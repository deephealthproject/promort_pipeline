import argparse
from cassandra.auth import PlainTextAuthProvider
from cassandra_dataset import CassandraDataset
from getpass import getpass
from envloader import EnvLoader

from pyspark import StorageLevel
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import time
from tqdm import trange, tqdm
import io
import numpy as np
import os
import random

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import pickle

# Run with
# /spark/bin/spark-submit --driver-memory 16G -c spark.driver.maxResultSize=0 --py-files cassandra_dataset.py,envloader.py,BPH.cpython-36m-x86_64-linux-gnu.so,/home/cesco/code/tmp/inet_256_rows.pckl,/DeepHealth/git/promort_pipeline/python/models.py --conf spark.cores.max=2 imagenet_spark_cass.py --nodes 2 --init-weights-fn /data/random_init.bin --out-dir ./  --max-patches 16 --batch-size 2 --epochs 3 --sync-iterations 2


### Function to create 
def get_net(net_name='vgg16', in_size=[256,256], num_classes=2, lr=1e-5, gpus=[1], init=eddl.HeNormal, dropout=None, l2_reg=None):

    ## Network definition
    in_ = eddl.Input([3, in_size[0], in_size[1]])

    if net_name == 'vgg16':
        out = models.VGG16(in_, num_classes, init=init, l2_reg=l2_reg, dropout=dropout)
    else:
        print('model %s not available' % net_name)
        sys.exit(-1)

    net = eddl.Model([in_], [out])
    eddl.build(
        net,
        eddl.rmsprop(lr),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(gpus, mem="low_mem") if gpus else eddl.CS_CPU()
        )

    return net


def set_net_weights(net, new_weights):
    l_l = net.layers
    for i, l in enumerate(l_l):
        w_l = new_weights[i]
        if w_l:
            bias = w_l[0]
            weights = w_l[1]
            l.update_weights(bias, weights)
            eddl.distributeParams(l)


def get_num_batches(el, seed):
    def ret(split_index):
        ## Environment loading
        el.value.start(seed)
        cd = el.value.cd
        return(cd.num_batches[split_index])
    return(ret)


def train(el, index_list, init_weights, sync_iterations, lr, gpus,
          dropout, l2_reg, seed):
    def ret(split_index):
        print ('Starting train function')
        t0 = time.time()
        
        ## Environment loading
        el.value.start(seed)
        cd = el.value.cd
        cd.set_indexes(index_list)
        net = el.value.net
        out = net.layers[-1]

        t1 = time.time()
        print ("Time to load the Environment  %.3f" % (t1-t0))
        t0 = t1
                
        ### Updating model weights 
        if isinstance(init_weights.value, str):
            eddl.load(net, str(init_weights.value))
            t1 = time.time()
            print ("Time to Load weights %.3f" % (t1-t0))
            t0 = t1

        elif init_weights.value:
            new_weights = init_weights.value 
            print (new_weights[-2][0].shape)

            t1 = time.time()
            print ("Receiving data from broadcast: %.3f" % (t1-t0))
            t0 = t1
            
            ## updating net weights layer by layer
            new_weights = [[Tensor(i) for i in l] for l in new_weights]  
            set_net_weights(net, new_weights)

            t1 = time.time()
            print ("Time to convert np array to tensor and set parameter to the net: %.3f" % (t1-t0))
            t0 = t1
        
        ###################
        ## Training step ##
        ###################
        metric_fn = eddl.getMetric("categorical_accuracy")
        loss_fn = eddl.getLoss("soft_cross_entropy")
        
        ### Main loop across epochs
        ### Training 
        if index_list[split_index] == 0: # Epoch start
            eddl.reset_loss(net) 
        
        metric_l = []
        loss_l = []
        
        ### Looping across batches of training data
        pbar = tqdm(range(sync_iterations))

        for b_index, b in enumerate(pbar):
            x, y = cd.load_batch(split_index)
    
            x.div_(255.0)
            tx, ty = [x], [y]
            eddl.train_batch(net, tx, ty)
            
            #print bratch train results
            loss = eddl.get_losses(net)[0]
            metr = eddl.get_metrics(net)[0]
            msg = "Batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(b + 1, sync_iterations, loss, metr)
            pbar.set_postfix_str(msg)
            loss_l.append(loss)
            metric_l.append(metr)
    
        pbar.close()

        t1 = time.time()
        print ("Time to perform training  %.3f" % (t1-t0))
        t0 = t1

        ## End of training. Get Loss, metric and weights
        p = eddl.get_parameters(net) # Get parameters from the model
        r = [[i.getdata() for i in l] for l in p] # Transform tensors to numpy array for spark serialization

        t1 = time.time()
        print ("Time to get parameters and convert them to numpy array  %.3f" % (t1-t0))
        t0 = t1
        
        res = (r, sum(loss_l), sum(metric_l))
        return res
    return ret

def sum_weights(in0, in1):
    # Weight lists, losses list, metric list
    w0, l0, m0 = in0
    w1, l1, m1 = in1
    
    print ("Reduce Function")
    t0 = time.time()
    tot_weights = [[(w0[i][j]+w1[i][j]) for j, _ in enumerate(l)] for i, l in enumerate(w0)]
    
    tot_losses = l0 + l1 
    
    tot_metrics = m0 + m1 

    t1 = time.time()
    print ("Reduce function time: %.3f" % (t1-t0))
    
    return (tot_weights, tot_losses, tot_metrics)


def run(args):
    try: 
        from private_data import inet_pass
    except ImportError:
        inet_pass = getpass('Insert Cassandra password: ')

    conf = SparkConf()\
        .setAppName("Distributed training")\
        .setMaster("spark://spark-master:7077")
    #conf.set('spark.scheduler.mode', 'FAIR')
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    
    ## Get parameters
    num = args.nodes # number of nodes
    
    if args.init_weights_fn:
        print ("Loading initialization weights")
        init_weights = args.init_weights_fn
    else:
        init_weights = None

    ### Parse GPU
    if args.gpu:
        gpus = [int(i) for i in args.gpu]
    else:
        gpus = []

    epochs = args.epochs
    sync_iterations = args.sync_iterations
    lr = args.lr
    augs_on = args.augs_on
    lsb = 1
    batch_size = args.batch_size
    dropout = args.dropout
    l2_reg = args.l2_reg
    max_patches = args.max_patches
    cass_row_fn = args.cass_row_fn
    cass_datatable = args.cass_datatable
    out_dir = args.out_dir
    seed = args.seed
    net_name = "vgg16"
    num_classes = 1000
    size = [256, 256]  # size of images
    net_init = eddl.HeNormal
    
    #########################
    ### Start parallel job ##
    #########################

    nodes = range(num)
    par_nodes = sc.parallelize(nodes, numSlices=num)

    loss_l = []
    acc_l = []
    val_loss_l = []
    val_acc_l = []

    t0 = time.time()
    
    el_bc = sc.broadcast(EnvLoader(inet_pass, num, augs_on,
                                   batch_size, max_patches,
                                   cass_row_fn, cass_datatable,
                                   net_name, size, num_classes,
                                   lr, gpus, net_init, 
                                   dropout, l2_reg))

    for ep in range(epochs):
        tmp_loss_l = []
        tmp_acc_l = []
        seed = random.getrandbits(32)
        num_batches = par_nodes\
                .map(get_num_batches(el_bc, seed))\
                .reduce(lambda x,y: min(x,y))
        n_steps = num_batches // sync_iterations
        for macro_step in range(n_steps):
            print ("KAIOOOOOO: %d, %d, %d" % (num_batches, n_steps, macro_step))
            init_weights_bc = sc.broadcast(init_weights)
            index_list = [macro_step * sync_iterations * batch_size] * num

            weights = par_nodes\
                .setName('Training function')\
                .map(train(el_bc, index_list, init_weights_bc,
                           sync_iterations, lr, gpus, dropout, l2_reg, seed))
            result = weights.setName('Reduce function').reduce(sum_weights)
            
            weights, losses, metrics = result
             
            # Convert list of numpy weigths to list of tensors
            init_weights = [[i / num for i in l] for l in weights]
                                                              
            tmp_loss_l.append(losses / (num * sync_iterations)) # Finalizing the average and appending to training losses history
            tmp_acc_l.append(metrics / (num * sync_iterations)) # Finalizing the average and appending to training metrics history 
            
            t1 = time.time()
            print (f"Time for macro epoch {ep}: {t1-t0:.3f}")
            t0 = t1
        
        loss_l.append(np.mean(tmp_loss_l))
        acc_l.append(np.mean(tmp_acc_l))

        if out_dir:
            # Store loss, metrics timeseries and weights
            history = {'loss': loss_l, 'acc': acc_l, 'val_loss': val_loss_l, 'val_acc': val_acc_l}
            pickle.dump(history, open(os.path.join(out_dir, 'history.pickle'), 'wb'))
            # Store weights


if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, metavar="INT", default=3, help='Number of nodes')
    parser.add_argument("--epochs", type=int, metavar="INT", default=10, help='Number of total epochs')
    parser.add_argument("--sync-iterations", type=int, metavar="INT", default=5, help='Number of step between weights sync')
    parser.add_argument("--max-patches", type=int, metavar="INT", default=1300000, help='Number of patches to use for all splits')
    parser.add_argument("--patience", type=int, metavar="INT", default=20, help='Number of epochs after which the training is stopped if validation accuracy does not improve (delta=0.001)')
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32, help='Batch size')
    parser.add_argument("--val-split-indexes", type=int, nargs='+', default=[], help='List of split indexs to be used as validation set in case of a multisplit dataset (e.g. for cross validation purpose')
    parser.add_argument("--test-split-indexes", type=int, nargs='+', default=[], help='List of split indexs to be used as validation set in case of a multisplit dataset (e.g. for cross validation purpose')
    parser.add_argument("--lsb", type=int, metavar="INT", default=1, help='(Multi-gpu setting) Number of batches to run before synchronizing the weights of the different GPUs')
    parser.add_argument("--seed", type=int, metavar="INT", default=None, help='Seed of the random generator to manage data load')
    parser.add_argument("--lr", type=float, metavar="FLOAT", default=1e-5, help='Learning rate')
    parser.add_argument("--lr_end", type=float, metavar="FLOAT", default=1e-2, help='Final learning rate. To be used with find-opt-lr option to scan learning rates')
    parser.add_argument("--dropout", type=float, metavar="FLOAT", default=None, help='Float value (0-1) to specify the dropout ratio' )
    parser.add_argument("--l2-reg", type=float, metavar="FLOAT", default=None, help='L2 regularization parameter')
    parser.add_argument("--gpu", nargs='+', default = [], help='Specify GPU mask. For example: 1 to use only gpu0; 1 1 to use gpus 0 and 1; 1 1 1 1 to use gpus 0,1,2,3')
    parser.add_argument("--save-weights", action="store_true", help='Network parameters are saved after each epoch')
    parser.add_argument("--augs-on", action="store_true", help='Activate data augmentations')
    parser.add_argument("--find-opt-lr", action="store_true", help='Scan learning rate with an increasing exponential law to find best lr')
    parser.add_argument("--out-dir", metavar="DIR", help="Specifies the output directory. If not set no output data is saved")
    parser.add_argument("--init-weights-fn", metavar="DIR", help="Filename of the .bin file with initial parameters of the network")
    parser.add_argument("--cass-row-fn", metavar="DIR", default='inet_256_rows.pckl',  help="Filename of cassandra rows file")
    parser.add_argument("--cass-datatable", metavar="DIR", default='imagenet.data_256',help="Name of cassandra datatable")
    run(parser.parse_args())
