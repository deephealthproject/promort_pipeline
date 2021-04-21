import argparse
from cassandra.auth import PlainTextAuthProvider
from cassandra_dataset import CassandraDataset
from getpass import getpass
from pyspark import StorageLevel
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import time
from tqdm import trange, tqdm
import io
import numpy as np
import os

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import models
import pickle

# Run with
# /spark/bin/spark-submit --driver-memory 16G -c spark.driver.maxResultSize=0 --py-files cassandra_dataset.py,BPH.cpython-36m-x86_64-linux-gnu.so,/home/cesco/code/tmp/inet_256_rows.pckl,/DeepHealth/git/promort_pipeline/python/models.py --conf spark.cores.max=8 imagenet_spark_cass.py --nodes 8 --init-weights-fn /data/random_init.bin


### Function to create 
def get_net(net_name='vgg16', in_size=[256,256], num_classes=2, lr=1e-5, augs=False, gpus=[1], lsb=1, init=eddl.HeNormal, dropout=None, l2_reg=None):

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
        #eddl.CS_GPU([1,1], mem="low_mem") if gpu else eddl.CS_CPU()
        eddl.CS_GPU(gpus, mem="low_mem", lsb=lsb) if gpus else eddl.CS_CPU()
        #eddl.CS_GPU(gpus, mem="low_mem") if gpus else eddl.CS_CPU()
        )

    #eddl.summary(net)
    #eddl.setlogfile(net, "promort_VGG16_classification")

    if augs:
        ## Set augmentations
        training_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugMirror(.5),
            ecvl.AugFlip(.5),
            ecvl.AugRotate([-10, 10])
        ])

        validation_augs = ecvl.SequentialAugmentationContainer([
        ])

        dataset_augs = [training_augs, validation_augs, None]

    else:
        dataset_augs = [None, None, None]

    return net, dataset_augs


def train(cass_row_fn, cass_datatable, inet_pass, max_split, init_weights, epochs, lr, augs_on, gpus, lsb, batch_size, dropout,l2_reg, max_patches=1300000, seed=123):
    def ret(split_index):
        print ('Starting train function')
        t0 = time.time()
        
        ### Get model 
        print ('Creating EDDL network...')
        net_name = "vgg16"
        num_classes = 1000
        size = [256, 256]  # size of images
 
        ### Get Network
        net_init = eddl.HeNormal
        net, dataset_augs = get_net(net_name='vgg16', in_size=size, num_classes=num_classes, lr=lr, augs=augs_on, gpus=gpus, lsb=lsb, init=net_init, dropout=dropout, l2_reg=l2_reg)
        out = net.layers[-1]

        t1 = time.time()
        print ("Time to create VGG16 %.3f" % (t1-t0))
        t0 = t1
                
        ### Cassandra Split Creation
        ap = PlainTextAuthProvider(username='inet', password=inet_pass)
        cd = CassandraDataset(ap, ['cassandra_db'], port=9042, seed=seed)
        cd.load_rows(cass_row_fn)
        cd.init_datatable(table=cass_datatable)
        cd.split_setup(batch_size=batch_size, split_ratios=[1]*max_split,
                       max_patches=max_patches, augs=dataset_augs)
       
        t1 = time.time()
        print ("Time for Cassandra set up %.3f" % (t1-t0))
        t0 = t1
        
        ### Updating model weights 
        if isinstance(init_weights.value, str):
            eddl.load(net, str(init_weights.value))
            #r = [[np.empty((40960,4096), dtype=np.float32)]]
            t1 = time.time()
            print ("Time to Load weights %.3f" % (t1-t0))
            t0 = t1

        elif init_weights.value:
            t1 = time.time()
            print ("Time waiting to convert np array to tensor and set parameter to the net: %.3f" % (t1-t0))
            t0 = t1

            new_weights = init_weights.value 
            print (new_weights[-2][0].shape)

            t1 = time.time()
            print ("Receiving data from broadcast: %.3f" % (t1-t0))
            t0 = t1
            new_weights = [[Tensor(i) for i in l] for l in new_weights]  
            
            ## updating net weights layer by layer
            l_l = net.layers
            for i, l in enumerate(l_l):
                w_l = new_weights[i]
                if w_l:
                    bias = w_l[0]
                    weights = w_l[1]
                    l.update_weights(bias, weights)

            t1 = time.time()
            print ("Time to convert np array to tensor and set parameter to the net: %.3f" % (t1-t0))
            t0 = t1
        
        ###################
        ## Training step ##
        ###################
        num_batches_tr = cd.num_batches[split_index]
        metric_fn = eddl.getMetric("categorical_accuracy")
        loss_fn = eddl.getLoss("soft_cross_entropy")
        
        print ("Strating Training")

        loss_l = []
        acc_l = []
        
        ### Main loop across epochs
        for e in range(epochs):

            ### Training 
            print("Epoch {:d}/{:d} - Training".format(e + 1, epochs),
                  flush=True)
            
            cd.rewind_splits(shuffle=True)
            eddl.reset_loss(net)
            total_metric = []
            total_loss = []
            
            ### Looping across batches of training data
            pbar = tqdm(range(num_batches_tr))

            for b_index, b in enumerate(pbar):
                x, y = cd.load_batch(split_index)
        
                x.div_(255.0)
                tx, ty = [x], [y]
                eddl.train_batch(net, tx, ty)
                
                #print bratch train results
                loss = eddl.get_losses(net)[0]
                metr = eddl.get_metrics(net)[0]
                msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(e + 1, epochs, b + 1, num_batches_tr, loss, metr)
                pbar.set_postfix_str(msg)
                total_loss.append(loss)
                total_metric.append(metr)
        
            loss_l.append(np.mean(total_loss))
            acc_l.append(np.mean(total_metric))

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
        
        res = (r, loss_l, acc_l)
        return res
    return ret

def sum_weights(in0, in1):
    # Weight lists, losses list, metric list
    w0, l0, m0 = in0
    w1, l1, m1 = in1
    
    print ("Reduce Function")
    t0 = time.time()
    tot_weights = [[(w0[i][j]+w1[i][j]) for j, _ in enumerate(l)] for i, l in enumerate(w0)]
    
    tot_losses = [l0[i] + l1[1] for i, _ in enumerate(l0)]
    
    tot_metrics = [m0[i] + m1[1] for i, _ in enumerate(m0)]
    
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
    sync_epochs = args.sync_epochs
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

    #########################
    ### Start parallel job ##
    #########################

    m_epochs = epochs // sync_epochs
    nodes = range(num)
    par_nodes = sc.parallelize(nodes, numSlices=num)

    loss_l = []
    acc_l = []
    val_loss_l = []
    val_acc_l = []

    t0 = time.time()

    for ep in range(m_epochs):
        init_weights_bc = sc.broadcast(init_weights)
        #init_weights_bc = init_weights
        weights = par_nodes\
            .setName('Training function')\
            .map(train(cass_row_fn, cass_datatable, inet_pass, num, init_weights_bc, sync_epochs, lr, augs_on, gpus, lsb, batch_size, dropout,l2_reg, max_patches, seed=123))
        result = weights\
            .setName('Reduce function')\
            .reduce(sum_weights)
        
        weights, losses, metrics = result
         
        # Convert list of numpy weigths to list of tensors
        init_weights = [[i / num for i in l] for l in weights]
        loss_l += [i / num for i in losses] # Finalizing the average and appending to training losses history
        acc_l += [i / num for i in metrics] # Finalizing the average and appending to training metrics history 
        
        t1 = time.time()
        print (f"Time for macro epoch {ep}: {t1-t0:.3f}")
        t0 = t1

        if out_dir:
            history = {'loss': loss_l, 'acc': acc_l, 'val_loss': val_loss_l, 'val_acc': val_acc_l}
            pickle.dump(history, open(os.path.join(out_dir, 'history.pickle'), 'wb'))


if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, metavar="INT", default=3, help='Number of nodes')
    parser.add_argument("--epochs", type=int, metavar="INT", default=10, help='Number of total epochs')
    parser.add_argument("--sync-epochs", type=int, metavar="INT", default=5, help='Number of epochs between weights sync')
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
