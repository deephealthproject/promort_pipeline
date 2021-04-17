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
# /spark/bin/spark-submit --py-files cassandra_dataset.py,BPH.cpython-36m-x86_64-linux-gnu.so,/home/cesco/code/tmp/inet_256_rows.pckl,/DeepHealth/git/promort_pipeline/python/models.py --conf spark.cores.max=10 imagenet_spark_cass.py --nodes 10 --init-weights-fn /DeepHealth/git/promort_pipeline/python/keras/vgg16_imagenet_init.bin


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

    eddl.summary(net)
    eddl.setlogfile(net, "promort_VGG16_classification")

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


def train(inet_pass, num, init_weights, lr, augs_on, gpus, lsb, dropout,l2_reg, seed=123):
    def ret(i):
        ap = PlainTextAuthProvider(username='inet', password=inet_pass)
        cd = CassandraDataset(ap, ['cassandra_db'], port=9042, seed=seed)
        cd.load_rows('inet_256_rows.pckl')
        cd.init_datatable(table='imagenet.data_256')
        cd.split_setup(batch_size=32, split_ratios=[1]*num,
                       max_patches=1300000, augs=[])

        ### Get model 
        net_name = "vgg16"
        num_classes = 1000
        size = [256, 256]  # size of images

        
        ### Get Network
        net_init = eddl.HeNormal
        net, dataset_augs = get_net(net_name='vgg16', in_size=size, num_classes=num_classes, lr=lr, augs=augs_on, gpus=gpus, lsb=lsb, init=net_init, dropout=dropout, l2_reg=l2_reg)
        out = net.layers[-1]
       
        # Load or set weights 
        if isinstance(init_weights, str):
            print ("Loading weitghts into the model")
            eddl.load(net, init_weights)
        elif init_weights:
            print (init_weights.value)
            #new_weights = [[Tensor(i) for i in l] for l in init_weights.value]  
            #eddl.set_parameters(net, new_weights)

        ### Here the training code for. Train epochs before sync
        x,y = cd.load_batch(i)
        
        p = eddl.get_parameters(net) # Get parameters from the model
        r = [[i.getdata() for i in l] for l in p] # Transform tensors to numpy array for spark serialization
 
        return r
    return ret


def average_weights(w0, w1):
    mean_weights = [[(w0[i][j]+w1[i][j])/2 for j, _ in enumerate(l)] for i, l in enumerate(w0)]
    return mean_weights


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

    lr = args.lr
    augs_on = args.augs_on
    lsb = 1
    dropout = args.dropout
    l2_reg = args.l2_reg
    

    ### Start parallel job
    nodes = range(num)
    par_nodes = sc.parallelize(nodes, numSlices=num)
    for ep in range(5):
        init_weights_bc = sc.broadcast(init_weights)
        data = par_nodes\
            .map(train(inet_pass, num, init_weights_bc, lr, augs_on, gpus, lsb, dropout,l2_reg, seed=123))\
            .reduce(average_weights)
        
        # Convert list of numpy weigths to list of tensors
        init_weights = data
            
        pickle.dump(data, open('test.pckl', 'wb'))

if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, metavar="INT", default=3, help='Number of nodes')
    parser.add_argument("--epochs", type=int, metavar="INT", default=50, help='Number of epochs')
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
    run(parser.parse_args())
