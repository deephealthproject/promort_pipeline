import argparse
from getpass import getpass
from envloader import EnvLoader
from mpi_functions import train

import time
from tqdm import trange, tqdm
import numpy as np
import os
import random
import pickle
import sys
#sys.path.append('./miniMPI')
from MMPI import miniMPI

# Run with:
# mpirun --mca btl tcp,self --bind-to none --mca btl_base_verbose 30 --mca btl_tcp_links 1 -n 2 --hostfile tmp/hostfile  python3 mpi_training.py --ltr 1 --epochs 2 --sync-iterations 2 --batch-size 32 --num-classes 1000 --lr 1e-6 --cass-row-fn /data/code/tmp/inet_256_rows.pckl

def run(args):
    try:
        from private_data import inet_pass
    except ImportError:
        inet_pass = getpass('Insert Cassandra password: ')

    MP = miniMPI(bl=2048)
    
    # Get parameters
    num_nodes = MP.mpi_size # number of nodes
    rank = MP.mpi_rank

    ltr = args.ltr # Number of local indipendent training before syncing
    
    init_weights_fn = args.init_weights_fn
    # Parse GPU
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
    num_classes = args.num_classes
    dropout = args.dropout
    l2_reg = args.l2_reg
    max_patches = args.max_patches
    cass_row_fn = args.cass_row_fn
    cass_datatable = args.cass_datatable
    out_dir = args.out_dir
    seed = args.seed
    net_name = "vgg16"
    size = [256, 256]  # size of images
    net_init = 'HeNormal'
    
    ## Each node gets its own environment
    num_splits = num_nodes * ltr

    el = EnvLoader(inet_pass, num_splits, augs_on,
                                   batch_size, max_patches,
                                   cass_row_fn, cass_datatable,
                                   net_name, size, num_classes,
                                   lr, gpus, net_init,
                                   dropout, l2_reg)

    #########################
    ### Start parallel job ##
    #########################
    
    results = train(MP, ltr, el, init_weights_fn, epochs, sync_iterations, lr, gpus, dropout, l2_reg, seed)    

    loss_l, acc_l, val_loss_l, val_acc_l = results
   
    if rank == 0:
        if out_dir:
            # Store loss, metrics timeseries and weights
            history = {'loss': loss_l, 'acc': acc_l,
                       'val_loss': val_loss_l, 'val_acc': val_acc_l}
            pickle.dump(history, open(os.path.join(
                out_dir, 'history.pickle'), 'wb'))
            # Store weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, metavar="INT",
                        default=3, help='Number of nodes')
    parser.add_argument("--ltr", type=int, metavar="INT",
                        default=1, help='Number of local indipendent training before averaging gradients')
    parser.add_argument("--epochs", type=int, metavar="INT",
                        default=10, help='Number of total epochs')
    parser.add_argument("--sync-iterations", type=int, metavar="INT",
                        default=5, help='Number of step between weights sync')
    parser.add_argument("--max-patches", type=int, metavar="INT",
                        default=1300000, help='Number of patches to use for all splits')
    parser.add_argument("--patience", type=int, metavar="INT", default=20,
                        help='Number of epochs after which the training is stopped if validation accuracy does not improve (delta=0.001)')
    parser.add_argument("--batch-size", type=int,
                        metavar="INT", default=32, help='Batch size')
    parser.add_argument("--num-classes", type=int,
                        metavar="INT", default=1000, help='Number of classes')
    parser.add_argument("--val-split-indexes", type=int, nargs='+', default=[],
                        help='List of split indexs to be used as validation set in case of a multisplit dataset (e.g. for cross validation purpose')
    parser.add_argument("--test-split-indexes", type=int, nargs='+', default=[],
                        help='List of split indexs to be used as validation set in case of a multisplit dataset (e.g. for cross validation purpose')
    parser.add_argument("--lsb", type=int, metavar="INT", default=1,
                        help='(Multi-gpu setting) Number of batches to run before synchronizing the weights of the different GPUs')
    parser.add_argument("--seed", type=int, metavar="INT", default=None,
                        help='Seed of the random generator to manage data load')
    parser.add_argument("--lr", type=float, metavar="FLOAT",
                        default=1e-5, help='Learning rate')
    parser.add_argument("--lr_end", type=float, metavar="FLOAT", default=1e-2,
                        help='Final learning rate. To be used with find-opt-lr option to scan learning rates')
    parser.add_argument("--dropout", type=float, metavar="FLOAT",
                        default=None, help='Float value (0-1) to specify the dropout ratio')
    parser.add_argument("--l2-reg", type=float, metavar="FLOAT",
                        default=None, help='L2 regularization parameter')
    parser.add_argument("--gpu", nargs='+', default=[],
                        help='Specify GPU mask. For example: 1 to use only gpu0; 1 1 to use gpus 0 and 1; 1 1 1 1 to use gpus 0,1,2,3')
    parser.add_argument("--save-weights", action="store_true",
                        help='Network parameters are saved after each epoch')
    parser.add_argument("--augs-on", action="store_true",
                        help='Activate data augmentations')
    parser.add_argument("--find-opt-lr", action="store_true",
                        help='Scan learning rate with an increasing exponential law to find best lr')
    parser.add_argument("--out-dir", metavar="DIR",
                        help="Specifies the output directory. If not set no output data is saved")
    parser.add_argument("--init-weights-fn", metavar="DIR",
                        help="Filename of the .bin file with initial parameters of the network")
    parser.add_argument("--cass-row-fn", metavar="DIR",
                        default='inet_256_rows.pckl',  help="Filename of cassandra rows file")
    parser.add_argument("--cass-datatable", metavar="DIR",
                        default='imagenet.data_256', help="Name of cassandra datatable")
    run(parser.parse_args())
