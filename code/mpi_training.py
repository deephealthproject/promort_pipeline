import argparse
from getpass import getpass
from mpi_envloader import EnvLoader
from mpi_functions import train

import time
from tqdm import trange, tqdm
import numpy as np
import os
import random
import pickle
import sys

from OPT_MPI import mpi_env

# Run with:
# mpirun --mca btl tcp,self --bind-to none --mca btl_base_verbose 30 -n 2 --hostfile tmp/hostfile  python3 mpi_training.py --ltr 1 --epochs 2 --sync-iterations 2 --batch-size 32 --num-classes 1000 --lr 1e-6 --cass-row-fn /data/code/tmp/inet_256_rows.pckl

def run(args):
    try:
        from private_data import inet_pass
    except ImportError:
        inet_pass = getpass('Insert Cassandra password: ')

    
    init_weights_fn = args.init_weights_fn
    # Parse GPU
    if args.gpu:
        gpus = [int(i) for i in args.gpu]
    else:
        gpus = []

    epochs = args.epochs
    n_sync = args.sync_iterations
    lr = args.lr
    augs_on = args.augs_on
    batch_size = args.batch_size
    num_classes = args.num_classes
    dropout = args.dropout
    l2_reg = args.l2_reg
    max_patches = args.max_patches
    cass_row_fn = args.cass_row_fn
    cass_datatable = args.cass_datatable
    out_dir = args.out_dir
    seed = args.seed
    net_name = args.net_name
    p_size = args.patch_size
    size = [p_size, p_size] 
    patience = args.patience
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    net_init = 'HeNormal'
    
    ## Each node gets its own environment
    
    el = EnvLoader(inet_pass, n_sync, val_ratio, test_ratio, augs_on, batch_size, max_patches,
                                   cass_row_fn, cass_datatable,
                                   net_name, size, num_classes,
                                   lr, gpus, net_init,
                                   dropout, l2_reg, seed)

    #########################
    ### Start parallel job ##
    #########################
    
    results = train(el, init_weights_fn, epochs, lr, gpus, dropout, l2_reg, seed, out_dir)    

    loss_l, acc_l, val_loss_l, val_acc_l = results
   
    rank = el.MP.mpi_rank
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
    parser.add_argument("--epochs", type=int, metavar="INT",
                        default=10, help='Number of total epochs')
    parser.add_argument("--sync-iterations", type=int, metavar="INT",
                        default=5, help='Number of step between weights sync')
    parser.add_argument("--max-patches", type=int, metavar="INT",
                        default=1300000, help='Number of patches to use for all splits')
    parser.add_argument("--patch-size", type=int, metavar="INT",
                        default=224, help='Patch side size')
    parser.add_argument("--patience", type=int, metavar="INT", default=20,
                        help='Number of epochs after which the training is stopped if validation accuracy does not improve (delta=0.001)')
    parser.add_argument("--batch-size", type=int,
                        metavar="INT", default=32, help='Batch size')
    parser.add_argument("--num-classes", type=int,
                        metavar="INT", default=1000, help='Number of classes')
    parser.add_argument("--val-ratio", type=float, default=0.0,
                        help='Validation split ratio')
    parser.add_argument("--test-ratio", type=float, default=0.0,
                        help='Test split ratio')
    parser.add_argument("--seed", type=int, metavar="INT", default=None,
                        help='Seed of the random generator to manage data load')
    parser.add_argument("--lr", type=float, metavar="FLOAT",
                        default=1e-5, help='Learning rate')
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
    parser.add_argument("--out-dir", metavar="DIR", default = './',
                        help="Specifies the output directory. If not set no output data is saved")
    parser.add_argument("--init-weights-fn", metavar="DIR",
                        help="Filename of the .bin file with initial parameters of the network")
    parser.add_argument("--cass-row-fn", metavar="DIR", required=True,  help="Filename of cassandra rows file")
    parser.add_argument("--cass-datatable", metavar="DIR", required=True, help="Name of cassandra datatable")
    parser.add_argument("--net-name",
                        default='resnet50',  help="Select a model between resnet50 and vgg16")
    run(parser.parse_args())
