# Copyright (c) 2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
PROMORT training model application.
"""

### Command line example
# python3 promort_tumor_training.py  --init-weights-fn keras/vgg16_imagenet_init_onlyconv.bin --epochs 50 --patience 20 --batch-size 64 --val-split-indexes 8 --test-split-indexes 9 --dropout 0.5 --save-weights --out-dir /tmp/promort_training_test --splits-fn cassandra_splits/10_splits/cosk_l1_bal_tcr_0.80_1.00.pckl --gpu 1

import argparse
import random
import sys
import os
from pathlib import Path
import time

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor


#from cassandra_dataset import CassandraDataset
#from cassandra.auth import PlainTextAuthProvider
from cassandradl import CassandraDataset
from cassandra.auth import PlainTextAuthProvider

from getpass import getpass
from tqdm import trange, tqdm
import numpy as np
import pickle 

import models 
import gc


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def accuracy(predictions, targets, epsilon=1e-12, th=0.5):
    """
    Computes accuracy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    targets = np.around(targets) # Rounds target values in the case of smooth labels
    N = predictions.shape[0]
    predictions[predictions >= th] = 1
    predictions[predictions < th] = 0    
    ca = np.sum((targets * predictions) + 1e-9) / N
    return ca



def get_net(net_name='vgg16_tumor', in_size=[256,256], num_classes=2, lr=1e-5, augs=False, gpus=[1], lsb=1, init=eddl.HeNormal, dropout=None, l2_reg=None, mem='low_mem'):
    
    ## Network definition
    in_ = eddl.Input([3, in_size[0], in_size[1]])
    
    if net_name == 'vgg16_tumor':
        out = models.VGG16_tumor(in_, num_classes, init=init, l2_reg=l2_reg, dropout=dropout)
    elif net_name == 'vgg16_gleason':
        out = models.VGG16_gleason(in_, num_classes, init=init, l2_reg=l2_reg, dropout=dropout)
    elif net_name == 'vgg16':
        out = models.VGG16(in_, num_classes, init=init, l2_reg=l2_reg, dropout=dropout)
    elif net_name == 'resnet50':
        out = models.ResNet50(in_, num_classes, init=init, l2_reg=l2_reg, dropout=dropout)
    else:
        print('model %s not available' % net_name)
        sys.exit(-1)

    net = eddl.Model([in_], [out])
    eddl.build(
        net,
        eddl.rmsprop(lr),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(gpus, mem=mem, lsb=lsb) if gpus else eddl.CS_CPU()
        )

    eddl.summary(net)
    eddl.setlogfile(net, "promort_VGG16_classification")
   
    if augs:
        ## Set augmentations
        training_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugMirror(.5),
            ecvl.AugFlip(.5),
            ecvl.AugRotate([-45, 45])
            #ecvl.AugAdditivePoissonNoise([0, 10]),
            #ecvl.AugGammaContrast([0.5, 1.5]),
            #ecvl.AugGaussianBlur([0, 0.8]),
            #ecvl.AugCoarseDropout([0, 0.3], [0.02, 0.05], 0.5)
        ])
        
        validation_augs = ecvl.SequentialAugmentationContainer([
        ])
        
        dataset_augs = [training_augs, validation_augs, None]
    
    else:
        dataset_augs = [None, None, None]

    return net, dataset_augs


def rescale_tensor(x, vgg_pretrained=True, mode='tf'):
    if mode == 'tf' and vgg_pretrained:
        # Data in -1,1 interval
        x.div_(255.0)
        x.mult_(2)
        x.add_(-1)
        return 
    elif mode == 'torch' or not vgg_pretrained:
        # Data in 0,1 interval
        x.div_(255.0)
        return 

def main(args):
    size = [256, 256]  # size of images

    ### Net name
    net_name = args.net_name

    ### Num classes
    num_classes = args.num_classes

    ### mem
    if args.full_mem:
        mem = 'full_mem'
    else:
        mem = 'low_mem'

    ### Parse GPU
    if args.gpu:
        gpus = [int(i) for i in args.gpu]
    else:
        gpus = []

    print ('GPUs mask: %r' % gpus)
    ### Get Network
    if args.init_conv == 'he':
        net_init = eddl.HeNormal
    elif args.init_conv == 'glorot':
        net_init = eddl.GlorotNormal

    net, dataset_augs = get_net(net_name=net_name, in_size=size, num_classes=num_classes, lr=args.lr, 
            augs=args.augs_on, gpus=gpus, lsb=args.lsb, init=net_init, dropout=args.dropout, 
            l2_reg=args.l2_reg, mem=mem)

    out = net.layers[-1]
    
    ## Load weights if requested
    if args.init_weights_fn:
        print ("Loading initialization weights")
        eddl.load(net, args.init_weights_fn)
    
    ## Check options
    if args.out_dir:
        working_dir = "model_cnn_%s_ps.%d_bs_%d_lr_%.2e" % (net_name, size[0], args.batch_size, args.lr)
        res_dir = os.path.join(args.out_dir, working_dir)
        try:
            os.makedirs(res_dir, exist_ok=True)
        except:
            print ("Directory already exists.")
            sys.exit()

    
    ########################################
    ### Set database and read split file ###
    ########################################

    if not args.cassandra_pwd_fn:
        cass_pass = getpass('Insert Cassandra password: ')	
    else:
        with open(args.cassandra_pwd_fn) as fd:
            cass_pass = fd.readline().rstrip()

    # create cassandra reader
    ap = PlainTextAuthProvider(username='prom', password=cass_pass)
    cd = CassandraDataset(ap, ['156.148.70.72'], seed=args.seed)
    #cd = CassandraDataset(ap, ['127.0.0.1'], seed=args.seed)
    
    # Smooth label param
    cd.smooth_eps = args.smooth_lab

    # Check if file exists
    if Path(args.splits_fn).exists():
        # Load splits 
        cd.load_splits(args.splits_fn, batch_size=args.batch_size, augs=dataset_augs, whole_batches=True)
    else:
        print ("Split file %s not found" % args.splits_fn)
        sys.exit(-1)

    # Check if a new data table has to be set
    if args.data_table:
        print (f"Setting data table to {args.data_table}")
        cd.table = args.data_table

    print (f"Using data table {args.data_table}")
    print ('Number of batches for each split (train, val, test):', cd.num_batches)
    
    ## validation index check and creation of split indexes lists
    n_splits = cd.num_splits
    if args.val_split_indexes:
        out_indexes = [i for i in args.val_split_indexes if i > (n_splits-1)]
        if out_indexes:
            print (f"Not valid validation split index: {out_indexes}")
            sys.exit(-1)

        val_splits = args.val_split_indexes
        test_splits = args.test_split_indexes
        train_splits = [i for i in range(n_splits) if (i not in val_splits) and (i not in test_splits)]
        num_batches_tr = np.sum([cd.num_batches[i] for i in train_splits])
        num_batches_val = np.sum([cd.num_batches[i] for i in val_splits])

        print ("Train splits: %r" % train_splits)
        print ("Val splits: %r" % val_splits)
        print ("Test splits: %r" % test_splits)
    
    else:
        if n_splits == 1:
            num_batches_tr = cd.num_batches[0]
            num_batches_val = 0
        else:
            num_batches_tr = cd.num_batches[0]
            num_batches_val = cd.num_batches[1]
    
    ################################
    #### Training and evaluation ###
    ################################

    print("Defining metric...", flush=True)
    
    metric_fn = eddl.getMetric("categorical_accuracy")
    loss_fn = eddl.getLoss("soft_cross_entropy")

    print("Starting training", flush=True)

    loss_l = []
    acc_l = []
    val_loss_l = []
    val_acc_l = []
    
    patience_cnt = 0
    val_acc_max = 0.0

    #### Code used to find best learning rate. Comment it to perform an actual training
    if args.find_opt_lr:
        max_epochs = args.epochs
        lr_start = args.lr
        lr_end = args.lr_end
        lr_f = lambda x: 10**(np.log10(lr_start) + ((np.log10(lr_end)-np.log10(lr_start))/max_epochs)*x)
    ####

    ### Main loop across epochs
    for e in range(args.epochs):
        ## SET LT
        if args.find_opt_lr:
            eddl.setlr(net, [lr_f(e)])

        ### Training 
        ## Set training mode 
        eddl.set_mode(net, 1) # TRMODE = 1, TSMODE = 0
        
        # Smooth label param
        cd.smooth_eps = args.smooth_lab
        cd._reset_indexes()

        cd.current_split = 0 ## Set the training split as the current one
        print("Epoch {:d}/{:d} - Training".format(e + 1, args.epochs),
              flush=True)
        
        cd.rewind_splits(shuffle=True)
        eddl.reset_loss(net)
        total_metric = []
        total_loss = []
        
        ### Looping across batches of training data
        pbar = tqdm(range(num_batches_tr))
        #pbar = tqdm(range(num_batches_tr))

        for b_index, b in enumerate(pbar):
            if args.val_split_indexes:
                x, y = cd.load_batch_cross(not_splits=val_splits+test_splits)
            else:
                x, y = cd.load_batch()
                #if x.shape[0] != args.batch_size:
                #    print (b_index, x.shape[0])
    
            rescale_tensor(x)
            #print (x.getdata())
            tx, ty = [x], [y]
            eddl.train_batch(net, tx, ty)

            #print bratch train results
            loss = eddl.get_losses(net)[0]
            #metr = eddl.get_metrics(net)[0]
            output = eddl.getOutput(out)
            result = output.getdata()
            target = y.getdata()
            metr = accuracy(result, target)
            
            total_loss.append(loss)
            total_metric.append(metr)
            msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(e + 1, args.epochs, b + 1, num_batches_tr, np.mean(total_loss), np.mean(total_metric))
            pbar.set_postfix_str(msg)

            if b_index % num_batches_tr == 0:
                cd.rewind_splits(shuffle=True)
    
        loss_l.append(np.mean(total_loss))
        acc_l.append(np.mean(total_metric))

        pbar.close()

        ### Evaluation on validation set batches
        eddl.set_mode(net, 0) # Set test mode. Dropout is deactivated
        # Smooth label param
        cd.smooth_eps = 0.0
        cd._reset_indexes()
        
        cd.current_split = 1 ## Set validation split as the current one
        tot_acc = []
        tot_loss = []
        
        print("Epoch %d/%d - Evaluation" % (e + 1, args.epochs), flush=True)
        
        pbar = tqdm(range(num_batches_val))

        for b_index, b in enumerate(pbar):
            if args.val_split_indexes:
                x, y = cd.load_batch_cross(not_splits=train_splits+test_splits)
            else:
                x, y = cd.load_batch()

            rescale_tensor(x)
            eddl.forward(net, [x])
            output = eddl.getOutput(out)
            
            result = output.getdata()
            target = y.getdata()
            ca = accuracy(result, target)
            ce = cross_entropy(result, target)

            tot_loss.append(ce)
            tot_acc.append(ca)

            msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) loss: {:.3f}, acc: {:.3f} ".format(
                e + 1,
                args.epochs,
                b + 1,
                num_batches_val,
                np.mean(tot_loss),
                np.mean(tot_acc),
            )
            pbar.set_postfix_str(msg)
            
        pbar.close()
        val_batch_acc_avg = np.mean(tot_acc)
        val_batch_loss_avg = np.mean(tot_loss)
        val_loss_l.append(val_batch_loss_avg)
        val_acc_l.append(val_batch_acc_avg)
    
        print("loss: {:.3f}, acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}\n".format(loss_l[-1], acc_l[-1], val_loss_l[-1], val_acc_l[-1]))

        ## Save weights 
        if args.save_weights:
            print("Saving weights")
            path = os.path.join(res_dir, "promort_%s_weights_ep_%s_vacc_%.2f.bin" % (net_name, e, val_acc_l[-1]))
            eddl.save(net, path, "bin")
    
        # Dump history at the end of each epoch so if the job is interrupted data are not lost.
        if args.out_dir:
            history = {'loss': loss_l, 'acc': acc_l, 'val_loss': val_loss_l, 'val_acc': val_acc_l}
            pickle.dump(history, open(os.path.join(res_dir, 'history.pickle'), 'wb'))
        
        ### Patience check
        if val_acc_l[-1] > val_acc_max:
            val_acc_max = val_acc_l[-1]
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt > args.patience:
            ## Exit and complete the training
            print ("Got maximum patience... training completed")
            break
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=50, help='Number of epochs')
    parser.add_argument("--patience", type=int, metavar="INT", default=20, help='Number of epochs after which the training is stopped if validation accuracy does not improve (delta=0.001)')
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32, help='Batch size')
    parser.add_argument("--num-classes", type=int, metavar="INT", default=2, help='Number of categories in the dataset')
    parser.add_argument("--val-split-indexes", type=int, nargs='+', default=[], help='List of split indexs to be used as validation set in case of a multisplit dataset (e.g. for cross validation purpose')
    parser.add_argument("--test-split-indexes", type=int, nargs='+', default=[], help='List of split indexs to be used as validation set in case of a multisplit dataset (e.g. for cross validation purpose')
    parser.add_argument("--lsb", type=int, metavar="INT", default=1, help='(Multi-gpu setting) Number of batches to run before synchronizing the weights of the different GPUs')
    parser.add_argument("--seed", type=int, metavar="INT", default=None, help='Seed of the random generator to manage data load')
    parser.add_argument("--lr", type=float, metavar="FLOAT", default=1e-6, help='Learning rate')
    parser.add_argument("--lr-end", type=float, metavar="FLOAT", default=1e-3, help='Final learning rate. To be used with find-opt-lr option to scan learning rates')
    parser.add_argument("--dropout", type=float, metavar="FLOAT", default=None, help='Float value (0-1) to specify the dropout ratio' )
    parser.add_argument("--l2-reg", type=float, metavar="FLOAT", default=None, help='L2 regularization parameter')
    parser.add_argument("--gpu", nargs='+', default = [], help='Specify GPU mask. For example: 1 to use only gpu0; 1 1 to use gpus 0 and 1; 1 1 1 1 to use gpus 0,1,2,3')
    parser.add_argument("--save-weights", action="store_true", help='Network parameters are saved after each epoch')
    parser.add_argument("--augs-on", action="store_true", help='Activate data augmentations')
    parser.add_argument("--find-opt-lr", action="store_true", help='Scan learning rate with an increasing exponential law to find best lr')
    parser.add_argument("--full-mem", action="store_true", help='Activate data augmentations')
    parser.add_argument("--smooth-lab", type=float, metavar="FLOAT", default=0.0, help='smooth labeling parameter')
    parser.add_argument("--out-dir", metavar="DIR",
                        help="Specifies the output directory. If not set no output data is saved")
    parser.add_argument("--init-weights-fn", metavar="DIR",
                        help="Filename of the .bin file with initial parameters of the network")
    parser.add_argument("--init-conv", default='he',
                        help="initialization method of convolutional layers")
    parser.add_argument("--splits-fn", metavar="STR", required=True,
                        help="Pickle file with cassandra splits")
    parser.add_argument("--data-table", metavar="STR", default=None,
                        help="set a different data table with respect to the one specified in the split file")
    parser.add_argument("--cassandra-pwd-fn", metavar="STR", default='/tmp/cassandra_pass.txt',
                        help="cassandra password")
    parser.add_argument("--net-name", metavar="STR", default='vgg16_tumor',
                        help="Select the neural net (vgg16|vgg16_tumor|vgg16_gleason|resnet50)")
    main(parser.parse_args())
