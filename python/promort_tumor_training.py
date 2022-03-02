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

from tqdm import trange, tqdm
import numpy as np
import pickle 
import json 
import copy 

import models 
from promort_functions import cross_entropy, accuracy, get_data_augs, get_cassandra_dl

def main(args):
    in_shape = [3, 256, 256]  # shape of input data (channels, w_size, h_size) 

    #####################################################################
    ### Get the built network and set data augmentations if requested ###
    #####################################################################
    net = models.get_net(net_name=args.net_name, in_shape=in_shape, num_classes=args.num_classes, 
            full_mem=args.full_mem, lr=args.lr, gpus=args.gpu, lsb=args.lsb, init=args.init_conv, 
            dropout=args.dropout, l2_reg=args.l2_reg, init_weights_fn=args.init_weights_fn)

    out = net.layers[-1]
    

    ########################################
    ### Set database and read split file ###
    ########################################
    data_preprocs, read_rgb = get_data_augs(args.preprocess_mode, args.augs_on, args.read_rgb)

    cd, num_batches_tr, num_batches_val, train_splits, val_splits, test_splits = get_cassandra_dl(splits_fn=args.splits_fn, 
            num_classes = args.num_classes, data_table=args.data_table, 
            smooth_lab=args.smooth_lab, seed=args.seed, cassandra_pwd_fn=args.cassandra_pwd_fn,
            batch_size=args.batch_size, dataset_augs=data_preprocs,  
            val_split_indexes=args.val_split_indexes, 
            test_split_indexes=args.test_split_indexes, 
            read_rgb=read_rgb,
            lab_map=args.lab_map, 
            full_batches=True)
   

    ###########################################################
    ### Set the output directory to store the model weights ###
    ###########################################################
    if args.out_dir:
        working_dir = "model_cnn_%s_ps.%dx%d_bs_%d_lr_%.2e" % (args.net_name, in_shape[0], in_shape[1], args.batch_size, args.lr)
        res_dir = os.path.join(args.out_dir, working_dir)
        try:
            os.makedirs(res_dir, exist_ok=True)
        except:
            print ("Directory already exists.")
            sys.exit()
        
        conf_dict = copy.copy(args.__dict__)
        cwd = os.getcwd()
        conf_dict['cwd'] = cwd
        conf_dict['cd.table'] = cd.table
        conf_dict['cd.metatable'] = cd.metatable
        conf_dict['cd.label_col'] = cd.label_col
        conf_dict['cd.num_classes'] = cd.num_classes

        conf_fn = os.path.join(res_dir, 'conf.json')
        with open(conf_fn, 'w') as f:
            json.dump(conf_dict, f, indent=2)


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

        ### Training ###
        ## Set training mode 
        
        eddl.set_mode(net, 1) # TRMODE = 1, TSMODE = 0
        
        # Smooth label param
        cd.set_smooth_eps(args.smooth_lab)

        cd.current_split = 0 ## Set the training split as the current one
        print("Epoch {:d}/{:d} - Training".format(e + 1, args.epochs),
              flush=True)
        
        cd.rewind_splits(shuffle=True)
        eddl.reset_loss(net)
        
        ### Looping across batches of training data
        pbar = tqdm(range(num_batches_tr))

        for b_index, b in enumerate(pbar):
            if args.val_split_indexes:
                x, y = cd.load_batch_cross(not_splits=val_splits+test_splits)
            else:
                x, y = cd.load_batch()
            
            tx, ty = [x], [y]
            eddl.train_batch(net, tx, ty)

            #print bratch train results
            loss = eddl.get_losses(net)[0]
            metr = eddl.get_metrics(net)[0]
            
            msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(e + 1, args.epochs, b + 1, num_batches_tr, loss, metr)
            pbar.set_postfix_str(msg)
            
            if b_index % num_batches_tr == 0:
                cd.rewind_splits(shuffle=True)
    
        loss_l.append(loss)
        acc_l.append(metr)

        pbar.close()

        ### Validation ###
        ## Set Test mode.

        eddl.set_mode(net, 0) 
        
        # Smooth label param
        cd.set_smooth_eps(0.0)
        
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
            
            eddl.forward(net, [x])
            output = eddl.getOutput(out)
            
            result = output.getdata()
            target = y.getdata()
            ca = accuracy(result, target)
            ce = cross_entropy(result, target)
            tot_loss.append(ce)
            tot_acc.append(ca)

            msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) loss: {:.3f}, acc: {:.3f}".format(
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

        
        ### Writing the results ###

        ## Save weights 
        if args.save_weights:
            print("Saving weights")
            path = os.path.join(res_dir, "promort_%s_weights_ep_%s_vacc_%.2f.bin" % (args.net_name, e, val_acc_l[-1]))
            eddl.save(net, path, "bin")
    
        # Dump history at the end of each epoch so if the job is interrupted data are not lost.
        if args.out_dir:
            history = {'loss': loss_l, 'acc': acc_l, 'val_loss': val_loss_l, 'val_acc': val_acc_l}
            pickle.dump(history, open(os.path.join(res_dir, 'history.pickle'), 'wb'))
        
        ### Early stopping implementation: Patience check
        if val_acc_l[-1] > val_acc_max:
            val_acc_max = val_acc_l[-1]
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt > args.patience:
            ## Exit and complete the training
            print ("Got maximum patience value... training completed")
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
    parser.add_argument("--lab-map", nargs='+', default = [], help='Specify label mapping. It is used to group original dataset labels to new class labels. For example: if the original dataset has [0, 1, 2, 3] classes using lab-map 0 0 1 1 maps the new label 0 to the old 0,1 classes and the new label 1 to the old 2,3 classes ')
    parser.add_argument("--save-weights", action="store_true", help='Network parameters are saved after each epoch')
    parser.add_argument("--augs-on", action="store_true", help='Activate data augmentations')
    parser.add_argument("--find-opt-lr", action="store_true", help='Scan learning rate with an increasing exponential law to find best lr')
    parser.add_argument("--full-mem", action="store_true", help='Activate data augmentations')
    parser.add_argument("--read-rgb", action="store_true", help='Load images in RGB format instead of the default BGR')
    parser.add_argument("--smooth-lab", type=float, metavar="FLOAT", default=0.0, help='smooth labeling parameter')
    parser.add_argument("--out-dir", metavar="DIR",
                        help="Specifies the output directory. If not set no output data is saved")
    parser.add_argument("--init-weights-fn", metavar="DIR", default='',
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
    parser.add_argument("--preprocess-mode", metavar="STR", default='div255',
                        help="Select the preprocessing mode of images. It can be useful for using imagenet pretrained nets (div255|pytorch|tf|caffe)")
    main(parser.parse_args())
