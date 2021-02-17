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

import argparse
import random
import sys
import os
from pathlib import Path

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

from cassandra_dataset import CassandraDataset

from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import trange, tqdm
import numpy as np
import pickle 

import models 
import gc

def get_net(net_name='vgg16', in_size=[256,256], num_classes=2, lr=1e-5, augs=False, gpu=True, init=eddl.HeNormal, dropout=None, l2_reg=None):
    
    ## Network definition
    in_ = eddl.Input([3, in_size[0], in_size[1]])
    
    if net_name == 'vgg16':
        out = models.VGG16_promort(in_, num_classes, init=init, l2_reg=l2_reg, dropout=dropout)
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
        eddl.CS_GPU([1,1], mem="low_mem") if gpu else eddl.CS_CPU()
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


def main(args):
    net_name = "vgg16"
    num_classes = 2
    size = [256, 256]  # size of images
    
    ### Get Network
    net_init = eddl.HeNormal
    net, dataset_augs = get_net(net_name='vgg16', in_size=size, num_classes=num_classes, lr=args.lr, augs=args.augs_on, gpu=args.gpu, init=net_init, dropout=args.dropout, l2_reg=args.l2_reg)
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
    #cd = CassandraDataset(ap, ['cassandra_db'])
    cd = CassandraDataset(ap, ['127.0.0.1'])

    # Check if file exists
    if Path(args.splits_fn).exists():
        # Load splits 
        cd.load_splits(args.splits_fn, batch_size=args.batch_size, augs=dataset_augs)
    else:
        print ("Split file not found")
        sys.exit(-1)

    print ('Number of batches for each split (train, val, test):', cd.num_batches)

    ################################
    #### Training and evaluation ###
    ################################

    print("Defining metric...", flush=True)
    
    metric = eddl.getMetric("categorical_accuracy")

    print("Starting training", flush=True)

    num_batches_tr = cd.num_batches[0]
    num_batches_val = cd.num_batches[1]
    
    loss_l = []
    acc_l = []
    val_loss_l = []
    val_acc_l = []
    

    #### Code used to find best learning rate. Comment it to perform an actual training
    #max_epochs = args.epochs
    #lr_start = 1e-6
    #lr_end = 1e-2
    #lr_f = lambda x: 10**(np.log10(lr_start) + ((np.log10(lr_end)-np.log10(lr_start))/max_epochs)*x)
    ####

    ### Main loop across epochs
    for e in range(args.epochs):
        ## SET LT
        #eddl.setlr(net, [lr_f(e)])

        ### Training 
        cd.current_split = 0 ## Set the training split as the current one
        print("Epoch {:d}/{:d} - Training".format(e + 1, args.epochs),
              flush=True)
        
        cd.rewind_splits(shuffle=True)
        eddl.reset_loss(net)
        total_metric = []
        total_loss = []

        ### Looping across batches of training data
        pbar = tqdm(range(num_batches_tr))
        
        for b_index, b in enumerate(pbar):
            x, y = cd.load_batch()
            x.div_(255.0)
            tx, ty = [x], [y]
            eddl.train_batch(net, tx, ty)
            
            #print bratch train results
            instances = (b_index+1) * args.batch_size
            loss = eddl.get_losses(net)[0]
            metr = eddl.get_metrics(net)[0]
            msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(e + 1, args.epochs, b + 1, num_batches_tr, loss, metr)
            pbar.set_postfix_str(msg)
            total_loss.append(loss)
            total_metric.append(metr)
    
        loss_l.append(np.mean(total_loss))
        acc_l.append(np.mean(total_metric))

        pbar.close()

        ### Evaluation on validation set batches
        cd.current_split = 1 ## Set validation split as the current one
        total_metric = []
        batch_loss = []
        
        print("Epoch %d/%d - Evaluation" % (e + 1, args.epochs), flush=True)
        
        pbar = tqdm(range(num_batches_val))

        for b_index, b in enumerate(pbar):
            n = 0
            x, y = cd.load_batch()
            x.div_(255.0)
            eddl.forward(net, [x])
            output = eddl.getOutput(out)

            sum_ = 0.0
            net.compute_loss()
            loss = eddl.get_losses(net)[0]
            batch_loss.append(loss)

            for k in range(x.getShape()[0]):
                result = output.select([str(k)])
                target = y.select([str(k)])
                ca = metric.value(target, result)
                total_metric.append(ca)
                sum_ += ca
                n += 1
            
            msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) loss: {:.3f}, acc: {:.3f} ".format(e + 1, args.epochs, b + 1, num_batches_val, loss, (sum_ / n))
            pbar.set_postfix_str(msg)
             
        pbar.close()
        val_batch_acc_avg = np.mean(total_metric)
        val_batch_loss_avg = np.mean(batch_loss)
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

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml_fn", default=None)
    parser.add_argument("--epochs", type=int, metavar="INT", default=50)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32)
    parser.add_argument("--lr", type=float, metavar="FLOAT", default=1e-5)
    parser.add_argument("--dropout", type=float, metavar="FLOAT", default=None)
    parser.add_argument("--l2_reg", type=float, metavar="FLOAT", default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--augs-on", action="store_true")
    parser.add_argument("--out-dir", metavar="DIR",
                        help="if set, save images in this directory")
    parser.add_argument("--init-weights-fn", metavar="DIR",
                        help="if set, a new set of weight are loaded to start the training")
    parser.add_argument("--splits-fn", metavar="STR",
                        help="if set, load splits data from a pickle file if it exists or save splits data to a pickle")
    parser.add_argument("--cassandra-pwd-fn", metavar="STR",
                        help="cassandra password")
    main(parser.parse_args())
