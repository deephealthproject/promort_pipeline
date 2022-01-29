"""
PROMORT example.
"""

import argparse
import random
import sys
import os
import glob
from pathlib import Path

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

from cassandradl import CassandraDataset

from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import trange, tqdm
import numpy as np
import pickle 

import models 
import gc


def get_best_weight_file(path):
    ### Get the weight file which gave the maximum validation accuracy
    hfn = os.path.join(path, 'history.pickle')
    history = pickle.load(open(hfn, 'rb'))
    val_acc = history['val_acc']
    ep_acc_l = [(i, val) for i,val in enumerate(val_acc)]
    ep_acc_l_sorted = sorted(ep_acc_l, key=lambda x: x[1], reverse=True)
    max_epoch = ep_acc_l_sorted[0][0]
    fn = [i for i in glob.glob(os.path.join(path, "*.bin")) if "ep_%d_vacc" % max_epoch in i][0]
   
    print ("Weight file used: %s" % fn)
    return fn


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
        eddl.rmsprop(1e-6),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(gpus, mem=mem, lsb=lsb) if gpus else eddl.CS_CPU()
        )

    eddl.summary(net)

    return net


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
    net_name = args.net_name
    num_classes = args.num_classes
    size = [256, 256]  # size of images
    
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

    ### Get Network
    net = get_net(net_name=net_name, in_size=size, num_classes=num_classes, gpus=gpus, mem=mem)
    out = net.layers[-1]
    
    ## Load weights if requested
    print ("Loading initialization weights")
    if args.weights_fn:
        weights_fn = args.weights_fn
    elif args.weights_path:
        weights_fn = get_best_weight_file(args.weights_path)
    else:
        print ("One of --weights_fn or --weights_path is required")
        sys.exit(-1)
   
    eddl.load(net, weights_fn)

    ## Check options
    print ("Creating output directory...")
    working_dir = "%s_%s" % (os.path.basename(args.splits_fn), os.path.basename(weights_fn))
    res_dir = os.path.join(args.out_dir, working_dir)
    os.makedirs(res_dir, exist_ok=True)
    fn = os.path.join(res_dir, "pred.csv")
    fd = open(fn, "w")
    fd.write("patch_id,normal_p,tumor_p,normal_gt,tumor_gt\n")
    
    
    #################################
    ### Set database to read data ###
    #################################

    if not args.cassandra_pwd_fn:
        cass_pass = getpass('Insert Cassandra password: ')	
    else:
        with open(args.cassandra_pwd_fn) as fdcass:
            cass_pass = fdcass.readline().rstrip()

    # create cassandra reader
    ap = PlainTextAuthProvider(username='prom', password=cass_pass)
    cd = CassandraDataset(ap, ['156.148.70.72'])

    if Path(args.splits_fn).exists():
        # Load splits 
        cd.load_splits(args.splits_fn, batch_size=args.batch_size, augs=[], whole_batches=True)
    else:
        print ("Split file %s not found" % args.splits_fn)
        sys.exit(-1)
        
    print ('Number of batches for each split (train, val, test):', cd.num_batches)

    ###################
    #### Evaluation ###
    ###################

    print("Defining metric...", flush=True)
    
    si = args.split_index
    num_batches = cd.num_batches[si]
    cd.current_split = si ## Set the current split 
    cd.rewind_splits(shuffle=False)
    rows = cd.row_keys[cd.split[si]]
    indexes = cd.current_index[si]

    eddl.reset_loss(net)
    metric = eddl.getMetric("categorical_accuracy") 
    
    ## Set net to test mode
    eddl.set_mode(net, 0)
    
    ### Evaluation on validation set batches
    total_metric = []
    
    pbar = tqdm(range(num_batches))

    for b_index, b in enumerate(pbar):
        n = 0
        x, y = cd.load_batch()
        x_dim = x.getShape()[0]
        rescale_tensor(x)
        eddl.forward(net, [x])
        output = eddl.getOutput(out)
        
        sum_ = 0.0
        ids = rows[indexes-x_dim:indexes] 
        
        for k in range(x_dim):
            result = output.select([str(k)])
            target = y.select([str(k)])
            ca = metric.value(target, result)
            total_metric.append(ca)
            sum_ += ca
            
            #p_id = str(ids[k][cd.id_col])
            p_id = str(ids[k])
            result_np = result.getdata()[0]
            gt_np = target.getdata()[0]
            normal_p = result_np[0]
            tumor_p = result_np[1]
            normal_gt = gt_np[0]
            tumor_gt = gt_np[1]

            fd.write('%s,%.2f,%.2f,%.2f,%.2f\n' % (p_id, normal_p, tumor_p, normal_gt, tumor_gt))

            n += 1
        
        indexes = cd.current_index[si]
        
        msg = "Batch {:d}/{:d}) - acc: {:.3f} ".format(b + 1, num_batches, (sum_ / n))
        pbar.set_postfix_str(msg)
         
    pbar.close()
    fd.close()
    total_avg = np.mean(total_metric)

    print("Total categorical accuracy: {:.3f}\n".format(total_avg))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32)
    parser.add_argument("--out-dir", metavar="DIR", required=True,
                        help="if set, save images in this directory")
    parser.add_argument("--weights-fn", metavar="DIR", 
                        help="filename of a weight file")
    parser.add_argument("--weights-path", metavar="DIR",
                        help="path to get the weight files which resulted in the best validation accuracy score")
    parser.add_argument("--splits-fn", metavar="STR", required=True,
                        help="split filename. It is pickle file")
    parser.add_argument("--split-index", type=int, default=1,
                        help="set the split that has to be evaluated")
    parser.add_argument("--cassandra-pwd-fn", metavar="STR", default='/tmp/cassandra_pass.txt',
                        help="cassandra password")
    parser.add_argument("--net-name", metavar="STR", default='vgg16_tumor',
                        help="Select the neural net (vgg16|vgg16_tumor|vgg16_gleason|resnet50)")
    parser.add_argument("--num-classes", type=int, metavar="INT", default=2, help='Number of categories in the dataset')
    parser.add_argument("--full-mem", action="store_true", help='Activate data augmentations')
    parser.add_argument("--gpu", nargs='+', default = [], help='Specify GPU mask. For example: 1 to use only gpu0; 1 1 to use gpus 0 and 1; 1 1 1 1 to use gpus 0,1,2,3')
    main(parser.parse_args())
