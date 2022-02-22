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

from promort_functions import get_data_augs, get_cassandra_dl

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


def main(args):
    net_name = args.net_name
    num_classes = args.num_classes
    in_shape = [3, 256, 256]  # shape of input data (channels, w_size, h_size) 
    
    ## Set weight file or get the weight file with the best validation accuracy if requested
    print ("Loading initialization weights")
    if args.weights_fn:
        weights_fn = args.weights_fn
    elif args.weights_path:
        weights_fn = get_best_weight_file(args.weights_path)
    else:
        print ("One of --weights_fn or --weights_path is required")
        sys.exit(-1)
   
    #####################################################################
    ### Get the built network and set data augmentations if requested ###
    #####################################################################
    net = models.get_net(net_name=args.net_name, in_shape=in_shape, num_classes=args.num_classes, 
            full_mem=args.full_mem, gpus=args.gpu, init_weights_fn=weights_fn)

    out = net.layers[-1]
    
    
    ## Check options
    print ("Creating output directory...")
    working_dir = "%s_%s" % (os.path.basename(args.splits_fn), os.path.basename(weights_fn))
    res_dir = os.path.join(args.out_dir, working_dir)
    os.makedirs(res_dir, exist_ok=True)
    fn = os.path.join(res_dir, "pred.csv")
    fd = open(fn, "w")
    fd.write("patch_id,normal_p,tumor_p,normal_gt,tumor_gt\n")
    
    ########################################
    ### Set database and read split file ###
    ########################################
    data_preprocs, read_rgb = get_data_augs(args.preprocess_mode, False, args.read_rgb)
   
    cd, num_batches_tr, num_batches_val = get_cassandra_dl(splits_fn=args.splits_fn, 
            num_classes = args.num_classes, seed=args.seed, cassandra_pwd_fn=args.cassandra_pwd_fn,
            batch_size=args.batch_size, dataset_augs=data_preprocs, 
            full_batches=True, 
            read_rgb=read_rgb,
            lab_map=args.lab_map)
    
    
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
    parser.add_argument("--seed", type=int, metavar="INT", default=None, help='Seed of the random generator to manage data load')
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
    parser.add_argument("--preprocess-mode", metavar="STR", default='div255',
                        help="Select the preprocessing mode of images. It can be useful for using imagenet pretrained nets (div255|pytorch|tf|caffe)")
    parser.add_argument("--read-rgb", action="store_true", help='Load images in RGB format instead of the default BGR')
    parser.add_argument("--lab-map", nargs='+', default = [], help='Specify label mapping. It is used to group original dataset labels to new class labels. For example: if the original dataset has [0, 1, 2, 3] classes using lab-map 0 0 1 1 maps the new label 0 to the old 0,1 classes and the new label 1 to the old 2,3 classes ')
    main(parser.parse_args())
