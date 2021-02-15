"""
PROMORT example.
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

def get_net(in_size=[256,256], num_classes=2, gpu=True):
    
    ## Network definition
    in_ = eddl.Input([3, in_size[0], in_size[1]])
    out = models.VGG16_promort(in_, num_classes)
    net = eddl.Model([in_], [out])
    eddl.build(
        net,
        eddl.rmsprop(1e-6),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        #eddl.CS_GPU([1,1], mem="low_mem") if gpu else eddl.CS_CPU()
        eddl.CS_GPU([1], mem="low_mem") if gpu else eddl.CS_CPU()
        )

    eddl.summary(net)
    eddl.setlogfile(net, "promort_VGG16_classification")
   
    return net


def main(args):
    net_name = "vgg16"
    num_classes = 2
    size = [256, 256]  # size of images
    
    ### Get Network
    net = get_net(in_size=size, num_classes=num_classes, gpu=args.gpu)
    out = net.layers[-1]
    
    ## Load weights if requested
    print ("Loading initialization weights")
    eddl.load(net, args.weights_fn)
    
    ## Check options
    print ("Creating output directory...")
    working_dir = "%s_%s" % (os.path.basename(args.splits_fn), os.path.basename(args.weights_fn))
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
    #cd = CassandraDataset(ap, ['cassandra_db'])
    cd = CassandraDataset(ap, ['127.0.0.1'])

    try:
        cd.load_splits(args.splits_fn, batch_size=args.batch_size, augs=[])
    except:
        print ("Split file not found")
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
    
    ### Evaluation on validation set batches
    total_metric = []
    
    pbar = tqdm(range(num_batches))

    for b_index, b in enumerate(pbar):
        ids = rows[indexes-cd.batch_size:indexes] 
        
        n = 0
        x, y = cd.load_batch()
        x.div_(255.0)
        eddl.forward(net, [x])
        output = eddl.getOutput(out)
        
        sum_ = 0.0
        
        for k in range(x.getShape()[0]):
            result = output.select([str(k)])
            target = y.select([str(k)])
            ca = metric.value(target, result)
            total_metric.append(ca)
            sum_ += ca
            
            p_id = str(ids[k]['patch_id'])
            result_np = result.getdata()[0]
            gt_np = target.getdata()[0]
            normal_p = result_np[1]
            tumor_p = result_np[0]
            normal_gt = gt_np[1]
            tumor_gt = gt_np[0]

            fd.write('%s,%.2f,%.2f,%.2f,%.2f\n' % (p_id, normal_p, tumor_p, normal_gt, tumor_gt))

            n += 1
        
        indexes = cd.current_index[si]
        
        msg = "Batch {:d}/{:d}) - acc: {:.3f} ".format(b + 1, num_batches, (sum_ / n))
        pbar.set_postfix_str(msg)
         
    pbar.close()
    fd.close()
    total_avg = np.mean(total_metric)

    print("Total categorical accuracy: {:.2f}\n".format(total_avg))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out-dir", metavar="DIR", required=True,
                        help="if set, save images in this directory")
    parser.add_argument("--weights-fn", metavar="DIR", required=True,
                        help="if set, a new set of weight are loaded to start the training")
    parser.add_argument("--splits-fn", metavar="STR", required=True,
                        help="split filename. It is pickle file")
    parser.add_argument("--split-index", type=int, default=1,
                        help="set the split that has to be evaluated")
    parser.add_argument("--cassandra-pwd-fn", metavar="STR",
                        help="cassandra password")
    main(parser.parse_args())
