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

def get_net(in_size=[256,256], num_classes=2, lr=1e-5, augs=False, gpu=True):
    
    ## Network definition
    in_ = eddl.Input([3, in_size[0], in_size[1]])
    out = models.VGG16_promort(in_, num_classes)
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

def check_db_rows_and_split(cd, args, splits=[7,1,2]):
    data_size = args.data_size
    
    if args.db_rows_fn:
        if Path(args.db_rows_fn).exists():
            # If rows file exists load them from file 
            cd.load_rows(args.db_rows_fn)
        else:
            # If rows do not exist, read them from db and save to a pickle
            cd.read_rows_from_db()
            cd.save_rows(args.db_rows_fn)
    else:
        # If a db_rows_fn is not specified just read them from db
        cd.read_rows_from_db()
        
    ## Create the split and save it
    cd.init_datatable(table='promort.data_osk')
    cd.split_setup(batch_size=args.batch_size, split_ratios=splits,
                                 max_patches=data_size, augs=[])
    cd.save_splits('/tmp/splits.pckl')


def main(args):
    net_name = "vgg16"
    num_classes = 2
    size = [256, 256]  # size of images
    
    ### Get Network
    net, dataset_augs = get_net(in_size=size, num_classes=num_classes, lr=args.lr, augs=args.augs_on, gpu=args.gpu)
    out = net.layers[-1]
    
    ## Load weights if requested
    if args.init_weights_fn:
        print ("Loading initialization weights")
        eddl.load(net, args.init_weights_fn)
    
    ## Check options
    if args.out_dir:
        working_dir = "model_cnn_%s_ps.%d_bs_%d_lr_%e" % (net_name, size[0], args.batch_size, args.lr)
        res_dir = os.path.join(args.out_dir, working_dir)
        try:
            os.makedirs(res_dir, exist_ok=True)
        except:
            print ("Directory already exists.")
            sys.exit()

    if args.out_dir and args.save_img:
        save_img = True
    else:
        save_img = False
    
    #################################
    ### Set database to read data ###
    #################################

    if not args.cassandra_pwd_fn:
        cass_pass = getpass('Insert Cassandra password: ')	
    else:
        with open(args.cassandra_pwd_fn) as fd:
            cass_pass = fd.readline().rstrip()

    # create cassandra reader
    ap = PlainTextAuthProvider(username='prom', password=cass_pass)
    #cd = CassandraDataset(ap, ['cassandra_db'])
    cd = CassandraDataset(ap, ['127.0.0.1'])

    cd.init_listmanager(meta_table='promort.ids_osk', id_col='patch_id',
                        split_ncols=2, num_classes=num_classes, 
                        partition_cols=['sample_name', 'sample_rep', 'label'])
    
    data_size = args.data_size
    
    if args.splits_fn:
        # Check if file exists
        if Path(args.splits_fn).exists():
            # Load splits 
            cd.load_splits(args.splits_fn, batch_size=args.batch_size, augs=dataset_augs)
        else:
            check_db_rows_and_split(cd, args)
            cd.save_splits(args.splits_fn)
            
    else:
        check_db_rows_and_split(cd, args)

    print ('Number of batches for each split (train, val, test):', cd.num_batches)

    ################################
    #### Training and evaluation ###
    ################################

    print("Defining metric...", flush=True)
    
    metric = eddl.getMetric("categorical_accuracy")

    print("Starting training", flush=True)

    ### Main loop across epochs
    num_batches_tr = cd.num_batches[0]
    num_batches_val = cd.num_batches[1]
    
    indices = list(range(args.batch_size))

    loss_l = []
    acc_l = []
    val_acc_l = []
    

    ### Main loop across epochs
    
    for e in range(args.epochs):

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
            #loss = net.fiterr[0]/instances
            #metr = net.fiterr[1]/instances
            msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(e + 1, args.epochs, b + 1, num_batches_tr, loss, metr)
            pbar.set_postfix_str(msg)
            total_loss.append(loss)
            total_metric.append(metr)

        loss_l.append(np.mean(total_loss))
        acc_l.append(np.mean(total_metric))

        pbar.close()
        
        if args.save_weights:
            print("Saving weights")
            path = os.path.join(res_dir, "promort_checkpoint_%s.bin" % e)
            eddl.save(net, path, "bin")

        ### Evaluation on validation set batches
        cd.current_split = 1 ## Set validation split as the current one
        total_metric = []
        
        if save_img:
            current_path = os.path.join(args.out_dir, "Epoch_%d" % e)
            for c in range(num_classes):
                c_dir = os.path.join(current_path, c)
                os.makedirs(c_dir, exist_ok=True)

        print("Epoch %d/%d - Evaluation" % (e + 1, args.epochs), flush=True)
        
        pbar = tqdm(range(num_batches_val))

        for b_index, b in enumerate(pbar):
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
                
                if save_img:
                    result_a = np.array(result, copy=False)
                    target_a = np.array(target, copy=False)
                    classe = np.argmax(result_a).item()
                    gt_class = np.argmax(target_a).item()
                    single_image = x.select([str(k)])
                    img_t = ecvl.TensorToView(single_image)
                    img_t.colortype_ = ecvl.ColorType.BGR
                    single_image.mult_(255.)
                    filename = d.samples_[d.GetSplit()[n]].location_[0]
                    head, tail = os.path.splitext(os.path.basename(filename))
                    bname = "%s_gt_class_%s.png" % (head, gt_class)
                    cur_path = os.path.join(
                        current_path, classe, bname
                    )
                    ecvl.ImWrite(cur_path, img_t)
                
                n += 1
            msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) - acc: {:.3f} ".format(e + 1, args.epochs, b + 1, num_batches_val, (sum_ / n))
            pbar.set_postfix_str(msg)
             
        pbar.close()
        total_avg = np.mean(total_metric)
        val_acc_l.append(total_avg)

        print("Total categorical accuracy: {:.2f}\n".format(total_avg))

    history = {'loss': loss_l, 'acc': acc_l, 'val_acc': val_acc_l}

    pickle.dump(history, open(os.path.join(res_dir, 'history.pickle'), 'wb'))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml_fn", default=None)
    parser.add_argument("--epochs", type=int, metavar="INT", default=50)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32)
    parser.add_argument("--lr", type=float, metavar="FLOAT", default=1e-5)
    parser.add_argument("--data-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--save-img", action="store_true")
    parser.add_argument("--augs-on", action="store_true")
    parser.add_argument("--out-dir", metavar="DIR",
                        help="if set, save images in this directory")
    parser.add_argument("--init-weights-fn", metavar="DIR",
                        help="if set, a new set of weight are loaded to start the training")
    parser.add_argument("--db-rows-fn", metavar="STR",
                        help="if set, load db rows from a pickle file if it exists or save rows to a pickle after reading image metadata from db")
    parser.add_argument("--splits-fn", metavar="STR",
                        help="if set, load splits data from a pickle file if it exists or save splits data to a pickle")
    parser.add_argument("--cassandra-pwd-fn", metavar="STR",
                        help="cassandra password")
    main(parser.parse_args())
