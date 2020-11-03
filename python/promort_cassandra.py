"""
PROMORT example.
"""

import argparse
import random
import sys

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

from cassandra_dataset import CassandraDataset

from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import trange, tqdm

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
        eddl.CS_GPU([1], mem="low_mem") if gpu else eddl.CS_CPU()
        )

    eddl.summary(net)
    eddl.setlogfile(net, "promort_VGG16_classification")
   
    if augs:
        ## Set augmentations
        training_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugResizeDim(size),
            ecvl.AugMirror(.5),
            ecvl.AugFlip(.5),
            ecvl.AugRotate([-45, 45])
            #ecvl.AugAdditivePoissonNoise([0, 10]),
            #ecvl.AugGammaContrast([0.5, 1.5]),
            #ecvl.AugGaussianBlur([0, 0.8]),
            #ecvl.AugCoarseDropout([0, 0.3], [0.02, 0.05], 0.5)
        ])
        
        validation_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugResizeDim(size),
        ])
        
        dataset_augs = [training_augs, validation_augs, None]
    
    else:
        dataset_augs = [None, None, None]

    return net, dataset_augs


def main(args):
    num_classes = 2
    size = [256, 256]  # size of images
    
    ### Get Network
    net, dataset_augs = get_net(in_size=size, num_classes=num_classes, lr=args.lr, augs=args.augs_on, gpu=args.gpu)
    out = net.layers[-1]

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
    
    cd = CassandraDataset(ap, ['127.0.0.1'],
                          table='promort.data_by_ids',
                          id_col='patch_id', num_classes=num_classes)
    
    if args.load_rows_fn:
        cd.load_rows(args.load_rows_fn)
    else:
        cd.read_rows_from_db(meta_table='promort.ids_by_metadata',
                             partition_cols=['sample_name', 'label'])
        if args.save_rows_fn:
            cd.save_rows(args.save_rows_fn)

    ### Split creation
    data_size = args.data_size
    cd.split_setup(batch_size=32, split_ratios=[8, 2],
                   max_patches=data_size, augs=dataset_augs, seed=None)


    #################
    #### Training ###
    #################

    print("Defining metric...", flush=True)
    
    metric = eddl.getMetric("categorical_accuracy")

    print("Starting training", flush=True)

    ### Main loop across epochs
    num_batches_tr = cd.num_batches[0]
    num_batches_val = cd.num_batches[1]
    
    indices = list(range(args.batch_size))

    ### Main loop across epochs
    for e in range(args.epochs):
        print("Epoch {:d}/{:d} - Training".format(e + 1, args.epochs),
              flush=True)
        if args.out_dir:
            current_path = os.path.join(args.out_dir, "Epoch_%d" % e)
            for c in d.classes_:
                c_dir = os.path.join(current_path, c)
                os.makedirs(c_dir, exist_ok=True)

        cd.rewind_splits(shuffle=True)
        eddl.reset_loss(net)
        total_metric = []
        
        ### Looping across batches of training data
        pbar = tqdm(range(num_batches_tr))
        for b_index, b in enumerate(pbar):
            x, y = cd.load_batch(0)
            x.div_(255.0)
            tx, ty = [x], [y]
            eddl.train_batch(net, tx, ty)
            
            #print bratch train results
            instances = (b_index+1) * args.batch_size
            loss = net.fiterr[0]/instances
            metr = net.fiterr[1]/instances
            msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(e + 1, args.epochs, b + 1, num_batches_tr, loss, metr)
            pbar.set_postfix_str(msg)
            
        pbar.close()
        
        if args.save_weights:
            print("Saving weights")
            eddl.save(net, "promort_checkpoint_%s.bin" % e, "bin")

        ### Evaluation on validation set batches
        print("Epoch %d/%d - Evaluation" % (e + 1, args.epochs), flush=True)
        
        pbar = tqdm(range(num_batches_val))

        for b_index, b in enumerate(pbar):
            n = 0
            x, y = cd.load_batch(1)
            x.div_(255.0)
            #eddl.forward(net, [x])
            #output = eddl.getOutput(out)
            #net.reset()
            net.forward([x])
            #output = out.output

            sum_ = 0.0
            """
            for k in range(x.getShape()[0]):
                result = output.select([str(k)])
                target = y.select([str(k)])
                ca = metric.value(target, result)
                total_metric.append(ca)
                sum_ += ca
                
                if args.out_dir:
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
                        current_path, d.classes_[classe], bname
                    )
                    ecvl.ImWrite(cur_path, img_t)
            
                n += 1
            
            msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) - acc: {:.2f} ".format(e + 1, args.epochs, b + 1, num_batches_tr, (sum_ / args.batch_size))
            pbar.set_postfix_str(msg)
            """ 
        pbar.close()
        #total_avg = sum(total_metric) / len(total_metric)
        #print("Total categorical accuracy: {:.2f}\n".format(total_avg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_ds", metavar="INPUT_DATASET")
    parser.add_argument("--epochs", type=int, metavar="INT", default=50)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32)
    parser.add_argument("--lr", type=float, metavar="FLOAT", default=1e-5)
    parser.add_argument("--data-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--augs-on", action="store_true")
    parser.add_argument("--out-dir", metavar="DIR",
                        help="if set, save images in this directory")
    parser.add_argument("--load-rows-fn", metavar="STR",
                        help="if set, load db rows from a pickle file and do read image metadata from db")
    parser.add_argument("--save-rows-fn", metavar="STR",
                        help="if set, save db rows to a pickle file after reading image metadata from db")
    parser.add_argument("--cassandra-pwd-fn", metavar="STR",
                        help="cassandra password")
    main(parser.parse_args())
