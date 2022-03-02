import sys
import numpy as np
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

from cassandradl import CassandraDataset
from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from pathlib import Path

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

def get_data_augs(preprocess_mode='div255', augs=False, center_crop=None, read_rgb=False):
    if preprocess_mode == 'div255':
        preprocess_l = [ecvl.AugToFloat32(1.), ecvl.AugDivBy255()] ## Image pixel in the [0,1] range
    elif preprocess_mode == 'caffe':
        preprocess_l = [ecvl.AugToFloat32(1.), ecvl.AugNormalize([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])]  ## Caffe Normalization for imagenet vgg16 tf models. BGR images assumed
    elif preprocess_mode == 'tf':
        preprocess_l = [ecvl.AugToFloat32(1.), ecvl.AugScaleFromTo(0., 255., -1.0, 1.0)] # Normalization for imagenet Resnet50 pretrained models. RGB images assumed
        read_rgb = True
    elif preprocess_mode == 'pytorch':
        preprocess_l = [ecvl.AugToFloat32(1.), ecvl.AugDivBy255(),  ecvl.AugNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])] ## Pytorch Normalization for imagenet pretrained models. RGB images assumed
        read_rgb = True
    else:
        print ("No preprocessing requested")
        preprocess_l = [] 
    
    if augs:
        ## Set augmentations
        trn_augs_l = [
            ecvl.AugMirror(.5),
            ecvl.AugFlip(.5),
            ecvl.AugRotate([-15, 15])
        ]
        
        val_augs_l = []
        
    else:
        trn_augs_l = []
        val_augs_l = []

    training_augs_l = trn_augs_l + preprocess_l
    validation_augs_l = val_augs_l + preprocess_l
    #print(training_augs_l, validation_augs_l)
    training_augs = ecvl.SequentialAugmentationContainer(training_augs_l)
    validation_augs = ecvl.SequentialAugmentationContainer(validation_augs_l)
     
    #if center_crop: ADD this feature
    
    dataset_augs = (training_augs, validation_augs)
    
    return dataset_augs, read_rgb

######################################
### Cassandra Dataloader functions ###
######################################

def get_cassandra_dl(splits_fn=None, num_classes=None, data_table=None, smooth_lab=0.0, seed=1234, cassandra_pwd_fn='/tmp/cassandra_pass.txt',
        batch_size=32, dataset_augs=[], val_split_indexes=None, test_split_indexes=None, 
        addr='156.148.70.72', user='prom', read_rgb=False, lab_map=[], full_batches=True):

    print ('READ RGB: %r' % read_rgb)
    if not cassandra_pwd_fn:
        cass_pass = getpass('Insert Cassandra password: ')	
    else:
        with open(cassandra_pwd_fn) as fd:
            cass_pass = fd.readline().rstrip()

    # create cassandra reader
    ap = PlainTextAuthProvider(username=user, password=cass_pass)
    cd = CassandraDataset(ap, [addr], seed=seed)
    
    # Smooth label param
    cd.smooth_eps = smooth_lab

    # Check if file exists
    if Path(splits_fn).exists():
        # Load splits 
        cd.load_splits(splits_fn)
    else:
        print ("Split file %s not found" % splits_fn)
        sys.exit(-1)


    # set batchsize
    cd.set_batchsize(bs=batch_size, full_batches=full_batches)
    
    # Remap lables if requested 
    if lab_map:
        lab_map = [int(i) for i in lab_map]
        cd.set_label_map(lab_map)
    
    cd.num_classes = num_classes 
    
    # Check if a new data table has to be set
    if data_table:
        print (f"Setting data table to {data_table}")
        cd.init_datatable(data_table)

    ## Set RGB images if requested 
    cd.set_rgb(read_rgb)

    print (f"Using data table {data_table}")
    print ('Number of batches for each split (train, val, test):', cd.num_batches)

    ## validation index check, creation of split indexes lists and data augmentation application
    n_splits = cd.num_splits
    trn_augs = dataset_augs[0]
    val_augs = dataset_augs[1]
    augs_l = [trn_augs for i in range(n_splits)]

    if val_split_indexes:
        out_indexes = [i for i in val_split_indexes if i > (n_splits-1)]
        if out_indexes:
            print (f"Not valid validation split index: {out_indexes}")
            sys.exit(-1)

        val_splits = val_split_indexes
        test_splits = test_split_indexes
        train_splits = [i for i in range(n_splits) if (i not in val_splits) and (i not in test_splits)]
        num_batches_tr = np.sum([cd.num_batches[i] for i in train_splits])
        num_batches_val = np.sum([cd.num_batches[i] for i in val_splits])
        
        for v_i in val_split_indexes:
            print ("Setting val preprocessing functions to split %d" % v_i)
            augs_l[v_i] = val_augs

        print ("Train splits: %r" % train_splits)
        print ("Val splits: %r" % val_splits)
        print ("Test splits: %r" % test_splits)
    
    else:
        train_splits = []
        val_splits = []
        test_splits = []
        if n_splits == 1:
            num_batches_tr = cd.num_batches[0]
            num_batches_val = 0
        else:
            num_batches_tr = cd.num_batches[0]
            num_batches_val = cd.num_batches[1]
            augs_l[1] = val_augs

    if test_split_indexes:
        for t_i in test_split_indexes:
            print ("Setting no preprocessing functions to split %d" % t_i)
            augs_l[t_i] = None

    ## Setting preprocessing abd  data augmentations
    cd.set_augmentations(augs_l)

    #print ("Preprocessin and augmentation list is:")
    #print (augs_l)

    return cd, num_batches_tr, num_batches_val, train_splits, val_splits, test_splits
