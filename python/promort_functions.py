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


def rescale_tensor(x, vgg_pretrained=True, mode='tf'):
    if mode == 'tf' and vgg_pretrained:
        # Data in -1,1 interval
        x.div_(255.0)
        x.mult_(2)
        x.add_(-1)
        return 
    elif mode == 'torch' or not vgg_pretrained:
        # Normalization
        mean=Tensor([0.485, 0.456, 0.406])
        std=Tensor([0.229, 0.224, 0.225])
        x.div_(255.0)
        return 
    else:
        x.div_(255.0)


def get_data_augs(augs=False):
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

    return dataset_augs

    ######################################
    ### Cassandra Dataloader functions ###
    ######################################

def get_cassandra_dl(splits_fn=None, data_table=None, smooth_lab=0.0, seed=1234, cassandra_pwd_fn='/tmp/cassandra_pass.txt',
        batch_size=32, dataset_augs=[], whole_batches=True, val_split_indexes=None, test_split_indexes=None):

    if not cassandra_pwd_fn:
        cass_pass = getpass('Insert Cassandra password: ')	
    else:
        with open(cassandra_pwd_fn) as fd:
            cass_pass = fd.readline().rstrip()

    # create cassandra reader
    ap = PlainTextAuthProvider(username='prom', password=cass_pass)
    cd = CassandraDataset(ap, ['156.148.70.72'], seed=seed)
    
    # Smooth label param
    cd.smooth_eps = smooth_lab

    # Check if file exists
    if Path(splits_fn).exists():
        # Load splits 
        cd.load_splits(splits_fn, batch_size=batch_size, augs=dataset_augs, whole_batches=whole_batches)
    else:
        print ("Split file %s not found" % splits_fn)
        sys.exit(-1)

    # Check if a new data table has to be set
    if data_table:
        print (f"Setting data table to {data_table}")
        cd.table = data_table

    print (f"Using data table {data_table}")
    print ('Number of batches for each split (train, val, test):', cd.num_batches)
    
    ## validation index check and creation of split indexes lists
    n_splits = cd.num_splits
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
    
    return cd, num_batches_tr, num_batches_val
