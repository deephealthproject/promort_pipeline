from cassandra_dataset import CassandraDataset

from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import trange, tqdm

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl

def VGG16(in_layer, num_classes):
    x = in_layer
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 64, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 128, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 256, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 512, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 512, [3, 3])), [2, 2], [2, 2])
    x = eddl.Reshape(x, [-1])
    x = eddl.ReLu(eddl.Dense(x, 256))
    x = eddl.Softmax(eddl.Dense(x, num_classes))
    return x

def cassandra_fit(cass_ds, net, epochs=3):
    cs = 0 # current split = training
    cass_ds.current_split = cs
    num_batches = cass_ds.num_batches[cs]
    # loop through the epochs
    for e in range(epochs):
        cass_ds.rewind_splits(cs, shuffle=True)
        eddl.reset_loss(net)
        # loop through batches
        for b in trange(num_batches):
            x,y = cass_ds.load_batch()
            x.div_(255.0)
            tx, ty = [x], [y]
            eddl.train_batch(net, tx, ty)
        # print loss
        eddl.print_loss(net, b)
        print()


def test_dataset():
    num_classes = 2
    size = [256, 256]  # size of images
    training_augs = ecvl.SequentialAugmentationContainer([
        #ecvl.AugResizeDim(size),
        ecvl.AugMirror(.5),
        ecvl.AugFlip(.5),
        ecvl.AugRotate([-180, 180]),
        #ecvl.AugAdditivePoissonNoise([0, 10]),
        #ecvl.AugGammaContrast([0.5, 1.5]),
        #ecvl.AugGaussianBlur([0, 0.8]),
        #ecvl.AugCoarseDropout([0, 0.3], [0.02, 0.05], 0.5),
    ])
    
    dataset_augs = [training_augs, None, None]

    in_ = eddl.Input([3, size[0], size[1]])
    out = VGG16(in_, num_classes)
    net = eddl.Model([in_], [out])
    eddl.build(
        net,
        eddl.rmsprop(1e-5),
        #eddl.sgd(0.001, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU([1], mem="low_mem") # eddl.CS_CPU()
        )
    eddl.summary(net)    

    # Read Cassandra password
    try: 
        from private_data import cass_pass
    except ImportError:
        cass_pass = getpass('Insert Cassandra password: ')
        
    # Init Cassandra dataset with server address
    ap = PlainTextAuthProvider(username='prom', password=cass_pass)
    cd = CassandraDataset(ap, ['cassandra_db'])

    # Flow 0: read rows from db, create splits and save everything
    cd.init_listmanager(meta_table='promort.ids_1', id_col='patch_id',
                        split_ncols=2, num_classes=2, 
                        partition_cols=['sample_name', 'sample_rep', 'label'])
    cd.read_rows_from_db()
    cd.save_rows('/tmp/rows.pckl')
    cd.init_datatable(table='promort.data_1')
    cd.split_setup(batch_size=32, split_ratios=[7,2,1],
                   max_patches=100000, augs=[])
    cd.save_splits('/tmp/splits.pckl')

    ## Flow 1: read saved rows, create and save splits
    #cd.load_rows('/tmp/rows.pckl')
    #cd.init_datatable(table='promort.data_1')
    #cd.split_setup(batch_size=32, split_ratios=[7,2,1],
    #               max_patches=100000, augs=[])
    #cd.save_splits('/tmp/splits.pckl')

    ## Flow 2: read saved splits
    #cd.load_splits('/tmp/splits.pckl', batch_size=32, augs=[])
    
    
    ## fit generator
    cassandra_fit(cd, net, epochs=1)

    # change batch size
    cd.set_batchsize(16)
    cassandra_fit(cd, net, epochs=1)

    # change partitioning and balance
    cd.split_setup(max_patches=1000, split_ratios=[10,1,1], balance=[2,1])
    cassandra_fit(cd, net, epochs=1)

