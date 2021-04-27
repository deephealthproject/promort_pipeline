from cassandra.auth import PlainTextAuthProvider
from cassandra_dataset import CassandraDataset

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

import models


class EnvLoader():
    def __init__(self, inet_pass, num, augs_on, batch_size,
                 max_patches, cass_row_fn, cass_datatable,
                 net_name, size, num_classes, lr, gpus, net_init, 
                 dropout, l2_reg):
        self.inet_pass = inet_pass
        self.num = num
        self.cd = None
        self.cass_row_fn = cass_row_fn
        self.cass_datatable = cass_datatable
        self.max_patches = max_patches
        self.batch_size = batch_size
        self.augs_on = augs_on
        self.seed = None
        self.net_name = net_name
        self.size = size
        self.num_classes = num_classes
        self.lr = lr
        self.gpus = gpus
        self.net_init = net_init
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.net = None
    def split_setup(self, seed):
        self.seed = seed
        self.cd.split_setup(batch_size=self.batch_size,
                            split_ratios=[1]*self.num,
                            max_patches=self.max_patches, seed=seed)
    def cassandra_setup(self, seed):
        print('####### Starting Cassandra dataset')
        ap = PlainTextAuthProvider(username='inet',
                                   password=self.inet_pass)
        cd = CassandraDataset(ap, ['cassandra_db'])
        cd.load_rows(self.cass_row_fn)
        cd.init_datatable(table=self.cass_datatable)
        self.cd = cd
        self.split_setup(seed)
    def start(self, seed):
        # cassandra
        if (self.cd is None): # first execution, create data structure
            self.cassandra_setup(seed)
        if (self.seed != seed): # seed changed, reshuffle all
            self.split_setup(seed)
        # eddl network
        if (self.net is None):
            self.net_setup()
        
    def net_setup(self):
        in_ = eddl.Input([3, self.size[0], self.size[1]])

        if self.net_name == 'vgg16':
            out = models.VGG16(in_, self.num_classes, init=self.net_init, 
                               l2_reg=self.l2_reg, dropout=self.dropout)
        else:
            print('model %s not available' % self.net_name)
            sys.exit(-1)

        self.net = eddl.Model([in_], [out])
        eddl.build(
            self.net,
            eddl.rmsprop(self.lr),
            ["soft_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_GPU(self.gpus, mem="low_mem") if self.gpus else eddl.CS_CPU()
            )


