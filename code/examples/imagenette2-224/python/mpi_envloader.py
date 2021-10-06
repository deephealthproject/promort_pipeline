import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

import yaml
import copy
import random
import os
import models

from OPT_MPI import mpi_env, sgd_mpi
import subprocess

class EnvLoader():
    def __init__(self, in_yml, n_sync, batch_size, net_init, augs_on,
                 net_name, size, num_classes, lr, gpus,
                 dropout, l2_reg, seed):
        
        self.MP = mpi_env(n_sync, bl=512)
        self.yml = in_yml
        self.per_rank_yml_l = []
        self.dataset_aug = None
        self.augs_on = augs_on
        self.num = self.MP.mpi_size
        self.batch_size = batch_size
        self.seed = seed
        self.net_name = net_name
        self.net_init = net_init
        self.size = size
        self.num_classes = num_classes
        self.lr = lr
        self.gpus = gpus
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.net = None
        self.ngpus = 0
        if gpus:
            # Discover how many gpus can be used
            result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
            self.ngpus = len(result.stdout.decode('utf-8').split('\n'))-1

    def start(self, seed):
        # augmentation setup 
        self.aug_setup()

        # yml per rank filename setup
        self.yml_setup()

        # eddl network
        if (self.net is None):
            self.net_setup()

    def aug_setup(self):
        if self.augs_on:
            training_augs = ecvl.SequentialAugmentationContainer([
            #ecvl.AugResizeDim(self.size),
            ecvl.AugMirror(.5),
            ecvl.AugFlip(.5),
            ecvl.AugRotate([-180, 180]),
            ecvl.AugAdditivePoissonNoise([0, 10]),
            ecvl.AugGammaContrast([0.5, 1.5]),
            ecvl.AugGaussianBlur([0, 0.8]),
            ecvl.AugCoarseDropout([0, 0.3], [0.02, 0.05], 0.5)
            ])
        
            validation_augs = ecvl.SequentialAugmentationContainer([
            #ecvl.AugResizeDim(self.size),
            ])
            
            self.dataset_augs = ecvl.DatasetAugmentations(
            [training_augs, validation_augs, None]
            )
        
        else:
            training_augs = ecvl.SequentialAugmentationContainer([])
            validation_augs = ecvl.SequentialAugmentationContainer([])

        self.dataset_augs = ecvl.DatasetAugmentations(
        [training_augs, validation_augs, None]
        )

    def yml_setup(self):
        for r in range(self.MP.mpi_size):
            fn = os.path.splitext(self.yml)[0] + '_%d' % self.MP.mpi_rank + '_%d' % r + '.yml'
            self.per_rank_yml_l.append(fn)
        self.get_per_rank_yaml()

    def get_per_rank_yaml(self):
        fn = self.yml
        n = self.MP.mpi_size
        # Read yaml
        with open(fn) as fd:
            buff = fd.read()
            yaml_orig = yaml.load(buff, Loader=yaml.FullLoader)

        ## Init per rank yaml
        yaml_ranks_l = []
        for i in range(n):
            yaml_ranks_l.append(copy.deepcopy(yaml_orig))

        ## Create per rank yaml
        split_keys = yaml_orig['split'].keys()
        for k in split_keys:
            s = yaml_orig['split'][k]
            split_size = len(s)

            partition_size = split_size // n

            random.shuffle(s)

            for i in range(n):
                yaml_tmp = yaml_ranks_l[i]
                start = i * partition_size
                new_split = s[start:start+partition_size]
                yaml_tmp['split'][k] = new_split

        ### Save per rank yaml
        for rank_index, yaml_tmp in enumerate(yaml_ranks_l):
            out_fn = self.per_rank_yml_l[rank_index]
            print (out_fn)
            yaml.dump(yaml_tmp, open(out_fn, 'w'))

    def net_setup(self):
        in_ = eddl.Input([3, self.size[0], self.size[1]])
        # set net init
        if (self.net_init=='HeNormal'):
            net_init = eddl.HeNormal
        elif (self.net_init=='GlorotNormal'):
            net_init = eddl.GlorotNormal
        else:
            raise ValueError('net_init can only be HeNormal or GlorotNormal')
        # set network
        if self.net_name == 'vgg16':
            print ('vgg16')
            out = models.VGG16(in_, self.num_classes, init=net_init,
                               l2_reg=self.l2_reg, dropout=self.dropout)
        elif self.net_name == 'resnet50':
            print ('resnet50')
            out = models.ResNet50(in_, self.num_classes, init=net_init,
                               l2_reg=self.l2_reg, dropout=self.dropout)
        else:
            raise ValueError('model %s not available' % self.net_name)
        # build network
        self.net = eddl.Model([in_], [out])
       
        if self.num == 1:
            ## Single node execution
                
            eddl.build(
                self.net,
                eddl.sgd(self.lr),
                ["soft_cross_entropy"],
                ["categorical_accuracy"],
                eddl.CS_GPU(self.gpus, mem="full_mem", lsb=1) if self.gpus else eddl.CS_CPU()
                )

        else:
            ## multinode and/or multigpu
            if self.gpus:
                gpu_mask = [0] * self.ngpus
                gpu_mask[self.MP.mpi_rank % self.ngpus] = 1
            else:
                gpu_mask = []

            eddl.build(
                self.net,
                sgd_mpi(self.MP, self.lr),
                ["soft_cross_entropy"],
                ["categorical_accuracy"],
                eddl.CS_GPU(gpu_mask, mem="full_mem", lsb=1) if self.gpus else eddl.CS_CPU()
                )

        eddl.summary(self.net)
