"""
PROMORT example.
"""

import argparse
import sys
import pickle
import numpy as np
sys.path.append('../')

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

import models 

def get_net(in_size=[256,256], num_classes=2, lr=1e-5, gpu=True):
    
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
        eddl.CS_GPU([1], mem="low_mem") if gpu else eddl.CS_CPU()
        )

    return net


def update_eddl_net_params(keras_params_d, net):
    conv_layers_l = [(l.name, l.params[0].getShape(), l.params[1].getShape(), l) for l in net.layers if 'conv' in l.name]
    
    for index, k in enumerate(sorted(keras_params_d.keys())):
        w_np = keras_params_d[k]['w']
        b_np = keras_params_d[k]['b']
        l = conv_layers_l[index]

        # Transpose to match eddl tensor shape
        w_np = np.transpose(w_np, (3, 2, 0, 1))

        # Shapes check
        eddl_w_shape = l[1]
        eddl_b_shape = l[2]

        print (l[0], k)
        print (eddl_w_shape, w_np.shape)
        print (eddl_b_shape, b_np.shape)
        
        # converting numpy arrays to tensor
        w_np_t = Tensor.fromarray(w_np)
        b_np_t = Tensor.fromarray(b_np)

        # Update of the parameters
        l[3].update_weights(w_np_t, b_np_t)


def check_params(keras_params_d, net):
    conv_layers_l = [(l.name, l.params[0].getdata(), l.params[1].getdata()) for l in net.layers if 'conv' in l.name]
    print ('-' * 50)
    print ("Parameters integrity check")
    
    for index, k in enumerate(sorted(keras_params_d.keys())):
        w_np = keras_params_d[k]['w']
        b_np = keras_params_d[k]['b']
        l = conv_layers_l[index]

        # Transpose to match eddl tensor shape
        w_np = np.transpose(w_np, (3, 2, 0, 1))
        print (l[0])
        if not (w_np == l[1]).all:
            print ('Weight different')
        if not (b_np == l[2]).all:
            print ('Bias differt')
        

def main(args):
    ### Get Network
    net = get_net()
    keras_params_d = pickle.load(open(args.in_fn, 'rb'))
   
    # Copy keras parameters to the eddl convolutional layers
    update_eddl_net_params(keras_params_d, net)

    # Check if everything is ok
    check_params(keras_params_d, net)
   
    # Save network weights
    eddl.save(net, args.out_fn, "bin") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-fn", metavar="DIR",
                        help="numpy file with vgg16 weights")
    parser.add_argument("--out-fn", metavar="DIR", default = "vgg16_imagenet_init.bin",
                        help="output weights filename")
    main(parser.parse_args())
