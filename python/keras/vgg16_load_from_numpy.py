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

def get_net(in_size=[256,256], num_classes=2, lr=1e-5, net_name='vgg16_tumor', gpu=True):
    
    ## Network definition
    in_ = eddl.Input([3, in_size[0], in_size[1]])
    if net_name == 'vgg16_tumor':
        out = models.VGG16_tumor(in_, num_classes)
    elif net_name == 'vgg16_gleason':
        out = models.VGG16_gleason(in_, num_classes)
    elif net_name == 'vgg16':
        out = models.VGG16(in_, num_classes)
    else:
        print('model %s not available' % net_name)
        sys.exit(-1)

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


def update_eddl_net_params(keras_params_d, net, include_top=True):
    if include_top:
        layers_l = [(l.name, l.params[0].getShape(), l.params[1].getShape(), l) for l in net.layers if 'conv' in l.name or 'dense' in l.name]
        keras_l = [k for k in sorted(keras_params_d.keys())]
    else:
        layers_l = [(l.name, l.params[0].getShape(), l.params[1].getShape(), l) for l in net.layers if 'conv' in l.name]
        keras_l = [k for k in sorted(keras_params_d.keys()) if 'conv' in k]
   
    
    for index, k in enumerate(keras_l):
        w_np = keras_params_d[k]['w']
        b_np = keras_params_d[k]['b']
        
        l = layers_l[index]

        # Transpose to match eddl tensor shape for convolutional layer
        if 'conv' in l[0]:
            print ("Conv before transpose", w_np.shape)
            w_np = np.transpose(w_np, (3, 2, 0, 1))
            
        
        # dense layer immediatly after the flattening. he order of weight is different because previous feature maps are channel last in Keras
        # but EDDL expects as they are channel first
        if l[0] == 'dense1':
            x = keras_params_d[keras_l[index-1]]['w'].shape[0]
            y = keras_params_d[keras_l[index-1]]['w'].shape[1]
            n_ch = keras_params_d[keras_l[index-1]]['w'].shape[3]
            print ('After flattening. #Channels of previous layers is: %d' % n_ch)
            # Converting w_np as the previous feature maps was channel first
            outputs = w_np.shape[1]
            print (w_np.shape)
        
            w_np_ch_f = np.zeros_like(w_np)
            
            for o in range(outputs):
                for offset in range(n_ch):
                    lll = w_np[offset::n_ch, o].shape[0]
                    w_np_ch_f[offset:offset+lll, o] = w_np[offset::n_ch, o]

        # Shapes check
        eddl_w_shape = np.array(l[1])
        eddl_b_shape = np.array(l[2])
        
        print (l[0], k)
        print (eddl_w_shape, w_np.shape)
        print (eddl_b_shape, b_np.shape)
        
        # converting numpy arrays to tensor
        w_np_t = Tensor.fromarray(w_np)
        b_np_t = Tensor.fromarray(b_np)

        # Update of the parameters
        l[3].update_weights(w_np_t, b_np_t)
        eddl.distributeParams(l[3])


def reset_eddl_net_params(net, weight='zeros', bias='zeros'):
    layers = net.layers
    for l in layers:
        name = l.name
        params = l.params
        
        w_np = None
        b_np = None
        
        for index, p in enumerate(params):
            if index == 0:
                if weight == 'zeros':
                    w_np = np.zeros_like(p.getdata())
                else:
                    w_np = np.ones_like(p.getdata())
        
            if index == 1: 
                if bias == 'zeros':
                    b_np = np.zeros_like(p.getdata())
                else:
                    b_np = np.ones_like(p.getdata())
        
        w_np_t = Tensor.fromarray(w_np)
        b_np_t = Tensor.fromarray(b_np)
        
        # Update of the parameters
        l.update_weights(w_np_t, b_np_t)      
        eddl.distributeParams(l)
        
        
def print_layer_outputs(net):
    ## Print layer outputs
    print ()
    print ('-'*47)
    print ('{0: <15} {1: >15} {2:>15}'.format('layer name', 'min_out_value', 'max_out_value'))
    for l in net.snets[0].layers:
        print ('{0: <15} {1:15e} {2:15e}'.format(l.name, np.min(l.output.getdata()), np.max(l.output.getdata())))  


def check_params(keras_params_d, net, include_top=False):
    if include_top:
        layers_l = [(l.name, l.params[0].getShape(), l.params[1].getShape(), l) for l in net.layers if 'conv' in l.name or 'dense' in l.name]
        keras_l = [k for k in sorted(keras_params_d.keys())]
    else:
        layers_l = [(l.name, l.params[0].getShape(), l.params[1].getShape(), l) for l in net.layers if 'conv' in l.name]
        keras_l = [k for k in sorted(keras_params_d.keys()) if 'conv' in k]
    
    print ('-' * 50)
    print ("Parameters integrity check")
    
    for index, k in enumerate(keras_l):
        w_np = keras_params_d[k]['w']
        b_np = keras_params_d[k]['b']
        l = layers_l[index]
        
        eddl_w_shape = np.array(l[1])
        eddl_b_shape = np.array(l[2])
        
        # Transpose to match eddl tensor shape
        if 'conv' in l[0]:
            w_np = np.transpose(w_np, (3, 2, 0, 1))
        
        print (l[0], k)
        if not (w_np.shape == eddl_w_shape).all():
            print ('Weight different')
        if not (b_np.shape == eddl_b_shape).all():
            print ('Bias different')
        

def main(args):
    num_classes = args.num_classes
    ### Get Network
    net = get_net(num_classes=num_classes, net_name=args.net_name)

    keras_params_d = pickle.load(open(args.in_fn, 'rb'))
   
    # Copy keras parameters to the eddl convolutional layers
    update_eddl_net_params(keras_params_d, net, args.include_top)

    # Check if everything is ok
    check_params(keras_params_d, net, args.include_top)
   
    # Save network weights
    eddl.save(net, args.out_fn, "bin") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-fn", metavar="DIR", required=True,
                        help="numpy file with vgg16 weights")
    parser.add_argument("--include-top", action="store_true",
                        help="Include dense classifier weights")
    parser.add_argument("--out-fn", metavar="DIR", default = "vgg16_weights.bin",
                        help="output weights filename")
    parser.add_argument("--num-classes", type=int, metavar="INT", default = 2,
                        help="Number of categories")
    parser.add_argument("--net-name", metavar="STR", default='vgg16_tumor',
                        help="Select the neural net (vgg16|vgg16_tumor|vgg16_gleason)")
    main(parser.parse_args())
