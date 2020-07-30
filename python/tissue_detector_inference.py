# Copyright (c) 2019-2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Fully connected network for tissue detection in histopathology images
"""

import argparse
import sys
import numpy as np

import pyeddl.eddl as eddl
import pyecvl.ecvl as ecvl
from pyeddl.tensor import Tensor

import models

def read_slide(slide_fn, level=4):
    levels = ecvl.OpenSlideGetLevels(slide_fn)
    dims = [0, 0] + levels[level]
    img = ecvl.OpenSlideRead(slide_fn, level, dims)
    t = ecvl.ImageToTensor(img)
    t_np = t.getdata()
    s = t_np.shape
    t_np = t_np.transpose((1,2,0)).reshape((s[1]*s[2], 3)) # Channel last and reshape
    t_eval = Tensor.fromarray(t_np) 
    print (t_eval.getShape())
    return t_eval, s


def get_mask(prob_T_l, s, th):
    mask_np_l = []
    for prob_T in prob_T_l:
        output_np = prob_T.getdata()
        pred_np = np.zeros(output_np.shape[0])
        pred_np[output_np[:, 1]>th] = 1
        mask_values = pred_np
        mask_np_l.append(mask_values)

    mask_values = np.vstack(mask_np_l)
    mask = mask_values.reshape((s[1], s[2]))

    return mask


def main(args):
    slide_fn = args.slide_fn
    level = args.level

    ## Load model
    net = models.tissue_detector_DNN()
    eddl.build(
        net,
        eddl.rmsprop(0.00001),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU() if args.gpu else eddl.CS_CPU()
    )

    eddl.load(net, args.weights_fn, "bin")
    eddl.summary(net)
    
    ## Load Slide
    slide_T, s = read_slide(slide_fn, level)

    ## Compute tissue mask
    #len_T = slide_T.getShape()[0]
    #bs = args.batch_size
    #nb = int(np.ceil((len_T / bs)))
    #print ("n. batches: %d" % nb)
    #output_l = []
    #for b in range(nb):
    #    start = bs*b
    #    stop = bs*(b+1) if bs*(b+1) < len_T else len_T
    #    b_T = slide_T.select(["%d:%d" % (start, stop)])
    
    output_l = eddl.predict(net, [slide_T])
    
    print (output_l)
    ## Binarize the output
    mask = get_mask(output_l, s, args.threshold)
    np.save("mask", mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide_fn", metavar="INPUT_DATASET")
    parser.add_argument("--weights_fn", type=str, metavar="MODEL FILENAME", default=30)
    parser.add_argument("--level", type=int, metavar="INT", default=4)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=2**24)
    parser.add_argument("--threshold", type=float, metavar="THRESHOLD TO CONVERT PROB TO PREDICTIONS", default=0.5)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
