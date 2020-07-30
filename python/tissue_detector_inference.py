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

def read_slide(slide_fn, level):
    levels = ecvl.OpenSlideGetLevels(slide_fn)  
    print (levels)
    return None

def main(args):
    slide_fn = args.slide_fn
    level = args.level

    ## Load model
    net = models.tissue_detector_DNN()
    eddl.load(net, fname=args.weights_fn)
    print ("Net description:")
    eddl.summary(net)

    ## Load Slide
    slideT = read_slide(slide_fn, level)

    ## Compute tissue mask
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide_fn", metavar="INPUT_DATASET")
    parser.add_argument("--weights_fn", type=str, metavar="MODEL FILENAME", default=30)
    parser.add_argument("--level", type=int, metavar="INT", default=4)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=8192)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
