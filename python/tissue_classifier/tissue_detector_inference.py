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
import pickle
import sys
import numpy as np

import pyeddl.eddl as eddl
import pyecvl.ecvl as ecvl
from pyeddl.tensor import Tensor

import tissue_detector as td


def main(args):
    slide_fn = args.slide_fn
    level = args.level

    t_dec = td.tissue_detector(model_fn=args.weights_fn, th=args.threshold)
    mask = t_dec.get_mask_tissue_from_slide(slide_fn,
                                            level=level,
                                            use_openslide=args.use_openslide)
    dimensions = ecvl.OpenSlideGetLevels(slide_fn)[0]
    if args.more_info_output:
        with open(args.output, 'wb') as f:
            pickle.dump(
                {
                    'mask': mask,
                    'dimensions': dimensions,
                    'extraction_level': args.level,
                    'threshold': args.threshold,
                    'input': args.slide_fn,
                    'weights': args.weights_fn,
                }, f)
    else:
        np.save(args.output, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide_fn", metavar="INPUT_DATASET")
    parser.add_argument("--weights_fn",
                        type=str,
                        metavar="MODEL FILENAME",
                        default=30)
    parser.add_argument("--level", type=int, metavar="INT", default=4)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=2**24)
    parser.add_argument("--threshold",
                        type=float,
                        metavar="THRESHOLD TO CONVERT PROB TO PREDICTIONS",
                        default=0.5)
    parser.add_argument("--use-openslide", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "-I",
        help="output will be a pickled dictionary with info about the run",
        dest="more_info_output",
        default=False,
        action="store_true")
    parser.add_argument("-o",
                        help="output path",
                        dest="output",
                        default="mask")
    main(parser.parse_args(sys.argv[1:]))
