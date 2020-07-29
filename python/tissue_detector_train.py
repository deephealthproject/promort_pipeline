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
from pyeddl.tensor import Tensor

def read_input(filename, split_ratio=0.7):
    data = np.load(filename)['d']
    # shuffle data
    sel_index = np.arange(data.shape[0])
    np.random.shuffle(sel_index)
    
    shuffled_data = data[sel_index]
    shuffled_data = np.c_[shuffled_data, np.zeros(shuffled_data.shape[0])] # Add column for two class labels
    shuffled_data[:,4][shuffled_data[:,3] == 0] = 1.

    # Split train test
    n_train = int (shuffled_data.shape[0] * split_ratio )
    
    train = shuffled_data[0:n_train]
    test = shuffled_data[n_train:]
    x_trn = train[:,:3]
    y_trn = train[:,3:]
    x_test = test[:,:3]
    y_test = test[:,3:]
    
    # Tensor creation
    x_train_t = Tensor.fromarray(x_trn.astype(np.float32))
    y_train_t = Tensor.fromarray(y_trn.astype(np.float32))
    x_test_t = Tensor.fromarray(x_test.astype(np.float32))
    y_test_t = Tensor.fromarray(y_test.astype(np.float32))

    return x_train_t, y_train_t, x_test_t, y_test_t


def main(args):

    num_classes = 2

    ## Read input dataset
    x_train, y_train, x_test, y_test = read_input(args.in_ds)

    ## Net architecture
    in_ = eddl.Input([3])

    layer = in_
    layer = eddl.ReLu(eddl.Dense(layer, 50))
    layer = eddl.ReLu(eddl.Dense(layer, 50))
    layer = eddl.ReLu(eddl.Dense(layer, 50))
    out = eddl.Softmax(eddl.Dense(layer, num_classes))
    net = eddl.Model([in_], [out])

    ## Net compilation
    eddl.build(
        net,
        eddl.rmsprop(0.00001),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU() if args.gpu else eddl.CS_CPU()
    )

    eddl.summary(net)

    ## Fit and evaluation
    eddl.fit(net, [x_train], [y_train], args.batch_size, args.epochs)
    eddl.evaluate(net, [x_test], [y_test])
    eddl.save(net, "tissue_detector_model.bin")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_ds", metavar="INPUT_DATASET")
    parser.add_argument("--epochs", type=int, metavar="INT", default=30)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=8192)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
