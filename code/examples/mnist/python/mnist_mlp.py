# Copyright (c) 2019-2021 CRS4
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

"""\
Basic MLP for MNIST.
"""

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from OPT_MPI import mpi_env, SGD_mpi, sgd_mpi

MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")

def main(args):
    n_sync = 1
    bl = 1024
    mpe = mpi_env(n_sync, bl)
    mpi_size = mpe.mpi_size
    mpi_rank = mpe.mpi_rank

    num_classes = 10

    in_ = eddl.Input([784])

    layer = in_
    layer = eddl.LeakyReLu(eddl.Dense(layer, 1024))
    layer = eddl.LeakyReLu(eddl.Dense(layer, 1024))
    layer = eddl.LeakyReLu(eddl.Dense(layer, 1024))
    out = eddl.Softmax(eddl.Dense(layer, num_classes), -1)
    net = eddl.Model([in_], [out])
    
    #opt_SGD_mpi = SGD_mpi(mpe, 1e-2, 0.9, 0.0, False)
    if args.gpu:
        gpu_mask = [0] * mpi_size
        gpu_mask[mpi_rank % mpi_size] = 1
    else:
        gpu_mask = []

    eddl.build(
        net,
        sgd_mpi(mpe, 1e-2, 0.9, 0.0, False),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(gpu_mask,mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )
    
    if mpi_rank == 0:
        eddl.summary(net)

    # Load dataset if needed
    eddl.download_mnist()

    # Broadcast initial params
    mpe.Broadcast_params(net)

    x_train = Tensor.load("mnist_trX.bin")
    y_train = Tensor.load("mnist_trY.bin")
    x_test = Tensor.load("mnist_tsX.bin")
    y_test = Tensor.load("mnist_tsY.bin")
    
    train_data_size = x_train.shape[0]
    test_data_size = x_test.shape[0]
    per_rank_ds = train_data_size // mpi_size
    start = per_rank_ds * mpi_rank
    x_train = Tensor(x_train.getdata()[start:start+per_rank_ds])
    y_train = Tensor(y_train.getdata()[start:start+per_rank_ds])

    x_train.div_(255.0)
    x_test.div_(255.0)

    eddl.fit(net, [x_train], [y_train], args.batch_size, args.epochs)
    
    if mpi_rank == 0:
        eddl.evaluate(net, [x_test], [y_test], bs=args.batch_size)
        print("All done")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=200)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES),
                        choices=MEM_CHOICES, default="full_mem")
    main(parser.parse_args(sys.argv[1:]))
