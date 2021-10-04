#!/bin/bash
DEST=/home/sgd_mpi/data
mkdir -p $DEST && cd $DEST
wget -q --show-progress https://www.dropbox.com/s/khrb3th2z6owd9t/mnist_trX.bin
wget -q --show-progress https://www.dropbox.com/s/m82hmmrg46kcugp/mnist_trY.bin
wget -q --show-progress https://www.dropbox.com/s/7psutd4m4wna2d5/mnist_tsX.bin
wget -q --show-progress https://www.dropbox.com/s/q0tnbjvaenb4tjs/mnist_tsY.bin

LINK_DEST=/home/sgd_mpi/code/examples/mnist/cpp
ln -s $DEST/mnist_trX.bin $LINK_DEST/mnist_trX.bin
ln -s $DEST/mnist_trY.bin $LINK_DEST/mnist_trY.bin
ln -s $DEST/mnist_tsX.bin $LINK_DEST/mnist_tsX.bin
ln -s $DEST/mnist_tsY.bin $LINK_DEST/mnist_tsY.bin

LINK_DEST=/home/sgd_mpi/code/examples/mnist/python
ln -s $DEST/mnist_trX.bin $LINK_DEST/mnist_trX.bin
ln -s $DEST/mnist_trY.bin $LINK_DEST/mnist_trY.bin
ln -s $DEST/mnist_tsX.bin $LINK_DEST/mnist_tsX.bin
ln -s $DEST/mnist_tsY.bin $LINK_DEST/mnist_tsY.bin
