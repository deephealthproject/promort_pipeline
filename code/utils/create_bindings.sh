#!/bin/bash
cd /home/sgd_mpi/code/opt_mpi/cpp
make clean
cd /home/sgd_mpi/code/opt_mpi/pybind
make
python3 setup.py build
ln -s /home/sgd_mpi/code/opt_mpi/pybind/build/lib.linux-x86_64-3.6/pyopt_mpi/OPT_MPI.cpython-36m-x86_64-linux-gnu.so  /home/sgd_mpi/code/examples/mnist/python/OPT_MPI.cpython-36m-x86_64-linux-gnu.so
ln -s /home/sgd_mpi/code/opt_mpi/pybind/build/lib.linux-x86_64-3.6/pyopt_mpi/OPT_MPI.cpython-36m-x86_64-linux-gnu.so  /home/sgd_mpi/code/examples/imagenette2-224/python/OPT_MPI.cpython-36m-x86_64-linux-gnu.so

