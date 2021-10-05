## Running the CPP example:
```bash
cd /home/sgd_mpi/code/examples/mnist/cpp
make
mpirun --n 2 --hostfile /home/sgd_mpi/code/hostfile mnist_mpi
```

## Running the Python example:
```bash
cd /home/sgd_mpi/code/examples/mnist/python
mpirun --n 2 --hostfile /home/sgd_mpi/code/hostfile python3 mnist_mlp.py --gpu
```
