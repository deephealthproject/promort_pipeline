# SGD MPI
This branch contains a basic implementation of an MPI version of EDDL SGD optimizer along with a distributed version of mnist code written both in CPP and Python.
The code requires a CUDA aware OpenMPI installation to exploit GPUDirect features for fast MPI communications among GPUs running both on the same host or on different hosts. 

A working environment is provided by the *Dockerfile* in the parent folder but it is mandatory to build and load the **gdrdrv** module in the host (https://github.com/NVIDIA/gdrcopy) before running the container. The version of **gdrcopy** on the docker image and the host must be the same. If for some reason the **gdrdrv** is already installed on the host, change the version in the dockerfile accordingly.


The **code** folder has the following sub-directories:
 * **opt_mpi**: The *cpp* folder includes the code to implement mpi functionalities along with the extension of the SGD optimizer. The *python* folder includes the code to create python bindings
 * **cpp**: *cpp* code of the mnist example
 * **python**: *python* code of the mnist example  

## How to Run the Docker container to test the examples
Run the following command from the parent folder to create the docker image and run the container:
```bash
### Create the docker image
docker build -t sgd_mpi .

### Run the container (change the gpu ids in the --gpus option if needed)
docker run -d --privileged --name sgd_mpi --gpus '"device=0, 1"' sgd_mpi:latest

### Get the prompt of the container
docker exec -ti sgd_mpi /bin/bash
```

A file named *hostfile* is already present in the container folder */home/sgd_mpi/code*. This is needed to make mpi aware about how many nodes and how many GPUs per node are available for the distributed computation. The *hostfile* provided contains one line:
```
HOSTNAME slots=2
```
assuming a computation environment of one node with 2 GPUs.

## Running the CPP example
Go to the folder *code/cpp* and issue the following commands:
```bash
make
mpirun --n 2 --hostfile ../hostfile mnist_mpi
```

## Running the Python example
Go to the folder *code/python* and run the example:
```bash
mpirun --n 2 --hostfile ../hostfile python3 mnist_mlp.py --gpu
```
