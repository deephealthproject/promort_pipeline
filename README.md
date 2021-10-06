# SGD MPI
This branch contains a basic MPI implementation of the EDDL SGD optimizer, along with two examples (mnist code, written both in C++ and Python and imagenette coded in Python). Note that in this version the synchronization among workers is performed at every iteration (after the training of each batch).

The code requires a CUDA aware OpenMPI and UCX installation to exploit GPUDirect features for fast MPI communications among GPUs (running both on the same host or on different hosts). UCX allows for transparent use of NVLink, Cuda IPC, InfiniBand and TCP.  
A working environment is provided by the *Dockerfile* in the parent folder, which automatically installs the latest version of [**gdrcopy**](https://github.com/NVIDIA/gdrcopy). It is also necessary to build and load the **gdrdrv** kernel module in the host before running the container. Please note that the versions of **gdrcopy** on the docker image and the host must be the same.

The **code** folder has the following sub-directories:
 * **opt_mpi**: The *cpp* sub-folder includes the code to implement mpi functionalities along with the extension of the SGD optimizer. The *pybind* sub-folder includes the code to create python bindings.
 * **examples**: Code of the mnist and imagenette examples.
 * **utils**: Some utility scripts to create python bindings, download datasets and generate yaml files starting from a dataset directory.  

## How to Run the Docker container to test the examples
Run the following commands from the parent folder to create the docker image and run the container:
```bash
### Clone the repository
git clone https://github.com/deephealthproject/promort_pipeline.git

### Switch to sgd_mpi branch
git checkout sgd_mpi

### Create the docker image
docker build -t sgd_mpi .

### Run the container (change the gpu ids in the --gpus option if needed)
docker run --privileged -d --rm --name sgd_mpi --gpus '"device=0, 1"' sgd_mpi:latest

### Get the prompt of the container
docker exec -ti sgd_mpi /bin/bash
```

A file named *hostfile* is already present in the container folder */home/sgd_mpi/code*. This is needed to make MPI aware about how many nodes and how many GPUs per node are available for the distributed computation. The *hostfile* provided contains one line:
```
HOSTNAME slots=2
```
assuming a computation environment of one node with 2 GPUs.

To run the examples read the **README** present in each example subfolder ([mnist](code/examples/mnist), [imagenette](code/examples/imagenette2-224/python)). 
