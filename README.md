# SGD MPI
This branch contains a basic implementation of an MPI version of EDDL SGD optimizer along with a distributed version of mnist code written both in CPP and Python.
The code requires a CUDA aware OpenMPI installation to exploit GPUDirect features for fast MPI communications among GPUs running both on the same host or on different hosts. A working environment is provided by the *Dockerfile* in the parent folder. The **code** folder has the following sub-directories:
 * **opt_mpi**: The *cpp* folder includes the code to implement mpi functionalities along with the extension of the SGD optimizer. The *python* folder includes the code to create python bindings
 * **cpp**: *cpp* code of the mnist example
 * **python**: *python* code of the mnist example  

## Running the Docker container to test the examples
Run the following command from the parent folder to create the docker image:
*docker build -t image_name .*

Then run the container with:
*docker run -d -it --name container_name --gpus '"device=gpu_index, gpu_index, ..."' image_name:latest*

Get the prompt of the container with:
*docker exec -ti container_name /bin/bash*

At this point you are located on parent folder inside the container and has to set some *mpi* stuff to get things working. In the case you are running on a single hosts with multiple GPUs there is only a Docker running container running but the definition of a *hostfile* is needed to run the mpi application properly. 
As an example a hostfile for a single container with two GPUs allocated is like:

*container_IP slots=2*

To get the container_IP just issue the *ip addr* command from bash.

## Running the CPP example
Go to the folder *code/cpp* and run the Makefile. If no errors occurred the run the application with the following command:

*mpirun --n 2 --hostfile path_to_hostfile mnist_mpi*

## Running the Python example
Go to the folder *code/python* and run *sh sh create_bindings.sh* to create python bindings. Run the example with:

*mpirun --n 2 --hostfile path_to_hostfile python3 mnist_mlp.py --gpu*
