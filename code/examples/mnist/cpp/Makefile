CXXFLAGS = -Wall -Wextra -pedantic -std=c++17 -O2
LFLAGS = -L/usr/local/lib/ -L/usr/local/cuda/lib64/ -l cudart  -l ecvl_eddl -l eddl -l mpi
OPT_MPI_DIR=/home/sgd_mpi/code/opt_mpi/cpp
IXXFLAGS = -I$(OPT_MPI_DIR) -I/usr/local/include/eddl -I/usr/local/include/ecvl -I/usr/local/cuda/include -I/usr/local/include/eigen3/
# All .o files go to build dir.
BUILD_DIR = ./
CPP = $(wildcard ./*.cpp) 
OBJ = $(CPP:%.cpp=$(BUILD_DIR)/%.o)
MAKE= make

.PHONY: clean all

all: mnist_mpi

clean:
	rm -f *.o mnist_mpi 

mnist_mpi: $(OBJ)
	mpiCC $(CXXFLAGS) $(IXXFLAGS) $^ $(OPT_MPI_DIR)/*.o -o $@ $(LFLAGS)

%.o : %.cpp
	cd $(OPT_MPI_DIR) && $(MAKE) -f Makefile
	mpiCC $(CXXFLAGS) $(IXXFLAGS) -c -o $@ $< $(LFLAGS)
