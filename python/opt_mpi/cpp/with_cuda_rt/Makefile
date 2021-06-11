CXXFLAGS = -Wall -Wextra -pedantic -std=c++17 -O2
LFLAGS = -L/usr/local/lib/ -L/usr/local/cuda/lib64/ -l cudart  -l ecvl_eddl -l eddl -l mpi
IXXFLAGS = -I/usr/local/include/eddl -I/usr/local/include/ecvl -I/usr/local/cuda/include
# All .o files go to build dir.
BUILD_DIR = ./
CPP = $(wildcard ./*.cpp)
OBJ = $(CPP:%.cpp=$(BUILD_DIR)/%.o)

.PHONY: clean all

all: mnist_mpi

clean:
	rm -f *.o mnist_mpi 

mnist_mpi: $(OBJ)
	mpiCC $(CXXFLAGS) $(IXXFLAGS) $^ -o $@ $(LFLAGS)

%.o : %.cpp
	mpiCC $(CXXFLAGS) $(IXXFLAGS) -c -o $@ $< $(LFLAGS)

dep.mk: *.cpp *.hpp
	g++ -MM $(IXXFLAGS) *.cpp > dep.mk

include dep.mk
