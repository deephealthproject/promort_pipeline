CXXFLAGS = -Wall -Wextra -pedantic -std=c++17 -O2
LFLAGS = -L/usr/local/lib/ -l ecvl_eddl -l eddl -l mpi
IXXFLAGS = -I/usr/local/include/eddl -I/usr/local/include/ecvl
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
	g++ -MM *.cpp > dep.mk

include dep.mk
