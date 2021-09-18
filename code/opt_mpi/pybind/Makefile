SRC_PARENT = ../cpp
CXXFLAGS = -fPIC -Wall -Wextra -pedantic -std=c++17 -O2
LFLAGS = -L/usr/local/lib/ -L/usr/local/cuda/lib64/ -l cudart  -l ecvl_eddl -l eddl -l mpi
IXXFLAGS = -I/usr/local/include/eddl -I/usr/local/include/ecvl -I/usr/local/cuda/include -I$(SRC_PARENT) -I/usr/local/include/eigen3
# All .o files go to build dir.
BUILD_DIR = ./
CPP = $(wildcard $(SRC_PARENT)/*.cpp)
OBJ = $(CPP:%.cpp=$(BUILD_DIR)/%.o)
OUT = libopt_mpi.so

.PHONY: clean all

all: $(OUT)

clean:
	rm -f $(SRC_PARENT)/*.o $(OUT) 

$(OUT): $(OBJ)
	mpiCC -shared $^ -o $@ $(LFLAGS)

%.o : %.cpp
	mpiCC $(CXXFLAGS) $(IXXFLAGS) -c -o $@ $< $(LFLAGS)

dep.mk: $(SRC_PARENT)/*.cpp $(SRC_PARENT)/*.hpp
	g++ -MM $(IXXFLAGS) *.cpp > dep.mk

include dep.mk
