CXXFLAGS = -Wall -Wextra -pedantic -std=c++17 -O2
LFLAGS = $(shell python3-config --ldflags)
BIND_INCL = $(shell python3 -m pybind11 --includes)
BIND_SUFF = $(shell python3-config --extension-suffix)
IXXFLAGS = $(BIND_INCL)
#MPICC = /usr/mpi/gcc/openmpi-4.0.3rc4/bin/mpiCC
MPICC = mpic++

.PHONY: clean all

all: MMPI

clean:
	rm -f *.o runme BPH$(BIND_SUFF)

MMPI: pybindings.cpp miniMPI.cpp
	$(MPICC) $(CXXFLAGS) $(IXXFLAGS) -shared -fPIC $^ -o $@$(BIND_SUFF) $(LFLAGS)

%.o : %.cpp
	$(MPICC) $(CXXFLAGS) $(IXXFLAGS) -c -o $@ $< $(LFLAGS)

dep.mk: *.cpp *.hpp
	g++ -MM *.cpp > dep.mk

include dep.mk
