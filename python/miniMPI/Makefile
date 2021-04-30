CXXFLAGS = -Wall -Wextra -pedantic -std=c++17 -O2
LFLAGS = $(shell python3-config --ldflags)
BIND_INCL = $(shell python3 -m pybind11 --includes)
BIND_SUFF = $(shell python3-config --extension-suffix)
IXXFLAGS = $(BIND_INCL)

.PHONY: clean all

all: MMPI

clean:
	rm -f *.o runme BPH$(BIND_SUFF)

MMPI: pybindings.cpp miniMPI.cpp
	mpiCC $(CXXFLAGS) $(IXXFLAGS) -shared -fPIC $^ -o $@$(BIND_SUFF) $(LFLAGS)

%.o : %.cpp
	mpiCC $(CXXFLAGS) $(IXXFLAGS) -c -o $@ $< $(LFLAGS)

dep.mk: *.cpp *.hpp
	g++ -MM *.cpp > dep.mk

include dep.mk
