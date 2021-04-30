#ifndef MINIMPI_H
#define MINIMPI_H

#include <string>
#include <vector>
using namespace std;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <mpi.h>

// list of lists of numpy array, densely packed in C style
typedef vector<vector<py::array_t<float, py::array::c_style | py::array::forcecast>>> LoL;

class miniMPI{
public:
  miniMPI(int bl=512);
  ~miniMPI();
  int mpi_rank;
  int mpi_size;
  size_t mpi_block;
  string mpi_hostname;
  float div;
  void Barrier();
  void LoLAverage(LoL& input, LoL& output);
};
#endif
