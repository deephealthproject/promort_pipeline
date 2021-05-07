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

// numpy array of floats, densely packed in C style
typedef py::array_t<float, py::array::c_style> cpyar;
// list of lists of numpy arrays
typedef vector<vector<cpyar>> LoL;

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
  void Gather(float x, cpyar& ret, int root=0);
  void LoLBcast(LoL& data, int root=0);
  void LoLAverage(LoL& input, LoL& output);
  float Allreduce(float input, string op="SUM");
};
#endif
