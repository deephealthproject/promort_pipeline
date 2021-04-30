#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;

#include "miniMPI.hpp"

PYBIND11_MODULE(MMPI, m) {
  py::class_<miniMPI>(m, "miniMPI")
    .def(py::init())
    .def_readonly("mpi_rank", &miniMPI::mpi_rank)
    .def_readonly("mpi_size", &miniMPI::mpi_size)
    .def_readonly("mpi_hostname", &miniMPI::mpi_hostname)
    .def("LoLAverage", &miniMPI::LoLAverage)
    .def("Barrier", &miniMPI::Barrier);
}
