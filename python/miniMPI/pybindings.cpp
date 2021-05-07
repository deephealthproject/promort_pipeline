#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;

#include "miniMPI.hpp"

PYBIND11_MODULE(MMPI, m) {
  py::class_<miniMPI>(m, "miniMPI")
    .def(py::init<int>(), "bl"_a=512)
    .def_readonly("mpi_rank", &miniMPI::mpi_rank)
    .def_readonly("mpi_size", &miniMPI::mpi_size)
    .def_readonly("mpi_hostname", &miniMPI::mpi_hostname)
    .def("Allreduce", &miniMPI::Allreduce)
    .def("LoLAverage", &miniMPI::LoLAverage, "input"_a, "output"_a)
    .def("Gather", &miniMPI::Gather, "x"_a, "ret"_a, "root"_a=0)
    .def("LoLBcast", &miniMPI::LoLBcast, "data"_a, "root"_a=0)
    .def("Barrier", &miniMPI::Barrier);
}
