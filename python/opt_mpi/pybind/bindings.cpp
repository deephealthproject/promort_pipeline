#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../cpp/with_cuda_rt/mpi_env.hpp"
#include "../cpp/with_cuda_rt/optim_sgd_mpi.hpp"

namespace py = pybind11;

PYBIND11_MODULE(OPT_MPI, m) {
    m.doc() = "pybind11 SGD_mpi and mpi_env for eddl layers"; // optional module docstring
    m.def("sgd_mpi", &sgd_mpi); // High level API
    
    py::class_<SGD_mpi>(m, "SGD_mpi")
       .def(py::init<mpi_env*, float, float, float, bool>())
       .def("clone", &SGD_mpi::clone)
       .def("applygrads", &SGD_mpi::applygrads)
       .def("sync_grads", &SGD_mpi::sync_grads)
       .def("sync_rank_0_parameters", &SGD_mpi::sync_rank_0_parameters);
    
    py::class_<mpi_env>(m, "mpi_env")
       .def(py::init<int, int>())
       .def("Barrier", &mpi_env::Barrier)
       .def("Bcast_Tensor", &mpi_env::Bcast_Tensor)
       .def("Allreduce_Tensor", &mpi_env::Allreduce_Tensor);
}
