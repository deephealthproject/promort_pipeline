#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../cpp/mpi_env.hpp"
#include "../cpp/optim_sgd_mpi.hpp"

namespace py = pybind11;

PYBIND11_MODULE(OPT_MPI, m) {
    m.doc() = "pybind11 SGD_mpi and mpi_env for eddl layers"; // optional module docstring
    //m.def("sgd_mpi", &sgd_mpi); // High level API

    m.def("sgd_mpi", (class Optimizer* (*)(mpi_env*, float, float, float, bool)) sgd_mpi, "C++: sgd_mpi(mpi_env*, float, float, float, bool) --> class Optimizer *", py::return_value_policy::reference, py::arg("MPE"), py::arg("lr") = 0.01f, py::arg("momentum") = 0.0f, py::arg("weight_decay") = 0.0f, py::arg("nesterov") = false);

    py::module _core = py::module::import("pyeddl._core");
    m.attr("SGD") = _core.attr("SGD");

    py::class_<SGD_mpi, std::unique_ptr<SGD_mpi>, SGD>(m, "SGD_mpi")
       .def(py::init<mpi_env*, float, float, float, bool>())
       .def("clone", &SGD_mpi::clone)
       .def("applygrads", &SGD_mpi::applygrads)
       .def("sync_params", &SGD_mpi::sync_params);
    
    py::class_<mpi_env>(m, "mpi_env")
       .def(py::init<int, int>(), py::arg("n_sync")=1, py::arg("bl")=512)
       .def_readonly("mpi_rank", &mpi_env::mpi_rank)
       .def_readonly("mpi_size", &mpi_env::mpi_size)
       .def("Barrier", &mpi_env::Barrier)
       .def("Bcast_Tensor", &mpi_env::Bcast_Tensor)
       .def("Gather_and_average", &mpi_env::Gather_and_average)
       .def("Allreduce_Tensor", &mpi_env::Allreduce_Tensor)
       .def("Broadcast_params", &mpi_env::Broadcast_params);
}
