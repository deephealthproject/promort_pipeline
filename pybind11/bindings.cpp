#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../cpp/initializers/he_init.h"

namespace py = pybind11;

PYBIND11_MODULE(_ext1, m) {
    m.doc() = "pybind11 he_initializer for eddl layers"; // optional module docstring
    m.def("he_normal", &he_normal);
    
    py::class_<IHeNormal>(m, "IHenormal")
       .def(py::init<int>())
       .def("apply", &IHeNormal::apply);

}

