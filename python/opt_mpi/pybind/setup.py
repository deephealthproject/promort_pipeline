from distutils.core import setup, Extension

import pybind11
import os


EXTRA_COMPILE_ARGS = ['-std=c++11', '-fvisibility=hidden']

ext = Extension(
    "pyopt_mpi.OPT_MPI",
    sources=["bindings.cpp"],
    include_dirs=[
        '/usr/local/cuda/include/',
        '/usr/local/include/eigen3/',
        os.getcwd(),
        pybind11.get_include(),
        pybind11.get_include(user=True)
    ],
    library_dirs=[os.getcwd()],
    runtime_library_dirs=[os.getcwd()],
    libraries=['opt_mpi', 'eddl', 'mpi'],
    extra_compile_args=EXTRA_COMPILE_ARGS,
    undef_macros=["NDEBUG"],
)


setup(
    name="pyopt_mpi",
    ext_modules=[ext]
)
