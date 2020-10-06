from distutils.core import setup, Extension

import pybind11
import os


EXTRA_COMPILE_ARGS = ['-std=c++11', '-fvisibility=hidden']

ext = Extension(
    "pyhe_init._ext1",
    sources=["bindings.cpp"],
    include_dirs=[
        os.getcwd(),
        pybind11.get_include(),
        pybind11.get_include(user=True)
    ],
    library_dirs=[os.getcwd(), '/usr/local/lib'],
    runtime_library_dirs=[os.getcwd(), '/usr/local/lib'],
    libraries=["he_init", 'eddl'],
    extra_compile_args=EXTRA_COMPILE_ARGS,
    undef_macros=["NDEBUG"],
)


setup(
    name="pyhe_init",
    ext_modules=[ext]
)
