Binding building howto:
1) bash build_cpp.sh
2) python3 setup.py build

Bindings modules are created in the subfolder build. To use them in your code, install them (python setup.py install) or put this code lines in your python code:

import sys
sys.path.insert(0, "your_path_to_this_pybind_folder/build/lib.linux-x86_64-3.6")
sys.path.insert(0, "your_path_to_this_pybind_folder/build/lib.linux-x86_64-3.8")

import pyhe_init._ext1
