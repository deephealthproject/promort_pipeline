import sys
sys.path.insert(0, "build/lib.linux-x86_64-3.6")
sys.path.insert(0, "build/lib.linux-x86_64-3.8")

import pyhe_init._ext1
from pyhe_init._ext1 import he_normal
import pyeddl.eddl as eddl

x = eddl.Input([3, 256,256])
x = eddl.ReLu(he_normal(eddl.Conv(x, 64, [3, 3]), 1))

print(x)
