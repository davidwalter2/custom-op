import os
from tensorflow.python.framework import load_library

# Use relative path to load shared object
_mylib = load_library.load_op_library(os.path.join(os.path.dirname(__file__), "_my_cholesky_op.so"))

# Register your op (update this to match your .cc name)
my_cholesky_op = _mylib.my_cholesky_op