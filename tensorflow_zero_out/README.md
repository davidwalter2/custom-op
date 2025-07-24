To build the custom operation:

export TF_LIB_DIR=/opt/venv/lib/python3.12/site-packages/tensorflow
export TF_HEADER_DIR=/opt/venv/lib/python3.12/site-packages/tensorflow/include
export TF_SHARED_LIBRARY_DIR=/opt/venv/lib/python3.12/site-packages/tensorflow
export TF_SHARED_LIBRARY_NAME="libtensorflow_framework.so.2"

bazel build --cxxopt='-std=c++17' //tensorflow_zero_out:python/ops/_my_cholesky_op.so