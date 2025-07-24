licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_my_cholesky_op.so',
    srcs = [
        "cc/kernels/my_cholesky_op_kernels.cc",
        "cc/ops/my_cholesky_op.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "my_cholesky_op_py",
    srcs = ([
        "python/ops/my_cholesky_op.py",
    ]),
    data = [
        ":python/ops/_my_cholesky_op.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "my_cholesky_op_py_test",
    srcs = [
        "python/ops/my_cholesky_op_test.py"
    ],
    main = "python/ops/my_cholesky_op_test.py",
    deps = [
        ":my_cholesky_op_py",
    ],
    srcs_version = "PY2AND3",
)

