#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("MyCholesky")
    .Input("input: T")
    .Output("output: T")        // Cholesky factor (lower-triangular)
    .Output("status: int32")    // Decomposition success/failure flag
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
        // Output has same shape as input
        c->set_output(0, input);
        // output[1] is a scalar (the status)
        c->set_output(1, c->Scalar());
        return ::tsl::OkStatus();
    })
    .Doc(R"doc(
A custom Cholesky decomposition op that outputs the lower-triangular matrix.

input: A symmetric positive-definite matrix.
output: Lower-triangular Cholesky factor.
status: 0 if success, 1 if failure.
)doc");

