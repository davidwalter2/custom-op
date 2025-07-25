#include "my_cholesky_op.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class MyCholeskyOp : public OpKernel {
 public:
  explicit MyCholeskyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    OP_REQUIRES(context, input_tensor.dims() == 2,
                errors::InvalidArgument("Input must be a 2D matrix"));
    const int64_t rows = input_tensor.dim_size(0);
    const int64_t cols = input_tensor.dim_size(1);
    OP_REQUIRES(context, rows == cols,
                errors::InvalidArgument("Input must be a square matrix"));

    Tensor* output_tensor = nullptr;
    Tensor* status_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &status_tensor));

    functor::MyCholeskyFunctor<Device, T>()(context, input_tensor, output_tensor, status_tensor);
  }
};

// CPU registration
REGISTER_KERNEL_BUILDER(Name("MyCholesky")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        MyCholeskyOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("MyCholesky")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        MyCholeskyOp<CPUDevice, double>);

#if GOOGLE_CUDA
// GPU registration
REGISTER_KERNEL_BUILDER(Name("MyCholesky")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        MyCholeskyOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("MyCholesky")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T"),
                        MyCholeskyOp<GPUDevice, double>);
#endif

}  // namespace tensorflow