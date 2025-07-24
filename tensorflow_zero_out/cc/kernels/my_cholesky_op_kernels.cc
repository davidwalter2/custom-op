#include "Eigen/Cholesky"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;

template <typename T>
class MyCholeskyOp : public OpKernel {
 public:
  explicit MyCholeskyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    OP_REQUIRES(context, input_tensor.dims() == 2,
                errors::InvalidArgument("Input must be a 2D matrix"));

    int64_t rows = input_tensor.dim_size(0);
    int64_t cols = input_tensor.dim_size(1);
    
    OP_REQUIRES(context, rows == cols,
                errors::InvalidArgument("Input must be a square matrix"));

    Tensor* output_tensor = nullptr;
    Tensor* status_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &status_tensor));
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    auto input_flat = input_tensor.flat<T>();
    auto output_flat = output_tensor->flat<T>();
    auto status_flat = status_tensor->flat<int32>();

    const Eigen::Map<const Matrix> input_matrix(input_flat.data(), rows, cols);
    Eigen::Map<Matrix> output_matrix(output_flat.data(), rows, cols);

    Eigen::LLT<Matrix> llt(input_matrix);

    if (llt.info() != Eigen::Success) {
      LOG(WARNING) << "Cholesky decomposition failed";
      const Matrix& L = llt.matrixL();  // Possibly partial
      Matrix reconstructed = L * L.transpose();
      Matrix diff = input_matrix - reconstructed;

      // Tolerance for identifying breakdown
      const T rtol = static_cast<T>(1e-5);
      const T atol = static_cast<T>(1e-8);

      Eigen::ArrayXX<bool> diff_mask = (diff.cwiseAbs().array() > atol + rtol * input_matrix.cwiseAbs().array());

      // Now find largest `i` such that diff_mask.block(0, 0, i, i).any() == false
      int fail_index = 0;
      for (int i = 1; i <= rows; ++i) {
        if ((diff_mask.block(0, 0, i, i)).any()) {
            break;
        }
        fail_index = i;
      }

      status_flat(0) = fail_index;
    }
    else{
      status_flat(0) = 0;
    }
    
    output_matrix = llt.matrixL();
  }
};

REGISTER_KERNEL_BUILDER(Name("MyCholesky").Device(DEVICE_CPU).TypeConstraint<float>("T"), MyCholeskyOp<float>);
REGISTER_KERNEL_BUILDER(Name("MyCholesky").Device(DEVICE_CPU).TypeConstraint<double>("T"), MyCholeskyOp<double>);