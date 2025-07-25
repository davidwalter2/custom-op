#include "my_cholesky_op.h"
#include "Eigen/Cholesky"

namespace tensorflow {
namespace functor {

template <typename T>
struct MyCholeskyFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_tensor,
                  Tensor* output_tensor, Tensor* status_tensor) {
    const int64_t rows = input_tensor.dim_size(0);
    const int64_t cols = input_tensor.dim_size(1);
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    auto input_flat = input_tensor.flat<T>();
    auto output_flat = output_tensor->flat<T>();
    auto status_flat = status_tensor->flat<int32>();

    const Eigen::Map<const Matrix> input_matrix(input_flat.data(), rows, cols);
    Eigen::Map<Matrix> output_matrix(output_flat.data(), rows, cols);

    Eigen::LLT<Matrix> llt(input_matrix);

    if (llt.info() != Eigen::Success) {
      const Matrix& L = llt.matrixL();
      Matrix reconstructed = L * L.transpose();
      Matrix diff = input_matrix - reconstructed;
      const T rtol = static_cast<T>(1e-5);
      const T atol = static_cast<T>(1e-8);

      Eigen::ArrayXX<bool> diff_mask = (diff.cwiseAbs().array() > atol + rtol * input_matrix.cwiseAbs().array());
      int fail_index = 0;
      for (int i = 1; i <= rows; ++i) {
        if ((diff_mask.block(0, 0, i, i)).any()) break;
        fail_index = i;
      }

      status_flat(0) = fail_index;
    } else {
      status_flat(0) = 0;
    }

    output_matrix = llt.matrixL();
  }
};

// Explicit template instantiation
template struct MyCholeskyFunctor<CPUDevice, float>;
template struct MyCholeskyFunctor<CPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow