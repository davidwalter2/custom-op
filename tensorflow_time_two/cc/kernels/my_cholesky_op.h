#ifndef MY_CHOLESKY_OP_H_
#define MY_CHOLESKY_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct MyCholeskyFunctor {
  void operator()(OpKernelContext* context, const Tensor& input_tensor,
                  Tensor* output_tensor, Tensor* status_tensor);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // MY_CHOLESKY_OP_H_