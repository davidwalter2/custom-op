#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "my_cholesky_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace tensorflow {
namespace functor {

template <typename T>
struct MyCholeskyFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_tensor,
                  Tensor* output_tensor, Tensor* status_tensor) {
    auto stream = context->eigen_device<GPUDevice>().stream();

    const int64_t n = input_tensor.dim_size(0);
    const int lda = n;

    const T* A = input_tensor.flat<T>().data();
    T* L = output_tensor->flat<T>().data();
    int* status = status_tensor->flat<int32>().data();

    // Copy input to output (cuSolver works in-place)
    cudaMemcpyAsync(L, A, sizeof(T) * n * n, cudaMemcpyDeviceToDevice, stream);

    cusolverDnHandle_t cusolverH = nullptr;
    cusolverDnCreate(&cusolverH);
    cusolverDnSetStream(cusolverH, stream);

    int workspace_size = 0;
    int* devInfo = nullptr;
    cudaMalloc(&devInfo, sizeof(int));

    if constexpr (std::is_same<T, float>::value) {
      cusolverDnSpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, n, L, lda, &workspace_size);
    } else if constexpr (std::is_same<T, double>::value) {
      cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, n, L, lda, &workspace_size);
    }

    T* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size * sizeof(T));

    if constexpr (std::is_same<T, float>::value) {
      cusolverDnSpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, n, L, lda, workspace, workspace_size, devInfo);
    } else if constexpr (std::is_same<T, double>::value) {
      cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, n, L, lda, workspace, workspace_size, devInfo);
    }

    cudaMemcpyAsync(status, devInfo, sizeof(int), cudaMemcpyDeviceToDevice, stream);

    cudaFree(workspace);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolverH);
  }
};

// Explicit template instantiation
template struct MyCholeskyFunctor<GPUDevice, float>;
template struct MyCholeskyFunctor<GPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow
#endif  // GOOGLE_CUDA