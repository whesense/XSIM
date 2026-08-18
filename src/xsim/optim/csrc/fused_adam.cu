// One-thread-per-element fused Adam (no weight decay, no amsgrad).
//
// PyTorch's fused Adam goes through multi_tensor_apply, which chunk-launches a
// tiny grid (a couple of blocks) tuned for lists of many small tensors. For our
// handful of multi-million-row Gaussian tensors that leaves ~98% of the SMs
// idle and the kernel runs memory-latency-bound. Here we launch a data-sized
// grid (grid-stride loop) so every SM is fed and the step becomes bandwidth-bound.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void fused_adam_kernel(
        scalar_t* __restrict__ p,
        const scalar_t* __restrict__ g,
        scalar_t* __restrict__ m,
        scalar_t* __restrict__ v,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1,
        const float bias_correction2,
        const int n) {
    const float step_size = lr / bias_correction1;
    const float bc2_sqrt = sqrtf(bias_correction2);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        const float grad = static_cast<float>(g[i]);
        float exp_avg = static_cast<float>(m[i]);
        float exp_avg_sq = static_cast<float>(v[i]);
        exp_avg = beta1 * exp_avg + (1.f - beta1) * grad;
        exp_avg_sq = beta2 * exp_avg_sq + (1.f - beta2) * grad * grad;
        // matches torch: denom = sqrt(v)/sqrt(bc2) + eps ; p -= step_size * m / denom
        const float denom = sqrtf(exp_avg_sq) / bc2_sqrt + eps;
        p[i] = static_cast<scalar_t>(static_cast<float>(p[i]) - step_size * exp_avg / denom);
        m[i] = static_cast<scalar_t>(exp_avg);
        v[i] = static_cast<scalar_t>(exp_avg_sq);
    }
}

// In-place update of p, exp_avg, exp_avg_sq for one tensor.
void fused_adam_step(
        at::Tensor p,
        at::Tensor grad,
        at::Tensor exp_avg,
        at::Tensor exp_avg_sq,
        double lr,
        double beta1,
        double beta2,
        double eps,
        double bias_correction1,
        double bias_correction2) {
    TORCH_CHECK(p.is_cuda(), "params must be CUDA");
    TORCH_CHECK(p.is_contiguous() && grad.is_contiguous() &&
                exp_avg.is_contiguous() && exp_avg_sq.is_contiguous(),
                "fused_adam requires contiguous tensors");
    TORCH_CHECK(p.numel() == grad.numel() && p.numel() == exp_avg.numel() &&
                p.numel() == exp_avg_sq.numel(), "size mismatch");

    const at::cuda::OptionalCUDAGuard guard(device_of(p));
    const int64_t n64 = p.numel();
    if (n64 == 0) return;
    // 32-bit indexing: our tensors are far below INT_MAX; guard so a future
    // huge tensor fails loudly instead of overflowing the index.
    TORCH_CHECK(n64 <= 0x7fffffffLL, "fused_adam: tensor too large for 32-bit indexing");
    const int n = static_cast<int>(n64);
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;  // grid-stride covers the remainder
    auto stream = at::cuda::getCurrentCUDAStream();

    // compute happens in fp32 inside the kernel regardless of storage dtype,
    // so half/bfloat16 params are fine (dispatched here, cast per-element)
    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            p.scalar_type(), "fused_adam_step", [&] {
        fused_adam_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            p.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
            exp_avg.data_ptr<scalar_t>(), exp_avg_sq.data_ptr<scalar_t>(),
            static_cast<float>(lr), static_cast<float>(beta1),
            static_cast<float>(beta2), static_cast<float>(eps),
            static_cast<float>(bias_correction1),
            static_cast<float>(bias_correction2), n);  // n is int32
    });
    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("step", &fused_adam_step, "fused Adam step (one thread per element)");
}
