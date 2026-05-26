#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "loader.h"
#include "model.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call;   \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Fused RMSNorm:
//   rms(x) = sqrt( (1/d) * sum(x_i^2) + eps )
//   y_i    = (x_i / rms(x)) * w_i
//
// Layout: one block per row (token). Threads cooperate over the hidden dim.
// Reduction: warp shuffle + shared-mem cross-warp combine. No intermediate
// global allocations — accumulation is in registers/shared mem only.
__global__ void rms_norm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int d_model,
    float eps
) {
    int row = blockIdx.x;
    float partial_sum = 0.0f;
    
    for (int j = threadIdx.x; j < d_model; j += blockDim.x) {
        float x_val = __bfloat162float(x[row * d_model + j]);
        partial_sum += x_val * x_val;
    }

    // Now reduce the threads within the same warp
    for (int delta = 16; delta > 0; delta /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, delta);
    }

    // Write warp to shared memory
    __shared__ float warp_sums[8];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0){
        warp_sums[warp_id] = partial_sum;
    }

    __syncthreads(); // Warp sync

    // Sum across warps
    float total_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        total_sum += warp_sums[i];
    }

    float rms = sqrtf(total_sum / d_model + eps);

    for (int j = threadIdx.x; j < d_model; j += blockDim.x) {
        float x_val = __bfloat162float(x[row * d_model + j]);
        float y_val = x_val / rms;
        output[row * d_model + j] = __float2bfloat16(y_val * __bfloat162float(weight[j]));
    }
}

std::vector<__nv_bfloat16> run_attention_forward(
    WeightMap& weights,
    const __nv_bfloat16* h_input,
    const std::string& weight_name,
    int seq_len,
    int d_model
) {
    __nv_bfloat16* d_weight = reinterpret_cast<__nv_bfloat16*>(weights[weight_name].d_ptr); // How to interpret the d_weight

    __nv_bfloat16* d_input = nullptr;
    __nv_bfloat16* d_output = nullptr;

}
