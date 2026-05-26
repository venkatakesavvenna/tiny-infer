#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "loader.h"

// Define the CUDA Check again
#define CUDA_CHECK(call) do { \
    cudaError_t err = call;   \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void embedding_lookup(
    const int32_t* token_ids,
    const __half* embed_table,
    __half* output,
    int seq_len,
    int d_model
){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;

    // Bounds check
    if (row >= seq_len || col >= d_model) return;

    // Actual Lookup -> Row Major Storage
    output[row * d_model + col] = embed_table[token_ids[row] * d_model + col];
}

std::vector<__half> run_embedding_lookup(
    WeightMap& weights,
    int32_t* h_token_ids, // host token ids
    int seq_len,
    int d_model
) {
    // 1. Get embed table pointer from weights map
    __half* embed_table = reinterpret_cast<__half*>(weights["model.embed_tokens.weight"].d_ptr);

    // 2. Allocate device memory for token ids and output
    int32_t* d_token_ids = nullptr;
    __half* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_token_ids, seq_len * sizeof(int32_t)));  
    CUDA_CHECK(cudaMalloc(&d_output, seq_len * d_model * sizeof(__half))); 

    // 3. Copy token ids to device
    CUDA_CHECK(cudaMemcpy(d_token_ids, h_token_ids, seq_len * sizeof(int32_t), cudaMemcpyHostToDevice));

    // 4. Launch Kernel
    dim3 blockDim(128);
    dim3 gridDim((d_model + blockDim.x - 1) / blockDim.x, seq_len);

    // 5. Run Kernel
    embedding_lookup<<<gridDim, blockDim>>>(d_token_ids, embed_table, d_output, seq_len, d_model);

    // Wait for the kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Allocate host memory for output   
    std::vector<__half> h_output(seq_len * d_model);

    // 7. Copy output to host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, seq_len * d_model * sizeof(__half), cudaMemcpyDeviceToHost));

    // 8. Free device memory
    CUDA_CHECK(cudaFree(d_token_ids));
    CUDA_CHECK(cudaFree(d_output));

    return h_output;
}