#pragma once // prevents the file being included twice

// Include string
#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_bf16.h>

struct TensorInfo {
    std::string name;           // Tensor name
    std::vector<int64_t> shape; // Shape
    std::string dtype;          // Dtype -> string
    void* d_ptr;                 // Pointer to data -> convention in CUDA (on device)
};

using WeightMap = std::unordered_map<std::string, TensorInfo>;

// Function Declaration
WeightMap load_weights(const std::string& path); // Load weights

void free_weights(WeightMap& weights);

size_t dtype_bytes(const std::string& dtype);

__global__ void bf16_to_fp32(const __nv_bfloat16* input, float* output, int n);

__global__ void sgemm(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m, int n, int k, float alpha, float beta);