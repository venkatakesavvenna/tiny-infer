#include <cuda_runtime.h>

__global__ void sgemm(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m, int n, int k, float alpha, float beta){
    // Flattened IDs remapping
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n){
        float sum = 0.0f;
        for (int i = 0; i < k; i++){
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = alpha * sum + beta * c[row * n + col];
    }
}