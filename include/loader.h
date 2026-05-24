#pragma once // prevents the file being included twice

// Include string
#include <string>
#include <vector>
#include <unordered_map>

struct TensorInfo {
    std::string name;           // Tensor name
    std::vector<int64_t> shape; // Shape 
    std::string dtype;          // Dtype -> string
    void* d_ptr;                 // Pointer to data -> convention in CUDA (on device)
};

using WeightMap = std::unordered_map<std::string, TensorInfo>;

// Function Declaration
WeightMap load_weights(const std::string& path); // Load weights