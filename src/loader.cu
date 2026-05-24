#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstring>

#include "loader.h"

// CUDA CHECK MACRO
#define CUDA_CHECK(call) do { \
    cudaError_t err = call;   \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,  cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }   \
} while (0) 

size_t dtype_bytes(const std::string& dtype) {
    if (dtype == "BF16") return 2; // F16 = 2 bytes
    if (dtype == "F32") return 4; // F16 = 4 bytes
    if (dtype == "I8") return 1;  // F16 = 1 bytes

    throw std::runtime_error("Unknown dtype: " + dtype);
}

WeightMap load_weights(const std::string& path){
    // 1. Open the file
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open: " + path);

    // 2. Get File Size
    fseek(f, 0, SEEK_END);          // jump the cursor to the end of file
    size_t file_size = ftell(f);    // Ask => where is cursor now?
    fseek(f, 0, SEEK_SET);          // jump the cursor back to the Start

    // 3. Read entire file into host buffer
    std::vector<uint8_t> buf(file_size);
    size_t bytes_read = fread(buf.data(), 1, file_size, f);
    if (bytes_read != file_size) throw std::runtime_error("File read error");
    fclose(f);

    // 4. Parse header length from first 8 bytes
    uint64_t header_len = 0;
    memcpy(&header_len, buf.data(), 8);

    // printf("Header Length: %zu\n", header_len);

    // 5. Extract JSON string
    std::string json(reinterpret_cast<char*>(buf.data() + 8), header_len);
    
    // 6. Parse JSON and load tensors
    WeightMap weights;

    // JSON Format
    // {
    //   "model.embed_tokens.weight": {
    //     "dtype": "F16",
    //     "shape": [128256, 2048],
    //     "data_offsets": [0, 524288000]
    //   },
    //   "model.layers.0.self_attn.q_proj.weight": { ... },
    //   ...
    // }

    // JSON: {"__metadata__":{"format":"pt"},"model.embed_tokens.weight":{"dtype":"BF16","sh

    // Parse json using custom function
    size_t start_idx = 0;
    size_t end_idx = 0;

    // Start Idx should be after the metadata prefix
    start_idx = json.find("\"__metadata__\":{\"format\":\"pt\"}");
    if (start_idx == std::string::npos) throw std::runtime_error("Invalid JSON format");

    start_idx += 31;

    while( start_idx < json.length() ) {
        // Print json after the start_idx

        start_idx = json.find("\"", start_idx);
        if (start_idx == std::string::npos) break;

        end_idx = json.find("\":", start_idx);


        std::string name = json.substr(start_idx + 1, end_idx - start_idx - 1);

        // Now, search for dtype
        start_idx = json.find("\"dtype\":\"", start_idx);
        if (start_idx == std::string::npos) break;

        end_idx = json.find("\",", start_idx);
        std::string dtype = json.substr(start_idx + 9, end_idx - start_idx - 9);

        // Now, search for shape
        start_idx = json.find("\"shape\":[", start_idx);
        if (start_idx == std::string::npos) break;

        end_idx = json.find("],", start_idx);
        // Shape is a vector of integers
        std::string shape = json.substr(start_idx + 9, end_idx - start_idx - 9);
        std::vector<int64_t> shape_ints;
        std::string::size_type prev = 0;
        std::string::size_type next = 0;
        while ((next = shape.find(',', prev)) != std::string::npos) {
            shape_ints.push_back(stoll(shape.substr(prev, next - prev)));
            prev = next + 1;
        }
        shape_ints.push_back(stoll(shape.substr(prev)));

        // printf("Shape: [");
        // for (auto& s: shape_ints) {
        //     printf("%ld, ", s);
        // }
        // printf("]\n");

        // A Pair for the data offsets
        std::pair<uint64_t, uint64_t> data_offsets;
        start_idx = json.find("\"data_offsets\":[", start_idx);
        if (start_idx == std::string::npos) break;

        end_idx = json.find("]},", start_idx);
        std::string data_offsets_str = json.substr(start_idx + 16, end_idx - start_idx - 16);

        // printf("Data Offsets: %s\n", data_offsets_str.c_str());

        std::string::size_type prev2 = 0;
        std::string::size_type next2 = 0;
        while ((next2 = data_offsets_str.find(',', prev2)) != std::string::npos) {
            data_offsets.first = stoll(data_offsets_str.substr(prev2, next2 - prev2));
            prev2 = next2 + 1;
        }
        data_offsets.second = stoll(data_offsets_str.substr(prev2));

        // printf("Data Offsets: [%lu, %lu]\n", data_offsets.first, data_offsets.second);

        // Find the next }
        start_idx = json.find('}', end_idx);
        if (start_idx == std::string::npos) break;

        // 1. Compute total bytes to allocate
        size_t nbytes = (data_offsets.second - data_offsets.first); // hint: num_elements * bytes_per_dtype

        // 2. Allocate on GPU
        void* d_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ptr, nbytes));

        const uint8_t* data_ptr = buf.data() + 8 + header_len + data_offsets.first;
        // 3. Copy from host to device
        CUDA_CHECK(cudaMemcpy(d_ptr, data_ptr, nbytes, cudaMemcpyHostToDevice));

        // 4. Store in weights map
        TensorInfo info; 

        info.name = name;
        info.dtype = dtype;
        info.shape = shape_ints;
        info.d_ptr = d_ptr;

        weights[name] = info;

        start_idx++;
    }

    // Code
    // printf("Loaded %zu tensors.\n", weights.size());

    return weights;
}