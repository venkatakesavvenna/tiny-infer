#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <stdexcept>

#include "loader.h"

// CUDA CHECK MACRO
#define CUDA_CHECK(call) do { \
    cudaError_t err = call;   \  
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,  cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }   \
} while (0) 

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
    fread(buf.data(), 1, file_size, f);
    fclose(f);

    // 4. Parse header length from first 8 bytes
    uint64_t header_len = 0;
    memcpy(&header_len, buf.data(), 8);

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

    // Parse json using custom function
    size_t start_idx = 0;
    size_t end_idx = 0;
    while( start_idx < json.length() ) {
        start_idx = json.find("\": {", start_idx);
        if (start_idx == std::string::npos) break;

        end_idx = json.rfind('"', start_idx - 1);
        std::string name = json.substr(end_idx + 1, start_idx - end_idx - 1);

        // Now, search for dtype
        start_idx = json.find("\"dtype\": \"", start_idx);
        if (start_idx == std::string::npos) break;

        end_idx = json.find("\",", start_idx);
        std::string dtype = json.substr(start_idx + 10, end_idx - start_idx - 10);


        // Now, search for shape
        start_idx = json.find("\"shape\": [", start_idx);
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

        // A Pair for the data offsets
        std::pair<uint64_t, uint64_t> data_offsets;
        start_idx = json.find("\"data_offsets\": [", start_idx);
        if (start_idx == std::string::npos) break;

        end_idx = json.find("],", start_idx);
        std::string data_offsets_str = json.substr(start_idx + 16, end_idx - start_idx - 16);

        std::string::size_type prev2 = 0;
        std::string::size_type next2 = 0;
        while ((next2 = data_offsets_str.find(',', prev2)) != std::string::npos) {
            data_offsets.first = stoul(data_offsets_str.substr(prev2, next2 - prev2));
            prev2 = next2 + 1;
        }
        data_offsets.second = stoul(data_offsets_str.substr(prev2));

        // Find the next }
        start_idx = json.find('}', end_idx);
        if (start_idx == std::string::npos) break;
    }

    // Code
    printf("JSON: %s\n", json.c_str());

    return weights;
}