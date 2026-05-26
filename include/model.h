#pragma once
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "loader.h"

std::vector<__nv_bfloat16> run_embedding_lookup(
    WeightMap &weights,
    int32_t *token_ids,
    int seq_len,
    int d_model
);

std::vector<__nv_bfloat16> run_rms_norm(
    WeightMap &weights,
    const __nv_bfloat16 *h_input,
    const std::string &weight_name,
    int seq_len,
    int d_model,
    float eps
);

