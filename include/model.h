#pragma once
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "loader.h"

std::vector<__half> run_embedding_lookup(
    WeightMap &weights,
    int32_t *token_ids,
    int seq_len,
    int d_model
);