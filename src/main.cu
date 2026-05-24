// #include "loader.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "loader.h"

// Validation - Check the shapes we know Llama 3.2 1B must have.

// Load via Huggingface 
// LlamaForCausalLM(
//   (model): LlamaModel(
//     (embed_tokens): Embedding(128256, 2048)
//     (layers): ModuleList(
//       (0-15): 16 x LlamaDecoderLayer(
//         (self_attn): LlamaAttention(
//           (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
//           (k_proj): Linear(in_features=2048, out_features=512, bias=False)
//           (v_proj): Linear(in_features=2048, out_features=512, bias=False)
//           (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
//         )
//         (mlp): LlamaMLP(
//           (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
//           (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
//           (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
//           (act_fn): SiLUActivation()
//         )
//         (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
//         (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
//       )
//     )
//     (norm): LlamaRMSNorm((2048,), eps=1e-05)
//     (rotary_emb): LlamaRotaryEmbedding()
//   )
//   (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
// )

int main(int argc, char** argv) {
    // argc = argument count
    // argv = argument values (arg[0] is the program name, argv[1] is the first argument, etc.)

    // 1. Check that a path was provided
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path>\n", argv[0]);
        return 1;
    }

    // 2. Load the weights
    WeightMap weights = load_weights(argv[1]);
    
    // 3. Do something with the weights
    for (auto& [name, info]: weights) {
        // Each entry is a key-value pair
        // key   = Tensor name (string)
        // value = TensorInfo struct
        printf("Name: %s\nShape: %zu\nDtype: %s\n", name.c_str(), info.shape.size(), info.dtype.c_str());
    }

    // 4. Print total count
    printf("Total Tensors: %zu\n", weights.size());
    return 0;
}