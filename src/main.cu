// #include "loader.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cuda_bf16.h>

#include "loader.h"
#include "model.h"

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

    // 5. Pass some test token ids to see if the tensors are loading properly
    int32_t token_ids[] = {791, 6864, 315, 9822, 374};
    int seq_len = 5;
    int d_model = 2048;

    // Write token ids to a bin as well, in the build folder
    FILE* fp_tokens = fopen("/code/build/token_ids.bin", "wb");
    fwrite(token_ids, sizeof(int32_t), seq_len, fp_tokens);
    fclose(fp_tokens);

    std::vector<__nv_bfloat16> output = run_embedding_lookup(weights, token_ids, seq_len, d_model);

    // Convert the above into fp32
    std::vector<float> output_fp32(output.size());
    for (int i = 0; i < output.size(); i++) {
        output_fp32[i] = __bfloat162float(output[i]);
    }

    // Write to disk in fp16
    FILE* fp = fopen("/code/build/embeddings.bin", "wb");
    fwrite(output_fp32.data(), sizeof(float), output_fp32.size(), fp);
    fclose(fp);

    // Run RMSNorm on the embedding output using layer-0 input_layernorm weights.
    std::vector<__nv_bfloat16> rms_out = run_rms_norm(
        weights,
        output.data(),
        "model.layers.0.input_layernorm.weight",
        seq_len,
        d_model,
        1e-5f
    );

    std::vector<float> rms_out_fp32(rms_out.size());
    for (size_t i = 0; i < rms_out.size(); i++) {
        rms_out_fp32[i] = __bfloat162float(rms_out[i]);
    }

    FILE* fp_rms = fopen("/code/build/rms_norm.bin", "wb");
    fwrite(rms_out_fp32.data(), sizeof(float), rms_out_fp32.size(), fp_rms);
    fclose(fp_rms);

    return 0;
}