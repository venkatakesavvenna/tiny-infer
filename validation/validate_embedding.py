import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/code/models/Llama-3.2-1B-Instruct"
TOKEN_IDS_PATH = "/code/build/token_ids.bin" # int32, written by C++ test harness
CUDA_OUTPUT_PATH = "/code/build/embeddings.bin" # float16, written by C++ test harness
D_MODEL = 2048

# 1. Load Huggingface model and pull the embedding table
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
embed = model.model.embed_tokens

# 2. Load token IDs from C++ test harness
token_ids = np.fromfile(TOKEN_IDS_PATH, dtype=np.int32)
seq_len = token_ids.shape[0]
print(f"seq_len = {seq_len}")

# # 3. Run HF embedding lookup (reference)
with torch.no_grad():
    ref = embed(torch.from_numpy(token_ids).long()).cpu().numpy()

# 4. Load our CUDA Kernel Output
cuda_out = np.from_file(CUDA_OUTPUT_PATH, dtype=np.float16).reshape(seq_len, D_MODEL)

# 5. Compare
diff = np.abs(ref.astype(np.float32)) - cuda_out.astype(np.float32)
max_diff = diff.max()
mean_diff = diff.mean()

print(f"Max Abs Diff: {max_diff:.6e}")

# 6. Max Difference
assert max_diff < 1e-3, f"FAIL: max diff {max_diff} >= 1e-3"