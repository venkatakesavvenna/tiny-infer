import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

MODEL_PATH = "/code/models/Llama-3.2-1B-Instruct"
TOKEN_IDS_PATH = "/code/build/token_ids.bin"
EMBED_PATH = "/code/build/embeddings.bin"
RMS_OUTPUT_PATH = "/code/build/rms_norm.bin"
WEIGHT_NAME = "model.layers.0.input_layernorm.weight"
D_MODEL = 2048
EPS = 1e-5

# 1. Load reference weight from HF checkpoint.
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
weight = dict(model.named_parameters())[WEIGHT_NAME].detach().float()  # [D_MODEL]

# 2. Load the embedding output that the CUDA harness fed into RMSNorm.
token_ids = np.fromfile(TOKEN_IDS_PATH, dtype=np.int32)
seq_len = token_ids.shape[0]
x = np.fromfile(EMBED_PATH, dtype=np.float32).reshape(seq_len, D_MODEL)
x_t = torch.from_numpy(x).float()

# 3. Reference RMSNorm (functional).
ref = F.rms_norm(x_t, normalized_shape=(D_MODEL,), weight=weight, eps=EPS).numpy()

# 4. Our CUDA output.
cuda_out = np.fromfile(RMS_OUTPUT_PATH, dtype=np.float32).reshape(seq_len, D_MODEL)

# 5. Compare.
diff = np.abs(ref - cuda_out)
max_diff = diff.max()
mean_diff = diff.mean()
print(f"seq_len    = {seq_len}")
print(f"Max Abs Diff : {max_diff:.6e}")
print(f"Mean Abs Diff: {mean_diff:.6e}")

assert max_diff < 1e-2, f"FAIL: max diff {max_diff} >= 1e-2"
print("Success!!!!!!")
