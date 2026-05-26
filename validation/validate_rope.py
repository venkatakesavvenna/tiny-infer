import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

MODEL_PATH = "/code/models/Llama-3.2-1B-Instruct"
TOKEN_IDS_PATH = "/code/build/token_ids.bin"
RMS_OUTPUT_PATH = "/code/build/rms_norm.bin"
Q_ROPE_PATH = "/code/build/q_rope.bin"
K_ROPE_PATH = "/code/build/k_rope.bin"

D_MODEL = 2048
HEAD_DIM = 64
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8

if __name__ == "__main__":
    # 1. Load reference model (fp32 for clean comparison).
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    model.eval()

    layer0 = model.model.layers[0]
    rotary_emb = model.model.rotary_emb  # HF computes inv_freq + scaling internally

    # 2. Load tokens + RMSNorm output that the CUDA harness already produced.
    token_ids = np.fromfile(TOKEN_IDS_PATH, dtype=np.int32)
    seq_len = int(token_ids.shape[0])
    rms_out = np.fromfile(RMS_OUTPUT_PATH, dtype=np.float32).reshape(seq_len, D_MODEL)
    x = torch.from_numpy(rms_out).float().unsqueeze(0)  # [1, seq_len, D_MODEL]

    # 3. Project to Q, K and reshape to [B, n_heads, seq, head_dim].
    with torch.no_grad():
        q = layer0.self_attn.q_proj(x)
        k = layer0.self_attn.k_proj(x)
        q = q.view(1, seq_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

        # Dump this data to disk.
        q.t().contiguous().cpu().numpy().tofile(Q_ROPE_PATH)
        k.t().contiguous().cpu().numpy().tofile(K_ROPE_PATH)

        # 4. Build position_ids [0..seq_len-1] and apply rotary embedding.
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        cos, sin = rotary_emb(x, position_ids)
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

    q_ref = q_rope.squeeze(0).cpu().numpy()  # [n_q_heads, seq, head_dim]
    k_ref = k_rope.squeeze(0).cpu().numpy()  # [n_kv_heads, seq, head_dim]

    # 5. Load CUDA outputs and compare.
    q_cuda = np.fromfile(Q_ROPE_PATH, dtype=np.float32).reshape(NUM_Q_HEADS, seq_len, HEAD_DIM)
    k_cuda = np.fromfile(K_ROPE_PATH, dtype=np.float32).reshape(NUM_KV_HEADS, seq_len, HEAD_DIM)

    q_diff = np.abs(q_ref - q_cuda)
    k_diff = np.abs(k_ref - k_cuda)
    print(f"seq_len    = {seq_len}")
    print(f"Q max diff : {q_diff.max():.6e}    mean: {q_diff.mean():.6e}")
    print(f"K max diff : {k_diff.max():.6e}    mean: {k_diff.mean():.6e}")

    assert q_diff.max() < 1e-2, f"FAIL: Q max diff {q_diff.max()} >= 1e-2"
    assert k_diff.max() < 1e-2, f"FAIL: K max diff {k_diff.max()} >= 1e-2"
    print("Success!!!!!!")
