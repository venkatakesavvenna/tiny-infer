# Tiny-Infer
## A 60-Day Execution Plan · 1 Hour/Day

**Target:** A minimal but real CUDA inference engine for Llama 3.2 1B, with speculative decoding and INT8 KV cache quantization — built from scratch, benchmarked honestly, written up publicly.

**Stack:** C++ · CUDA · Python (tokenizer only) · Llama 3.2 1B Instruct

**Output:** 3 blog posts · 1 public GitHub repo · 1 benchmark table that tells the whole story

---

## The North Star

Every day of work should move one of these numbers:

| Metric | Baseline target | Optimized target |
|---|---|---|
| Tokens/sec (greedy, bs=1) | > 5 tok/s | > 40 tok/s |
| Time-to-first-token | measured | measurably reduced |
| Memory (4096 ctx) | measured | < 50% of baseline |
| Speculative decoding speedup | — | > 1.5x at bs=1 |

If a day's work doesn't move a number or unblock moving a number, you've drifted.

---

## Rules

**Rule 1: Correctness before speed.** At every stage, outputs must match HuggingFace transformers before you optimize. Wrong-but-fast is worthless.

**Rule 2: Every stage ends with a number.** Tokens/sec, memory usage, perplexity — something measurable. No "it works" without a benchmark.

**Rule 3: Don't reimplement what doesn't teach you anything.** Use HuggingFace tokenizers. Use safetensors loading. The learning is in the compute path, not the file format.

**Rule 4: When you're stuck for more than one session, post in GPU Mode.** Don't silently debug for three days. The community exists for this.

**Rule 5: Commit daily.** Even if it's broken. The git log is the learning log.

---

## Repo Structure (set this up on Day 1)

```
your-inference-engine/
├── src/
│   ├── main.cu              # Entry point
│   ├── model.cu             # Transformer forward pass
│   ├── attention.cu         # Your attention kernel (plugged in Week 3)
│   ├── kv_cache.cu          # Static KV cache (Week 2), paged (Week 4)
│   ├── rmsnorm.cu           # Fused RMSNorm kernel
│   ├── rope.cu              # Fused RoPE kernel
│   ├── ffn.cu               # FFN / SwiGLU
│   ├── sampler.cu           # Greedy sampling, then speculative
│   ├── quantization.cu      # INT8 KV cache (Month 2)
│   └── speculative.cu       # Speculative decoding (Month 2)
├── include/
│   ├── model.h
│   ├── attention.h
│   └── ...
├── python/
│   └── tokenizer.py         # Thin wrapper around HF tokenizers
├── benchmarks/
│   └── bench.sh             # Reproducible benchmark script
├── results/
│   └── README.md            # Running benchmark table — update every week
├── CMakeLists.txt
├── README.md                # Architecture diagram + benchmark table
└── BLOG.md                  # Notes that become blog posts
```

---

## Month 1 — Build the Engine

### Week 1: Weight Loading + Forward Pass (Days 1–7)

**Goal:** A single forward pass whose logits match HuggingFace transformers to float32 tolerance.

**Reference:** tiny-vllm for structure. HuggingFace transformers for numerical validation.

---

#### Day 1 — Repo setup + safetensors loading

**What to do:**
- Create the repo with the structure above
- Set up CMakeLists.txt with CUDA support
- Download `model.safetensors` from `meta-llama/Llama-3.2-1B-Instruct` on HuggingFace
- Write a loader that reads the safetensors header, maps tensor names to shapes and dtypes
- Allocate GPU memory for each tensor and copy weights to device
- Print all tensor names and shapes — verify they match what you expect

**Validation:** `model.embed_tokens.weight` shape is `[128256, 2048]`. All 32 layer tensors present.

**Commit message:** `feat: weight loading — all tensors on GPU`

---

#### Day 2 — Embedding lookup

**What to do:**
- Write a CUDA kernel: given a sequence of token IDs, look up their embeddings
- Input: `int32* token_ids [seq_len]`, `float16* embed_table [vocab_size, d_model]`
- Output: `float16* embeddings [seq_len, d_model]`
- Write a Python script that runs the same operation in HuggingFace and compares outputs

**Validation:** Max absolute difference between your embeddings and HuggingFace < 1e-3.

**Commit message:** `feat: embedding lookup kernel`

---

#### Day 3 — RMSNorm kernel

**What to do:**
- Implement RMSNorm: `output = (x / rms(x)) * weight`
- Write it as a fused CUDA kernel — one kernel, no intermediate allocations
- Each thread block handles one row (one token's hidden state)
- Use shared memory for the partial sum reduction

**Key math:**
```
rms(x) = sqrt( (1/d) * sum(x_i^2) + eps )
output_i = (x_i / rms(x)) * weight_i
```

**Validation:** Compare against `torch.nn.functional.rms_norm` on random input. Max diff < 1e-3.

**Note:** This is the kernel you'll later fuse with other ops. Write it cleanly.

**Commit message:** `feat: fused RMSNorm kernel`

---

#### Day 4 — RoPE (Rotary Position Embedding)

**What to do:**
- Implement RoPE applied to Q and K tensors before attention
- Rotate pairs of dimensions using sine/cosine of position-dependent frequencies
- Precompute the cos/sin tables on the CPU, copy to GPU once at startup

**Key math:**
```
theta_i = 1 / (10000 ^ (2i / d_head))
cos_m_i = cos(m * theta_i)
sin_m_i = sin(m * theta_i)

q'[2i]   = q[2i]   * cos_m_i - q[2i+1] * sin_m_i
q'[2i+1] = q[2i+1] * cos_m_i + q[2i]   * sin_m_i
```

**Validation:** Compare against HuggingFace LlamaRotaryEmbedding output. Max diff < 1e-3.

**Commit message:** `feat: RoPE kernel`

---

#### Day 5 — Naive attention

**What to do:**
- Implement multi-head attention — naive version, no Flash Attention yet
- Steps: QKV projection → reshape to heads → RoPE on Q and K → QK^T → scale → softmax → AV → output projection
- Use `cublasHgemm` for the matrix multiplications
- Implement softmax yourself (you'll understand why Flash Attention exists after this)

**Validation:** Compare attention output against HuggingFace LlamaAttention. Max diff < 1e-2 (FP16 accumulation drift is expected here).

**Note:** Write down how much GPU memory the attention matrix takes at seq_len=1024. This number motivates Flash Attention.

**Commit message:** `feat: naive multi-head attention`

---

#### Day 6 — FFN (SwiGLU)

**What to do:**
- Implement the feed-forward network: `FFN(x) = (SiLU(W1*x) * W3*x) W2`
- SiLU: `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`
- Use cublas for the linear layers, write the fused SiLU+multiply as a custom kernel

**Validation:** Compare against HuggingFace LlamaMLP. Max diff < 1e-2.

**Commit message:** `feat: SwiGLU FFN`

---

#### Day 7 — Full single forward pass

**What to do:**
- Wire everything together: embed → [RMSNorm → attention → residual → RMSNorm → FFN → residual] × 32 → final RMSNorm → lm_head
- Run on a real prompt: `"The capital of France is"`
- Compare final logits against HuggingFace

**Validation:** Top-1 predicted token from your logits matches HuggingFace top-1 token. This is the Week 1 milestone.

**Benchmark:** Record time for a single forward pass. This is your Day 7 baseline.

**Commit message:** `feat: full forward pass — logits match HuggingFace`

**Week 1 checkpoint:** Post your Day 7 benchmark in GPU Mode. "Single forward pass on Llama 3.2 1B: X ms." Get feedback.

---

### Week 2: Autoregressive Generation + KV Cache (Days 8–14)

**Goal:** Coherent text generation. "The capital of France is" → "Paris ..."

---

#### Day 8 — Static KV cache

**What to do:**
- Pre-allocate K and V tensors for the full context window: `[max_seq_len, n_heads, head_dim]` per layer
- Modify attention to write K and V into the cache at the current position
- Modify attention to read from positions 0..current_pos during decode

**What changes:** Attention now takes a `pos` argument. QK^T is computed between the current Q and all cached K values.

**Validation:** Output of a single decode step must match a full forward pass on the same tokens.

**Commit message:** `feat: static KV cache`

---

#### Day 9 — Greedy sampling

**What to do:**
- Implement argmax over the logits vocabulary (128256 tokens)
- Write a CUDA kernel for this — don't do it on CPU
- Return the token ID with the highest logit

**Note:** Use a parallel reduction. The vocabulary is large enough that a naive loop is measurably slow.

**Commit message:** `feat: greedy sampling kernel`

---

#### Day 10 — Generation loop

**What to do:**
- Wire the decode loop: forward pass → sample token → append to sequence → repeat
- Run: `"The capital of France is"` → generate 20 tokens
- Use your Python tokenizer wrapper to decode the output

**Validation:** Output is coherent English. "Paris" or equivalent should appear.

**This is the first real milestone. The engine works.**

**Benchmark:** Tokens/sec for greedy generation. Record this as your Day 10 baseline — every future optimization is measured against it.

**Commit message:** `feat: autoregressive generation loop`

---

#### Day 11 — Measure everything

**What to do:**
- Write `benchmarks/bench.sh` — a reproducible script that measures:
  - Time to first token (prefill latency)
  - Tokens/sec (decode throughput)
  - Peak GPU memory usage (`nvidia-smi`)
- Run at prompt lengths: 16, 64, 256, 512 tokens
- Fill in `results/README.md` with your first real benchmark table

**This table is the foundation of your README and your blog post.**

**Commit message:** `bench: baseline measurement harness`

---

#### Day 12 — Profile with Nsight

**What to do:**
- Run `nsys profile` on your generation loop
- Open the timeline: where is time actually going?
- Run `ncu` on the attention kernel specifically — check occupancy, memory throughput, arithmetic intensity
- Write down the top 3 bottlenecks you find

**Expected finding:** Attention is memory-bound. The softmax is eating time. This is why Flash Attention exists.

**Commit message:** `docs: Nsight profiling notes — identified bottlenecks`

---

#### Day 13 — Code cleanup + README first draft

**What to do:**
- Write `README.md`: what this is, how to build it, the benchmark table from Day 11
- Add architecture comments to the non-obvious kernel code
- Review your tensor naming — make it consistent

**Commit message:** `docs: README and inline comments`

---

#### Day 14 — Rest / catch-up

Use this day for whichever of Days 8–13 ran over. Don't skip it — debugging always takes longer than expected.

---

### Week 3: Plug In Your Kernels (Days 15–21)

**Goal:** Replace naive ops with your custom kernels. Measure the improvement at each step.

---

#### Day 15 — Plug in Flash Attention forward

**What to do:**
- Take your Flash Attention forward kernel from your previous project
- Integrate it: replace the naive QK^T → softmax → AV path
- The interface: takes Q, K, V tensors and returns output O — same as before

**Important:** The KV cache integration is the tricky part. Your Flash Attention kernel needs to handle the case where K and V come from the cache (variable length) not just a fresh tensor.

**Validation:** Outputs must match naive attention to within 1e-2. Any larger divergence is a bug.

**Commit message:** `feat: Flash Attention forward integrated`

---

#### Day 16 — Debug Flash Attention integration

**Blocked time for debugging.** Common issues:
- Causal masking off-by-one in the decode path
- Head dimension indexing wrong when reading from KV cache
- Softmax numerical instability at long sequences

Post in GPU Mode if stuck after 45 minutes. This is a known hard day.

**Commit message:** `fix: Flash Attention KV cache integration`

---

#### Day 17 — Benchmark attention before/after

**What to do:**
- Run `ncu` on both naive and Flash Attention kernels
- Compare: memory bandwidth used, arithmetic intensity, kernel time
- Update your benchmark table with attention-only timing
- Write down the roofline position of both kernels

**Expected result:** Flash Attention should be 2–4x faster on memory bandwidth, and your overall tokens/sec should improve by a meaningful amount.

**Commit message:** `bench: Flash Attention vs naive — results`

---

#### Day 18 — Fused RMSNorm kernel

**What to do:**
- You wrote RMSNorm on Day 3. Now fuse it with the residual add:
  `output = RMSNorm(x + residual)`
- This eliminates one read/write pass over the hidden state
- Replace all RMSNorm calls in the forward pass

**Validation:** Outputs match unfused version. Kernel time reduced.

**Commit message:** `feat: fused residual + RMSNorm kernel`

---

#### Day 19 — Fused RoPE kernel

**What to do:**
- Fuse RoPE into the QKV projection: apply rotations immediately after computing Q and K, before the attention kernel
- This eliminates a separate kernel launch and keeps Q/K in registers

**Validation:** Outputs match. Profile confirms one fewer kernel in the timeline.

**Commit message:** `feat: fused RoPE kernel`

---

#### Day 20 — Full benchmark after custom kernels

**What to do:**
- Run your full benchmark suite from Day 11
- Compare against Day 10 baseline
- Update `results/README.md`
- The table should show clear improvement at every prompt length

**Expected improvement:** 2–5x tokens/sec over Day 10 baseline. If less, profile again.

**Commit message:** `bench: Week 3 results — custom kernels vs baseline`

---

#### Day 21 — Nsight deep dive

**What to do:**
- Profile the full decode loop with all custom kernels
- Confirm attention is now compute-bound (not memory-bound)
- Find the next bottleneck — it's probably the FFN linear layers
- Write up your findings in `BLOG.md` — this becomes blog post 1

**Commit message:** `docs: Nsight analysis — roofline position after optimization`

---

### Week 4: Paged KV Cache (Days 22–30)

**Goal:** Replace static KV cache with a paged allocator. Measure memory savings.

---

#### Day 22 — Read the paper, draw the design

**What to do:**
- Read the vLLM PagedAttention paper (Kwon et al., 2023) — all of it
- Draw the block table on paper: physical blocks, logical positions, the mapping between them
- Write pseudocode for your allocator before writing any CUDA

**Do not write code today.** The design work is the work.

**Key concepts to understand:**
- Why fixed KV cache allocation causes fragmentation
- What a "physical block" is (fixed-size chunk of KV cache memory)
- How the block table maps sequence positions to physical blocks
- What happens when a sequence grows beyond its current blocks

---

#### Day 23 — Block allocator

**What to do:**
- Implement `BlockAllocator`: manages a pool of fixed-size physical blocks
- Interface:
  ```
  BlockAllocator(int num_blocks, int block_size)
  int allocate()        // returns block ID, or -1 if OOM
  void free(int block_id)
  int num_free_blocks()
  ```
- Block size: 16 tokens per block (matches vLLM default)
- Track free blocks with a simple free list

**Validation:** Allocate all blocks, free half, allocate again. No leaks.

**Commit message:** `feat: KV cache block allocator`

---

#### Day 24 — Block table

**What to do:**
- Implement the block table: maps a sequence's logical token positions to physical block IDs
- Interface:
  ```
  BlockTable(BlockAllocator* allocator)
  void append_token()          // allocates new block if needed
  int get_block_id(int pos)    // returns physical block for position pos
  int get_block_offset(int pos) // offset within block
  ```

**Commit message:** `feat: block table for sequence KV management`

---

#### Day 25 — Paged attention kernel

**What to do:**
- Modify your attention kernel to handle paged KV access
- Instead of reading K and V from a contiguous tensor, read from scattered physical blocks
- The kernel receives: Q tensor, block table (array of block IDs), physical KV memory pool
- For each KV position: look up block ID → compute physical address → load K or V

**This is the hardest kernel you'll write in this project.** The indexing is subtle.

**Commit message:** `feat: paged attention kernel (WIP)`

---

#### Day 26 — Debug paged attention

**Blocked time for debugging.** Common issues:
- Off-by-one in block offset calculation
- Wrong stride when addressing physical KV blocks
- Causal mask not accounting for block boundaries

**Validation test:** Paged attention output must exactly match your static KV cache output on the same input. Any difference is a bug.

Post in GPU Mode if stuck. This is the hardest day of Month 1.

**Commit message:** `fix: paged attention — correctness verified`

---

#### Day 27 — Integration and correctness

**What to do:**
- Replace static KV cache with paged KV cache throughout the forward pass
- Run full generation: output must match Day 10 baseline exactly
- Run at context lengths: 512, 1024, 2048, 4096

**Validation:** Token-for-token identical output to static KV cache version.

**Commit message:** `feat: paged KV cache fully integrated`

---

#### Day 28 — Memory benchmark

**What to do:**
- Measure peak GPU memory at context lengths: 512, 1024, 2048, 4096
- Compare paged vs static KV cache memory usage
- Update benchmark table
- Measure if tokens/sec changed (it shouldn't — correctness first)

**Expected result:** Paged KV cache uses less memory due to no pre-allocation of unused positions.

**Commit message:** `bench: paged vs static KV cache memory comparison`

---

#### Day 29 — Blog post 1 draft

**Write: "Building a Minimal LLM Inference Engine in CUDA"**

Structure:
1. Why I built this (the learning goal)
2. The architecture — what each component does
3. Week 1–2: Getting correctness (the forward pass, the decode loop)
4. Week 3: Plugging in custom kernels — the benchmark numbers
5. Week 4: Paged KV cache — why it matters, what I measured
6. What's next (speculative decoding, INT8 KV cache)

**Include:** Your benchmark table. Nsight screenshots. The architecture diagram.

**Tone:** Write like you're explaining to a friend who knows ML but hasn't done kernel work. Not a tutorial — a story of what you built and what you learned.

---

#### Day 30 — Publish + share

**What to do:**
- Polish and publish blog post 1 (Medium, personal site, or dev.to)
- Post in GPU Mode: "I built a minimal CUDA inference engine. Here's what I learned and what the numbers look like: [link]"
- Tag it on your GitHub repo README
- Update your resume with the project

**This is the Month 1 checkpoint. You have a working engine. Everything from here is making it better.**

---

## Month 2 — Push the Limits

### Week 5: Speculative Decoding — Theory + Setup (Days 31–37)

**Goal:** A working speculative decode loop. Output must be identical to standard greedy decoding.

**Read before Day 31:** Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)

---

#### Day 31 — The math, by hand

**What to do:**
- Read the speculative decoding paper, sections 1–3
- Work through the token acceptance probability derivation by hand:

```
Given:
  p(x) = target model probability for token x
  q(x) = draft model probability for token x

Accept draft token with probability: min(1, p(x) / q(x))

If rejected, sample from adjusted distribution:
  p'(x) = normalize(max(0, p(x) - q(x)))
```

- Write out a 3-token example on paper: what gets accepted, what gets rejected, what gets resampled

**Do not write code today.** The math is the work.

**Key insight to internalize:** The output distribution is exactly equal to sampling from the target model directly. Speculative decoding doesn't change quality — it changes speed.

---

#### Day 32 — Load draft model

**What to do:**
- Load a second model as the drafter — use Llama 3.2 1B as drafter, you'll need a slightly larger model as the verifier target, OR use Llama 3.2 1B as target and a smaller distilled model as drafter
- Simplest setup: use your existing engine with Llama 3.2 1B as both — just to verify the plumbing works before adding a real draft model
- Verify the draft model generates tokens correctly with your existing generation loop

**Commit message:** `feat: draft model loading`

---

#### Day 33 — Draft loop

**What to do:**
- Implement `generate_draft_tokens(prompt, K)`:
  - Run the draft model autoregressively for K steps
  - Return K candidate tokens AND the draft model's probability distribution for each
- Start with K=4

**Key implementation detail:** The draft model needs its own KV cache — separate from the target model's cache.

**Commit message:** `feat: draft token generation loop`

---

#### Day 34 — Verification pass

**What to do:**
- Implement the verification step:
  - Run the target model on `[prompt + K draft tokens]` in a single forward pass
  - Get target model probabilities for each of the K positions
- This is the key efficiency gain: one forward pass verifies K tokens instead of K separate forward passes

**Commit message:** `feat: parallel verification pass`

---

#### Day 35 — Acceptance/rejection sampling

**What to do:**
- Implement the acceptance kernel:
  ```
  for each draft token i:
    r = uniform random in [0, 1]
    if r < min(1, p_target[i] / p_draft[i]):
      accept token i
    else:
      reject token i, sample from adjusted distribution, stop
  ```
- Write this as a CUDA kernel — the random number generation is the tricky part (use cuRAND)

**Commit message:** `feat: acceptance/rejection sampling kernel`

---

#### Day 36 — End-to-end speculative decode

**What to do:**
- Wire everything together:
  1. Generate K draft tokens
  2. Run parallel verification
  3. Accept/reject with your sampling kernel
  4. Advance position by number of accepted tokens
  5. Repeat
- Run: `"The capital of France is"` → generate 50 tokens

**Validation:** Token-for-token output must match standard greedy decoding. This is non-negotiable. Any difference means your acceptance sampling math is wrong.

**This is the hardest correctness validation in the project. Don't move on until it passes.**

**Commit message:** `feat: speculative decoding end-to-end`

---

#### Day 37 — First speculative decoding benchmark

**What to do:**
- Measure: tokens/sec with and without speculative decoding at K=4
- Measure acceptance rate on 3 different prompt types:
  - Factual: "The capital of France is..."
  - Creative: "Once upon a time in a land..."
  - Code: "def fibonacci(n):"
- Record everything in your benchmark table

**Expected finding:** Acceptance rate varies significantly by prompt type. Code prompts tend to have high acceptance (predictable next tokens). Creative prompts lower.

**Commit message:** `bench: speculative decoding first results`

---

### Week 6: Speculative Decoding — Optimize + Measure (Days 38–44)

---

#### Day 38 — Sweep K values

**What to do:**
- Run your benchmark at K = 1, 2, 4, 6, 8
- For each K: tokens/sec, acceptance rate, overhead per speculation step
- Find the K that maximizes throughput for your hardware and model

**Expected finding:** There's a sweet spot. Too-large K means more rejected tokens and wasted draft compute. Too-small K means not enough speculation benefit.

**Commit message:** `bench: K sweep for speculative decoding`

---

#### Day 39 — Prompt type analysis

**What to do:**
- Run 10 different prompts across 3 categories: factual, creative, code
- Measure acceptance rate for each
- Build intuition for when speculative decoding wins and loses

**Key insight to develop:** Speculative decoding is a bet that the draft model's distribution is close to the target's. When that bet is right (predictable text), you win big. When it's wrong (surprising text), you waste compute.

**Commit message:** `bench: acceptance rate by prompt type`

---

#### Day 40 — Find the batch size crossover

**What to do:**
- Run your benchmark at batch sizes: 1, 2, 4, 8 (you'll need to implement simple batching if you haven't)
- Measure where speculative decoding stops helping
- This is the most interesting finding in the whole project — the crossover point

**Expected finding:** Speculative decoding wins at batch_size=1 and breaks even or loses at batch_size=4+. The reason: at high batch sizes, the target model is already compute-bound, and speculation adds overhead without adding throughput.

**Commit message:** `bench: speculative decoding batch size crossover`

---

#### Day 41 — Profile the verification kernel

**What to do:**
- Profile the parallel verification step with ncu
- Find: where is the overhead? Draft generation, verification, or acceptance sampling?
- Often the verification step is faster than expected because it runs in parallel

**Commit message:** `docs: speculative decoding profiling notes`

---

#### Day 42 — Write up the findings

**What to do:**
- Fill in `BLOG.md` with your speculative decoding results
- The interesting parts: the K sweep curve, the prompt type analysis, the batch size crossover
- The surprising finding: what did you measure that you didn't expect?

---

#### Day 43 — Blog post 2 draft

**Write: "Implementing Speculative Decoding From Scratch — What the Benchmarks Actually Showed"**

Structure:
1. What speculative decoding is (the math, briefly)
2. How I implemented it (the three components: draft, verify, accept)
3. The correctness validation — why this is harder than it sounds
4. The benchmark results: K sweep, prompt types, batch size crossover
5. The surprising finding
6. When to use it and when not to

**This is the post that gets shared. Write for someone who has heard of speculative decoding but never looked at the internals.**

---

#### Day 44 — Publish blog post 2

**What to do:**
- Polish and publish
- Post in GPU Mode and FlashInfer Slack
- Engage with comments — this is how you meet people

---

### Week 7: INT8 KV Cache Quantization (Days 45–51)

**Goal:** Halve KV cache memory with INT8 quantization. Measure the quality tradeoff honestly.

**Read before Day 45:** The KV cache quantization section of the vLLM blog post on INT8 KV cache, and the SmoothQuant paper section 3.

---

#### Day 45 — The math and the plan

**What to do:**
- Understand per-tensor symmetric INT8 quantization:
  ```
  scale = max(abs(x)) / 127
  x_int8 = round(x / scale).clamp(-128, 127)
  x_reconstructed = x_int8 * scale
  ```
- Understand where the error comes from and why it grows with sequence length
- Plan the implementation: quantize on write to KV cache, dequantize on read

**Do not write code today.** Design on paper.

---

#### Day 46 — INT8 quantization kernel (KV write)

**What to do:**
- Write a CUDA kernel that quantizes a FP16 K or V tensor to INT8 as it's written to the KV cache
- Store: INT8 values + FP32 scale per tensor
- Keep the scale on GPU in a separate small tensor

**Commit message:** `feat: INT8 KV cache write quantization kernel`

---

#### Day 47 — INT8 dequantization kernel (KV read)

**What to do:**
- Write a CUDA kernel that dequantizes INT8 K and V to FP16 as they're read in the attention kernel
- Fuse this into your paged attention kernel — don't add a separate kernel launch
- Modify attention to: load INT8 block → dequantize inline → use in attention computation

**Commit message:** `feat: INT8 KV cache dequantization fused into attention`

---

#### Day 48 — Correctness + quality measurement

**What to do:**
- Run generation with INT8 KV cache. Output will differ from FP16 — this is expected.
- Measure perplexity on a small set of prompts: compare FP16 vs INT8 KV cache
- How to measure perplexity: run the model on a fixed text, compare log probabilities

**Expected finding:** Perplexity degradation is small (< 0.5 points) at short contexts and grows at very long contexts (> 2048 tokens). This is the quality cliff you're looking for.

**Commit message:** `bench: INT8 KV cache quality measurement`

---

#### Day 49 — Memory benchmark

**What to do:**
- Measure peak GPU memory at context lengths: 512, 1024, 2048, 4096 with FP16 vs INT8 KV cache
- Expected: ~50% memory reduction for the KV cache (INT8 vs FP16)
- Measure if tokens/sec changes (dequantization overhead)

**Commit message:** `bench: INT8 KV cache memory savings`

---

#### Day 50 — Find the quality cliff

**What to do:**
- Run generation at increasing context lengths: 256, 512, 1024, 2048, 4096
- At each length: measure perplexity delta (INT8 vs FP16)
- Find the context length where quality degradation becomes visible
- This is the most practically useful finding in this section

**Expected finding:** Quality is essentially identical up to ~1024 tokens and begins diverging at very long contexts. The exact cliff depends on your model and quantization scheme.

**Commit message:** `bench: INT8 KV quality cliff analysis`

---

#### Day 51 — Full stack benchmark

**What to do:**
- Run the full combined stack: paged KV cache + Flash Attention + speculative decoding + INT8 KV cache
- Compare against your Day 10 baseline
- Build the final benchmark table: Baseline → +Custom Kernels → +Paged KV → +Spec Decode → +INT8 KV

This table is the story of the whole project.

| Configuration | Tokens/sec | Memory (2048 ctx) | Notes |
|---|---|---|---|
| Day 10 baseline | X | X | Naive everything |
| + Custom kernels | X | X | Flash Attn, fused RMSNorm |
| + Paged KV cache | X | X | No memory pre-allocation |
| + Speculative decode (K=4) | X | X | 1.5–2x at bs=1 |
| + INT8 KV cache | X | X | ~50% KV memory reduction |

**Commit message:** `bench: full stack benchmark table`

---

### Week 8: Polish and Ship (Days 52–60)

---

#### Day 52 — Blog post 3 draft

**Write: "INT8 KV Cache Quantization — Halving Memory With Measurable Quality Tradeoffs"**

Structure:
1. Why KV cache memory is the bottleneck at long contexts
2. How INT8 quantization works (the math)
3. Implementation: quantize on write, dequantize on read, fused into paged attention
4. The benchmark results: memory savings, tokens/sec overhead
5. The quality cliff: where does INT8 hurt you?
6. The combined picture: what you get when you run everything together

---

#### Day 53 — Publish blog post 3

Post everywhere: GPU Mode, FlashInfer Slack, LinkedIn, Twitter/X.

---

#### Day 54 — README overhaul

**What to do:**
- Write the final README:
  - What this project is (one paragraph)
  - Architecture diagram (draw it — ASCII is fine, a real diagram is better)
  - The final benchmark table from Day 51
  - How to build and run
  - Link to all three blog posts
  - What's not implemented (be honest: no batching, no beam search, single GPU)

---

#### Day 55 — Code cleanup

**What to do:**
- Remove all dead code
- Add comments on every non-obvious kernel decision
- Make sure every kernel has a comment explaining: what it does, what the memory access pattern is, why it's written this way
- Run clang-format

---

#### Day 56 — The performance summary post

**What to do:**
- Write one final short post: "The complete picture — every optimization I made and what each one bought"
- This is the benchmark table with two paragraphs per row explaining the implementation and the result
- This is the most useful reference post — engineers who want to do the same thing will bookmark it

---

#### Day 57 — Post in GPU Mode

**What to do:**
- Post the full project with the benchmark table
- Ask specific questions: "My speculative decoding acceptance rate on code prompts is X — is that expected? What would improve it?"
- Specific questions get specific answers from the people who know

---

#### Day 58 — 5-minute walkthrough video (optional but high signal)

**What to do:**
- Record a screen share walking through the codebase
- Not a tutorial — just "here's the repo structure, here's the interesting kernel, here's the benchmark"
- 5 minutes max
- Post it with the GitHub repo

This is optional but consistently outperforms written READMEs for getting people to actually look at your work.

---

#### Day 59 — Resume + LinkedIn update

**What to do:**
- Add to resume under Projects:
  ```
  Minimal CUDA LLM Inference Engine                                    2026
  CUDA C++ · Flash Attention · Speculative Decoding · INT8 Quantization
  - Built a minimal Llama 3.2 1B inference engine from scratch in CUDA/C++,
    implementing custom Flash Attention, fused RMSNorm, paged KV cache,
    speculative decoding (K=4), and INT8 KV cache quantization
  - Achieved Xtok/s vs Xtok/s baseline; 50% KV memory reduction; 1.8x
    speculative decoding speedup at batch size 1
  - 3 published technical blog posts; open source on GitHub
  ```
- Fill in the actual numbers from your benchmark table

---

#### Day 60 — Rest

You built an inference engine.

---

## Reference

### Papers (read in this order)

1. Vaswani et al. — "Attention Is All You Need" (2017) — read before Week 1
2. Dao et al. — "FlashAttention" (2022) — read before Day 15
3. Kwon et al. — "Efficient Memory Management for LLM Serving with PagedAttention" (2023) — read before Day 22
4. Leviathan et al. — "Fast Inference from Transformers via Speculative Decoding" (2023) — read before Day 31
5. Xiao et al. — "SmoothQuant" (2022), Section 3 — read before Day 45

### Tools

- `nsys profile ./your_binary` — timeline profiling
- `ncu --set full ./your_binary` — kernel-level profiling
- `nvidia-smi --query-gpu=memory.used --format=csv -l 1` — live memory monitoring
- `nvcc -lineinfo` — enable source line info in profiles

### Reference codebases (read, don't copy)

- tiny-vllm: `github.com/jmaczan/tiny-vllm` — structure and weight loading
- llm.c: `github.com/karpathy/llm.c` — clean C/CUDA style reference
- vLLM source: paged attention implementation in `csrc/attention/`

### Communities

- GPU Mode Discord — post benchmarks, ask specific questions
- FlashInfer Slack — for attention kernel questions specifically

---

## Weekly Checkpoint Template

Copy this into `results/README.md` every Sunday:

```
## Week N checkpoint — [date]

### What I built this week
- 

### Benchmark numbers
| Metric | This week | Last week | Day 10 baseline |
|---|---|---|---|
| Tokens/sec | | | |
| Memory (2048 ctx) | | | |
| Time to first token | | | |

### What surprised me
- 

### What I'm stuck on
- 

### GPU Mode post: [link if posted]
```

---

*Start Day 1. Build the thing.*
