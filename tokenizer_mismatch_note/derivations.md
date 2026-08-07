# Arithmetic Sheet — "The Price of a Mismatched Tokenizer"

All numbers computed with python3; arithmetic shown step by step. FLOP convention: ≈2N FLOPs per decoded token for N active parameters (Kaplan et al., 2020).

---

## (i) Embedding / LM-head parameter and FLOP shares

**Config sources:**
- **Gemma-2-2B** — HF `google/gemma-2-2b` `config.json`: `vocab_size=256000`, `hidden_size=2304`, `tie_word_embeddings=true`; total params 2,614,341,888 (HF model card, ≈2.6B). *(Correction 2026-07-29: an earlier draft used 256,128; the config value is 256,000 — confirmed by the layer-wise decomposition reproducing the exact total only with 256,000.)*
- **Llama-3.1-8B** — HF `meta-llama/Llama-3.1-8B` `config.json`: `vocab_size=128256`, `hidden_size=4096`, `tie_word_embeddings=false`; total params 8,030,261,248 (HF model card, ≈8.03B).
- **Llama-3.1-70B** — HF `meta-llama/Llama-3.1-70B` `config.json`: `vocab_size=128256`, `hidden_size=8192`, `tie_word_embeddings=false`; total params 70,553,706,496 (HF model card, ≈70.6B).

### Gemma-2-2B (tied)
- V·d = 256,000 × 2,304 = **589,824,000** ≈ 0.59B (derived)
- Embedding params (tied → count once) = 589,824,000 (derived)
- N_non-emb = 2,614,341,888 − 589,824,000 = **2,024,517,888** ≈ 2.02B (derived)
- Vd / N_non-emb = 589,824,000 / 2,024,517,888 = **0.2913 → 29.13%** (derived)
- Vd / (N_non-emb + Vd) = 589,824,000 / 2,614,341,888 = **0.2256 → 22.56%** (derived) ✓ sanity check: lands in the 20–25% band

### Llama-3.1-8B (untied)
- V·d = 128,256 × 4,096 = **525,336,576** ≈ 0.53B (derived)
- Embedding params (untied → 2·Vd) = 2 × 525,336,576 = 1,050,673,152 (derived)
- N_non-emb = 8,030,261,248 − 1,050,673,152 = **6,979,588,096** ≈ 6.98B (derived)
- Vd / N_non-emb = 525,336,576 / 6,979,588,096 = **0.0753 → 7.53%** (derived)
- Vd / (N_non-emb + Vd) = 525,336,576 / 7,504,924,672 = **0.0700 → 7.00%** (derived)

### Llama-3.1-70B (untied)
- V·d = 128,256 × 8,192 = **1,050,673,152** ≈ 1.05B (derived)
- Embedding params (untied → 2·Vd) = 2 × 1,050,673,152 = 2,101,346,304 (derived)
- N_non-emb = 70,553,706,496 − 2,101,346,304 = **68,452,360,192** ≈ 68.5B (derived)
- Vd / N_non-emb = 1,050,673,152 / 68,452,360,192 = **0.0153 → 1.53%** (derived)
- Vd / (N_non-emb + Vd) = 1,050,673,152 / 69,503,033,344 = **0.0151 → 1.51%** (derived) ✓ sanity check: < 2%

### Wasted FLOP share (dead-vocab fraction f)
Formula: wasted share ≈ f · Vd / N_non-emb.
- Gemma-2-2B at f = 0.5: 0.5 × 590,118,912 / 2,024,222,976 = **0.1458 → 14.58%** of non-embedding compute wasted per decoded token (derived)

### Wasted parameters at f = 0.3, fp16 (2 bytes/param)
Formula: f·Vd (tied) or f·2Vd (untied), × 2 bytes.
- Gemma-2-2B (tied): 0.3 × 590,118,912 = 177,035,674 params × 2 B = 354,071,347 B ≈ **354.1 MB** (derived)
- Llama-3.1-8B (untied): 0.3 × 1,050,673,152 = 315,201,946 params × 2 B = 630,403,891 B ≈ **630.4 MB** (derived)
- Llama-3.1-70B (untied): 0.3 × 2,101,346,304 = 630,403,891 params × 2 B = 1,260,807,782 B ≈ **1,260.8 MB ≈ 1.26 GB** (derived)

---

## (ii) KV-cache bytes per token — Llama-3.1-8B

Config source: HF `meta-llama/Llama-3.1-8B` `config.json`: `num_hidden_layers=32`, `num_key_value_heads=8`, `head_dim = hidden_size/num_attention_heads = 4096/32 = 128`.

bytes/token = layers × kv_heads × head_dim × 2 (K and V) × 2 (fp16 bytes)
= 32 × 8 × 128 × 2 × 2 = **131,072 B = 128 KiB per token** (derived; note KiB, not KB)

**×(1+L) example at L = 0.3, nominal 8k conversation (8,192 tokens):**
- Extra tokens = 8,192 × 0.3 = **2,457.6 tokens** (derived)
- Extra KV-cache = 2,457.6 × 131,072 B = 322,122,547 B ≈ **322.1 MB (307.2 MiB)** (derived)
- (Reference: full 8,192-token cache = 8,192 × 131,072 B = 1,073,741,824 B = exactly 1.00 GiB ≈ 1.07 GB.) (derived)

---

## (iii) Training logits tensor

B·T = 8,192 tokens, V = 128,256, fp32 (4 bytes):
8,192 × 128,256 × 4 = **4,202,692,608 B ≈ 4.20 GB (3.91 GiB)** (derived) ✓ ≈ 4 GB as expected — one 8k sequence's logits alone.

---

## (iv) L = 0.3 worked example

L = token-count inflation of the mismatched tokenizer over the matched one.
- Decode latency / per-token cost / KV-cache footprint: × (1 + L) = **×1.30** (derived)
- Effective context window: 1 / (1 + L) = 1/1.3 = **0.7692** → context loss = 1 − 1/1.3 = **0.2308 → 23.08%** (derived)
- Attention FLOPs (quadratic in sequence length): (1 + L)² = 1.3² = **1.69 → +69%** (derived)

---

## (v) Effective-context-loss mini-table

Loss = 1 − 1/(1+L):

| L | 1/(1+L) | context loss (derived) |
|------|---------|------------------------|
| 0.1 | 0.9091 | 9.09% |
| 0.3 | 0.7692 | 23.08% |
| 0.5 | 0.6667 | 33.33% |
| 1.0 | 0.5000 | 50.00% |

---

## (vi) Coverage identity sanity check

Derivation (2 lines):
1. A fraction (1−ρ) of tokens is uncovered; each such token is split into s pieces, i.e., contributes s tokens instead of 1 → extra (s−1) tokens per affected token.
2. Extra tokens per original token: L = (1−ρ)·(s−1); covered tokens (mass ρ) contribute 0 extra.

Check: (1−ρ) = 0.10, s = 3 → L = (3−1) × 0.10 = **0.20** (derived) ✓ matches the claimed 2 × 0.10.

---

## (vii) Rule-of-thumb check

Extra cross-entropy of 0.1 bits/char on top of H(Q) ≈ 1.2–1.4 bits/char for English → fractional token inflation L ≈ 0.1/H:
- H = 1.4: 0.1/1.4 = 0.0714 → **7.14%** (derived)
- H = 1.3: 0.1/1.3 = 0.0769 → **7.69%** (derived) ✓ ≈ 7.7%
- H = 1.2: 0.1/1.2 = 0.0833 → **8.33%** (derived)

Range: **7.1–8.3%**, central value ≈ 7.7% (derived).

---

## (viii) Composite formula worked example

cost(Q)/cost_opt(Q) ≈ (1 + D_KL/H) · (1 + f·Vd/N_non-emb), at D_KL/H = 0.3, f = 0.5, Gemma-2-2B ratio Vd/N_non-emb = 0.29134 from (i):

- Second factor: 1 + 0.5 × 0.291340 = 1.145670 (derived)
- Product: 1.3 × 1.145670 = **1.4894 → ≈ 49% cost overhead** vs. a matched-tokenizer, right-sized-vocab baseline (derived)

*(First-order in f·Vd/N. The exact ratio against a right-sized-vocabulary baseline is (N+Vd)/(N+(1−f)Vd) = 1.1270, giving 1.465 rather than 1.489; the first-order form overstates by ~2.4 points at f = 0.5.)*
