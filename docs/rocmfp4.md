# ROCmFP4 (STRIX) quantization support

This branch adds first-class support for the **ROCmFP4** GGUF quantization format
(`Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST`), used by the ROCmFPX / rocmfp4-llama
quantization stack for AMD RDNA GPUs ("STRIX" format), on top of the DFlash2
speculative-decoding work from [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342).

Supported out of the box: model loading, GGUF conversion/quantization
(`llama-quantize`), CPU inference, and the full set of CUDA/HIP kernels
(MMVQ vector dot, MMQ tiled GEMM including the RDNA3 WMMA path, dequantization
to f32/f16/bf16, `MUL_MAT_ID` for MoE).

## Format

| | `Q4_0_ROCMFP4` (type 100) | `Q4_0_ROCMFP4_FAST` (type 101) |
|---|---|---|
| block | 32 values, 18 bytes | 32 values, 17 bytes |
| nibble layout | split: low nibbles of the 16 bytes = values 0..15, high nibbles = values 16..31 | same |
| codebook | Codebook10 (`kvalues_rocmfp4`, E2M1-like ±{0..10}) | same |
| scales | 2× unsigned UE4M3 per block (one per 16-value half), **half-scale** semantics (official ue4m3 map ÷ 2) | 1× UE4M3 per block |

Two things make this format unusual compared to stock `Q4_0`/`MXFP4`:

1. **Split-nibble layout.** The 16 bytes of each block do *not* pack two
   consecutive values per byte. Instead the low nibbles of *all* 16 bytes hold
   the first 16 values and the high nibbles hold the last 16
   (`qs[j] = val(j) | val(j+16) << 4`). Kernels that assume the Q4_0-style
   consecutive layout produce silently-wrong-but-plausible results — the MMQ
   tile loader and the dequantize kernel therefore decode nibble halves, not
   byte halves.
2. **Half-scale UE4M3.** Scale bytes follow the standard unsigned E4M3 map
   divided by 2 (`2^(exp-9)` for normal values). The CUDA side decodes through
   `rocmfp4_ue4m3_to_fp32_half()`; encodings > 0x7E (inf/NaN) clamp to 0.

## Reference serving configuration (RX 7900 XTX / gfx1100)

`scripts/serve-dflash2.sh` is the fully commented reference launcher. The
measurement context: Qwen3.8-27B (hybrid: 48 DeltaNet linear layers + 16 full
attention layers), 256K context, fp4 KV cache, RDNA3 wave32.

| measured (same card, same prompt) | tok/s |
|---|---|
| fp4 baseline, no speculation | 41 |
| + DFlash2 draft (n5/p0.4), creative prose | 30-35 |
| + DFlash2 draft, reasoning / chain-of-thought content | 68 |
| + ngram-map-k4v stacked, repetitive content | 98 |

**Speculative acceptance is dominated by content predictability**, not by the
implementation: two independent DFlash2 stacks draft byte-identical tokens.
Measure with a fixed content type or the numbers mean nothing.

### Every flag and why

**Quantization / memory path**

- `-ctk q4_0 -ctv q4_0` — quantize the KV cache to 4 bits. On this hybrid
  architecture only the 16 full-attention layers carry KV; at 4 bits the whole
  256K cache is ≈4.7 GiB instead of ≈19 GiB at f16, which is what makes 256K
  context fit a 24 GiB card at all. Decode is bandwidth-bound, so a 4x smaller
  KV stream also speeds up every token.
- `LLAMA_ATTN_ROT_DISABLE=1` (environment) — mandatory with a quantized KV
  cache on HIP. llama.cpp's KV-quant accuracy path re-encodes K/V through a
  Hadamard rotation (`k_rot`/`v_rot` in the graph); the hybrid DeltaNet cache
  path does not implement it. Without this variable the process crashes.
- `-ngl 99` / `-ngld 99` — full offload of target and draft models.

**Speculative decoding**

- `--spec-type ngram-map-k4v,draft-dflash` — two stacked speculators, tried in
  listed order. ngram hits are free (no draft-model forward) and near-exact;
  DFlash2 covers everything the n-gram table misses.
- `--spec-ngram-map-k4v-size-n 12 / size-m 48 / min-hits 1` — look up the last
  N=12 tokens as a key; on a hit, draft up to M=48 continuation tokens; a
  single prior occurrence is enough (`min-hits 1`). This is the 2.4x lever on
  repetitive content (boilerplate, templates, long conversations) and idle
  otherwise.
- `--spec-draft-n-max 5` — DFlash2 drafts one block of `block_size=8` rows
  (anchor + 7 mask slots) but only the first `n_max` mask positions are kept.
  Measured sweep: n4 = 70, n5 = 76, n6 = 73 tok/s on reasoning content.
- `--spec-draft-p-min 0.4` — stop appending draft tokens when the selector's
  confidence at the argmax (softmax over the top-k transition scores) drops
  below 0.4. Sweep: p0.5 = 67, p0.45 = 73, **p0.4 = 76**, p0.3 = 74 tok/s.
  Low-confidence tail drafts cost a full verification pass each and are
  rejected too often to pay for it.

**Context / stability**

- `-c 262144` — 256K context. Fits 24 GiB only *with* the q4_0 KV cache above;
  at f16 KV, 128K is already out of memory.
- `--no-kv-unified` — llama.cpp's unified-KV mode lets slots share cache pages,
  but hybrid/recurrent layers (DeltaNet fixed state) cannot be re-based across
  requests; with `parallel > 1` and unified KV every new request re-prefills
  the *entire* shared history (measured: 53K-token recompute ≈ 90 s, growing
  linearly with conversation depth). Disabling unified KV restores per-request
  caches: turn-2 prefill drops to a ~27-token increment.
- `--parallel 1` — one slot; pairs with `--no-kv-unified` above.
- `--ctx-checkpoints 2` — keeps 2 rollback snapshots of the target + draft KV
  for partial-acceptance rollback in speculative decoding. Default is 32
  snapshots (≈320 MB each here — 10 GiB!). 2 covers the practical rollback
  depth at n_max=5. Use `0` only if you never partially accept drafts.
- `--cache-ram 4096` — spill prompt-cache entries to host RAM (4 GiB) instead
  of recomputing them on revisit.
- `-b 2048 -ub 512` — logical/physical batch sizes; bounds prefill compute
  buffers (relevant at 97% VRAM occupancy).
- `-fa on` — fused attention kernels.
- `--no-warmup` — skip the startup warm-up pass (avoids transiently allocating
  a second compute buffer on an already-full card).

## Validated on RX 7900 XTX (gfx1100, ROCm 7.14)

- `test-backend-ops`: **12967/12967 pass**, including real-model shapes
  (k=5120, k=96) and MMQ configs J=16..128 added as regression guards.
- Qwen3.8-27B end-to-end: coherent generation, multimodal (mmproj) loading,
  long-conversation stability per the configuration above.

## Build

```bash
cmake -B build-rocmfp4 -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build-rocmfp4 -j
```

## Credits

- Format and reference decode/encode: [ROCmFPX](https://github.com/charlie12345/ROCmFPX)
  and [rocmfp4-llama](https://github.com/charlie12345/rocmfp4-llama) by charlie12345.
- DFlash2 speculative decoding: PR #27342 by SubSir.
- Port integration, split-nibble MMQ tile layout and RDNA3 validation: this fork.

License: MIT (same as upstream llama.cpp).
