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

## Validated on RX 7900 XTX (gfx1100, ROCm 7.14)

- `test-backend-ops`: **12967/12967 pass**, including real-model shapes
  (k=5120, k=96) and MMQ configs J=16..128 added as regression guards.
- Qwen3.8-27B end-to-end: coherent generation; baseline 41.1 tok/s (fp4 KV
  cache q4_0/q4_0, `LLAMA_ATTN_ROT_DISABLE=1`); with DFlash2 + ngram-map-k4v
  speculative decoding 68-98 tok/s depending on content predictability.

## Build

```bash
cmake -B build-rocmfp4 -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build-rocmfp4 -j
```

## Usage

```bash
# serve a ROCmFP4 gguf with DFlash2 speculative decoding
llama-server -m model.ROCmFP4-STRIX.gguf \
  -md model-DFlash2-draft.gguf \
  --spec-type ngram-map-k4v,draft-dflash \
  --spec-draft-n-max 5 --spec-draft-p-min 0.4 \
  -ctk q4_0 -ctv q4_0 \
  -ngl 99 -ngld 99 -c 262144 -fa on
```

Note: when using a quantized KV cache on HIP, set `LLAMA_ATTN_ROT_DISABLE=1`.

## Credits

- Format and reference decode/encode: [ROCmFPX](https://github.com/charlie12345/ROCmFPX)
  and [rocmfp4-llama](https://github.com/charlie12345/rocmfp4-llama) by charlie12345.
- DFlash2 speculative decoding: PR #27342 by SubSir.
- Port integration, split-nibble MMQ tile layout and RDNA3 validation: this fork.

License: MIT (same as upstream llama.cpp).
