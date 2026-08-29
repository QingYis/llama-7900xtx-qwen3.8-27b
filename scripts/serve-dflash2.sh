#!/usr/bin/env bash
# ============================================================================
# Qwen3.8-27B ROCmFP4 + DFlash2 + ngram-map-k4v reference serving script
# (RX 7900 XTX / gfx1100 tested; see docs/rocmfp4.md for the rationale of
#  every flag below)
#
# Usage:
#   bash scripts/serve-dflash2.sh                 # defaults below
#   MODEL=... DRAFT=... PORT=8080 CTX=131072 bash scripts/serve-dflash2.sh
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${BIN_DIR:-$REPO/build-rocmfp4/bin}"          # built with -DGGML_HIP=ON
MODEL="${MODEL:-$REPO/../models/Qwen3.8-27B-heretic-ara.ROCmFP4-STRIX.gguf}"
DRAFT="${DRAFT:-$REPO/../models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}"
MMPROJ="${MMPROJ:-$REPO/../models/mmproj-Qwen3.8-27B-Q8_0.gguf}"   # MMPROJ="" to disable vision

export PATH="$HOME/.local/bin:/opt/rocm/bin:$PATH"
# Required whenever the KV cache is quantized on HIP: skips the rotated-KV
# re-encoding path that this hybrid (DeltaNet + full-attention) architecture
# does not implement. Omit it ONLY for f16 KV caches.
export LLAMA_ATTN_ROT_DISABLE=1

SPEC_FLAGS=()
if [ -n "${DRAFT:-}" ] && [ -f "$DRAFT" ]; then
    # Two stacked speculators, evaluated in order (ngram first, free hits win):
    #  - ngram-map-k4v: exact n-gram lookup over the prompt/history, drafts up
    #    to M=48 tokens with zero draft-model cost. Dominant on repetitive
    #    content (2.4x baseline); silently idle otherwise.
    #  - draft-dflash: block-diffusion draft model (block=8, selector top-k
    #    lattice). n_max=5 / p_min=0.4 is the measured throughput sweet spot:
    #    lower p_min drafts into low-confidence regions and verification cost
    #    outgrows the win, higher n_max wastes the tail positions.
    SPEC_FLAGS=(
        --spec-type ngram-map-k4v,draft-dflash
        --spec-draft-n-max 5
        --spec-draft-p-min 0.4
        --spec-ngram-map-k4v-size-n 12
        --spec-ngram-map-k4v-size-m 48
        --spec-ngram-map-k4v-min-hits 1
        -ngld 99
    )
fi

exec "$BIN_DIR/llama-server" \
    -m "$MODEL" \
    ${MMPROJ:+--mmproj "$MMPROJ"} \
    "${SPEC_FLAGS[@]}" \
    -ctk q4_0 -ctv q4_0 \
    -ngl 99 \
    -c "${CTX:-262144}" \
    -b 2048 -ub 512 \
    -fa on --jinja \
    --no-kv-unified --parallel 1 \
    --ctx-checkpoints 2 --cache-ram 4096 \
    --no-warmup \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-1234}" \
    "$@"
