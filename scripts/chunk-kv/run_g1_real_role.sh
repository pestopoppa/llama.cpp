#!/bin/bash
# G1 shift exactness on a REAL role (Qwen3-VL-30B-A3B text path, attention-only, IMROPE): the same test-chunk-kv
# binary as the CI gate, pointed at the real GGUF with f32 K/V caches. Needs an inference window (30B model loads
# and decodes ~100 tokens a few times); refuses unless WINDOW_CONFIRMED=1.
#   WINDOW_CONFIRMED=1 BUILD=<build dir> OUT=<dir> scripts/chunk-kv/run_g1_real_role.sh
# Run under:  region-lock run --role bench --cpu-list <quarter> -- ...   (CPU) ; take NO region claim for a GPU run.
set -euo pipefail

[ "${WINDOW_CONFIRMED:-0}" = "1" ] || { echo "refusing: needs an inference window (set WINDOW_CONFIRMED=1)"; exit 3; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${BUILD:-$ROOT/build-cpu}"
MODEL="${MODEL:-/mnt/raid0/llm/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf}"
OUT="${OUT:-$ROOT/results/chunk-kv-g1-$(date -u +%Y%m%dT%H%M%SZ)}"
THREADS="${THREADS:-48}"
EXTRA="${EXTRA:-}"

[ -f "$MODEL" ] || { echo "FAIL: model $MODEL missing"; exit 2; }
BIN="$BUILD/bin/test-chunk-kv"
[ -x "$BIN" ] || { echo "FAIL: $BIN not built"; exit 2; }

# stale-artifact guard: the library must be newer than the last commit touching the chunk code, and contain the API
LIB="$(ls "$BUILD"/bin/libllama.so* | head -1)"
nm -D "$LIB" | grep -q llama_chunk_kv_inject || { echo "FAIL: $LIB lacks llama_chunk_kv_inject (stale build)"; exit 2; }
[ "$(stat -c %Y "$LIB")" -ge "$(git -C "$ROOT" log -1 --format=%ct -- src/llama-kv-cache.cpp src/llama-context.cpp)" ] \
    || { echo "FAIL: $LIB is older than the source commit (stale build)"; exit 2; }

# each tree has its own ggml: prove this binary loads ITS libraries (verify_ggml_linkage.sh lives in the research repo)
export LD_LIBRARY_PATH="$BUILD/bin"
/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh "$BIN" "$BUILD/bin"

mkdir -p "$OUT"
{
  echo "date=$(date -u +%FT%TZ) model=$MODEL commit=$(git -C "$ROOT" rev-parse HEAD) threads=$THREADS"
  "$BIN" -m "$MODEL" -ctk f32 -ctv f32 -t "$THREADS" $EXTRA
} 2>&1 | tee "$OUT/g1.log"
echo "log: $OUT/g1.log"
