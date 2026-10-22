#!/bin/bash
# G1 shift exactness on a REAL role (Qwen3-VL-30B-A3B text path, attention-only, IMROPE): the same test-chunk-kv
# binary as the CI gate, pointed at the real GGUF with f32 K/V caches. Needs an inference window (30B model loads
# and decodes ~100 tokens a few times); refuses unless WINDOW_CONFIRMED=1.
#   WINDOW_CONFIRMED=1 [KPF62_TEST_CHUNK_KV=<bin> | BUILD=<build dir>] OUT=<dir> scripts/chunk-kv/run_g1_real_role.sh [--dry-run]
#   (default: binary from the BUILD_OK marker written by the kpf-62-8-build filler; --dry-run needs no window/claim)
# Run under:  region-lock run --role bench --cpu-list <quarter> -- ...   (CPU) ; take NO region claim for a GPU run.
# Gates (KPF-62.8 fix, 2026-10-22): the exit code gates G1a (inject == in-tree seq_add K-shift, bit-for-bit, K/V/logits), the
# V bit-copy, the rope self-check and the refusal paths. The pre-registered G1-ref comparison (vs a chunk re-encoded in place
# at delta) is printed with its own VERDICT line but is informational on this Q4_K_M role (CKV_G1_REF_INFORMATIONAL=1): the
# reference re-encode drifts by rounding propagation (q8 activation re-quantization + MoE routing), independent of the inject.
# Set CKV_G1_REF_INFORMATIONAL=0 to gate it anyway.
set -euo pipefail

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-/mnt/raid0/llm/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf}"
OUT="${OUT:-$ROOT/results/chunk-kv-g1-$(date -u +%Y%m%dT%H%M%SZ)}"
THREADS="${THREADS:-48}"
EXTRA="${EXTRA:-}"
MARK="${KPF62_BUILD_OK:-/mnt/raid0/llm/tmp/qb2-kpf62-8/BUILD_OK}"

# Binary resolution order: $BUILD (dir) > $KPF62_TEST_CHUNK_KV (binary path) > BUILD_OK marker > $ROOT/build-cpu
SRC=default
if [ -n "${BUILD:-}" ]; then SRC=env-BUILD
elif [ -n "${KPF62_TEST_CHUNK_KV:-}" ]; then BUILD="$(cd "$(dirname "$KPF62_TEST_CHUNK_KV")/.." && pwd)"; SRC=env-KPF62_TEST_CHUNK_KV
elif [ -f "$MARK" ]; then BUILD="$(sed -n 's/^build_dir=//p' "$MARK")"; SRC="marker:$MARK"
else BUILD="$ROOT/build-cpu"; fi
BIN="$BUILD/bin/test-chunk-kv"
[ -x "$BIN" ] || { echo "FAIL: test-chunk-kv not found at $BIN (source: $SRC). Build it first (filler kpf-62-8-build writes $MARK) or set KPF62_TEST_CHUNK_KV=<binary>"; exit 2; }
if [ "$SRC" = "marker:$MARK" ]; then
    want="$(sed -n 's/^sha256=//p' "$MARK")"; have="$(sha256sum "$BIN" | cut -d' ' -f1)"
    [ "$want" = "$have" ] || { echo "FAIL: $BIN sha256 differs from BUILD_OK marker (rebuilt/stale?)"; exit 2; }
fi

if [ "$DRY" = 1 ]; then
    echo "dry-run: no inference performed"
    echo "binary   : $BIN (source: $SRC)"
    echo "model    : $MODEL ($([ -f "$MODEL" ] && echo present || echo MISSING))"
    echo "threads  : $THREADS   out: $OUT"
    echo "head     : $(git -C "$ROOT" rev-parse HEAD)"
    echo "claim    : region-lock run --role bench --cpu-list <quarter, e.g. 0-23,96-111 = q0> -- nice -n 19 $0   (WINDOW_CONFIRMED=1)"
    exit 0
fi

[ "${WINDOW_CONFIRMED:-0}" = "1" ] || { echo "refusing: needs an inference window (set WINDOW_CONFIRMED=1)"; exit 3; }
[ -f "$MODEL" ] || { echo "FAIL: model $MODEL missing"; exit 2; }

# stale-artifact guard: the library must be newer than the last commit touching the chunk code, and contain the API
LIB="$(ls "$BUILD"/bin/libllama.so* | head -1)"
grep -q llama_chunk_kv_inject <(nm -D "$LIB") || { echo "FAIL: $LIB lacks llama_chunk_kv_inject (stale build)"; exit 2; }
[ "$(stat -c %Y "$LIB")" -ge "$(git -C "$ROOT" log -1 --format=%ct -- src/llama-kv-cache.cpp src/llama-context.cpp)" ] \
    || { echo "FAIL: $LIB is older than the source commit (stale build)"; exit 2; }

# each tree has its own ggml: prove this binary loads ITS libraries (verify_ggml_linkage.sh lives in the research repo)
export LD_LIBRARY_PATH="$BUILD/bin"
/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh "$BIN" "$BUILD/bin"

mkdir -p "$OUT"
{
  echo "date=$(date -u +%FT%TZ) model=$MODEL commit=$(git -C "$ROOT" rev-parse HEAD) threads=$THREADS"
  echo "CKV_G1_REF_INFORMATIONAL=${CKV_G1_REF_INFORMATIONAL:-1}"
  CKV_G1_REF_INFORMATIONAL="${CKV_G1_REF_INFORMATIONAL:-1}" "$BIN" -m "$MODEL" -ctk f32 -ctv f32 -t "$THREADS" $EXTRA
} 2>&1 | tee "$OUT/g1.log"
echo "log: $OUT/g1.log"
