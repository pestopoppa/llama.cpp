#!/bin/bash
# KPF-62.4/.5/.6 build + unit tests on CPU (tiny generated models only; no real model, no server).
# Builds libllama, llama-server (so the server wiring compiles) and the chunk-KV tests, then runs:
#   test-server-chunk-store, test-server-reorder              (header-only units)
#   test-chunk-kv            llama qwen3 qwen3vl qwen3vlmoe qwen35   (slice 1 regression, G1a)
#   test-chunk-kv-recompute  llama qwen3 qwen3vl qwen3vlmoe qwen35   (KPF-62.4 G0: G0a, G0a-al, G0a-inj, head; qwen35 = refusal)
#                            + CKV_N_CHUNK=200 and the Q4_K_M fixtures (qwen3, qwen3vlmoe) at CKV_N_CHUNK=512
#   test-chunk-kv-g3 --g3-self-test  qwen3 qwen3vlmoe                (KPF-62.6 harness sanity: all-inject bit-identical, store hits)
#   g3_gate.py selftest                                              (gate statistics)
# Run it under a build-role region claim, foreground, niced; the filler job kpf62_build_job.sh does exactly that.
#   BUILD=<dir> (default $ROOT/build-cpu-kpf62, resumable)  JOBS=<n> (default 8)  OUT=<dir for logs>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${BUILD:-$ROOT/build-cpu-kpf62}"
JOBS="${JOBS:-8}"
OUT="${OUT:-$BUILD/kpf62-test-logs}"
PY="${PY:-/usr/bin/python3}"
mkdir -p "$OUT"

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_NATIVE=ON -DGGML_HIP=OFF -DLLAMA_CURL=OFF >"$OUT/configure.log" 2>&1

cmake --build "$BUILD" -j "$JOBS" --target llama llama-server llama-quantize test-llama-archs \
      test-chunk-kv test-chunk-kv-recompute test-chunk-kv-g3 test-server-chunk-store test-server-reorder >"$OUT/build.log" 2>&1

# the binary under test must contain the API (a stale build proves nothing); no `grep -q` under pipefail
LIB="$(ls "$BUILD"/bin/libllama.so* 2>/dev/null | head -1)"
[ -n "$LIB" ] || { echo "FAIL: no libllama in $BUILD/bin"; exit 2; }
for sym in llama_chunk_kv_inject llama_chunk_kv_export llama_chunk_kv_rope_selfcheck llama_chunk_kv_decode_at; do
    nm -D "$LIB" | grep "$sym" >/dev/null || { echo "FAIL: $sym missing from $LIB (stale build)"; exit 2; }
done
strings "$BUILD/bin/llama-server" | grep -- "--chunk-store-mib" >/dev/null || { echo "FAIL: llama-server lacks --chunk-store-mib (stale build)"; exit 2; }

export LD_LIBRARY_PATH="$BUILD/bin"
MODELS="$BUILD/tests/test-models"
mkdir -p "$MODELS"
"$BUILD/bin/test-llama-archs" -o "$MODELS" >"$OUT/test-llama-archs.log" 2>&1 \
    || echo "note: test-llama-archs exited nonzero (pre-existing deepseek41 generator failure); using whatever models were written"

rc=0
SUMMARY="$OUT/SUMMARY.txt"
: >"$SUMMARY"
run() { # NAME CMD...
    local name=$1; shift
    if "$@" >"$OUT/$name.log" 2>&1; then
        echo "PASS $name" | tee -a "$SUMMARY"
    else
        echo "FAIL $name (log $OUT/$name.log)" | tee -a "$SUMMARY"; rc=1
    fi
}
model_of() { local f="$MODELS/$1-dense.gguf"; [ -f "$f" ] || f="$MODELS/$1-moe.gguf"; echo "$f"; }

run test-server-chunk-store "$BUILD/bin/test-server-chunk-store"
run test-server-reorder     "$BUILD/bin/test-server-reorder"

for arch in llama qwen3 qwen3vl qwen3vlmoe qwen35; do
    f="$(model_of $arch)"
    [ -f "$f" ] || { echo "SKIP $arch: not generated" | tee -a "$SUMMARY"; continue; }
    SKIPK=""; [ "$arch" = llama ] && SKIPK=1   # random rope_freqs in the generated llama model (test-chunk-kv.cpp)
    run "test-chunk-kv-$arch" env CKV_SKIP_K_GATE="${SKIPK}" "$BUILD/bin/test-chunk-kv" -m "$f" -ctk f32 -ctv f32 -t 8
    run "test-chunk-kv-recompute-$arch" "$BUILD/bin/test-chunk-kv-recompute" -m "$f" -ctk f32 -ctv f32 -t 8
    [ "$arch" = qwen35 ] || run "test-chunk-kv-recompute-$arch-n200" env CKV_N_CHUNK=200 "$BUILD/bin/test-chunk-kv-recompute" -m "$f" -ctk f32 -ctv f32 -t 8
done

for arch in qwen3 qwen3vlmoe; do
    f="$(model_of $arch)"
    [ -f "$f" ] || { echo "SKIP $arch q4_k_m: not generated" | tee -a "$SUMMARY"; continue; }
    q="$MODELS/${arch}-q4_k_m.gguf"
    if [ ! -f "$q" ]; then
        "$BUILD/bin/llama-quantize" "$f" "$q" Q4_K_M 8 >"$OUT/quantize-$arch.log" 2>&1 || { echo "FAIL quantize $arch" | tee -a "$SUMMARY"; rc=1; continue; }
    fi
    run "test-chunk-kv-recompute-$arch-q4km-n512" env CKV_N_CHUNK=512 CKV_N_BATCH=128 "$BUILD/bin/test-chunk-kv-recompute" -m "$q" -ctk f32 -ctv f32 -t 8
    run "test-chunk-kv-g3-selftest-$arch" "$BUILD/bin/test-chunk-kv-g3" -m "$(model_of $arch)" -ctk f32 -ctv f32 -t 8 --g3-self-test
done

run g3_gate-selftest "$PY" -I "$ROOT/scripts/chunk-kv/g3_gate.py" selftest

echo "summary: $SUMMARY"
exit $rc
