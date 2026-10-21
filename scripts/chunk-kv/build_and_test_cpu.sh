#!/bin/bash
# Build the chunk-KV P1 tests on CPU and run them against the generated tiny models. CPU only, no GPU, no servers.
# Run it under the region lock, foreground, niced (host rules for build work):
#   /workspace/repos/epyc-orchestrator/scripts/region-lock run --role build --cpu-list 0-95 --timeout-s 7200 -- \
#       nice -n 19 scripts/chunk-kv/build_and_test_cpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${BUILD:-$ROOT/build-cpu}"
JOBS="${JOBS:-48}"

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF >/dev/null

cmake --build "$BUILD" -j "$JOBS" --target llama test-llama-archs test-chunk-kv test-server-chunk-store

# the binary under test must contain the API (a stale build proves nothing)
LIB="$(ls "$BUILD"/bin/libllama.so* 2>/dev/null | head -1)"
[ -n "$LIB" ] || { echo "FAIL: no libllama in $BUILD/bin"; exit 2; }
for sym in llama_chunk_kv_inject llama_chunk_kv_export llama_chunk_kv_rope_selfcheck; do
    nm -D "$LIB" | grep -q "$sym" || { echo "FAIL: $sym missing from $LIB (stale build)"; exit 2; }
done

MODELS="$BUILD/tests/test-models"
mkdir -p "$MODELS"
"$BUILD/bin/test-llama-archs" -o "$MODELS" || echo "note: test-llama-archs exited nonzero (pre-existing deepseek41 generator failure); using whatever models were written"

"$BUILD/bin/test-server-chunk-store"

rc=0
for arch in llama qwen3 qwen3vl qwen3vlmoe qwen35; do
    f="$MODELS/${arch}-dense.gguf"; [ -f "$f" ] || f="$MODELS/${arch}-moe.gguf"
    if [ ! -f "$f" ]; then echo "SKIP $arch: $f not generated"; continue; fi
    echo "=== test-chunk-kv $arch ==="
    SKIPK=""; [ "$arch" = llama ] && SKIPK=1   # random rope_freqs in the generated llama model
    CKV_SKIP_K_GATE="${SKIPK:+1}" LD_LIBRARY_PATH="$BUILD/bin" "$BUILD/bin/test-chunk-kv" -m "$f" -ctk f32 -ctv f32 -t 8 || rc=1
done
exit $rc
