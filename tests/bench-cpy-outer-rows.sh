#!/bin/bash
set -euo pipefail

tree=${1:-/mnt/raid0/llm/llama.cpp-experimental-glm53-20260908}
build=${2:-$tree/build-glm53-cpu}
out=${3:-/mnt/raid0/llm/tmp/glm53-validation-20260908/cpy-outer-rows-micro}
region_lock=/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock

mkdir -p "$out"
sha256sum "$build/bin/bench-cpy-outer-rows" "$build/bin/libggml-cpu.so.0.16.0" > "$out/artifact-sha256.txt"

cmd=(
    env -i
    PATH=/usr/bin:/bin
    LD_LIBRARY_PATH="$build/bin"
    OMP_NUM_THREADS=48
    OMP_DYNAMIC=false
    OMP_PLACES=cores
    OMP_PROC_BIND=spread
    OMP_WAIT_POLICY=active
    taskset -c 0-95
    numactl --interleave=all
    "$build/bin/bench-cpy-outer-rows"
)
printf '%q ' "${cmd[@]}" > "$out/bench-argv-env.txt"
printf '\n' >> "$out/bench-argv-env.txt"

"$region_lock" run --cpu-list 0-95 --role benchmark --timeout-s 0 \
    --tag bench:glm53-cpy-outer-rows -- "${cmd[@]}" \
    > "$out/bench.stdout" 2> "$out/bench.stderr"
