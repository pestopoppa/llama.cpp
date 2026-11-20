#!/bin/bash
set -euo pipefail
# AK-PROF-SCLK-ENERGY: MMQ tile-selection environment knobs (HIP/CUDA build, one GPU, no model weights).
# Every knob must (1) leave results correct vs the CPU backend and (2) actually reach the kernel (route log / marker).
# usage: test-mmq-tile-env.sh /abs/path/test-backend-ops /abs/output-dir [backend, default ROCm0]
if (( $# < 2 )); then
    echo "usage: $0 /absolute/path/test-backend-ops /absolute/output-dir [backend]" >&2
    exit 2
fi
bin=$1
out=$2
be=${3:-ROCm0}
mkdir -p "$out"

run() { # name, env...
    local name=$1; shift
    if ! env "$@" GGML_CUDA_LOG_MMVQ_ROUTE=1 "$bin" test -b "$be" -o MUL_MAT -p type_a=q8_0 > "$out/$name.out" 2> "$out/$name.err"; then
        echo "FAIL $name: test-backend-ops rc != 0 (see $out/$name.out)" >&2
        exit 1
    fi
    if ! grep -Eq '[0-9]+/[0-9]+ tests passed' "$out/$name.out" || grep -q 'FAIL' "$out/$name.out"; then
        echo "FAIL $name: correctness" >&2
        exit 1
    fi
}

big_mmq() { grep -E 'route=MMQ .*ne11=(2[0-9]{2}|[3-9][0-9]{2}|[0-9]{4,})$' "$out/$1.err" | head -1; }

# Knob 1: GGML_CUDA_MMQ_FORCE=1 must route a large-batch Q8_0 MUL_MAT to MMQ; unset must keep the default route.
run default
run force GGML_CUDA_MMQ_FORCE=1
if [[ -z "$(big_mmq force)" ]]; then
    echo "FAIL force: no MMQ route at ne11 >= 200 with GGML_CUDA_MMQ_FORCE=1" >&2
    exit 1
fi
echo "PASS GGML_CUDA_MMQ_FORCE"
