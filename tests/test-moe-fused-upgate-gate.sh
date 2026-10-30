#!/bin/bash
set -euo pipefail
# CH-19 C6 source gate. Run against one rebuilt CPU image; no model weights needed.
if (( $# != 2 )); then
    echo "usage: $0 /absolute/path/test-moe-fused-upgate /absolute/output-dir" >&2
    exit 2
fi
bin=$1
out=$2
mkdir -p "$out"
ref="$out/reference.bin"
GGML_IQK=1 GGML_IQK_Q8_0=1 "$bin" --fused off write "$ref" > "$out/reference.out" 2> "$out/reference.err"
if grep -Fq '[iqk] ACTIVE: fused MoE up+gate+SwiGLU (type=' "$out/reference.err"; then
    echo "Q8_0 fused marker appeared with --fused off" >&2
    exit 1
fi
for cell in 0 1; do
    GGML_IQK=1 GGML_IQK_Q8_0=1 "$bin" --fused on cmp "$ref" "$cell" > "$out/q4-${cell}.out" 2> "$out/q4-${cell}.err"
    if grep -Fq '[iqk] ACTIVE: fused MoE up+gate+SwiGLU (type=' "$out/q4-${cell}.err"; then
        echo "Q8_0 extension marker appeared in Q4_K control $cell" >&2
        exit 1
    fi
done
for cell in 2 3; do
    GGML_IQK=1 GGML_IQK_Q8_0=1 "$bin" --fused off cmp "$ref" "$cell" > "$out/q8-off-${cell}.out" 2> "$out/q8-off-${cell}.err"
    if grep -Fq '[iqk] ACTIVE: fused MoE up+gate+SwiGLU (type=' "$out/q8-off-${cell}.err"; then
        echo "Q8_0 fused marker appeared in off control $cell" >&2
        exit 1
    fi
    GGML_IQK=1 GGML_IQK_Q8_0=1 "$bin" --fused on cmp "$ref" "$cell" > "$out/q8-on-${cell}.out" 2> "$out/q8-on-${cell}.err"
    grep -Fq '[iqk] ACTIVE: fused MoE up+gate+SwiGLU (type=' "$out/q8-on-${cell}.err"
done
GGML_IQK=1 GGML_IQK_Q8_0=0 "$bin" --fused off write "$out/q8-gate-off-reference.bin" > "$out/q8-gate-off-reference.out" 2> "$out/q8-gate-off-reference.err"
GGML_IQK=1 GGML_IQK_Q8_0=0 "$bin" --fused on cmp "$out/q8-gate-off-reference.bin" 2 > "$out/q8-gate-off.out" 2> "$out/q8-gate-off.err"
if grep -Fq '[iqk] ACTIVE: fused MoE up+gate+SwiGLU (type=' "$out/q8-gate-off.err"; then
    echo "Q8_0 fused marker appeared with GGML_IQK_Q8_0=0" >&2
    exit 1
fi

GGML_IQK=1 GGML_IQK_Q8_0=1 "$bin" --workspace-gap > "$out/workspace-gap.out" 2> "$out/workspace-gap.err"
grep -Fq "workspace-gap: raw=" "$out/workspace-gap.err"
grep -Fq "cached OFF->ON->OFF" "$out/workspace-gap.err"
grep -Fq "[iqk] ACTIVE: fused MoE up+gate+SwiGLU (type=" "$out/workspace-gap.err"
printf '%s\n' 'CH19_Q8_GATE PASS'
