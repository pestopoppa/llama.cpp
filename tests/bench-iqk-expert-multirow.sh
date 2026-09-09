#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: $0 /path/to/bench-iqk-expert-multirow output.jsonl [repeats=200] [threads=48]" >&2
    exit 2
fi

bench_bin=$1
output=$2
repeats=${3:-200}
threads=${4:-48}

: > "$output"
for round in 1 2 3 4 5; do
    for type in q4 q5; do
        for rows in 1 2 3 4; do
            if (( round % 2 == 1 )); then
                arms=(serial multirow)
            else
                arms=(multirow serial)
            fi
            for arm in "${arms[@]}"; do
                if [[ $arm == multirow ]]; then
                    enabled=1
                else
                    enabled=0
                fi
                GLM53_EXPERT_BENCH_ROUND=$round GGML_IQK_EXPERT_MULTIROW=$enabled \
                    "$bench_bin" "$type" "$rows" "$repeats" "$threads" >> "$output"
            done
        done
    done
done
