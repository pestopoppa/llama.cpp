#!/bin/bash
# Tokenize the same real-source prospective alternate for one additional role.
# Uses vocabulary only; no model weights/context/inference. Invoke under CPU/RAM claim.
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo "usage: $0 ROLE MODEL_GGUF HEADER_BYTES NEW_OUTPUT_ROOT" >&2
    exit 2
fi
role=$1
model=$2
header=$3
out=$4
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=/mnt/raid0/llm/tmp/kv-prefix-fork
src=/mnt/raid0/llm/data/eval
lib=/mnt/micron/rtg58-kpf623-g3-20261102/bin
vocab=${KPF62_VOCAB_HELPER:-$root/kpf62-musique-20261102/g3-vocab-tokenize}
selection=$root/kpf62-musique-20261102/selection.json
mkdir -m 700 "$out"
python3 "$here/g3_musique_tokenize.py" --source "$src/zeroscrolls/musique/musique/test.jsonl" \
    --selection "$selection" --model "$model" --model-header-bytes "$header" \
    --tokenizer "$vocab" --lib-dir "$lib" --out "$out/musique" > "$out/musique.log"
python3 "$here/g3_longqa_acquire.py" \
    --source "squality=$src/zeroscrolls/squality/squality/test.jsonl" \
    --source "narrativeqa=$src/zeroscrolls/narrative_qa/narrative_qa/test.jsonl" \
    --out "$out/longqa" --tokenizer "$vocab" --model "$model" --lib-dir "$lib" \
    --header-bytes "$header" --music "$out/musique" > "$out/longqa.log"
python3 "$here/g3_longqa_acquire.py" \
    --leval "leval-legal=$src/leval/LEval/Generation/legal_contract_qa.jsonl" \
    --leval "leval-natural=$src/leval/LEval/Generation/natural_question.jsonl" \
    --leval "leval-narrative=$src/leval/LEval/Generation/narrative_qa.jsonl" \
    --leval "leval-longchat=$src/leval/LEval/Exam/topic_retrieval_longchat.jsonl" \
    --leval "leval-meeting=$src/leval/LEval/Generation/meeting_summ.jsonl" \
    --out "$out/final" --tokenizer "$vocab" --model "$model" --lib-dir "$lib" \
    --header-bytes "$header" --music "$out/longqa" > "$out/final.log"
python3 "$here/g3_manifest.py" --corpus "$out/final/corpus.jsonl" --tasks "$out/final/tasks.jsonl" \
    --role "$role" --task-source 'prospective ZeroSCROLLS and L-Eval alternate; not B7' \
    --out "$out/final/manifest.json"
