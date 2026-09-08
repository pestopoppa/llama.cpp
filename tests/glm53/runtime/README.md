# GLM-5.3 text/native-MTP validation runtime

This directory contains one implemented candidate-side correctness producer and
three offline/server evidence utilities. It does not contain a completed k-pool
producer. No build or inference has been run from this directory.

The current host has about 26 days of uptime, so timings collected now are
observation-grade. A trusted speed claim still requires the P-BENCH-4 host reset
and contention gates. This Codex session has no roster identity. Do not execute
the commands below until the operator grants the task-scoped physical-lock
lease recorded by the owning session.

## Implemented target rollback gate

`../../test-glm5next-mtp-rollback.cpp` is llama.cpp test source (not yet compiled)
using public and common APIs. For every forced accepted length `k=0..draft_max`, it:

1. decodes a full synthetic target verification suffix;
2. removes the rejected suffix with `llama_memory_seq_rm`;
3. replays the same accepted prefix in a fresh target context;
4. fully restores that replay checkpoint into a target context that already
   contains recurrent state; and
5. compares argmax tokens and finite logits for several continuation steps.

It checks observable target-context equivalence after rollback. It does not
claim that the synthetic suffix came from the NextN head, does not compare raw
allocator state, and does not exercise restore into a used MTP context.

Candidate integration is limited to copying the source into `tests/` and adding
the executable to `tests/CMakeLists.txt` with the same common/llama dependencies
as `test-recurrent-state-rollback`. The candidate owner remains the sole writer.

Configuration and compilation both acquire the physical build lock because
CMake performs compiler probes. The initial CPU build must disable HIP:

```bash
RL=/workspace/repos/epyc-orchestrator/scripts/region-lock
SRC=/path/to/authorized/glm53-candidate
BUILD="$SRC/build-glm53-cpu"

"$RL" run --cpu-list 0-95 --role build --timeout-s 0 \
  --tag glm53-validation-configure -- \
  cmake -S "$SRC" -B "$BUILD" -DGGML_HIP=OFF -DLLAMA_BUILD_TESTS=ON

INF70_AGENT=glm53-validation \
  /mnt/raid0/llm/tmp/inf70/build_locked.sh "$BUILD" -j 40 \
  --target test-glm5next-mtp-rollback llama-server
```

Run the target test on the CPU backend under the granted region lock:

```bash
mkdir -p ./runs
GLM53_TEST_DRAFT_MAX=5 \
GLM53_TEST_CYCLES=2 \
GLM53_TEST_CONTINUATION=4 \
GLM53_TEST_LOGIT_TOLERANCE=1e-5 \
GLM53_TEST_JSONL="$PWD/runs/rollback.jsonl" \
  "$RL" run --cpu-list 0-95 --role bench --timeout-s 0 \
  --tag glm53-target-rollback -- \
  taskset -c 0-95 numactl --interleave=all \
  "$BUILD/bin/test-glm5next-mtp-rollback" -m /path/to/model.gguf \
  -ngl 0 --device none -c 128 -b 32 -ub 32

./validate_glm53.py \
  --rollback ./runs/rollback.jsonl --draft-max 5 --logit-epsilon 1e-5 \
  --output ./runs/rollback-report.json
```

The tolerance is declared before execution. Do not change it after inspecting a
failure. The test refuses a non-GLM5Next architecture, a non-recurrent model,
insufficient context/batch capacity, missing cases, nonfinite logits, token
differences, or a logit difference above the declared tolerance.

## Native-MTP server gate

Use the checked serve-plan generator in `../serve/` rather than copying
the stale binary constants from the INF-70 draft recipe. It preserves the
champion-derived CPU recipe: taskset 0-95, NUMA interleave, 48 threads, one slot,
8192 context, no mmap, FA on, f16 K/V, ngl0, and the exact IQK/fused-decode/
FA-split/hugepage environment. Supply the candidate binary explicitly.

```bash
python3 ../serve/generate_serve_plan.py \
  --binary "$BUILD/bin/llama-server" --model /path/to/model.gguf \
  --draft-max 5 --output ./runs/mtp-plan.json
```

Start the emitted command only inside the granted physical lock, preserve the
server log and captured PID, and verify candidate binary/library identity before
the run. Use only that captured PID for shutdown. `probe_server.py` writes every
exact request payload, canonical response, server counters, and five measured
512-token requests after warmup:

```bash
./probe_server.py --port 18497 --out ./runs/mtp
```

Run a second server arm with the identical plan except native speculation is
disabled, then compare the matched greedy responses:

```bash
./probe_server.py --port 18497 --out ./runs/plain
./compare_server_arms.py --plain ./runs/plain --mtp ./runs/mtp \
  > ./runs/matched-greedy.json
```

The comparator requires five pairs, exact saved payload equality, exactly 512
integer token IDs per response, and exact token equality. This end-to-end gate
must pass alongside evidence of actual `draft-mtp` graph dispatch. Preserve the
raw server log and parse its per-verification `accepted A/D draft tokens` lines:
at least one completed event must have `A > 0`, and at least one must have
`0 <= A < D`. Aggregate `draft_n - draft_n_accepted` is only an unaccepted count
because terminal drafts may never be verified; it is not rejection evidence.
Forced target suffix cases are not native draft acceptance evidence.

`validate_glm53.py --mtp-arm ... --speed-shape` validates a prospective arm
sidecar using metric `accepted_output_tokens_per_second`, direction
`higher_better`, protocol `P-BENCH-4`, `n=5`, candidate/model/binary/recipe
identity, host caveats, contention evidence, and in-window sample references.
The write-side evidence hook must create that sidecar; do not invent counters or
a grading rule in this harness. The existing ClaimTuple ladder owns grading.

## Mandatory gate still awaiting a producer

K-pool threshold and chunked-prefill execution remains mandatory before a
support claim. `validate_glm53.py --pool` validates records for either the tiny
fixture contract (`kpool=4, topk=8`) or the real model (`kpool=4, topk=2048`). It
requires one full and one chunked case at all six boundary lengths, exact member
and incomplete-tail coverage, aligned complete groups of four, equal next token,
and a predeclared numeric logit tolerance. No executable producer for these
records is present here, so this gate is not complete.

The tiny files in `fixtures/` are schema-validator self-tests only. They are not
model evidence.

## Offline self-test

These commands perform no compilation or inference:

```bash
python3 -m py_compile validate_glm53.py probe_server.py compare_server_arms.py make_test_fixtures.py
./make_test_fixtures.py
./validate_glm53.py --rollback fixtures/rollback-pass.jsonl \
  --draft-max 5 --logit-epsilon 0 --output /tmp/glm53-rollback-validator.json
./validate_glm53.py --pool fixtures/pool-pass.jsonl \
  --kpool 4 --topk 2048 --pool-logit-epsilon 0 \
  --output /tmp/glm53-pool-validator.json
```

Calling `validate_glm53.py` without any evidence input must refuse a vacuous
PASS.
