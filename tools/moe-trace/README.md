# llama-moe-trace (INF-72 RT-1)

Per-token MoE routing tap. Records the router's top-k expert selection of every MoE layer for every token
through the backend eval callback, **weights untouched**, and writes the `epyc.moe.routing_trace.v1` contract
(manifest + `rows.bin` + `ids.i16`, sha256-bound) plus the EOR / IR_t rider. Experimental branch only
(cut from `production-consolidated-v10` @ `ffc1bac82`); `tools/` change, no library code touched.

Port of `llama-moe-trace` from github.com/Shriniwas410/cacheable-by-design @ `03366ef1` (MIT).
Handoff: `/workspace/handoffs/active/moe-routing-tap-and-locality-measurement.md` (RT-1..RT-3, RT-2.1).
Spec: `/workspace/repos/epyc-inference-research/scripts/kernel_rnd/moe_routing_tap/CPP_CHANGE_SPEC.md`.

## What it emits

| file (per shard dir) | content |
|---|---|
| `routing_trace.json` | manifest, validates against `routing_trace.schema.json` and `trace_shape.validate_manifest` |
| `routing_trace.rows.bin` | 16-byte row per token: `u32 sample_id, u32 pos, u32 step, u16 spec_block, u8 phase (0 prefill / 1 decode), u8 spec_accepted (255 = null)` |
| `routing_trace.ids.i16` | `int16 [rows, n_layers, max_top_k]`, short layers padded with `-1` |
| `routing_trace.eor.json` | EOR / IR_t per layer and phase, with the `K/N` chance baseline (8/256 = 3.13 %, 10/512 = 1.95 %) |
| `routing_trace.provenance.json` | build info, corpus/token sha256, layer `il` map, guard statistics, argv, env |
| `routing_trace.argsort.i16` | only with `MOE_TRACE_SRP=1`: full descending expert order (`ffn_moe_argsort`), `int16 [rows, n_layers, n_experts]` |

A run that fails (hazard, incomplete tap, error) has **no** `routing_trace.json` in its in-flight shard and an
`<MOE_TRACE_OUT>/FAILED` marker. Absence of the manifest means "not a trace".

## The four adjustments (RT-1) and the RT-3 hazard

1. **int16** ids; `n_experts` / `top_k` are read **per layer from the graph** (the top-k view's argsort parent), not from a global.
2. `-b N` + `llama_memory_clear` per window gives `[samples, N, top_k]` by reshape. `-ub` is forced equal to `-b` so one `llama_decode` is one forward.
3. SRP additionally needs `ffn_moe_argsort-<il>` (cb-named at `src/llama-graph.cpp:2060`); SCH needs only the top-k the tap always emits.
4. Any arch whose MoE goes through `build_moe_ffn` is covered: `qwen35moe`, `qwen3next` and `qwen4exp` (the Q38FN fleet model, 512 experts / top-10).
   `qwen4exp` has a **fused single-token decode** that bypasses the graph; the tool sets `GGML_FUSED_DECODE_OFF=1` and refuses any forward
   with a missing layer (exit 2) instead of writing a short trace. No `glm5next` arch exists in this tree.
5. **RT-3 stride hazard** (`ggml/src/ggml.c:5443 ggml_argsort_top_k`, `src/llama-graph.cpp:2059-2062`; the handoff's `ggml.c:5373` anchor has drifted):
   `ffn_moe_topk` is a non-contiguous view of the `[n_expert, n_tokens]` argsort result. It is read **row by row at `i * nb[1]`**. A flat read returns
   whole argsort rows - every expert exactly the same number of times, no adjacent reuse - a plausible-looking negative. Three independent defences:
   the reader (`read_rows_strided`), a **startup selftest** that proves the reader on a synthetic strided tensor and proves the guard catches the flat read
   (exit 4 if either fails), and a **runtime guard** on every prefill forward and layer (`hazard_check`, exit 3). The same guard runs offline in
   `moe_trace_convert.py verify`.

## Build

```bash
cmake -S . -B build-moe-trace -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
cmake --build build-moe-trace --target llama-moe-trace -j4        # builds the libs too: slow on a busy host
cmake --build build-moe-trace --target test-moe-trace-core && ctest --test-dir build-moe-trace -R moe-trace
```

`-DLLAMA_BUILD_MOE_TRACE=OFF` removes the tool from the build. The tool uses only `llama.h` + `llama-common`, so a binary built here runs against the
production store's libs of the SAME commit (as `llama-weight-handoff` does): link the object against `kernels/production/cpu/*.so` and prove it with
`LD_LIBRARY_PATH=<store> verify_ggml_linkage.sh <binary> <store build>`.

## Run

```bash
MOE_TRACE_OUT=/path/out MOE_TRACE_DOMAIN=prose MOE_TRACE_MAX_SAMPLES=512 MOE_TRACE_SHARD_SAMPLES=64 \
GGML_IQK=1 llama-moe-trace -m model.gguf -f corpus.txt -c 1024 -b 512 -t 16 -tb 16 -ngl 0 --no-warmup
MOE_TRACE_SELFTEST_ONLY=1 llama-moe-trace      # no model: reader + guard + sha256 + int16 proof
```

Env knobs are documented at the top of `moe-trace.cpp`. Exit codes: 0 done, 10 stopped (stop file / deadline / SIGTERM, resumable from `MOE_TRACE_START_SAMPLE`),
1 error, 2 tap incomplete, 3 hazard, 4 selftest failed.

## Read and verify

```bash
python3 tools/moe-trace/moe_trace_convert.py verify  <shard>   # hashes, schema, row invariants, offline hazard guard
python3 tools/moe-trace/moe_trace_convert.py summary <shard>
python3 tools/moe-trace/moe_trace_convert.py npz     <shard> --out shard.npz   # (L, T, k) for routing-lab 02/03/04
# in epyc-inference-research: trace_shape.load_run(shard) -> (manifest, rows, ids); trace_shape.validate_rows; derive_counts
```

## Tests (no model)

`tests/test-moe-trace-core.cpp` (ctest `test-moe-trace-core`; also writes a synthetic run through the real writer) and
`tests/test_moe_trace_convert.py` (`python3 -m pytest tools/moe-trace/tests`): stride reader, flat-read detection for 256/512 experts, near-uniform
detection, sub-chance-alone is not fatal, EOR hand values, sha256 vectors, int16 > 127, C++ -> Python contract, tamper detection, and the research
`trace_shape` validator on the C++-written manifest.

## Not done / out of scope

* No speculative-block tap: `spec_block` is the forward's `n_tokens`, `spec_accepted` is always null. DFlash2/MTP verify blocks need their own harness.
* Not run on a model by this change (compute-gated). The first filler run is the first real-graph exercise; the selftest and the per-forward completeness
  checks are what stand between a wrong read and a wrong number.
* Write-side belief wiring (SC66) is RT-2's, not this tool's.
