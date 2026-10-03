# EXL3-6 CPU — status and resume notes

Branch `experimental/exl3-cpu-20261003` in worktree `/mnt/raid0/llm/worktrees/llama-exl3-cpu-20261003`,
forked from the global champion `90c12df42` (`ak/champion/llama-cpp-ffc1bac82eec`; re-fetched before
every commit, still the tip). Design: `EXL3-CPU-DESIGN.md`. Plan of record: epyc-root
`handoffs/active/exl3-cpu-mi210-implementation.md` (EXL3-6). Nothing pushed.

## Done

| Phase | What | Evidence |
|---|---|---|
| 1 design | `EXL3-CPU-DESIGN.md` | — |
| 2 importer | `tools/exl3/exl3_import.py` (+ `exl3np.py`): EXL3 safetensors + UD GGUF → 2-split GGUF; modes `native` (EXL3_K band8 + `.exl3_suh/.exl3_svh`) and `materialize` (F16, canonical stage roundings); GDN V-head reorder as 128-block perms; `--check` correlates every EXL3 tensor against the UD tensor | `tools/exl3/test_exl3np.py`: decode + materialized-weight **hashes bit-exact** vs EXL3-1 on all 10 mul1 fixtures; `--check` layers 0-3: block corr ≥ 0.979 (experts K3) / ≥ 0.9988 (dense K5); band8 round trip exact |
| 3 ggml + CPU | `GGML_TYPE_EXL3_1..8` (ids 60..67 = buun), `ggml-cpu/exl3.cpp` executor on MUL_MAT / MUL_MAT_ID with aux srcs; int8 GEMV (exllamav3 VBMI/VNNI extraction, per-128-block act scales, exact affine mul1 constant), fp32 GEMM prefill (canonical fp16 codebook); GEMV split-K for < 32-unit ops; `test-backend-ops` cases for a future HIP executor | `build/bin/test-exl3`: ALL PASS — scalar ≡ VNNI ≡ VBMI bit-exact and multi-thread ≡ 1-thread bit-exact (GEMV + GEMM, dense + experts, rows 1-13, K1-8, split-K on/off); vs an independent canonical reference GEMV rel-L2 ≤ 1.0e-2, GEMM ≤ 5e-4; real fixtures real-mul1-k3/k4 GEMV 7.0e-3, GEMM 4.2-4.7e-4. Sealed as 8 `epyc.exl3.verifier.v1` rows (all pass): `/mnt/raid0/llm/tmp/exl3-cpu-20261003/evidence/verifier-20261003T175404Z/` (commit eb5809ba0; re-emit after the last code commit with `tools/exl3/emit_verifier.py`) |
| 4 integration | loader creates aux next to each EXL3 weight (`exl3.layout` checked, skipped tensors counted), `llama_model::exl3_aux` → `build_lora_mm{,_id}` attach; draft heads borrowing an EXL3 target tensor carry its aux; INF-64 fused decode bypassed for EXL3 models | 4-layer truncation (GDN, QSA attention, PLE, K3 experts, K5 dense/shared, **EXL3 K5 `lm_head`**), native vs materialized F16, `llama-perplexity --kl-divergence` (wiki.test.raw, c128): **KLD mean 9.6e-5**, median 1.7e-5, max 0.004, same-top 97.0 % (ub 128); decode path (ub 1): **KLD mean 1.63e-4**, max 0.0049. Earlier UD-head run: 1.47e-4 (ub 128) / 2.46e-4 (ub 1). `llama-completion` generation path works on the truncation; **MTP draft smoke**: `llama-server -md mtp-…-shared-Q8_0.gguf --spec-type draft-mtp` on the 4-layer EXL3 trunk loads the draft against the target (borrowing the EXL3 `output` with its aux), drafts 9 tokens (0 accepted — truncated trunk) and returns the same greedy text as plain decoding. Uncontended decode of the 4-layer model on 8 cores: 13 ms/token |
| 5 (prep) full artifact | `/mnt/raid0/llm/models/exl3-gguf/Qwen3.8-Flash-Next-EXL3-3.05bpw/model-0000{1,2}-of-00002.gguf` — 49.7 GB + shared PLE split; 2113 tensors: 445 EXL3 (144 `exl3_3` experts/indexer-free, 301 `exl3_5` dense + head) with all 890 aux present, `exl3.layout = band8-v1` | importer `--check` over all 48 layers + head: 589 block-correlation checks, worst 0.9787 (layer-0 expert vs UD IQ3_S), head 0.9985 vs UD Q6_K. **Not loaded in llama.cpp yet — needs a window** |

Single-core cache-resident diagnostic (`test-exl3 --bench 1`, not a measurement; cores 176-183 are
SMT siblings of AK-loop cores, so absolute numbers swing up to ~2x): K3 9-10.4 GB/s, K4 17-19 GB/s,
K5 12-14 GB/s of weights per core (exllamav3's own kernel: 9.4 / 17.3 GB/s in INF-71 X1); int8
GEMV vs fp32 GEMM cross over at 8 rows (default `gemv_max_rows = 8`).

Commits (oldest first): `c02961978` design/importer/types/executor/test · `6bbc96c6f` llama
wiring · `36b169cb1` split-K + bench · `eb5809ba0` verifier emitter · `5219ecd3f` doc/no-alloc ·
`01ca2d7e7` crossover bench · `2557de12e` test-backend-ops cases · `cfba227f0` draft-borrow aux ·
`40f309091` split-K threshold. (+ the commit carrying this file.)

Artifacts (not in git):
* shared PLE split: `/mnt/raid0/llm/models/exl3-gguf/qwen4exp-ple-table-ud-iq4nl.split2of2.gguf` (28.8 GB, UD IQ4_NL n-gram table) — every EXL3 artifact's `model-00002-of-00002.gguf` is a symlink to it
* 4-layer native model `/mnt/raid0/llm/tmp/exl3-cpu-20261003/gguf/native-L4/` (EXL3 head) and its reference logits
  `/mnt/raid0/llm/tmp/exl3-cpu-20261003/kld/mat-L4-head.c128.kld` (from the materialized F16 pair, which was deleted to
  free 22 GB — recreate with `exl3_import.py --mode materialize --layers 4 --out …/gguf/mat-L4` if the reference must change)
* full native GGUF `/mnt/raid0/llm/models/exl3-gguf/Qwen3.8-Flash-Next-EXL3-3.05bpw/` (built by `…/make_after_shard7.sh`, log `…/logs/make_after_shard7.log`; its `exl3.source.shards` KV also lists the ngram/mtp files present at import time — fixed in the importer since)
* logs `…/logs/`, KLD base `…/kld/mat-L4.c128.kld`

## Next (in order)

1. **Full model — needs a main-session window** (49.7 GB weights + 29 GB PLE split):
   load + greedy generation smoke, then speed and quality windows. Use the codified recipe
   (`epyc-inference-research/scripts/lib/qwen38_flash_next_recipe.py`, `build_serve_command(bindir=<worktree>/build/bin, model=<full>/model-00001-of-00002.gguf)`;
   it already sets `GGML_FUSED_DECODE_OFF=1`, so the UD anchors and EXL3 both run the graph path).
   Note: there is no high-precision Q38FN GGUF on disk (only UD-IQ4_XS, IQ4_XS-uniform, MTP) — a
   KLD comparison needs a base (Q8_0/BF16 logits) or falls back to PPL / relative KLD; disk is 96 %.
2. Performance (before the speed window, or driven by its profile): (a) each EXL3 op pays one
   internal barrier (prep), split-K ops two — consider folding prep into the producer op or
   per-thread redundant prep for tiny-k ops; (b) merged gate|up with per-segment `suh` (one prep,
   one op); (c) VNNI tier target attributes for portable/multi-variant builds (the file assumes
   `-march=native` on Zen 5 today); (d) prefill: per-unit band decode dominates for 9-30-row MoE
   groups; (e) the EXL3 indexer `index_qk_proj` (K3, fused q|k → split by view; ~1 % bytes).
3. Re-run `python3 tools/exl3/emit_verifier.py --output /mnt/raid0/llm/tmp/exl3-cpu-20261003/evidence/verifier-<ts>`
   after the last code change so the sealed rows bind the final commit; add model-level
   measurement rows (`epyc.exl3.measurement.v1`) from the booked windows.
4. Later: MCG in the importer (the executor has it), padded dims, EXL3 n-gram rows (buun EXL3N), HIP backend
   (the `test_exl3_mul_mat` cases in test-backend-ops are its harness).

### Window runbook (main session)

```bash
cd /mnt/raid0/llm/worktrees/llama-exl3-cpu-20261003
M=/mnt/raid0/llm/models/exl3-gguf/Qwen3.8-Flash-Next-EXL3-3.05bpw/model-00001-of-00002.gguf
# smoke (greedy, plain graph path; same env as the codified recipe)
env -u LD_LIBRARY_PATH OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false \
    GGML_IQK=1 GGML_FUSED_DECODE_OFF=1 GGML_FA_SPLIT_KV=0 GGML_NOHUGEPAGE_PROCESS=1 \
    taskset -c 0-95 numactl --interleave=all build/bin/llama-completion -m $M -t 48 -fa on -ctk f16 -ctv f16 \
    --no-mmap --temp 0 -n 64 -no-cnv -p "The capital of France is"
# serving / speed: qwen38_flash_next_recipe.build_serve_command(bindir="<worktree>/build/bin", model=M, mtp=True|False)
# quality: llama-perplexity -m $M -f /mnt/raid0/llm/data/wiki.test.raw -c 512 (vs the same on UD-IQ4_XS / IQ4_XS-uniform)
```
Knobs: `GGML_EXL3_SPLITK_TARGET` (default 64; 1 disables split-K), `GGML_EXL3_ISA=scalar|vnni|vbmi`.

## Resume

```bash
cd /mnt/raid0/llm/worktrees/llama-exl3-cpu-20261003
# AK loop first: step must not contain measuring / A/A / accumulate / serving gate
python3 -c "import json;print(json.load(open('/mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/store/loop-status.json'))['step'])"
nice -n 19 taskset -c 176-183 cmake --build build -j 6 --target test-exl3 llama-perplexity llama-completion llama-cli llama-server
env -u LD_LIBRARY_PATH nice -n 19 taskset -c 176-183 build/bin/test-exl3          # expect ALL PASS
OPENBLAS_NUM_THREADS=4 nice -n 19 taskset -c 176-183 /mnt/raid0/llm/venvs/llama-gguf-convert/bin/python tools/exl3/test_exl3np.py
G=/mnt/raid0/llm/tmp/exl3-cpu-20261003/gguf K=/mnt/raid0/llm/tmp/exl3-cpu-20261003/kld
env -u LD_LIBRARY_PATH nice -n 19 taskset -c 176-183 build/bin/llama-perplexity -m $G/mat-L4/model-00001-of-00002.gguf \
  -f /mnt/raid0/llm/data/wiki.test.raw -c 128 --chunks 8 -t 8 -b 128 -ub 128 --kl-divergence-base $K/mat-L4.c128.kld
env -u LD_LIBRARY_PATH nice -n 19 taskset -c 176-183 build/bin/llama-perplexity -m $G/native-L4/model-00001-of-00002.gguf \
  -c 128 --chunks 8 -t 8 -b 128 -ub 128 --kl-divergence-base $K/mat-L4.c128.kld --kl-divergence   # and -ub 1
```
Build config: gcc-15, Release, GGML_NATIVE=ON, GGML_OPENMP=ON, LLAMA_CURL=OFF, tests ON (the
champion's native-openmp-gcc15-cpu-v1 recipe). Always run with `env -u LD_LIBRARY_PATH`.
Importer: `/mnt/raid0/llm/venvs/llama-gguf-convert/bin/python tools/exl3/exl3_import.py --help`.

## Blockers / notes

* No blocker for phases 1-4. Phase 5 (full model) needs a booked window: memory/CPU heavy.
* Licences: exllamav3 (MIT, turboderp) — tile-row extraction ported with the MIT notice in
  `exl3.cpp`; buun-llama.cpp engine `1d0f493c` (MIT, ggml authors) — integration convention (type
  ids 60..67, aux-source slots) followed, no code copied; EXL3-1/2 are project-owned.
* Numerics: decode uses int8 activations (per-128-block scales) → ~0.7 % per-op output rel-L2 vs
  canonical (same class as llama.cpp's Q8_K activations); prefill is fp32 (≈ the fp16 rounding
  floor of the canonical reference). The artifact keeps the UD BF16 indexer and IQ4_NL n-gram
  table, so it is not byte-identical to turboderp's model in those two places.
* INF-64 fused decode (`qwen4exp-fused.cpp`) is engaged by default for single-token decode in
  this tree and is single-threaded (4-layer F16 probe at 8 threads: 135.8 ms/token fused vs 36.9
  ms graph); the codified recipe disables it. EXL3 models always take the graph path.
* Housekeeping: the scratchpad `/tmp/claude-1000/-workspace/654db1ee-…/scratchpad` is shared with
  the main session — one early file write there (`moe.txt`) may have overwritten an older file of
  the same name; all later scratch files of this session live in `/mnt/raid0/llm/tmp/exl3-cpu-20261003/scratch/`.
