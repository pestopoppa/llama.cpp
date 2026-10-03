# EXL3-6 CPU — status and resume notes

Branch `experimental/exl3-cpu-20261003` in worktree `/mnt/raid0/llm/worktrees/llama-exl3-cpu-20261003`,
forked from the global champion `90c12df42` (`ak/champion/llama-cpp-ffc1bac82eec`, re-fetched at each
commit; still the tip). Design: `EXL3-CPU-DESIGN.md`. Plan of record: epyc-root
`handoffs/active/exl3-cpu-mi210-implementation.md` (EXL3-6). Nothing pushed.

## Done

| Phase | What | Evidence |
|---|---|---|
| 1 design | `EXL3-CPU-DESIGN.md` | — |
| 2 importer | `tools/exl3/exl3_import.py` (+ `exl3np.py`): EXL3 safetensors + UD GGUF → 2-split GGUF; modes `native` (EXL3_K band8 + `.exl3_suh/.exl3_svh`) and `materialize` (F16, canonical stage roundings); GDN V-head reorder as 128-block perms; `--check` correlates every EXL3 tensor against the UD tensor | `tools/exl3/test_exl3np.py`: decode + materialized-weight **hashes bit-exact** vs EXL3-1 on all 10 mul1 fixtures; `--check` on layers 0-3: block corr ≥ 0.979 (experts K3) / ≥ 0.9988 (dense K5); band8 round trip exact |
| 3 ggml + CPU | `GGML_TYPE_EXL3_1..8` (ids 60..67 = buun), `ggml-cpu/exl3.cpp` executor on MUL_MAT / MUL_MAT_ID with aux srcs; int8 GEMV (exllamav3 VBMI/VNNI extraction, per-128-block act scales), fp32 GEMM prefill fallback | `build/bin/test-exl3`: ALL PASS — scalar ≡ VNNI ≡ VBMI bit-exact (GEMV and GEMM, dense and experts, rows 1-13, K1-8); vs independent canonical reference GEMV rel-L2 0.5-1.0e-2, GEMM 3.5-5e-4; real fixtures real-mul1-k3/k4 GEMV 7.0e-3, GEMM 4.2-4.7e-4 |
| 4 integration | loader creates aux next to each EXL3 weight (`exl3.layout` checked), `llama_model::exl3_aux` → `build_lora_mm{,_id}` attach; fused INF-64 decode bypassed for EXL3 models | 4-layer truncation, native vs materialized F16 (`llama-perplexity --kl-divergence`, wiki.test.raw, c128): **KLD mean 1.47e-4** (ub 128) / **2.46e-4** (ub 1, GEMV path); max 0.004 / 0.006 |

Commits: `c02961978` (design, importer, ggml types, executor, test), `6bbc96c6f` (llama wiring).

Artifacts (not in git):
* shared PLE split: `/mnt/raid0/llm/models/exl3-gguf/qwen4exp-ple-table-ud-iq4nl.split2of2.gguf` (28.8 GB, UD IQ4_NL table)
* `/mnt/raid0/llm/tmp/exl3-cpu-20261003/gguf/{native-L4,mat-L4}/` (4-layer test models, UD head) —
  recreate with `/mnt/raid0/llm/tmp/exl3-cpu-20261003/make_l4.sh`
* logs: `/mnt/raid0/llm/tmp/exl3-cpu-20261003/logs/`, KLD base: `…/kld/mat-L4.c128.kld`

## Next (in order)

1. **EXL3 `lm_head`** once shard 7 lands (`model-00007-of-00007.safetensors`): rerun the importer
   without `--keep-ud-head` on L4 (native + materialize) and repeat the two KLD runs.
2. **Full-model native GGUF** (all 48 layers) when the download finishes:
   `exl3_import.py --mode native --check --out /mnt/raid0/llm/models/exl3-gguf/Qwen3.8-Flash-Next-EXL3-3.05bpw`
   (~52 GB + the shared PLE split; disk had 160 GB free at 17:50Z with ~65 GB of download still
   to land — check `df` first). A full materialized F16 reference does not fit; the full-model
   comparison is KLD vs the UD / uniform IQ4_XS references in a booked window.
3. Load + short generation smoke on the full model — **needs a main-session window** (memory/CPU).
4. Performance work (before any speed window): (a) gate|up merge with per-segment `suh` (MoE N=1
   has only 50 units/op for 96 threads), or split-K; (b) the prep barrier (2 barriers/op); (c) VNNI
   tier target attributes for portable builds (today the file assumes -march=native on Zen 5);
   (d) GEMM microkernel and decode for prefill; (e) port EXL3 into the INF-64 fused decode path
   (production decode path) or measure graph-vs-fused overhead explicitly.
5. Wire `test-exl3` results into the EXL3 verifier writer (`epyc.exl3.verifier.v1`) and the
   Vidya adapter (research `scripts/kernel_rnd/exl3/evidence.py`) — the belief-kernel rule.
6. Later: indexer `index_qk_proj` as EXL3 (fused q|k, split by view), MCG codebook types
   (68..75), padded dims, EXL3 n-gram rows (buun EXL3N), HIP backend.

## Resume

```bash
cd /mnt/raid0/llm/worktrees/llama-exl3-cpu-20261003
# check the AK loop first: step must not contain measuring / A/A / accumulate / serving gate
python3 -c "import json;print(json.load(open('/mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/store/loop-status.json'))['step'])"
nice -n 19 taskset -c 176-183 cmake --build build -j 6 --target test-exl3 llama-perplexity llama-cli llama-server
env -u LD_LIBRARY_PATH nice -n 19 taskset -c 176-183 build/bin/test-exl3          # expect ALL PASS
OPENBLAS_NUM_THREADS=4 nice -n 19 taskset -c 176-183 /mnt/raid0/llm/venvs/llama-gguf-convert/bin/python tools/exl3/test_exl3np.py
# 4-layer KLD (base computed once from mat-L4):
G=/mnt/raid0/llm/tmp/exl3-cpu-20261003/gguf
env -u LD_LIBRARY_PATH nice -n 19 taskset -c 176-183 build/bin/llama-perplexity -m $G/native-L4/model-00001-of-00002.gguf \
  -c 128 --chunks 8 -t 8 -b 128 -ub 128 --kl-divergence-base /mnt/raid0/llm/tmp/exl3-cpu-20261003/kld/mat-L4.c128.kld --kl-divergence
```
Build config: gcc-15, Release, GGML_NATIVE=ON, GGML_OPENMP=ON, LLAMA_CURL=OFF, tests ON (the
champion's native-openmp-gcc15-cpu-v1 recipe). Always run with `env -u LD_LIBRARY_PATH`.

## Blockers / notes

* No blocker for phases 1-4 on truncated models. Full-model runs need a booked window.
* Licences: exllamav3 (MIT, turboderp) — tile-row extraction ported with notice in `exl3.cpp`;
  buun-llama.cpp engine `1d0f493c` (MIT, ggml authors) — integration convention (type ids,
  aux-source slots) followed, no code copied; EXL3-1/2 are project-owned.
* Numerics: decode uses int8 activations (per-128 block scales) → ~0.7 % per-op output rel-L2 vs
  canonical, the same class as llama.cpp's Q8_K activations; prefill path is fp32 (≈ the fp16
  rounding floor of the canonical reference).
