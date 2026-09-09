# GLM5Next reachability of champion Qwen CPU optimizations

This audit is for the experimental GLM5Next candidate derived from champion
`ef81196d5bdd4190b46dff4ae7eecc333a46c8ce`. It asks whether optimizations
already developed for Qwen3.8/Qwen4Next are compiled and reachable from the
GLM5Next CPU graph. It does not claim a production result.

The inspected pre-change CPU library is
`build-glm53-cpu/bin/libggml-cpu.so.0.16.0`, SHA-256
`b4a77e4a63175cb49e14fd85d1415c615665bf906c7198782af937574d7f1d7b`.
`strings` finds `GGML_MMID_SLAB`, `GGML_QSPLIT`, `GGML_ROWCOL_SPLIT`,
`GGML_TINY_SOLO`, `GGML_EMPTY_SKIP`, `GGML_VEC_Q8K`, `GGML_IQK_Q8_0`,
`GGML_ROWEXACT_N`, and `GGML_FA_SPLIT_KV`. Presence alone is not evidence of
execution; the table separates binary presence from GLM runtime evidence.

| Optimization | GLM status and evidence | Shape or dispatch condition |
|---|---|---|
| IQK MoE and `GGML_MMID_SLAB` | Active. The GLM decode profile contains `iqk_mul_mat_moe_rows`; server logs report IQK MoE for weight types 12/13/14 and activation type 99. | `ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp` accepts supported K-quant expert weights. GLM expert tensors are `(4096,2048,288)` or `(2048,4096,288)`, top-8 of 288; batch-1 uses the flat expert-row slab path. |
| `GGML_ROWCOL_SPLIT` | Compiled, default on, and architecture-neutral. GLM F32 elementwise nodes meet its row/column predicates where applicable. | `ggml/src/ggml-cpu/common.h`; active when the operation-specific row split has fewer rows than workers and enough columns/elements. |
| `GGML_TINY_SOLO` / `GGML_EMPTY_SKIP` | Compiled and default on. Single-row eligible chains and empty rollback nodes can use them. Multi-row native-MTP tensors do not satisfy the default one-row solo limit. | `ggml/src/ggml-cpu/ggml-cpu.c`; destination and source must be F32, the op must be whitelisted, rows must be at most 1 by default, and destination elements at most 4096. |
| `GGML_QSPLIT` | Compiled and default on for multi-row IQK dense operations. Before this change GLM dense Q8 bypassed IQK, so Q8 could not reach QSPLIT. The new Q8 minimum-row gate exposes it only for qualifying prefill batches. | `ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp`; requires IQK dense dispatch, fewer activation rows than workers, and more than one row unless the explicit single-row width threshold is met. It does not apply to `MUL_MAT_ID`. |
| `GGML_VEC_Q8K` | Compiled but ineligible for the dominant GLM weights. | `ggml/src/ggml-cpu/iqk/iqk_config.h` and IQK dispatch use Q8_K activation packing for IQ2/IQ3/IQ4_XS weights. GLM dense Q8_0 and Q4_K/Q5_K/Q6_K expert paths use other activation formats, including Q8_2_X4 for K-quants. |
| Native dense Q8 | Active and dominant in the observed decode. | `ggml/src/ggml-cpu/arch/x86/quants.c`; the plain profile attributes 35.67% of sampled decode self cycles to `ggml_vec_dot_q8_0_q8_0`. These CPU-cycle samples are not a wall-time fraction. |
| IQK dense Q8 | Previously opt-in for every row count. A one-run profile improved 2K prefill from 113.324 to 153.692 prompt tokens/s, while decode changed 6.574 to 6.627 tokens/s and changed the token trajectory. | `GGML_IQK=1 GGML_IQK_Q8_0=1`. This candidate adds `GGML_IQK_Q8_0_MIN_ROWS`, default 1. Dense eligibility uses `src1.ne[1]`, the per-matrix token axis; `ne[2]`/`ne[3]` are broadcast/head batches and do not inflate it. MMID eligibility uses `ids.ne[1]`, intentionally excluding top-k width. The real GLM trace records dense prompt/single/verification token rows 26/1/3-4, including Q8 MLA shapes `[256,1,64,1]` and `[512,1,64,1]`. Thus the canonical threshold 32 keeps those decode and verification matrices native while allowing long prefill matrices to use IQK. |
| Fused GDN, lightning indexer, DSV4 mHC, flash attention | Active. Server load messages report all fused paths; profile symbols include each family. | GLM builds the corresponding fused ops in `src/models/glm5-next.cpp`. Flash attention is enabled by the canonical context; `GGML_FA_SPLIT_KV=0` keeps the alternate single-row split disabled. |
| Whole Qwen4Exp fused decoder | Ineligible by architecture and intentionally disabled by the canonical recipe. | Implemented in `src/models/qwen4exp-fused.cpp`; GLM does not advertise that model-level fused decoder and the recipe sets `GGML_FUSED_DECODE_OFF=1`. |

`GGML_ROWEXACT_N` originally made its decision again on local tiles and per-expert row groups. This could exactify the 8-row tail of a 40-row dense prefill or a small expert bucket inside a 26-token MMID prefill. The candidate now computes dense eligibility once from full `Ny` and MMID eligibility once from global `ids.ne[1]`, then passes that decision into the low-level IQK operation. Default `GGML_ROWEXACT_N=0` is unchanged; the opt-in semantics intentionally change for large batches with small local groups. The fused up/gate IQK API is not used by GLM and retains its local-`Ny` behavior.

No missing champion decode kernel was found that only needed a GLM architecture
whitelist. The measured GLM mHC decode work is small in the profile
(`DSV4_HC_POST` 0.49%, `PRE` 0.06%, `COMB` approximately 0% sampled self
cycles), and PRE/POST already partition over embedding elements. Widening the
global multi-row solo predicate is not justified by these data.

The most relevant prior controls are preserved rather than repeated:

- `GGML_VEC_Q8K` plus `GGML_QSPLIT` was a +5.24% Qwen served-stack result and is
  already present. See `/mnt/raid0/llm/tmp/inf70/wrapup-20260908/settled_markers.txt`.
- `GGML_ROWCOL_SPLIT` measured +5.3% in the Qwen champion leave-one-out and is
  already default on. See `/mnt/raid0/llm/tmp/inf70/agents/champion1/REPORT.md`.
- Widening `GGML_TINY_SOLO_ROWS` measured -0.16% MTP and -1.04% plain and was
  rejected. See `/mnt/raid0/llm/tmp/inf70/wrapup-20260908/settled_markers.txt`.
- `GGML_SCALE_SPLIT` and `GGML_SOLO_YIELD_ROWCOL` were rejected; see
  `/workspace/docs/design/inf70-close-out-20260908/FIX1-RESULT.md`.
- The prior Q8 8x8 AVX-512BW branch is a closed appendix. Its one measured
  retained experiment, RMS_NORM intra-op parallel reduction, regressed 4.41 to
  4.02 tokens/s (-8.8%) at 96 threads; the remaining kernel body never cleared
  its separate gate. See `/workspace/docs/design/champion-consolidation-audit-20260908.md`
  lines 105-114. It is not reintroduced without a GLM measurement isolating
  arithmetic from memory and synchronization cost.
- Qwen gate/up and QKV graph fusion removed about 50 of about 590 barriers per
  token but measured -2.11%, +0.25%, and -0.57% for the two individual and
  combined arms. See the same consolidation audit lines 95-103. This rejects a
  generic barrier-fusion transplant; it does not reject GLM parallel verification.

The GLM profile and its evidence limits are documented in
`/workspace/docs/reference/models/glm53-cpu-profile-20260908.md`. The real GLM shape trace is retained at
`/mnt/raid0/llm/tmp/glm53-validation-20260908/runtime/mm-shape-trace-20260908T223524Z/`.
Validation of the Q8 cutoff and logical-row correction used the diagnostic-free intermediate library below. This build predates phase-scoped row exactness and failed the later plain original-reference gate:

- Fresh processes at `GGML_IQK_Q8_0_MIN_ROWS=32` observed IQK inactive at
  rows 1, 4, and 31, and active at rows 32, 33, 40, and 48. All seven real
  CPU-backend/reference cases passed. Evidence: `/mnt/raid0/llm/tmp/glm53-validation-20260908/q8-route-boundary/`.
- The same seven cases passed at both `GGML_ROWEXACT_N=0` and 16. The backend
  harness used hardware concurrency (192 workers) and tolerance comparisons; it
  does not by itself prove exact N0/N16 equality. Evidence:
  `/mnt/raid0/llm/tmp/glm53-validation-20260908/final-q8-boundaries/`.
- The real GLM target comparator at N=0 reproduced the known divergent argmax
  13931. At N=16, all five sequential/batched/rollback comparisons had maximum
  absolute logit difference zero and argmax 1246, matching serial. Evidence:
  `/mnt/raid0/llm/tmp/glm53-validation-20260908/runtime/rowexact-logical-control-20260908T224928Z/`;
  N16 result SHA-256 `b2301ba14d1cd75b97d3678d2018a6d083f82b3486f08bb65838e932ec127c02`, comparator binary SHA-256 `ed4abf02810019a962b559409bad1eaace6f995d9fe663640276df5e24f826a5`.
- Superseded intermediate `libggml-cpu.so` SHA-256 is
  `1cc8346f94305cde426be3b23e4076586432af3a098fb044c5c994103870d0ae`;
  its `llama-server` SHA-256 is
  `c23fe97c585072496142a7d99eecc2e128841f9d77bd1544d5c69398850748b7`.
  The library contains `GGML_IQK_Q8_0_MIN_ROWS` and contains no
  `GGML_MM_SHAPE_TRACE`, `MM_SHAPE`, or `MMID_SHAPE` strings.

The superseded intermediate source diff used for those binaries is retained at
`/mnt/raid0/llm/tmp/glm53-validation-20260908/final-q8-source-before-benchmark.patch`,
SHA-256 `bfd3dbe8a9ea36f2d02258c3b808d065a0e94a1beeee2aed54db89acc289674d`.


Committed phase-scoped source is candidate commit `04ffb8ad0` (`cpu: gate Q8
prefill and scope row-exact MTP verification`). Final rebuilt binary identities:

- `libggml-cpu.so`: `8c5a352b3b899aed15b2dcb827d9a55bbac53f5baece2ec904c9c9043a3024bf`
- `libllama.so`: `f81b1fc28c677ff93fd117ab94413d3ce7962714a399fad448d4411485c4e07f`
- `libllama-server-impl.so`: `a3ad4f4554942cad305d7fc32d50f6cf95557d298e2339e7e89f09db186a8c60`
- `llama-server`: `8ce86a370cad067bcc1e9b2bacc7ca74364ab6a94df1d2f3546284c95de3c9fe`

## Phase-scoped row exactness

A server regression gate exposed a phase ambiguity in the width-only row-exact
control. The canonical server splits this 26-token prompt into checkpoint
chunks of 22 and 4 tokens. A global `GGML_ROWEXACT_N=16` therefore changed the
four-token prompt tail and moved the greedy stream at generated token 6, even
without speculative decoding or context reuse. A fresh-context 64-token run
reproduced the change, ruling out stale recurrent or index-pool state.

The final implementation carries a zero-safe `disable_rowexact` bit from the
CPU backend through `ggml_cplan` and `ggml_compute_params`; generic and IQK
matrix dispatch honor it. `llama_set_rowexact()` reapplies the context policy
at every scheduler compute, and cached CPU graph-plan execution refreshes the
policy from its backend. `LLAMA_SPEC_EXACT=row` initializes target and draft
contexts disabled, requires `-np 1` and a positive threshold, and enables the
target only for batches whose rows are all explicitly tracked verification
rows. A guard restores the disabled state on normal and exceptional exits.

The real-model phase gate at
`runtime/rowexact-phase-gate-20260908T233319Z` passed six comparisons with
zero maximum logit difference: disabled split-22+4 prefill versus the original
reference, enable/disable restoration on a reused context, two verification
batches, and a rollback continuation. Its result JSON SHA-256 is
`f293f7840f25f57f29eb99501876a91313de60112af8d04083adcf4d80eec462`.
A fresh phase-scoped plain 64-token server run then matched the preserved
original token stream, including token 6 = 1246. These are experimental CPU
results and do not establish production suitability.
