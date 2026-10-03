# EXL3 on the llama.cpp CPU backend — design (EXL3-6, CPU half)

Branch `experimental/exl3-cpu-20261003`, forked from the global champion `90c12df42`
(`ak/champion/llama-cpp-ffc1bac82eec`). Target: turboderp `Qwen3.8-Flash-Next-exl3` branch
`3.05bpw_h5_ng5` (arch `qwen4exp`). Plan of record: epyc-root
`handoffs/active/exl3-cpu-mi210-implementation.md` (EXL3-6). Status / resume notes:
`EXL3-CPU-STATUS.md` beside this file.

## 0. Format facts this design relies on (verified, not re-derived)

From EXL3-1 (`epyc-inference-research` `scripts/kernel_rnd/exl3`, commit `bc9ac47b`), INF-71 X0
(`docs/design/exl3-mul1-ggml-type.md`) and the shard-1 safetensors header of the 3.05 branch:

* Each quantized linear is four tensors: `trellis` int16 `[k/16, n/16, 16*K]` (native order: tile
  `(kt, nt)` at `(kt*n/16 + nt)*16K` words), `suh` f16 `[k]`, `svh` f16 `[n]`, `mul1` int32 scalar
  (the codebook multiplier marker; must equal `0x83DCD12D`).
* Tile = 16 inputs x 16 outputs = 256 weights. State `i` is the 16-bit window of the tile's
  circular `256*K`-bit stream (little-endian u32 words, MSB first) starting at
  `((i+257)*K - 16) mod 256K`; state `i` sits at tile coordinate given by `make_tc_perm`
  (EXL3-1 `state_index`).
* `mul1` codebook (canonical): `w = fp16( fp16(0x6400 + bytesum(s*0x83DCD12D)) * fp16(0x1EEE) + fp16(0xC931) )`,
  one rounding. Pre-rounding this is exactly affine: `w = k_inv*bytesum + c0`,
  `k_inv = 1774/262144`, `c0 = 1024*k_inv + fp16(0xC931) = -3.453125`. (exllamav3's own CPU path
  uses `(bytesum-510)*k_inv`, i.e. `c0' = -3.451309`; we use the exact `c0`.)
* Forward: `y = svh ⊙ H128( Q^T · H128(suh ⊙ x) )`, `H128` the orthonormal Sylvester Hadamard
  applied per 128-block (`suh`/`svh` fold sign, channel scale, codebook scale, global scale).
  `suh` differs per tensor and per expert, so the input transform is per (tensor, expert, row).
* 3.05 branch tensor classes (shard-1 header): routed experts K3 (`[160,40,48]` gate/up,
  `[40,160,48]` down), shared expert / attention q,k,v,o / GDN `in_proj_qkv`, `in_proj_z`,
  `out_proj` K5, indexer `index_qk_proj` K3 (q|k fused), `lm_head` K5 (`head_bits 5`). Not
  quantized (f16/bf16 in the checkpoint): router `gate`, `shared_expert_gate`, GDN
  `in_proj_a/b`, `A_log`, `dt_bias`, conv1d, norms, hyper-connection weights, PLE
  `key_proj`/`value_proj`, `embed_tokens`. Every quantized dim of this model is a multiple of
  128, so no padded tensors occur (the importer refuses padding rather than half-supporting it).

## 1. GGUF carriage

**Types.** `GGML_TYPE_EXL3_1 .. GGML_TYPE_EXL3_8` (ids 60..67, `mul1`) and
`GGML_TYPE_EXL3M_1 .. GGML_TYPE_EXL3M_8` (ids 68..75, `mcg`), names `exl3_K` / `exl3m_K`,
`blck_size 16`, `type_size 2*K` bytes — the same ids, names and per-block size as the
buun-llama.cpp donor (intake-1575, MIT), so the two trees agree on what a type id means. The
type encodes K and the codebook (buun's 76..83 = 3INST stay unused). `mcg` is not affine in a
byte sum, so it always runs the fp32 path (validated on EXL3-1's real GLM-5.3 MCG fixture); the
importer handles `mul1` checkpoints only (Q38FN is `mul1`). K is per tensor, so mixed-K
models need nothing extra. Rows are **not** independently decodable: `to_float` is not provided
for row use; the only consumer is the matrix executor (section 3).

**Shapes and layout.** The trellis tensor keeps the logical weight shape the arch expects
(`[k, n]` dense, `[k, n, n_expert]` experts) so `create_tensor` shape checks and
`ggml_nbytes` work unchanged (`k*n*K/8` bytes per matrix). The bytes are repacked at import
from native tile order into **band8**: tile `(kt, nt)` at
`((nt/8)*(k/16)*8 + kt*8 + nt%8) * 16K` words. A band = 8 output tiles = 128 outputs = one
output Hadamard block, stored as one contiguous k-stream. This is exllamav3's `swz=1` layout
(its measured 3.4x over per-column traversal on cold DRAM) and EXL3-2's `band8`. Experts are
contiguous per expert (`nb[2]`). Repack is a lossless permutation; the GGUF records the
canonical source identity (HF repo + revision, per-file sha256 of the source shards) under
`exl3.*` KV keys and `exl3.layout = "band8-v1"`; the loader refuses a file whose layout key is
missing or different.

**Aux tensors.** For a weight `X.weight` of an EXL3 type the GGUF carries `X.exl3_suh`
(f16, `[k]` or `[k, n_expert]`) and `X.exl3_svh` (f16, `[n]` or `[n, n_expert]`), byte copies of
the checkpoint vectors (after the block permutations of section 5). The model loader creates
them next to the weight in the same context/buffer, generically for any arch (hook in
`llama_model_loader::create_tensor`), and records `weight -> {suh, svh}` in a map owned by the
model. The `mul1` marker is checked at import and not stored.

## 2. Where the Hadamard steps go: aux sources on MUL_MAT / MUL_MAT_ID

The input and output transforms do not fit `vec_dot` (a tile spans 16 rows; the output H128
mixes 128 rows). Rather than a new op, the EXL3 executor runs **inside `MUL_MAT` / `MUL_MAT_ID`
with the transform vectors attached as extra sources** — the buun convention:

| op | src0 | src1 | src2 | src3 | src4 |
|---|---|---|---|---|---|
| `MUL_MAT` | trellis `[k,n]` | x f32 | `svh` f16 `[n]` | `suh` f16 `[k]` | – |
| `MUL_MAT_ID` | trellis `[k,n,E]` | x f32 | ids i32 | `svh` f16 `[n,E]` | `suh` f16 `[k,E]` |

Why: the loader's buffer-type probes, the scheduler, the allocator and graph reuse all treat
these as ordinary matmuls (they iterate every `src[j]`), and the graph-side change is confined
to `build_lora_mm` / `build_lora_mm_id` (look up the weight in the model's aux map, attach).
Fail-closed: the CPU backend aborts on an EXL3 matmul without both aux tensors, and
`supports_op` accepts EXL3 only for these two ops with F32 activations and 128-aligned dims
(probe graphs without aux are accepted, as in buun, because placement probes precede
attachment). Per-expert / per-tensor transforms therefore live in the operator contract, not
in a quant trait (EXL3-6 requirement).

## 3. CPU executor

Entry: `ggml_compute_forward_mul_mat{,_id}` divert to `ggml_cpu_exl3_compute()` when `src0`
is an EXL3 type, before any vec_dot_type conversion; `ggml_graph_plan` asks
`ggml_cpu_exl3_work_size()` for the scratch. Two phases, one barrier:

1. **Prep** (rows split over threads): for every (row r, expert e) instance,
   `x' = H128(suh_e ⊙ x_r) / sqrt(128)` in fp32, then symmetric int8 quantization **per
   128-input block** (scale `amax_b/127`, RNE, clamp ±127); store the activation splat
   (`x8` replicated x4 as int32 — the `vpdpbusd` signed operand), per-block factors
   `f_b = k_inv*scale_b`, and the per-row constant `corr = c0 * Σ_b scale_b*Σx8_b`. For the
   prefill path the unquantized `x'` is kept instead.
2. **GEMV** (decode/verify, rows ≤ 8 per instance): work unit = (expert instance, band of 128
   outputs, ≤4-row chunk). The band kernel streams the band's k-stream once, extracts each tile
   row's 16 states into one zmm (lane = output column), multiplies by the `mul1` constant and
   accumulates `bytesum*x8` with one `vpdpbusd` per 16 weights per row (exllamav3 VNNI/VBMI
   extraction, ported — MIT); every 8 tiles (one 128-input block) the int32 accumulators are
   flushed into fp32 with `f_b`. The unit then adds `corr`, applies the output `H128`,
   `/sqrt(128)` and `svh` for its own 128 outputs and stores — **no reduction across threads, no
   second barrier**, results independent of thread count.
3. **Prefill** (rows > 8 per instance): dequant → GEMM. The unit (instance, band) decodes its
   band to fp32 with the canonical fp16 codebook (`[k x 128]`, k-chunked to stay in L2) and
   runs a register-blocked fp32 GEMM over all rows routed to the instance, then the same output
   epilogue. Rows are grouped by expert exactly as `mul_mat_id` does.

ISA tiers: scalar reference (always built), AVX-512 VNNI, AVX-512 VBMI (Zen 5 default;
`GGML_EXL3_ISA=scalar|vnni|vbmi` caps for tests). K1..K8 are compile-time specialised; the 3.05
branch uses K3 and K5.

**Threading.** Units are handed out with the threadpool chunk counter (first unit = thread
index, then `chunk_add`); every unit is computed entirely by one thread, so the assignment never
changes the arithmetic. **GEMV split-K**: decode ops with few bands (attn_k/v 4, ssm_out and
attn_output 20, shared expert 5/20) stream from too few cores to reach DRAM bandwidth,
so for ops with ≤ 64 instances and fewer than 32 GEMV units the k range is split into S slices
(S from the op's shape and routing only — S = ceil(64 / units), cap k/256;
`GGML_EXL3_SPLITK_TARGET`, default 64, 1 disables); slices write raw partials, one barrier, then a
fixed-order reduction + the shared epilogue. Ops that already have ≥ 32 units (attn_qkv 80,
MoE gate/up 50, MoE down 200 at N=1) stream from enough cores and skip the extra barrier. Outputs are
bit-identical for every thread count and ISA tier (tested). Not in v1: merged gate|up with
per-segment `suh`, fusing prep into the previous op (each EXL3 op costs one extra barrier, two
with split-K).

## 4. Dense path

Every EXL3-quantized dense projection (attention q/k/v/o, GDN qkv/z/out, shared expert,
`lm_head`) uses the same executor via `MUL_MAT` with N = tokens. The head is K5,
`[2560, 248320]` = 1940 bands — the largest single tensor and the best-subscribed op.
The indexer `index_qk_proj` (K3, q|k fused, 640 outputs) is **not** imported in v1: the
UD GGUF's BF16 `indexer.{q,k}_proj` stay (1.6 M params per full-attention layer, negligible
bytes); a later step can import it fused and split the output by view.

## 5. Tensor mapping and the GDN V-head reorder

The importer maps EXL3 names onto the UD GGUF names (`in_proj_qkv→attn_qkv`, `in_proj_z→attn_gate`,
`out_proj→ssm_out`, `q/k/v/o_proj→attn_q/k/v/output`, experts→`ffn_{gate,up,down}_exps`
stacked over 512 experts, shared expert→`ffn_*_shexp`, `lm_head→output`). llama.cpp's
qwen3.5-family converter reorders GDN V heads from grouped to tiled order
(`_LinearAttentionVReorderBase`). The heads are 128 wide — exactly one Hadamard block — so the
reorder is a permutation of whole 128-blocks, which commutes with the block-diagonal H128:
* output-side (`attn_qkv` V part, `attn_gate`): permute the band8 bands and the matching `svh`
  blocks;
* input-side (`ssm_out`): permute the 8-tile k-groups inside every band and the `suh` blocks.
The importer validates every mapping by correlating a materialised EXL3 matrix against the UD
GGUF's dequantised tensor (a wrong permutation reads as correlation ≈ 0).

## 6. n-gram table (`per_layer_token_embd`) and everything else

Kept from the UD GGUF (IQ4_NL, 29 GB, row-gathered — not a matmul). To avoid duplicating it per
test artifact, the importer writes it once into a standalone GGUF that serves as split 2 of 2
(`split.no = 1`); every EXL3 artifact is a 2-split model whose second split is a symlink to that
file. All other non-EXL3 tensors (hyper-connection weights, norms, router, GDN `ssm_a/alpha/beta/
conv1d/dt/norm`, PLE projections, `token_embd`, indexer q/k), the tokenizer and all arch KV are
copied byte-for-byte from the UD GGUF (same model, same converter conventions). The 3.05
branch's own n-gram rows (buun's `EXL3N` type) are out of scope for v1.

## 7. Correctness contract (tolerances)

Propositions, each with its own oracle (EXL3-R10):

1. **Packed decode**: our state extraction + canonical codebook == EXL3-1 scalar decode,
   bit-exact (all 256 states, K1..K8, real fixtures).
2. **Executor arithmetic**: every ISA tier == the scalar Q8 reference in this tree, **bit-exact**
   (integer accumulation is exact; the fp32 epilogue order is fixed: per-block flush in
   ascending block order, `+corr`, butterfly H128, `*(1/sqrt128)`, `*svh`).
3. **Operator output vs canonical** (`materialized_weight_fp32_fma_v1` from EXL3-1): declared
   envelope relative-L2 ≤ 0.02 per output vector on the real fixtures and real tensors (int8
   activation noise; EXL3-2 measured < 0.025 with one scale per row, we use per-128 scales);
   the prefill path (fp32 activations, canonical codebook) ≤ 2e-3.
4. **Model level**: an EXL3-native GGUF and a *materialised* GGUF (same tensors reconstructed
   to F16 by the importer, everything else identical) must agree on logits of a truncated
   (first-N-layers) model within a declared KLD / top-1 envelope; then full generation.
   Token equality alone never stands in for (1)–(3).

## 7a. Evidence

`tools/exl3/emit_verifier.py` runs `test-exl3 --json` pinned and seals one
`epyc.exl3.verifier.v1` row per proposition through the research repo's single EXL3 writer
(producer `epyc.exl3.evidence/v1`), so the rows project through root's `exl3-verifier` Vidya
adapter. Model-level KLD receipts (measurement schema) are left to the booked windows.

## 8. What is deliberately not done in v1

3INST codebook, MCG in the importer (the executor has it), padded (non-128) dims, the EXL3 n-gram rows, MTP, the INF-64 fused
single-token decode path (`qwen4exp-fused.cpp` is bypassed when the model has EXL3 tensors —
it calls vec_dot directly; it is single-threaded and production recipes already disable it with
`GGML_FUSED_DECODE_OFF=1`, so EXL3 and the GGUF anchors both run the graph path), HIP. Each is a refusal at load or a documented fallback, never a
silent wrong answer.
