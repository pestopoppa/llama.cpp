# K28 fused chunked GDN prototype notes

This worktree is a default-off proof gate for a future fused chunked GDN recurrence kernel. It must not change behavior unless `GGML_CUDA_GDN_CHUNKED_PROTOTYPE=1` is set, and the current scaffold still falls back to the existing serial CUDA kernel.

## Current source anchors

- `ggml/src/ggml-cuda/gated_delta_net.cu`: `gdn_chunked_prototype_requested()` parses the default-off env flag.
- `ggml/src/ggml-cuda/gated_delta_net.cu`: `try_launch_gated_delta_net_chunked_prototype()` is the future launch site.
- `ggml/src/ggml-cuda/gated_delta_net.cu`: `launch_gated_delta_net()` still owns the existing serial recurrence dispatch.
- `src/models/delta-net-base.cpp`: `build_delta_net_chunking()` is the reference graph implementation for the chunked equations.
- `src/models/delta-net-base.cpp`: `build_delta_net_fused()` feeds prefill chunks through `ggml_gated_delta_net(..., K=1)`.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_try_gdn_cache_fusion()` identifies the `K > 1` snapshot cache bridge. The prototype must keep falling back there.

## First supported shape

The first real kernel should only launch when all of these are true:

- scalar-gate GDA only: `src_g->ne[0] == 1`
- `dst->type == GGML_TYPE_F32`
- `K == 1` and `keep_rs == false`
- no fused-cache bridge: `cache == nullptr`
- `S_v == 128`
- `n_seqs == 1`
- long prefill: `n_tokens >= 64`

All KDA, BF16-state, snapshot, fused-cache, multi-sequence, smaller-state, and uncertain shapes must remain on `gdn_dispatch()`.

## Minimal kernel sequence

Use one block per `(head, seq)` and loop over chunks in block order. Start with `CS=64` for GDA. Keep FP32 math first; do not introduce MFMA or BF16 until the fusion-only path has correctness and timing evidence.

For each chunk:

1. Load the incoming transposed state layout used by the serial kernel: `state[col * S_v + row]`.
2. Load chunk-local `q`, `k`, `v`, scalar `g`, and `beta`.
3. Build the lower-triangular decay-masked `kb` and `kq` terms following `build_delta_net_chunking()`.
4. Solve `(I + lower(kb)) * attn = -lower(kb)`, then add identity, matching the `ggml_solve_tri()` path.
5. Compute `v_new = v_t - k_cd * state`.
6. Emit `attn_inter + v_attn` into the same output layout used by `gated_delta_net_cuda()`.
7. Update the final recurrent state only. Do not write per-token snapshots in this prototype path.

The proof gate should be discarded if it cannot beat the existing serial op on `test-backend-ops -o GATED_DELTA_NET` without changing full-model behavior when the env flag is unset.
