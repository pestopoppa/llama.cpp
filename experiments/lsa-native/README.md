# Native LSA experimental module

This module builds the C5 CPU/HIP kernels and validated native map reader. It does
not yet select these kernels in a Qwen runtime graph. The reviewed HIGH-risk
GGML/cache/graph change remains stopped pending explicit operator approval. No
Python worker, torch library, or map conversion runs in native inference.

Default configure leaves native LSA disabled. Build/testing is correctness work
and requires an actual narrow build-role region claim; all build paths must be
under the handoff scratch root. Example from `/workspace` (absolute paths are
required because region-lock changes directory):

```sh
TREE=/mnt/raid0/llm/tmp/rope-ctx-assessment/lsa-screen-20261103/llama-native-c5
BUILD=/mnt/raid0/llm/tmp/rope-ctx-assessment/lsa-screen-20261103/build-native-module-cpu
/workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list 0-1 --role build --timeout-s 2 --tag lsa-native-configure -- cmake -S "$TREE/experiments/lsa-native" -B "$BUILD" -DLLAMA_LSA_NATIVE=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BLAS=OFF -DBUILD_SHARED_LIBS=ON
/workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list 0-1 --role build --timeout-s 2 --tag lsa-native-build -- cmake --build "$BUILD" -j2
LD_LIBRARY_PATH="$BUILD/ggml/src" /workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list 0-1 --role build --timeout-s 2 --tag lsa-native-test -- ctest --test-dir "$BUILD" --output-on-failure
```

HIP compile uses a distinct build directory. The optional path is validated with
ROCm clang18 and gfx90a, including the source-compatible shared CPU module:

```sh
HIPBUILD=/mnt/raid0/llm/tmp/rope-ctx-assessment/lsa-screen-20261103/build-native-module-hip
/workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list 94-95 --role build --timeout-s 2 --tag lsa-native-hip-configure -- cmake -S "$TREE/experiments/lsa-native" -B "$HIPBUILD" -DLLAMA_LSA_NATIVE=ON -DLLAMA_LSA_NATIVE_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BLAS=OFF -DBUILD_SHARED_LIBS=ON
/workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list 94-95 --role build --timeout-s 2 --tag lsa-native-hip-build -- cmake --build "$HIPBUILD" -j2
LD_LIBRARY_PATH="$HIPBUILD/ggml/src:/opt/rocm/lib" /workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list 94-95 --role build --timeout-s 2 --tag lsa-native-hip-host-test -- ctest --test-dir "$HIPBUILD" --output-on-failure
```

Normal CTest registers `test-lsa-native-hip` without arguments and interprets
return code 77 as SKIPPED. The source exits in that branch before `hipSetDevice`
or any explicit HIP initialization; the default exit77 and CTest skip were
verified from the newly built binary. Device tests require explicit `--run-hip`
and separately held resource claims. `lsa-inspect-hip-memory` calls only the pure
host estimator and tests admission caps and invalid capacity. No GPU execution,
model inference, or server launch is part of the CMake validation.

## LSAMAP1 binary format

All integers and BF16 words are little endian. Nothing is inferred from a
filename. Fixed header is 108 bytes:

| Offset | Bytes | Meaning |
|---|---:|---|
| 0 | 8 | `LSAMAP1` followed by NUL |
| 8 | 4 | Version = 1 |
| 12 | 4 | Query head count |
| 16 | 4 | KV head count |
| 20 | 4 | Physical head dimension |
| 24 | 4 | Rotary prefix dimension |
| 28 | 4 | Index dimension = 64 |
| 32 | 4 | Global attention layer count |
| 36 | 4 | Model identity UTF-8 byte length |
| 40 | 4 | Map provenance identity UTF-8 byte length |
| 44 | 32 | SHA256 of source PT bytes |
| 76 | 32 | SHA256 of all bytes following the header |

Payload is the model identity (no NUL), map provenance identity (no NUL), then
ascending layer records. Each record is layer ID (u32), Q BF16 row-major buffer,
then K BF16 row-major buffer. Each buffer has
`kv_heads * (head_dim - rope_dim) * 64` words. The released 27B geometry has
24/4/256/64 heads/dimension/rotary and 768x64 maps; the 35B geometry has
16/2/256/64 and 384x64 maps. Source keys are exactly
`<layer>.W_Q_idx` / `<layer>.W_K_idx`. Map training produces the 35B tensors;
this exporter never substitutes or reshapes the released 27B tensors into 35B.

The reader requires the caller's approved source hash, exact model identity,
geometry and global layer IDs. Use `name@sha256:GGUF_DIGEST` as model identity
and bind it to a verified model manifest when graph integration is enabled.
A recorded source hash is provenance, not an authenticated signature: the
expected hash is supplied by the trusted caller. Reader checks file bounds,
header/version, exact payload byte count/hash, finite BF16 buffers, unique exact
layer coverage and absence of trailing bytes; rejection leaves the prior bundle
untouched. Per-layer map identity includes payload digest and layer ID, so map
replacement invalidates an index. The loader has no libtorch/OpenSSL dependency.

Offline conversion (CPU-only torch, `weights_only=True`, no download):

```sh
python export_maps.py SOURCE.pt OUTPUT.lsamap --source-sha256 APPROVED_DIGEST --model-identity 'MODEL@sha256:GGUF_DIGEST' --identity RELEASE_OR_TRAINING_ID --geometry 27b --layers 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63
lsa-inspect-map OUTPUT.lsamap 'MODEL@sha256:GGUF_DIGEST' APPROVED_DIGEST 27b 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63
```

The exporter verifies the supplied source hash before torch load and writes the
output exclusively. A pre-existing bound artifact is never overwritten.

## Graph design pending approval

Append `GGML_OP_LSA_NATIVE` before the existing op COUNT so existing op IDs stay
unchanged. Its CPU/HIP dispatch calls native kernels directly on backend buffers;
HIP receives the backend stream, never a CPU-copy callback. Runtime owns one
index per request/layer; normalized logical Q/K before RoPE are projected, only
new K is appended, and physical Hk/Hv rotations plus full quantized KV stay in the
attention path. Selection produces real selected reads. Retain QK norm, RoPE,
GDN and the Qwen output gate. The first route supports a single causal text
sequence with contiguous physical slots starting at zero; unsupported masks,
slots, backends, DCA, MTP/speculation and geometry use existing dense graphs.
Graph reuse must re-check route eligibility. Clear/reuse/copy/shift/restore
invalidates request epoch; suffix deletion truncates the per-layer index.
Selection ambiguity or capacity failure returns a typed request for the original
C5 dense flash-attention path. The adapter implements no replacement dense kernel
and exposes no partial output.

HIP query projection/selection/attention is chunked (32 or 64 queries); new K is
appended once before chunks, causal visibility bounds cluster publication.
Persistent index memory and temporary scratch require explicit byte estimates
and configured caps before allocation. Full-ubatch per-layer attention scratch
is prohibited (2048*24*4612*4 bytes exceeds 900 MB per layer). GPU attention
workspace must be bounded or reused; runtime resource failure chooses dense.

## Validation scope

`test-lsa-map` covers both geometries, known SHA256 vectors, exact source/model
binding, wrong geometry/layer sets, duplicates, malformed header, truncation,
payload mutation, hash-valid nonfinite maps, trailing bytes, oversized lengths,
and atomic rejection. `test-lsa-native` covers kernel/index behavior; its external
author fixture is a separate parity gate. A passing standalone test is not an
actual Qwen runtime or CPU/HIP parity claim. No model inference is authorized by
this module.


## Allocator admission at a 32-query microbatch

The linked `hip_memory_requirements` API was queried from the optional CMake
build without device initialization. For 27B (24 Q heads, four KV heads),
`max_slots=4612`, and one context per global layer:

| Maximum keys | Query chunk | Persistent bytes/context | Workspace bytes/context | Total bytes/16 contexts |
|---:|---:|---:|---:|---:|
| 131072 | 32 | 38994176 | 16265236 | 884150592 |
| 131072 | 2048 | 38994176 | 1040973844 | 17279488320 |
| 1048576 | 32 | 310577152 | 30945300 | 5464359232 |
| 1048576 | 2048 | 310577152 | 1980497940 | 36657201472 |

These are estimator outputs, not measured VRAM residency. They cover allocations
owned by each HIP context: maps, projected keys, buckets, ambiguity flags, bucket
scores, attention scores and fault/stat fields. They exclude model weights, full
physical KV, runtime/backend overhead, allocator alignment, and caller-owned query,
selection and output buffers. The caller must budget their sum before creating
contexts and set a per-context `max_device_bytes` cap; the API returns
`resource_error` when the cap is too small. A 32-query limit bounds attention
scratch; it does not cap persistent index growth. Full-ubatch contexts must never
be selected automatically. Until a runtime owns chunking, admission and explicit
dense fallback, this module remains a component with no graph activation.

## Current component evidence and qualification limits

The optional HIP CMake build passes three CPU-only CTest entries (native kernels,
map validation, host memory admission); the device entry is deliberately skipped.
Its freshly built native CPU suite has 70 passing assertions. The released-map
export/reader check separately proves exact BF16 map-buffer bits across all 16
Q/K layer pairs for both PCA and distilled maps. That is map serialization
correctness, not index projection conformance.

The unchanged author oracle retains **four failed strict assertions** in
`native-cpu-fixtures-v4/fixture-report.json`: the BF16 projected-key step bound
fails in random27b, released PCA layer3, and released distilled layer3 fixtures;
the random27b fixture also differs in ordered selected indices at position5120.
All nine coupled query sets have equal support, and eight ordered lists match;
the differing order changes the selected-attention output by about 5.59e-9 in
that fixture. Those diagnostic observations do not turn a failed conformance
gate into a pass. No native graph activation or 27B author-conformance
qualification is claimed. Existing synthetic HIP device-unit evidence belongs to
its separately frozen binary under `hip-native-unit-evidence-20261103-final`;
the optional CMake build adds build/host validation only.


## Standalone runtime adapter

### One-row HIP selected-index witness

Set `LLAMA_LSA_INDEX_WITNESS` to a new absolute output path together with the
existing pinned native manifest to request one diagnostic capture at global
layer 3, causal query position 5131. The HIP runtime copies the selected row,
projected query, and the first 5132 projected keys from the same live index
cache immediately after selection. It admits at most 4612 selected slots and
publishes a complete binary file atomically without replacing an existing
path. Capture or publication failure takes the dense failure path. The extra
HIP stream synchronization and host copies make this **diagnostic only**; no
latency result from that run is valid. When the variable is unset, no capture
copies, synchronization, or file operations occur.

The file stores the manifest-verified model and map SHA256 digests, released
`.pt` map-source SHA256, request slot/epoch, target layer/position, selection budgets, vectors,
and ordered indices. Replay with
`python3 experiments/lsa-native/replay_index_witness.py FILE --oracle AUTHOR_PY
--model-sha256 HEX --map-sha256 HEX --map-source-sha256 HEX`. The reader requires
the existing pinned author-oracle digest and reports both ordered and set
parity, retaining both ordered index lists. Exit 0 means exact ordered parity;
exit 2 means identical membership with different ordering and is **inconclusive**
until tie behavior and attention-output impact are analyzed; exit 1 means a
true membership mismatch. The future runner must independently bind the **exact rendered
development token IDs**, candidate source commit/build identity, and expected
hashes before accepting this
file. Setting an environment variable or finding a witness file does not prove
the request was the approved development prompt or source build. The capture
requires the author oracle's exact default budgets `(4096, 1024, 512, 4)` and
fails closed on any other tuple. This capture has no held-out
target and does not inspect model outputs.

`llama-lsa-runtime` is a separate target above `llama-lsa-native`; the native
module never links back to it. New files are `llama-lsa-runtime.{h,cpp}`, the
private `llama-lsa-runtime-impl.h`, and optional `llama-lsa-runtime-hip.cpp`.
There are no GGML, KV-cache, or Qwen graph callers. Building this target does
not activate sparse attention in a model.

A runtime is bound to one validated model/map bundle. It owns an index per
request slot and global-attention layer. `begin_request(slot, epoch, backend)`
requires a strictly newer epoch on reuse. `invalidate` handles clear, copy,
restore, shift and slot reuse; `truncate` removes a suffix from every existing
layer. Invalidated or partially failed history stays dense until a fresh request
epoch starts at position0. Restored quantized physical KV is never used to
reconstruct missing logical index keys. Replacing maps preserves model identity
and geometry, invalidates all requests, and requires a new graph map binding.

Each frame supplies normalized pre-RoPE logical **new** Q/K separately from
physical Q/full K/V after RoPE and quantization rotations. Host positions and
new physical slot IDs must equal `first+i`. The caller also attests that existing
physical cells are the sole causal text sequence's contiguous prefix; its future
KV-cache adapter must prove that from real cells and call invalidation for cache
mutations. DCA, speculation/MTP, multiple sequences, unsupported layouts and
incomplete history return `sparse=false` with a typed reason. The caller then
invokes original C5 flash attention. The runtime never synthesizes dense output.

Execution appends only new logical keys once, then projects/selects/attends in
query chunks of at most32. Output is staged internally. Success returns a lease
with runtime identity, generation, backend, device and stream; failure returns no
lease. **Any runtime mutation invalidates every prior lease**, including mutations
for another slot. Callers serialize the lease-consumption interval as well as
runtime calls. `publish_cpu` validates identity/generation/backend/shape/count
before writing a destination. A future HIP GGML consumer must queue publication
on the returned stream before the next mutation (or synchronize safely). No
atomicity is claimed for a catastrophic device fault during that consumer copy.

Host admission accounts for map buffers, bounded request/layer metadata,
transaction/projection/selection staging and conservatively reserved CPU indexes.
HIP admission sums every live layer context and each request's shared transaction
output/query/index/position scratch before allocation; `max_device_bytes` is
mandatory for HIP. Context `max_queries` equals the query chunk, not full model
ubatch. HIP views must be supported, bounded allocations on the current device;
changing a live request's device or stream returns dense. Allocator alignment and
external model/full-KV/backend memory still require headroom outside the recorded
requested-byte budget.

`test-lsa-runtime` currently passes64 CPU assertions with real native attention,
including 32/32/1 chunking, incremental keys, independent causal output, late-chunk
failure injection, stale/forged/cross-slot leases, rollback, clear/copy/restore/map
invalidation, typed original-dense dispatch, aggregate admission and ambiguous
Ward fallback. The HIP bridge compiles for gfx90a. One separately approved synthetic device
bridge invocation passed from frozen project inputs:65 new keys with32/32/1 query
chunks, incremental append, suffix replacement, restore guards, host-pointer
rejection followed by a valid new epoch, late-chunk selected-V NaN fallback with
an untouched destination, poisoned-cache rejection, fresh-zero recovery, and
aggregate device admission. Eighteen nonzero-VRAM/KFD samples overlapped that
invocation. Its exact binary and input hashes are under
`runtime-adapter-20261103/hip-bridge-unit-r1`; it uses analytic maps and F32
physical Q/K/V, and proves that bounded synthetic bridge case only. The normal
CMake device test still skips with exit77 before explicit HIP initialization.
No model graph or author-conformance qualification follows from this unit. The four strict author failures
above remain failed. Source/binary evidence is under `runtime-adapter-20261103`.
The prospective CI verifier requires a clean tested repository; this intentionally
uncommitted review candidate cannot yet receive that verifier's receipt.
