# Chapter 01: llama.cpp Toolchain & Patches

## Introduction

This project uses a **fork of llama.cpp** at `github.com/pestopoppa/llama.cpp` with local optimizations for AMD EPYC 9655 "Turin" architecture. The fork includes parallel tensor repack (2.2x model loading speedup), sliding window attention (SWA) fixes for speculative decoding, and prompt lookup ported to llama-server.

The toolchain uses **git worktrees** to isolate production and experimental work, preventing branch conflicts when multiple agents share access. Production inference MUST use the `production-consolidated-v2-v2` branch (rebased 2026-03-03) — feature work happens in separate worktrees.

## Git Worktree Architecture

The codebase is split into two physical directories sharing a single git history. Production lives at `/mnt/raid0/llm/llama.cpp` and must always stay on the `production-consolidated-v2` branch — all benchmarks and orchestration use this build. Experimental work happens in `/mnt/raid0/llm/llama.cpp-experimental`, where you can switch branches freely without affecting production.

<details>
<summary>Directory layout and worktree rules</summary>

| Directory | Branch | Purpose |
|-----------|--------|---------|
| `/mnt/raid0/llm/llama.cpp` | `production-consolidated-v2` | **Production** - benchmarks, stable inference |
| `/mnt/raid0/llm/llama.cpp-experimental` | `feature/*` branches | **Experimental** - new features, research |

**Production directory** (`/mnt/raid0/llm/llama.cpp`):
- **NEVER** checkout a different branch
- **NEVER** commit experimental work
- Stay on `production-consolidated-v2` at all times
- All benchmarks and orchestration use this build

**Experimental directory** (`/mnt/raid0/llm/llama.cpp-experimental`):
- Switch branches freely
- Test new features without affecting production
- Build with `./build/bin/llama-*` binaries
- Changes here don't affect production

</details>

<details>
<summary>Common operations and branch verification</summary>

<details>
<summary>Code: worktree management commands</summary>

```bash
# Check current worktrees
cd /mnt/raid0/llm/llama.cpp
git worktree list

# Expected output:
# /mnt/raid0/llm/llama.cpp               6b43356a1 [production-consolidated-v2]
# /mnt/raid0/llm/llama.cpp-experimental  xxxxxxxx [feature/paged-attention]

# Start experimental work
cd /mnt/raid0/llm/llama.cpp-experimental
git checkout production-consolidated-v2
git checkout -b feature/my-new-feature

# Build experimental version
cmake -B build -DGGML_NATIVE=ON -DGGML_AVX512=ON
cmake --build build -j 96

# Verify binary version
./build/bin/llama-cli --version
```

</details>

<details>
<summary>Code: branch safety verification</summary>

```bash
# Manual verification
cd /mnt/raid0/llm/llama.cpp
git branch --show-current
# Output: production-consolidated-v2

# If wrong branch, fix with:
git checkout production-consolidated-v2
```

</details>

**Critical**: Never run benchmarks or live inference on a feature branch. Results won't be reproducible.

</details>

## Production Patches

The fork includes three major optimizations, two already merged upstream. These patches represent the project's direct contributions to the llama.cpp ecosystem — each one solves a real bottleneck we hit during inference optimization.

<details>
<summary>Patch 1: Parallel Tensor Repack (PR #18239)</summary>

**Status**: Merged upstream
**Speedup**: 2.2x faster model loading on 96-core systems

Parallelizes the tensor repack operation that converts GGUF tensors to runtime format. Before this patch, model loading was single-threaded and took 45-60 seconds for 235B Q4_K_M models. With parallel repack, loading drops to 20-25 seconds.

**Implementation**: Splits tensor repack across available threads using OpenMP parallel loop with dynamic scheduling. Each thread processes independent tensor blocks, writing to pre-allocated output buffers.

| Method | Load Time | Speedup |
|--------|-----------|---------|
| Original (single-threaded) | 54.2s | 1.0x |
| Parallel repack | 24.8s | **2.2x** |

</details>

<details>
<summary>Patch 2: SWA Speculation Fix (PR #18720)</summary>

**Status**: Merged upstream
**Speedup**: Enables spec decode for SWA models (was crashing)

Fixed `std::bad_alloc` crash when using speculative decoding with sliding window attention (SWA) models. The crash occurred in `llama_kv_cache::slot_info` during KV cache initialization because SWA requires consecutive context positions, incompatible with speculation's non-sequential token prediction.

**Root Cause**: Draft model predicted tokens at position `N+K`, but target model with SWA expected consecutive positions `[N, N+1, ..., N+K]`. KV cache allocation failed when trying to allocate non-contiguous slots.

**Fix**: Added SWA compatibility check in speculative decoder initialization. If target model uses SWA, disable speculation and fall back to standard generation.

**Models Affected**: Gemma-3 series (SWA with window size 8192).

</details>

<details>
<summary>Patch 3: Prompt Lookup for llama-server</summary>

**Status**: Local patch (not yet submitted)
**Speedup**: 8.6-12.7x on document QA tasks

Ported the prompt lookup optimization from `llama-cli` to `llama-server` to enable document summarization acceleration in the orchestrator stack. Prompt lookup detects repeated n-grams between prompt and generation, copying them directly instead of predicting token-by-token.

**Best Results**:
- Summarization: 95.18 t/s (12.7x)
- Code editing: 25.82 t/s (8.6x)
- Requires source material in context

**Implementation** (re-ported to `production-consolidated-v2`, 2026-03-06):
- `--lookup` CLI flag enables per-slot ngram cache built from prompt tokens
- Spec-first/lookup-fallback architecture: draft model tried first, prompt lookup as fallback when no draft or draft produces poor candidates
- Per-slot `ngram_cache_context` isolation for parallel request support
- Cache initialized from prompt tokens at `SLOT_STATE_GENERATING`, updated after each decode and speculative acceptance
- Compatible with `-md` (draft model): `--lookup` + `-md` enables both strategies simultaneously

<details>
<summary>Code: prompt lookup via API and CLI</summary>

```bash
# Server: standalone lookup (no draft model needed)
llama-server -m model.gguf --lookup --port 8080
curl -X POST http://localhost:8080/v1/chat/completions \
  -d '{"model":"m","messages":[{"role":"user","content":"Summarize: [document]"}]}'

# Server: spec + lookup combined (draft model + prompt lookup fallback)
llama-server -m model.gguf -md draft.gguf --lookup --draft-max 16 --port 8080

# CLI: prompt lookup (uses different flag name)
llama-cli -m model.gguf -f prompt.txt --lookup-ngram-min 3
```

</details>
</details>

## Build System

Building llama.cpp for this hardware means enabling AVX-512 and its extensions (VBMI, VNNI) to take full advantage of Zen 5's true 512-bit execution units. The same cmake flags apply to both production and experimental directories.

<details>
<summary>Build configuration and verification</summary>

<details>
<summary>Code: production build</summary>

```bash
cd /mnt/raid0/llm/llama.cpp

# Configure with AVX-512 and native CPU optimizations
cmake -B build \
  -DGGML_NATIVE=ON \
  -DGGML_AVX512=ON \
  -DGGML_AVX512_VBMI=ON \
  -DGGML_AVX512_VNNI=ON \
  -DCMAKE_BUILD_TYPE=Release

# Build with all cores
cmake --build build -j 96

# Install binaries (optional)
cmake --install build --prefix /mnt/raid0/llm/llama.cpp/install
```

</details>

<details>
<summary>Code: experimental build</summary>

```bash
cd /mnt/raid0/llm/llama.cpp-experimental

# Same configuration
cmake -B build \
  -DGGML_NATIVE=ON \
  -DGGML_AVX512=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j 96

# Test experimental binary
./build/bin/llama-cli --version
./build/bin/llama-cli -m /mnt/raid0/llm/models/test.gguf -p "Hello"
```

</details>

<details>
<summary>Code: AVX-512 verification</summary>

```bash
# Check AVX-512 support in binary
./build/bin/llama-cli --version | grep AVX512

# Expected output:
# AVX512 = 1
# AVX512_VBMI = 1
# AVX512_VNNI = 1
```

</details>

**Note**: EPYC 9655 "Turin" has true 512-bit AVX-512 execution units (not double-pumped like Intel Alder/Raptor Lake). AVX-512 VNNI provides 2x INT8 throughput over AVX2.

</details>

## Binary Usage Patterns

Three binaries handle different inference modes: `llama-cli` for interactive and batch work, `llama-speculative` for draft-model acceleration, and `llama-server` for production API serving. Each has its own flags and typical launch patterns.

<details>
<summary>Binary commands and examples</summary>

<details>
<summary>Code: llama-cli (Interactive/Batch)</summary>

```bash
# Standard completion
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -p "Write a function to compute factorial" \
  -n 512 -t 96 --temp 0

# Prompt lookup (for document QA)
/mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m model.gguf \
  -f prompt_with_source.txt \
  --lookup-ngram-min 3 \
  -t 96
```

</details>

<details>
<summary>Code: llama-speculative (Draft Model)</summary>

```bash
# External draft model (11x speedup on code)
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -md /mnt/raid0/llm/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf \
  --draft-max 24 -t 96 -p "prompt"
```

</details>

<details>
<summary>Code: llama-server (Production Orchestrator)</summary>

```bash
# HOT tier server (port 8080: frontdoor)
/mnt/raid0/llm/llama.cpp/build/bin/llama-server \
  -m /mnt/raid0/llm/models/Qwen3-Coder-30B-A3B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --threads 96 --parallel 4 \
  --override-kv qwen3moe.expert_used_count=int:6 \
  --ctx-size 32768 --no-mmap

# Worker server (port 8082: spec + lookup)
/mnt/raid0/llm/llama.cpp/build/bin/llama-server \
  -m /mnt/raid0/llm/models/Qwen2.5-7B-Instruct-f16.gguf \
  -md /mnt/raid0/llm/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf \
  --draft-max 24 --lookup-ngram-min 3 \
  --host 0.0.0.0 --port 8082 --threads 96 --parallel 4
```

</details>
</details>

## Rebase History

### production-consolidated-v2 (2026-03-03)

Rebased `production-consolidated` onto `origin/master` as `production-consolidated-v2`:

- **16 custom commits** applied (from 25 original)
- **9 skipped**: upstream-merged (lookup crash, lookahead, MTMD), superseded (prompt lookup → upstream ngram spec), doc-only, merge commit
- **4 conflicts resolved**: qwen3next.cpp (upstream API change), llama-kv-cache.cpp (SWA API x2), server-context.cpp (slot erase)
- **SSM checkpoint** cherry-picked in (conditional Go for code tasks)
- **Build passes**: `cmake -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_CURL=ON`
- **Smoke tests pass**: Qwen3-Coder-30B (34.0 t/s), Qwen3.5-35B-A3B (10.9 t/s)

### Prompt Lookup Re-port (2026-03-06)

The `--lookup` feature (commit `8e35dbc01` on `production-consolidated`) was lost during the v1→v2 rebase. Cherry-pick failed with 10 conflicts across `server-context.cpp` and `server-task.cpp`. Manually ported across 5 files (88 insertions):

- `common/arg.cpp` — `--lookup` CLI flag definition
- `common/common.h` — `bool lookup` in `common_params`
- `tools/server/server-task.h` — `bool lookup` in `task_params`
- `tools/server/server-task.cpp` — JSON serialization, defaults, parsing
- `tools/server/server-context.cpp` — ngram cache fields in `server_slot`, cache lifecycle (init/reset/update), `can_speculate()` gate, spec-first/lookup-fallback draft generation

Tested: standalone `--lookup`, combined `-md` + `--lookup` (spec+lookup). Both produce valid completions.

## Known Limitations

Two model families have hard incompatibilities that will silently produce garbage or crash if you ignore them. These aren't bugs to be fixed — they're architectural constraints of the models themselves.

<details>
<summary>SSM models and BOS token mismatch</summary>

### SSM Models (Qwen3-Next)

SSM architecture models require consecutive context positions for state propagation — draft token rejection during speculation corrupts the recurrent state which encodes cumulative history.

**Experimental: SSM State Checkpointing** (2026-03-03, on `production-consolidated-v2`): A checkpoint/restore mechanism saves the full recurrent state (~63 MB for Qwen3.5-35B-A3B) before speculation, restores on rejection, then re-advances through accepted tokens. Go/No-Go benchmark showed 1.56x speedup on code generation (92% accept rate) but 0.89x regression on summarization (51.9% accept rate). **Verdict**: Enable spec decode for code generation tasks only; non-code tasks should still avoid speculation with SSM models.

<details>
<summary>Code: correct vs incorrect SSM usage</summary>

```bash
# ❌ WRONG - will produce garbage
llama-speculative -m Qwen3-Next-80B-A3B-Q4_K_M.gguf -md draft.gguf

# ✅ CORRECT - expert reduction only
llama-cli -m Qwen3-Next-80B-A3B-Q4_K_M.gguf \
  --override-kv qwen3next.expert_used_count=int:2
```

</details>

### Qwen3-Coder-480B BOS Token

The 480B model has BOS token mismatch (`BOS=','`) that breaks all speculation:

<details>
<summary>Code: correct 480B usage</summary>

```bash
# ❌ Speculation will fail
llama-speculative -m Qwen3-Coder-480B-A35B-Q4_K_M.gguf -md draft.gguf

# ✅ Use expert reduction only
llama-cli -m Qwen3-Coder-480B-A35B-Q4_K_M.gguf \
  --override-kv qwen3moe.expert_used_count=int:3
```

</details>

**Result**: 10.3 t/s with MoE3 (vs 3.0 t/s baseline), but no speculation compatibility.

</details>

## Troubleshooting

When things go wrong, it's usually one of two things: wrong branch or wrong directory. These quick checks cover the most common issues.

<details>
<summary>Common issues and fixes</summary>

### "Build is using wrong version"

<details>
<summary>Code: version verification and recovery</summary>

```bash
pwd  # Should be /mnt/raid0/llm/llama.cpp for production
git branch --show-current  # Should be production-consolidated-v2
./build/bin/llama-cli --version  # Verify commit hash
```

If on wrong branch:
```bash
cd /mnt/raid0/llm/llama.cpp
git checkout production-consolidated-v2
cmake --build build -j 96  # Rebuild
```

</details>

### "I accidentally worked on production-consolidated-v2"

<details>
<summary>Code: recovery steps</summary>

1. Stash or commit changes: `git stash` or `git commit -am "WIP"`
2. Create feature branch: `git checkout -b feature/my-work`
3. Switch production back: `cd /mnt/raid0/llm/llama.cpp && git checkout production-consolidated-v2`
4. Move work to experimental: `cd /mnt/raid0/llm/llama.cpp-experimental && git cherry-pick <hash>`

</details>
</details>

<details>
<summary>References</summary>

- Fork: https://github.com/pestopoppa/llama.cpp
- Upstream: https://github.com/ggml-org/llama.cpp
- PR #18239: Parallel tensor repack (merged)
- PR #18720: SWA speculation fix (merged)
- `docs/reference/LLAMA_CPP_WORKTREES.md` - Detailed worktree workflow

</details>

---
