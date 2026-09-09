# GLM-5.3 run evidence wrapper

The wrapper acquires the physical `0-95` region lock before starting its owned
server PID. It uses the exact argv/environment in a generated serve plan,
verifies linkage, hashes source/binary loaded libraries, records bounded GGUF
header identity, samples contention during requests, and terminates only its
exact captured server PID. It never drops caches or changes another process.

Resolve and check the plan, candidate, probe, and all six pinned shard stats
without acquiring a lock or starting the server:

```bash
python3 run_glm53_arm.py --preflight-only --mode smoke \
  --plan ../serve/candidate_serve_plan.json \
  --source-tree /mnt/raid0/llm/llama.cpp-experimental-glm53-20260908 \
  --manifest ../expected_manifest.json
```

Smoke (startup plus one 32-token request; no P-BENCH-4 arm sidecar):

```bash
python3 run_glm53_arm.py --mode smoke --plan ../serve/candidate_serve_plan.json \
  --out evidence-smoke --source-tree /mnt/raid0/llm/llama.cpp-experimental-glm53-20260908 \
  --manifest ../expected_manifest.json
```

Full observation (existing warmup plus five 512-token requests and `arm.json`):

```bash
python3 run_glm53_arm.py --mode full --plan ../serve/candidate_serve_plan.json \
  --out evidence-mtp-full --arm-id glm53-mtp \
  --source-tree /mnt/raid0/llm/llama.cpp-experimental-glm53-20260908 \
  --manifest ../expected_manifest.json
```

Replace the plan with `candidate_plain_plan.json` and arm id with `glm53-plain` for
the matched control. Run `python3 -m unittest -v test_run_glm53_arm.py` for pure
tests. The arm remains an observation unless the complete owning measurement
protocol and host gates attest it.

Validate a completed full native-MTP sidecar with the existing schema validator:

```bash
python3 ../runtime/validate_glm53.py --mtp-arm evidence-mtp-full/arm.json \
  --speed-shape --output evidence-mtp-full/arm-validation.json
```

`profile_phases.py` is a bounded profiler instrument for a running server whose
exact PID is supplied as `GLM_SERVER_PID`. It constructs one long prompt, records
one cache-populating one-token request as the prefill window, and then records a
128-token request that reuses the cached prompt as the decode window. The phase
labels describe those request windows; they are not a claim that every sample in
a window belongs exclusively to one graph phase. Startup, tokenization, template
rendering, and report generation happen outside the recorded windows.

The instrument owns and stops only the two `perf` PIDs it launches. Its evidence
is observational unless the caller also records the server recipe, binary and
library identities, physical region lock, contention samples, and model identity.

The final suite drivers preserve the pinned canonical, quality, repeated-decode,
long-prefill, and distinct-prompt inputs used by the committed GLM validation.
They connect to an already owned server; the wrapper still owns locking, process
lifecycle, library identity, and server logs.

```bash
python3 final_plain_suite.py --port 18497 --out final-plain \
  --scorer-root /path/to/epyc-inference-research/scripts/benchmark
python3 final_mtp_suite.py --port 18497 --out final-mtp \
  --scorer-root /path/to/epyc-inference-research/scripts/benchmark
python3 -m unittest -v test_run_glm53_arm.py test_final_workload_clients.py
```

The canonical client's content-only SALAD label is a degeneracy diagnostic. It
is not a quality score for responses whose generated text is carried in the
separate reasoning field.
