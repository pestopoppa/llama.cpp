# GLM-5.3 serve-plan generator

`generate_serve_plan.py` emits JSON and a quoted inner shell command. It never
starts a process. Launch remains gated on explicit task authority and a physical
region lock for CPUs 0-95.

The generator requires the candidate `llama-server`, first model shard, and the
current codified recipe as explicit inputs. At generation time it parses and
checks the copied affinity, OMP/GGML environment, base server, FA, and KV-cache
constants against that source, then records its SHA-256. It copies no champion
binary, build number, or binary digest.

```bash
python3 generate_serve_plan.py \
  --binary /path/to/candidate/build/bin/llama-server \
  --model /path/to/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  --recipe-source /path/to/qwen38_flash_next_recipe.py \
  --output candidate_serve_plan.json
```

The default plan is CPU-only, text-only native MTP. It uses the model's embedded
NextN block with `--spec-type draft-mtp`; it never emits `-md`. Serve defaults
are draft-max 3 and p-min 0. Acceptance/performance validation separately tests
draft-max 1, 3, and 5.

Generate the matched plain diagnostic by adding `--plain-diagnostic`. This mode
removes only the speculative flags and keeps the binary, model, environment,
CPU placement, context, KV precision, and other server flags matched.

Pure tests:

```bash
python3 -m unittest -v test_generate_serve_plan.py
```

The emitted `shell_preview` is the inner command only. Do not execute it until
the owning session has task authority, the physical region lock, a completed
candidate build, and candidate-local ggml linkage verification.
