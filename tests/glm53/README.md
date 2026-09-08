# Portable GLM-5.3 artifact tests

Copy these four source files into `tests/glm53/` in a llama.cpp candidate:

* `glm53_artifact.py`
* `generate_tiny_fixture.py`
* `test_glm53_artifact.py`
* `README.md`

Do not copy generated `.gguf` files, `__pycache__`, or the host-specific
`expected_manifest.json`. Tests generate four ~349 KB GGUFs in a temporary
directory and remove them automatically.

From the candidate repository root, run the normal portable test:

```bash
python3 -m unittest discover -s tests/glm53 -p 'test_*.py' -v
```

The tools resolve `gguf-py` in this order: an explicit `--gguf-py` argument,
the `GGUF_PY` environment variable, then the nearest ancestor repository's
`gguf-py/`. The unit test uses the last two forms and contains no host path.

Opt into the six-shard local artifact test explicitly:

```bash
GLM53_MODEL_DIR=/path/to/UD-Q4_K_XL \
python3 -m unittest discover -s tests/glm53 -p 'test_*.py' -v
```

An identity manifest is optional evidence, independent of loader compatibility:

```bash
GLM53_MODEL_DIR=/path/to/UD-Q4_K_XL \
GLM53_IDENTITY_MANIFEST=/path/to/expected_manifest.json \
python3 -m unittest discover -s tests/glm53 -p 'test_*.py' -v
```

Direct checker use:

```bash
python3 tests/glm53/glm53_artifact.py /path/to/UD-Q4_K_XL \
  --identity-manifest /path/to/expected_manifest.json
```

The small fixtures preserve three KDA dense blocks, one MLA/DSA MoE block,
mHC with four streams, pool size four, vector KDA gates, and one NextN block.
The two architecture/key spellings and present/absent MTP index-sharing and
tail-selection flags use identical deterministic tensor payloads. Optional
NextN embedding and output tensors remain absent so the global fallbacks are
covered structurally.

## Real-model target batch divergence diagnostic

`../test-glm5next-real-divergence.cpp` reproduces the observed greedy split at the
six-token accepted prefix using target-only public decoding APIs. It compares
strictly sequential logits with cumulative batched decoding, a clean four-row
speculative-shaped batch, and the same shape after a prior one-token rollback.
The recorded CPU run used source SHA-256
`4ebb20360fefaf1ee3bb2ab1ba936735dbfa97c357362fc1245c8fffceecc500`
and is preserved outside the source tree at
`/mnt/raid0/llm/tmp/glm53-validation-20260908/runtime/real-divergence-20260908T212632Z/`.

This is a numerical behavior diagnostic with a predeclared `1e-5` logit
tolerance. A nonzero exit records a detected difference; it is not a general
model-quality or rollback-correctness verdict. The clean four-row comparison
can demonstrate that rollback is not required to reproduce a divergence, while
the after-rollback comparison remains separately reported.
