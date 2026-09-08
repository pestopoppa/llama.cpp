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
