# Development recall diagnostic

`recall_runner.py` is an explicit one-cell diagnostic for `dev-code-00` at 32,768
tokens, query 32,767. It accepts only the frozen development token JSON, the
released reference PCA `.pt` map, and the Qwen3.8-27B-Q8_0 model. It does not
select items from a cohort or read heldout token payloads.

The runner requires `--execute`, an owned GPU filler grant
(`FILLER_ID`, `FILLER_NEEDGPU=1`, `FILLER_STOPFILE`, `FILLER_DEADLINE`,
`FILLER_END`, `FILLER_PIDFILE`), and a new build proof. A build proof has schema
`lsa-recall-build-proof-v1`, the exact source commit of this runner and hook,
`binary_sha256`, `libllama_sha256`, and `dsos` mapping absolute library paths to
SHA-256 digests. The runner checks a recall-specific marker in `libllama` and
verifies the process's mapped llama/ggml libraries. It refuses an absent or
unproved future build.

The model receipt is a prior full-hash witness with `sha256`, `dev`, `ino`,
`size`, `head_sha256`, and `tail_sha256`; the latter two hash the first and last
65,536 bytes. The runner checks inode, size, and edge bytes before launch, and
does not rehash the complete model during the grant.

The callback outputs only one aggregate per global layer: selected attention
mass, top-4096 key recall, selected budget, query, layer, mode, released-map
digest, token digest, and model scale/softcap. It reconstructs dense attention
from dequantized physical cached K with a double-precision dot; it is not a
bitwise GPU attention-kernel comparison. The source hook is disabled unless
`LLAMA_LSA_RECALL_OUT` is explicitly set. A real diagnostic remains unrun until
the future build, grant, and runner invocation are reviewed.
