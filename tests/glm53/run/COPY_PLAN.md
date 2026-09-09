# Candidate source copy

Destination: `tests/glm53/` in the GLM-5.3 candidate.

- Copy `run_glm53_arm.py`, `test_run_glm53_arm.py`, `profile_phases.py`,
  `README.md`, and this file into `run/`.
- Keep the final workload clients and suite drivers together with
  `canonical24-prompts.json`, `quality-manifest.json`,
  `prefill2029-tokens.json`, and `response_classifier.py`. These JSON files are
  small pinned measurement inputs, not generated responses.
- Update the shared `runtime/probe_server.py`.
- Update the shared `serve/generate_serve_plan.py`.
- Keep categorical auditors together in `runtime/`; `audit_serial_plain.py` imports `audit_paired_evidence.py`.
- Register `tests/test-glm5next-real-divergence.cpp` as a test executable.

The runner resolves shared instruments through sibling `../runtime` and
`../serve` directories. Do not duplicate them under `run/`. Do not copy
generated plans, preflight JSON, runtime evidence, model manifests, checksums,
bytecode caches, or binary fixtures. Runtime callers supply `--plan`,
`--source-tree`, and `--manifest`.
