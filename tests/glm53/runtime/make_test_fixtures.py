#!/usr/bin/env python3
"""Generate tiny synthetic PASS records for testing validate_glm53.py only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "fixtures"
ROOT.mkdir(exist_ok=True)

def semantic_state(seed):
    return {name: {"comparison": "exact", "max_abs_diff": 0.0, "tolerance": 0.0,
                   "rollback_live_sha256": seed * 64, "replay_live_sha256": seed * 64}
            for name in ("convolution", "kda_recurrent", "mla_indexer", "kpool", "speculative")}

rollback = []
for cycle in (1, 2):
    for k in range(6):
        rollback.append({
            "schema": "epyc.glm53.mtp_rollback_case.v1",
            "producer": "test-glm5next-mtp-rollback/public-api-v1",
            "model_path": "/fixture/glm5next.gguf", "architecture": "glm5next",
            "recipe_id": "glm53-target-verify-seq-rm-v1", "cycle": cycle,
            "forced_accepted_prefix": k, "drafted_tokens": 5,
            "accepted_prefix_tokens": list(range(k)), "replay_prefix_tokens": list(range(k)),
            "continuation_steps": 4, "continuation_tokens_equal": True,
            "max_abs_logit_diff": 0.0, "declared_logit_tolerance": 0.0,
            "used_context_max_abs_logit_diff": 0.0, "logits_finite": True,
            "used_context_restore": True, "used_context_tokens_equal": True,
            "state_scope": "observable continuation after real target suffix seq_rm; raw unused storage excluded",
        })
(ROOT / "rollback-pass.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rollback))

pool = []
for n in (2047, 2048, 2049, 2051, 2052, 2053):
    for variant, ubatch in (("full", 4096), ("chunked", 257)):
        pool.append({
            "schema": "epyc.glm53.kpool_prefill_case.v1", "case_id": f"n{n}",
            "prefill_tokens": n, "prefill_variant": variant, "ubatch": ubatch,
            "kpool": 4, "topk": 2048, "select_incomplete_tail": True,
            "selected_tokens": min(n - n % 4, 2048) + n % 4,
            "selected_member_positions": list(range(min(n - n % 4, 2048))),
            "selected_tail_positions": list(range(n - n % 4, n)),
            "next_logits_sha256": "c" * 64, "next_token": 101,
            "max_abs_logit_diff_vs_pair": 0.0, "declared_logit_tolerance": 0.0,
            "semantic_state": semantic_state("d"),
        })
(ROOT / "pool-pass.jsonl").write_text("".join(json.dumps(x) + "\n" for x in pool))
