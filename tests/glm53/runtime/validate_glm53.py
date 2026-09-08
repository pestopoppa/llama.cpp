#!/usr/bin/env python3
"""Fail-closed validator for GLM-5.3 native-MTP runtime evidence.

This script performs no inference.  It validates producer-authored JSON/JSONL
records emitted by the candidate runtime tests and server client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "epyc.glm53.native_mtp_validation.v1"
ROLLBACK_SCHEMA = "epyc.glm53.mtp_rollback_case.v1"
ARM_SCHEMA = "epyc.glm53.server_arm.v1"
POOL_SCHEMA = "epyc.glm53.kpool_prefill_case.v1"


class Refusal(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Refusal(f"cannot read valid JSON {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Refusal(f"{path}:{lineno}: row is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise Refusal(f"cannot read valid JSONL {path}: {exc}") from exc
    if not rows:
        raise Refusal(f"{path}: no rows")
    return rows


def require(row: dict[str, Any], keys: Iterable[str], where: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise Refusal(f"{where}: missing {missing}")


def finite_positive(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal(f"{where}: expected number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise Refusal(f"{where}: expected finite positive number, got {value!r}")
    return result


def finite_nonnegative(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal(f"{where}: expected number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise Refusal(f"{where}: expected finite nonnegative number, got {value!r}")
    return result


def compare_semantic_components(components: Any, where: str) -> None:
    """Compare producer-normalized live state, never raw allocator/storage bytes."""
    if not isinstance(components, dict) or not components:
        raise Refusal(f"{where}: semantic_state must be a non-empty object")
    required = {"convolution", "kda_recurrent", "mla_indexer", "kpool", "speculative"}
    if set(components) != required:
        raise Refusal(f"{where}: semantic_state keys must be exactly {sorted(required)}")
    for name, component in components.items():
        cwhere = f"{where}.{name}"
        if not isinstance(component, dict):
            raise Refusal(f"{cwhere}: expected object")
        require(component, ("comparison", "max_abs_diff", "tolerance",
                            "rollback_live_sha256", "replay_live_sha256"), cwhere)
        diff = finite_nonnegative(component["max_abs_diff"], cwhere + ".max_abs_diff")
        tolerance = finite_nonnegative(component["tolerance"], cwhere + ".tolerance")
        comparison = component["comparison"]
        if comparison not in ("exact", "numeric"):
            raise Refusal(f"{cwhere}: comparison must be exact or numeric")
        if comparison == "exact" and (tolerance != 0 or
                component["rollback_live_sha256"] != component["replay_live_sha256"]):
            raise Refusal(f"{cwhere}: exact normalized-live-state comparison differs")
        if diff > tolerance:
            raise Refusal(f"{cwhere}: max abs {diff} exceeds declared tolerance {tolerance}")


def validate_arm(path: Path, *, require_mtp: bool, speed_shape: bool) -> dict[str, Any]:
    doc = load_json(path)
    if not isinstance(doc, dict) or doc.get("schema") != ARM_SCHEMA:
        raise Refusal(f"{path}: expected schema {ARM_SCHEMA}")
    require(doc, ("arm_id", "date", "metric", "metric_direction", "protocol_id",
                  "reps", "evidence_status", "argv", "environment", "binary",
                  "candidate", "model", "host", "host_caveats", "context_recipe",
                  "in_window_sample_refs", "requests", "native_mtp", "contention"), str(path))
    if doc["metric"] != "accepted_output_tokens_per_second" or doc["metric_direction"] != "higher_better":
        raise Refusal(f"{path}: wrong speed unit or metric direction")
    if doc["protocol_id"] != "P-BENCH-4" or int(doc["reps"]) != 5:
        raise Refusal(f"{path}: owning protocol/reps must be P-BENCH-4/n=5")
    if doc["evidence_status"] not in ("observation", "correctness"):
        raise Refusal(f"{path}: evidence_status must be observation or correctness")
    if not isinstance(doc["host_caveats"], list) or not doc["host_caveats"]:
        raise Refusal(f"{path}: host caveats are required")
    if not isinstance(doc["in_window_sample_refs"], list) or not doc["in_window_sample_refs"]:
        raise Refusal(f"{path}: in-window sample refs are required")
    requests = doc["requests"]
    if not isinstance(requests, list) or not requests:
        raise Refusal(f"{path}: requests must be non-empty")
    drafted = accepted = predicted = 0
    rates: list[float] = []
    for i, row in enumerate(requests):
        where = f"{path}:requests[{i}]"
        require(row, ("kind", "predicted_n", "predicted_per_second",
                      "draft_n", "draft_n_accepted", "canonical_response_sha256"), where)
        pred = int(row["predicted_n"])
        dft = int(row["draft_n"] or 0)
        acc = int(row["draft_n_accepted"] or 0)
        if pred < 1 or dft < 0 or acc < 0 or acc > dft:
            raise Refusal(f"{where}: impossible token counters")
        predicted += pred
        drafted += dft
        accepted += acc
        rate = finite_positive(row["predicted_per_second"], where + ".predicted_per_second")
        if row["kind"] == "measure":
            rates.append(rate)
            if speed_shape:
                if pred != 512 or row.get("finish_reason") != "length":
                    raise Refusal(f"{where}: P-BENCH-4 shape needs 512 tokens and length stop")
    if require_mtp:
        if doc["native_mtp"].get("spec_type") != "draft-mtp":
            raise Refusal(f"{path}: spec_type is not draft-mtp")
        if drafted == 0:
            raise Refusal(f"{path}: zero drafted tokens; MTP could be silently disabled")
        if accepted == 0:
            raise Refusal(f"{path}: no accepted draft token observed")
        if not doc["native_mtp"].get("draft_graph_dispatch_observed"):
            raise Refusal(f"{path}: no native draft-head graph dispatch evidence")
        events = doc["native_mtp"].get("verification_events")
        if not isinstance(events, list) or not events:
            raise Refusal(f"{path}: missing parsed per-verification server-log events")
        verified_accept = verified_reject = False
        for i, event in enumerate(events):
            where = f"{path}:native_mtp.verification_events[{i}]"
            if not isinstance(event, dict):
                raise Refusal(f"{where}: expected object")
            require(event, ("accepted", "drafted", "log_ref"), where)
            ev_accepted, ev_drafted = int(event["accepted"]), int(event["drafted"])
            if ev_drafted < 1 or not 0 <= ev_accepted <= ev_drafted or not event["log_ref"]:
                raise Refusal(f"{where}: impossible counters or missing raw-log reference")
            verified_accept |= ev_accepted > 0
            verified_reject |= ev_accepted < ev_drafted
        if not verified_accept or not verified_reject:
            raise Refusal(f"{path}: per-verification logs do not cover both acceptance and rejection")
    if speed_shape and len(rates) != 5:
        raise Refusal(f"{path}: P-BENCH-4 shape needs exactly five measured requests")
    median = statistics.median(rates) if rates else None
    mad = statistics.median(abs(x - median) for x in rates) if rates else None
    return {"arm_id": doc["arm_id"], "predicted": predicted, "drafted": drafted,
            "accepted": accepted, "unaccepted": drafted - accepted,
            "acceptance": accepted / drafted if drafted else None,
            "median_tps": median, "mad_tps": mad, "n_speed": len(rates)}


def validate_rollback(path: Path, draft_max: int, eps: float) -> dict[str, Any]:
    rows = load_jsonl(path)
    seen: set[tuple[int, int]] = set()
    cycles: set[int] = set()
    for i, row in enumerate(rows):
        where = f"{path}:{i + 1}"
        if row.get("schema") != ROLLBACK_SCHEMA:
            raise Refusal(f"{where}: expected schema {ROLLBACK_SCHEMA}")
        require(row, ("producer", "model_path", "architecture", "recipe_id",
                      "cycle", "forced_accepted_prefix", "drafted_tokens",
                      "accepted_prefix_tokens", "replay_prefix_tokens",
                      "continuation_steps", "continuation_tokens_equal",
                      "max_abs_logit_diff", "used_context_max_abs_logit_diff",
                      "used_context_tokens_equal", "logits_finite", "used_context_restore",
                      "declared_logit_tolerance", "state_scope"), where)
        cycle = int(row["cycle"])
        if row["architecture"] not in ("glm5next", "glm5-next"):
            raise Refusal(f"{where}: architecture alias not GLM5Next")
        k = int(row["forced_accepted_prefix"])
        if not 0 <= k <= draft_max:
            raise Refusal(f"{where}: forced prefix {k} outside [0,{draft_max}]")
        if int(row["drafted_tokens"]) < max(1, k):
            raise Refusal(f"{where}: drafted token count does not cover forced prefix")
        if list(row["accepted_prefix_tokens"]) != list(row["replay_prefix_tokens"]):
            raise Refusal(f"{where}: replay did not use the exact accepted prefix")
        if int(row["continuation_steps"]) < 2 or row["continuation_tokens_equal"] is not True:
            raise Refusal(f"{where}: observable continuation differs or horizon is too short")
        declared = finite_nonnegative(row["declared_logit_tolerance"], where + ".declared_logit_tolerance")
        if declared != eps:
            raise Refusal(f"{where}: record tolerance {declared} differs from predeclared CLI {eps}")
        diff = finite_nonnegative(row["max_abs_logit_diff"], where + ".max_abs_logit_diff")
        used_diff = finite_nonnegative(row["used_context_max_abs_logit_diff"], where + ".used_context_max_abs_logit_diff")
        if (row["logits_finite"] is not True or row["used_context_restore"] is not True
                or row["used_context_tokens_equal"] is not True):
            raise Refusal(f"{where}: nonfinite logits or restore-into-used-context missing")
        if diff > declared or used_diff > declared:
            raise Refusal(f"{where}: logits differ (max abs {diff} > {declared})")
        if "observable continuation" not in str(row["state_scope"]):
            raise Refusal(f"{where}: state scope is missing or overclaims raw-state equality")
        key = (cycle, k)
        if key in seen:
            raise Refusal(f"{where}: duplicate cycle/prefix {key}")
        seen.add(key)
        cycles.add(cycle)
    missing = [(cycle, k) for cycle in cycles for k in range(draft_max + 1)
               if (cycle, k) not in seen]
    if missing:
        raise Refusal(f"{path}: missing forced prefix cases {missing}")
    if len(cycles) < 2:
        raise Refusal(f"{path}: need at least two cycles to cover repeated accept/reject restoration")
    return {"cycles": len(cycles), "draft_max": draft_max, "cases": len(rows),
            "prefixes": list(range(draft_max + 1)), "epsilon": eps}


def validate_pool(path: Path, kpool: int, topk: int, eps: float) -> dict[str, Any]:
    rows = load_jsonl(path)
    if kpool != 4 or topk not in (8, 2048):
        raise Refusal("supported contracts are kpool=4/topk=8 fixture or kpool=4/topk=2048 real model")
    by_tokens: dict[int, dict[str, Any]] = {}
    variants: dict[tuple[int, str], dict[str, Any]] = {}
    for i, row in enumerate(rows):
        where = f"{path}:{i + 1}"
        if row.get("schema") != POOL_SCHEMA:
            raise Refusal(f"{where}: expected schema {POOL_SCHEMA}")
        require(row, ("case_id", "prefill_tokens", "ubatch", "kpool", "topk",
                      "select_incomplete_tail", "selected_tokens", "selected_member_positions",
                      "selected_tail_positions", "next_logits_sha256", "next_token",
                      "max_abs_logit_diff_vs_pair", "declared_logit_tolerance",
                      "semantic_state"), where)
        if int(row["kpool"]) != kpool or int(row["topk"]) != topk:
            raise Refusal(f"{where}: model pool metadata drift")
        if row["select_incomplete_tail"] is not True:
            raise Refusal(f"{where}: incomplete-tail selection must be enabled")
        n = int(row["prefill_tokens"])
        selected = int(row["selected_tokens"])
        expected = min(n - n % kpool, topk) + n % kpool
        if selected != expected:
            raise Refusal(f"{where}: selected {selected}, expected {expected} complete-pool+tail tokens")
        members = list(row["selected_member_positions"])
        tail = list(row["selected_tail_positions"])
        if len(members) != min(n - n % kpool, topk) or len(tail) != n % kpool:
            raise Refusal(f"{where}: selected member/tail position counts are wrong")
        expected_tail = list(range(n - n % kpool, n))
        if tail != expected_tail:
            raise Refusal(f"{where}: incomplete tail positions {tail}, expected {expected_tail}")
        if any(p < 0 or p >= n - n % kpool for p in members) or len(set(members)) != len(members):
            raise Refusal(f"{where}: invalid or duplicated complete-pool member positions")
        groups = sorted(members)
        for start in range(0, len(groups), kpool):
            group = groups[start:start + kpool]
            if len(group) != kpool or group != list(range(group[0], group[0] + kpool)) or group[0] % kpool:
                raise Refusal(f"{where}: selected members do not form aligned complete groups of {kpool}")
        variant = str(row.get("prefill_variant", ""))
        if variant not in ("full", "chunked"):
            raise Refusal(f"{where}: prefill_variant must be full or chunked")
        key = (n, variant)
        if key in variants:
            raise Refusal(f"{where}: duplicate token-count/variant {key}")
        variants[key] = row
        if variant == "full":
            by_tokens[n] = row
    required = {topk - 1, topk, topk + 1, topk + kpool - 1,
                topk + kpool, topk + kpool + 1}
    missing = sorted(required - set(by_tokens))
    if missing:
        raise Refusal(f"{path}: missing threshold/tail full-prefill token counts {missing}")
    compared = 0
    for n in sorted(required):
        if (n, "full") not in variants or (n, "chunked") not in variants:
            raise Refusal(f"{path}: token count {n} lacks exactly one full and one chunked row")
        a, b = variants[(n, "full")], variants[(n, "chunked")]
        if a["next_token"] != b["next_token"]:
            raise Refusal(f"{path}: n={n}: full/chunk next token differs")
        for row in (a, b):
            declared = finite_nonnegative(row["declared_logit_tolerance"], f"{path}:n={n}.tolerance")
            if declared != eps:
                raise Refusal(f"{path}: n={n}: record tolerance {declared} differs from CLI {eps}")
            diff = finite_nonnegative(row["max_abs_logit_diff_vs_pair"], f"{path}:n={n}.diff")
            if diff > declared:
                raise Refusal(f"{path}: n={n}: logit difference {diff} exceeds {declared}")
        if eps == 0 and a["next_logits_sha256"] != b["next_logits_sha256"]:
            raise Refusal(f"{path}: n={n}: exact full/chunk logit digests differ")
        compare_semantic_components(a["semantic_state"], f"{path}:n={n}.full.semantic_state")
        compare_semantic_components(b["semantic_state"], f"{path}:n={n}.chunked.semantic_state")
        compared += 1
    if compared < len(required):
        raise Refusal(f"{path}: need full/chunk comparisons for all six boundary cases")
    return {"cases": len(rows), "full_chunk_pairs": compared,
            "boundary_tokens": sorted(required), "max_selected_tokens": topk + kpool - 1}


def write_report(output: Path, results: dict[str, Any]) -> None:
    payload = {"schema": SCHEMA, "verdict": "PASS", **results}
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=Path, action="append", default=[])
    ap.add_argument("--mtp-arm", type=Path, action="append", default=[])
    ap.add_argument("--rollback", type=Path)
    ap.add_argument("--pool", type=Path)
    ap.add_argument("--draft-max", type=int, default=5)
    ap.add_argument("--logit-epsilon", type=float, default=1e-5)
    ap.add_argument("--pool-logit-epsilon", type=float, default=0.0)
    ap.add_argument("--kpool", type=int, default=4)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--speed-shape", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not (args.arm or args.mtp_arm or args.rollback or args.pool):
        raise Refusal("no inputs: refusing a vacuous PASS")
    results: dict[str, Any] = {"arms": []}
    for path in args.arm:
        results["arms"].append(validate_arm(path, require_mtp=False,
                                             speed_shape=args.speed_shape))
    for path in args.mtp_arm:
        results["arms"].append(validate_arm(path, require_mtp=True,
                                             speed_shape=args.speed_shape))
    if args.rollback:
        results["rollback"] = validate_rollback(args.rollback, args.draft_max,
                                                 args.logit_epsilon)
    if args.pool:
        results["pool"] = validate_pool(args.pool, args.kpool, args.topk,
                                         args.pool_logit_epsilon)
    write_report(args.output, results)
    print(json.dumps({"verdict": "PASS", **results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Refusal as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        raise SystemExit(2)
