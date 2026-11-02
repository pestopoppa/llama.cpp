#!/usr/bin/env python3
"""g3_gate.py -- chunk-KV G3 approximation gate (KPF-62.6), stdlib only.

The measuring binary is tests/test-chunk-kv-g3.cpp (per-position KL / top-1 / logits NMSE). This script owns
everything else, so the gate logic lives in one place:

  split    print the request ids of a split of a corpus:  sha256(utf-8 id) as an integer, mod 5 == 0 -> dev, else gate
           (P0_THRESHOLDS section 8, frozen dev/gate split rule)
  prereg   check the pre-registration: the corpus manifest's sha256 must equal the one recorded in P0_THRESHOLDS
           section 8 ("- manifest path / sha256: <path> / <sha256>"). TBD or a mismatch refuses (exit 3).
  gate     pool a run directory (positions.jsonl, requests.jsonl, meta.json) and print the G3 table with verdicts
           (P0_THRESHOLDS section 4). Without --dev the prereg check runs first and must pass; with --dev the output is
           labelled DEV and is never a gate result.
  selftest unit checks of the statistics on a synthetic fixture (no model, no files outside a temp dir).

Thresholds (copied from P0_THRESHOLDS section 4, never tuned here):
  mean KL <= 0.05 nats, p95 KL <= 0.25, top-1 >= 0.95, logits NMSE median <= 0.02, p95 <= 0.10,
  retention >= 0.95 (0.85-0.95 = opt-in only, < 0.85 = excluded), naive mean KL / selector mean KL >= 5.
Pooled statistics over all (request, position) pairs; percentiles are nearest-rank.
"""
import argparse
import hashlib
import json
import math
import os
import re
import random
import sys
import types

try:
    import g3_manifest
except ModuleNotFoundError:
    # Some callers load this script by file path rather than putting its directory on sys.path.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import g3_manifest
import tempfile

GATES = {
    "mean_kl": ("<=", 0.05),
    "p95_kl": ("<=", 0.25),
    "top1": (">=", 0.95),
    "nmse_median": ("<=", 0.02),
    "nmse_p95": ("<=", 0.10),
    "naive_ratio": (">=", 5.0),
    "retention": (">=", 0.95),
}
RETENTION_OPT_IN = 0.85
BUCKETS = [("1k-4k", 1024, 4096), ("4k-8k", 4096, 8192), ("8k-16k", 8192, 16384), ("16k-32k", 16384, 32769)]
CORPUS_MIN_REQUESTS = 200
CORPUS_MIN_PER_BUCKET = 50
CHUNKS_MIN, CHUNKS_MAX = 2, 16


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def is_dev(rid):
    return int(hashlib.sha256(rid.encode("utf-8")).hexdigest(), 16) % 5 == 0


def nearest_rank(sorted_x, p):
    if not sorted_x:
        return float("nan")
    k = max(1, math.ceil(p / 100.0 * len(sorted_x)))
    return sorted_x[min(k, len(sorted_x)) - 1]


def median(sorted_x):
    return nearest_rank(sorted_x, 50.0)


def check(name, value):
    op, thr = GATES[name]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NOT COMPUTED"
    ok = value <= thr if op == "<=" else value >= thr
    return "PASS" if ok else "FAIL"


def prereg_sha(thresholds_path):
    """sha256 recorded in P0_THRESHOLDS section 8, or None (TBD / absent)."""
    with open(thresholds_path, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"^## 8\..*?$(.*?)(^## |\Z)", text, re.S | re.M)
    if not m:
        return None, None
    sec = m.group(1)
    line = re.search(r"^- manifest path / sha256:\s*(.*)$", sec, re.M)
    if not line:
        return None, None
    val = line.group(1).strip()
    sha = re.search(r"\b([0-9a-f]{64})\b", val)
    return (sha.group(1) if sha else None), val


def cmd_prereg(args):
    sha, raw = prereg_sha(args.thresholds)
    if sha is None:
        print(f"REFUSED: P0_THRESHOLDS section 8 has no manifest sha256 (value: {raw!r}). The corpus manifest must be "
              f"committed and its hash recorded before any G3 gate number exists.")
        return 3
    have = sha256_file(args.manifest)
    if have != sha:
        print(f"REFUSED: manifest {args.manifest} sha256 {have} != pre-registered {sha}")
        return 3
    print(f"OK: manifest sha256 {have} matches P0_THRESHOLDS section 8")
    return 0


def cmd_split(args):
    n = 0
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rid = json.loads(line)["id"]
            if (args.split == "dev") == is_dev(rid):
                print(rid)
                n += 1
    print(f"{n} {args.split} ids", file=sys.stderr)
    return 0


def load_run(run_dir):
    reqs = {}
    with open(os.path.join(run_dir, "requests.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                key = (r["id"], r["mode"])
                if key in reqs:
                    raise ValueError(f"duplicate request record: {key}")
                reqs[key] = r
    pos = {}
    with open(os.path.join(run_dir, "positions.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                pos.setdefault((p["id"], p["mode"]), []).append(p)
    meta = {}
    mp = os.path.join(run_dir, "meta.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    return reqs, pos, meta


def pool(rows):
    kl = sorted(r["kl"] for r in rows)
    nm = sorted(r["nmse"] for r in rows)
    n = len(rows)
    return {
        "n_positions": n,
        "mean_kl": (sum(kl) / n) if n else float("nan"),
        "p95_kl": nearest_rank(kl, 95.0),
        "top1": (sum(r["top1"] for r in rows) / n) if n else float("nan"),
        "nmse_median": median(nm),
        "nmse_p95": nearest_rank(nm, 95.0),
    }


def bucket_of(n_tokens):
    for name, lo, hi in BUCKETS:
        if lo <= n_tokens < hi:
            return name
    return "out-of-range"


def evaluate(reqs, pos, sel, naive, retention=None):
    ids = sorted({rid for (rid, m) in reqs if m == sel})
    out = {"selector_mode": sel, "naive_mode": naive, "n_requests": len(ids)}

    # harness sanity: every all-inject arm must reproduce the reference bit-for-bit
    sanity = [r for (rid, m), r in reqs.items() if m == "all-inject"]
    out["sanity_all_inject"] = {"n": len(sanity), "bit_identical": sum(1 for r in sanity if r.get("bit_identical_to_ref"))}
    out["sanity_ok"] = bool(ids) and len(sanity) == len(ids) and all(
        (rid, "all-inject") in reqs and reqs[(rid, "all-inject")].get("bit_identical_to_ref") is True
        for rid in ids
    )

    # A missing, duplicated or mismatched arm must not become a favorable pooled score.
    coverage = []
    for mode in (sel, naive, "all-inject"):
        if {rid for rid, m in reqs if m == mode} != set(ids):
            coverage.append(f"{mode}: request identities differ")
    if retention is not None and (not isinstance(retention, (int, float)) or
                                  isinstance(retention, bool) or not math.isfinite(retention)):
        coverage.append("invalid retention")
    for rid in ids:
        arm_keys = []
        for mode in (sel, naive, "all-inject"):
            request = reqs.get((rid, mode))
            rows = pos.get((rid, mode), [])
            if not request or not rows:
                coverage.append(f"{rid}/{mode}: missing request or position rows")
                continue
            keys = [(p.get("row"), p.get("pos")) for p in rows]
            if len(set(keys)) != len(keys) or request.get("n_rows") != len(rows):
                coverage.append(f"{rid}/{mode}: duplicate rows or row count mismatch")
            for p in rows:
                if any(not isinstance(p.get(k), (int, float)) or isinstance(p.get(k), bool) or
                       not math.isfinite(p[k]) for k in ("kl", "nmse", "top1")) or p.get("top1") not in (0, 1):
                    coverage.append(f"{rid}/{mode}: invalid metric")
                    break
            arm_keys.append(set(keys))
        if len(arm_keys) == 3 and not (arm_keys[0] == arm_keys[1] == arm_keys[2]):
            coverage.append(f"{rid}: arm positions differ")
    out["coverage_violations"] = coverage

    sel_rows = [p for rid in ids for p in pos.get((rid, sel), [])]
    naive_rows = [p for rid in ids for p in pos.get((rid, naive), [])]
    out["pooled"] = pool(sel_rows)
    out["naive_pooled"] = pool(naive_rows) if naive_rows else None
    nm, sm = (out["naive_pooled"] or {}).get("mean_kl"), out["pooled"]["mean_kl"]
    out["pooled"]["naive_ratio"] = (nm / sm if sm and sm > 0 else (float("inf") if nm else float("nan"))) if nm is not None else None
    out["pooled"]["retention"] = retention
    out["pooled"]["r_mean"] = (sum(reqs[(rid, sel)]["r"] for rid in ids) / len(ids)) if ids else float("nan")
    out["per_request_mean_kl"] = {rid: reqs[(rid, sel)]["mean_kl"] for rid in ids}

    out["verdicts"] = {k: check(k, out["pooled"].get(k)) for k in GATES}

    by_bucket = {}
    for rid in ids:
        b = bucket_of(reqs[(rid, sel)]["n_tokens"])
        by_bucket.setdefault(b, []).append(rid)
    out["buckets"] = {}
    for name, _, _ in BUCKETS + [("out-of-range", 0, 0)]:
        bids = by_bucket.get(name, [])
        if not bids:
            continue
        rows = [p for rid in bids for p in pos.get((rid, sel), [])]
        nrows = [p for rid in bids for p in pos.get((rid, naive), [])]
        st = pool(rows)
        nmb = pool(nrows)["mean_kl"] if nrows else None
        st["naive_ratio"] = (nmb / st["mean_kl"]) if (nmb is not None and st["mean_kl"] > 0) else None
        st["n_requests"] = len(bids)
        st["verdicts"] = {k: check(k, st.get(k)) for k in GATES if k != "retention"}
        out["buckets"][name] = st

    shape = []
    if len(ids) < CORPUS_MIN_REQUESTS:
        shape.append(f"{len(ids)} requests < {CORPUS_MIN_REQUESTS}")
    for name, _, _ in BUCKETS:
        n = len(by_bucket.get(name, []))
        if n < CORPUS_MIN_PER_BUCKET:
            shape.append(f"bucket {name}: {n} requests < {CORPUS_MIN_PER_BUCKET}")
    bad_chunks = [rid for rid in ids if not (CHUNKS_MIN <= reqs[(rid, sel)]["n_chunks"] <= CHUNKS_MAX)]
    if bad_chunks:
        shape.append(f"{len(bad_chunks)} requests outside {CHUNKS_MIN}-{CHUNKS_MAX} chunks")
    out["corpus_shape_violations"] = shape

    v = out["verdicts"]
    if not out["sanity_ok"] or out["coverage_violations"]:
        overall = "INVALID (missing, mismatched or invalid arm data or all-inject sanity failed)"
    elif any(x == "FAIL" for x in v.values()):
        overall = "FAIL"
    elif shape:
        overall = "INCOMPLETE (corpus shape: " + "; ".join(shape) + ")"
    elif any(x == "NOT COMPUTED" for x in v.values()):
        overall = "INCOMPLETE (one or more gate metrics not computed)"
    else:
        overall = "PASS"
    if retention is not None and RETENTION_OPT_IN <= retention < GATES["retention"][1] and overall == "FAIL":
        if all(x == "PASS" for k, x in v.items() if k != "retention"):
            overall = "OPT-IN ONLY (retention in [0.85, 0.95))"
    out["overall"] = overall
    return out


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf"
        return f"{x:.4g}"
    return str(x)


def render(res, label):
    p = res["pooled"]
    lines = [f"# G3 {label}: selector mode `{res['selector_mode']}` vs naive `{res['naive_mode']}`", "",
             f"requests {res['n_requests']}, scored positions {p['n_positions']}, mean r {fmt(p['r_mean'])}; "
             f"sanity all-inject bit-identical {res['sanity_all_inject']['bit_identical']}/{res['sanity_all_inject']['n']}", "",
             "| metric | value | gate | verdict |", "|---|---|---|---|"]
    for k, (op, thr) in GATES.items():
        lines.append(f"| {k} | {fmt(p.get(k))} | {op} {thr} | {res['verdicts'][k]} |")
    lines += ["", "| bucket | requests | positions | mean KL | p95 KL | top-1 | NMSE med | NMSE p95 | naive ratio |", "|---|---|---|---|---|---|---|---|---|"]
    for name, st in res["buckets"].items():
        lines.append(f"| {name} | {st['n_requests']} | {st['n_positions']} | {fmt(st['mean_kl'])} | {fmt(st['p95_kl'])} | {fmt(st['top1'])} | "
                     f"{fmt(st['nmse_median'])} | {fmt(st['nmse_p95'])} | {fmt(st['naive_ratio'])} |")
    if res["corpus_shape_violations"]:
        lines += ["", "corpus shape: " + "; ".join(res["corpus_shape_violations"])]
    lines += ["", f"OVERALL: {res['overall']}"]
    return "\n".join(lines)


def bound_cohort(manifest, reqs, pos, selector, naive):
    """Bind every native arm to the frozen held-out cohort, not merely to a manifest hash."""
    expected = {r["id"]: r for r in manifest["requests"] if r["split"] == "gate"}
    for mode in (selector, naive, "all-inject"):
        got = {rid for rid, arm in reqs if arm == mode}
        if got != set(expected):
            raise ValueError(f"{mode}: result ids do not equal frozen gate cohort")
        for rid, declared in expected.items():
            row = reqs[(rid, mode)]
            if row.get("n_tokens") != declared["n_tokens"] or row.get("n_chunks") != declared["n_chunks"]:
                raise ValueError(f"{rid}/{mode}: result shape differs from frozen cohort")
            n = declared["n_tokens"] - declared["suffix_start"] + 64
            if row.get("n_cont") != 64 or row.get("n_rows") != n:
                raise ValueError(f"{rid}/{mode}: missing declared suffix/64 continuation rows")
            actual = [(p.get("row"), p.get("pos")) for p in pos.get((rid, mode), [])]
            if sorted(actual) != [(i, declared["suffix_start"] + i) for i in range(n)]:
                raise ValueError(f"{rid}/{mode}: position coverage differs from frozen corpus")
    if any(rid not in expected or mode not in (selector, naive, "all-inject") for rid, mode in pos):
        raise ValueError("position rows contain undeclared request or mode")
    return expected


def _load_pinned_scorer(path, expected_sha):
    """Execute the exact digest-pinned project scorer bytes, never a path substitute."""
    if not path:
        raise ValueError("frozen text tasks require --scorer-module")
    with open(path, "rb") as source:
        data = source.read()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("scorer module digest mismatch")
    module = types.ModuleType("_kpf62_pinned_debug_scorer")
    module.__file__ = os.path.abspath(path)
    exec(compile(data, module.__file__, "exec"), module.__dict__)
    if not callable(getattr(module, "score_answer", None)):
        raise ValueError("pinned scorer has no score_answer")
    return module.score_answer


def paired_retention(path, tasks_path, expected, scorer_module=None):
    """Score paired raw outputs against each frozen task and pinned scorer."""
    tasks = g3_manifest.records(tasks_path)
    pairs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("task outcome is not an object")
            key = (row.get("id"), row.get("arm"))
            if key in pairs or key[0] not in expected or key[1] not in ("full", "reuse"):
                raise ValueError("duplicate or unexpected task outcome")
            task = tasks[key[0]]
            if task.get("metric") == "exact_token_match":
                output = row.get("output_tokens")
                if not isinstance(output, list) or any(type(x) is not int or x < 0 for x in output):
                    raise ValueError("missing or invalid task output tokens")
            elif task.get("metric") == "orchestrator_text_v1":
                output = row.get("output_text")
                if not isinstance(output, str):
                    raise ValueError("missing or invalid task output text")
            else:
                raise ValueError("unsupported task metric")
            if row.get("error") is not None or row.get("complete") is not True:
                raise ValueError("task output is incomplete or failed")
            pairs[key] = output
    required = {(rid, arm) for rid in expected for arm in ("full", "reuse")}
    if set(pairs) != required:
        raise ValueError("paired task outcome coverage is incomplete")
    ids = sorted(expected)
    text_hashes = {tasks[rid]["scorer_sha256"] for rid in ids
                   if tasks[rid].get("metric") == "orchestrator_text_v1"}
    if len(text_hashes) > 1:
        raise ValueError("multiple scorer module digests in one cohort")
    scorer = _load_pinned_scorer(scorer_module, next(iter(text_hashes))) if text_hashes else None
    def score(rid, arm):
        task = tasks[rid]
        if task["metric"] == "exact_token_match":
            return int(pairs[(rid, arm)] == task["target_tokens"])
        if task.get("scoring_method") not in ("substring", "exact_match", "f1"):
            raise ValueError("unsupported or non-deterministic scoring method")
        try:
            value = scorer(pairs[(rid, arm)], task["expected"], task["scoring_method"], task["scoring_config"])
        except Exception as exc:
            raise ValueError(f"{rid}/{arm}: canonical scorer failed") from exc
        if type(value) is not bool:
            raise ValueError("canonical scorer did not return a boolean")
        return int(value)
    full = [score(rid, "full") for rid in ids]
    reuse = [score(rid, "reuse") for rid in ids]
    total = sum(full)
    if not total:
        raise ValueError("full-recompute score is zero; retention is undefined")
    ratio = sum(reuse) / total
    # Paired bootstrap is reported, not gated (P0 §4). Preserve undefined resamples.
    rng = random.Random(62026)
    estimates, undefined = [], 0
    for _ in range(10000):
        sample = [rng.randrange(len(ids)) for _ in ids]
        a, b = sum(full[i] for i in sample), sum(reuse[i] for i in sample)
        if a:
            estimates.append(b / a)
        else:
            undefined += 1
    estimates.sort()
    return {"score_full": total / len(ids), "score_reuse": sum(reuse) / len(ids),
            "retention": ratio, "n_tasks": len(ids), "bootstrap_seed": 62026,
            "bootstrap_samples": 10000, "bootstrap_undefined": undefined,
            "scorer_sha256": next(iter(text_hashes)) if text_hashes else None,
            "item_scores": [{"id": rid, "full": full[i], "reuse": reuse[i],
                             "metric": tasks[rid]["metric"]} for i, rid in enumerate(ids)],
            "bootstrap_95": [nearest_rank(estimates, 2.5), nearest_rank(estimates, 97.5)] if estimates else None}


def cmd_gate(args):
    label = "DEV (not a gate result)" if args.dev else "GATE"
    if not args.dev:
        if not args.manifest:
            print("REFUSED: a gate run needs --manifest (pre-registration check); use --dev for a non-gating read")
            return 3
        rc = cmd_prereg(args)
        if rc != 0:
            return rc
    reqs, pos, meta = load_run(args.run)
    retention, score = None, None
    if not args.dev:
        if not args.corpus or not args.tasks or not args.retention:
            print("REFUSED: --corpus, --tasks and per-item --retention are required for GATE")
            return 3
        try:
            manifest = g3_manifest.verify(args.manifest, args.corpus, args.tasks)
            if meta.get("n_cont") != 64:
                raise ValueError("native run did not declare 64 continuation positions")
            expected = bound_cohort(manifest, reqs, pos, args.selector, args.naive)
            score = paired_retention(args.retention, args.tasks, expected, getattr(args, "scorer_module", None))
            retention = score["retention"]
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"REFUSED: invalid frozen cohort or paired outcomes: {exc}")
            return 3
    elif args.retention:
        print("REFUSED: unbound aggregate task retention is not accepted; use a frozen GATE cohort")
        return 3
    res = evaluate(reqs, pos, args.selector, args.naive, retention)
    res["task_scores"] = score
    res["label"] = label
    res["meta"] = meta
    if args.manifest:
        res["manifest_sha256"] = sha256_file(args.manifest)
    text = render(res, label)
    print(text)
    if args.out:
        with open(args.out + ".json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        with open(args.out + ".md", "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if args.dev:
        return 0
    return 0 if res["overall"] == "PASS" else 1


def cmd_selftest(_args):
    fails = []

    def expect(cond, what):
        if not cond:
            fails.append(what)

    expect(nearest_rank([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95.0) == 10, "p95 nearest-rank of 1..10 is 10")
    expect(nearest_rank([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50.0) == 5, "median nearest-rank of 1..10 is 5")
    expect(nearest_rank(list(range(1, 101)), 95.0) == 95, "p95 nearest-rank of 1..100 is 95")
    expect(is_dev("x") == (int(hashlib.sha256(b"x").hexdigest(), 16) % 5 == 0), "split rule")

    with tempfile.TemporaryDirectory() as d:
        # 2 requests, selector mean KL 0.01, naive 0.1 -> ratio 10; all-inject bit-identical
        with open(os.path.join(d, "requests.jsonl"), "w") as f:
            for rid, n in (("a", 2000), ("b", 5000)):
                for mode, kl in (("head", 0.01), ("none", 0.1), ("all-inject", 0.0)):
                    f.write(json.dumps({"id": rid, "mode": mode, "n_tokens": n, "n_chunks": 3, "r": 0.2 if mode == "head" else 0.0,
                                        "mean_kl": kl, "n_rows": 10, "bit_identical_to_ref": True}) + "\n")
        with open(os.path.join(d, "positions.jsonl"), "w") as f:
            for rid in ("a", "b"):
                for k in range(10):
                    f.write(json.dumps({"id": rid, "mode": "head", "row": k, "pos": k, "kl": 0.01, "top1": 1, "nmse": 0.001}) + "\n")
                    f.write(json.dumps({"id": rid, "mode": "none", "row": k, "pos": k, "kl": 0.1, "top1": 1, "nmse": 0.01}) + "\n")
                    f.write(json.dumps({"id": rid, "mode": "all-inject", "row": k, "pos": k, "kl": 0.0, "top1": 1, "nmse": 0.0}) + "\n")
        reqs, pos, _ = load_run(d)
        res = evaluate(reqs, pos, "head", "none", retention=None)
        expect(abs(res["pooled"]["mean_kl"] - 0.01) < 1e-12, "pooled mean KL")
        expect(abs(res["pooled"]["naive_ratio"] - 10.0) < 1e-9, "naive ratio 10")
        expect(res["verdicts"]["mean_kl"] == "PASS" and res["verdicts"]["retention"] == "NOT COMPUTED", "verdicts")
        expect(res["overall"].startswith("INCOMPLETE"), "2 requests is an incomplete corpus")
        expect(set(res["buckets"]) == {"1k-4k", "4k-8k"}, "buckets")
        res2 = evaluate(reqs, pos, "head", "none", retention=0.9)
        expect(res2["verdicts"]["retention"] == "FAIL", "retention 0.9 fails the 0.95 gate")

        del reqs[("a", "all-inject")]
        expect(evaluate(reqs, pos, "head", "none", retention=1)["overall"].startswith("INVALID"), "missing sanity arm refuses")
        reqs[("a", "all-inject")] = reqs[("b", "all-inject")].copy()
        pos[("a", "none")].pop()
        expect(evaluate(reqs, pos, "head", "none", retention=1)["overall"].startswith("INVALID"), "missing position refuses")

        # thresholds file parsing
        tp = os.path.join(d, "P0.md")
        open(tp, "w").write("## 8. Corpus manifest\n\n- manifest path / sha256: TBD\n\n## 9. Amendments\n")
        expect(prereg_sha(tp)[0] is None, "TBD -> no sha")
        mf = os.path.join(d, "m.json")
        open(mf, "w").write("{}\n")
        sha = sha256_file(mf)
        open(tp, "w").write(f"## 8. Corpus manifest\n\n- manifest path / sha256: /x/m.json / {sha}\n\n## 9. Amendments\n")
        expect(prereg_sha(tp)[0] == sha, "sha parsed")

    if fails:
        for x in fails:
            print("SELFTEST FAIL:", x)
        return 1
    print("g3_gate selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split"); s.add_argument("--corpus", required=True); s.add_argument("--split", choices=["dev", "gate"], required=True)
    s = sub.add_parser("prereg"); s.add_argument("--thresholds", required=True); s.add_argument("--manifest", required=True)
    s = sub.add_parser("gate")
    s.add_argument("--run", required=True)
    s.add_argument("--thresholds", default="/mnt/raid0/llm/tmp/cacheclip-port-20261021/P0_THRESHOLDS.md")
    s.add_argument("--manifest")
    s.add_argument("--selector", default="head")
    s.add_argument("--naive", default="none")
    s.add_argument("--corpus", help="frozen tokenized corpus JSONL matching the manifest")
    s.add_argument("--tasks", help="frozen task target JSONL matching the manifest")
    s.add_argument("--retention", help="paired per-item full/reuse raw output JSONL")
    s.add_argument("--scorer-module", help="digest-pinned canonical orchestrator debug_scorer.py for frozen text tasks")
    s.add_argument("--dev", action="store_true")
    s.add_argument("--out", help="write OUT.json and OUT.md")
    sub.add_parser("selftest")
    args = ap.parse_args()
    return {"split": cmd_split, "prereg": cmd_prereg, "gate": cmd_gate, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
