#!/usr/bin/env python3
"""Emit a GLM-5.3 native-MTP CPU serve plan; never execute it."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any

EXACT_ENV = {
    "PATH": "/usr/bin:/bin",
    "OMP_PROC_BIND": "spread",
    "OMP_PLACES": "cores",
    "OMP_WAIT_POLICY": "active",
    "OMP_DYNAMIC": "false",
    "GGML_IQK": "1",
    "GGML_FUSED_DECODE_OFF": "1",
    "GGML_FA_SPLIT_KV": "0",
    "GGML_NOHUGEPAGE_PROCESS": "1",
}
PREFIX = ["taskset", "-c", "0-95", "numactl", "--interleave=all"]


def verify_recipe_source(path: Path | str) -> dict[str, Any]:
    """Read constants from the cited recipe and refuse a transcription drift."""
    source_path = Path(path).expanduser().resolve(strict=True)
    raw = source_path.read_bytes()
    tree = ast.parse(raw, filename=str(source_path))
    values: dict[str, Any] = {}

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, (ast.List, ast.Tuple)): return [evaluate(item) for item in node.elts]
        if isinstance(node, ast.Dict): return {evaluate(key): evaluate(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.Name): return values[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str" and len(node.args) == 1:
            return str(evaluate(node.args[0]))
        raise ValueError(f"unsupported recipe expression: {ast.dump(node)}")

    wanted = {"THREADS", "CONTEXT", "PARALLEL_SLOTS", "SERVE_PREFIX", "CANONICAL_OMP_ENV",
              "CHAMPION_GGML_ENV", "BASE_SERVER_FLAGS", "FLASH_ATTN_FLAGS", "KV_CACHE_FLAGS"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                values[name] = evaluate(node.value)
    missing = sorted(wanted - values.keys())
    if missing: raise ValueError(f"cited recipe lacks parseable constants: {missing}")
    expected_base = ["--no-webui", "-np", "1", "-c", "8192", "-t", "48", "--no-mmap", "-lv", "4"]
    comparisons = {
        "SERVE_PREFIX": (values["SERVE_PREFIX"], PREFIX),
        "CANONICAL_OMP_ENV": (values["CANONICAL_OMP_ENV"], {k: EXACT_ENV[k] for k in ("OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "OMP_DYNAMIC")}),
        "CHAMPION_GGML_ENV": (values["CHAMPION_GGML_ENV"], {k: EXACT_ENV[k] for k in ("GGML_IQK", "GGML_FUSED_DECODE_OFF", "GGML_FA_SPLIT_KV", "GGML_NOHUGEPAGE_PROCESS")}),
        "BASE_SERVER_FLAGS": (values["BASE_SERVER_FLAGS"], expected_base),
        "FLASH_ATTN_FLAGS": (values["FLASH_ATTN_FLAGS"], ["-fa", "on"]),
        "KV_CACHE_FLAGS": (values["KV_CACHE_FLAGS"], ["-ctk", "f16", "-ctv", "f16"]),
    }
    drift = [name for name, (got, expected) in comparisons.items() if got != expected]
    if drift: raise ValueError(f"copied serve constants drift from cited recipe: {drift}")
    return {"path": str(source_path), "sha256": hashlib.sha256(raw).hexdigest(), "constants_verified": sorted(comparisons)}


def build_plan(
    binary: Path | str,
    model: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = 18497,
    context: int = 8192,
    threads: int = 48,
    draft_max: int = 3,
    draft_p_min: float = 0.0,
    mtp: bool = True,
    evidence_trace: bool = False,
    reasoning_off: bool = False,
    recipe_source: Path | str | None = None,
) -> dict[str, Any]:
    binary = Path(binary).expanduser().resolve(strict=False)
    model = Path(model).expanduser().resolve(strict=False)
    if binary.name != "llama-server":
        raise ValueError("--binary must name llama-server")
    if model.suffix != ".gguf":
        raise ValueError("--model must name a .gguf file")
    if not 1 <= port <= 65535:
        raise ValueError("port must be in 1..65535")
    if min(context, threads, draft_max) < 1:
        raise ValueError("context, threads, and draft-max must be positive")
    if not 0.0 <= draft_p_min <= 1.0:
        raise ValueError("draft-p-min must be in [0,1]")

    env = {**EXACT_ENV, "LD_LIBRARY_PATH": str(binary.parent)}
    provenance = verify_recipe_source(recipe_source) if recipe_source is not None else {"status": "not_supplied"}
    argv = PREFIX + [
        str(binary),
        "--no-webui",
        "-np", "1",
        "-c", str(context),
        "-t", str(threads),
        "--no-mmap",
        "-lv", "4",
        "--device", "none",
        "-ngl", "0",
        "-fa", "on",
        "-ctk", "f16",
        "-ctv", "f16",
        "--host", host,
        "--port", str(port),
        "-m", str(model),
    ]
    if mtp:
        argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_max),
                 "--spec-draft-p-min", format(draft_p_min, "g")]
    if evidence_trace:
        env["LLAMA_TRACE"] = "1"
    if reasoning_off:
        argv += ["--reasoning", "off"]
    if "-md" in argv or "--model-draft" in argv:
        raise AssertionError("GLM-5.3 must use its embedded NextN head")
    shell = "env -i " + " ".join(
        [f"{key}={shlex.quote(value)}" for key, value in env.items()]
        + [shlex.quote(token) for token in argv]
    )
    result: dict[str, Any] = {
        "schema": "glm53-native-mtp-serve-plan-v1",
        "execution": "not_executed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "binary": str(binary),
        "model": str(model),
        "environment": env,
        "argv": argv,
        "shell_preview": shell,
        "contract": {
            "text_only": True,
            "cpu_only": True,
            "embedded_nextn_available": True,
            "native_mtp_enabled": mtp,
            "separate_draft_model": False,
            "spec_type": "draft-mtp" if mtp else "none (matched plain diagnostic)",
        },
        "execution_preconditions": {
            "operator_task_authority": "required before launch",
            "physical_region_lock": "required before launch for region 0-95",
            "candidate_build": "binary must exist and be built from the intended candidate tip",
            "ggml_linkage": "verify candidate-local linkage before launch",
            "shell_preview_scope": "inner command only; generation grants no authority and starts no process",
        },
        "derivation": {
            "base_recipe": "epyc-inference-research/data/inf70-prod1-recipe-draft-2026-09-08/lib/qwen38_flash_next_recipe.py",
            "base_recipe_fields": ["SERVE_PREFIX", "CANONICAL_OMP_ENV", "CHAMPION_GGML_ENV", "BASE_SERVER_FLAGS", "FLASH_ATTN_FLAGS", "KV_CACHE_FLAGS"],
            "preserved": "CPU affinity/NUMA order, no-mmap, 48 threads, 8192 context, one slot, f16 KV, FA on, exact OMP/GGML launch environment",
            "glm_specific": "embedded draft-mtp omits -md; serve draft-max=3 and p-min=0 are unoptimized correctness-first defaults",
            "validation_parameters": "acceptance/performance validation separately sweeps draft-max 1,3,5; draft-max 5 is not the default serve value",
            "identity_policy": "binary and model are required caller inputs; no stale champion binary/build/digest is copied",
            "recipe_source": provenance,
            "runtime_evidence_overrides": {"LLAMA_TRACE": "1 enables source-gated accepted A/D records"} if evidence_trace else {},
            "reasoning_mode": "off via server flag for consistently formatted useful output" if reasoning_off else "recipe default",
        },
    }
    stable = {key: value for key, value in result.items() if key != "shell_preview"}
    result["plan_sha256"] = hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18497)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--draft-max", type=int, default=3)
    parser.add_argument("--draft-p-min", type=float, default=0.0)
    parser.add_argument("--plain-diagnostic", action="store_true", help="emit matched no-spec control; default remains native MTP")
    parser.add_argument("--evidence-trace", action="store_true", help="add LLAMA_TRACE=1 for per-verification accepted A/D records")
    parser.add_argument("--reasoning-off", action="store_true", help="disable template thinking consistently across matched arms")
    parser.add_argument("--recipe-source", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = build_plan(args.binary, args.model, host=args.host, port=args.port,
                      context=args.context, threads=args.threads,
                      draft_max=args.draft_max, draft_p_min=args.draft_p_min,
                      mtp=not args.plain_diagnostic, evidence_trace=args.evidence_trace,
                      reasoning_off=args.reasoning_off, recipe_source=args.recipe_source)
    rendered = json.dumps(plan, indent=2) + "\n"
    if args.output: args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
