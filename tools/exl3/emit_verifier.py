#!/usr/bin/env python3
"""Run build/bin/test-exl3 and seal its propositions as epyc.exl3.verifier.v1 rows.

Uses the research repo's single prospective writer (scripts/kernel_rnd/exl3/evidence.py,
producer epyc.exl3.evidence/v1) so the rows project through root's existing `exl3-verifier`
Vidya adapter. A failed proposition is written as a failed verifier row, never dropped.

usage: emit_verifier.py --output <fresh dir> [--build build] [--research /workspace/repos/epyc-inference-research]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--build", type=Path, default=TREE / "build")
    ap.add_argument("--research", type=Path, default=Path("/workspace/repos/epyc-inference-research"))
    ap.add_argument("--taskset", default="176-183")
    args = ap.parse_args()

    sys.path.insert(0, str(args.research))
    from scripts.kernel_rnd.exl3 import evidence, contract  # noqa: E402

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixtures = args.research / "scripts/kernel_rnd/exl3/fixtures"
    binary = (args.build / "bin/test-exl3").resolve()
    lib = (args.build / "bin/libggml-cpu.so.0").resolve()
    src = (TREE / "ggml/src/ggml-cpu/exl3.cpp").resolve()
    result_path = out / "test-exl3.json"
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    cmd = ["nice", "-n", "19", "taskset", "-c", args.taskset, str(binary), "--json", str(result_path), str(fixtures)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (out / "test-exl3.stdout").write_text(proc.stdout + proc.stderr)
    res = json.loads(result_path.read_text())

    commit = subprocess.run(["git", "-C", str(TREE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(TREE), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip() != ""
    gcc = subprocess.run(["/usr/bin/gcc-15", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    date = datetime.now(timezone.utc).isoformat()
    run_id = "exl3-llama-cpu-" + date
    host = {"platform": platform.platform(), "cpu": cpu, "isa": res["isa"], "pid": os.getpid(),
            "backend": "llama_cpp_cpu", "inference": False, "taskset": args.taskset,
            "tree": str(TREE), "commit": commit, "dirty": dirty, "rc": proc.returncode}
    host_path = out / "execution-identity.json"
    with host_path.open("xb") as stream:
        stream.write(contract.canonical(host))
    checker = Path(__file__).resolve()
    identities = {
        "source": {"id": f"llama.cpp experimental/exl3-cpu-20261003@{commit}{'+dirty' if dirty else ''}",
                   "sha256": evidence.file_hash(src)},
        "binary": {"id": str(binary), "sha256": evidence.file_hash(binary)},
        "library": {"id": str(lib), "sha256": evidence.file_hash(lib)},
        "toolchain": {"id": gcc, "sha256": evidence.digest(gcc)},
        "hardware": {"id": cpu, "sha256": evidence.digest({"cpu": cpu, "isa": res["isa"]})},
        "residency": {"id": f"CPU process (taskset {args.taskset})", "sha256": evidence.file_hash(host_path)},
    }
    n = 0
    failed = 0
    for p in res["props"]:
        fx = fixtures / p["fixture"]
        if fx.is_dir():
            man = json.loads((fx / "manifest.json").read_text())
            fixture_file = fx / "fixture.json"
            model = {"id": man["source"]["repository"] + "@" + man["source"]["revision"], "sha256": evidence.digest(man["source"])}
            artifact = {"id": man["matrices"][0]["id"], "sha256": man["artifact_sha256"]}
            inputs = [fx / "manifest.json", fixture_file, fx / "trellis.bin", fx / "suh.bin", fx / "svh.bin"]
        else:
            fixture_file = result_path
            model = {"id": "synthetic random trellis (test-exl3 seed 1234)", "sha256": evidence.digest("test-exl3-seed-1234")}
            artifact = {"id": "test-exl3 synthetic suite", "sha256": evidence.file_hash(TREE / "tests/test-exl3.cpp")}
            inputs = [TREE / "tests/test-exl3.cpp"]
        inputs += [result_path, host_path, binary, lib, src, checker]
        read_set = [{"path": str(Path(x).resolve()), "sha256": evidence.file_hash(x)} for x in inputs]
        claim = p["proposition"] + f" [value {p['value']:.6g}, envelope {p['envelope']:.6g}]"
        row = dict(schema=evidence.VERIFIER, run_id=run_id, date=date, category="CANDIDATE", protocol_id="",
                   protocol_eligible=False, arm="llama-cpu-exl3", comparator="materialized_weight_fp32_fma_v1",
                   identities=dict(identities, model=model, artifact=artifact), claim=claim, backend="llama_cpp_cpu",
                   fixture=p["fixture"], path=p["path"], decided_proposition=claim,
                   verdict="pass" if p["pass"] else "fail",
                   checker={"id": "epyc.exl3.llama_cpu.emit_verifier/v1", "path": str(checker), "sha256": evidence.file_hash(checker)},
                   fixture_sha256=evidence.file_hash(fixture_file), read_set=read_set, read_set_sha256=evidence.digest(read_set))
        evidence.write(out, row)
        n += 1
        failed += not p["pass"]
    print(json.dumps({"rows": n, "failed": failed, "test_rc": proc.returncode, "output": str(out)}, sort_keys=True))
    return 1 if failed or proc.returncode else 0


if __name__ == "__main__":
    sys.exit(main())
