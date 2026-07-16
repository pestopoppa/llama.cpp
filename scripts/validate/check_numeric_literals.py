#!/usr/bin/env python3
"""Report numeric literals introduced in changed source files."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hh",
    ".cu",
    ".m",
    ".mm",
    ".py",
    ".sh",
}
NUMERIC_RE = re.compile(r"(?<![A-Za-z_])(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?:[uUlLfF]*)")


def _run_git(args: list[str], root: Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git command failed")
    return proc.stdout


def _changed_files(root: Path, base: str) -> list[Path]:
    output = _run_git(["diff", "--name-only", "--diff-filter=ACMR", base, "--"], root)
    paths: list[Path] = []
    for raw in output.splitlines():
        path = root / raw
        if path.exists() and path.suffix in SOURCE_SUFFIXES:
            paths.append(path)
    return sorted(paths)


def _scan_file(path: Path, root: Path) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#")):
            continue
        for match in NUMERIC_RE.finditer(line):
            literal = match.group(0)
            if literal in {"0", "1", "2"}:
                continue
            findings.append({"line": lineno, "literal": literal, "text": stripped[:160]})
    return {"path": str(path.relative_to(root)), "count": len(findings), "findings": findings[:50]}


def build_report(root: Path, base: str, max_new_literals: int | None) -> dict[str, object]:
    files = _changed_files(root, base)
    file_reports = [_scan_file(path, root) for path in files]
    total = sum(int(row["count"]) for row in file_reports)
    over_limit = max_new_literals is not None and total > max_new_literals
    return {
        "schema_version": "numeric_literal_audit.v1",
        "base": base,
        "files_scanned": len(files),
        "numeric_literals": total,
        "limit": max_new_literals,
        "ok": not over_limit,
        "files": file_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="Git base ref for changed-file scan.")
    parser.add_argument("--max-new-literals", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    root = Path(_run_git(["rev-parse", "--show-toplevel"], Path.cwd()).strip())
    try:
        report = build_report(root, args.base, args.max_new_literals)
    except RuntimeError as exc:
        print(f"check_numeric_literals: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"numeric literal audit: {report['numeric_literals']} literal(s) "
            f"across {report['files_scanned']} changed source file(s)"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
