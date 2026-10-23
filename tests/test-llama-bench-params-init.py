#!/usr/bin/env python3
"""Hardware-free structural contract: llama-bench never parses indeterminate scalars.

Regression for the 2026-08-11 defect (0492c2319): ``parse_cmd_params()``
declared ``cmd_params params;`` (an aggregate with no default member
initialisers) and copied only some scalars from ``cmd_params_defaults``.
``autokernel_harden`` was therefore indeterminate and roughly half of all
llama-bench processes ran the hardened protocol without ``--autokernel-harden``.

The defect is undefined behaviour, so a runtime test can pass by luck (a zeroed
stack frame).  This test checks the source instead, deterministically:

1. ``parse_cmd_params`` value-initialises ``params`` (``cmd_params params{};``);
2. every non-vector field of ``struct cmd_params`` is assigned from
   ``cmd_params_defaults`` before the argv loop;
3. the struct is never pre-seeded wholesale from the defaults (that would merge
   user-supplied vector values with the defaults), and every vector field is
   filled from the defaults only when empty.
"""
import re
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tools/llama-bench/llama-bench.cpp"


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE
    text = source.read_text(encoding="utf-8")

    m = re.search(r"^struct cmd_params \{\n(.*?)^\};", text, re.S | re.M)
    if not m:
        raise SystemExit("struct cmd_params not found")
    fields = []
    for line in m.group(1).splitlines():
        line = line.split("//")[0].strip()
        if not line:
            continue
        fm = re.match(r"(.+?)\s+(\w+);$", line)
        if not fm:
            raise SystemExit(f"unparsed cmd_params member: {line!r}")
        fields.append((fm.group(1).strip(), fm.group(2)))
    scalars = [name for typ, name in fields if not typ.startswith("std::vector<")]
    vectors = [name for typ, name in fields if typ.startswith("std::vector<")]
    if "autokernel_harden" not in scalars or len(vectors) < 10:
        raise SystemExit(f"cmd_params parse looks wrong: scalars={scalars}")

    start = text.index("static cmd_params parse_cmd_params(int argc, char ** argv) {")
    loop = text.index("for (int i = 1; i < argc; i++)", start)
    prologue = text[start:loop]
    end = text.index("\nstatic ", loop)
    body = text[start:end]

    errors = []
    if not re.search(r"\bcmd_params\s+params\s*\{\s*\}\s*;", prologue):
        errors.append("parse_cmd_params must value-initialise: `cmd_params params{};`")
    for name in scalars:
        pat = rf"\bparams\.{name}\s*=\s*cmd_params_defaults\.{name}\s*;"
        if not re.search(pat, prologue):
            errors.append(f"scalar `{name}` is not copied from cmd_params_defaults before the argv loop")
    if re.search(r"\bparams\s*=\s*cmd_params_defaults\s*;", body) or \
       re.search(r"\bcmd_params\s+params\s*=\s*cmd_params_defaults\b", body):
        errors.append("params must not be pre-seeded wholesale: vector fields are append-then-fill-if-empty")
    for name in vectors:
        if name in ("hf_repo", "hf_file"):
            continue  # no defaults by design
        pat = rf"if \(params\.{name}\.empty\(\)\)\s*\{{\s*params\.{name}\s*=\s*cmd_params_defaults\.{name};"
        if not re.search(pat, body):
            errors.append(f"vector `{name}` has no fill-if-empty from cmd_params_defaults")
    if errors:
        raise SystemExit("llama-bench parse_cmd_params init contract violated:\n  " + "\n  ".join(errors))
    print(f"ok: {len(scalars)} scalars copied, {len(vectors)} vectors fill-if-empty, params value-initialised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
