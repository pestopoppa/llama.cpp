#!/usr/bin/env python3
"""Hardware-free structural contract for the AutoKernel bench handshake.

This test deliberately does not execute llama-bench or initialize a backend.
It protects the only ordering guarantee the discovery runner relies on: model
load/context construction and warmups happen before ``ready``; the first timed
clock begins only after the token-bound ``continue`` acknowledgement.
"""
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "tools/llama-bench/llama-bench.cpp"


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    required = (
        '"--autokernel-ready-file"',
        '"--autokernel-continue-file"',
        '"--autokernel-ready-token"',
        "O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW",
        "O_RDONLY | O_CLOEXEC | O_NOFOLLOW",
        "epyc.autokernel.ready_continue.v1",
        "autokernel_write_ready_and_wait(params)",
        "--autokernel-ready-* handshake failed before timed repetitions",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"missing ready/continue contract tokens: {missing}")
    barrier = text.index("autokernel_write_ready_and_wait(params)")
    first_timed_clock = text.index("uint64_t t_start = get_time_ns()")
    if barrier >= first_timed_clock:
        raise SystemExit("ready/continue barrier is not before the first timed repetition")
    warmup = text.index("// Every context gets different warmup content.")
    if barrier <= warmup:
        raise SystemExit("ready/continue barrier is not after the warmup phase")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
