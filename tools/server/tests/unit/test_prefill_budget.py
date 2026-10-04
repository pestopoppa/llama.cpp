import os
import re
import tempfile
import threading

import pytest
from utils import *

# --prefill-budget-decoding / --prefill-budget-target-ms: while any slot is generating, cap the prompt tokens
# added per server iteration; with no slot generating, prompts still fill n_batch. Chunking the prompt must not
# change any output: greedy completions are compared against the same requests run alone.

server: ServerProcess

DECODE_PROMPT = "Once upon a time there was a little girl who"
LONG_PROMPT = " ".join([
    "The knight rode across the mountains and rivers to find the golden sword hidden in the enchanted forest.",
    "He met dragons and fairies and wizards who helped him on his noble quest to save the kingdom.",
] * 6)

ITER_RE = re.compile(r"prefill-budget: iter \d+: decode_rows = (\d+) \(slots (\d+)\), prompt = (\d+) \(slots (\d+)\), budget = (\d+)")


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 2048
    server.n_batch = 128
    server.n_slots = 2
    server.kv_unified = True
    server.temperature = 0.0
    server.n_predict = 64
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


def solo(prompt: str, n_predict: int) -> str:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": False,
    })
    assert res.status_code == 200
    return res.body["content"]


def decode_with_neighbour_prefill(n_decode: int, n_long: int) -> tuple[str, str]:
    """slot A starts decoding; once it has streamed a token, slot B sends a long prompt"""
    generating = threading.Event()
    out = {}

    def decoder():
        content = ""
        for data in server.make_stream_request("POST", "/completion", data={
            "prompt": DECODE_PROMPT, "n_predict": n_decode, "temperature": 0.0, "stream": True,
            "cache_prompt": False,
        }):
            if data.get("content"):
                content += data["content"]
                generating.set()
        out["decode"] = content

    t = threading.Thread(target=decoder)
    t.start()
    assert generating.wait(60)
    out["long"] = solo(LONG_PROMPT, n_long)
    t.join()
    return out["decode"], out["long"]


def iter_rows():
    with open(server.log_path) as f:
        return [tuple(int(x) for x in m.groups()) for m in ITER_RE.finditer(f.read())]


@pytest.mark.parametrize("budget,target_ms", [
    (None, None),
    (8, None),
    (None, 1),
    (16, 1),
])
def test_prefill_budget_greedy_identity(budget, target_ms):
    global server
    server.prefill_budget_decoding = budget
    server.prefill_budget_target_ms = target_ms
    server.start()

    ref_decode = solo(DECODE_PROMPT, 400)
    ref_long = solo(LONG_PROMPT, 16)

    got_decode, got_long = decode_with_neighbour_prefill(400, 16)
    assert got_decode == ref_decode
    assert got_long == ref_long

    rows = iter_rows()
    mixed = [r for r in rows if r[0] > 0 and r[2] > 0]   # decode rows + prompt tokens in the same iteration
    solo_rows = [r for r in rows if r[0] == 0]
    assert len(mixed) > 0, "the neighbour prefill never shared an iteration with the decoding slot"

    # solo prefill keeps the full n_batch
    assert max(r[2] for r in solo_rows) == 128
    assert all(r[4] == 0 for r in solo_rows)

    if budget is None and target_ms is None:
        # today's behaviour: the neighbour's prompt fills the batch up to n_batch
        assert all(r[4] == 0 for r in mixed)
        assert max(r[2] for r in mixed) > 16
    elif budget is not None:
        # a fixed budget binds in every mixed iteration (the adaptive floor, 64, is above it)
        assert all(r[4] == budget for r in mixed)
        assert all(r[2] <= budget for r in mixed)
    else:
        # adaptive only: when it binds, it is at least its floor and below n_batch, and it is respected
        for r in mixed:
            assert r[4] == 0 or (64 <= r[4] < 128 and r[2] <= r[4])
