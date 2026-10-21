import os
import pytest
from utils import *

# Opt-in reorder planner (chunk-KV reuse P1a). Needs a tiny local model: LLAMA_TEST_MODEL=/path/stories15M-q4_0.gguf
MODEL = os.environ.get("LLAMA_TEST_MODEL")

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    if not MODEL:
        pytest.skip("LLAMA_TEST_MODEL not set")
    server = ServerPreset.tinyllama2()
    server.model_file = MODEL
    server.model_hf_repo = None
    server.model_hf_file = None
    server.server_port = int(os.environ.get("LLAMA_TEST_PORT", "18763"))
    server.n_slots = 1
    server.n_ctx = 2048
    server.n_predict = 8
    server.temperature = 0.0
    server.seed = 1
    yield
    server.stop()  # this file is run with --noconftest (the shared conftest preloads every preset from HF)


def doc(base, n):
    return list(range(base, base + n))


# the tiny model trains with a 128-token context (the server caps the slot to it): keep prompts under ~100 tokens
SYS = doc(300, 8)
A, B, C = doc(400, 24), doc(500, 24), doc(600, 24)
Q = doc(700, 16)
Q2 = doc(800, 16)


def segs(first=8, n=72):
    return {"segments": [{"offset": first, "n_tokens": n}]}


def complete(prompt, reorder=None, cache=True):
    data = {"prompt": prompt, "cache_prompt": cache, "n_predict": 8, "temperature": 0.0, "seed": 1, "return_tokens": True}
    if reorder is not None:
        data["kv_reorder"] = reorder
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200, res.body
    return res.body


def test_off_has_no_reorder_fields():
    server.start()
    body = complete(SYS + A + B + C + Q)
    assert "kv_reorder" not in body["timings"]
    body2 = complete(SYS + C + A + B + Q)
    assert "kv_reorder" not in body2["timings"]
    assert body2["timings"]["cache_n"] == len(SYS)  # plain LCP behaviour, the order is NOT touched


def test_reordered_request_becomes_prefix_hit_and_matches_full_recompute():
    server.start()
    complete(SYS + A + B + C + Q)
    body = complete(SYS + C + A + B + Q2, reorder=segs())
    rep = body["timings"]["kv_reorder"]
    assert rep["applied"] is True
    assert rep["baseline_prefix"] == len(SYS)
    assert rep["reordered_prefix"] == len(SYS + A + B + C)
    assert body["timings"]["cache_n"] >= len(SYS + A + B + C) - 1
    reordered_out = body["tokens"]

    # reference: the reordered prompt, cold server, no planner
    server.stop()
    server.start()
    ref = complete(SYS + A + B + C + Q2, cache=False)
    assert ref["tokens"] == reordered_out
    assert "kv_reorder" not in ref["timings"]


def test_no_cached_prompt_is_a_noop():
    server.start()
    body = complete(SYS + C + A + B + Q, reorder=segs())
    rep = body["timings"]["kv_reorder"]
    assert rep["applied"] is False
    assert body["timings"]["cache_n"] == 0


def test_tokens_outside_segments_are_never_moved():
    server.start()
    complete(SYS + A + B + C + Q)
    # only B and C are marked: A stays pinned at its place, so nothing can be moved in front of it
    body = complete(SYS + C + A + B + Q2, reorder={"segments": [{"offset": 32, "n_tokens": 24}, {"offset": 56, "n_tokens": 24}]})
    assert body["timings"]["kv_reorder"]["applied"] is False


@pytest.mark.parametrize("bad", [
    {"segments": []}, {"segments": [{"offset": 0}]}, {"segments": "x"}, "x",
])
def test_malformed_field_is_rejected(bad):
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": SYS + A, "kv_reorder": bad})
    assert res.status_code == 400


def test_out_of_range_segment_degrades_to_normal_path():
    server.start()
    complete(SYS + A + B + C + Q)
    body = complete(SYS + C + A + B + Q2, reorder={"segments": [{"offset": 20, "n_tokens": 100000}]})
    rep = body["timings"]["kv_reorder"]
    assert rep["applied"] is False and rep["reason"] == "invalid segments"
