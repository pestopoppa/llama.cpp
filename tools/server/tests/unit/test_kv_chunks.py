import os
import pytest
from utils import *

# Chunk-KV reuse (P1, KPF-62.5 server wiring, G0 at the server level). Needs a tiny local attention-only model:
#   LLAMA_TEST_MODEL=/path/stories15M-q4_0.gguf LLAMA_SERVER_BIN_PATH=<build>/bin/llama-server \
#       pytest --noconftest tools/server/tests/unit/test_kv_chunks.py
# The store is enabled through the environment (LLAMA_ARG_CHUNK_STORE_MIB), because the test harness has no generic
# extra-argument field.
MODEL = os.environ.get("LLAMA_TEST_MODEL")

server = ServerPreset.tinyllama2()


def make_server(store_mib):
    global server
    server = ServerPreset.tinyllama2()
    server.model_file = MODEL
    server.model_hf_repo = None
    server.model_hf_file = None
    server.server_port = int(os.environ.get("LLAMA_TEST_PORT", "18764"))
    server.n_slots = 1
    server.n_ctx = 2048
    server.n_batch = 16          # small batches so the G0a alignment (mode "all") is exercised
    server.n_ubatch = 16
    server.n_predict = 8
    server.temperature = 0.0
    server.seed = 1
    if store_mib:
        os.environ["LLAMA_ARG_CHUNK_STORE_MIB"] = str(store_mib)
    else:
        os.environ.pop("LLAMA_ARG_CHUNK_STORE_MIB", None)
    return server


@pytest.fixture(autouse=True)
def create_server():
    if not MODEL:
        pytest.skip("LLAMA_TEST_MODEL not set")
    make_server(64)
    yield
    server.stop()  # this file is run with --noconftest (the shared conftest preloads every preset from HF)
    os.environ.pop("LLAMA_ARG_CHUNK_STORE_MIB", None)


def doc(base, n):
    return list(range(base, base + n))


# the tiny model trains with a 128-token context (the server caps the slot to it): keep prompts under ~100 tokens
SYS = doc(300, 8)
A, B = doc(400, 24), doc(500, 24)
G = doc(600, 6)
Q = doc(700, 12)
PROMPT = SYS + A + G + B + Q                       # A at 8, B at 38
CHUNKS = [{"offset": len(SYS), "n_tokens": len(A)}, {"offset": len(SYS + A + G), "n_tokens": len(B)}]


def complete(prompt, chunks=None, cache=True, n_probs=4):
    data = {"prompt": prompt, "cache_prompt": cache, "n_predict": 8, "temperature": 0.0, "seed": 1,
            "return_tokens": True, "n_probs": n_probs}
    if chunks is not None:
        data["kv_chunks"] = chunks
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200, res.body
    return res.body


def strip(body):
    b = dict(body)
    for k in ("timings", "id", "created", "id_slot", "generation_settings"):
        b.pop(k, None)
    return b


def test_absent_field_reports_nothing():
    server.start()
    body = complete(PROMPT)
    assert "kv_chunks" not in body["timings"]


def test_store_off_ignores_the_field_and_matches_g0b():
    make_server(0)
    server.start()
    ref = complete(PROMPT, cache=False)
    body = complete(PROMPT, chunks=CHUNKS, cache=False)
    rep = body["timings"]["kv_chunks"]
    assert rep["applied"] is False and "off" in rep["reason"]
    assert strip(body) == strip(ref)


def test_mode_all_is_bit_identical_to_the_plain_request_g0a():
    server.start()
    ref = complete(PROMPT, cache=False)
    server.stop()
    server.start()
    body = complete(PROMPT, chunks={"chunks": CHUNKS, "recompute": "all"}, cache=False)
    rep = body["timings"]["kv_chunks"]
    assert rep["applied"] is True and rep["approx"] is False
    assert rep["n_chunk_injected"] == 0 and rep["n_chunk_hit"] == 0  # store bypassed
    assert rep["assembled_until"] == (len(SYS + A + G + B) // 16) * 16
    assert strip(body) == strip(ref)  # tokens AND n_probs probabilities, exactly


def test_head_mode_reuses_chunks_at_new_offsets():
    server.start()
    b1 = complete(PROMPT, chunks={"chunks": CHUNKS, "recompute": "head", "window": 4}, cache=False)
    r1 = b1["timings"]["kv_chunks"]
    assert r1["applied"] is True
    assert r1["n_chunk_miss"] == 2 and r1["n_chunk_hit"] == 0
    assert r1["n_chunk_injected"] == len(A) + len(B) and r1["n_chunk_recomputed"] == 8
    assert r1["approx"] is True

    # same chunks after a different predecessor, B first: both are hits (content key, no position)
    pre = doc(320, 5)
    p2 = pre + B + G + A + Q
    c2 = [{"offset": len(pre), "n_tokens": len(B)}, {"offset": len(pre + B + G), "n_tokens": len(A)}]
    b2 = complete(p2, chunks={"chunks": c2, "recompute": "none"}, cache=False)
    r2 = b2["timings"]["kv_chunks"]
    assert r2["n_chunk_hit"] == 2 and r2["n_chunk_miss"] == 0 and r2["n_chunk_recomputed"] == 0
    assert r2["chunk_store_hit_bytes"] > 0


def test_approximate_state_never_seeds_an_exact_request():
    server.start()
    complete(PROMPT, chunks={"chunks": CHUNKS, "recompute": "none"})
    body = complete(PROMPT)  # same prompt, no field: only the exact part before the first chunk may be reused
    assert body["timings"]["cache_n"] == len(SYS)
    # reference with the same batch partition: a cold server that has exactly SYS cached
    server.stop()
    server.start()
    complete(SYS + doc(900, 4))
    ref = complete(PROMPT)
    assert ref["timings"]["cache_n"] == len(SYS)
    assert strip(body) == strip(ref)


def test_chunk_inside_the_cached_prefix_is_dropped():
    server.start()
    complete(PROMPT)
    body = complete(PROMPT + doc(800, 4), chunks={"chunks": CHUNKS, "recompute": "none"})
    rep = body["timings"]["kv_chunks"]
    assert rep["applied"] is False and rep["n_dropped_cached"] == 2


@pytest.mark.parametrize("bad", [
    [], "x", [{"offset": 0}], {"chunks": []}, {"chunks": CHUNKS, "recompute": "most"}, {"chunks": CHUNKS, "window": -1},
])
def test_malformed_field_is_rejected(bad):
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": PROMPT, "kv_chunks": bad})
    assert res.status_code == 400


def test_invalid_spans_degrade_to_the_normal_path():
    server.start()
    ref = complete(PROMPT, cache=False)
    body = complete(PROMPT, chunks=[{"offset": 8, "n_tokens": 100000}], cache=False)
    rep = body["timings"]["kv_chunks"]
    assert rep["applied"] is False
    assert strip(body) == strip(ref)
