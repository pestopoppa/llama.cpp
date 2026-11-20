import os
import pytest
from utils import *

# PFXC (prefix-cache for hybrid/recurrent models in agentic tool loops):
#  A1  post-generation checkpoint + tool-result checkpoint boundaries + checkpoint_min_step 1024
#  A2  copy-load of a prompt-cache entry rejected by the f_keep >= 0.25 rule + the prefix-anchor checkpoint is never FIFO-evicted
# Correctness: a state resumed from a checkpoint / copy-loaded cache entry must reproduce the fresh-prefill logits.
# Set LLAMA_TEST_HYBRID_GGUF to a small hybrid/recurrent GGUF with a chat template (e.g. LFM2.5-1.2B, Qwen3.5-0.8B).

HYBRID = os.environ.get("LLAMA_TEST_HYBRID_GGUF")

pytestmark = pytest.mark.skipif(not HYBRID, reason="LLAMA_TEST_HYBRID_GGUF not set")

SYSTEM = "You are a careful coding agent working in a git repository. " + " ".join(
    f"Rule {i}: keep every change minimal, verify with a test, and report the exact command you ran." for i in range(60))


def _server(tmp_path, monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    srv = ServerPreset.tinyllama2()
    srv.model_file = HYBRID
    srv.model_hf_repo = None
    srv.model_hf_file = None
    srv.n_ctx = 8192
    srv.n_batch = 512
    srv.n_slots = 1
    srv.jinja = True
    srv.temperature = 0.0
    srv.cache_ram = 2048
    srv.log_path = str(tmp_path / "server.log")
    return srv


def _chat(srv, messages, n=12, cache_prompt=True):
    res = srv.make_request("POST", "/v1/chat/completions", data={
        "messages": messages, "max_tokens": n, "temperature": 0.0, "seed": 1,
        "logprobs": True, "top_logprobs": 5, "cache_prompt": cache_prompt,
    })
    assert res.status_code == 200, res.body
    return res.body


def _top(body):
    return [[(t["token"], t["logprob"]) for t in c["top_logprobs"]] for c in body["choices"][0]["logprobs"]["content"]]


def _assert_same(a, b, tol=1e-3):
    ta, tb = _top(a), _top(b)
    assert len(ta) == len(tb)
    for sa, sb in zip(ta, tb):
        assert sa[0][0] == sb[0][0], (sa, sb)  # identical greedy token
        da, db = dict(sa), dict(sb)
        for tok in set(da) & set(db):
            assert abs(da[tok] - db[tok]) < tol, (tok, da[tok], db[tok])


def _conv(task, steps):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Task {task}: fix the failing test in module m{task}."}]
    for i in range(steps):
        msgs.append({"role": "assistant", "content": f"I will run step {i} of task {task}."})
        msgs.append({"role": "user", "content": f"Output of step {i}: " + " ".join(f"line {j} ok" for j in range(40))})
    return msgs


def test_resumed_state_matches_fresh_prefill(tmp_path, monkeypatch):
    srv = _server(tmp_path, monkeypatch)
    srv.start()
    try:
        m1 = _conv(1, 2)
        r1 = _chat(srv, m1)
        gen = r1["choices"][0]["message"]["content"]
        m2 = m1 + [{"role": "assistant", "content": gen}, {"role": "user", "content": "Now run the full suite and summarise."}]
        resumed = _chat(srv, m2)
        assert resumed["timings"]["cache_n"] > 0
        fresh = _chat(srv, m2, cache_prompt=False)
        assert fresh["timings"]["cache_n"] == 0
        _assert_same(resumed, fresh)
    finally:
        srv.stop()
    log = open(srv.log_path).read()
    assert "failed to restore context checkpoint" not in log


def test_post_generation_checkpoint_reuses_generated_turn(tmp_path, monkeypatch):
    srv = _server(tmp_path, monkeypatch)
    srv.start()
    try:
        m1 = _conv(2, 1)
        r1 = _chat(srv, m1, n=24)
        n1 = r1["timings"]["prompt_n"] + r1["timings"]["cache_n"]
        gen = r1["choices"][0]["message"]["content"]
        m2 = m1 + [{"role": "assistant", "content": gen}, {"role": "user", "content": "continue"}]
        r2 = _chat(srv, m2)
        # without the post-generation checkpoint the reuse stops at the end-4 prompt checkpoint (<= n1 - 4)
        assert r2["timings"]["cache_n"] > n1 - 4, (r2["timings"], n1)
    finally:
        srv.stop()


def test_copy_load_shared_prefix(tmp_path, monkeypatch):
    srv = _server(tmp_path, monkeypatch, LLAMA_ARG_CACHE_RAM_COPY_MIN=64)
    srv.start()
    try:
        big = _conv(3, 12)       # long conversation, the shared system prefix is < 25 % of it
        _chat(srv, big)
        other = [{"role": "system", "content": "Generate a short title."}, {"role": "user", "content": "hello"}]
        _chat(srv, other)        # evicts the long conversation into the prompt cache
        new_task = _conv(4, 0)   # shares only the system prefix with the cached long conversation
        r = _chat(srv, new_task)
        assert r["timings"]["cache_n"] > 0, r["timings"]
        fresh = _chat(srv, new_task, cache_prompt=False)
        _assert_same(r, fresh)
    finally:
        srv.stop()
    log = open(srv.log_path).read()
    assert "copy-loading cached prompt" in log
