import os
import pytest
from utils import *

# We use a F16 MOE gguf as main model, and q4_0 as draft model

server = ServerPreset.stories15m_moe()

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"

def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    # set default values
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"


@pytest.fixture(autouse=True)
def fixture_create_server():
    return create_server()


def test_with_and_without_draft():
    global server
    server.model_draft = None  # disable draft model
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
    })
    assert res.status_code == 200
    content_no_draft = res.body["content"]
    server.stop()

    # create new server with draft model
    create_server()
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
    })
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    content_draft = res.body["content"]

    assert content_no_draft == content_draft


# ---------------------------------------------------------------------------------------------------------------
# SPEC-STALE-1: a stop inside an accepted draft must leave the same slot cache as the non-speculative run
# (upstream #32771 shape) and an accepted EOG must end the round (upstream #29638 / #28232 cases).
# Local model files are taken from env vars so the tests run offline; each test skips when its file is absent.

SPEC_STALE_TARGET = os.environ.get("LLAMA_TEST_SPEC_STALE_TARGET", "")   # stories15M_MOE-F16.gguf
SPEC_STALE_DRAFT  = os.environ.get("LLAMA_TEST_SPEC_STALE_DRAFT", "")    # stories15M-q4_0.gguf
SPEC_STALE_TINY   = os.environ.get("LLAMA_TEST_SPEC_STALE_TINY", "")     # stories260K-f32.gguf
SPEC_STALE_HYBRID = os.environ.get("LLAMA_TEST_SPEC_STALE_HYBRID", "")   # a small RS/hybrid gguf (e.g. LFM2.5-1.2B)
# extra spec-on/off parity configs on served drafter types, ';'-separated "<name>|<model>|<spec-type>[|<draft model>]"
SPEC_STALE_EXTRA  = os.environ.get("LLAMA_TEST_SPEC_STALE_EXTRA", "")

STOP_PROMPT = "I believe the meaning of life is"


def _spec_stale_server(model, spec_type, model_draft=None, n_slots=1):
    srv = ServerPreset.stories15m_moe()
    srv.model_hf_repo = None
    srv.model_hf_file = None
    srv.model_file = model
    srv.model_draft = model_draft
    srv.spec_type = spec_type
    srv.n_slots = n_slots
    srv.n_ctx = 2048 * n_slots
    srv.fa = "off"
    srv.n_threads = int(os.environ.get("LLAMA_TEST_THREADS", "8"))
    if spec_type == "draft-simple":
        srv.spec_draft_n_min = 4
        srv.spec_draft_n_max = 8
    return srv


def _stop_cache_state(srv, content, n_slots, n_predict, max_words=20):
    """per stop word: (word, tokens_cached of the stopped request, cache_n of the follow-up on the same slot)"""
    res = []
    words = list(dict.fromkeys(content.split()))[:max_words]
    for k, word in enumerate(words):
        id_slot = k % n_slots
        base = {"prompt": STOP_PROMPT, "temperature": 0.0, "top_k": 1, "n_predict": n_predict, "id_slot": id_slot}
        stopped = srv.make_request("POST", "/completion", data={**base, "stop": [" " + word]})
        assert stopped.status_code == 200, stopped.body
        # the follow-up continues the text, so it reuses the tokens cached by the stopped request
        nxt = srv.make_request("POST", "/completion", data={**base, "prompt": STOP_PROMPT + content, "n_predict": 1})
        assert nxt.status_code == 200, nxt.body
        res.append((word, stopped.body["tokens_cached"], nxt.body["timings"]["cache_n"]))
    return res


def _parity(model, spec_type, model_draft, n_slots, n_predict=32):
    global server
    server = _spec_stale_server(model, None, None, n_slots)
    server.start()
    ref = server.make_request("POST", "/completion", data={"prompt": STOP_PROMPT, "temperature": 0.0, "top_k": 1, "n_predict": n_predict})
    assert ref.status_code == 200
    content = ref.body["content"]
    state_off = _stop_cache_state(server, content, n_slots, n_predict)
    server.stop()

    server = _spec_stale_server(model, spec_type, model_draft, n_slots)
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": STOP_PROMPT, "temperature": 0.0, "top_k": 1, "n_predict": n_predict})
    assert res.status_code == 200
    assert res.body["content"] == content
    assert res.body["timings"].get("draft_n_accepted", 0) > 0
    state_on = _stop_cache_state(server, content, n_slots, n_predict)
    server.stop()

    diff = [(a, b) for a, b in zip(state_off, state_on) if a != b]
    print(f"spec-stale parity {spec_type} np={n_slots}: {len(state_off) - len(diff)}/{len(state_off)} stop words match; diffs={diff}")
    assert len(state_off) >= 10
    assert diff == []


@pytest.mark.skipif(not (os.path.isfile(SPEC_STALE_TARGET) and os.path.isfile(SPEC_STALE_DRAFT)), reason="SPEC-STALE target/draft not set")
@pytest.mark.parametrize("n_slots", [1, 2])
def test_spec_stale_stop_inside_draft_same_cache_as_without_draft(n_slots):
    _parity(SPEC_STALE_TARGET, "draft-simple", SPEC_STALE_DRAFT, n_slots)


def _extra_configs():
    out = []
    for item in filter(None, SPEC_STALE_EXTRA.split(";")):
        f = item.split("|")
        out.append(pytest.param(f[1], f[2], f[3] if len(f) > 3 else None, id=f[0]))
    return out or [pytest.param(None, None, None, id="none", marks=pytest.mark.skip(reason="LLAMA_TEST_SPEC_STALE_EXTRA not set"))]


@pytest.mark.parametrize("n_slots", [1, 2])
@pytest.mark.parametrize("model,spec_type,model_draft", _extra_configs())
def test_spec_stale_parity_served_types(model, spec_type, model_draft, n_slots):
    _parity(model, spec_type, model_draft, n_slots, n_predict=int(os.environ.get("LLAMA_TEST_SPEC_STALE_NPRED", "32")))


def _eog_tail_case(model):
    """#28232: an ngram draft of 48 EOG tokens after the forced 'a' must not be accepted past the first EOG"""
    global server
    server = _spec_stale_server(model, "ngram-simple", None, 1)
    server.n_ctx = 1024
    server.n_batch = 512
    server.n_ubatch = 512
    server.spec_draft_n_max = 48
    server.start()

    props = server.make_request("GET", "/props")
    assert props.status_code == 200, props.body
    eos = server.make_request("POST", "/tokenize", data={"content": props.body["eos_token"], "parse_special": True})
    tok_x = server.make_request("POST", "/tokenize", data={"content": "x"})
    tok_bos_x = server.make_request("POST", "/tokenize", data={"content": "x", "add_special": True})
    assert len(eos.body["tokens"]) == 1, eos.body
    id_eos, id_x = eos.body["tokens"][0], tok_x.body["tokens"][0]
    id_bos = tok_bos_x.body["tokens"][0] if len(tok_bos_x.body["tokens"]) >= 2 else None
    head = [id_bos, id_x] if id_bos is not None else [id_x]

    cal = server.make_request("POST", "/completion", data={
        "prompt": head, "cache_prompt": False, "temperature": 0.0, "top_k": 1, "n_predict": 2,
        "return_tokens": True, "grammar": 'root ::= "a"'})
    assert cal.status_code == 200, cal.body
    assert cal.body["tokens"][-1] == id_eos, cal.body
    id_a = cal.body["tokens"][0]

    prompt_tokens = head + [id_x] * 11 + [id_a] + [id_eos] * 48 + [id_x] * 11
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_tokens, "cache_prompt": True, "temperature": 0.0, "top_k": 1, "n_predict": 64,
        "return_tokens": True, "grammar": 'root ::= "a"'})
    assert res.status_code == 200, res.body
    server.stop()
    print(f"spec-stale eog tail: tokens={res.body['tokens']} stop_type={res.body['stop_type']} "
          f"timings={ {k: res.body['timings'].get(k) for k in ('draft_n', 'draft_n_accepted', 'predicted_n')} } "
          f"tokens_cached={res.body['tokens_cached']} len={len(prompt_tokens)}")
    return res, prompt_tokens, id_a, id_eos


@pytest.mark.skipif(not os.path.isfile(SPEC_STALE_TINY), reason="LLAMA_TEST_SPEC_STALE_TINY not set")
def test_spec_stale_ngram_eog_draft_tail_is_not_accepted():
    res, prompt_tokens, id_a, id_eos = _eog_tail_case(SPEC_STALE_TINY)
    assert res.body["tokens"] == [id_a, id_eos]
    assert res.body["stop_type"] == "eos"
    assert res.body["timings"].get("draft_n", 0) > 0
    assert res.body["timings"]["draft_n_accepted"] == 0
    # spec-off caches the prompt and 'a', never the EOG
    assert res.body["tokens_cached"] == len(prompt_tokens) + 1


@pytest.mark.skipif(not os.path.isfile(SPEC_STALE_HYBRID), reason="LLAMA_TEST_SPEC_STALE_HYBRID not set")
def test_spec_stale_eog_row0_checkpoint_round_terminates():
    # recurrent target + a 48-token draft > n_rs_seq: the EOG is the target's pick at row 0, so the round restores
    # its checkpoint and replays a one-token draft; generation must end in that one extra round, not loop
    res, prompt_tokens, id_a, id_eos = _eog_tail_case(SPEC_STALE_HYBRID)
    assert res.body["tokens"] == [id_a, id_eos]
    assert res.body["stop_type"] == "eos"
    assert res.body["timings"]["predicted_n"] == 2
    assert res.body["timings"]["draft_n_accepted"] <= 1
    assert res.body["tokens_cached"] == len(prompt_tokens) + 1


def test_different_draft_min_draft_max():
    global server
    test_values = [
        (1, 2),
        (1, 4),
        (4, 8),
        (4, 12),
        (8, 16),
    ]
    last_content = None
    for draft_min, draft_max in test_values:
        server.stop()
        server.spec_draft_n_min = draft_min
        server.spec_draft_n_max = draft_max
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 16,
        })
        assert res.status_code == 200
        if last_content is not None:
            assert last_content == res.body["content"]
        last_content = res.body["content"]


def test_per_request_draft_max_and_isolation():
    global server
    server.start()

    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
    }
    request_cap = server.spec_draft_n_min
    assert 0 < request_cap < server.spec_draft_n_max

    # A zero request cap disables speculation and is reported as effective.
    res_disabled = server.make_request("POST", "/completion", data={
        **request,
        "speculative.n_max": 0,
    })
    assert res_disabled.status_code == 200
    assert res_disabled.body["generation_settings"]["speculative.n_max"] == 0
    assert "draft_n" not in res_disabled.body["timings"]

    # A non-zero request cap is plumbed independently of the launch default.
    res_capped = server.make_request("POST", "/completion", data={
        **request,
        "speculative.n_max": request_cap,
    })
    assert res_capped.status_code == 200
    assert res_capped.body["generation_settings"]["speculative.n_max"] == request_cap
    assert res_capped.body["timings"]["draft_n"] > 0

    # Requests cannot exceed the launch-time capacity; reporting reflects the
    # effective clamped value, not the unfulfillable requested value.
    res_clamped = server.make_request("POST", "/completion", data={
        **request,
        "speculative.n_max": server.spec_draft_n_max + 4,
    })
    assert res_clamped.status_code == 200
    assert res_clamped.body["generation_settings"]["speculative.n_max"] == server.spec_draft_n_max
    assert res_clamped.body["timings"]["draft_n"] > 0

    # Omitting the field on the next request restores the launch default; the
    # zero cap from the first request must not leak through the reused slot.
    res_default = server.make_request("POST", "/completion", data=request)
    assert res_default.status_code == 200
    assert res_default.body["generation_settings"]["speculative.n_max"] == server.spec_draft_n_max
    assert res_default.body["timings"]["draft_n"] > 0

    assert res_disabled.body["content"] == res_capped.body["content"]
    assert res_capped.body["content"] == res_clamped.body["content"]
    assert res_clamped.body["content"] == res_default.body["content"]


def test_slot_ctx_not_exceeded():
    global server
    server.n_ctx = 256
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_with_ctx_shift():
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    assert res.body["tokens_predicted"] == 256
    assert res.body["truncated"] == True


def test_n_probs_with_speculative_decoding():
    # TD-1d.2: the speculative-accept path used to leave every accepted token's
    # probabilities unset (prob = 1.0, no top_logprobs), regardless of n_probs --
    # so a request combining speculative decoding with n_probs silently got no
    # probabilities for the (typically large majority of) tokens that were
    # accepted from the draft instead of sampled directly. Greedy decoding
    # (temperature 0) with a matched tiny draft model reproduces a high
    # acceptance rate deterministically, so this exercises the non-post-sampling
    # (raw target-logits) branch of the fix on essentially every predicted token.
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_probs": 10,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
    })
    assert res.status_code == 200
    # sanity check that speculative decoding actually ran for this request --
    # otherwise this test would pass trivially via the non-speculative path.
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] > 0

    assert "completion_probabilities" in res.body
    assert len(res.body["completion_probabilities"]) == 16
    for tok in res.body["completion_probabilities"]:
        assert "id" in tok and tok["id"] > 0
        assert "token" in tok and type(tok["token"]) == str
        assert "logprob" in tok and tok["logprob"] <= 0.0
        assert len(tok["top_logprobs"]) == 10
        for prob in tok["top_logprobs"]:
            assert "id" in prob and prob["id"] > 0
            assert "token" in prob and type(prob["token"]) == str
            assert "logprob" in prob and prob["logprob"] <= 0.0


def test_n_probs_post_sampling_with_speculative_decoding():
    # Same defect as test_n_probs_with_speculative_decoding(), but for the
    # post_sampling_probs branch, which snapshots the TARGET model's post-chain
    # candidate distribution at accept time (common_sampler_sample_and_accept_n's
    # out_dists) rather than reading raw logits by index.
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Today was the day. Today I would finally become a",
        "n_probs": 10,
        "temperature": 1.0,
        "n_predict": 16,
        "post_sampling_probs": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0

    assert "completion_probabilities" in res.body
    assert len(res.body["completion_probabilities"]) == 16
    for tok in res.body["completion_probabilities"]:
        assert "id" in tok and tok["id"] > 0
        assert "token" in tok and type(tok["token"]) == str
        assert "prob" in tok and 0.0 < tok["prob"] <= 1.0
        assert "top_probs" in tok and type(tok["top_probs"]) == list
        assert len(tok["top_probs"]) > 0
        for prob in tok["top_probs"]:
            assert "id" in prob and prob["id"] > 0
            assert "token" in prob and type(prob["token"]) == str
            # 0.0 probability tokens should never be returned by the server
            assert "prob" in prob and 0.0 < prob["prob"] <= 1.0


def _completion_probs(prompt: str, n_predict: int, post_sampling: bool):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_probs": 5,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": n_predict,
        "post_sampling_probs": post_sampling,
    })
    assert res.status_code == 200
    return res.body


@pytest.mark.parametrize("post_sampling", [False, True])
def test_n_probs_serial_verify_matches_plain(monkeypatch, post_sampling: bool):
    # DS41-T4.1.2: LLAMA_SPEC_EXACT=serial (and the DSpark recurrent-target default at
    # temp 0) verifies each draft token with its own single-token decode, overwriting
    # ctx_tgt's logits every step, so the accept loop used to return no probabilities
    # for any token accepted through it. The serial loop now captures prob/top-n_probs
    # at each sampled row before the next decode. Every row on this path is a
    # single-token decode, exactly like plain decoding, so the per-token ids AND
    # probabilities must equal the no-draft run.
    global server
    prompt = "I believe the meaning of life is"
    n_predict = 24
    lp_key, top_key = ("prob", "top_probs") if post_sampling else ("logprob", "top_logprobs")

    server.model_draft = None
    server.spec_type = None
    server.start()
    plain = _completion_probs(prompt, n_predict, post_sampling)
    server.stop()

    create_server()
    monkeypatch.setenv("LLAMA_SPEC_EXACT", "serial")
    server.start()
    serial = _completion_probs(prompt, n_predict, post_sampling)

    # the serial path actually ran and accepted draft tokens (otherwise this would pass
    # trivially through the non-speculative populate_token_probs path)
    assert serial["timings"]["draft_n"] > 0
    assert serial["timings"]["draft_n_accepted"] > 0

    assert serial["content"] == plain["content"]
    p_rows = plain["completion_probabilities"]
    s_rows = serial["completion_probabilities"]
    assert len(p_rows) == len(s_rows) == n_predict
    for i, (p, s) in enumerate(zip(p_rows, s_rows)):
        assert s["id"] == p["id"], f"row {i}"
        assert len(s[top_key]) > 0, f"row {i}: no top-n on the serial path"
        assert s[lp_key] == pytest.approx(p[lp_key], abs=1e-6), f"row {i}"
        assert [t["id"] for t in s[top_key]] == [t["id"] for t in p[top_key]], f"row {i}"
        for a, b in zip(s[top_key], p[top_key]):
            assert a[lp_key] == pytest.approx(b[lp_key], abs=1e-6), f"row {i}"


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 2),
    (2, 2),
])
def test_multi_requests_parallel(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.start()
    tasks = []
    for _ in range(n_requests):
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
        })))
    results = parallel_function_calls(tasks)
    for res in results:
        assert res.status_code == 200
        assert match_regex("(wise|kind|owl|answer)+", res.body["content"])


@pytest.mark.parametrize("n_slots", [2, 4])
def test_serial_exact_concurrent_slots(n_slots: int, monkeypatch):
    # a serial verification decode replaces the shared batch's logits: before the fix the next
    # slot aborted in common_sampler_sample with "invalid logits id"
    global server
    monkeypatch.setenv("LLAMA_SPEC_EXACT", "serial")
    server.n_slots = n_slots
    server.n_ctx = 2048 * n_slots
    server.start()
    prompts = ["I believe the meaning of life is", "Once upon a time", "The sky is", "In the forest"][:n_slots]
    data = [{"prompt": p, "temperature": 0.0, "top_k": 1, "n_predict": 32, "cache_prompt": False} for p in prompts]
    sequential = [server.make_request("POST", "/completion", data=d).body["content"] for d in data]
    tasks = [(server.make_request, ("POST", "/completion", d)) for d in data]
    results = parallel_function_calls(tasks)
    assert [r.status_code for r in results] == [200] * n_slots
    assert [r.body["content"] for r in results] == sequential
