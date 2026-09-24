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
