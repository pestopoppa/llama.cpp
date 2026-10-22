# [SPEC-RNG.1.1] server-level acceptance for the position-keyed counter RNG (--rng-poskey / request field "rng_poskey").
#
# Claim: with the switch on, SEEDED sampling output is identical across (a) np 1 vs np 4 (requests interleaved in 4 slots)
# and (b) speculative depth 0 vs 3 (draft-mtp), on a fixed prompt set. Exact: 0 differing outputs.
#
# Model: LLAMA_TEST_RNG_MODEL=/path/model.gguf. (b) needs an MTP-capable model (e.g. Qwen3.6-35B-A3B-MTP-Q8_0.gguf);
# set KPF_RNG_SPEC=0 to skip (b) on a model without an MTP head. Port: PORT env, default 18473.
# The speculative rejection-sampling path (block drafters DFlash2/DSpark with temperature>0, slot.spec_dists) is NOT covered:
# it keeps the target distribution but is not sample-identical to depth 0 by construction; draft-mtp uses the exact-match path.
#
# Cases (each one boots its own server):
#   test_poskey_np1_equals_np4                  --rng-poskey on, request without the field  (argv switch)
#   test_poskey_depth0_equals_depth3            request field "rng_poskey": true, server argv off (request switch)
#   test_control_stateful_depth0_differs_depth3 switch OFF at depth 0 vs 3 MUST differ somewhere (non-vacuity of the depth case)
#   test_kill_switch_env_zero                   LLAMA_RNG_POSKEY=0 forces the stateful path even with argv on: depth0 != depth3 as in the control
#   test_env_one_does_not_enable                LLAMA_RNG_POSKEY=1 with the switch off changes nothing vs. unset (ONE-champion rule)
import concurrent.futures
import os

import pytest
from utils import *

server: ServerProcess


@pytest.fixture(scope="module", autouse=True)
def do_something():
    # overrides conftest's preset download: with a local model nothing is fetched and no preset server is started on a default port
    if not os.environ.get("LLAMA_TEST_RNG_MODEL"):
        ServerPreset.load_all()

PROMPTS = [
    "The lighthouse keeper kept a careful log. Every evening he wrote down",
    "In a small harbour town, the baker woke before dawn and began to",
    "Question: why is the sky blue at noon but red at sunset? Answer:",
    "Write a short story about a cat who learns to read the weather:",
    "List five unusual uses for a paperclip, one per line:",
    "The committee met on Tuesday to decide whether the bridge should be",
    "Explain in two sentences how a hash table handles collisions:",
    "Once upon a time, a clerk found forty volumes of storm records and",
]
N_PREDICT = 48
BASE_SEED = 1234


def make_server(n_slots: int, depth: int, rng_poskey: str | None, env: dict | None = None) -> ServerProcess:
    model = os.environ.get("LLAMA_TEST_RNG_MODEL")
    if not model:
        pytest.skip("LLAMA_TEST_RNG_MODEL not set")
    s = ServerPreset.tinyllama2()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.server_port = int(os.environ.get("PORT", "18473"))
    if os.environ.get("LLAMA_TEST_THREADS"):
        s.n_threads = int(os.environ["LLAMA_TEST_THREADS"])
    s.n_slots = n_slots
    s.n_ctx = n_slots * 1024
    s.kv_unified = True
    s.temperature = 0.8
    s.n_predict = N_PREDICT
    s.rng_poskey = rng_poskey
    if depth > 0:
        if os.environ.get("KPF_RNG_SPEC") == "0":
            pytest.skip("KPF_RNG_SPEC=0")
        s.spec_type = "draft-mtp"
        s.spec_draft_n_max = depth
    return s


def run_set(s: ServerProcess, concurrent_n: int, **extra) -> list[list[int]]:
    """the fixed prompt set, seeded per prompt; concurrent_n requests in flight at once (np interleaving)"""
    def one(i: int) -> list[int]:
        body = {"prompt": PROMPTS[i], "n_predict": N_PREDICT, "seed": BASE_SEED + i, "temperature": 0.8, "top_k": 40,
                "top_p": 0.95, "cache_prompt": False, "return_tokens": True, "backend_sampling": False}
        body.update(extra)
        res = s.make_request("POST", "/completion", data=body, timeout=600)
        assert res.status_code == 200, res.body
        return res.body["tokens"]
    out: list[list[int]] = [[] for _ in PROMPTS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_n) as ex:
        futs = {ex.submit(one, i): i for i in range(len(PROMPTS))}
        for f in concurrent.futures.as_completed(futs):
            out[futs[f]] = f.result()
    return out


def n_diff(a: list[list[int]], b: list[list[int]]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def boot_run(n_slots, depth, argv_poskey, concurrent_n, env=None, **extra):
    global server
    server = make_server(n_slots, depth, argv_poskey)
    if env:
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
    try:
        server.start()
        return run_set(server, concurrent_n, **extra)
    finally:
        server.stop()
        if env:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def test_poskey_np1_equals_np4():
    a = boot_run(1, 0, "on", 1)
    b = boot_run(4, 0, "on", 4)
    assert all(len(x) == N_PREDICT for x in a + b), "short outputs make the comparison vacuous"
    assert n_diff(a, b) == 0, ("np1 vs np4 differ", n_diff(a, b), a, b)


def test_poskey_depth0_equals_depth3():
    a = boot_run(1, 0, "off", 1, rng_poskey=True)
    b = boot_run(1, 3, "off", 1, rng_poskey=True)
    assert all(len(x) == N_PREDICT for x in a + b)
    assert n_diff(a, b) == 0, ("depth0 vs depth3 differ", n_diff(a, b), a, b)


def test_control_stateful_depth0_differs_depth3():
    a = boot_run(1, 0, "off", 1)
    b = boot_run(1, 3, "off", 1)
    assert n_diff(a, b) >= 1, "the stateful control never differs between depth 0 and 3: the depth cases would be vacuous (raise temperature / lengthen)"


def test_kill_switch_env_zero():
    a = boot_run(1, 0, "on", 1, env={"LLAMA_RNG_POSKEY": "0"})
    b = boot_run(1, 3, "on", 1, env={"LLAMA_RNG_POSKEY": "0"})
    assert n_diff(a, b) >= 1, "LLAMA_RNG_POSKEY=0 did not force the stateful path (depth0 == depth3 although the kill switch is set)"


def test_env_one_does_not_enable():
    a = boot_run(1, 3, "off", 1)
    b = boot_run(1, 3, "off", 1, env={"LLAMA_RNG_POSKEY": "1"})
    assert n_diff(a, b) == 0, "LLAMA_RNG_POSKEY=1 changed the output with the switch off: the env var must not enable the feature"
