# [SPEC-RNG.1.1] server-level acceptance for the position-keyed counter RNG (--rng-poskey / request field "rng_poskey").
#
# What is NOT claimed: byte-identical output across np 1 vs concurrent np 4, or across draft depth 0 vs 3. On CPU, different ubatch
# shapes take different GEMM/MoE paths, so the target logits themselves drift; the non-poskey (stateful) control drifts identically
# (6/8 prompts). On the draft-mtp path the exact-match verify does one draw per emitted token, so poskey changes no draw alignment there.
#
# What IS claimed, with the same seeded 8-prompt set (BASE_SEED + i per prompt):
#   (1) same-shape control: an np4 server fed the 8 requests ONE AT A TIME equals an np1 server, 8/8 identical (no shape change, so the
#       logits are bit-identical and the position-keyed draws must be too);
#   (2) wherever two arms (np1 vs concurrent np4; depth 0 vs 3) DO diverge, the divergence is explained by logit drift and not by a
#       misaligned RNG: at the first divergent token d the two arms' post-sampling top-k probability vectors differ at d or earlier, and
#       the two chosen tokens are a low-margin disagreement (their post-sampling probabilities are within DRIFT_PROB_TOL).
#
# PREDECLARED RULE (written before the run): PASS = (1) 8/8 identical, and (2) zero divergences with identical probs up to d
# The count of diverging prompts is REPORTED (printed) and never fails a test by itself.
#
# Model: LLAMA_TEST_RNG_MODEL=/path/model.gguf. The depth pair needs an MTP-capable model (e.g. Qwen3.6-35B-A3B-MTP-Q8_0.gguf);
# set KPF_RNG_SPEC=0 to skip it on a model without an MTP head. Port: PORT env, default 18473.
# The speculative rejection-sampling path (block drafters DFlash2/DSpark with temperature>0, slot.spec_dists) is NOT covered:
# it keeps the target distribution but is not sample-identical to depth 0 by construction; draft-mtp uses the exact-match path.
#
# Cases (each one boots its own server(s); the divergence test is parametrized over the two former pairs):
#   test_poskey_np4_serial_equals_np1                 --rng-poskey on; np4 server, requests sent serially, vs np1: 8/8 identical
#   test_poskey_divergence_is_low_margin[np1_vs_np4]  np1 vs CONCURRENT np4 (argv switch), n_probs=10 post_sampling_probs
#   test_poskey_divergence_is_low_margin[depth0_vs_depth3]  request field "rng_poskey": true, server argv off, draft-mtp depth 3
#   test_control_stateful_depth0_differs_depth3       LOGIT-DRIFT WITNESS: switch OFF at depth 0 vs 3 MUST differ somewhere, i.e. the
#                                                     drift the divergence tests rely on is real (non-vacuity)
#   test_kill_switch_env_zero                         LLAMA_RNG_POSKEY=0 forces the stateful path even with argv on: depth0 != depth3 as in the control
#   test_env_one_does_not_enable                      LLAMA_RNG_POSKEY=1 with the switch off changes nothing vs. unset (ONE-champion rule)
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


def run_set_bodies(s: ServerProcess, concurrent_n: int, **extra) -> list[dict]:
    """the fixed prompt set, seeded per prompt; concurrent_n requests in flight at once (np interleaving; 1 = strictly serial).
    Returns the raw /completion response body per prompt."""
    def one(i: int) -> dict:
        body = {"prompt": PROMPTS[i], "n_predict": N_PREDICT, "seed": BASE_SEED + i, "temperature": 0.8, "top_k": 40,
                "top_p": 0.95, "cache_prompt": False, "return_tokens": True, "backend_sampling": False}
        body.update(extra)
        res = s.make_request("POST", "/completion", data=body, timeout=600)
        assert res.status_code == 200, res.body
        return res.body
    out: list[dict] = [{} for _ in PROMPTS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_n) as ex:
        futs = {ex.submit(one, i): i for i in range(len(PROMPTS))}
        for f in concurrent.futures.as_completed(futs):
            out[futs[f]] = f.result()
    return out


def run_set(s: ServerProcess, concurrent_n: int, **extra) -> list[list[int]]:
    return [b["tokens"] for b in run_set_bodies(s, concurrent_n, **extra)]


def n_diff(a: list[list[int]], b: list[list[int]]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def boot_run(n_slots, depth, argv_poskey, concurrent_n, env=None, full=False, **extra):
    """full=True returns the raw response bodies (for the n_probs tests) instead of the token lists"""
    global server
    server = make_server(n_slots, depth, argv_poskey)
    if env:
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
    try:
        server.start()
        return (run_set_bodies if full else run_set)(server, concurrent_n, **extra)
    finally:
        server.stop()
        if env:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# 0.15 absolute tolerance on the post-sampling probability of the two diverging tokens (see check_divergence). Why this number: a
# divergence with identical seeded draws needs the CDF boundary between the two tokens to have moved across the draw u, and the drift of
# a CPU GEMM/MoE path change is O(1e-3) in logit space, i.e. a few 1e-3 in probability. 0.15 is ~two orders above that, so a drift flip
# passes with a wide margin, while a confident disagreement (e.g. 0.9 vs 0.3: an unrelated draw, i.e. a misaligned RNG) fails. The
# tolerance is only a backstop: the primary criterion is "probability vectors differ at or before d".
DRIFT_PROB_TOL = 0.15


def prob_vec(entry: dict) -> list[tuple[int, float]]:
    """one position's post-sampling candidate vector: [(token id, prob), ...] in server order, plus the chosen token's own entry"""
    return [(e["id"], e["prob"]) for e in entry["top_probs"]] + [(entry["id"], entry["prob"])]


def check_divergence(label: str, a: list[dict], b: list[dict]) -> int:
    """a, b: raw response bodies per prompt (n_probs=10, post_sampling_probs=true). Returns the number of diverging prompts (reported only);
    asserts the divergences are logit drift, never a misaligned RNG."""
    n_div = 0
    misaligned: list[str] = []
    too_confident: list[str] = []
    for i, (x, y) in enumerate(zip(a, b)):
        tx, ty = x["tokens"], y["tokens"]
        assert len(tx) == len(ty) == N_PREDICT, (label, i, "short outputs make the comparison vacuous", len(tx), len(ty))
        px, py = x["completion_probabilities"], y["completion_probabilities"]
        assert len(px) == len(tx) and len(py) == len(ty), (label, i, "probabilities missing for some tokens (n_probs not honoured)", len(px), len(py), len(tx))
        if tx == ty:
            continue
        n_div += 1
        d = next(k for k in range(N_PREDICT) if tx[k] != ty[k])          # first divergent token index
        drifted_at = next((k for k in range(d + 1) if prob_vec(px[k]) != prob_vec(py[k])), None)
        pa, pb = px[d]["prob"], py[d]["prob"]
        print(f"[{label}] prompt {i}: first divergence d={d} tokens {tx[d]}/{ty[d]} post-sampling probs {pa:.4f}/{pb:.4f} "
              f"prob vectors first differ at {drifted_at}")
        if drifted_at is None:
            # same context up to d, identical probability vectors at every position <= d, yet different tokens: the draw itself differs
            misaligned.append(f"prompt {i} d={d}")
        elif abs(pa - pb) > DRIFT_PROB_TOL:
            too_confident.append(f"prompt {i} d={d} probs {pa:.4f}/{pb:.4f}")
    print(f"[{label}] diverging prompts: {n_div}/{len(PROMPTS)} (reported, never a failure by itself)")
    assert not misaligned, (label, "divergence with IDENTICAL probability vectors up to d: the RNG is misaligned", misaligned)
    assert not too_confident, (label, f"divergence is a confident disagreement (|dp| > {DRIFT_PROB_TOL}), not low-margin drift", too_confident)
    return n_div


def test_poskey_np4_serial_equals_np1():
    """same-shape control: np4 server, the 8 requests sent one at a time (never concurrently), vs an np1 server, same seeds"""
    a = boot_run(1, 0, "on", 1)
    b = boot_run(4, 0, "on", 1)
    assert all(len(x) == N_PREDICT for x in a + b), "short outputs make the comparison vacuous"
    assert n_diff(a, b) == 0, ("np4 serial vs np1 differ", n_diff(a, b), a, b)


@pytest.mark.parametrize("pair", ["np1_vs_np4", "depth0_vs_depth3"])
def test_poskey_divergence_is_low_margin(pair):
    """former byte-identity pairs, now: any divergence must be explained by logit drift (see module docstring and DRIFT_PROB_TOL)"""
    probs = {"n_probs": 10, "post_sampling_probs": True}
    if pair == "np1_vs_np4":
        a = boot_run(1, 0, "on", 1, full=True, **probs)
        b = boot_run(4, 0, "on", 4, full=True, **probs)        # concurrent: 4 requests in flight, interleaved in 4 slots
    else:
        a = boot_run(1, 0, "off", 1, full=True, rng_poskey=True, **probs)
        b = boot_run(1, 3, "off", 1, full=True, rng_poskey=True, **probs)
    check_divergence(pair, a, b)


def test_control_stateful_depth0_differs_depth3():
    """LOGIT-DRIFT WITNESS: with the switch OFF, depth 0 vs 3 must differ somewhere; this proves the target logits drift between the
    two arms, which is what the divergence tests attribute their divergences to (and that they are not vacuous)"""
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
