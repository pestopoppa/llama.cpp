# [KPF-17] cross-slot prefix fork (--slot-fork-min-tokens), server side.
#
# Model: LLAMA_TEST_FORK_MODEL=/path/model.gguf (a hybrid such as Qwen3.5-0.8B exercises the checkpoint
# path, a pure-attention model the kv path); without it the stock tinyllama preset is used.
# Port: PORT env, default 18471 (never a standard port).
#
# Prompts are natural text tokenized by the running server, concatenated at the token level so the shared
# prefix is exact (random token ids make the model emit invalid UTF-8, which /completion rejects). Greedy. "fresh" means the same
# prompt with "slot_fork": false and "cache_prompt": false (full prefill, no fork, no RAM-cache reuse).

import json
import os
import threading
import time

import pytest
from utils import *

server: ServerProcess

FORK_MIN = 16

TRUNK_TEXT = (
    "The lighthouse keeper kept a careful log. Every evening he wrote down the wind, the height of the waves, "
    "the ships that passed, and the colour of the sky at sunset. Over thirty years the log grew to forty volumes. "
    "When the keeper retired, the harbour office asked a young clerk to read all of them and summarise what had "
    "changed. The clerk noticed that storms had become more frequent in autumn, that fewer fishing boats passed "
    "each decade, and that the keeper's handwriting grew smaller and neater as the years went by. She also found "
    "small drawings in the margins: gulls, a cat that lived in the tower for a while, and once a whale. "
)
SUFFIX_TEXT = [
    "Question: what animal did the clerk find drawn in the margins? Answer:",
    "Question: how many volumes did the log have? Answer:",
    "Question: which season had more storms? Answer:",
    "Question: who read the log after the keeper retired? Answer:",
    "Write one sentence about the keeper's handwriting:",
    "List three things the keeper wrote down every evening:",
    "Question: what happened to the number of fishing boats? Answer:",
    "Give the log a short title:",
]

TRUNK: list[int] = []
SUF: dict[int, list[int]] = {}
TRUNK_LEN = 0


def tokenize(text: str) -> list[int]:
    res = server.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200
    return res.body["tokens"]


def build_prompts():
    global TRUNK, SUF, TRUNK_LEN
    TRUNK = tokenize(TRUNK_TEXT)
    TRUNK_LEN = len(TRUNK)
    SUF = {k: tokenize(" " + t) for k, t in enumerate(SUFFIX_TEXT)}
    assert TRUNK_LEN > 4 * FORK_MIN


def make_server(fork_min: int = FORK_MIN, **kw) -> ServerProcess:
    s = ServerPreset.tinyllama2()
    model = os.environ.get("LLAMA_TEST_FORK_MODEL")
    if model:
        s.model_hf_repo = None
        s.model_hf_file = None
        s.model_file = model
    s.server_port = int(os.environ.get("PORT", "18471"))
    if os.environ.get("LLAMA_TEST_THREADS"):
        s.n_threads = int(os.environ["LLAMA_TEST_THREADS"])
    s.temperature = 0.0
    s.n_slots = 4
    s.n_ctx = 4 * 2048
    s.kv_unified = True
    s.server_slots = True
    s.server_metrics = True
    s.slot_fork_min_tokens = fork_min
    s.n_predict = 24
    # [KPF-17/19] env knobs so the same cases run on MTP / smaller batches without editing the file
    if os.environ.get("KPF_SPEC_TYPE"):
        s.spec_type = os.environ["KPF_SPEC_TYPE"]
        s.spec_draft_n_max = int(os.environ.get("KPF_SPEC_N_MAX", "4"))
    if os.environ.get("KPF_N_BATCH"):
        s.n_batch = int(os.environ["KPF_N_BATCH"])
        s.n_ubatch = min(s.n_batch, int(os.environ.get("KPF_N_UBATCH", s.n_batch)))
    if os.environ.get("KPF_SLOT_SAVE_DIR"):  # needed by /slots?action=erase in the KPF-17 cases
        s.slot_save_path = os.environ["KPF_SLOT_SAVE_DIR"]
    if os.environ.get("KPF_DFT_FULL") == "1":
        s.slot_fork_dft_full = True
    for k, v in kw.items():
        setattr(s, k, v)
    return s


@pytest.fixture(scope="module", autouse=True)
def do_something():
    # overrides conftest's preset download: with a local model nothing is fetched and no preset
    # server is started (those would listen on the default port)
    if not os.environ.get("LLAMA_TEST_FORK_MODEL"):
        ServerPreset.load_all()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = make_server()
    yield
    server.stop()


def complete(prompt: list[int], id_slot: int, n_predict: int = 24, **extra) -> dict:
    data = {"prompt": prompt, "id_slot": id_slot, "n_predict": n_predict, "cache_prompt": True,
            "temperature": 0.0, "return_tokens": True}
    data.update(extra)
    res = server.make_request("POST", "/completion", data=data, timeout=600)
    assert res.status_code == 200, res.body
    return res.body


def slots() -> list[dict]:
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    return res.body


def check_pool_invariant(ss: list[dict]):
    pool = ss[0]["kv_pool"]
    priv = sum(s["kv_cells"]["private"] for s in ss)
    assert pool["used"] == priv + pool["shared"], (pool, priv)
    for s in ss:
        assert s["kv_pool"] == pool  # one snapshot


def assert_forked(body: dict, what: str, n_min: int = FORK_MIN, src: int | None = None) -> int:
    """generic guard: a case whose purpose is to exercise the fork fails if the fork did not happen (2026-10-21: the
    decoding-source and current-step cases passed for weeks without a single do_slot_fork in their logs)"""
    t = body["timings"]
    assert t.get("n_fork_tokens", 0) >= max(1, n_min), (what, "the fork did not happen", t)
    if src is not None:
        assert t.get("fork_src_slot") == src, (what, "forked from the wrong slot", t)
    return t["n_fork_tokens"]


def test_fork_off_adds_nothing():
    global server
    server = make_server(fork_min=0)
    server.start()
    build_prompts()
    complete(TRUNK, 0, n_predict=0)
    body = complete(TRUNK + SUF[0], 1)
    t = body["timings"]
    assert "n_fork_tokens" not in t and "fork_src_kind" not in t
    for s in slots():
        assert "kv_cells" not in s and "fork" not in s


def test_fork_from_idle_trunk_equals_fresh():
    server.start()
    build_prompts()
    complete(TRUNK, 0, n_predict=0, checkpoint_at=[-1])

    child = complete(TRUNK + SUF[1], 1)
    t = child["timings"]
    assert t["n_fork_tokens"] >= TRUNK_LEN, t
    assert t["fork_src_slot"] == 0
    assert t["fork_src_kind"] in ("end", "checkpoint", "kv")
    assert t["cache_n"] == TRUNK_LEN
    assert t["prompt_n"] == len(SUF[1])

    fresh = complete(TRUNK + SUF[1], 2, slot_fork=False, cache_prompt=False)
    assert fresh["timings"]["n_fork_tokens"] == 0
    assert fresh["timings"]["cache_n"] == 0  # cache_prompt false: full prefill
    assert child["tokens"] == fresh["tokens"]


def test_second_child_after_source_purge():
    # default cache_idle_slots + --cache-ram: launching child 1 purges the idle trunk slot.
    # child 2 must then fork from child 1 (the junction travels with the fork), and the purge of
    # the source must not disturb child 1's cells
    global server
    server = make_server(cache_ram=512)
    server.start()
    build_prompts()
    complete(TRUNK, 0, n_predict=0, checkpoint_at=[-1])

    c1 = complete(TRUNK + SUF[1], 1)
    assert c1["timings"]["n_fork_tokens"] >= TRUNK_LEN

    c2 = complete(TRUNK + SUF[2], 2)
    t2 = c2["timings"]
    assert t2["n_fork_tokens"] >= TRUNK_LEN, t2  # kv mode forks the full LCP, which may run into a shared suffix head
    assert t2["fork_src_slot"] in (0, 1)

    ss = slots()
    check_pool_invariant(ss)

    for k, c in ((1, c1), (2, c2)):
        fresh = complete(TRUNK + SUF[k], 3, slot_fork=False, cache_prompt=False)
        assert c["tokens"] == fresh["tokens"], k


def test_no_cache_idle_slots_keeps_trunk_shared():
    global server
    server = make_server(no_cache_idle_slots=True)
    server.start()
    build_prompts()
    complete(TRUNK, 0, n_predict=0, checkpoint_at=[-1])
    for k in (1, 2, 3):
        c = complete(TRUNK + SUF[k], k)
        assert c["timings"]["n_fork_tokens"] >= TRUNK_LEN, (k, c["timings"])

    ss = slots()
    check_pool_invariant(ss)
    pool = ss[0]["kv_pool"]
    logical = sum(s["kv_cells"]["private"] + s["kv_cells"]["shared"] for s in ss)
    # the trunk is held once, by four slots
    assert pool["shared"] >= TRUNK_LEN
    assert logical - pool["used"] >= 3 * TRUNK_LEN

    # overwriting the source slot drops only its bits: the forks' cells stay
    complete(tokenize("An unrelated prompt about baking bread at home, step by step."), 0, slot_fork=False, cache_prompt=False)
    ss = slots()
    check_pool_invariant(ss)
    for k in (1, 2, 3):
        assert ss[k]["kv_cells"]["private"] + ss[k]["kv_cells"]["shared"] >= TRUNK_LEN

    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    assert "llamacpp:slot_forks_total 3" in res.body
    assert "llamacpp:kv_cells_used" in res.body and "llamacpp:kv_cells_logical" in res.body


def test_fork_from_busy_source():
    global server
    server = make_server(no_cache_idle_slots=True)
    server.start()
    build_prompts()

    busy_prompt = TRUNK + SUF[4]
    result = {}

    def run_busy():
        result["busy"] = complete(busy_prompt, 0, n_predict=256, checkpoint_at=[TRUNK_LEN])

    th = threading.Thread(target=run_busy)
    th.start()
    # wait until slot 0 is generating. Not next_token.n_decoded: /slots reports it from the slot's PREVIOUS task until
    # the new one generates. n_prompt_tokens past the prompt can only be advanced by the current task.
    deadline = time.time() + 300
    while True:
        s0 = slots()[0]
        if s0["is_processing"] and s0.get("n_prompt_tokens", 0) > len(busy_prompt):
            break
        if not th.is_alive() or time.time() > deadline:
            pytest.fail(f"source never started generating (n_prompt_tokens {s0.get('n_prompt_tokens')})")
        time.sleep(0.05)

    child = complete(TRUNK + SUF[5], 1)
    busy = th.is_alive()
    th.join()
    t = child["timings"]
    assert_forked(child, "busy source", TRUNK_LEN, src=0)
    assert busy, "source finished before the child returned: not a fork from a busy source"

    # child and source ran concurrently, the baselines alone: batch composition alone can move greedy
    # output on a unified KV pool (RTG-57), so only the head is asserted here. KPF-18 item 1/3 measure this
    # against the matched control (same composition, fork off).
    fresh_child = complete(TRUNK + SUF[5], 2, slot_fork=False, cache_prompt=False)
    assert child["tokens"][:4] == fresh_child["tokens"][:4]
    print("busy-source child: full greedy match =", child["tokens"] == fresh_child["tokens"])

    fresh_busy = complete(busy_prompt, 3, n_predict=256, slot_fork=False, cache_prompt=False)
    assert result["busy"]["tokens"][:4] == fresh_busy["tokens"][:4]
    print("busy source: full greedy match =", result["busy"]["tokens"] == fresh_busy["tokens"])


def test_checkpoint_at_is_pinned():
    server.start()
    build_prompts()
    body = complete(TRUNK, 0, n_predict=0, checkpoint_at=[64, -1])
    s0 = slots()[0]
    ck = {c["n_tokens"]: c for c in s0["checkpoints"]}
    kind_needs_ckpt = len(s0["checkpoints"]) > 0
    if not kind_needs_ckpt:
        pytest.skip("pure-attention model: checkpoints are not used (kv fork mode)")
    assert ck[64]["pinned"] and ck[TRUNK_LEN]["pinned"], s0["checkpoints"]
    assert body["timings"]["prompt_n"] == TRUNK_LEN


def test_fork_chain_holds_each_junction_checkpoint_once():
    # trunk with a pinned end checkpoint, then a chain of children (end fork, then checkpoint forks): no slot may hold
    # two checkpoints of the same state (n_tokens, pos_max); on a 27B each one is ~150 MiB
    server.no_cache_idle_slots = True
    server.start()
    build_prompts()
    complete(TRUNK, 0, n_predict=0, checkpoint_at=[-1])
    if not slots()[0]["checkpoints"]:
        pytest.skip("pure-attention model: checkpoints are not used (kv fork mode)")
    for k in (1, 2, 3):
        body = complete(TRUNK + SUF[k], k)
        assert body["timings"]["n_fork_tokens"] == TRUNK_LEN, body["timings"]
    for s in slots():
        keys = [(c["n_tokens"], c["pos_max"]) for c in s["checkpoints"]]
        assert len(keys) == len(set(keys)), (s["id"], s["checkpoints"])
        at_junction = [c for c in s["checkpoints"] if c["n_tokens"] == TRUNK_LEN]
        if s["id"] > 0:
            assert len(at_junction) == 1 and at_junction[0]["pinned"], (s["id"], s["checkpoints"])


def test_pinned_junction_slot_file_roundtrip():
    # the slot file layout is upstream's (llama payload | SCKP | SDFT) and has no `pinned` flag: a pinned fork
    # junction saved to a slot file comes back as an ordinary (unpinned) checkpoint. The restore must still work
    # and a later fork must match fresh (known limitation, see RECUT2.md).
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="kpf-slotfile-")
    try:
        server.slot_save_path = tmp
        server.no_cache_idle_slots = True
        server.start()
        build_prompts()
        complete(TRUNK, 0, n_predict=0, checkpoint_at=[-1])
        child = complete(TRUNK + SUF[1], 1)  # slot 1 inherits a pinned junction through the fork
        assert child["timings"]["n_fork_tokens"] == TRUNK_LEN, child["timings"]
        s1 = slots()[1]
        if not s1["checkpoints"]:
            pytest.skip("pure-attention model: checkpoints are not used (kv fork mode)")
        assert any(c["pinned"] and c["n_tokens"] == TRUNK_LEN for c in s1["checkpoints"]), s1["checkpoints"]
        n_ckpt = len(s1["checkpoints"])

        res = server.make_request("POST", "/slots/1?action=save", data={"filename": "junction.bin"})
        assert res.status_code == 200, res.body
        assert res.body["n_saved"] > 0
        res = server.make_request("POST", "/slots/2?action=restore", data={"filename": "junction.bin"})
        assert res.status_code == 200, res.body
        assert res.body["n_restored"] > 0

        s2 = slots()[2]
        assert len(s2["checkpoints"]) == n_ckpt, (s2["checkpoints"], n_ckpt)
        assert not any(c["pinned"] for c in s2["checkpoints"]), s2["checkpoints"]  # pinned=false after restore

        # a later request on the restored slot, and a fork whose source is the restored slot, both match fresh
        again = complete(TRUNK + SUF[2], 2)
        fresh2 = complete(TRUNK + SUF[2], 3, slot_fork=False, cache_prompt=False)
        assert again["tokens"] == fresh2["tokens"]
        assert again["timings"]["cache_n"] >= TRUNK_LEN, again["timings"]  # the restored state was reused
        later = complete(TRUNK + SUF[3], 0)
        fresh3 = complete(TRUNK + SUF[3], 3, slot_fork=False, cache_prompt=False)
        assert later["tokens"] == fresh3["tokens"]
        check_pool_invariant(slots())
    finally:
        server.slot_save_path = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoint_at_rejects_bad_input():
    server.start()
    build_prompts()
    res = server.make_request("POST", "/completion", data={"prompt": TRUNK, "n_predict": 0, "checkpoint_at": "end"})
    assert res.status_code == 400
    res = server.make_request("POST", "/completion", data={"prompt": TRUNK, "n_predict": 0, "checkpoint_at": [0]})
    assert res.status_code == 400


# ---- P2 interface (workspace-89): content epoch, prefix-hash ladder, ids in timings, /props, message checkpoints


def test_p2_epoch_hash_ladder_and_ids():
    global server
    server = make_server(no_cache_idle_slots=True)
    server.start()
    build_prompts()

    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    sf = res.body["slot_fork"]
    assert sf["min_tokens"] == FORK_MIN and sf["mode"] in ("kv", "checkpoint")

    complete(TRUNK, 0, n_predict=0, checkpoint_at=[-1])
    e0 = {s["id"]: s["content_epoch"] for s in slots()}

    c = complete(TRUNK + SUF[1], 1)
    t = c["timings"]
    assert_forked(c, "fork-into epoch", FORK_MIN, src=0)  # the epoch / ladder assertions below are about a fork
    assert t["id_slot"] == 1 and t["id_task"] >= 0
    ss = slots()
    assert ss[1]["id_task"] == t["id_task"]
    # fork-into bumps the destination; the source is untouched
    assert ss[1]["content_epoch"] > e0[1]
    assert ss[0]["content_epoch"] == e0[0]
    # the ladder agrees on the shared trunk
    src_h = {ck["n_tokens"]: ck["prefix_hash"] for ck in ss[0]["checkpoints"]}
    dst_h = {ck["n_tokens"]: ck["prefix_hash"] for ck in ss[1]["checkpoints"]}
    for n in set(src_h) & set(dst_h):
        assert src_h[n] == dst_h[n], n
    if sf["mode"] == "kv":
        assert ss[0]["prefix_hash"] != ss[1]["prefix_hash"]

    # an appending continuation on the same slot does not bump the epoch
    e1 = ss[1]["content_epoch"]
    complete(TRUNK + SUF[1] + c["tokens"], 1, n_predict=4)
    assert slots()[1]["content_epoch"] == e1

    # a diverging prompt on the same slot truncates below what it held: bump
    complete(TRUNK + SUF[2], 1, n_predict=4, slot_fork=False)
    assert slots()[1]["content_epoch"] > e1

    # OAI chat timings carry the ids (non-streamed and streamed)
    msgs = [{"role": "user", "content": "Say hello."}]
    res = server.make_request("POST", "/v1/chat/completions", data={"messages": msgs, "max_tokens": 4, "temperature": 0})
    assert res.status_code == 200
    assert "id_slot" in res.body["timings"] and "id_task" in res.body["timings"]
    res = server.make_request("POST", "/v1/responses", data={"input": "Say hello.", "max_output_tokens": 4, "temperature": 0})
    assert res.status_code == 200
    assert "id_slot" in res.body["timings"] and "id_task" in res.body["timings"]


def test_p2_checkpoint_at_message_index():
    server.start()
    build_prompts()
    msgs = [{"role": "system", "content": "You answer questions about a lighthouse log."},
            {"role": "user", "content": TRUNK_TEXT},
            {"role": "user", "content": "Question: how many volumes did the log have?"}]
    # the junction a fan-out trunk needs: the end of the shared messages (here system + first user message)
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": msgs, "max_tokens": 2, "temperature": 0, "id_slot": 0,
        "checkpoint_at": [{"message": 1, "at": "end"}]})
    assert res.status_code == 200
    if server_mode() == "kv":
        pytest.skip("kv mode (pure attention): checkpoints are not used")
    ck = slots()[0]["checkpoints"]
    pinned = [c for c in ck if c["pinned"]]
    assert pinned, ck
    n_prompt = res.body["usage"]["prompt_tokens"]
    assert all(0 < c["n_tokens"] < n_prompt for c in pinned)

    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": msgs, "max_tokens": 2, "checkpoint_at": [{"message": "x"}]})
    assert res.status_code == 400


def test_p2_off_has_no_new_fields():
    global server
    server = make_server(fork_min=0)
    server.start()
    build_prompts()
    res = server.make_request("GET", "/props")
    assert "slot_fork" not in res.body
    complete(TRUNK, 0, n_predict=2)
    s0 = slots()[0]
    for k in ("content_epoch", "prefix_hash", "kv_cells", "kv_pool", "fork", "checkpoints"):
        assert k not in s0
    res = server.make_request("POST", "/v1/responses", data={"input": "Say hello.", "max_output_tokens": 2})
    assert res.status_code == 200 and "timings" not in res.body


def server_mode() -> str:
    return server.make_request("GET", "/props").body.get("slot_fork", {}).get("mode", "none")


# ---------------------------------------------------------------------------------------------------------------
# [KPF-17.1 .. 17.4, 19.1] deterministic conformance cases. Each docstring names the MUTATION (env LLAMA_KPF_MUTATE
# on the server) that removes its guard; the mutation run must make the case FAIL (see the cpu-queue job scripts).
# ---------------------------------------------------------------------------------------------------------------

LOGPROB_FLOOR = 0.05  # fresh-vs-fresh floor on the first generated token's logprob (CPU, differing batch shapes)
# A greedy flip is accepted only where the fresh run's top-2 logprobs are this close. Any change of float evaluation order
# (batch boundaries, batch composition, KV cell placement) moves CPU logprobs: measured 2026-10-20 on gemma-3-1b, the
# slot's OWN checkpoint continuation (no fork at all), the fork and a fresh prefill of TRUNK+SUF[1] differ by up to 0.06
# nats at token 1, and at token 3 the top-2 (19566 vs 3026) are 0.07-0.25 nats apart, so job 23's "fork vs fresh"
# divergence at index 3 was that near-tie flipping, not the fork (the logit-level floor is in test-kpf-swa-checkpoint).
TIE_MARGIN = 0.5


def need_ckpt_mode():
    if server_mode() in ("kv", "none"):
        pytest.skip("pure-attention model (kv fork mode): checkpoints are not used")


def fresh(prompt: list[int], id_slot: int, n_predict: int = 24, **extra) -> dict:
    # top-2 per position: same_as_fresh needs the runner-up and its margin
    extra["n_probs"] = max(2, int(extra.get("n_probs", 0)))
    return complete(prompt, id_slot, n_predict=n_predict, slot_fork=False, cache_prompt=False, **extra)


def first_logprob(body: dict):
    try:
        return body["completion_probabilities"][0]["logprob"]
    except (KeyError, IndexError, TypeError):
        return None


def same_as_fresh(a: dict, b: dict, what: str):
    """a equals the fresh run b token for token, except that ONE greedy flip to b's runner-up is accepted at a position
    where b's top-2 are within TIE_MARGIN; past such a flip the two continuations legitimately differ, so the comparison
    stops there. A wrong state diverges where b is confident, or to a token outside b's top-2."""
    ta, tb = a["tokens"], b["tokens"]
    probs = b.get("completion_probabilities") or []
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x == y:
            continue
        tops = probs[i].get("top_logprobs", []) if i < len(probs) else []
        margin = tops[0]["logprob"] - tops[1]["logprob"] if len(tops) >= 2 else None
        assert margin is not None and margin < TIE_MARGIN and x == tops[1]["id"] and y == tops[0]["id"], \
            (what, f"diverged at {i}", ta, tb, tops[:2])
        print(f"{what}: greedy flip at a near-tie of the fresh run, index {i}, top-2 margin {margin:.3f} nats")
        break
    else:
        assert len(ta) == len(tb), (what, ta, tb)
    la, lb = first_logprob(a), first_logprob(b)
    if la is not None and lb is not None:
        assert abs(la - lb) < LOGPROB_FLOOR, (what, la, lb)


def ckpts(i: int) -> list[dict]:
    return slots()[i].get("checkpoints", [])


def assert_ckpt_offsets(ss: list[dict], what: str):
    """every checkpoint's recorded n_tokens must be pos_max + 1 (positions == token indices on text models)"""
    for s in ss:
        for c in s.get("checkpoints", []):
            assert c["pos_max"] + 1 == c["n_tokens"], (what, s["id"], c)


def diverge_token(prompt: list[int], n: int) -> int:
    for cand in (tokenize(" zebra")[0], tokenize(" lighthouse")[0], tokenize(" quantum")[0]):
        if cand != prompt[n]:
            return cand
    raise AssertionError("no diverging token")


def erase_slot(i: int):
    res = server.make_request("POST", f"/slots/{i}?action=erase")
    assert res.status_code == 200, res.body


def test_kpf171_pinned_ntokens_equal_requested():
    """KPF-17.1(1): ckpt.n_tokens == its requested checkpoint_at position, also while another slot runs draft-verify
    batches (KPF_SPEC_TYPE=draft-mtp); fork / slot-file / clone-to-fork / prompt-cache round trips keep it.
    MUTATION: ckpt_ntokens_skew (n_tokens recorded one past the data) -> the /slots offset assertion and the fork fail."""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="kpf171-")
    try:
        server.slot_save_path = tmp
        server.n_batch = server.n_batch or 40
        server.n_ubatch = server.n_batch
        # with cache_idle_slots on (the default) every launch purges every idle slot into the RAM cache: slot 0 would be
        # empty by the slot-file step below and that step would compare two empty lists (job 23 ran it so, vacuously)
        server.no_cache_idle_slots = True
        server.start()
        build_prompts()
        need_ckpt_mode()
        req = [48, 96, TRUNK_LEN]
        # slot 1 decodes (draft-verify batches under MTP) while slot 0 prefills in 40-token chunks
        dec: dict = {}
        t = threading.Thread(target=lambda: dec.update(complete(TRUNK[:60] + SUF[0], 1, n_predict=48)))
        t.start()
        body = complete(TRUNK, 0, n_predict=0, checkpoint_at=req)
        t.join()
        pinned = sorted(c["n_tokens"] for c in ckpts(0) if c["pinned"])
        assert pinned == sorted(req), (pinned, req)
        assert_ckpt_offsets(slots(), "after prefill")

        # restored fork == fresh, junction positions preserved by clone-to-fork
        child = complete(TRUNK + SUF[1], 2)
        assert child["timings"]["n_fork_tokens"] == TRUNK_LEN, child["timings"]
        same_as_fresh(child, fresh(TRUNK + SUF[1], 3, n_probs=1), "fork vs fresh")
        inherited = sorted(c["n_tokens"] for c in ckpts(2) if c["pinned"])
        assert inherited and set(inherited) <= set(req), inherited
        assert_ckpt_offsets(slots(), "after fork")

        # slot-file round trip keeps n_tokens
        assert sorted(c["n_tokens"] for c in ckpts(0) if c["pinned"]) == sorted(req), ckpts(0)
        res = server.make_request("POST", "/slots/0?action=save", data={"filename": "k.bin"})
        assert res.status_code == 200, res.body
        res = server.make_request("POST", "/slots/3?action=restore", data={"filename": "k.bin"})
        assert res.status_code == 200, res.body
        assert sorted(c["n_tokens"] for c in ckpts(3)) == sorted(c["n_tokens"] for c in ckpts(0))
        assert_ckpt_offsets(slots(), "after slot-file restore")

        # prompt-cache round trip (evict slot 0 to the RAM cache with an unrelated prompt, then load it back)
        other = tokenize("Completely unrelated text about volcanoes and the way molten rock moves through the crust. " * 3)
        complete(other, 0, n_predict=2)
        back = complete(TRUNK + SUF[2], 0)
        same_as_fresh(back, fresh(TRUNK + SUF[2], 3, n_probs=1), "prompt-cache round trip")
        assert_ckpt_offsets(slots(), "after prompt-cache round trip")
    finally:
        server.slot_save_path = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_kpf171_offset_oracle_under_concurrency():
    """KPF-17.1(2): np=4, n_batch=40 so chunks end at arbitrary offsets; for EVERY checkpoint (and the idle source's
    'end' state) a fork probe whose prompt equals source[:n_tokens]+diverging token must fork at exactly n_tokens (no
    silent skip by the ckpt.pos_max + 1 != p guard) and equal a cold prefill.
    MUTATION: ckpt_ntokens_skew -> pos_max+1 != n_tokens (asserted) / skipped probes; with no_posmax_guard added the
    stale-labelled restore diverges from the cold prefill."""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="kpf171b-")
    try:
        server.slot_save_path = tmp
        server.n_batch = 40
        server.n_ubatch = 40
        # every probe launch would purge the three idle sources (cache_idle_slots): job 23's second probe found no
        # source at all (n_fork_tokens 0, kind none). The oracle needs the sources to stay.
        server.no_cache_idle_slots = True
        server.start()
        build_prompts()
        need_ckpt_mode()
        prompts = {i: TRUNK + SUF[i] for i in range(3)}
        ats = [37, 71, 93, -1]
        res: dict = {}

        def run(i):
            res[i] = complete(prompts[i], i, n_predict=4, checkpoint_at=ats)

        th = [threading.Thread(target=run, args=(i,)) for i in prompts]
        [x.start() for x in th]
        [x.join() for x in th]
        assert_ckpt_offsets(slots(), "after concurrent prefill")
        probes = 0
        for i, pr in prompts.items():
            cks = ckpts(i)
            # the end-state of the idle source (kind "end") is probed with n = len(prompt) + n_predict - 1 tokens held
            targets = sorted({c["n_tokens"] for c in cks})
            for n in targets:
                if n < FORK_MIN or n >= len(pr):
                    continue
                probe = pr[:n] + [diverge_token(pr, n)] + SUF[3]
                erase_slot(3)
                child = complete(probe, 3, n_predict=4)
                assert child["timings"]["n_fork_tokens"] == n, (i, n, child["timings"], cks)
                same_as_fresh(child, fresh(probe, 3, n_predict=4, n_probs=1), f"slot {i} ckpt {n}")
                probes += 1
        assert probes >= 6, probes
        check_pool_invariant(slots())
    finally:
        server.slot_save_path = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_kpf171_abort_then_retry():
    """KPF-17.1(3): abort a trunk-first request at several offsets mid-prefill, send a sibling right after the cancel lands:
    output == fresh and the fork never reads the aborted slot's cells beyond its decoded n_past.

    How a slot cancelled mid-prefill is treated (server-context.cpp): the cancel is a queue task, handled between
    update_slots iterations (SERVER_TASK_TYPE_CANCEL -> release()); prompt tokens are appended only in the iteration
    whose batch decodes them, and a failed decode prompt_clear()s every processing slot. So at any fork launch a source's
    prompt tokens equal its decoded cells, and the kv-mode guard `src_pos_max < p - 1` in find_fork_source is
    defence in depth that no reachable state exercises: LLAMA_KPF_MUTATE=no_posmax_guard ALONE is undetectable by
    construction (job 22, 2026-10-20). In kv mode this case asserts that invariant directly (n_prompt_tokens == cells).
    MUTATIONS (kv mode, pure-attention model): abort_keeps_tail is a fault injection (a cancelled mid-prefill slot keeps
    the next n_batch prompt tokens it never decoded, i.e. the state upstream's "handle ret == 2 (abort)" TODO would create)
    -> must still PASS, the guard skips that source; abort_keeps_tail,no_posmax_guard -> must FAIL (forks cells that
    were never computed). On hybrids the checkpoint list only holds decoded positions, so this case is the output oracle.

    Job 22 never aborted anything: the old request was non-streaming, so requests.post returned only after the server
    finished (every slot-0 release showed the full 703 tokens). The client now hangs up with a read timeout, and then
    sends NOTHING for 1.2 s: a non-streaming handler notices the hang-up only when its 1 s result wait times out
    (server_response::recv_with_timeout restarts the full wait on every notify, i.e. on ANY task's result), so polling
    /slots right away starves the disconnect check and the task runs to completion. The cancel therefore lands on a
    1 s tick of the request's wait; the delays below put the hang-up before ticks 1, 1, 2, 2 and 3."""
    import requests
    inject = "abort_keeps_tail" in os.environ.get("LLAMA_KPF_MUTATE", "").split(",")
    server.n_batch = 40
    server.n_ubatch = 40
    server.start()
    build_prompts()
    kv = server_mode() == "kv"
    # long enough that a cancel landing up to ~1 s (HTTP_POLLING_SECONDS) after the hang-up is still mid-prefill: TRUNK*12
    # (1647 tokens) prefilled in 1.2 s on gemma-3-1b with 24 threads and no round was cancelled. TRUNK*24 keeps slot 0 +
    # the sibling + the fresh run inside the 8192-cell pool.
    big = TRUNK * 24
    url = f"http://{server.server_host}:{server.server_port}/completion"
    aborted, tail_seen, forked = 0, 0, 0
    for k, delay in enumerate((0.1, 0.5, 1.2, 1.6, 2.2)):
        try:
            requests.post(url, json={"prompt": big + SUF[0], "id_slot": 0, "n_predict": 4, "cache_prompt": True,
                                     "checkpoint_at": [TRUNK_LEN, 2 * TRUNK_LEN, 3 * TRUNK_LEN]}, timeout=(10, delay))
        except requests.exceptions.ReadTimeout:
            pass  # the client hung up: the server cancels the task at its next poll
        time.sleep(1.2)  # no traffic: let the handler's 1 s wait time out and see the closed connection
        for _ in range(200):
            if not slots()[0]["is_processing"]:
                break
            time.sleep(0.05)
        s0 = slots()[0]
        assert not s0["is_processing"], s0
        n_tok = s0.get("n_prompt_tokens", 0)
        cells = s0["kv_cells"]["private"] + s0["kv_cells"]["shared"]
        aborted += int(n_tok < len(big))
        if kv:
            if inject:
                tail_seen += int(n_tok > cells)
            else:
                assert n_tok == cells, (delay, n_tok, cells)  # the invariant that makes the guard unreachable
        sib_prompt = big + SUF[1 + k]
        sib = complete(sib_prompt, 1, n_predict=8)
        n_f = sib["timings"]["n_fork_tokens"]
        forked += int(n_f > 0)
        print(f"abort@{delay}: slot 0 n_prompt_tokens {n_tok} cells {cells}, sibling n_fork_tokens {n_f} "
              f"({sib['timings'].get('fork_src_kind')})")
        if kv:
            assert n_f <= cells, (delay, n_f, cells, sib["timings"])  # never beyond the aborted slot's decoded cells
            assert n_f == 0 or sib["timings"]["fork_src_slot"] == 0, sib["timings"]
        same_as_fresh(sib, fresh(sib_prompt, 2, n_predict=8, n_probs=1), f"abort@{delay}")
        assert_ckpt_offsets(slots(), f"abort@{delay}")
        for i in (0, 1, 2):  # slot 2 (the fresh run) shares big with the next sibling: it must not be a fork source
            erase_slot(i)
    assert aborted >= 1, "no round was cancelled mid-prefill: the case would be vacuous"
    if kv and inject:
        assert tail_seen >= 1, "fault injection abort_keeps_tail never created a source ahead of its cells"
    else:
        # the guard skips a source that is ahead of its cells (kind none) only under the injection
        assert forked >= 1, "no sibling forked from the aborted slot: the case would be vacuous"
    check_pool_invariant(slots())


def test_kpf171_key_check_negative_control():
    """KPF-17.1(4): P1=S+a, P2=S+b with a[0]!=b[0]; run P1,P2,P1: P2 never restores P1's junction beyond S.
    MUTATION: fork_no_lcp (the token key ignored) = the forced stale restore; P2 then forks beyond S and diverges."""
    server.start()
    build_prompts()
    need_ckpt_mode()
    a, b = SUF[0], SUF[4]
    assert a[0] != b[0]
    p1, p2 = TRUNK + a, TRUNK + b
    f1, f2 = fresh(p1, 3, n_probs=1), fresh(p2, 2, n_probs=1)
    assert f1["tokens"] != f2["tokens"]  # the control is sensitive: stale state would show
    complete(p1, 0, n_predict=24, checkpoint_at=[TRUNK_LEN, TRUNK_LEN + 3, -1])
    r2 = complete(p2, 1)
    assert_forked(r2, "P2 after P1", FORK_MIN, src=0)  # the restore up to S must happen, or nothing is tested
    assert r2["timings"]["n_fork_tokens"] <= TRUNK_LEN, r2["timings"]
    same_as_fresh(r2, f2, "P2 after P1")
    erase_slot(1)
    r1 = complete(p1, 2)
    same_as_fresh(r1, f1, "P1 again")


def test_kpf172_source_diverges_below_junction():
    """KPF-17.2(1): a source rewinding below its own pinned junction loses that checkpoint; a NEW request with the same
    trunk still forks bit-exactly from the child slot holding the pinned clone (cache_idle_slots on and off).
    MUTATION: fork_skip_ckpt_load (dst never receives the recurrent state)."""
    for idle_cache in (True, False):
        global server
        server.stop()
        server = make_server(no_cache_idle_slots=not idle_cache)
        server.start()
        build_prompts()
        need_ckpt_mode()
        complete(TRUNK, 0, n_predict=0, checkpoint_at=[-1])
        c1 = complete(TRUNK + SUF[1], 1)
        assert c1["timings"]["n_fork_tokens"] == TRUNK_LEN, c1["timings"]
        assert any(c["pinned"] and c["n_tokens"] == TRUNK_LEN for c in ckpts(1))
        if idle_cache:
            # cache_idle_slots on a unified KV: launching c1 already purged the idle source (prompt_clear, pinned
            # checkpoints included), so slot 0 is below the junction now. An explicit rewind request here would purge
            # the idle CHILD the same way (job 23 did that: c2 then forked 18 tokens from slot 0's new prompt), so the
            # real case is the next trunk request reaching the child before any other launch.
            s0 = slots()[0]
            assert s0["kv_cells"]["private"] + s0["kv_cells"]["shared"] == 0, s0
        else:
            # slot 0 rewinds below the junction
            complete(TRUNK[:40] + SUF[2], 0, n_predict=2, slot_fork=False)
        assert not [c for c in ckpts(0) if c["n_tokens"] >= TRUNK_LEN], ckpts(0)
        c2 = complete(TRUNK + SUF[3], 2)
        assert c2["timings"]["n_fork_tokens"] == TRUNK_LEN, (idle_cache, c2["timings"])
        assert c2["timings"]["fork_src_slot"] == 1, c2["timings"]
        same_as_fresh(c2, fresh(TRUNK + SUF[3], 3, n_probs=1), f"idle_cache={idle_cache}")
        check_pool_invariant(slots())


def test_kpf172_fork_same_iteration_as_source_erase():
    """KPF-17.2(2): a fork launched in the same update_slots iteration in which the source's checkpoint erase loop
    (--ctx-checkpoints 2) or prompt_clear runs: dst holds its own cloned checkpoint, output == fresh. Timing is raced
    over 8 rounds. MUTATION: fork_skip_ckpt_load."""
    global server
    server.stop()
    server = make_server(ctx_checkpoints=2)
    server.start()
    build_prompts()
    need_ckpt_mode()
    n_forks = 0
    for r in range(8):
        erase_slot(0)
        erase_slot(1)
        complete(TRUNK, 0, n_predict=0, checkpoint_at=[-1])
        bar = threading.Barrier(2)
        out: dict = {}

        def src_side():
            bar.wait()
            out["a"] = complete(TRUNK[: 30 + r] + SUF[r % 8], 0, n_predict=4, slot_fork=False, checkpoint_at=[20, -1])

        def dst_side():
            bar.wait()
            out["b"] = complete(TRUNK + SUF[(r + 1) % 8], 1)

        th = [threading.Thread(target=src_side), threading.Thread(target=dst_side)]
        [x.start() for x in th]
        [x.join() for x in th]
        same_as_fresh(out["b"], fresh(TRUNK + SUF[(r + 1) % 8], 2, n_probs=1), f"round {r}")
        n_f = out["b"]["timings"]["n_fork_tokens"]
        if n_f > 0:
            # dst holds its OWN pinned clone of the junction (was `... or True`, i.e. never checked)
            n_forks += 1
            assert out["b"]["timings"]["fork_src_slot"] == 0, out["b"]["timings"]
            assert any(c["pinned"] and c["n_tokens"] == n_f for c in ckpts(1)), (r, n_f, ckpts(1))
        assert_ckpt_offsets(slots(), f"round {r}")
    # the race is real only if dst forked in some round (job 20: 6 of 8 rounds forked from slot 0)
    assert n_forks >= 1, "dst never forked from the source in 8 rounds: the case would be vacuous"


def test_kpf172_fork_from_decoding_source_no_leak():
    """KPF-17.2(3): checkpoint_at beyond the prompt (inside the generation / unverified draft window) creates nothing, and a
    fork taken while the source decodes (KPF_SPEC_TYPE=draft-mtp: k pending drafts) leaves the source untouched.
    MUTATION: fork_clobber_src (fork disturbs its source), fork_skip_ckpt_load.

    Jobs 20 and 21 (2026-10-21) never forked here (no do_slot_fork line in either log), so fork_clobber_src went
    undetected: the wait loop read next_token.n_decoded, which /slots reports from the PREVIOUS task until the new one
    generates (48 = the base run), so it returned at once while slot 0 was still prefilling (n_prompt_tokens 64 < the
    137-token junction), the source held no checkpoint the child could use, and the child prefilled from scratch. Now:
    wait on n_prompt_tokens past the prompt, require the child to have forked from slot 0, check the source's state
    directly (cells == tokens, same length as the unforked run) and compare its tokens against an A/A control (same
    concurrency, child with slot_fork off) instead of a fixed tolerance."""
    server.start()
    build_prompts()
    need_ckpt_mode()
    prompt = TRUNK + SUF[0]
    at = [TRUNK_LEN, len(prompt) + 3, len(prompt) + 10]
    n_src = 96  # long enough that the source is still decoding while the child forks and prefills

    def src_state() -> tuple[int, int]:
        s0 = slots()[0]
        return s0.get("n_prompt_tokens", 0), s0["kv_cells"]["private"] + s0["kv_cells"]["shared"]

    base = complete(prompt, 0, n_predict=n_src, checkpoint_at=at)
    assert all(c["n_tokens"] <= len(prompt) for c in ckpts(0)), ckpts(0)  # beyond-prompt positions never checkpointed
    base_tok, base_cells = src_state()
    assert base_tok == base_cells, (base_tok, base_cells)

    def concurrent(child_fork: bool) -> tuple[dict, dict]:
        for i in (0, 1, 2):
            erase_slot(i)
        out: dict = {}
        t = threading.Thread(target=lambda: out.update(complete(prompt, 0, n_predict=n_src, checkpoint_at=at)))
        t.start()
        deadline = time.time() + 300
        while True:  # slot 0 has decoded >= 3 generated tokens (n_prompt_tokens counts them); never a stale field
            s0 = slots()[0]
            if s0["is_processing"] and s0.get("n_prompt_tokens", 0) >= len(prompt) + 3:
                break
            assert t.is_alive() and time.time() < deadline, ("source never reached decoding", s0.get("n_prompt_tokens"))
            time.sleep(0.02)
        child = complete(TRUNK + SUF[1], 1, n_predict=24, slot_fork=child_fork)
        busy = t.is_alive()
        t.join()
        if child_fork:
            tm = child["timings"]
            assert tm["n_fork_tokens"] >= FORK_MIN and tm["fork_src_slot"] == 0, tm  # the fork path really ran
            assert busy, "source finished before the child returned: not a fork from a decoding source"
        return out, child

    def matched(a: list[int], b: list[int]) -> int:
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    # A/A control: the same concurrency without the fork measures how far batch composition alone moves the source
    ctrl, _ = concurrent(False)
    out, child = concurrent(True)
    tok, cells = src_state()
    # the source's own state, directly: every position it holds is a computed cell, and it holds as many as unforked
    assert tok == cells, ("source cells != its tokens after the fork", tok, cells)
    assert tok == base_tok, ("source length changed by the fork", tok, base_tok)
    m_ctrl, m_fork = matched(ctrl["tokens"], base["tokens"]), matched(out["tokens"], base["tokens"])
    print(f"source vs unforked: A/A control matches {m_ctrl}/{len(base['tokens'])}, forked run {m_fork}")
    if m_ctrl == len(base["tokens"]):
        assert out["tokens"] == base["tokens"], "source continuation changed by the fork (A/A control is exact)"
    else:
        assert m_fork >= m_ctrl, ("source diverged earlier than the A/A control", m_fork, m_ctrl)
    same_as_fresh(child, fresh(TRUNK + SUF[1], 2, n_predict=24, n_probs=1), "fork from decoding source")
    assert_ckpt_offsets(slots(), "decoding-source fork")


def test_kpf172_fork_from_checkpoint_of_current_step():
    """KPF-17.2(4): siblings arriving at staggered moments while the trunk-first source creates its pinned checkpoints
    never fork from a half-built one: every sibling == fresh, forks land only on completed junctions.
    MUTATION: ckpt_ntokens_skew + no_posmax_guard (a mislabelled / not-yet-valid checkpoint becomes selectable)."""
    server.n_batch = 40
    server.n_ubatch = 40
    server.start()
    build_prompts()
    need_ckpt_mode()
    big = TRUNK * 8
    js = [TRUNK_LEN, 2 * TRUNK_LEN, 3 * TRUNK_LEN]
    # Job 20 (tiny hybrid) never exercised this: every sibling forked from an IDLE source (the trunk had finished, and
    # the trunk request itself forked from the previous round's fresh run on slot 2 instead of prefilling). Now the
    # trunk prefills from scratch (slot_fork off, slots 2/3 erased), the delays are fractions of a measured solo
    # prefill, and at least one sibling must fork from slot 0 while it is still prefilling.
    solo = complete(big + SUF[0], 3, n_predict=1, slot_fork=False, cache_prompt=False)
    t_pp = solo["timings"]["prompt_ms"] / 1000.0
    erase_slot(3)
    mid_forks = 0
    for k, frac in enumerate((0.1, 0.3, 0.5, 0.7, 0.9)):
        delay = frac * t_pp
        for i in (0, 1, 2):
            erase_slot(i)
        t = threading.Thread(target=lambda: complete(big + SUF[0], 0, n_predict=2, checkpoint_at=js, slot_fork=False))
        t.start()
        time.sleep(delay)
        s0 = slots()[0]
        mid = s0["is_processing"] and s0.get("n_prompt_tokens", 0) < len(big)
        sib_prompt = big[: 2 * TRUNK_LEN] + SUF[1 + k]
        sib = complete(sib_prompt, 1, n_predict=8)
        t.join()
        n_f = sib["timings"]["n_fork_tokens"]
        print(f"delay {delay:.2f}s ({frac} of prefill): source mid-prefill {mid} (n_prompt_tokens "
              f"{s0.get('n_prompt_tokens')}), sibling n_fork_tokens {n_f} from slot {sib['timings'].get('fork_src_slot')}")
        assert n_f == 0 or n_f in js or n_f >= 2 * TRUNK_LEN - 1, (delay, sib["timings"])
        if mid and n_f > 0 and sib["timings"]["fork_src_slot"] == 0:
            mid_forks += 1
        same_as_fresh(sib, fresh(sib_prompt, 2, n_predict=8, n_probs=1), f"delay {delay:.2f}")
        assert_ckpt_offsets(slots(), f"delay {delay:.2f}")
    assert mid_forks >= 1, "no sibling forked from the source while it was creating its junctions: vacuous"


def pinned_of(i: int) -> list[int]:
    return [c["n_tokens"] for c in ckpts(i) if c["pinned"]]


def seq_prompt(k: int) -> list[int]:
    return (TRUNK + SUF[0] + SUF[1] + SUF[2] + SUF[3] + SUF[4] + SUF[5])[: 40 + 12 * k]


def test_kpf173_pinned_cap_per_slot():
    """KPF-17.3: --slot-pinned-checkpoints 3: more sequential checkpoint_at requests than the cap never exceed it; the
    newest junction survives and the oldest drops first; forked children's inherited clones count toward their own cap; an
    idle slot keeps its pinned junctions after ordinary release and loses them only on prompt_clear (erase).
    Checkpoint memory is bounded. MUTATION: pin_cap_off."""
    global server
    server.stop()
    server = make_server(ctx_checkpoints=8, slot_pinned_checkpoints=3)
    server.start()
    build_prompts()
    need_ckpt_mode()
    cap, lens = 3, []
    n_ctx_ckpt = 8
    max_one = 0.0
    for k in range(10):
        pr = seq_prompt(k)
        lens.append(len(pr))
        complete(pr, 0, n_predict=0, checkpoint_at=[-1])
        p = pinned_of(0)
        assert len(p) <= cap, (k, p)
        assert p == lens[-cap:], (k, p, lens)  # newest kept, oldest dropped first
        # bounded over the series: the COUNT is bounded (--ctx-checkpoints, pinned capped) and each checkpoint is one
        # state. A checkpoint's size is the state size, constant for a recurrent state but growing with the token count
        # up to the window on an SWA cache, so "total size plateaus" (job 23's assertion) only holds on hybrids
        cks = ckpts(0)
        assert len(cks) <= max(n_ctx_ckpt, cap + 1), (k, cks)
        max_one = max([max_one] + [c["size_mib"] for c in cks])
        pinned_mib = sum(c["size_mib"] for c in cks if c["pinned"])
        assert pinned_mib <= cap * max_one + 1e-6, (k, pinned_mib, max_one, cks)
        assert sum(c["size_mib"] for c in cks) <= max(n_ctx_ckpt, cap + 1) * max_one + 1e-6, (k, cks)
    # ordinary release did not drop them (the slot is idle now)
    assert pinned_of(0) == lens[-cap:]
    # a child inherits the clones and they count toward ITS cap
    child = complete(seq_prompt(9) + SUF[6], 1, n_predict=2)
    assert child["timings"]["n_fork_tokens"] > 0, child["timings"]
    assert 0 < len(pinned_of(1)) <= cap, pinned_of(1)
    for k in range(10, 16):
        complete(seq_prompt(k) + SUF[6], 1, n_predict=0, checkpoint_at=[-1])
        assert len(pinned_of(1)) <= cap, (k, pinned_of(1))
    # prompt_clear releases
    erase_slot(0)
    assert pinned_of(0) == []


def test_kpf173_cap_zero_equals_per_request_bound():
    """KPF-17.3: --slot-pinned-checkpoints 0 reproduces today's behaviour (pinned accumulate across requests; only the
    per-request checkpoint_at bound applies)."""
    global server
    server.stop()
    server = make_server(ctx_checkpoints=4, slot_pinned_checkpoints=0)
    server.start()
    build_prompts()
    need_ckpt_mode()
    for k in range(9):
        complete(seq_prompt(k), 0, n_predict=0, checkpoint_at=[-1])
    assert len(pinned_of(0)) == 9, pinned_of(0)  # > --ctx-checkpoints: no per-slot bound, as before


# KPF-17.4 (SWA): LLAMA_TEST_FORK_MODEL=gemma-3-1b-it (n_swa 512). Records the pinned junction pos_min against the
# acceptance window and requires fork == fresh. The C++ test tests/test-kpf-swa-checkpoint.cpp carries the logit-level
# comparison and the real-vs-recorded pos_min mutation.
def test_kpf174_swa_pinned_junction_fork_equals_fresh():
    """MUTATION: ckpt_ntokens_skew, fork_skip_ckpt_load."""
    server.n_batch = 256
    server.n_ubatch = 256
    server.start()
    build_prompts()
    props = server.make_request("GET", "/props").body
    if props.get("slot_fork", {}).get("mode") != "checkpoint":
        pytest.skip("not an SWA/hybrid checkpoint-mode model")
    long_t = tokenize(TRUNK_TEXT * 8)
    if len(long_t) < 700:
        pytest.skip("trunk shorter than the SWA window")
    complete(long_t, 0, n_predict=0, checkpoint_at=[len(long_t) // 2, -1])
    rec = [{"n_tokens": c["n_tokens"], "pos_min": c["pos_min"], "pos_max": c["pos_max"], "pinned": c["pinned"]}
           for c in ckpts(0)]
    rp = os.environ.get("KPF_RECORD")
    if rp:
        with open(rp, "a") as f:
            f.write(json.dumps({"case": "kpf174_swa_ckpt_ranges", "ckpts": rec, "n_trunk": len(long_t)}) + "\n")
    child = complete(long_t + SUF[1], 1)
    assert_forked(child, "swa junction fork", len(long_t) // 2, src=0)
    same_as_fresh(child, fresh(long_t + SUF[1], 2, n_probs=1), "swa junction fork")
    assert_ckpt_offsets(slots(), "swa")


def test_kpf191_fork_drafter_state_equals_unforked():
    """KPF-19.1: forking an MTP-drafter slot gives the destination the same drafter state as an unforked slot: same
    tokens and same draft_n / draft_n_accepted. Run with and without KPF_DFT_FULL=1 (--slot-fork-dft-full).
    On a PLAIN drafter cache the attention-only share already hands dst the whole drafter KV, so the fork is independent of
    what a PARTIAL_ONLY load does: MUTATION dft_partial_noop (emulates #30514) must PASS with the flag off AND on (job 40
    found it undetected; that is the correct reading, not a blind spot). The guard that the drafter case is NOT vacuous is
    MUTATION dft_no_transfer (dst gets no drafter state): it must FAIL with the flag off and on."""
    if not os.environ.get("KPF_SPEC_TYPE"):
        pytest.skip("needs KPF_SPEC_TYPE=draft-mtp on an MTP model")
    server.start()
    build_prompts()
    need_ckpt_mode()
    complete(TRUNK, 0, n_predict=0, checkpoint_at=[-1])
    child = complete(TRUNK + SUF[1], 1, n_predict=32)
    assert child["timings"]["n_fork_tokens"] == TRUNK_LEN, child["timings"]
    ref = fresh(TRUNK + SUF[1], 2, n_predict=32)
    assert ref["timings"].get("draft_n", 0) > 0, ref["timings"]
    assert child["tokens"] == ref["tokens"]
    assert child["timings"].get("draft_n") == ref["timings"].get("draft_n"), (child["timings"], ref["timings"])
    assert child["timings"].get("draft_n_accepted") == ref["timings"].get("draft_n_accepted"), (child["timings"], ref["timings"])
