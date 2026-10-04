# [KPF-17] cross-slot prefix fork (--slot-fork-min-tokens), server side.
#
# Model: LLAMA_TEST_FORK_MODEL=/path/model.gguf (a hybrid such as Qwen3.5-0.8B exercises the checkpoint
# path, a pure-attention model the kv path); without it the stock tinyllama preset is used.
# Port: PORT env, default 18471 (never a standard port).
#
# Prompts are natural text tokenized by the running server, concatenated at the token level so the shared
# prefix is exact (random token ids make the model emit invalid UTF-8, which /completion rejects). Greedy. "fresh" means the same
# prompt with "slot_fork": false and "cache_prompt": false (full prefill, no fork, no RAM-cache reuse).

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
    # wait until slot 0 is generating
    deadline = time.time() + 300
    while time.time() < deadline:
        s0 = slots()[0]
        if s0["is_processing"] and s0["next_token"][0]["n_decoded"] > 0:
            break
        time.sleep(0.05)
    else:
        pytest.fail("source never started generating")

    child = complete(TRUNK + SUF[5], 1)
    th.join()
    t = child["timings"]
    assert t["n_fork_tokens"] >= TRUNK_LEN, t
    assert t["fork_src_slot"] == 0

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
