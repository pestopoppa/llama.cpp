import json
import os
import pytest
import requests
from utils import *

# PFXC-TOOLPARSE error contract (binding, coordinator 2026-11-20):
#  non-streaming: HTTP 500, {"error":{"code":500,"type":"tool_call_parse_error","message":...,"retryable":true,"raw_excerpt":<=256 chars}}
#  streaming:     HTTP 200, one SSE event `data: {"error":{...same object...}}`, then the stream closes WITHOUT `data: [DONE]`;
#                 no tool_calls delta for the bad call is emitted before it (tool calls are buffered until the final parse).
#  recoverable (split) cases never fire it.
# The model's output is forced with an assistant prefill (continue_final_message) rendered by the served Qwen3.6-35B-A3B
# template, so any small GGUF works. Set LLAMA_TEST_TOOLPARSE_GGUF to run.

GGUF = os.environ.get("LLAMA_TEST_TOOLPARSE_GGUF")
pytestmark = pytest.mark.skipif(not GGUF, reason="LLAMA_TEST_TOOLPARSE_GGUF not set")

TOOLS = [{"type": "function", "function": {"name": "write", "description": "Write a file", "parameters": {
    "type": "object", "properties": {"content": {"type": "string"}, "filePath": {"type": "string"}}, "required": ["content", "filePath"]}}}]

MALFORMED = "<tool_call>\n<function=write>\n<parameter=content>\nhello</parameter></function><parameter=filePath>\n/a.txt\n</parameter>\n</function>\n</tool_call>"
UNRECOVERABLE = "<tool_call>\n<function=write>\n<parameter=content>\nx<tool_call>y\n</parameter>\n<parameter=filePath>\n/a\n</parameter>\n</function>\n</tool_call>"
GOOD = "<tool_call>\n<function=write>\n<parameter=content>\nhello\n</parameter>\n<parameter=filePath>\n/a.txt\n</parameter>\n</function>\n</tool_call>"


@pytest.fixture(scope="module", autouse=True)
def do_something():
    pass  # overrides conftest's ServerPreset.load_all(): this test needs no download


@pytest.fixture
def srv():
    s = ServerPreset.tinyllama2()
    s.model_file = GGUF
    s.model_hf_repo = None
    s.model_hf_file = None
    s.jinja = True
    s.chat_template_file = "../../../models/templates/Qwen3.6-35B-A3B.jinja"
    s.n_ctx = 4096
    s.n_slots = 1
    s.n_batch = 512
    if os.environ.get("LLAMA_TEST_SERVER_LOG"):
        s.log_path = os.environ["LLAMA_TEST_SERVER_LOG"]
    s.start()
    yield s
    s.stop()


def body(prefill, stream):
    return {"messages": [{"role": "user", "content": "write it"}, {"role": "assistant", "content": prefill}],
            "continue_final_message": True, "add_generation_prompt": False, "tools": TOOLS, "max_tokens": 1, "temperature": 0.0, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}


def post(srv, data, stream):
    return requests.post(f"http://{srv.server_host}:{srv.server_port}/v1/chat/completions", json=data, stream=stream, timeout=120)


def check_error_obj(e):
    assert e["code"] == 500 and e["type"] == "tool_call_parse_error" and e["retryable"] is True
    assert isinstance(e["message"], str) and e["message"]
    assert isinstance(e["raw_excerpt"], str) and 0 < len(e["raw_excerpt"]) <= 256 and e["raw_excerpt"].startswith("<tool_call>")


@pytest.mark.parametrize("prefill", [UNRECOVERABLE, MALFORMED])
def test_non_streaming_contract(srv, prefill):
    r = post(srv, body(prefill, False), False)
    if prefill == MALFORMED and r.status_code == 200:  # recovered by the split: must be clean
        args = json.loads(r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        assert args == {"content": "hello", "filePath": "/a.txt"}
        return
    assert r.status_code == 500, r.text
    check_error_obj(r.json()["error"])


@pytest.mark.parametrize("prefill", [UNRECOVERABLE, MALFORMED])
def test_streaming_contract(srv, prefill):
    r = post(srv, body(prefill, True), True)
    assert r.status_code == 200
    events = [l[len("data: "):] for l in r.iter_lines(decode_unicode=True) if l and l.startswith("data: ")]
    assert "[DONE]" not in events, events
    errs = [json.loads(e) for e in events if e.startswith("{") and "error" in json.loads(e)]
    assert len(errs) == 1, events
    check_error_obj(errs[0]["error"])
    assert events[-1] == json.dumps(errs[0], separators=(",", ":")) or "error" in json.loads(events[-1]), events
    for e in events[:-1]:
        for ch in json.loads(e).get("choices", []):
            assert not ch.get("delta", {}).get("tool_calls"), e  # no partial tool_calls delta before the error


def test_good_call_never_fires(srv):
    r = post(srv, body(GOOD, False), False)
    assert r.status_code == 200, r.text
    args = json.loads(r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"content": "hello", "filePath": "/a.txt"}
    r = post(srv, body(GOOD, True), True)
    events = [l[len("data: "):] for l in r.iter_lines(decode_unicode=True) if l and l.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert not any("tool_call_parse_error" in e for e in events)
