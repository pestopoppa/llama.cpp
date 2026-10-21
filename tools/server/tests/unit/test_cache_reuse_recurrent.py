import os
import pytest
from utils import *

# Guard: --cache-reuse (KV-shift chunk reuse) must be disabled for models carrying recurrent state
# (hybrid attention+recurrent, pure recurrent): llama_memory_seq_add() only moves cell.pos for the
# recurrent part and leaves its state stale -> silently wrong output.
# Set LLAMA_TEST_HYBRID_GGUF to a small hybrid/recurrent GGUF (e.g. Qwen3.5-0.8B, LFM2) to run.

HYBRID = os.environ.get("LLAMA_TEST_HYBRID_GGUF")

pytestmark = pytest.mark.skipif(not HYBRID, reason="LLAMA_TEST_HYBRID_GGUF not set")


def _hybrid_server(tmp_path):
    srv = ServerPreset.tinyllama2()
    srv.model_file = HYBRID
    srv.model_hf_repo = None
    srv.model_hf_file = None
    srv.n_ctx = 512
    srv.n_slots = 1
    srv.n_predict = 8
    srv.log_path = str(tmp_path / "server.log")
    return srv


def test_hybrid_disables_cache_reuse(tmp_path):
    srv = _hybrid_server(tmp_path)
    srv.n_cache_reuse = 16
    srv.start()
    try:
        res = srv.make_request("POST", "/completion", data={"prompt": "Hello world, this is a test.", "n_predict": 4})
        assert res.status_code == 200
        assert res.body["generation_settings"]["n_cache_reuse"] == 0
    finally:
        srv.stop()
    log = open(srv.log_path).read()
    assert "cache_reuse is not supported by this context, it will be disabled" in log
