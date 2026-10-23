"""Model-free tests for moe_trace_convert.py (INF-72 RT-1 / RT-3). Needs numpy; compiles the C++ core test with g++.

Run:  /usr/bin/python3 -m pytest tools/moe-trace/tests -q     (or: python3 -m unittest discover tools/moe-trace/tests)

Covers
  * the cross-language contract: the run that C++ run_writer writes (tests/test-moe-trace-core.cpp) is read back,
    hash-checked and schema-checked here, and (when epyc-inference-research is on disk) by its trace_shape validator
  * the RT-3 stride hazard offline: a trace built from a FLAT read of a strided router tensor is rejected, a
    realistic one is accepted, and a guard that evaluated nothing is not a pass
  * tamper detection: flipped ids byte, truncated rows, int8 manifest, broken token order
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import moe_trace_convert as conv  # noqa: E402

SHAPE_DIR = Path(os.environ.get("EPYC_TRACE_SHAPE_DIR", "/workspace/repos/epyc-inference-research/scripts/kernel_rnd/moe_routing_tap"))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_run(d: Path, layers, samples, rows, ids):
    """Independent Python writer of the on-disk contract (mirrors moe-trace-core.h run_writer)."""
    d.mkdir(parents=True, exist_ok=True)
    rows_b = np.asarray(rows, dtype=conv.ROW_DTYPE).tobytes()
    ids_b = np.asarray(ids, dtype="<i2").tobytes()
    (d / "routing_trace.rows.bin").write_bytes(rows_b)
    (d / "routing_trace.ids.i16").write_bytes(ids_b)
    m = {"schema": conv.SCHEMA_ID, "model": "fixture", "arch": "fixture-moe", "batch_size": 512,
         "layers": [{"layer": i, "n_experts": ne, "top_k": k, "id_dtype": "int16"} for i, (ne, k) in enumerate(layers)],
         "samples": samples, "columns": conv.COLUMNS,
         "ids": {"dtype": "int16", "shape_note": "[rows, n_layers, top_k] row-major; layers with smaller top_k pad with -1", "sha256": _sha(ids_b)},
         "rows_sha256": _sha(rows_b)}
    (d / "routing_trace.json").write_text(json.dumps(m, indent=1))
    return m


def synth_argsort(n_experts, n_tokens, seed):
    """Skewed popularity + temporal persistence: what real routing looks like. -> [n_tokens, n_experts] descending order."""
    rng = np.random.default_rng(seed)
    pop = 1.0 / (1.0 + (np.arange(n_experts) % 29)) + 0.2 * rng.random(n_experts)
    carry = np.zeros(n_experts)
    out = np.empty((n_tokens, n_experts), dtype=np.int32)
    for t in range(n_tokens):
        carry = 0.6 * carry + 0.4 * rng.random(n_experts)
        out[t] = np.argsort(-(pop + carry), kind="stable")
    return out


def one_forward_run(d: Path, n_experts, k, n_tokens, mode):
    argsort = synth_argsort(n_experts, n_tokens, 5)
    if mode == "strided":
        sel = argsort[:, :k]                                   # row by row at i * nb1 -> first k of each row
    else:
        sel = argsort.reshape(-1)[: n_tokens * k].reshape(n_tokens, k)   # FLAT read: whole argsort rows
    ids = sel.astype(np.int16).reshape(n_tokens, 1, k)
    rows = np.array([(0, t, 0, n_tokens, 0, 255) for t in range(n_tokens)],
                    dtype=[("sample_id", "<u4"), ("pos", "<u4"), ("step", "<u4"), ("spec_block", "<u2"), ("phase", "u1"), ("acc", "u1")])
    return write_run(d, [(n_experts, k)], [{"sample_id": 0, "domain": "fixture", "start_row": 0, "n_rows": n_tokens}], rows, ids)


class CppContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="moe_trace_test_"))
        cls.runp = None
        binp = os.environ.get("MOE_TRACE_CORE_TEST_BIN")
        if not binp:
            gxx = shutil.which("g++")
            if gxx:
                binp = str(cls.tmp / "test-moe-trace-core")
                subprocess.run([gxx, "-std=c++17", "-O1", "-o", binp, str(HERE / "test-moe-trace-core.cpp")], check=True)
        if binp:
            cls.runp = cls.tmp / "cpp_run"
            subprocess.run([binp, str(cls.runp)], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def need(self):
        if self.runp is None:
            self.skipTest("no g++ and no MOE_TRACE_CORE_TEST_BIN")
        return self.runp

    def test_cpp_written_run_verifies(self):
        r = self.need()
        # the 17-row fixture is far below 2*n_experts selections, so the guard has nothing to judge: verify must SAY so
        probs = conv.verify(r, guard=True)
        self.assertEqual(len(probs), 1)
        self.assertIn("evaluated NOTHING", probs[0])
        self.assertEqual(conv.verify(r, guard=False), [])

    def test_cpp_manifest_and_rows_decode(self):
        r = self.need()
        m, rows, ids = conv.load_run(r)
        self.assertEqual(rows.size, 17)
        self.assertEqual(ids.shape, (17, 2, 6))
        self.assertEqual([l["n_experts"] for l in m["layers"]], [256, 128])
        self.assertEqual([l["top_k"] for l in m["layers"]], [6, 4])
        self.assertEqual(int(ids[0, 0, 0]), 200)                       # > 127: survives (int8 would wrap)
        self.assertTrue((ids[:, 1, 4:] == -1).all())                  # layer 1 pads slots 4..5
        self.assertEqual(rows["phase"].tolist(), [0] * 5 + [1] * 3 + [0] * 4 + [0] * 3 + [1] * 2)
        self.assertEqual(rows["step"].tolist(), [0] * 5 + [1, 2, 3] + [0] * 4 + [0] * 3 + [1, 2])
        self.assertEqual(rows["spec_block"].tolist(), [5] * 5 + [1] * 3 + [4] * 4 + [3] * 3 + [1] * 2)
        self.assertTrue((rows["acc"] == 255).all())

    def test_cpp_manifest_satisfies_research_trace_shape(self):
        r = self.need()
        if not (SHAPE_DIR / "trace_shape.py").is_file():
            self.skipTest("epyc-inference-research trace_shape.py not on disk")
        sys.path.insert(0, str(SHAPE_DIR))
        try:
            import trace_shape as ts
        finally:
            sys.path.pop(0)
        m, rows, ids = conv.load_run(r)
        self.assertEqual(ts.validate_manifest(m), [])
        py_rows = [[int(x["sample_id"]), int(x["pos"]), "prefill" if x["phase"] == 0 else "decode", int(x["step"]),
                    int(x["spec_block"]), None if x["acc"] == 255 else bool(x["acc"])] for x in rows]
        py_ids = [[[int(e) for e in ids[i, l]] for l in range(ids.shape[1])] for i in range(rows.size)]
        self.assertEqual(ts.validate_rows(m, py_rows, py_ids), [])

    def test_npz_refuses_mixed_top_k_and_converts_uniform(self):
        r = self.need()
        with self.assertRaises(ValueError):
            conv.to_npz(r, str(self.tmp / "x.npz"))
        d = self.tmp / "uniform"
        one_forward_run(d, 256, 8, 512, "strided")
        msg = conv.to_npz(d, str(self.tmp / "u.npz"))
        z = np.load(self.tmp / "u.npz")
        self.assertEqual(z["experts"].shape, (1, 512, 8))
        self.assertEqual(z["experts"].dtype, np.int16)
        self.assertIn("num_experts=256", msg)


class StrideHazardOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="moe_trace_hz_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_strided_read_is_accepted(self):
        for ne, k in ((256, 8), (512, 10)):
            d = self.tmp / f"ok{ne}"
            one_forward_run(d, ne, k, 512, "strided")
            self.assertEqual(conv.verify(d), [], f"{ne} experts")

    def test_flat_read_is_rejected_with_the_signature(self):
        for ne, k in ((256, 8), (512, 10)):
            d = self.tmp / f"bad{ne}"
            one_forward_run(d, ne, k, 512, "flat")
            probs = conv.verify(d)
            self.assertTrue(probs, f"flat read of {ne} experts passed the guard")
            self.assertIn("STRIDE-HAZARD", probs[0])
            self.assertIn("every expert selected the same number of times", probs[0])
            self.assertIn("SUB-CHANCE", probs[0])
            self.assertEqual(conv.main(["verify", str(d)]), 1)

    def test_python_guard_matches_the_cpp_semantics(self):
        argsort = synth_argsort(256, 512, 3)
        good = conv.hazard(argsort[:, :8], 256)
        self.assertTrue(good["evaluated"] and not good["fatal"] and good["ir_mean"] > good["chance"])
        bad = conv.hazard(argsort.reshape(-1)[:512 * 8].reshape(512, 8), 256)
        self.assertTrue(bad["fatal"] and bad["uniform"] and bad["count_min"] == bad["count_max"] == 16)
        self.assertFalse(conv.hazard(argsort[:1, :8], 256)["evaluated"])            # decode step: nothing to judge

    def test_sub_chance_reuse_alone_is_a_result_not_a_hazard(self):
        ne, k, n = 64, 4, 128
        sel = np.zeros((n, k), dtype=np.int32)
        for t in range(n):
            sel[t] = np.arange(k) + (4 if t % 2 else 0)            # skewed to 8 experts, adjacent tokens disjoint
        h = conv.hazard(sel, ne)
        self.assertTrue(h["evaluated"] and h["sub_chance"] and not h["fatal"])


class Tamper(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="moe_trace_tp_"))
        self.d = self.tmp / "run"
        one_forward_run(self.d, 256, 8, 512, "strided")
        self.assertEqual(conv.verify(self.d), [])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_flipped_ids_byte(self):
        p = self.d / "routing_trace.ids.i16"
        b = bytearray(p.read_bytes()); b[100] ^= 1; p.write_bytes(bytes(b))
        self.assertTrue(any("ids.i16 sha256" in x for x in conv.verify(self.d)))

    def test_truncated_rows(self):
        p = self.d / "routing_trace.rows.bin"
        p.write_bytes(p.read_bytes()[:-16])
        self.assertTrue(conv.verify(self.d))

    def test_int8_manifest(self):
        p = self.d / "routing_trace.json"
        m = json.loads(p.read_text()); m["layers"][0]["id_dtype"] = "int8"; p.write_text(json.dumps(m))
        self.assertTrue(any("int16" in x for x in conv.verify(self.d)))

    def test_broken_token_order_even_with_correct_hashes(self):
        m, rows, ids = conv.load_run(self.d)
        rows = rows.copy(); rows["pos"][5], rows["pos"][6] = rows["pos"][6], rows["pos"][5]
        write_run(self.tmp / "run2", [(256, 8)], m["samples"], rows, ids)
        self.assertTrue(any("token order" in x for x in conv.verify(self.tmp / "run2")))

    def test_missing_manifest_means_not_a_trace(self):
        (self.d / "routing_trace.json").unlink()
        self.assertTrue(any("missing routing_trace.json" in x for x in conv.verify(self.d)))


if __name__ == "__main__":
    unittest.main()
