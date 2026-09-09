#!/usr/bin/env python3
"""Pure tests for the portable GLM-5.3 workload clients."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response(tokens, prompt_n=3, rate=10.0):
    return {
        "choices": [{"finish_reason": "length", "message": {"content": "ok"}}],
        "__verbose": {"tokens": tokens, "prompt": "prompt"},
        "tokens": tokens,
        "content": "ok",
        "stop_type": "limit",
        "stopped_limit": True,
        "timings": {"predicted_n": len(tokens), "predicted_per_second": rate,
                    "prompt_n": prompt_n, "prompt_per_second": 20.0,
                    "cache_n": 0, "draft_n": 0, "draft_n_accepted": 0},
    }


class FinalWorkloadClientsTest(unittest.TestCase):
    def invoke(self, module, *args):
        with mock.patch.object(sys, "argv", [module.__file__, *map(str, args)]):
            return module.main()

    def test_canonical_captures_exact_tokens(self):
        module = load("canonical_workload_client")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); prompts = td / "prompts.json"
            prompts.write_text(json.dumps([{"id": "one", "class": "test", "prompt": "hello"}]))
            with mock.patch.object(module, "post", return_value=response(list(range(16)))):
                self.invoke(module, "--port", 1, "--out", td / "out", "--prompts", prompts)
            row = json.loads((td / "out/requests.json").read_text())[0]
            self.assertEqual(row["tokens"], list(range(16)))

    def test_canonical_rejects_boolean_token(self):
        module = load("canonical_workload_client")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); prompts = td / "prompts.json"
            prompts.write_text(json.dumps([{"id": "one", "class": "test", "prompt": "hello"}]))
            with mock.patch.object(module, "post", return_value=response([True])):
                with self.assertRaisesRegex(RuntimeError, "exact token IDs"):
                    self.invoke(module, "--port", 1, "--out", td / "out", "--prompts", prompts)

    def test_canonical_rejects_nonfinite_rate(self):
        module = load("canonical_workload_client")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); prompts = td / "prompts.json"
            prompts.write_text(json.dumps([{"id": "one", "class": "test", "prompt": "hello"}]))
            bad=response([1]); bad["timings"]["predicted_per_second"]=float("nan")
            with mock.patch.object(module, "post", return_value=bad):
                with self.assertRaisesRegex(RuntimeError, "valid timings"):
                    self.invoke(module, "--port", 1, "--out", td / "out", "--prompts", prompts)

    def test_prefill_fixture_and_schema(self):
        module = load("prefill2029_client")
        self.assertEqual(len(json.loads(module.TOKENS.read_text())), 2029)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with mock.patch.object(module, "post", return_value=response([7], prompt_n=2029, rate=1.0)):
                self.invoke(module, "--port", 1, "--out", td / "out")
            rows = json.loads((td / "out/requests.json").read_text())
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row["cache_n"] == 0 for row in rows))

    def test_multiprompt_captures_two_exact_outputs(self):
        module = load("multiprompt512_client")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); prompts = td / "prompts.json"
            prompts.write_text(json.dumps([
                {"id": "gsm8k_00739", "prompt": "a"},
                {"id": "simpleqa_general_01814", "prompt": "b"},
            ]))
            def post(_port, route, _body, _timeout):
                return {"prompt": "rendered"} if route == "/apply-template" else response([4] * 512)
            with mock.patch.object(module, "post", side_effect=post):
                self.invoke(module, "--port", 1, "--out", td / "out", "--prompts", prompts)
            rows = json.loads((td / "out/requests.json").read_text())
            self.assertEqual([len(row["tokens"]) for row in rows], [512, 512])

    def test_quality_uses_explicit_scorer(self):
        module = load("quality_workload_client")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); scorer = td / "scorer"; scorer.mkdir()
            (scorer / "debug_scorer.py").write_text("def score_answer(*args): return True\n")
            manifest = td / "manifest.json"
            manifest.write_text(json.dumps({"rows": [{"id": "q", "prompt": "p", "expected": "ok",
                "scoring_method": "exact", "scoring_config": {}}]}))
            with mock.patch.object(module, "post", return_value=response([1, 2])):
                self.invoke(module, "--port", 1, "--out", td / "out", "--manifest", manifest,
                            "--scorer-root", scorer)
            self.assertTrue(json.loads((td / "out/results.json").read_text())[0]["passed_content"])


if __name__ == "__main__":
    unittest.main()
