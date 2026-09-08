#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from generate_serve_plan import EXACT_ENV, PREFIX, build_plan


BINARY = "/candidate/build/bin/llama-server"
MODEL = "/models/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf"


class ServePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="glm53-recipe-")
        cls.recipe = Path(cls.temp.name) / "recipe.py"
        cls.recipe.write_text('''
THREADS = 48
CONTEXT = 8192
PARALLEL_SLOTS = 1
SERVE_PREFIX = ["taskset", "-c", "0-95", "numactl", "--interleave=all"]
CANONICAL_OMP_ENV = {"OMP_PROC_BIND":"spread","OMP_PLACES":"cores","OMP_WAIT_POLICY":"active","OMP_DYNAMIC":"false"}
CHAMPION_GGML_ENV = {"GGML_IQK":"1","GGML_FUSED_DECODE_OFF":"1","GGML_FA_SPLIT_KV":"0","GGML_NOHUGEPAGE_PROCESS":"1"}
BASE_SERVER_FLAGS = ["--no-webui", "-np", str(PARALLEL_SLOTS), "-c", str(CONTEXT), "-t", str(THREADS), "--no-mmap", "-lv", "4"]
FLASH_ATTN_FLAGS = ["-fa", "on"]
KV_CACHE_FLAGS = ["-ctk", "f16", "-ctv", "f16"]
''')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def plan(self, **kwargs):
        return build_plan(BINARY, MODEL, recipe_source=self.recipe, **kwargs)

    def test_exact_cpu_native_mtp_argv(self) -> None:
        plan = self.plan()
        self.assertEqual(plan["argv"][:len(PREFIX)], PREFIX)
        self.assertEqual(plan["argv"][len(PREFIX)], BINARY)
        self.assertEqual(plan["argv"][plan["argv"].index("-m") + 1], MODEL)
        self.assertEqual(plan["argv"][plan["argv"].index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(plan["argv"][plan["argv"].index("--spec-draft-n-max") + 1], "3")
        self.assertEqual(plan["argv"][plan["argv"].index("--spec-draft-p-min") + 1], "0")
        self.assertNotIn("-md", plan["argv"])
        self.assertNotIn("--model-draft", plan["argv"])
        self.assertEqual(plan["argv"][plan["argv"].index("--device") + 1], "none")
        self.assertEqual(plan["argv"][plan["argv"].index("-ngl") + 1], "0")

    def test_environment_is_exact_and_candidate_local(self) -> None:
        plan = self.plan()
        self.assertEqual(plan["environment"], {**EXACT_ENV, "LD_LIBRARY_PATH": "/candidate/build/bin"})
        self.assertTrue(plan["shell_preview"].startswith("env -i "))
        self.assertEqual(len(plan["plan_sha256"]), 64)
        self.assertEqual(len(plan["derivation"]["recipe_source"]["sha256"]), 64)
        self.assertIn("required before launch", plan["execution_preconditions"]["physical_region_lock"])

    def test_inputs_and_tuning_are_rendered_without_execution(self) -> None:
        plan = self.plan(port=19001, context=16384, threads=64, draft_max=5, draft_p_min=0.25)
        for flag, value in (("--port", "19001"), ("-c", "16384"), ("-t", "64"),
                            ("--spec-draft-n-max", "5"), ("--spec-draft-p-min", "0.25")):
            index = max(i for i, token in enumerate(plan["argv"]) if token == flag)
            self.assertEqual(plan["argv"][index + 1], value)
        self.assertEqual(plan["execution"], "not_executed")

    def test_plain_diagnostic_is_matched_and_has_no_spec_flags(self) -> None:
        plan = self.plan(mtp=False)
        self.assertNotIn("--spec-type", plan["argv"])
        self.assertNotIn("--spec-draft-n-max", plan["argv"])
        self.assertIn("plain diagnostic", plan["contract"]["spec_type"])

    def test_rejects_invalid_paths_and_values(self) -> None:
        with self.assertRaises(ValueError): build_plan("/candidate/not-server", MODEL, recipe_source=self.recipe)
        with self.assertRaises(ValueError): build_plan(BINARY, "/models/not-gguf.bin", recipe_source=self.recipe)
        with self.assertRaises(ValueError): self.plan(port=0)
        with self.assertRaises(ValueError): self.plan(draft_p_min=1.1)

    def test_cited_recipe_drift_fails_closed(self) -> None:
        bad = Path(self.temp.name) / "bad.py"
        bad.write_text(self.recipe.read_text().replace('THREADS = 48', 'THREADS = 96'))
        with self.assertRaisesRegex(ValueError, "drift"):
            build_plan(BINARY, MODEL, recipe_source=bad)


if __name__ == "__main__": unittest.main()
