#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from generate_tiny_fixture import generate
from glm53_artifact import (
    expected_tensor_shapes,
    fixture_contract,
    load_reader,
    real_contract,
    scan,
    validate_identity,
    validate_loader,
)


class TinyFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="glm53-fixtures-")
        cls.paths = generate(Path(cls.temp.name))
        cls.reader_cls = load_reader()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_independent_real_schema_has_all_1412_tensors(self) -> None:
        shapes = expected_tensor_shapes(real_contract())
        self.assertEqual(len(shapes), 1412)
        self.assertEqual(shapes["blk.45.nextn.eh_proj.weight"], [8192, 4096])
        self.assertEqual(shapes["blk.45.indexer_compressor_ape.weight"], [128, 4])

    def test_both_aliases_and_flag_modes_share_complete_tiny_schema(self) -> None:
        expected = expected_tensor_shapes(fixture_contract())
        self.assertEqual(len(self.paths), 4)
        for path in self.paths:
            with self.subTest(path=path.name):
                actual = scan([path], self.reader_cls)
                got = {item["name"]: item["shape"] for item in actual["tensors"]}
                self.assertEqual(got, expected)
                self.assertNotIn("blk.4.nextn.embed_tokens.weight", got)
                self.assertNotIn("blk.4.nextn.shared_head_head.weight", got)

    def test_all_four_variants_have_identical_tensor_payloads(self) -> None:
        def payload_digest(path: Path) -> str:
            reader = self.reader_cls(str(path), "r")
            digest = hashlib.sha256()
            for item in sorted(reader.tensors, key=lambda value: value.name):
                digest.update(item.name.encode())
                digest.update(item.data.tobytes())
            return digest.hexdigest()
        self.assertEqual(len({payload_digest(path) for path in self.paths}), 1)

    def test_explicit_flags_are_true_and_default_variants_omit_them(self) -> None:
        for path in self.paths:
            with self.subTest(path=path.name):
                metadata = scan([path], self.reader_cls)["metadata"]
                prefix = metadata["general.architecture"]
                names = [f"{prefix}.attention.indexer.kpool_select_tail",
                         f"{prefix}.attention.indexer.index_share_mtp"]
                if "explicit" in path.name:
                    self.assertTrue(all(metadata[name] is True for name in names))
                else:
                    self.assertTrue(all(name not in metadata for name in names))


@unittest.skipUnless(os.environ.get("GLM53_MODEL_DIR"), "set GLM53_MODEL_DIR for local-artifact validation")
class OptInLocalArtifactTests(unittest.TestCase):
    def test_real_headers_match_loader_contract_and_optional_identity(self) -> None:
        model_dir = Path(os.environ["GLM53_MODEL_DIR"])
        actual = scan(sorted(model_dir.glob("*.gguf")), load_reader())
        result = validate_loader(actual, real_contract())
        self.assertTrue(result["loader_schema_compatible"], result["errors"])
        if path := os.environ.get("GLM53_IDENTITY_MANIFEST"):
            manifest = json.loads(Path(path).read_text())
            self.assertEqual(validate_identity(actual, manifest), [])


if __name__ == "__main__": unittest.main()
