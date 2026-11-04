#!/usr/bin/env python3
"""Small model-free wire and ordering checks for the diagnostic reader."""

import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

MODULE = Path(__file__).with_name("replay_index_witness.py")
spec = importlib.util.spec_from_file_location("replay_index_witness", MODULE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def payload(indices=(4, 2), length=None, budgets=(4096, 1024, 512, 4)):
    if length is None:
        length = len(indices)
    model, map_hash, map_source = (c * 64 for c in (b"a", b"b", b"c"))
    head = replay.HEADER.pack(b"LSAWIT1\0", 5132, 4612, length, 3, 5131,
                              *budgets, 0, 7, model, map_hash, map_source)
    return head + struct.pack("<64f", *([0.0] * 64)) + struct.pack("<328448f", *([0.0] * 328448)) + struct.pack(f"<{len(indices)}i", *indices)


class WitnessTests(unittest.TestCase):
    def test_order_distinct_from_membership(self):
        exact = replay.compare_rows([4, 2], [4, 2])
        self.assertEqual(exact["disposition"], "ordered_exact")
        self.assertEqual(replay.disposition_exit_code(exact["disposition"]), 0)
        order_only = replay.compare_rows([4, 2], [2, 4])
        self.assertEqual(order_only["disposition"], "inconclusive_order_only")
        self.assertTrue(order_only["set_equal"])
        self.assertEqual(order_only["first_order_mismatch"], 0)
        self.assertEqual(order_only["native_indices"], [4, 2])
        self.assertEqual(order_only["author_indices"], [2, 4])
        self.assertEqual(replay.disposition_exit_code(order_only["disposition"]), 2)
        membership = replay.compare_rows([4, 2], [4, 3])
        self.assertEqual(membership["disposition"], "membership_mismatch")
        self.assertFalse(membership["set_equal"])
        self.assertEqual(replay.disposition_exit_code(membership["disposition"]), 1)

    def test_wire_and_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "witness"
            path.write_bytes(payload())
            witness = replay.read_witness(path)
            self.assertEqual(witness["indices"], [4, 2])
            self.assertEqual(witness["epoch"], 7)
            self.assertEqual(witness["map_source_sha256"], "c" * 64)
            path.write_bytes(payload()[:-1])
            with self.assertRaisesRegex(ValueError, "payload length"):
                replay.read_witness(path)
            path.write_bytes(payload((5132,)))
            with self.assertRaisesRegex(ValueError, "invalid index"):
                replay.read_witness(path)
            path.write_bytes(payload(budgets=(4095, 1024, 513, 4)))
            with self.assertRaisesRegex(ValueError, "budgets differ"):
                replay.read_witness(path)


if __name__ == "__main__":
    unittest.main()
