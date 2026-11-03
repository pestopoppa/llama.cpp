import importlib.util
from pathlib import Path
import unittest
import os
import struct
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("reference", Path(__file__).with_name("reference.py"))
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


class ReferenceTest(unittest.TestCase):
    def test_real_maps_and_grouping(self):
        root = Path("/tmp/lsa-release-inspect")
        torch.manual_seed(7)
        q, k = torch.randn(4, 24, 256), torch.randn(4, 4, 256)
        for kind, name in (("pca", "pca-step0"), ("distilled", "d2048")):
            maps = ref.load_maps(root / f"qwen38_27b__long-16k-1M-heavy-key-and-query-{name}.pt", kind)
            for layer in ref.LAYERS:
                wq, wk = maps[f"{layer}.W_Q_idx"], maps[f"{layer}.W_K_idx"]
                iq, ik = ref.project(q, k, wq, wk)
                grouped = torch.stack([sum(q[:, h, 64:].float() for h in range(g * 6, g * 6 + 6)) for g in range(4)], 1).flatten(1)
                expected_q = F.normalize((grouped.bfloat16() @ wq.bfloat16()).float(), dim=-1)
                expected_k = F.normalize((k[:, :, 64:].flatten(1).bfloat16() @ wk.bfloat16()).float(), dim=-1).bfloat16()
                torch.testing.assert_close(iq, expected_q, atol=0, rtol=0)
                torch.testing.assert_close(ik, expected_k, atol=0, rtol=0)

    def test_selection_and_all_keys_oracle(self):
        from lsa.core.reference import build_buckets, read

        torch.manual_seed(9)
        ik = F.normalize(torch.randn(5200, 64), dim=-1).bfloat16()
        positions = torch.tensor([0, 3, 511, 512, 513, 1023, 1024, 4095, 4607, 5199])
        iq = F.normalize(torch.randn(len(positions), 64), dim=-1)
        expected_buckets = build_buckets(ik, 4, 8, 16, 512)
        centroids, members = ref.buckets(ik)
        torch.testing.assert_close(members, expected_buckets.bucket_members, atol=0, rtol=0)
        torch.testing.assert_close(centroids, expected_buckets.bucket_centroids, atol=1e-6, rtol=1e-6)
        expected, _ = read(iq, positions, expected_buckets, 1024, 4096, 4, 512)
        actual = ref.select(iq, ik, positions)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for pos, indices in zip(positions.tolist(), actual):
            held = indices[indices >= 0].tolist()
            self.assertEqual(len(held), len(set(held)))
            self.assertTrue(all(0 <= key <= pos for key in held))
            self.assertTrue(set(range(max(0, pos - 511), pos + 1)).issubset(held))
            self.assertTrue(set(range(min(4, pos + 1))).issubset(held))

    def test_worker_mask_and_rejects_holes(self):
        torch.manual_seed(12)
        q, k = torch.randn(2, 24, 256), torch.randn(520, 4, 256)
        mask = torch.full((2, 520), -torch.inf)
        mask[0, :512] = 0
        mask[1, :] = 0
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "in", Path(tmp) / "out"
            def write():
                source.write_bytes(struct.pack("=II", 2, 520) + q.numpy().tobytes() + k.numpy().tobytes() + mask.numpy().tobytes())
            def run(mode):
                return subprocess.run([sys.executable, str(Path(__file__).with_name("mask_worker.py")), str(source), str(output), "3", mode,
                    "/tmp/lsa-release-inspect/qwen38_27b__long-16k-1M-heavy-key-and-query-pca-step0.pt"], capture_output=True, env={**os.environ, "LSA_THREADS": "1"})
            write()
            self.assertEqual(run("all").returncode, 0)
            actual = torch.frombuffer(bytearray(output.read_bytes()), dtype=torch.float32).reshape(2, 520)
            self.assertTrue(torch.equal(actual, mask))
            self.assertEqual(run("pca").returncode, 0)
            actual = torch.frombuffer(bytearray(output.read_bytes()), dtype=torch.float32).reshape(2, 520)
            self.assertTrue(torch.equal(actual, mask))  # at this length the window plus closed block retain all
            mask[1, 10] = -torch.inf
            write()
            self.assertNotEqual(run("pca").returncode, 0)


if __name__ == "__main__":
    unittest.main()
