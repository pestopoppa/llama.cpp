"""No model: physical Q/K oracle budget, causal mask and deterministic ties."""
import math
import os
import struct
import unittest

import torch

from mask_worker import ORACLE_TOP_K, ORACLE_WINDOW, ORACLE_SINK, compute, oracle_config, oracle_indices, oracle_mass


def tensors(n):
    q = torch.zeros((1, 24, 256), dtype=torch.float32)
    k = torch.zeros((n, 4, 256), dtype=torch.float32)
    return q, k


def payload(q, k, position):
    mask = torch.full((len(q), len(k)), -float('inf'))
    mask[:, :position + 1] = 0
    return bytearray(struct.pack('=II', len(q), len(k)) + q.numpy().tobytes() + k.numpy().tobytes() + mask.numpy().tobytes())


class OracleTopKTests(unittest.TestCase):
    def test_boundaries_and_stable_ties(self):
        q, k = tensors(4613)
        for position in (0, 3, 511, 512, 4611):
            got = oracle_indices(q[0], k, position, 0.0625, 0).tolist()
            self.assertEqual(set(got), set(range(position + 1)))
            self.assertEqual(len(got), position + 1)
        got = oracle_indices(q[0], k, 4612, 0.0625, 0).tolist()
        expected = list(range(4, 4100)) + list(range(4101, 4613)) + list(range(4))
        self.assertEqual(got, expected)  # stable tie picks lower distant index
        self.assertEqual(len(got), ORACLE_TOP_K + ORACLE_WINDOW + ORACLE_SINK)
        self.assertEqual(len(got), len(set(got)))

    def test_physical_attention_mass_changes_selected_key(self):
        q, k = tensors(4613)
        q[0, :, 0] = 1
        k[4100, :, 0] = 10
        got = oracle_indices(q[0], k, 4612, 0.0625, 2).tolist()
        self.assertIn(4100, got)
        self.assertNotIn(4099, got)
        self.assertEqual(len(got), 4612)
        self.assertTrue(all(0 <= x <= 4612 for x in got))

    def test_teacher_mass_matches_independent_per_head_softmax(self):
        torch.manual_seed(73)
        q = torch.randn((24, 256))
        k = torch.randn((7, 4, 256))
        scale, cap = 0.0625, 1.5
        expected = torch.zeros(7)
        for head in range(24):
            logits = torch.tensor([float(torch.dot(q[head], k[key, head // 6])) * scale for key in range(7)])
            logits = cap * torch.tanh(logits / cap)
            expected += torch.softmax(logits, dim=0) / 24
        actual = oracle_mass(q, k, 6, scale, cap)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=6)

    def test_nonuniform_gqa_heads_change_distant_ranking(self):
        torch.manual_seed(91)
        q, k = tensors(4613)
        q[0, :, :4] = torch.randn(24, 4)
        k[:, :, :4] = torch.randn(4613, 4, 4)
        scale, cap = 0.25, 1.2
        mean = torch.zeros(4613)
        for head in range(24):
            logits = (k[:, head // 6, :4] @ q[0, head, :4]) * scale
            logits = cap * torch.tanh(logits / cap)
            mean += torch.softmax(logits, dim=0) / 24
        expected_rank = torch.argsort(mean[4:4101], descending=True, stable=True)[:4096] + 4
        expected = torch.cat((expected_rank, torch.arange(4101, 4613), torch.arange(4))).tolist()
        actual = oracle_indices(q[0], k, 4612, scale, cap).tolist()
        torch.testing.assert_close(oracle_mass(q[0], k, 4612, scale, cap), mean, rtol=1e-5, atol=1e-8)
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(len(actual), 4612)

    def test_worker_returns_full_causal_mask(self):
        q, k = tensors(4613)
        out = torch.frombuffer(bytearray(compute(payload(q, k, 4612), '3', 'oracle_topk', None, (0.0625, 0))),
                               dtype=torch.float32).reshape(1, 4613)[0]
        selected = torch.isfinite(out).nonzero().flatten().tolist()
        self.assertEqual(selected, list(range(4100)) + list(range(4101, 4613)))
        self.assertTrue(torch.all(out[selected] == 0))
        self.assertTrue(torch.isneginf(out[4100]))
        damaged = payload(q, k, 4612)
        mask_start = 8 + q.numel() * 4 + k.numel() * 4
        struct.pack_into('=f', damaged, mask_start + 10 * 4, -float('inf'))
        with self.assertRaisesRegex(ValueError, 'contiguous causal sequence'):
            compute(damaged, '3', 'oracle_topk', None, (0.0625, 0))

    def test_bad_configuration_and_nonfinite_qk_refused(self):
        q, k = tensors(4)
        for scale, cap in ((0, 0), (math.nan, 0), (1, -1), (1, math.inf)):
            with self.assertRaises(ValueError):
                oracle_indices(q[0], k, 3, scale, cap)
        q[0, 0, 0] = math.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite Q/K'):
            oracle_indices(q[0], k, 3, 0.0625, 0)
        old_scale = os.environ.pop('LLAMA_LSA_ORACLE_SCALE', None)
        old_cap = os.environ.pop('LLAMA_LSA_ORACLE_SOFTCAP', None)
        try:
            with self.assertRaisesRegex(ValueError, 'missing'):
                oracle_config('oracle_topk')
        finally:
            if old_scale is not None: os.environ['LLAMA_LSA_ORACLE_SCALE'] = old_scale
            if old_cap is not None: os.environ['LLAMA_LSA_ORACLE_SOFTCAP'] = old_cap


if __name__ == '__main__':
    unittest.main()
