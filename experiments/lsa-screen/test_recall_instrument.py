"""Small synthetic checks for the development recall definition and fence."""
import os
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mask_worker import oracle_mass, verify_recall_binding, verify_recall_request
from recall_runner import LAYERS, MAP_SHA, TOKEN_SHA, validate_aggregates


class RecallInstrumentTests(unittest.TestCase):
    def test_aggregate_requires_all_sixteen_bound_layers(self):
        rows = [dict(item='dev-code-00', split='dev', nominal=32768, token_sha256=TOKEN_SHA,
                     query=32767, layer=layer, mode='pca', released_pt_map_sha256=MAP_SHA,
                     budget=4612, top_k=4096, top_hits=2048, top_recall=0.5,
                     selected_mass=0.75, scale=0.0625, softcap=0,
                     method='reconstructed_dense_dequantized_k_double_dot') for layer in sorted(LAYERS)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'aggregates.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(len(validate_aggregates(path)), 16)
            rows[-1]['layer'] = 3
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'sixteen global layers'):
                validate_aggregates(path)

    def test_frozen_dev_binding_uses_decompressed_ids_and_released_map(self):
        frozen = Path('/mnt/raid0/llm/tmp/rope-ctx-assessment/lsa-screen-20261103/root/artifacts/lsa-screen-20261103')
        manifest = frozen / 'manifest.json'
        token = frozen / 'inputs/dev-code-00-32768.json.gz'
        map_path = Path('/tmp/lsa-release-inspect/qwen38_27b__long-16k-1M-heavy-key-and-query-pca-step0.pt')
        if not all(p.is_file() for p in (manifest, token, map_path)):
            self.skipTest('frozen development assets unavailable')
        env = {
            'LLAMA_LSA_RECALL_OUT': '/tmp/unused-recall.jsonl',
            'LLAMA_LSA_RECALL_ITEM': 'dev-code-00',
            'LLAMA_LSA_RECALL_SPLIT': 'dev',
            'LLAMA_LSA_RECALL_NOMINAL': '32768',
            'LLAMA_LSA_RECALL_POSITIONS': '32767',
            'LLAMA_LSA_RECALL_TOKEN_SHA256': '9a6948657870e895f48aabc0f3acc15fdf273ccf66f72345f0a482191dc6a480',
            'LLAMA_LSA_RECALL_MAP_SHA256': 'baaa28cb00b4378b3d33e02dc971e2781482d5c4d45e29e9140143bdf7832e65',
            'LLAMA_LSA_RECALL_MANIFEST': str(manifest),
            'LLAMA_LSA_RECALL_TOKEN_FILE': str(token),
        }
        with patch.dict(os.environ, env, clear=True):
            verify_recall_binding('pca', str(map_path))
            ids = json.loads(gzip.decompress(token.read_bytes()))
            verify_recall_request(ids, token)
            ids[-1] += 1
            with self.assertRaisesRegex(ValueError, 'differs from frozen'):
                verify_recall_request(ids, token)

    def test_attention_mass_and_top_key_are_distinct(self):
        q = torch.zeros((24, 256))
        k = torch.zeros((6, 4, 256))
        q[:, 0] = 1
        k[4, :, 0] = 12
        mass = oracle_mass(q, k, 5, 0.25, 0)
        selected = torch.tensor([0, 1, 2, 3, 5])
        self.assertGreater(float(mass[4]), float(mass[selected].sum()))
        self.assertEqual(int(torch.isin(torch.argsort(mass, descending=True, stable=True)[:1], selected).sum()), 0)

    def test_heldout_and_unpinned_positions_rejected_before_file_read(self):
        base = {
            'LLAMA_LSA_RECALL_OUT': '/tmp/unused-recall.jsonl',
            'LLAMA_LSA_RECALL_ITEM': 'heldout-code-00',
            'LLAMA_LSA_RECALL_SPLIT': 'heldout',
            'LLAMA_LSA_RECALL_NOMINAL': '32768',
            'LLAMA_LSA_RECALL_POSITIONS': '32767',
            'LLAMA_LSA_RECALL_TOKEN_SHA256': '9a6948657870e895f48aabc0f3acc15fdf273ccf66f72345f0a482191dc6a480',
            'LLAMA_LSA_RECALL_MAP_SHA256': 'baaa28cb00b4378b3d33e02dc971e2781482d5c4d45e29e9140143bdf7832e65',
        }
        with patch.dict(os.environ, base, clear=True):
            with self.assertRaisesRegex(ValueError, 'outside frozen development cell'):
                verify_recall_binding('pca', '/does/not/exist')
            os.environ['LLAMA_LSA_RECALL_ITEM'] = 'dev-code-00'
            os.environ['LLAMA_LSA_RECALL_SPLIT'] = 'dev'
            os.environ['LLAMA_LSA_RECALL_POSITIONS'] = '32766'
            with self.assertRaisesRegex(ValueError, 'outside frozen development cell'):
                verify_recall_binding('pca', '/does/not/exist')


if __name__ == '__main__':
    unittest.main()
