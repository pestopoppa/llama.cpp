"""Model-free regression checks for the prospective G3 coverage gate."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'g3_gate.py'
spec = importlib.util.spec_from_file_location('g3_gate', PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class CoverageGateTest(unittest.TestCase):
    def setUp(self):
        self.reqs, self.pos = {}, {}
        for bucket, length in enumerate((2048, 6000, 12000, 20000)):
            for n in range(50):
                rid = f'{bucket}-{n}'
                for mode in ('head', 'none', 'all-inject'):
                    key = (rid, mode)
                    self.reqs[key] = {'id': rid, 'mode': mode, 'n_tokens': length,
                                      'n_chunks': 2, 'n_rows': 1, 'r': 0.2,
                                      'mean_kl': 0.01,
                                      'bit_identical_to_ref': mode == 'all-inject'}
                    self.pos[key] = [{'row': 0, 'pos': 0, 'kl': 0.1 if mode == 'none' else 0.01,
                                      'nmse': 0.001, 'top1': 1}]

    def test_complete_cohort_and_missing_sanity(self):
        self.assertEqual(gate.evaluate(self.reqs, self.pos, 'head', 'none', 1)['overall'], 'PASS')
        del self.reqs[('0-0', 'all-inject')]
        self.assertTrue(gate.evaluate(self.reqs, self.pos, 'head', 'none', 1)['overall'].startswith('INVALID'))

    def test_missing_naive_position_and_nonfinite_metric_refuse(self):
        self.pos[('0-0', 'none')] = []
        self.assertTrue(gate.evaluate(self.reqs, self.pos, 'head', 'none', 1)['overall'].startswith('INVALID'))
        self.pos[('0-0', 'none')] = [{'row': 0, 'pos': 0, 'kl': float('nan'), 'nmse': 0, 'top1': 1}]
        self.assertTrue(gate.evaluate(self.reqs, self.pos, 'head', 'none', 1)['overall'].startswith('INVALID'))

    def test_kdev_arm_uses_same_gate_and_missing_data_refuses(self):
        for rid, mode in list(self.reqs):
            if mode == 'head':
                self.reqs[(rid, 'kdev')] = dict(self.reqs.pop((rid, mode)), mode='kdev')
                self.pos[(rid, 'kdev')] = self.pos.pop((rid, mode))
        self.assertEqual(gate.evaluate(self.reqs, self.pos, 'kdev', 'none', 1)['overall'], 'PASS')
        self.pos[('0-0', 'kdev')] = []
        self.assertTrue(gate.evaluate(self.reqs, self.pos, 'kdev', 'none', 1)['overall'].startswith('INVALID'))

    def test_missing_retention_is_never_a_pass(self):
        self.assertTrue(gate.evaluate(self.reqs, self.pos, 'head', 'none')['overall'].startswith('INCOMPLETE'))

    def test_duplicate_request_record_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = json.dumps({'id': 'one', 'mode': 'head'})
            (root / 'requests.jsonl').write_text(record + '\n' + record + '\n')
            (root / 'positions.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError, 'duplicate request'):
                gate.load_run(root)


if __name__ == '__main__':
    unittest.main()
