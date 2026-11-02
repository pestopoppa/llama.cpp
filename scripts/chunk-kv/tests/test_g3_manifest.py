"""Adversarial, model-free checks for frozen cohort/task binding."""
import importlib.util
import json
import sys
import types
from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import g3_manifest as gm  # noqa: E402
spec = importlib.util.spec_from_file_location('g3_gate', SCRIPTS / 'g3_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class FrozenGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cp, self.tp, self.mp, self.op = (self.root / x for x in ('corpus.jsonl', 'tasks.jsonl', 'manifest.json', 'outputs.jsonl'))
        self.patchers = [patch.object(gm, 'BUCKETS', ((12, 16), (16, 20), (20, 24), (24, 28))),
                         patch.object(gm, 'GATE_MIN_REQUESTS', 4), patch.object(gm, 'GATE_MIN_PER_BUCKET', 1)]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)
        ids = []
        n = 0
        while len(ids) < 4:
            rid = f'case-{n}'
            if gm.split(rid) == 'gate':
                ids.append(rid)
            n += 1
        self.ids = ids
        corpus, tasks = [], []
        for i, rid in enumerate(ids):
            # Every chunk repeats, with a different predecessor per request.
            tokens = [100 + i, 11, 12, 13, 22, 23, 24] + [90] * (5 + i * 4)
            corpus.append({'id': rid, 'tokens': tokens,
                           'chunks': [{'offset': 1, 'n_tokens': 3}, {'offset': 4, 'n_tokens': 3}]})
            tasks.append({'id': rid, 'metric': 'exact_token_match', 'target_tokens': [7, i]})
        self.cp.write_text(''.join(json.dumps(x) + '\n' for x in corpus))
        self.tp.write_text(''.join(json.dumps(x) + '\n' for x in tasks))
        self.manifest = gm.make_manifest(self.cp, self.tp, 'test-role', 'fixture-only')
        self.mp.write_text(json.dumps(self.manifest))
        self.outcomes = [{'id': rid, 'arm': arm, 'output_tokens': [7, i], 'complete': True}
                         for i, rid in enumerate(ids) for arm in ('full', 'reuse')]
        self.op.write_text(''.join(json.dumps(x) + '\n' for x in self.outcomes))
        self.reqs = {(rid, mode): {'n_tokens': 12 + i * 4, 'n_chunks': 2, 'n_cont': 64, 'n_rows': 69 + i * 4}
                     for i, rid in enumerate(ids) for mode in ('head', 'none', 'all-inject')}
        self.pos = {(rid, mode): [{'row': j, 'pos': 7 + j} for j in range(self.reqs[(rid, mode)]['n_rows'])]
                    for rid, mode in self.reqs}

    def test_digest_shape_and_result_identity(self):
        manifest = gm.verify(self.mp, self.cp, self.tp)
        expected = gate.bound_cohort(manifest, self.reqs, self.pos, 'head', 'none')
        self.assertEqual(set(expected), set(self.ids))
        self.pos[(self.ids[0], 'head')].pop()
        with self.assertRaisesRegex(ValueError, 'position coverage'):
            gate.bound_cohort(manifest, self.reqs, self.pos, 'head', 'none')
        self.pos[(self.ids[0], 'head')].append({'row': 68, 'pos': 75})
        del self.reqs[(self.ids[0], 'head')]
        with self.assertRaisesRegex(ValueError, 'result ids'):
            gate.bound_cohort(manifest, self.reqs, self.pos, 'head', 'none')
        self.cp.write_text(self.cp.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            gm.verify(self.mp, self.cp, self.tp)

    def test_paired_outcomes_rederived_and_missing_refuses(self):
        expected = {r['id']: r for r in self.manifest['requests']}
        result = gate.paired_retention(self.op, self.tp, expected)
        self.assertEqual(result['retention'], 1)
        self.assertEqual(result['bootstrap_95'], [1, 1])
        self.op.write_text(''.join(json.dumps(x) + '\n' for x in self.outcomes[:-1]))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            gate.paired_retention(self.op, self.tp, expected)
        self.op.write_text(''.join(json.dumps(x) + '\n' for x in self.outcomes + self.outcomes[:1]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            gate.paired_retention(self.op, self.tp, expected)

    def test_gate_requires_frozen_manifest_and_per_item_scores(self):
        # Exercise the real gate entry with a tiny shape override, never a real verdict.
        run = self.root / 'run'
        run.mkdir()
        (run / 'meta.json').write_text(json.dumps({'n_cont': 64}))
        with (run / 'requests.jsonl').open('w') as f, (run / 'positions.jsonl').open('w') as pos:
            for (rid, mode), row in self.reqs.items():
                f.write(json.dumps(dict(row, id=rid, mode=mode, r=0.2, mean_kl=0.01,
                                        bit_identical_to_ref=mode == 'all-inject')) + '\n')
                for point in self.pos[(rid, mode)]:
                    pos.write(json.dumps(dict(point, id=rid, mode=mode, kl=0.1 if mode == 'none' else 0.01,
                                              nmse=0.001, top1=1)) + '\n')
        threshold = self.root / 'P0.md'
        threshold.write_text(f'## 8. Corpus manifest\n- manifest path / sha256: x / {gm.sha(self.mp)}\n## 9. Amendments\n')
        args = types.SimpleNamespace(dev=False, thresholds=str(threshold), manifest=str(self.mp),
                                     corpus=str(self.cp), tasks=str(self.tp), retention=str(self.op),
                                     run=str(run), selector='head', naive='none', out=None)
        with patch.object(gate, 'BUCKETS', [('a', 12, 16), ('b', 16, 20), ('c', 20, 24), ('d', 24, 28)]), \
             patch.object(gate, 'CORPUS_MIN_REQUESTS', 4), patch.object(gate, 'CORPUS_MIN_PER_BUCKET', 1), \
             redirect_stdout(StringIO()):
            self.assertEqual(gate.cmd_gate(args), 0)
            self.op.write_text(json.dumps({'score_full': 1, 'score_reuse': 1}))
            self.assertEqual(gate.cmd_gate(args), 3)

    def test_pinned_canonical_text_scorer_is_bound_per_item(self):
        import hashlib
        scorer = self.root / 'scorer.py'
        scorer.write_text('def score_answer(answer, expected, scoring_method, scoring_config):\n'
                          '    return expected.lower() in answer.lower()\n')
        digest = hashlib.sha256(scorer.read_bytes()).hexdigest()
        tasks = gm.records(self.tp)
        for rid in self.ids:
            tasks[rid] = {'id': rid, 'metric': 'orchestrator_text_v1', 'expected': 'Oxford',
                          'scoring_method': 'substring', 'scoring_config': {}, 'scorer_sha256': digest}
        self.tp.write_text(''.join(json.dumps(x) + '\n' for x in tasks.values()))
        manifest = gm.make_manifest(self.cp, self.tp, 'role', 'fixture')
        self.mp.write_text(json.dumps(manifest))
        self.assertEqual(gm.verify(self.mp, self.cp, self.tp), manifest)
        outcomes = [{'id': rid, 'arm': arm, 'output_text': 'Oxford University', 'complete': True}
                    for rid in self.ids for arm in ('full', 'reuse')]
        self.op.write_text(''.join(json.dumps(x) + '\n' for x in outcomes))
        expected = {x['id']: x for x in manifest['requests']}
        result = gate.paired_retention(self.op, self.tp, expected, str(scorer))
        self.assertEqual(result['retention'], 1)
        self.assertEqual(len(result['item_scores']), len(self.ids))
        self.assertEqual(result['scorer_sha256'], digest)
        scorer.write_text('def score_answer(*args): return True\n')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            gate.paired_retention(self.op, self.tp, expected, str(scorer))
        tasks[self.ids[0]]['scoring_method'] = 'llm_judge'
        self.tp.write_text(''.join(json.dumps(x) + '\n' for x in tasks.values()))
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            gm.make_manifest(self.cp, self.tp, 'role', 'fixture')

    def test_missing_task_wrong_metrics_or_changed_layout_refuse(self):
        rows = gm.records(self.tp)
        rows[self.ids[0]]['metric'] = 'unfixed_metric'
        self.tp.write_text(''.join(json.dumps(x) + '\n' for x in rows.values()))
        with self.assertRaisesRegex(ValueError, 'metric'):
            gm.make_manifest(self.cp, self.tp, 'test-role', 'fixture-only')
        rows[self.ids[0]]['metric'] = 'exact_token_match'
        self.tp.write_text(''.join(json.dumps(x) + '\n' for x in rows.values()))
        corpus = gm.records(self.cp)
        corpus[self.ids[0]]['chunks'][0]['offset'] = 3
        self.cp.write_text(''.join(json.dumps(x) + '\n' for x in corpus.values()))
        with self.assertRaisesRegex(ValueError, 'chunk'):
            gm.make_manifest(self.cp, self.tp, 'test-role', 'fixture-only')


if __name__ == '__main__':
    unittest.main()
