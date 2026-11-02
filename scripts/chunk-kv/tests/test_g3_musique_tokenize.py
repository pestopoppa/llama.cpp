import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('g3_musique_tokenize', ROOT / 'g3_musique_tokenize.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import g3_musique_acquire as acquire


def fixture(tmp_path):
    source = tmp_path / 'source.jsonl'
    rows = []
    for name, prefix in [('a', 'X'), ('b', 'XX')]:
        prompt = prefix + 'a' * 1100 + 'b' * 200 + '?'
        rows.append({'id': name, 'input': prompt, 'output': 'yes',
                     'document_start_index': len(prefix), 'document_end_index': len(prefix) + 1300,
                     'inner_docs_start_indices': [len(prefix), len(prefix) + 1100]})
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    selection = tmp_path / 'selection.json'; selection.write_text(json.dumps(acquire.collect(source)[0]))
    model = tmp_path / 'model'; model.write_bytes(b'header')
    libs = tmp_path / 'lib'; libs.mkdir(); (libs / 'libllama.so.0').write_bytes(b'llama'); (libs / 'libggml.so.0').write_bytes(b'ggml')
    exe = tmp_path / 'tokenizer'
    exe.write_text('''#!/usr/bin/env python3
import json,sys
for line in sys.stdin:
 r=json.loads(line); text=r['prompt']; spans=r['spans']
 print(json.dumps({'id':r['id'], 'tokens':list(text.encode()), 'target_tokens':list(r['target'].encode()), 'chunks':[{'offset':s['start_byte'],'n_tokens':s['end_byte']-s['start_byte']} for s in spans], 'rejected_chunks':0}))
'''); exe.chmod(0o755)
    return source, selection, model, exe, libs


def test_retains_observed_requests_but_refuses_to_call_small_set_gate_ready(tmp_path):
    src, sel, model, exe, libs = fixture(tmp_path)
    out = tmp_path / 'out'
    report = m.materialize(src, sel, model, exe, libs, 6, out)
    assert report['retained_requests'] == 2 and report['status'] == 'NOT_G3_ELIGIBLE'
    assert 'B7 traffic' in report['limitations'][0]
    assert len((out / 'corpus.jsonl').read_text().splitlines()) == 2
    assert len((out / 'tasks.jsonl').read_text().splitlines()) == 2
    assert len(report['library_sha256']) == 2


def test_fails_closed_on_source_drift(tmp_path):
    src, sel, model, exe, libs = fixture(tmp_path)
    src.write_text(src.read_text().replace('yes', 'no'))
    with pytest.raises(ValueError, match='source missing or changed'):
        m.materialize(src, sel, model, exe, libs, 6, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
