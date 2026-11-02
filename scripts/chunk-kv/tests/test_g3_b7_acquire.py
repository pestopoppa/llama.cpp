import importlib.util
from pathlib import Path
import pytest

P = Path(__file__).resolve().parents[1] / 'g3_b7_acquire.py'
s = importlib.util.spec_from_file_location('g3_b7_acquire', P)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


def record():
    prompt = 'prefix AAAA mid BBBB suffix'
    chunks = []
    for txt in ('AAAA', 'BBBB'):
        chunks.append({'start': prompt.index(txt), 'length': len(txt), 'sha256_12': m.sha(txt.encode())[:12]})
    r = {'task_id': 't', 'request_id': 'r', 'layout_match': True,
         'request_prompt_sha256': m.sha(prompt.encode()),
         'blocks': [{'kind': 'doc_chunk', 'chunks': chunks}]}
    r['record_sha256'] = m.record_hash(r)
    return r, {'record_sha256': r['record_sha256'], 'prompt': prompt}, {'task_id': 't', 'target': 'answer'}


def test_accepts_bound_original_and_independent_gold():
    r, c, g = record()
    o = m.acquire([r], [c], [g])
    assert len(o) == 1 and o[0]['id'] == 'r' and len(o[0]['chunks']) == 2
    assert 'prompt' not in o[0] and 'target' not in o[0]


def test_fail_closed_for_missing_full_prompt_identity_and_mismatches():
    r, c, g = record()
    r.pop('request_prompt_sha256'); r['record_sha256'] = m.record_hash(r); c['record_sha256'] = r['record_sha256']
    with pytest.raises(ValueError, match='full-prompt'): m.acquire([r], [c], [g])
    r, c, g = record(); c['prompt'] += '!'
    with pytest.raises(ValueError, match='full-prompt'): m.acquire([r], [c], [g])
    r, c, g = record(); r['blocks'][0]['chunks'][0]['sha256_12'] = '0' * 12; r['record_sha256'] = m.record_hash(r); c['record_sha256'] = r['record_sha256']
    with pytest.raises(ValueError, match='chunk bytes'): m.acquire([r], [c], [g])
    r, c, g = record(); g['target'] = ''
    with pytest.raises(ValueError, match='target missing'): m.acquire([r], [c], [g])
