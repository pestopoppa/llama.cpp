import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('g3_musique_acquire', ROOT / 'g3_musique_acquire.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_preserves_exact_source_and_real_moved_chunks(tmp_path):
    rows = [
        {'id': 'a', 'input': 'Qaaaabbbb?', 'output': 'yes', 'document_start_index': 1, 'document_end_index': 9, 'inner_docs_start_indices': [1, 5]},
        {'id': 'b', 'input': 'QQaaaabbbb!', 'output': 'no', 'document_start_index': 2, 'document_end_index': 10, 'inner_docs_start_indices': [2, 6]},
        {'id': 'c', 'input': 'xaaaazzzz?', 'output': 'no', 'document_start_index': 1, 'document_end_index': 9, 'inner_docs_start_indices': [1, 5]},
    ]
    src = tmp_path / 'source.jsonl'
    src.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    result = mod.collect(src)[0]
    assert [x['id'] for x in result['requests']] == ['a', 'b']
    assert result['requests'][0]['prompt_sha256'] == mod.digest(rows[0]['input'].encode())
    assert result['requests'][0]['target_sha256'] == mod.digest(b'yes')
    assert len(result['requests'][0]['chunks']) == 2
    assert 'not B7' in result['scope']


def test_reject_invalid_spans_and_duplicate_ids(tmp_path):
    import pytest
    src = tmp_path / 'source.jsonl'
    item = {'id': 'a', 'input': 'Qabcd?', 'output': 'yes', 'document_start_index': 1, 'document_end_index': 5, 'inner_docs_start_indices': [1, 1]}
    src.write_text(json.dumps(item) + '\n')
    with pytest.raises(ValueError): mod.collect(src)
    item['inner_docs_start_indices'] = [1, 3]
    src.write_text(json.dumps(item) + '\n' + json.dumps(item) + '\n')
    with pytest.raises(ValueError): mod.collect(src)
