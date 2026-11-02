"""Prospective long-QA selection preserves exact original data and provenance."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g3_longqa_acquire import make_records, prepare  # noqa: E402


def example(rid, question, target='answer'):
    story = 'alpha paragraph\n\nbeta paragraph\n\ngamma paragraph\n'
    a = len('Story:\n')
    b = a + len(story)
    prefix = 'Story:\n' + story + '\n'
    query = 'Question: ' + question
    return {'id': rid, 'pid': rid + '_0', 'input': prefix + query + '\nAnswer:',
            'output': target, 'document_start_index': a, 'document_end_index': b,
            'query_start_index': len(prefix), 'query_end_index': len(prefix) + len(query)}


def write(path, rows):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))


def test_original_and_gold_are_preserved_and_prefix_is_a_real_question(tmp_path):
    path = tmp_path / 'source.jsonl'
    a, b = example('a', 'what?'), example('b', 'who?')
    write(path, [a, b, {**a, 'pid':'a_1', 'output':'different gold'}])
    records = make_records(path, 'squality')
    assert len(records) == 2
    for record, original in zip(records, [a, b], strict=True):
        question = original['input'][original['query_start_index']:original['query_end_index']]
        assert record['prompt'] == question + '\n\n' + original['input']
        assert record['target'] == original['output']
        encoded = record['prompt'].encode()
        chunks = [encoded[s['start_byte']:s['end_byte']] for s in record['spans']]
        assert b''.join(chunks) == original['input'][original['document_start_index']:original['document_end_index']].encode()
    assert records[0]['question_prefix_sha256'] != records[1]['question_prefix_sha256']


def test_unchanged_predecessors_and_invalid_bounds_refuse(tmp_path):
    path = tmp_path / 'source.jsonl'
    write(path, [example('a', 'same'), example('b', 'same')])
    assert make_records(path, 'source') == []
    bad = example('c', 'bad')
    bad['document_end_index'] = 999999
    write(path, [bad])
    with pytest.raises(ValueError, match='invalid source spans'):
        make_records(path, 'source')


def test_exclusive_output_and_transformation_recorded(tmp_path):
    path = tmp_path / 'source.jsonl'
    write(path, [example('a', 'what?'), example('b', 'who?')])
    out = tmp_path / 'out'
    record = prepare([('source', path)], out, 180_000)
    assert out.stat().st_mode & 0o777 == 0o700
    assert record['selection'].startswith('first ZeroSCROLLS target per question')
    assert 'constructed' in ' '.join(record['limitations'])
    assert len(record['requests']) == 2
    with pytest.raises(FileExistsError):
        prepare([('source', path)], out, 180_000)


def test_leval_preserves_document_questions_and_targets(tmp_path):
    from g3_longqa_acquire import make_leval_records
    path = tmp_path / 'leval.jsonl'
    source = {'input': 'first paragraph\n\nsecond paragraph\n\nthird paragraph\n',
              'instructions': ['who?', 'where?'], 'outputs': ['alice', 'paris']}
    write(path, [source])
    records = make_leval_records(path, 'leval')
    assert len(records) == 2
    for i, r in enumerate(records):
        assert r['prompt'] == 'Question: ' + source['instructions'][i] + '\n\nContext:\n' + source['input'] + '\n\nAnswer:'
        assert r['target'] == source['outputs'][i]
        data = r['prompt'].encode()
        assert b''.join(data[s['start_byte']:s['end_byte']] for s in r['spans']) == source['input'].encode()
    assert records[0]['question_prefix_sha256'] != records[1]['question_prefix_sha256']
    source['outputs'] = ['only one']
    write(path, [source])
    with pytest.raises(ValueError, match='question/answer list'):
        make_leval_records(path, 'leval')
