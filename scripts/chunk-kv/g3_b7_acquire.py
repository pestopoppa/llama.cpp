#!/usr/bin/env python3
"""Validate B7 source records against independently retained prompt/gold bytes.

The current B7 layout logs intentionally contain only 12-hex chunk hashes and
character offsets, not prompt bytes or independently scored targets. They are NOT
sufficient to reconstruct a G3 corpus. This tool will accept a future bounded
export only if the original self-hashed producer record also binds the FULL prompt
SHA-256, and the independent gold file supplies an exact target for its task ID.
It does not claim that a self-hash proves a record's producer or authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(b):
    return hashlib.sha256(b).hexdigest()


def record_hash(record):
    body = {k: v for k, v in record.items() if k != 'record_sha256'}
    return sha(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())


def read_rows(path):
    rows = []
    with Path(path).open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f'{path}:{n}: object required')
                rows.append(row)
    return rows


def unique_index(rows, key):
    out = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in out:
            raise ValueError(f'missing or duplicate {key}')
        out[value] = row
    return out


def acquire(records, captures, gold):
    originals = unique_index(records, 'record_sha256')
    capture_by_hash = unique_index(captures, 'record_sha256')
    gold_by_task = unique_index(gold, 'task_id')
    if not capture_by_hash:
        raise ValueError('no captures')
    out = []
    for key, capture in capture_by_hash.items():
        event = originals.get(key)
        if event is None or record_hash(event) != key:
            raise ValueError('missing, corrupt or unmatched native record')
        task_id, rid, prompt = event.get('task_id'), event.get('request_id'), capture.get('prompt')
        if not isinstance(task_id, str) or not task_id or not isinstance(rid, str) or not rid or not isinstance(prompt, str):
            raise ValueError('request, task and retained prompt required')
        if event.get('layout_match') is not True:
            raise ValueError(f'{rid}: no affirmative layout match')
        # A 12-character chunk hash does not bind the unlogged prompt. Require
        # a full request digest from the *original* producer record; never
        # silently synthesize that warrant during reconstruction.
        prompt_hash = event.get('request_prompt_sha256')
        if not isinstance(prompt_hash, str) or len(prompt_hash) != 64 or prompt_hash != sha(prompt.encode()):
            raise ValueError(f'{rid}: original full-prompt SHA-256 absent or mismatched')
        answer = gold_by_task.get(task_id)
        if not answer or not isinstance(answer.get('target'), str) or not answer['target']:
            raise ValueError(f'{rid}: independent task target missing')
        spans = []
        for block in event.get('blocks') or []:
            if not isinstance(block, dict) or block.get('kind') not in ('reference_code', 'doc_chunk'):
                continue
            for c in block.get('chunks') or []:
                if not isinstance(c, dict):
                    raise ValueError(f'{rid}: invalid chunk record')
                a, n, h = c.get('start'), c.get('length'), c.get('sha256_12')
                if type(a) is not int or type(n) is not int or n <= 0 or a < 0 or a + n > len(prompt):
                    raise ValueError(f'{rid}: chunk placement missing or invalid')
                if not isinstance(h, str) or len(h) != 12 or sha(prompt[a:a+n].encode())[:12] != h:
                    raise ValueError(f'{rid}: chunk bytes do not match native record')
                spans.append({'start_char': a, 'end_char': a+n, 'sha256': sha(prompt[a:a+n].encode())})
        spans.sort(key=lambda x: (x['start_char'], x['end_char']))
        if not 2 <= len(spans) <= 16 or any(a['end_char'] > b['start_char'] for a, b in zip(spans, spans[1:])):
            raise ValueError(f'{rid}: need 2-16 non-overlapping observed chunks')
        out.append({'id': rid, 'task_id': task_id, 'record_sha256': key,
                    'prompt_sha256': prompt_hash, 'target_sha256': sha(answer['target'].encode()), 'chunks': spans})
    if len({x['id'] for x in out}) != len(out):
        raise ValueError('duplicate request id')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    try:
        records, captures, gold = map(read_rows, (a.records, a.captures, a.gold))
        rows = acquire(records, captures, gold)
        result = {'schema': 'epyc.kpf62.b7-acquisition.v1', 'records_sha256': sha(a.records.read_bytes()),
                  'captures_sha256': sha(a.captures.read_bytes()), 'gold_sha256': sha(a.gold.read_bytes()),
                  'rows': rows, 'status': 'SOURCE_BOUND_NOT_TOKENIZED_OR_GATE_READY'}
        with a.out.open('x', encoding='utf-8') as f:
            json.dump(result, f, sort_keys=True, indent=2); f.write('\n')
        print(f"source-bound requests={len(rows)}, sha256={sha(a.out.read_bytes())}")
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        p.exit(2, f'REFUSED: {exc}\n')


if __name__ == '__main__':
    main()
