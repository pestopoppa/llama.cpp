#!/usr/bin/env python3
"""Exploratory paired token-F1 on dev IDs; never the frozen G3 task gate."""
import argparse
import collections
import hashlib
import json
import re
import string
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\b(a|an|the)\b', ' ', text).split()


def token_f1(answer, reference):
    a, b = normalize(answer), normalize(reference)
    if not a or not b:
        return float(a == b)
    overlap = sum((collections.Counter(a) & collections.Counter(b)).values())
    return 2 * overlap / (len(a) + len(b))


def score(manifest, selection, inputs, outcomes, ids):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty or duplicate requested IDs')
    declared = {x['id']: x for x in manifest['requests']}
    selected = {x['id']: x for x in selection['requests']}
    sources = {x['id']: x for x in inputs if x['id'] in ids}
    if len(sources) != len(ids) or any(declared.get(i, {}).get('split') != 'dev' for i in ids):
        raise ValueError('missing source or non-dev ID')
    pairs = {}
    for row in outcomes:
        key = (row.get('id'), row.get('arm'))
        if key[0] not in ids or key[1] not in ('full', 'reuse') or key in pairs:
            raise ValueError('unexpected or duplicate outcome')
        if row.get('complete') is not True or row.get('error') is not None or not isinstance(row.get('output_text'), str):
            raise ValueError('failed, truncated or missing outcome')
        pairs[key] = row['output_text']
    if set(pairs) != {(i, arm) for i in ids for arm in ('full', 'reuse')}:
        raise ValueError('incomplete paired coverage')
    rows = []
    for rid in ids:
        reference = sources[rid]['target']
        if hashlib.sha256(reference.encode()).hexdigest() != selected.get(rid, {}).get('target_sha256'):
            raise ValueError('target no longer matches source selection')
        rows.append({'id': rid, 'full_f1': token_f1(pairs[(rid, 'full')], reference),
                     'reuse_f1': token_f1(pairs[(rid, 'reuse')], reference)})
    full, reuse = sum(r['full_f1'] for r in rows), sum(r['reuse_f1'] for r in rows)
    return {'schema': 'epyc.kpf62.dev-token-f1.v1', 'classification': 'exploratory dev observation; not G3 gate or B7',
            'n': len(rows), 'full_mean_f1': full / len(rows), 'reuse_mean_f1': reuse / len(rows),
            'ratio': reuse / full if full else None, 'items': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'selection', 'inputs', 'outcomes', 'ids', 'out'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    manifest, selection = json.loads(Path(a.manifest).read_text()), json.loads(Path(a.selection).read_text())
    ids = [x.strip() for x in Path(a.ids).read_text().splitlines() if x.strip()]
    result = score(manifest, selection, read_jsonl(a.inputs), read_jsonl(a.outcomes), ids)
    result['input_sha256'] = {name: digest(getattr(a, name)) for name in ('manifest', 'selection', 'inputs', 'outcomes', 'ids')}
    Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
