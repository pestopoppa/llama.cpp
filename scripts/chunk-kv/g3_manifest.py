#!/usr/bin/env python3
"""Freeze and verify tokenized G3 task corpora; no inference or result grading.

Input JSONL corpus entries contain id, tokens, chunks [{offset,n_tokens}]. Task
JSONL entries specify either exact_token_match/target_tokens or an explicitly pinned
orchestrator_text_v1 deterministic scorer, method, config and expected text. Both inputs
must be independently sourced, not synthesized from model results. The output
manifest binds exact bytes and predeclared SHA256(id) % 5 split. A gate cohort
must satisfy P0's 200/50-per-bucket requirements; shared chunks must occur under
at least two distinct offsets or predecessor token sequences. This module never
claims that inputs are actual B7 traffic: source identity and task suitability
remain the corpus owner's responsibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

SCHEMA = 'epyc.kpf62.g3-manifest.v1'
BUCKETS = ((1024, 4096), (4096, 8192), (8192, 16384), (16384, 32769))
GATE_MIN_REQUESTS = 200
GATE_MIN_PER_BUCKET = 50


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def split(rid):
    return 'dev' if int(hashlib.sha256(rid.encode('utf-8')).hexdigest(), 16) % 5 == 0 else 'gate'


def records(path):
    out = {}
    with open(path, encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
                raise ValueError(f'{path}:{number}: missing id')
            if row['id'] in out:
                raise ValueError(f'{path}:{number}: duplicate id')
            out[row['id']] = row
    if not out:
        raise ValueError(f'{path}: empty')
    return out


def _tokens(value):
    return isinstance(value, list) and bool(value) and all(type(x) is int and x >= 0 for x in value)


def inspect(corpus_path, tasks_path):
    corpus, tasks = records(corpus_path), records(tasks_path)
    if set(corpus) != set(tasks):
        raise ValueError('corpus and task ids differ')
    chunk_occurrences, rows = {}, []
    for rid in sorted(corpus):
        item, task = corpus[rid], tasks[rid]
        toks, chunks = item.get('tokens'), item.get('chunks')
        if not _tokens(toks) or not BUCKETS[0][0] <= len(toks) < BUCKETS[-1][1]:
            raise ValueError(f'{rid}: invalid token sequence or length')
        if not isinstance(chunks, list) or not 2 <= len(chunks) <= 16:
            raise ValueError(f'{rid}: invalid chunk count')
        previous_end = 0
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError(f'{rid}: invalid chunk')
            off, n = chunk.get('offset'), chunk.get('n_tokens')
            if type(off) is not int or type(n) is not int or off < previous_end or n <= 0 or off + n >= len(toks):
                raise ValueError(f'{rid}: invalid or overlapping chunk, or missing suffix')
            body = json.dumps(toks[off:off+n], separators=(',', ':')).encode()
            predecessor = json.dumps(toks[:off], separators=(',', ':')).encode()
            key = hashlib.sha256(body).hexdigest()
            chunk_occurrences.setdefault(key, set()).add((off, hashlib.sha256(predecessor).hexdigest()))
            previous_end = off + n
        metric = task.get('metric')
        if metric == 'exact_token_match':
            if not _tokens(task.get('target_tokens')):
                raise ValueError(f'{rid}: unsupported metric or invalid target tokens')
        elif metric == 'orchestrator_text_v1':
            if (task.get('scoring_method') not in ('substring', 'exact_match', 'f1')
                    or not isinstance(task.get('expected'), str) or not task['expected'].strip()
                    or not isinstance(task.get('scoring_config'), dict)
                    or not isinstance(task.get('scorer_sha256'), str)
                    or not re.fullmatch(r'[0-9a-f]{64}', task['scorer_sha256'])):
                raise ValueError(f'{rid}: unsupported metric or incomplete pinned text scorer')
        else:
            raise ValueError(f'{rid}: unsupported metric')
        rows.append({'id': rid, 'split': split(rid), 'n_tokens': len(toks), 'n_chunks': len(chunks),
                     'suffix_start': previous_end, 'metric': task['metric']})
    if any(len(v) < 2 for v in chunk_occurrences.values()):
        raise ValueError('each chunk must occur at >=2 offsets or predecessor histories')
    gate = [r for r in rows if r['split'] == 'gate']
    if len(gate) < GATE_MIN_REQUESTS:
        raise ValueError('gate split requires >=200 requests')
    if any(sum(lo <= r['n_tokens'] < hi for r in gate) < GATE_MIN_PER_BUCKET for lo, hi in BUCKETS):
        raise ValueError('gate split requires >=50 requests per length bucket')
    return rows


def make_manifest(corpus_path, tasks_path, role, source):
    if not isinstance(role, str) or not role.strip() or not isinstance(source, str) or not source.strip():
        raise ValueError('role and task-source description required')
    corpus_hash, tasks_hash = sha(corpus_path), sha(tasks_path)
    rows = inspect(corpus_path, tasks_path)
    if corpus_hash != sha(corpus_path) or tasks_hash != sha(tasks_path):
        raise ValueError('corpus or task bytes changed while freezing')
    return {'schema': SCHEMA, 'role': role, 'task_source': source,
            'corpus_sha256': corpus_hash, 'tasks_sha256': tasks_hash, 'requests': rows}


def verify(manifest_path, corpus_path, tasks_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA:
        raise ValueError('unsupported G3 manifest schema')
    expected = make_manifest(corpus_path, tasks_path, manifest.get('role'), manifest.get('task_source'))
    if manifest != expected:
        raise ValueError('manifest differs from corpus or task bytes/metadata')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--role', required=True)
    parser.add_argument('--task-source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        result = make_manifest(args.corpus, args.tasks, args.role, args.task_source)
        # Exclusive create prevents overwriting a frozen manifest.
        with os.fdopen(os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), 'w', encoding='utf-8') as f:
            json.dump(result, f, sort_keys=True, indent=2)
            f.write('\n')
        print(f'prepared {args.out}: sha256={sha(args.out)}')
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f'REFUSED: {exc}\n')


if __name__ == '__main__':
    main()
