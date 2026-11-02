#!/usr/bin/env python3
"""Tokenize the pinned MuSiQue selection with a vocab-only llama tokenizer.

Prospective alternate, not a B7 capture or a G3 gate: original source prompts and
answers are not altered. Writes request tokens and target tokens to a NEW directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from g3_manifest import inspect, split
from g3_musique_acquire import digest


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def materialize(source: Path, selection: Path, model: Path, tokenizer: Path, lib_dir: Path, header_bytes: int, out: Path):
    doc = json.loads(selection.read_text(encoding='utf-8'))
    if doc.get('schema') != 'epyc.kpf62.musique-source-selection.v1' or file_sha(source) != doc['source_sha256']:
        raise ValueError('selection source missing or changed')
    if header_bytes <= 0 or header_bytes > model.stat().st_size:
        raise ValueError('invalid model vocabulary-header extent')
    expected = {r['source_line']: r for r in doc['requests']}
    if len(expected) != len(doc['requests']):
        raise ValueError('duplicate source lines')
    lines = []
    with source.open(encoding='utf-8') as f:
        for num, line in enumerate(f, 1):
            if num not in expected:
                continue
            row = json.loads(line); ref = expected[num]
            prompt, target = row['input'], row['output']
            if (row['id'] != ref['id'] or digest(line.encode()) != ref['source_line_sha256'] or
                    digest(prompt.encode()) != ref['prompt_sha256'] or digest(target.encode()) != ref['target_sha256']):
                raise ValueError(f'{num}: source drift')
            spans = []
            for c in ref['chunks']:
                a, b = c['start_char'], c['end_char']
                if not (0 <= a < b <= len(prompt)) or digest(prompt[a:b].encode()) != c['sha256']:
                    raise ValueError(f'{num}: chunk span drift')
                spans.append({'start_byte': len(prompt[:a].encode()), 'end_byte': len(prompt[:b].encode())})
            lines.append(json.dumps({'id': row['id'], 'prompt': prompt, 'target': target, 'spans': spans}, ensure_ascii=False))
    if len(lines) != len(expected):
        raise ValueError('source line missing')
    before = file_sha(tokenizer)
    library_paths = sorted({p.resolve() for pattern in ('libllama*.so*', 'libggml*.so*')
                            for p in lib_dir.glob(pattern) if p.is_file()})
    if not any(p.name.startswith('libllama.') for p in library_paths) or not any(p.name.startswith('libggml') for p in library_paths):
        raise ValueError('candidate llama/ggml libraries missing')
    library_hashes = {str(p): file_sha(p) for p in library_paths}
    with model.open('rb') as f:
        header_hash = digest(f.read(header_bytes))
    env = dict(os.environ, LD_LIBRARY_PATH=str(lib_dir.resolve()))
    proc = subprocess.run([str(tokenizer.resolve()), str(model.resolve())], input='\n'.join(lines) + '\n',
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False)
    if proc.returncode:
        raise ValueError(f'vocabulary tokenizer failed rc={proc.returncode}: {proc.stderr[-1000:]}')
    if (file_sha(tokenizer) != before or file_sha(source) != doc['source_sha256'] or
            any(file_sha(Path(path)) != value for path, value in library_hashes.items())):
        raise ValueError('inputs changed during tokenization')
    with model.open('rb') as f:
        if digest(f.read(header_bytes)) != header_hash:
            raise ValueError('vocabulary header changed during tokenization')
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    if len(rows) != len(expected) or {r['id'] for r in rows} != {r['id'] for r in doc['requests']}:
        raise ValueError('tokenizer request coverage mismatch')
    original = len(rows)
    # Retain exact source prompts only. Drop invalid/short/long and boundary merges;
    # repeatedly prune orphaned exact token chunks until every retained chunk has
    # >=2 observed placements and every retained request has >=2 chunks.
    rows = [r for r in rows if 1024 <= len(r['tokens']) <= 32768 and len(r['chunks']) >= 2 and r['target_tokens']]
    while True:
        seen = defaultdict(set)
        for r in rows:
            for c in r['chunks']:
                a, n = c['offset'], c['n_tokens']
                key = tuple(r['tokens'][a:a+n]); pred = tuple(r['tokens'][:a])
                seen[key].add((a, hashlib.sha256(json.dumps(pred).encode()).hexdigest()))
        eligible = {k for k, v in seen.items() if len(v) >= 2}
        updated = []
        for r in rows:
            chunks = [c for c in r['chunks'] if tuple(r['tokens'][c['offset']:c['offset']+c['n_tokens']]) in eligible]
            if len(chunks) >= 2:
                updated.append({**r, 'chunks': chunks})
        old = [(r['id'], len(r['chunks'])) for r in rows]
        new = [(r['id'], len(r['chunks'])) for r in updated]
        rows = updated
        if old == new:
            break
    out.mkdir(parents=True, exist_ok=False)
    corpus, tasks = out / 'corpus.jsonl', out / 'tasks.jsonl'
    with corpus.open('x') as c, tasks.open('x') as t:
        for r in rows:
            c.write(json.dumps({'id': r['id'], 'tokens': r['tokens'], 'chunks': r['chunks']}, separators=(',', ':')) + '\n')
            t.write(json.dumps({'id': r['id'], 'metric': 'exact_token_match', 'target_tokens': r['target_tokens']}, separators=(',', ':')) + '\n')
    try:
        inspect(corpus, tasks)
        status = 'STRUCTURALLY_ELIGIBLE_NOT_APPROVED'
        issue = None
    except ValueError as exc:
        status = 'NOT_G3_ELIGIBLE'
        issue = str(exc)
    counts = {str(k): sum(1 for r in rows if split(r['id']) == 'gate' and k <= len(r['tokens']) < hi)
              for k, hi in ((1024, 4096), (4096, 8192), (8192, 16384), (16384, 32769))}
    report = {'schema': 'epyc.kpf62.musique-alternate.v1', 'source_sha256': doc['source_sha256'],
              'selection_sha256': file_sha(selection), 'model_path': str(model.resolve()),
              'model_header_bytes': header_bytes, 'model_header_sha256': header_hash,
              'tokenizer_sha256': before, 'lib_dir': str(lib_dir.resolve()),
              'library_sha256': library_hashes,
              'original_requests': original, 'retained_requests': len(rows), 'gate_bucket_counts': counts,
              'tokenized_corpus_sha256': file_sha(corpus), 'tokenized_tasks_sha256': file_sha(tasks),
              'status': status, 'structural_refusal': issue,
              'limitations': ['alternate dataset, not B7 traffic', 'no full model-byte identity asserted',
                              'no G3 decision or task metric approval', 'raw input and gold are unmodified',
                              'no model inference performed']}
    (out / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'selection', 'model', 'tokenizer', 'lib-dir', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--model-header-bytes', type=int, required=True)
    a = p.parse_args()
    try:
        report = materialize(a.source, a.selection, a.model, a.tokenizer, a.lib_dir, a.model_header_bytes, a.out)
        print(json.dumps(report, sort_keys=True))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        p.exit(2, f'REFUSED: {exc}\n')


if __name__ == '__main__':
    main()
