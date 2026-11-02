#!/usr/bin/env python3
"""Prepare a source-bound prospective long-QA G3 alternate, without inference.

A question-first prefix is copied from the dataset's own original question; the
entire original prompt and the first retained human target remain byte-for-byte.
The resulting prompt is a *new format*, not observed serving or B7 traffic. Its
exact-token answer score is diagnostic pending independent metric admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from g3_manifest import inspect, split


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def make_records(source: Path, name: str, *, char_limit: int = 180_000) -> list[dict]:
    """One distinct question per dataset ID; exact duplicated prompts retained once."""
    rows, seen_ids = [], set()
    with source.open(encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            x = json.loads(line)
            rid, prompt, target = x.get('id'), x.get('input'), x.get('output')
            if not isinstance(rid, str) or not rid or rid in seen_ids:
                continue
            if not isinstance(prompt, str) or not isinstance(target, str) or not target:
                continue
            a, b, q0, q1 = (x.get(k) for k in ('document_start_index', 'document_end_index',
                                                 'query_start_index', 'query_end_index'))
            if any(type(v) is not int for v in (a, b, q0, q1)) or not (0 <= a < b <= q0 < q1 <= len(prompt)):
                raise ValueError(f'{name}:{number}: invalid source spans')
            if len(prompt) > char_limit:
                continue  # bounded candidate search, not token truncation
            question, document = prompt[q0:q1], prompt[a:b]
            if not question.strip() or len(document) < 2:
                continue
            # Split existing document bytes at an existing newline. No document
            # text is deleted, normalized, fabricated, or padded.
            middle = len(document) // 2
            cut = document.find('\n', middle)
            if cut < 0:
                cut = document.rfind('\n', 0, middle)
            if cut <= 0 or cut >= len(document) - 1:
                continue
            cut += 1
            prefix = question + '\n\n'
            rendered = prefix + prompt
            starts = (len(prefix) + a, len(prefix) + a + cut, len(prefix) + b)
            spans = []
            for left, right in zip(starts, starts[1:]):
                spans.append({'start_byte': len(rendered[:left].encode()),
                              'end_byte': len(rendered[:right].encode())})
            rows.append({'id': f'{name}:{rid}', 'prompt': rendered, 'target': target, 'spans': spans,
                         'source_line': number, 'source_line_sha256': sha(line.encode()),
                         'source_prompt_sha256': sha(prompt.encode()),
                         'question_prefix_sha256': sha(prefix.encode()),
                         'rendered_prompt_sha256': sha(rendered.encode()),
                         'target_sha256': sha(target.encode()), 'document_sha256': sha(document.encode())})
            seen_ids.add(rid)
    docs = defaultdict(set)
    for row in rows:
        docs[row['document_sha256']].add(row['question_prefix_sha256'])
    return [r for r in rows if len(docs[r['document_sha256']]) >= 2]


def make_leval_records(source: Path, name: str, *, char_limit: int = 180_000) -> list[dict]:
    """Use original L-Eval document/instruction/gold tuples; format prospectively."""
    rows = []
    with source.open(encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            x = json.loads(line)
            document, questions, answers = x.get('input'), x.get('instructions'), x.get('outputs')
            if not isinstance(document, str) or len(document) > char_limit:
                continue
            if not isinstance(questions, list) or not isinstance(answers, list) or len(questions) != len(answers):
                raise ValueError(f'{name}:{number}: invalid question/answer list')
            pairs = [(i, q, a) for i, (q, a) in enumerate(zip(questions, answers, strict=True))
                     if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip()]
            if len({q for _, q, _ in pairs}) < 2:
                continue
            middle = len(document) // 2
            cut = document.find('\n', middle)
            if cut < 0:
                cut = document.rfind('\n', 0, middle)
            if cut <= 0 or cut >= len(document) - 1:
                continue
            cut += 1
            for i, question, target in pairs:
                prefix = 'Question: ' + question + '\n\nContext:\n'
                rendered = prefix + document + '\n\nAnswer:'
                starts = (len(prefix), len(prefix) + cut, len(prefix) + len(document))
                spans = [{'start_byte': len(rendered[:a].encode()), 'end_byte': len(rendered[:b].encode())}
                         for a, b in zip(starts, starts[1:])]
                rows.append({'id': f'{name}:{number}:{i}', 'prompt': rendered, 'target': target, 'spans': spans,
                             'source_line': number, 'question_index': i, 'source_line_sha256': sha(line.encode()),
                             'source_prompt_sha256': sha(document.encode()),
                             'question_prefix_sha256': sha(prefix.encode()),
                             'rendered_prompt_sha256': sha(rendered.encode()),
                             'target_sha256': sha(target.encode()), 'document_sha256': sha(document.encode())})
    return rows


def prepare(sources: list[tuple[str, Path]], out: Path, char_limit: int, leval_sources: list[tuple[str, Path]] | None = None) -> dict:
    inputs = []
    for name, path in sources:
        before = file_sha(path)
        inputs += make_records(path, name, char_limit=char_limit)
        if before != file_sha(path):
            raise ValueError(f'{name}: source changed during read')
    leval_sources = leval_sources or []
    for name, path in leval_sources:
        before = file_sha(path)
        inputs += make_leval_records(path, name, char_limit=char_limit)
        if before != file_sha(path):
            raise ValueError(f'{name}: source changed during read')
    if len({r['id'] for r in inputs}) != len(inputs):
        raise ValueError('duplicate cross-source IDs')
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    with (out / 'tokenize-input.jsonl').open('x', encoding='utf-8') as f:
        for r in inputs:
            f.write(json.dumps({k: r[k] for k in ('id', 'prompt', 'target', 'spans')}, ensure_ascii=False) + '\n')
    evidence = {'schema': 'epyc.kpf62.longqa-alternate-selection.v1',
                'selection': 'first ZeroSCROLLS target per question; prepend original question to intact prompt; L-Eval original document, question and answer in a constructed question-first template',
                'char_limit': char_limit, 'sources': {name: {'path': str(p.resolve()), 'sha256': file_sha(p)} for name, p in sources + leval_sources},
                'requests': [{k: v for k, v in r.items() if k not in ('prompt', 'target', 'spans')} for r in inputs],
                'limitations': ['prospective alternate, not B7 or observed traffic',
                                'question-first prompts are constructed, not original serving templates',
                                'one source target selected; exact-token scorer not admitted',
                                'char bound excludes long source prompts without truncation']}
    (out / 'selection.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    return evidence


def finalize(out: Path, tokenizer: Path, model: Path, lib_dir: Path, header_bytes: int, music: Path | None):
    token_input, token_output = out / 'tokenize-input.jsonl', out / 'tokenized.jsonl'
    files = sorted({p.resolve() for pat in ('libllama*.so*', 'libggml*.so*') for p in lib_dir.glob(pat) if p.is_file()})
    if not files or header_bytes <= 0 or header_bytes > model.stat().st_size:
        raise ValueError('missing libraries or invalid vocabulary extent')
    identity = {'tokenizer_sha256': file_sha(tokenizer), 'token_input_sha256': file_sha(token_input),
                'libraries': {str(p): file_sha(p) for p in files}}
    with model.open('rb') as f:
        identity['model_header_sha256'] = sha(f.read(header_bytes))
    with token_input.open('rb') as i, token_output.open('xb') as o, (out / 'tokenizer.stderr').open('xb') as e:
        proc = subprocess.run([str(tokenizer), str(model)], stdin=i, stdout=o, stderr=e,
                              env=dict(os.environ, LD_LIBRARY_PATH=str(lib_dir)), check=False)
    if proc.returncode:
        raise ValueError(f'tokenizer rc={proc.returncode}; output retained')
    if file_sha(tokenizer) != identity['tokenizer_sha256'] or file_sha(token_input) != identity['token_input_sha256'] or any(file_sha(Path(p)) != h for p, h in identity['libraries'].items()):
        raise ValueError('inputs changed during tokenization')
    with model.open('rb') as f:
        if sha(f.read(header_bytes)) != identity['model_header_sha256']:
            raise ValueError('model header changed during tokenization')
    expected = {r['id'] for r in json.loads((out / 'selection.json').read_text())['requests']}
    rows = [json.loads(line) for line in token_output.open() if line.strip()]
    if len(rows) != len(expected) or {r['id'] for r in rows} != expected:
        raise ValueError('tokenizer coverage mismatch')
    rows = [r for r in rows if 1024 <= len(r['tokens']) <= 32768 and len(r['chunks']) >= 2 and r['target_tokens']]
    # Retain only chunks independently appearing in >=2 distinct predecessor histories.
    while True:
        places = defaultdict(set)
        for r in rows:
            for c in r['chunks']:
                a, n = c['offset'], c['n_tokens']
                key = sha(json.dumps(r['tokens'][a:a+n], separators=(',', ':')).encode())
                places[key].add((a, sha(json.dumps(r['tokens'][:a], separators=(',', ':')).encode())))
        eligible = {k for k, v in places.items() if len(v) >= 2}
        next_rows = []
        for r in rows:
            chunks = [c for c in r['chunks'] if sha(json.dumps(r['tokens'][c['offset']:c['offset']+c['n_tokens']], separators=(',', ':')).encode()) in eligible]
            if len(chunks) >= 2:
                next_rows.append({**r, 'chunks': chunks})
        old = [(r['id'], len(r['chunks'])) for r in rows]
        new = [(r['id'], len(r['chunks'])) for r in next_rows]
        rows = next_rows
        if old == new:
            break
    corpus, tasks = out / 'corpus.jsonl', out / 'tasks.jsonl'
    music_inputs = None
    if music:
        music_inputs = {name: file_sha(music / name) for name in ('corpus.jsonl', 'tasks.jsonl', 'report.json')}
    with corpus.open('x') as c, tasks.open('x') as t:
        for r in rows:
            c.write(json.dumps({'id': r['id'], 'tokens': r['tokens'], 'chunks': r['chunks']}, separators=(',', ':')) + '\n')
            t.write(json.dumps({'id': r['id'], 'metric': 'exact_token_match', 'target_tokens': r['target_tokens']}, separators=(',', ':')) + '\n')
        if music:
            with (music / 'corpus.jsonl').open() as mc, (music / 'tasks.jsonl').open() as mt:
                for cl, tl in zip(mc, mt, strict=True):
                    if json.loads(cl)['id'] != json.loads(tl)['id']:
                        raise ValueError('MuSiQue task mismatch')
                    c.write(cl); t.write(tl)
    if music_inputs and any(file_sha(music / name) != digest for name, digest in music_inputs.items()):
        raise ValueError('combined alternate inputs changed during copy')
    counts = Counter()
    with corpus.open() as f:
        for line in f:
            r = json.loads(line); n = len(r['tokens'])
            if split(r['id']) == 'gate':
                counts[str(next(lo for lo, hi in ((1024,4096),(4096,8192),(8192,16384),(16384,32769)) if lo <= n < hi))] += 1
    try:
        inspected = inspect(corpus, tasks)
        status, error = 'STRUCTURALLY_ELIGIBLE_NOT_APPROVED', None
    except ValueError as exc:
        inspected, status, error = [], 'NOT_G3_ELIGIBLE', str(exc)
    report = {'schema':'epyc.kpf62.longqa-alternate.v1', 'source_selection_sha256': file_sha(out / 'selection.json'),
              'input_records': len(expected), 'long_records_retained': len(rows), 'gate_bucket_counts': dict(counts),
              'combined_alternate_inputs': music_inputs,
              'corpus_sha256':file_sha(corpus), 'tasks_sha256':file_sha(tasks), 'identity':identity,
              'status':status, 'refusal':error, 'manifest_request_count':len(inspected),
              'limitations':['not B7 traffic', 'question-first prompts are constructed from original questions and prompts',
                             'exact-token-match task metric not admitted', 'no model inference']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', action='append', default=[], help='NAME=ZeroSCROLLS.jsonl')
    ap.add_argument('--leval', action='append', default=[], help='NAME=L-Eval.jsonl')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--char-limit', type=int, default=180_000)
    ap.add_argument('--tokenizer', required=True, type=Path)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--lib-dir', required=True, type=Path)
    ap.add_argument('--header-bytes', type=int, required=True)
    ap.add_argument('--music', type=Path)
    args = ap.parse_args()
    try:
        sources = []
        for item in args.source:
            name, path = item.split('=', 1)
            if not name or ':' in name:
                raise ValueError('invalid source name')
            sources.append((name, Path(path)))
        leval_sources = []
        for item in args.leval:
            name, path = item.split('=', 1)
            if not name or ':' in name:
                raise ValueError('invalid source name')
            leval_sources.append((name, Path(path)))
        if len({name for name, _ in sources + leval_sources}) != len(sources + leval_sources) or not (sources or leval_sources):
            raise ValueError('duplicate source name or no source')
        prepare(sources, args.out, args.char_limit, leval_sources)
        print(json.dumps(finalize(args.out, args.tokenizer, args.model, args.lib_dir, args.header_bytes, args.music), sort_keys=True))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        ap.exit(2, f'REFUSED: {exc}\n')


if __name__ == '__main__':
    main()
