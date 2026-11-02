#!/usr/bin/env python3
"""Create a *prospective alternate* MuSiQue source inventory, without inference.

This deliberately does not call the output a B7 capture or a qualified G3 corpus.
It preserves exact original request and answer hashes, original paragraph offsets,
and source-file line identities; it never writes private or synthetic prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from g3_manifest import split


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def collect(path: Path) -> list[dict]:
    source_digest = digest(path.read_bytes())
    candidates = []
    with path.open(encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            x = json.loads(line)
            rid, text, target = x.get('id'), x.get('input'), x.get('output')
            starts, lo, hi = x.get('inner_docs_start_indices'), x.get('document_start_index'), x.get('document_end_index')
            if not isinstance(rid, str) or not rid or not isinstance(text, str) or not isinstance(target, str) or not target:
                continue
            if not isinstance(starts, list) or not starts or type(lo) is not int or type(hi) is not int:
                continue
            if starts[0] != lo or any(type(v) is not int for v in starts) or starts != sorted(set(starts)) or not (0 <= lo < hi <= len(text)):
                raise ValueError(f'{lineno}: invalid document spans')
            if any(not lo <= start < hi for start in starts):
                raise ValueError(f'{lineno}: chunk outside document')
            chunks = []
            for a, b in zip(starts, starts[1:] + [hi]):
                body = text[a:b]
                if not body:
                    raise ValueError(f'{lineno}: empty source chunk')
                chunks.append({'start_char': a, 'end_char': b, 'sha256': digest(body.encode()),
                               'predecessor_sha256': digest(text[:a].encode())})
            candidates.append({'id': rid, 'source_line': lineno, 'source_line_sha256': digest(line.encode()),
                               'prompt_sha256': digest(text.encode()), 'target_sha256': digest(target.encode()),
                               'prompt_chars': len(text), 'chunks': chunks, 'split': split(rid)})
    if not candidates or len({r['id'] for r in candidates}) != len(candidates):
        raise ValueError('empty or duplicate source IDs')
    # Use only paragraphs recurring at two real positions/predecessors.  Iterate to
    # a fixed point because dropping an unqualified request can orphan a chunk.
    rows = candidates
    while True:
        occurrences = defaultdict(set)
        for row in rows:
            for c in row['chunks']:
                occurrences[c['sha256']].add((c['start_char'], c['predecessor_sha256']))
        eligible = {h for h, places in occurrences.items() if len(places) >= 2}
        keep = [r for r in rows if sum(c['sha256'] in eligible for c in r['chunks']) >= 2]
        if len(keep) == len(rows):
            break
        rows = keep
    occurrences = defaultdict(set)
    for row in rows:
        for c in row['chunks']:
            occurrences[c['sha256']].add((c['start_char'], c['predecessor_sha256']))
    eligible = {h for h, places in occurrences.items() if len(places) >= 2}
    # Cap 16 without inventing segments; a later token-boundary and G3 audit must
    # rerun recurrence checks because a capped selection may orphan an occurrence.
    selected = []
    for r in rows:
        c = [x for x in r['chunks'] if x['sha256'] in eligible][:16]
        selected.append({**{k: v for k, v in r.items() if k != 'chunks'}, 'chunks': c})
    return [{'schema': 'epyc.kpf62.musique-source-selection.v1', 'source_path': str(path.resolve()),
             'source_sha256': source_digest, 'selection': 'unaltered MuSiQue requests, repeated original paragraph bytes',
             'scope': 'prospective alternate only; not B7 traffic, not tokenized, not a G3 manifest',
             'requests': selected}]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = collect(args.source)[0]
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')
    print(f"selected={len(data['requests'])}, gate={sum(r['split']=='gate' for r in data['requests'])}, sha256={digest(args.out.read_bytes())}")


if __name__ == '__main__':
    main()
