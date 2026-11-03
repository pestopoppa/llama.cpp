#!/usr/bin/env python3
"""Offline export tests and released-map native BF16 roundtrip; no inference."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--artifact-dir', type=Path, required=True)
p.add_argument('--native-inspector', type=Path, required=True)
args = p.parse_args()
args.artifact_dir.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('export_maps', Path(__file__).with_name('export_maps.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import torch
torch.set_num_threads(1)
failures = []
passed = []
def check(value, label):
    (passed if value else failures).append(label)
    print(('PASS ' if value else 'FAIL ') + label)

def must_reject(maps, label, **changes):
    options = dict(model_identity='Qwen3.8-35B-fixture', source_sha256='a'*64,
                   identity='synthetic-test', q_heads=16, kv_heads=2,
                   head_dim=256, rope_dim=64, layers=[3])
    options.update(changes)
    try:
        module.encode_maps(maps, **options)
    except (ValueError, TypeError):
        check(True, label)
    else:
        check(False, label)

maps = {'3.W_Q_idx': torch.zeros((384, 64), dtype=torch.bfloat16),
        '3.W_K_idx': torch.ones((384, 64), dtype=torch.bfloat16)}
blob = module.encode_maps(maps, model_identity='Qwen3.8-35B-fixture', source_sha256='a'*64,
                          identity='synthetic-test', q_heads=16, kv_heads=2,
                          head_dim=256, rope_dim=64, layers=[3])
check(blob[:8] == b'LSAMAP1\0' and len(blob) == 108 + len('Qwen3.8-35B-fixture') + len('synthetic-test') + 4 + 384*64*4,
      '35B384x64 exact format length')
must_reject({'3.W_K_idx': maps['3.W_K_idx']}, 'missing Q buffer rejected')
must_reject({'3.W_Q_idx': maps['3.W_Q_idx']}, 'missing K buffer rejected')
must_reject(dict(maps, **{'7.W_Q_idx': maps['3.W_Q_idx']}), 'extra query layer rejected')
must_reject(maps, 'wrong global layer set rejected', layers=[7])
must_reject(maps, 'duplicate global layers rejected', layers=[3, 3])
must_reject(dict(maps, **{'3.W_Q_idx': torch.zeros((768,64))}), '27B tensor cannot masquerade as35B')
bad = dict(maps); bad['3.W_Q_idx'] = bad['3.W_Q_idx'].clone(); bad['3.W_Q_idx'][0,0] = float('nan')
must_reject(bad, 'nonfinite tensor rejected')
bad = dict(maps); bad['3.W_Q_idx'] = torch.full((384,64), 3.4e38)
must_reject(bad, 'finite FP32 overflow during BF16 conversion rejected')
must_reject(maps, 'invalid source digest rejected', source_sha256='no')
must_reject(maps, 'empty model binding rejected', model_identity='')
must_reject(maps, 'embedded NUL identity rejected', identity='x\0y')

release_dir = Path('/tmp/lsa-release-inspect')
layers = list(range(3,64,4))
releases = {
    'pca': ('qwen38_27b__long-16k-1M-heavy-key-and-query-pca-step0.pt',
            'baaa28cb00b4378b3d33e02dc971e2781482d5c4d45e29e9140143bdf7832e65'),
    'distilled': ('qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt',
                  '24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b'),
}
receipts = []
for kind, (filename, digest) in releases.items():
    source = release_dir / filename
    check(hashlib.sha256(source.read_bytes()).hexdigest() == digest, f'{kind} approved PT source digest')
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        continue
    output = args.artifact_dir / f'{kind}.lsamap'
    dump = args.artifact_dir / f'{kind}.native-buffers.bin'
    # Deterministic fixture generation; exclusive artifact semantics in exporter.
    command = [os.sys.executable, str(Path(__file__).with_name('export_maps.py')),
               str(source), str(output), '--source-sha256', digest,
               '--model-identity', 'Qwen3.8-27B', '--identity', f'released-{kind}',
               '--geometry', '27b', '--layers', ','.join(map(str,layers))]
    result = subprocess.run(command, capture_output=True, text=True)
    check(result.returncode == 0, f'{kind} exporter CLI succeeds')
    if result.returncode:
        print(result.stderr)
        continue
    loaded = subprocess.run([str(args.native_inspector), str(output), 'Qwen3.8-27B', digest,
                             '27b', ','.join(map(str,layers)), str(dump)], capture_output=True, text=True)
    check(loaded.returncode == 0, f'{kind} native loader accepts exact binding')
    if loaded.returncode:
        print(loaded.stderr)
        continue
    original = torch.load(source, map_location='cpu', weights_only=True)
    expected = bytearray()
    for layer in layers:
        expected += struct.pack('<I',layer)
        for axis in ('Q','K'):
            words = original[f'{layer}.W_{axis}_idx'].to(torch.bfloat16).contiguous().view(torch.int16).reshape(-1).tolist()
            expected += struct.pack('<'+'H'*len(words),*(x & 65535 for x in words))
    check(dump.read_bytes() == expected, f'{kind} native Q/K BF16 bit equality all16layers')
    rejected = subprocess.run([str(args.native_inspector), str(output), 'Qwen3.8-27B', 'b'*64,
                               '27b', ','.join(map(str,layers))], capture_output=True, text=True)
    check(rejected.returncode == 1, f'{kind} native approved-source mismatch rejected')
    receipts.append(dict(kind=kind, source=str(source), source_sha256=digest,
                         output=str(output), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                         native_dump_sha256=hashlib.sha256(dump.read_bytes()).hexdigest(),
                         bit_equality=dump.read_bytes() == expected,
                         loader_stdout=loaded.stdout.strip()))
receipt = dict(schema='lsa-native-map-roundtrip-v1', passed=passed, failed=failures,
               releases=receipts, scope='offline export/native map unit validation; no inference, GPU, or graph claim')
(args.artifact_dir / 'roundtrip.json').write_text(json.dumps(receipt, indent=2)+'\n')
raise SystemExit(bool(failures))
