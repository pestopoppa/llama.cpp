#!/usr/bin/env python3
"""Write bounded conformance fixtures using the unchanged, hash-pinned Python oracle."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess

import torch


ORACLE_SHA256 = "96e44dc399114e0709a0f6c0820c0ddcc2c62cb28d69f7a675389ae17db68ac2"


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def load_oracle(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ORACLE_SHA256:
        raise ValueError(f"oracle digest mismatch: {digest}")
    spec = importlib.util.spec_from_file_location("lsa_bound_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_cluster_fixture(path, oracle, keys, query, positions):
    keys = keys.to(torch.bfloat16).float().contiguous()
    centroids, members = oracle.buckets(keys)
    selected = oracle.select(query, keys, positions)
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sII", b"LSAFIX1\0", len(keys), len(query)))
        for tensor in (keys, query.float(), positions.int(), members, centroids):
            stream.write(tensor_bytes(tensor))
        for row in selected:
            row = row[row >= 0]
            stream.write(struct.pack("<i", len(row)))
            stream.write(tensor_bytes(row))


def write_projection_fixture(path, oracle, generator, map_path=None, map_kind=None):
    q = torch.randn(3, 24, 256, generator=generator)
    k = torch.randn(5121, 4, 256, generator=generator)
    if map_path:
        maps = oracle.load_maps(map_path, map_kind)
        wq, wk = maps["3.W_Q_idx"].to(torch.bfloat16), maps["3.W_K_idx"].to(torch.bfloat16)
    else:
        wq = (torch.randn(768, 64, generator=generator) / 28).to(torch.bfloat16)
        wk = (torch.randn(768, 64, generator=generator) / 28).to(torch.bfloat16)
    iq, ik = oracle.project(q, k, wq, wk)
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sII", b"LSAPRO2\0", len(q), len(k)))
        for tensor in (wq, wk, q, k, iq, ik.float()):
            stream.write(tensor_bytes(tensor))
        positions = torch.tensor([0, 4096, 5120])
        stream.write(tensor_bytes(positions.int()))
        for row in oracle.select(iq, ik, positions):
            row = row[row >= 0]
            stream.write(struct.pack("<i", len(row)))
            stream.write(tensor_bytes(row))

    # Diagnose the initial failed exact-key assertion without changing its record.
    a = k[:, :, 64:].reshape(-1, 768).to(torch.bfloat16).float()
    b = wk.float()
    raw_author = (a.to(torch.bfloat16) @ wk).float()
    raw_sequential = torch.zeros_like(raw_author)
    for coordinate in range(768):
        raw_sequential.add_(a[:, coordinate, None] * b[coordinate, None, :])
    raw_sequential = raw_sequential.to(torch.bfloat16).float()
    raw_high_precision = (a.double() @ b.double()).to(torch.bfloat16).float()
    norm_sq = torch.zeros(len(a))
    for coordinate in range(64):
        norm_sq.add_(raw_sequential[:, coordinate].square())
    sequential_norm = norm_sq.sqrt().clamp_min(1e-12)
    native_model = (raw_sequential / sequential_norm[:, None]).to(torch.bfloat16).float()
    expected = ik.float()
    mismatches = (native_model != expected).nonzero()
    samples = []
    for row, column in mismatches[:12].tolist():
        samples.append({
            "row": row, "column": column,
            "raw_author_bf16_bits": int(raw_author[row, column].to(torch.bfloat16).view(torch.int16)) & 65535,
            "raw_sequential_bf16_bits": int(raw_sequential[row, column].to(torch.bfloat16).view(torch.int16)) & 65535,
            "raw_fp64_bf16_bits": int(raw_high_precision[row, column].to(torch.bfloat16).view(torch.int16)) & 65535,
            "norm_author_fp32": float(torch.linalg.vector_norm(raw_author[row])),
            "norm_sequential_fp32": float(sequential_norm[row]),
            "norm_fp64": float(torch.linalg.vector_norm(raw_author[row].double())),
            "normalized_author_fp32": float(raw_author[row, column] / torch.linalg.vector_norm(raw_author[row])),
            "normalized_sequential_fp32": float(raw_sequential[row, column] / sequential_norm[row]),
            "normalized_fp64": float(raw_author[row, column].double() / torch.linalg.vector_norm(raw_author[row].double())),
            "expected_key": float(expected[row, column]), "sequential_key": float(native_model[row, column]),
        })
    diagnostic = {
        "key_elements": expected.numel(),
        "raw_sequential_bf16_mismatches": int((raw_sequential != raw_author).sum()),
        "raw_fp64_bf16_mismatches": int((raw_high_precision != raw_author).sum()),
        "normalized_bf16_mismatches": len(mismatches), "samples": samples,
        "map_path": str(map_path) if map_path else None,
        "map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest() if map_path else None,
    }
    return iq, ik, diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-test", type=Path)
    parser.add_argument("--pca-map", type=Path)
    parser.add_argument("--distilled-map", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    oracle = load_oracle(args.oracle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    snapshot = args.output_dir / "source-snapshot"
    snapshot.mkdir(exist_ok=True)
    source_hashes = {}
    for relative in ("src/llama-lsa-native.h", "src/llama-lsa-native.cpp", "tests/test-lsa-native.cpp", "tests/test-lsa-native-reference.py"):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
        source_hashes[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    native_test = args.native_test
    binary_hash = None
    if native_test:
        frozen_binary = args.output_dir / "test-lsa-native-frozen"
        shutil.copy2(native_test, frozen_binary)
        native_test = frozen_binary
        binary_hash = hashlib.sha256(native_test.read_bytes()).hexdigest()
    generator = torch.Generator().manual_seed(20261103)
    profiles = {}
    random = torch.nn.functional.normalize(torch.randn(1537, 64, generator=generator), dim=-1)
    profiles["random"] = random
    identical = torch.zeros_like(random)
    identical[:, 0] = 1
    profiles["identical"] = identical
    paired = torch.zeros_like(random)
    near_tie = torch.zeros_like(random)
    for position in range(len(random)):
        pool = position % 512 // 4
        paired[position, pool // 2] = 1
        near_tie[position, pool // 2] = 1
        near_tie[position, (pool // 2 + 1) % 64] = (1 if pool % 2 else -1) / 256
    profiles["paired-duplicates"] = paired
    profiles["near-ties"] = near_tie
    final_round = torch.zeros_like(random)
    for position in range(len(random)):
        pool = position % 512 // 4
        final_round[position, pool - 124 if pool >= 124 else pool // 2] = 1
    profiles["final-round-ties"] = final_round
    queries = torch.nn.functional.normalize(torch.randn(5, 64, generator=generator), dim=-1)
    positions = torch.tensor([0, 511, 512, 1024, 1536])
    records = []

    def record(path, mode):
        item = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mode": mode}
        if native_test:
            result = subprocess.run([str(native_test), mode, str(path)], text=True, capture_output=True, check=False)
            item.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            print(json.dumps(item), flush=True)
        records.append(item)

    for name, keys in profiles.items():
        path = args.output_dir / f"{name}.lsafix"
        write_cluster_fixture(path, oracle, keys, queries, positions)
        record(path, "--reference-fixture")
    projection = args.output_dir / "random27b.lsapro"
    iq, ik, diagnostic = write_projection_fixture(projection, oracle, generator)
    record(projection, "--projection-fixture")
    path = args.output_dir / "projected27b.lsafix"
    write_cluster_fixture(path, oracle, ik, iq, torch.tensor([0, 4096, 5120]))
    record(path, "--reference-fixture")
    diagnostics = {"random27b": diagnostic}
    for kind, map_path in (("pca", args.pca_map), ("distilled", args.distilled_map)):
        if not map_path:
            continue
        path = args.output_dir / f"released27b-{kind}-layer3.lsapro"
        _, _, diagnostic = write_projection_fixture(path, oracle, generator, map_path, kind)
        diagnostics[kind] = diagnostic
        record(path, "--projection-fixture")
    report = {
        "oracle": str(args.oracle), "oracle_sha256": ORACLE_SHA256,
        "torch_version": torch.__version__, "seed": 20261103,
        "scope": "synthetic CPU unit fixtures, no model inference or performance claim",
        "records": records,
        "projection_diagnostics": diagnostics,
        "source_snapshot": str(snapshot), "source_sha256": source_hashes,
        "native_test": str(native_test) if native_test else None, "native_test_sha256": binary_hash,
    }
    (args.output_dir / "fixture-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(any(item.get("returncode", 0) != 0 for item in records))


if __name__ == "__main__":
    raise SystemExit(main())
