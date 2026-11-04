"""Explicit one-cell reference recall diagnostic. Never selects a cohort row."""
import argparse
import gzip
import hashlib
import json
import math
import os
import signal
import threading
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from mask_worker import verify_recall_binding, verify_recall_request


ITEM = 'dev-code-00'
TOKEN_SHA = '9a6948657870e895f48aabc0f3acc15fdf273ccf66f72345f0a482191dc6a480'
MAP_SHA = 'baaa28cb00b4378b3d33e02dc971e2781482d5c4d45e29e9140143bdf7832e65'
MODEL_SHA = 'a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348'
LAYERS = set(range(3, 64, 4))
MARKER = b'LSA recall: duplicate layer aggregate'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_model_receipt(path, model):
    receipt = json.loads(Path(path).read_text())
    stat = Path(model).stat()
    if receipt.get('sha256') != MODEL_SHA or receipt.get('dev') != stat.st_dev or \
            receipt.get('ino') != stat.st_ino or receipt.get('size') != stat.st_size:
        raise ValueError('model full-hash receipt identity mismatch')
    with Path(model).open('rb') as stream:
        head = stream.read(65536)
        stream.seek(max(0, stat.st_size - 65536))
        tail = stream.read(65536)
    if hashlib.sha256(head).hexdigest() != receipt.get('head_sha256') or \
            hashlib.sha256(tail).hexdigest() != receipt.get('tail_sha256'):
        raise ValueError('model edge bytes differ from full-hash receipt')
    return receipt


def grant():
    required = ('FILLER_ID', 'FILLER_NEEDGPU', 'FILLER_STOPFILE', 'FILLER_DEADLINE', 'FILLER_END', 'FILLER_PIDFILE')
    if any(not os.environ.get(name) for name in required) or os.environ['FILLER_NEEDGPU'] != '1':
        raise ValueError('owned GPU filler grant required')
    stopfile = Path(os.environ['FILLER_STOPFILE'])
    deadline = float(os.environ['FILLER_DEADLINE'])
    pidfile = Path(os.environ['FILLER_PIDFILE'])
    if not math.isfinite(deadline) or int(os.environ['FILLER_END']) <= deadline or deadline <= time.time() or stopfile.exists():
        raise ValueError('filler stopped or deadline passed')
    return stopfile, deadline, pidfile


def check_grant(stopfile, deadline):
    if stopfile.exists() or time.time() >= deadline:
        raise RuntimeError('filler stop or deadline reached')


def check_log_bound(path):
    if Path(path).stat().st_size > 2 * 1024 * 1024:
        raise ValueError('reference server log exceeds evidence bound')


def source_commit():
    return subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]),
                                    'rev-parse', 'HEAD'], text=True).strip()


def verify_build(proof_path, binary, library):
    proof = json.loads(Path(proof_path).read_text())
    if proof.get('schema') != 'lsa-recall-build-proof-v1' or proof.get('source_commit') != source_commit():
        raise ValueError('recall source commit is absent from build proof')
    if sha(binary) != proof.get('binary_sha256') or sha(library) != proof.get('libllama_sha256'):
        raise ValueError('recall binary/library digest differs from build proof')
    if MARKER not in Path(library).read_bytes():
        raise ValueError('recall callback marker absent from libllama')
    dsos = proof.get('dsos')
    if not isinstance(dsos, dict) or not dsos or str(Path(library).resolve()) not in dsos:
        raise ValueError('recall build proof lacks mapped-library inventory')
    for path, digest in dsos.items():
        if not Path(path).is_absolute() or sha(path) != digest:
            raise ValueError('recall build library digest mismatch')
    return proof


def verify_mapped_libraries(pid, proof, rehash=True):
    allowed = proof['dsos']
    mapped = {str(Path(line.split()[-1]).resolve())
              for line in Path(f'/proc/{pid}/maps').read_text().splitlines()
              if line.split()[-1].startswith('/') and
              any(name in line for name in ('libllama', 'libggml'))}
    if not mapped or not mapped.issubset(allowed):
        raise ValueError('reference process mapped an unproved llama/ggml library')
    if rehash:
        for path in mapped:
            if sha(path) != allowed[path]:
                raise ValueError('mapped reference library digest changed')
    return sorted(mapped)


def verify_owned_listener(pid, port):
    inodes = set()
    for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
        fields = line.split()
        if fields[1].endswith(f':{port:04X}') and fields[3] == '0A':
            inodes.add(fields[9])
    if not inodes:
        raise ValueError('reference port has no listening socket')
    for fd in Path(f'/proc/{pid}/fd').iterdir():
        try:
            link = os.readlink(fd)
        except OSError:
            continue
        if link.startswith('socket:[') and link[8:-1] in inodes:
            return
    raise ValueError('reference listener is not owned by spawned server PID')


def validate_aggregates(path):
    raw = Path(path).read_bytes()
    if len(raw) > 64 * 1024 or not raw.endswith(b'\n'):
        raise ValueError('recall aggregate missing or oversized')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 16 or {row.get('layer') for row in rows} != LAYERS:
        raise ValueError('recall does not have exactly sixteen global layers')
    required = {'item': ITEM, 'split': 'dev', 'nominal': 32768, 'token_sha256': TOKEN_SHA,
                'query': 32767, 'mode': 'pca', 'released_pt_map_sha256': MAP_SHA,
                'top_k': 4096, 'method': 'reconstructed_dense_dequantized_k_double_dot'}
    for row in rows:
        if any(row.get(key) != value for key, value in required.items()):
            raise ValueError('recall aggregate binding mismatch')
        if not (1 <= row.get('budget', 0) <= 4612 and 0 <= row.get('top_hits', -1) <= 4096):
            raise ValueError('recall budget or top hits invalid')
        for key in ('top_recall', 'selected_mass', 'scale', 'softcap'):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('recall aggregate nonfinite value')
        if not (0 <= row['top_recall'] <= 1 and 0 <= row['selected_mass'] <= 1.000001 and
                abs(row['top_recall'] - row['top_hits'] / 4096) < 1e-12 and row['scale'] > 0 and row['softcap'] >= 0):
            raise ValueError('recall aggregate range mismatch')
    return rows


def fetch_bounded(url, body=None, timeout=1, limit=1024 * 1024):
    req = urllib.request.Request(url, data=body,
                                 headers={'Content-Type': 'application/json'} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError('server response exceeds declared bound')
        return data


def sample_gpu(pid):
    base = Path(f'/sys/class/kfd/kfd/proc/{pid}')
    vram = {}
    if base.is_dir():
        for path in base.glob('vram_*'):
            try:
                vram[path.name] = int(path.read_text().strip())
            except (OSError, ValueError):
                vram[path.name] = None
    return {'monotonic_ns': time.monotonic_ns(), 'pid': pid, 'kfd_present': base.exists(),
            'vram_bytes': vram}


def monitored_request(url, body, server, proof, stopfile, deadline, log_path):
    result = {}
    def worker():
        try:
            result['body'] = fetch_bounded(url, body, timeout=360, limit=1024 * 1024)
        except Exception as exc:
            result['error'] = exc
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    samples = []
    while thread.is_alive():
        check_grant(stopfile, deadline)
        check_log_bound(log_path)
        if server.poll() is not None:
            raise RuntimeError('reference server exited during request')
        mapped = verify_mapped_libraries(server.pid, proof, rehash=False)
        sample = sample_gpu(server.pid)
        sample['mapped_dsos'] = mapped
        samples.append(sample)
        thread.join(timeout=0.5)
    if 'error' in result:
        raise result['error']
    if not any(s['kfd_present'] and any(type(v) is int and v > 0 for v in s['vram_bytes'].values()) for s in samples):
        raise ValueError('no in-request own-PID KFD/VRAM witness')
    return result['body'], samples


def owned_group_members(group):
    members = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                stat = (path / 'stat').read_text()
                state = stat[stat.rfind(')') + 2]
                if state != 'Z' and os.getpgid(int(path.name)) == group:
                    members.append(int(path.name))
            except (OSError, ValueError):
                pass
    return members


def stop_owned_group(server, group):
    if server is None:
        return
    members = owned_group_members(group)
    if members:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    members = owned_group_members(group)
    if members:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    remaining = owned_group_members(group)
    if remaining:
        raise RuntimeError(f'owned process group still alive: {remaining}')


def run(args):
    if not args.execute:
        raise ValueError('explicit --execute required')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    aggregate = output / 'recall-aggregates.jsonl'
    stopfile, deadline, pidfile = grant()
    proof = verify_build(args.build_proof, args.binary, args.libllama)
    verify_model_receipt(args.model_receipt, args.model)
    env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_LSA_') and
           k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'GGML_BACKEND_PATH')}
    env.update({
        'LLAMA_LSA_SCREEN': 'pca', 'LLAMA_LSA_WORKER': str(Path(__file__).with_name('mask_worker.py').resolve()),
        'LLAMA_LSA_PYTHON': sys.executable, 'LLAMA_LSA_MAP': str(Path(args.map).resolve()),
        'LLAMA_LSA_RECALL_OUT': str(aggregate), 'LLAMA_LSA_RECALL_ITEM': ITEM,
        'LLAMA_LSA_RECALL_SPLIT': 'dev', 'LLAMA_LSA_RECALL_NOMINAL': '32768',
        'LLAMA_LSA_RECALL_POSITIONS': '32767', 'LLAMA_LSA_RECALL_TOKEN_SHA256': TOKEN_SHA,
        'LLAMA_LSA_RECALL_MAP_SHA256': MAP_SHA,
        'LLAMA_LSA_RECALL_MANIFEST': str(Path(args.manifest).resolve()),
        'LLAMA_LSA_RECALL_TOKEN_FILE': str(Path(args.tokens).resolve()),
        'LD_LIBRARY_PATH': str(Path(args.libllama).resolve().parent) + ':/opt/rocm/lib',
    })
    # The worker validates the manifest and released map bytes before it accepts a graph payload.
    # Run the same check before starting a server, without relying on the worker's env copy.
    old = {key: os.environ.get(key) for key in env if key.startswith('LLAMA_LSA_')}
    try:
        os.environ.update({key: value for key, value in env.items() if key.startswith('LLAMA_LSA_')})
        verify_recall_binding('pca', args.map)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    argv = [str(Path(args.binary).resolve()), '-m', str(Path(args.model).resolve()),
            '--host', '127.0.0.1', '--port', str(args.port), '-np', '1', '-c', '33792',
            '-ub', '2048', '--flash-attn', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0',
            '-ngl', 'all', '--device', 'ROCm0', '--spec-type', 'none', '--fit', 'off', '--no-mmap']
    server = None
    group = None
    started = time.monotonic()
    request_sha = None
    try:
        with (output / 'server.log').open('wb') as log:
            check_grant(stopfile, deadline)
            server = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            group = server.pid
            with pidfile.open('a') as pid_stream:
                pid_stream.write(str(server.pid) + '\n')
            until = min(time.monotonic() + 180, time.monotonic() + max(0, deadline - time.time()))
            while time.monotonic() < until:
                check_grant(stopfile, deadline)
                check_log_bound(output / 'server.log')
                if server.poll() is not None:
                    raise RuntimeError('reference server exited before readiness')
                try:
                    fetch_bounded(f'http://127.0.0.1:{args.port}/health', timeout=1, limit=4096)
                    break
                except Exception:
                    time.sleep(0.5)
            else:
                raise TimeoutError('reference server readiness deadline')
            verify_owned_listener(server.pid, args.port)
            verify_mapped_libraries(server.pid, proof)
            # Re-read the frozen bytes and compare the actual list immediately before serialization.
            ids = json.loads(gzip.decompress(Path(args.tokens).read_bytes()))
            verify_recall_request(ids, args.tokens)
            body = json.dumps({'prompt': ids, 'n_predict': 1, 'cache_prompt': False,
                               'seed': 42, 'temperature': 0.0, 'top_k': 1,
                               'stream': False}, separators=(',', ':')).encode()
            request_sha = hashlib.sha256(body).hexdigest()
            response, samples = monitored_request(f'http://127.0.0.1:{args.port}/completion',
                                                   body, server, proof, stopfile, deadline,
                                                   output / 'server.log')
            (output / 'response.json').write_bytes(response)
            with (output / 'samples.jsonl').open('w') as samples_out:
                for sample in samples:
                    samples_out.write(json.dumps(sample, sort_keys=True) + '\n')
        rows = validate_aggregates(aggregate)
        verify_mapped_libraries(server.pid, proof)
        result = {'schema': 'lsa-reference-recall-diagnostic-v1', 'status': 'capture-valid',
                  'item': ITEM, 'nominal': 32768, 'query': 32767, 'mode': 'pca',
                  'source_commit': proof['source_commit'], 'binary_sha256': proof['binary_sha256'],
                  'libllama_sha256': proof['libllama_sha256'], 'model_sha256': MODEL_SHA,
                  'released_pt_map_sha256': MAP_SHA, 'token_sha256': TOKEN_SHA,
                  'request_sha256': request_sha, 'aggregate_rows': len(rows),
                  'aggregate_sha256': sha(aggregate), 'samples_sha256': sha(output / 'samples.jsonl'),
                  'sample_count': len(samples), 'elapsed_s': time.monotonic() - started,
                  'scope': 'one reconstructed dense attention diagnostic; no GPU kernel parity or quality claim'}
        (output / 'RESULT.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    except Exception as exc:
        (output / 'FAILURE.json').write_text(json.dumps({'schema': 'lsa-reference-recall-diagnostic-v1',
                                                          'status': 'invalid', 'reason': str(exc)[:512],
                                                          'request_sha256': request_sha}, sort_keys=True) + '\n')
        raise
    finally:
        stop_owned_group(server, group)
        log_path = output / 'server.log'
        if log_path.is_file() and log_path.stat().st_size > 2 * 1024 * 1024:
            with log_path.open('rb+') as stream:
                stream.truncate(2 * 1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    for name in ('build-proof', 'binary', 'libllama', 'model', 'model-receipt', 'map',
                 'manifest', 'tokens', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        output = Path(args.output)
        if output.is_dir() and not (output / 'FAILURE.json').exists():
            (output / 'FAILURE.json').write_text(json.dumps({
                'schema': 'lsa-reference-recall-diagnostic-v1', 'status': 'invalid',
                'reason': str(exc)[:512]}, sort_keys=True) + '\n')
        raise


if __name__ == '__main__':
    main()
