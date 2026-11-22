#!/usr/bin/python3
# CPU probe of llama-server (served 27B + DFlash2 argv, -ngl 0): one fresh server per arm, greedy and temp-1 requests,
# witness lines from the log. Arms: OFF (knobs unset) and ON (LLAMA_DEVICE_KQ_MASK=1 LLAMA_DEVICE_LATTICE=1).
# Terminates only the captured pid and verifies it with ps -p.
import json, os, subprocess, sys, time, urllib.request
W = '/mnt/raid0/llm/llama-worktrees/gpuround2f-20261121'
B = W + '/build-gpu/bin'
PORT = 18931
ARGV = ['/usr/bin/taskset', '-c', '72-87', B + '/llama-server', '-m', '/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf',
        '--host', '127.0.0.1', '--port', str(PORT), '-c', '16384', '-np', '2', '-t', '16', '-tb', '16', '-b', '2048', '-ub', '2048',
        '-ctk', 'f16', '-ctv', 'f16', '-ngl', '0', '-fa', 'on', '-lv', '4',
        '-md', '/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf', '-ngld', '0', '--spec-type', 'draft-dflash', '--spec-draft-n-max', '7']
PROMPT = 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'


def req(temp, seed):
    body = json.dumps({'prompt': PROMPT, 'n_predict': 32, 'temperature': temp, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0,
                       'seed': seed, 'cache_prompt': False, 'ignore_eos': True, 'return_tokens': True}).encode()
    r = urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{PORT}/completion', body, {'Content-Type': 'application/json'}), timeout=1200)
    j = json.loads(r.read())
    return j.get('tokens'), (j.get('timings') or {}).get('draft_n_accepted'), (j.get('timings') or {}).get('draft_n')


def arm(tag, knobs):
    log = f'{W}/build-srv-probe-{tag}.log'
    base = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}  # knobs set in BOTH arms
    env = {'PATH': '/usr/bin:/bin', 'HOME': os.environ.get('HOME', '/tmp'), 'LD_LIBRARY_PATH': B, 'HIP_VISIBLE_DEVICES': '-1', **base, **knobs}
    lf = open(log, 'wb')
    p = subprocess.Popen(ARGV, stdout=lf, stderr=subprocess.STDOUT, env=env)
    out = {'tag': tag}
    try:
        t0 = time.time()
        while time.time() - t0 < 900 and p.poll() is None:
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2).read()
                break
            except Exception:
                time.sleep(2)
        out['greedy'] = req(0.0, 7)
        out['temp1'] = req(1.0, 7)
        # two concurrent greedy requests (np 2): multi-sequence verify batches exercise the output reorder
        import threading
        res = [None, None]
        def w(i):
            res[i] = req(0.0, 11 + i)
        th = [threading.Thread(target=w, args=(i,)) for i in range(2)]
        [t.start() for t in th]; [t.join() for t in th]
        out['conc'] = res
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=30)
        out['dead'] = subprocess.run(['/usr/bin/ps', '-p', str(p.pid)], capture_output=True).returncode != 0
        out['pid'] = p.pid
    txt = open(log, errors='replace').read()
    out['witness'] = [l[-160:] for l in txt.splitlines() if ('kround' in l) or 'ENABLED' in l and ('DEVICE_CTL' in l or 'KV_IDX' in l or 'KV_RING' in l or 'DEVICE_CPOS' in l or 'SCOPY' in l or 'DFEED' in l)]
    out['mismatch'] = 'MISMATCH' in txt; out['ctl'] = [l for l in txt.splitlines() if 'device_hits' in l or 'token_hits' in l or 'llama_kv_cache: LLAMA_DEVICE_' in l or 'LLAMA_DEVICE_SCOPY' in l or 'LLAMA_DEVICE_DFEED' in l or 'DSEL_RNG' in l or 'FIXED_T' in l or 'DFT_KV' in l]
    out['rs'] = sorted(set(l[-120:] for l in txt.splitlines() if 'rs_replay ' in l or 'RS_REP' in l))
    return out


off = arm('OFF', {})
on = arm('ON', json.loads(sys.argv[1]) if len(sys.argv) > 1 else {'LLAMA_DEVICE_KQ_MASK': '1', 'LLAMA_DEVICE_LATTICE': '1'})
for a in (off, on):
    print(a['tag'], 'pid', a['pid'], 'dead', a['dead'], 'witness', a['witness'], 'mismatch', a['mismatch'], a['ctl'])
    print('  rs', a['rs'])
    print('  greedy acc/draft', a['greedy'][1], a['greedy'][2], ' temp1 acc/draft', a['temp1'][1], a['temp1'][2])
print('greedy tokens ON == OFF:', on['greedy'][0] == off['greedy'][0], len(on['greedy'][0] or []))
print('concurrent greedy ON == OFF:', [on['conc'][i][0] == off['conc'][i][0] for i in range(2)], 'acc/draft OFF', [(x[1], x[2]) for x in off['conc']], 'ON', [(x[1], x[2]) for x in on['conc']])
print('temp1 tokens produced ON:', len(on['temp1'][0] or []), 'temp1 tokens ON == OFF:', on['temp1'][0] == off['temp1'][0])
