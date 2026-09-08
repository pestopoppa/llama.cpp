#!/usr/bin/env python3
"""Profile one long prefill and one cache-reused decode request."""
import argparse, hashlib, json, os, signal, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

def now(): return datetime.now(timezone.utc).isoformat()
def dump(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def post(port,route,body,timeout):
    q=Request(f"http://127.0.0.1:{port}{route}",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urlopen(q,timeout=timeout) as r:return json.load(r)
def stop(p):
    if p.poll() is None:
        p.send_signal(signal.SIGINT)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
def render_tokens(port,timeout,target):
    unit="A CPU cache hierarchy moves frequently used data through registers, L1, L2, L3, and main memory. "
    lo,hi=1,512; best=None
    while lo<=hi:
        n=(lo+hi)//2; payload={"messages":[{"role":"user","content":unit*n}],"add_generation_prompt":True}
        rendered=post(port,"/apply-template",payload,timeout)["prompt"]
        tokens=post(port,"/tokenize",{"content":rendered},timeout)["tokens"]
        if len(tokens)<=target:best=(payload,rendered,tokens);lo=n+1
        else:hi=n-1
    if best is None:raise RuntimeError("could not construct bounded long prompt")
    return best
def capture(perf,pid,out,port,body,timeout):
    out.mkdir(); rec_err=(out/"perf-record.stderr").open("w"); stat_err=(out/"perf-stat.stderr").open("w")
    rec=stat=None
    try:
        rec=subprocess.Popen([perf,"record","-F","99","-e","cycles:u","--call-graph","dwarf,4096","-p",str(pid),"-o",str(out/"perf.data")],stdout=subprocess.DEVNULL,stderr=rec_err)
        stat=subprocess.Popen([perf,"stat","-x,","-e","instructions:u,cache-references:u,cache-misses:u,task-clock","-p",str(pid),"-o",str(out/"perf-stat.csv")],stdout=subprocess.DEVNULL,stderr=stat_err)
        started=now(); time.sleep(.5)
        if rec.poll() is not None or stat.poll() is not None:raise RuntimeError(f"profiler exited before request record={rec.returncode} stat={stat.returncode}")
        request_start=now(); response=post(port,"/completion",body,timeout); request_end=now()
    finally:
        if stat is not None:stop(stat)
        if rec is not None:stop(rec)
        rec_err.close();stat_err.close()
    dump(out/"request.json",body);dump(out/"response.json",response)
    dump(out/"window.json",{"profiler_started_utc":started,"request_start_utc":request_start,"request_end_utc":request_end,"profiler_stopped_utc":now(),"server_pid":pid,"perf_record_pid":rec.pid,"perf_stat_pid":stat.pid,"perf_record_rc":rec.returncode,"perf_stat_rc":stat.returncode})
    with (out/"perf-symbols.txt").open("w") as f: subprocess.run([perf,"report","-i",str(out/"perf.data"),"--stdio","--no-children","--percent-limit","0.1"],stdout=f,stderr=subprocess.STDOUT,timeout=120,check=True)
    with (out/"perf-callgraph.txt").open("w") as f: subprocess.run([perf,"report","-i",str(out/"perf.data"),"--stdio","--percent-limit","0.5"],stdout=f,stderr=subprocess.STDOUT,timeout=120,check=True)
    return response
def main():
    a=argparse.ArgumentParser();a.add_argument("--port",type=int,required=True);a.add_argument("--out",type=Path,required=True);a.add_argument("--timeout",type=int,default=3600);a.add_argument("--perf",default="/usr/bin/perf");x=a.parse_args();x.out.mkdir(parents=True,exist_ok=False)
    pid=int(os.environ["GLM_SERVER_PID"]); payload,rendered,tokens=render_tokens(x.port,x.timeout,2048)
    dump(x.out/"apply-template-request.json",payload);(x.out/"rendered-prompt.txt").write_text(rendered);dump(x.out/"prompt-tokens.json",tokens)
    base={"prompt":tokens,"temperature":0.0,"top_k":1,"seed":42,"cache_prompt":True,"stream":False,"ignore_eos":True,"return_tokens":True}
    pre=capture(x.perf,pid,x.out/"prefill",x.port,{**base,"n_predict":1},x.timeout)
    dec=capture(x.perf,pid,x.out/"decode",x.port,{**base,"n_predict":128},x.timeout)
    dump(x.out/"requests.json",[{"kind":"prefill-profile","predicted_n":pre["timings"]["predicted_n"],"predicted_per_second":pre["timings"]["predicted_per_second"],"draft_n":0,"draft_n_accepted":0,"canonical_response_sha256":hashlib.sha256(json.dumps(pre,sort_keys=True,separators=(',',':')).encode()).hexdigest()},{"kind":"decode-profile","predicted_n":dec["timings"]["predicted_n"],"predicted_per_second":dec["timings"]["predicted_per_second"],"draft_n":0,"draft_n_accepted":0,"canonical_response_sha256":hashlib.sha256(json.dumps(dec,sort_keys=True,separators=(',',':')).encode()).hexdigest()}])
    print(json.dumps({"server_pid":pid,"prompt_tokens":len(tokens),"prefill_timings":pre["timings"],"decode_timings":dec["timings"]},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
