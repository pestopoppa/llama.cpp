#!/usr/bin/env python3
"""Five exact, uncached 2029-token prefill observations with one decode token."""
import argparse, hashlib, json, math
from pathlib import Path
from urllib.request import Request, urlopen

TOKENS=Path(__file__).with_name("prefill2029-tokens.json")
TOKENS_FILE_SHA256="b39534e20f95b09a1e1df800e5672f9359718dba1a19ab0b89f0dad6a7a397ea"
def post(port,body,timeout):
    q=Request(f"http://127.0.0.1:{port}/completion",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urlopen(q,timeout=timeout) as r:return json.load(r)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--port",type=int,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--timeout",type=int,default=3600);ap.add_argument("--prompt-tokens",type=Path,default=TOKENS);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False); raw=a.prompt_tokens.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=TOKENS_FILE_SHA256:raise RuntimeError("pinned prompt token file drift")
    tokens=json.loads(raw)
    if len(tokens)!=2029 or not all(isinstance(x,int) and not isinstance(x,bool) for x in tokens):raise RuntimeError("invalid pinned 2029-token prompt")
    (a.out/"provenance.json").write_text(json.dumps({"prompt_tokens_path":str(a.prompt_tokens.resolve()),"prompt_tokens_sha256":hashlib.sha256(raw).hexdigest()},indent=2,sort_keys=True)+"\n")
    base={"prompt":tokens,"temperature":0.0,"top_k":1,"seed":42,"cache_prompt":False,"stream":False,"ignore_eos":True,"return_tokens":True,"n_predict":1}
    (a.out/"prompt-tokens.json").write_bytes(raw);(a.out/"request-base.json").write_text(json.dumps(base,indent=2,sort_keys=True)+"\n")
    rows=[]
    for rep in range(1,6):
        doc=post(a.port,base,a.timeout);(a.out/f"response-{rep}.json").write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
        t=doc.get("timings") or {}
        if t.get("prompt_n")!=2029 or t.get("cache_n")!=0:raise RuntimeError(f"rep{rep}: expected prompt_n=2029 cache_n=0, got {t}")
        rate=t.get("prompt_per_second")
        if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<=0:raise RuntimeError(f"rep{rep}: invalid prompt rate")
        outt=doc.get("tokens")
        if not isinstance(outt,list) or not all(isinstance(x,int) and not isinstance(x,bool) for x in outt):raise RuntimeError(f"rep{rep}: missing exact output tokens")
        rows.append({"kind":"prefill2029","rep":rep,"prompt_n":2029,"cache_n":0,"prompt_per_second":rate,"prompt_ms":t.get("prompt_ms"),"predicted_n":t.get("predicted_n"),"predicted_per_second":t.get("predicted_per_second"),"tokens":outt,"canonical_response_sha256":hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"draft_n":t.get("draft_n",0),"draft_n_accepted":t.get("draft_n_accepted",0)})
        (a.out/"requests.json").write_text(json.dumps(rows,indent=2,sort_keys=True)+"\n")
        print(json.dumps({k:v for k,v in rows[-1].items() if k!="tokens"}),flush=True)
if __name__=="__main__":raise SystemExit(main())
