#!/usr/bin/env python3
"""Two distinct deterministic 512-token prompts for plain/MTP parity."""
import argparse,hashlib,json,math
from pathlib import Path
from urllib.request import Request,urlopen

PROMPTS=Path(__file__).with_name("canonical24-prompts.json")
QIDS=("gsm8k_00739","simpleqa_general_01814")
def post(port,route,body,timeout):
    q=Request(f"http://127.0.0.1:{port}{route}",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urlopen(q,timeout=timeout) as r:return json.load(r)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--port",type=int,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--timeout",type=int,default=3600);ap.add_argument("--prompts",type=Path,default=PROMPTS);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);raw=a.prompts.read_bytes();pool={x["id"]:x["prompt"] for x in json.loads(raw)};rows=[]
    (a.out/"provenance.json").write_text(json.dumps({"prompts_path":str(a.prompts.resolve()),"prompts_sha256":hashlib.sha256(raw).hexdigest()},indent=2,sort_keys=True)+"\n")
    for i,qid in enumerate(QIDS):
        fmt={"messages":[{"role":"user","content":pool[qid]}],"add_generation_prompt":True,"chat_template_kwargs":{"enable_thinking":False}}
        rendered=post(a.port,"/apply-template",fmt,a.timeout).get("prompt")
        if not isinstance(rendered,str) or not rendered:raise RuntimeError(f"{qid}: missing rendered prompt")
        req={"prompt":rendered,"temperature":0.0,"top_k":1,"seed":42,"cache_prompt":False,"stream":False,"ignore_eos":True,"return_tokens":True,"n_predict":512}
        doc=post(a.port,"/completion",req,a.timeout);tokens=doc.get("tokens");tim=doc.get("timings") or {}
        if not isinstance(tokens,list) or len(tokens)!=512 or not all(isinstance(x,int) and not isinstance(x,bool) for x in tokens):raise RuntimeError(f"{qid}: expected 512 exact token IDs")
        rate=tim.get("predicted_per_second")
        if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<=0:raise RuntimeError(f"{qid}: invalid predicted rate")
        (a.out/f"apply-template-{i}.json").write_text(json.dumps(fmt,indent=2,sort_keys=True)+"\n")
        (a.out/f"rendered-{i}.txt").write_text(rendered);(a.out/f"request-{i}.json").write_text(json.dumps(req,indent=2,sort_keys=True)+"\n");(a.out/f"response-{i}.json").write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
        rows.append({"kind":"multiprompt512","ordinal":i+1,"id":qid,"predicted_n":512,"predicted_per_second":rate,"prompt_n":tim.get("prompt_n"),"cache_n":tim.get("cache_n"),"draft_n":tim.get("draft_n",0),"draft_n_accepted":tim.get("draft_n_accepted",0),"tokens":tokens,"tokens_sha256":hashlib.sha256(json.dumps(tokens,separators=(",",":")).encode()).hexdigest(),"rendered_prompt_sha256":hashlib.sha256(rendered.encode()).hexdigest(),"canonical_response_sha256":hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(",",":")).encode()).hexdigest()})
        (a.out/"requests.json").write_text(json.dumps(rows,indent=2,sort_keys=True)+"\n");print(json.dumps({k:v for k,v in rows[-1].items() if k!="tokens"}),flush=True)
if __name__=="__main__":raise SystemExit(main())
