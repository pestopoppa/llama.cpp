#!/usr/bin/env python3
"""Run the bounded adapter-backed GLM quality screen against one server."""
import argparse, hashlib, importlib.util, json, math, sys, time
from pathlib import Path
from urllib.request import Request, urlopen

def load_scorer(root):
    path = root / "debug_scorer.py"
    spec = importlib.util.spec_from_file_location("glm53_debug_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scorer: {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.score_answer

def post(port,body,timeout):
    q=Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urlopen(q,timeout=timeout) as r:return json.load(r)
def sha_text(value): return hashlib.sha256(value.encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--port",type=int,required=True);ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--manifest",type=Path,default=Path(__file__).with_name("quality-manifest.json"));ap.add_argument("--scorer-root",type=Path,required=True);ap.add_argument("--timeout",type=int,default=1800);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False); manifest=json.loads(a.manifest.read_text()); rows=[]; score_answer=load_scorer(a.scorer_root)
    (a.out/"manifest.json").write_bytes(a.manifest.read_bytes())
    scorer_path=a.scorer_root/"debug_scorer.py"
    (a.out/"provenance.json").write_text(json.dumps({"manifest_path":str(a.manifest.resolve()),"manifest_sha256":hashlib.sha256(a.manifest.read_bytes()).hexdigest(),"scorer_path":str(scorer_path.resolve()),"scorer_sha256":hashlib.sha256(scorer_path.read_bytes()).hexdigest()},indent=2,sort_keys=True)+"\n")
    for i,item in enumerate(manifest["rows"]):
        req={"messages":[{"role":"user","content":item["prompt"]}],"max_tokens":256,"temperature":0.0,
             "cache_prompt":False,"reasoning_budget_tokens":0,"return_tokens":True,"verbose":True}
        start=time.time(); doc=post(a.port,req,a.timeout); wall=time.time()-start
        (a.out/f"request-{i:02d}.json").write_text(json.dumps(req,indent=2,sort_keys=True)+"\n")
        (a.out/f"response-{i:02d}.json").write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
        ch=doc["choices"][0]; msg=ch.get("message") or {}; content=msg.get("content") or ""; reasoning=msg.get("reasoning_content") or ""
        verbose=doc.get("__verbose") or {}; tokens=verbose.get("tokens"); tim=doc.get("timings") or {}
        if not isinstance(tokens,list) or not all(isinstance(x,int) and not isinstance(x,bool) for x in tokens): raise RuntimeError(f"{item['id']}: missing exact token IDs")
        if len(tokens)!=tim.get("predicted_n"): raise RuntimeError(f"{item['id']}: token/timing count mismatch")
        rate=tim.get("predicted_per_second")
        if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<=0: raise RuntimeError(f"{item['id']}: invalid predicted rate")
        cfg=item.get("scoring_config") or {}
        row={"ordinal":i+1,"id":item["id"],"expected":item["expected"],"scoring_method":item["scoring_method"],
             "scoring_config":cfg,"passed_content":score_answer(content,item["expected"],item["scoring_method"],cfg),
             "passed_combined":score_answer(reasoning+"\n"+content,item["expected"],item["scoring_method"],cfg),
             "tokens":tokens,"tokens_sha256":hashlib.sha256(json.dumps(tokens,separators=(",",":")).encode()).hexdigest(),
             "content":content,"content_sha256":sha_text(content),"reasoning_content":reasoning,"reasoning_content_sha256":sha_text(reasoning),
             "reasoning_budget_tokens":0,"reasoning_chars":len(reasoning),"content_chars":len(content),
             "rendered_prompt_tail":str(verbose.get("prompt",""))[-80:],"finish_reason":ch.get("finish_reason"),
             "predicted_n":len(tokens),"predicted_per_second":rate,"wall_seconds":wall}
        rows.append(row)
        payload=json.dumps(rows,indent=2,sort_keys=True)+"\n"
        (a.out/"results.json").write_text(payload);(a.out/"requests.json").write_text(payload)
        print(json.dumps({k:v for k,v in row.items() if k not in ("tokens","content","reasoning_content")}),flush=True)
    print(json.dumps({"completed":len(rows),"passed_content":sum(r["passed_content"] for r in rows),"passed_combined":sum(r["passed_combined"] for r in rows)}))
if __name__=="__main__": raise SystemExit(main())
