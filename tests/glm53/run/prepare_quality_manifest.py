#!/usr/bin/env python3
"""Resolve a bounded GLM quality screen through the existing benchmark adapters."""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path

QIDS = ("gsm8k_00739", "gsm8k_00115", "gsm8k_00041",
        "simpleqa_general_01814", "hellaswag_06230")
CANONICAL = Path(__file__).with_name("canonical24-prompts.json")

def load_lookup(bench):
    path = bench / "lookup_question.py"
    sys.path.insert(0, str(bench))
    spec = importlib.util.spec_from_file_location("glm53_lookup_question", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load lookup adapter: {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.find_question_in_adapter, module.parse_question_id

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--research-root",type=Path,required=True);ap.add_argument("--canonical",type=Path,default=CANONICAL); a=ap.parse_args()
    bench=a.research_root/"scripts/benchmark";find_question_in_adapter,parse_question_id=load_lookup(bench)
    canonical_rows=json.loads(a.canonical.read_text())
    canonical={x["id"]:(i,x) for i,x in enumerate(canonical_rows)}; rows=[]
    for qid in QIDS:
        suite,sub,suffix=parse_question_id(qid)
        q=find_question_in_adapter(suite,sub,suffix,qid)
        if not q: raise RuntimeError(f"adapter did not resolve {qid}")
        ordinal,crow=canonical[qid]
        if crow["prompt"] != q["prompt"]: raise RuntimeError(f"canonical prompt drift for {qid}")
        row={k:q.get(k) for k in ("id","suite","tier","prompt","expected","scoring_method","scoring_config")}
        row["canonical_source_ordinal"]=ordinal
        row["canonical_prompt_sha256"]=hashlib.sha256(crow["prompt"].encode()).hexdigest()
        row["adapter_prompt_sha256"]=hashlib.sha256(q["prompt"].encode()).hexdigest()
        row["canonical_prompt_equal_adapter"]=True
        rows.append(row)
    doc={"schema":"glm53-bounded-quality-manifest-v1","selection":"fixed canonical production-mix subset: 3 GSM8K, 1 SimpleQA, 1 HellaSwag",
         "rows":rows,"provenance":{"canonical_prompts":{"path":str(a.canonical.resolve()),"sha256":sha(a.canonical)},
         "lookup_question":{"path":str(bench/"lookup_question.py"),"sha256":sha(bench/"lookup_question.py")},
         "dataset_adapters":{"path":str(bench/"dataset_adapters.py"),"sha256":sha(bench/"dataset_adapters.py")},
         "debug_scorer":{"path":str(bench/"debug_scorer.py"),"sha256":sha(bench/"debug_scorer.py")}}}
    a.out.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"out":str(a.out),"rows":[(r["id"],r["expected"],r["scoring_method"]) for r in rows]}))
if __name__=="__main__": raise SystemExit(main())
