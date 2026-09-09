#!/usr/bin/env python3
"""Capture the canonical 24-prompt production mix with exact generated IDs."""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from response_classifier import classify

PROMPTS = Path(__file__).with_name("canonical24-prompts.json")

def post(port,body,timeout):
 q=Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urlopen(q,timeout=timeout) as r:return json.load(r)
def body(text):return {'messages':[{'role':'user','content':text}],'max_tokens':200,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False},'temperature':0.0,'return_tokens':True,'verbose':True}
def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--timeout',type=int,default=1800);p.add_argument('--prompts',type=Path,default=PROMPTS);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 prompts=json.loads(a.prompts.read_text()); rows=[]
 provenance={'prompts_path':str(a.prompts.resolve()),'prompts_sha256':hashlib.sha256(a.prompts.read_bytes()).hexdigest(),'client_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'endpoint':'/v1/chat/completions','max_tokens':200,'temperature':0.0,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False}}
 (a.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 warm=post(a.port,body('hi'),a.timeout);(a.out/'warmup-response.json').write_text(json.dumps(warm,indent=2)+'\n')
 for i,item in enumerate(prompts):
  req=body(item['prompt']); t=time.time(); doc=post(a.port,req,a.timeout); wall=time.time()-t
  (a.out/f'request-{i:02d}.json').write_text(json.dumps(req,indent=2,sort_keys=True)+'\n');(a.out/f'response-{i:02d}.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
  ch=(doc.get('choices')or[{}])[0]; content=(ch.get('message')or{}).get('content','')or''; verbose=doc.get('__verbose')or{}; tokens=verbose.get('tokens')
  if not isinstance(tokens,list) or not all(isinstance(x,int) and not isinstance(x,bool) for x in tokens):raise RuntimeError(f'{item["id"]}: server did not return exact token IDs in __verbose.tokens')
  tim=doc.get('timings')or{}; rate=tim.get('predicted_per_second')
  if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate <= 0: raise RuntimeError(f'{item["id"]}: missing valid timings.predicted_per_second')
  c=classify({'tokens':tokens,'stop_type':ch.get('finish_reason'),'content':content}); rendered=str(verbose.get('prompt',''))
  row={'kind':'measure','ordinal':i+1,'id':item['id'],'class':item['class'],'predicted_n':len(tokens),'predicted_per_second':rate,'prompt_n':tim.get('prompt_n'),'prompt_per_second':tim.get('prompt_per_second'),'draft_n':tim.get('draft_n',0),'draft_n_accepted':tim.get('draft_n_accepted',0),'finish_reason':ch.get('finish_reason'),'canonical_response_sha256':hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'tokens':tokens,'content_sha256':hashlib.sha256(content.encode()).hexdigest(),'verdict':c.get('cls'),'n_uniq':c.get('uniq'),'words':c.get('words'),'rendered_prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'rendered_has_open_think_tail':rendered.rstrip().endswith('<think>'),'reasoning_content_present':bool((ch.get('message')or{}).get('reasoning_content'))}
  rows.append(row);(a.out/'requests.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in row.items() if k not in ('tokens',)}),flush=True)
 print(json.dumps({'completed':len(rows),'token_weighted_tps':sum(r['predicted_n'] for r in rows if r['predicted_n']>=16)/sum(r['predicted_n']/r['predicted_per_second'] for r in rows if r['predicted_n']>=16)},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
