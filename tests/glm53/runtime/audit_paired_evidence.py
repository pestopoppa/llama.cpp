#!/usr/bin/env python3
"""Fail-closed categorical audit of paired GLM5Next plain/native-MTP arms."""
from __future__ import annotations
import argparse, hashlib, json, math, re, statistics, sys
from pathlib import Path

EVENT_RE = re.compile(r"\btask\s+(\d+)\s+\|\s+accepted\s+(\d+)\s*/\s*(\d+)\s+draft tokens(?: \(restore checkpoint\))?\s*$")
LAUNCH_RE = re.compile(r"\btask\s+(\d+)\s+\|\s+processing task\b")
RELEASE_RE = re.compile(r"\btask\s+(\d+)\s+\|\s+stop processing:")
EVAL_RE = re.compile(r"\btask\s+(\d+)\s+\|\s+eval time\s*=.*?/\s*(\d+) tokens\b")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

class Refusal(RuntimeError): pass

def load(path: Path):
    try: return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc: raise Refusal(f"{path}: invalid/missing JSON: {exc}") from exc

def sha_bytes(data: bytes): return hashlib.sha256(data).hexdigest()
def sha_file(path: Path):
    try: return sha_bytes(path.read_bytes())
    except OSError as exc: raise Refusal(f"{path}: unreadable: {exc}") from exc

def require(cond, message):
    if not cond: raise Refusal(message)

def parse_log(root: Path):
    path=root/'server.log'
    try: lines=path.read_text(errors='replace').splitlines()
    except OSError as exc: raise Refusal(f"{path}: unreadable: {exc}") from exc
    launched=set(); released=set(); eval_n={}; events=[]
    for lineno,line in enumerate(lines,1):
        if m:=LAUNCH_RE.search(line): launched.add(int(m.group(1)))
        if m:=RELEASE_RE.search(line): released.add(int(m.group(1)))
        if m:=EVAL_RE.search(line): eval_n[int(m.group(1))]=int(m.group(2))
        if m:=EVENT_RE.search(line):
            task,a,d=map(int,m.groups())
            require(d>=1 and 0<=a<=d, f"{path}:{lineno}: impossible verification event")
            events.append({'task':task,'accepted':a,'drafted':d,'log_ref':f'server.log:{lineno}',
                           'line_sha256':sha_bytes(line.encode())})
    measured=sorted(t for t,n in eval_n.items() if n==512 and t in launched and t in released)
    require(len(measured)==5, f"{path}: expected exactly five completed 512-token task windows, got {measured}")
    return {'path':path,'lines':lines,'events':events,'measured_tasks':measured,
            'measured_events':[e for e in events if e['task'] in measured]}

def audit_event_sidecar(root: Path, log):
    arm=load(root/'arm.json'); obs=load(root/'run-observation.json')
    side=arm.get('native_mtp',{}).get('verification_events')
    require(isinstance(side,list), f"{root}/arm.json: verification_events missing")
    require(side==obs.get('verification_events'), f"{root}: arm/run-observation verification events differ")
    raw_by_ref={e['log_ref']:e for e in log['events']}
    for i,e in enumerate(side):
        require(isinstance(e,dict), f"{root}: event[{i}] not object")
        ref=e.get('log_ref'); require(ref in raw_by_ref, f"{root}: event[{i}] ref does not resolve: {ref}")
        raw=raw_by_ref[ref]
        require(e.get('accepted')==raw['accepted'] and e.get('drafted')==raw['drafted'], f"{root}: event[{i}] counters differ from raw log")
        require(e.get('line_sha256')==raw['line_sha256'], f"{root}: event[{i}] line hash differs from raw log")
    require(len(side)==len(log['events']), f"{root}: sidecar omits or adds raw verification events")
    return arm

def expected_request(req, where):
    exact={'n_predict':512,'temperature':0.0,'top_k':1,'seed':42,'cache_prompt':False,
           'stream':False,'ignore_eos':True,'return_tokens':True}
    for k,v in exact.items(): require(req.get(k)==v and type(req.get(k)) is type(v), f"{where}: expected {k}={v!r}")
    require(isinstance(req.get('prompt'),str) and req['prompt'], f"{where}: nonempty rendered prompt required")

def audit_responses(root: Path, arm):
    probe=root/'probe'; rows=load(probe/'requests.json')
    require(isinstance(rows,list), f"{probe}/requests.json: expected list")
    measures=[r for r in rows if r.get('kind')=='measure']
    require(len(measures)==5, f"{root}: expected five measured sidecar rows")
    require(rows==arm.get('requests'), f"{root}: arm requests differ from probe requests.json")
    tokens=[]; payloads=[]; hashes=[]
    for i,row in enumerate(measures,1):
        reqp=probe/f'request-measure-{i}.json'; rspp=probe/f'response-measure-{i}.json'
        req=load(reqp); rsp=load(rspp); expected_request(req,str(reqp))
        tok=rsp.get('tokens'); require(isinstance(tok,list) and len(tok)==512, f"{rspp}: expected exactly 512 token IDs")
        require(all(isinstance(x,int) and not isinstance(x,bool) for x in tok), f"{rspp}: token IDs must be integers, not booleans")
        timings=rsp.get('timings') or {}
        require(timings.get('predicted_n')==512, f"{rspp}: timings.predicted_n must be 512")
        require(rsp.get('stop_type')=='limit' or rsp.get('stopped_limit') is True, f"{rspp}: response did not stop at requested limit")
        digest=sha_file(rspp)
        require(HEX64_RE.fullmatch(str(row.get('canonical_response_sha256',''))) is not None, f"{root}: malformed response hash row {i}")
        require(row['canonical_response_sha256']==digest, f"{root}: response hash mismatch row {i}")
        require(row.get('predicted_n')==512 and row.get('finish_reason')=='length', f"{root}: normalized measured row {i} has wrong shape/finish")
        tokens.append(tok); payloads.append(req); hashes.append(digest)
    require(all(x==payloads[0] for x in payloads[1:]), f"{root}: measured payloads differ within arm")
    require(all(x==tokens[0] for x in tokens[1:]), f"{root}: greedy token arrays differ within arm")
    return {'tokens':tokens,'payloads':payloads,'hashes':hashes,'rows':measures}

def strip_mtp(argv):
    value_flags={'--spec-type','--spec-draft-n-max','--spec-draft-p-min'}; out=[]; i=0
    while i<len(argv):
        if argv[i] in value_flags:
            require(i+1<len(argv), f"argv: {argv[i]} missing value"); i+=2
        else: out.append(argv[i]); i+=1
    return out

def identity_pair(mtp_root,plain_root,mtp_arm,plain_arm):
    mp=load(mtp_root/'plan.json'); pp=load(plain_root/'plan.json')
    require(mp.get('contract',{}).get('native_mtp_enabled') is True, 'MTP plan does not enable native MTP')
    require(pp.get('contract',{}).get('native_mtp_enabled') is False, 'plain plan does not disable native MTP')
    require(mp.get('contract',{}).get('spec_type')=='draft-mtp', 'MTP plan spec_type mismatch')
    require('--spec-type' not in pp.get('argv',[]), 'plain argv contains --spec-type')
    require(strip_mtp(mp.get('argv',[]))==pp.get('argv',[]), 'plan argv differ beyond MTP flags')
    require(mp.get('environment')==pp.get('environment'), 'plan environments differ')
    require(mp.get('binary')==pp.get('binary') and mp.get('model')==pp.get('model'), 'plan binary/model differ')
    ms=load(mtp_root/'source-identity.json'); ps=load(plain_root/'source-identity.json')
    for k in ('head','status_sha256','status','tracked_diff_sha256','key_sources'):
        require(ms.get(k)==ps.get(k), f"source identity differs: {k}")
    require(ms.get('status')==[], 'candidate source tree was not clean')
    mm=load(mtp_root/'model-identity.json'); pm=load(plain_root/'model-identity.json')
    for k in ('manifest_sha256','metadata_sha256','tensor_set_sha256','tensor_count','shards'):
        require(mm.get(k)==pm.get(k), f"model identity differs: {k}")
    mb=load(mtp_root/'loaded-libraries.json'); pb=load(plain_root/'loaded-libraries.json')
    require(mb.get('binary',{}).get('sha256')==pb.get('binary',{}).get('sha256'), 'server binary hashes differ')
    require(mtp_arm.get('evidence_status')=='observation' and plain_arm.get('evidence_status')=='observation', 'paired arms must remain observation status')
    return {'candidate_head':ms['head'],'binary_sha256':mb['binary']['sha256'],
            'model_metadata_sha256':mm['metadata_sha256'],'mtp_plan_sha256':mp.get('plan_sha256'),
            'plain_plan_sha256':pp.get('plan_sha256')}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mtp',type=Path,required=True); ap.add_argument('--plain',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    mr,pr=a.mtp.resolve(),a.plain.resolve()
    ml,pl=parse_log(mr),parse_log(pr)
    ma,pa=audit_event_sidecar(mr,ml),audit_event_sidecar(pr,pl)
    require(ma.get('native_mtp',{}).get('spec_type')=='draft-mtp', 'MTP arm spec_type mismatch')
    require(pa.get('native_mtp',{}).get('spec_type')=='none', 'plain arm spec_type must be none')
    require(not pl['events'], 'plain raw log contains draft verification events')
    me=ml['measured_events']
    require(any(e['accepted']>0 for e in me), 'measured MTP windows contain no accepted draft')
    require(any(e['accepted']<e['drafted'] for e in me), 'measured MTP windows contain no verified rejection')
    ident=identity_pair(mr,pr,ma,pa)
    mx,px=audit_responses(mr,ma),audit_responses(pr,pa)
    require(mx['payloads']==px['payloads'], 'plain/MTP measured request payloads differ')
    pair_diffs=[]
    for i,(mt,pt) in enumerate(zip(mx['tokens'],px['tokens']),1):
        first=next((j for j,(x,y) in enumerate(zip(mt,pt)) if x!=y),None)
        pair_diffs.append({'pair':i,'equal':first is None,'first_mismatch_index':first,
                           'mtp_token':None if first is None else mt[first],
                           'plain_token':None if first is None else pt[first],
                           'mtp_context':None if first is None else mt[max(0,first-6):min(len(mt),first+7)],
                           'plain_context':None if first is None else pt[max(0,first-6):min(len(pt),first+7)]})
    across=all(x['equal'] for x in pair_diffs)
    mrates=[float(x['predicted_per_second']) for x in mx['rows']]
    prates=[float(x['predicted_per_second']) for x in px['rows']]
    result={'schema':'epyc.glm53.paired_evidence_audit.v1','verdict':'PASS' if across else 'FAIL','classification':'categorical validation; no grading ladder',
            'identity':ident,'mtp_dir':str(mr),'plain_dir':str(pr),'measured_tasks':{'mtp':ml['measured_tasks'],'plain':pl['measured_tasks']},
            'measured_mtp_events':{'count':len(me),'accepted_events':sum(e['accepted']>0 for e in me),'rejection_events':sum(e['accepted']<e['drafted'] for e in me),
                                   'raw_refs':[e['log_ref'] for e in me]},
            'requests':{'pairs':5,'tokens_per_response':512,'within_arm_exact':True,'across_arms_exact':across,'payloads_exact':True,
                        'pair_differences':pair_diffs,
                        'ignore_eos_scope':'fixed-length stress trajectory; does not establish natural EOS/termination parity, broad prompt fidelity, or logit equality'},
            'performance_observation':{'mtp_predicted_per_second':mrates,'plain_predicted_per_second':prates,
                                       'mtp_median':statistics.median(mrates),'plain_median':statistics.median(prates),
                                       'speedup_valid':across,
                                       'limitation':None if across else 'semantic trajectory mismatch invalidates comparative speedup'},
            'response_sha256':{'mtp':mx['hashes'],'plain':px['hashes']}}
    encoded=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode(); a.output.write_bytes(encoded)
    a.output.with_suffix(a.output.suffix+'.sha256').write_text(f"{sha_bytes(encoded)}  {a.output.name}\n")
    print(json.dumps(result,indent=2,sort_keys=True)); return 0 if across else 3

if __name__=='__main__':
    try: raise SystemExit(main())
    except Refusal as exc: print(f'REFUSE: {exc}',file=sys.stderr); raise SystemExit(2)
