#!/usr/bin/env python3
"""Strict categorical audit of the GLM5Next serial-spec diagnostic against plain."""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path

from audit_paired_evidence import (Refusal, audit_event_sidecar, audit_responses, load,
                                   parse_log, require, sha_bytes, strip_mtp)

def mad(xs):
    m=statistics.median(xs); return statistics.median(abs(x-m) for x in xs)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--serial',type=Path,required=True)
    ap.add_argument('--plain',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    sr,pr=a.serial.resolve(),a.plain.resolve(); sl,pl=parse_log(sr),parse_log(pr)
    sa,pa=audit_event_sidecar(sr,sl),audit_event_sidecar(pr,pl)
    sp,pp=load(sr/'plan.json'),load(pr/'plan.json')
    require(sp.get('contract',{}).get('native_mtp_enabled') is True and sp.get('contract',{}).get('spec_type')=='draft-mtp',
            'serial plan is not native MTP')
    require(pp.get('contract',{}).get('native_mtp_enabled') is False and str(pp.get('contract',{}).get('spec_type','')).startswith('none'),
            'plain plan is not plain')
    se,pe=dict(sp.get('environment',{})),dict(pp.get('environment',{}))
    require(se.pop('LLAMA_SPEC_EXACT',None)=='serial' and 'LLAMA_SPEC_EXACT' not in pe,
            'required sole LLAMA_SPEC_EXACT=serial delta absent')
    require(se==pe,'environment differs beyond LLAMA_SPEC_EXACT=serial')
    require(strip_mtp(sp.get('argv',[]))==pp.get('argv',[]),'argv differs beyond native MTP flags')
    require(sp.get('binary')==pp.get('binary') and sp.get('model')==pp.get('model'),'plan binary/model differ')
    ss,ps=load(sr/'source-identity.json'),load(pr/'source-identity.json')
    for k in ('head','status_sha256','status','tracked_diff_sha256','key_sources'):
        require(ss.get(k)==ps.get(k),f'source identity differs: {k}')
    require(ss.get('status')==[],'candidate source was not clean')
    sm,pm=load(sr/'model-identity.json'),load(pr/'model-identity.json')
    for k in ('manifest_sha256','metadata_sha256','tensor_set_sha256','tensor_count','shards'):
        require(sm.get(k)==pm.get(k),f'model identity differs: {k}')
    sb,pb=load(sr/'loaded-libraries.json'),load(pr/'loaded-libraries.json')
    require(sb.get('binary',{}).get('sha256')==pb.get('binary',{}).get('sha256'),'binary hashes differ')
    require(sa.get('evidence_status')=='observation' and pa.get('evidence_status')=='observation','status must be observation')
    require(not pl['events'],'plain log contains verification events')
    ev=sl['measured_events']; require(ev,'serial measured windows lack verification events')
    require(any(e['accepted']>0 for e in ev),'serial measured windows lack accepted drafts')
    require(any(e['accepted']<e['drafted'] for e in ev),'serial measured windows lack verified rejection')
    sx,px=audit_responses(sr,sa),audit_responses(pr,pa)
    require(sx['payloads']==px['payloads'],'saved request payloads differ')
    diffs=[]
    for i,(st,pt) in enumerate(zip(sx['tokens'],px['tokens']),1):
        first=next((j for j,(x,y) in enumerate(zip(st,pt)) if x!=y),None)
        diffs.append({'pair':i,'equal':first is None,'first_mismatch_index':first})
    equal=all(x['equal'] for x in diffs)
    srate=[float(x['predicted_per_second']) for x in sx['rows']]
    prate=[float(x['predicted_per_second']) for x in px['rows']]
    out={'schema':'epyc.glm53.serial_plain_audit.v1','verdict':'PASS' if equal else 'FAIL',
         'classification':'categorical diagnostic validation; no grading ladder',
         'identity':{'candidate_head':ss['head'],'binary_sha256':sb['binary']['sha256'],
                     'model_metadata_sha256':sm['metadata_sha256'],'serial_plan_sha256':sp.get('plan_sha256'),
                     'plain_plan_sha256':pp.get('plan_sha256')},
         'plan_delta':{'environment':{'LLAMA_SPEC_EXACT':'serial'},'argv':'native MTP flags only'},
         'measured_tasks':{'serial':sl['measured_tasks'],'plain':pl['measured_tasks']},
         'measured_serial_events':{'count':len(ev),'drafted':sum(e['drafted'] for e in ev),
                                   'accepted':sum(e['accepted'] for e in ev),
                                   'accepted_events':sum(e['accepted']>0 for e in ev),
                                   'verified_rejection_events':sum(e['accepted']<e['drafted'] for e in ev),
                                   'raw_refs':[e['log_ref'] for e in ev]},
         'responses':{'pairs':5,'tokens_per_response':512,'saved_payloads_exact':True,
                      'within_arm_exact':True,'across_arms_exact':equal,'pair_differences':diffs,
                      'serial_sha256':sx['hashes'],'plain_sha256':px['hashes']},
         'timing_observation':{'serial_rates':srate,'plain_rates':prate,
                               'serial_median':statistics.median(srate),'serial_mad':mad(srate),
                               'plain_median':statistics.median(prate),'plain_mad':mad(prate),
                               'native_parallel_speed_claim_valid':False,
                               'scope':'LLAMA_SPEC_EXACT=serial diagnostic path; not normal parallel native MTP'},
         'interpretation':'serial target verification reproduces plain greedy trajectory' if equal else 'serial target verification differs from plain'}
    encoded=(json.dumps(out,indent=2,sort_keys=True)+'\n').encode(); a.output.write_bytes(encoded)
    a.output.with_suffix(a.output.suffix+'.sha256').write_text(f'{sha_bytes(encoded)}  {a.output.name}\n')
    print(json.dumps(out,indent=2,sort_keys=True)); return 0 if equal else 3

if __name__=='__main__':
    try: raise SystemExit(main())
    except Refusal as exc: print(f'REFUSE: {exc}',file=sys.stderr); raise SystemExit(2)
