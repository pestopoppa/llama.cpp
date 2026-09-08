#!/usr/bin/env python3
"""Run one locked GLM-5.3 server arm and preserve its observation evidence."""
from __future__ import annotations

import argparse, hashlib, json, os, re, resource, signal, socket, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

# Select the INFO verification record. The later DEBUG `..., new n_tokens = N`
# line describes the same verification and must not double-count it.
EVENT_RE = re.compile(r"\baccepted\s+(\d+)\s*/\s*(\d+)\s+draft tokens(?: \(restore checkpoint\))?\s*$")
MTP_CONTEXT = "creating MTP draft context against the target model"

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def dump(path: Path, obj): path.write_text(json.dumps(obj, indent=2, sort_keys=True)+"\n")

def parse_events(path: Path):
    out=[]
    for n,line in enumerate(path.read_text(errors="replace").splitlines(),1):
        m=EVENT_RE.search(line)
        if not m: continue
        a,d=map(int,m.groups())
        if d < 1 or a > d: continue
        out.append({"accepted":a,"drafted":d,"verified_rejected":d-a,
                    "classification":"verified_partial_or_zero_accept" if a<d else "verified_all_accepted",
                    "log_ref":f"{path.name}:{n}","line_sha256":hashlib.sha256(line.encode()).hexdigest()})
    return out

def source_identity(tree: Path):
    def git(*args): return subprocess.run(["git","-C",str(tree),*args],text=True,capture_output=True,check=True).stdout
    head=git("rev-parse","HEAD").strip(); status=git("status","--porcelain=v1","--untracked-files=normal")
    diff=subprocess.run(["git","-C",str(tree),"diff","--binary","HEAD","--"],capture_output=True,check=True).stdout
    files={}
    for rel in ("src/models/glm5-next.cpp","src/models/glm5-next.h","common/speculative.cpp","tools/server/server-context.cpp"):
        p=tree/rel
        if p.is_file(): files[rel]={"sha256":sha(p),"size":p.stat().st_size}
    return {"tree":str(tree),"head":head,"status_sha256":hashlib.sha256(status.encode()).hexdigest(),
            "status":status.splitlines(),"tracked_diff_sha256":hashlib.sha256(diff).hexdigest(),"key_sources":files}

def loaded_libs(pid: int, bindir: Path):
    paths=set()
    for line in Path(f"/proc/{pid}/maps").read_text(errors="replace").splitlines():
        p=line.split()[-1] if "/" in line else ""
        if any(x in Path(p).name for x in ("libggml","libllama","libmtmd")): paths.add(p)
    rows=[{"path":p,"resolved":str(Path(p).resolve()),"sha256":sha(Path(p).resolve())} for p in sorted(paths)]
    foreign=[x["resolved"] for x in rows if not Path(x["resolved"]).is_relative_to(bindir.resolve())]
    names=[Path(x["resolved"]).name for x in rows]
    if foreign or not any(n.startswith("libggml-base.so") for n in names) or not any(n.startswith("libggml-cpu.so") for n in names):
        raise RuntimeError(f"mapped-library gate failed: foreign={foreign}, names={names}")
    return rows

def thread_ticks(exclude_pid):
    rows={}
    for pp in Path("/proc").glob("[0-9]*"):
        try: pid=int(pp.name)
        except ValueError: continue
        if pid==exclude_pid: continue
        for sp in (pp/"task").glob("[0-9]*/stat"):
            try:
                raw=sp.read_text(); end=raw.rfind(")"); rest=raw[end+2:].split(); tid=int(sp.parent.name)
                cpu=int(rest[36]); ticks=int(rest[11])+int(rest[12]); comm=raw[raw.find("(")+1:end]
                if 0<=cpu<=191: rows[(pid,tid)]={"ticks":ticks,"cpu":cpu,"comm":comm}
            except (OSError,ValueError,IndexError): continue
    return rows

def sample_contention(server_pid: int, stop: threading.Event, path: Path, interval: float):
    hz=os.sysconf("SC_CLK_TCK"); previous=thread_ticks(server_pid); previous_at=time.monotonic()
    with path.open("w") as f:
        while not stop.is_set():
            stop.wait(interval)
            current=thread_ticks(server_pid); at=time.monotonic(); elapsed=at-previous_at; active=[]
            for key,row in current.items():
                if key in previous and row["ticks"]>previous[key]["ticks"]:
                    active.append({"pid":key[0],"tid":key[1],"last_cpu":row["cpu"],"comm":row["comm"],"cpu_seconds_delta":(row["ticks"]-previous[key]["ticks"])/hz})
            f.write(json.dumps({"at_utc":now(),"interval_seconds":elapsed,"loadavg":Path("/proc/loadavg").read_text().strip(),"region":"physical cores represented by logical CPUs 0-191; server affinity is CPUs 0-95","metric":"per-thread /proc stat scheduler-tick delta over this interval","active_other_threads":active},sort_keys=True)+"\n"); f.flush()
            previous,previous_at=current,at

def wait_ready(port, proc, timeout):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if proc.poll() is not None: raise RuntimeError(f"server exited during startup rc={proc.returncode}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as r:
                if r.status==200: return
        except Exception: pass
        time.sleep(1)
    raise RuntimeError("server health timeout")

def stop_owned(proc):
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)

def smoke(port, out, timeout, prompt, n_predict):
    fmt={"messages":[{"role":"user","content":prompt}],"add_generation_prompt":True}
    (out/"apply-template-request.json").write_text(json.dumps(fmt,indent=2,sort_keys=True)+"\n")
    formatted=post_json(port,"/apply-template",fmt,timeout); dump(out/"apply-template-response.json",formatted); rendered=formatted["prompt"]
    (out/"rendered-prompt.txt").write_text(rendered)
    body={"prompt":rendered,"temperature":0.0,"top_k":1,"seed":42,"cache_prompt":False,"stream":False,"ignore_eos":True,"return_tokens":True,"n_predict":n_predict}
    raw=json.dumps(body,sort_keys=True).encode(); (out/"smoke-request.json").write_bytes(raw+b"\n")
    req=Request(f"http://127.0.0.1:{port}/completion",data=raw,headers={"Content-Type":"application/json"})
    with urlopen(req,timeout=timeout) as r: response=r.read()
    (out/"smoke-response.json").write_bytes(response+b"\n")
    doc=json.loads(response); t=doc.get("timings",{})
    return [{"kind":"smoke","predicted_n":t.get("predicted_n"),"predicted_per_second":t.get("predicted_per_second"),"draft_n":t.get("draft_n",0),"draft_n_accepted":t.get("draft_n_accepted",0),"finish_reason":(doc.get("choices")or[{}])[0].get("finish_reason"),"canonical_response_sha256":hashlib.sha256(response).hexdigest()}]

def post_json(port, route, body, timeout):
    raw=json.dumps(body,sort_keys=True).encode(); req=Request(f"http://127.0.0.1:{port}{route}",data=raw,headers={"Content-Type":"application/json"})
    with urlopen(req,timeout=timeout) as r: return json.loads(r.read())

def model_identity(manifest: Path, model: Path):
    doc=json.loads(manifest.read_text()); inv=doc["inventory"]
    checks=[]
    for row in inv["shards"]:
        p=model.parent/row["file"]; st=p.stat()
        ok=st.st_size==row["size"] and st.st_mtime_ns==row["mtime_ns"]
        checks.append({"file":str(p),"size":st.st_size,"mtime_ns":st.st_mtime_ns,"matches_pinned_stat":ok})
    if not all(x["matches_pinned_stat"] for x in checks): raise RuntimeError("model shard stat differs from pinned bounded-header manifest")
    return {"manifest":str(manifest),"manifest_sha256":sha(manifest),"read_policy":inv["read_policy"],
            "scope_limit":"GGUFReader mmaps files; identity touches headers and bounded first/last samples only, never traverses tensor payloads",
            "metadata_sha256":inv["metadata_sha256"],"tensor_set_sha256":inv["tensor_set_sha256"],"tensor_count":inv["tensor_count"],"shards":inv["shards"],"current_stat_checks":checks}

def preflight(a):
    plan=json.loads(a.plan.resolve(strict=True).read_text())
    if plan.get("schema")!="glm53-native-mtp-serve-plan-v1": raise RuntimeError("unexpected serve plan schema")
    binary=Path(plan["binary"]).resolve(strict=True); model=Path(plan["model"]).resolve(strict=True)
    if not os.access(binary,os.X_OK): raise RuntimeError("candidate server is not executable")
    if plan.get("contract",{}).get("native_mtp_enabled") not in (True,False): raise RuntimeError("plan lacks native_mtp_enabled boolean")
    if a.source_tree.resolve(strict=True) not in binary.parents: raise RuntimeError("plan binary is outside candidate source tree")
    if a.mode in ("full","screen"): a.probe.resolve(strict=True)
    a.plan_generator.resolve(strict=True); a.validator.resolve(strict=True)
    ident=model_identity(a.manifest.resolve(strict=True),model)
    return {"schema":"glm53-run-preflight-v1","plan":str(a.plan.resolve()),"binary":str(binary),"model":str(model),"source_tree":str(a.source_tree.resolve()),"probe":str(a.probe.resolve()),"native_mtp_enabled":plan["contract"]["native_mtp_enabled"],"model_identity":ident}

def instrument_identity(a):
    files={"wrapper":Path(__file__).resolve(),"probe":a.probe.resolve(),"plan":a.plan.resolve(),
           "plan_generator":a.plan_generator.resolve(),"validator":a.validator.resolve()}
    return {name:{"path":str(path),"sha256":sha(path),"size":path.stat().st_size} for name,path in files.items()}

def inside(a):
    out=a.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(a.plan.read_text()); argv=plan["argv"]; env=plan["environment"]
    if env.get("LD_LIBRARY_PATH")!=str(Path(plan["binary"]).parent): raise RuntimeError("plan lacks exact candidate LD_LIBRARY_PATH")
    if "--port" not in argv or int(argv[argv.index("--port")+1]) != a.port: raise RuntimeError("--port differs from serve plan")
    host=argv[argv.index("--host")+1]
    probe_socket=socket.socket(); probe_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,0)
    try: probe_socket.bind((host,a.port))
    except OSError as exc: raise RuntimeError(f"refusing launch: {host}:{a.port} is already bound") from exc
    finally: probe_socket.close()
    dump(out/"plan.json",plan); dump(out/"instrument-identity.json",instrument_identity(a)); dump(out/"source-identity.json",source_identity(a.source_tree)); dump(out/"model-identity.json",model_identity(a.manifest,Path(plan["model"])))
    link=subprocess.run([str(a.linkage),plan["binary"],str(a.source_tree)],env=env,text=True,capture_output=True)
    (out/"linkage.txt").write_text(link.stdout+link.stderr)
    if link.returncode: raise RuntimeError(f"linkage verifier failed rc={link.returncode}")
    log=(out/"server.log").open("w"); proc=None; stop=threading.Event(); th=None
    try:
        def child_limits(): resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        proc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,text=True,start_new_session=True,preexec_fn=child_limits)
        dump(out/"server-process.json",{"pid":proc.pid,"owned_by_wrapper_pid":os.getpid(),"started_at_utc":now(),"argv":argv,"environment":env})
        wait_ready(a.port,proc,a.startup_timeout)
        actual_argv=[x.decode(errors="replace") for x in Path(f"/proc/{proc.pid}/cmdline").read_bytes().split(b"\0") if x]
        actual_env={}
        for row in Path(f"/proc/{proc.pid}/environ").read_bytes().split(b"\0"):
            if b"=" in row:
                k,v=row.split(b"=",1); actual_env[k.decode(errors="replace")]=v.decode(errors="replace")
        dump(out/"process-readback.json",{"pid":proc.pid,"argv":actual_argv,"environment":actual_env})
        binary=Path(plan["binary"])
        dump(out/"loaded-libraries.json",{"captured_at_utc":now(),"pid":proc.pid,"binary":{"path":str(binary),"sha256":sha(binary),"size":binary.stat().st_size},"libraries":loaded_libs(proc.pid,binary.parent)})
        th=threading.Thread(target=sample_contention,args=(proc.pid,stop,out/"contention.jsonl",a.sample_interval),daemon=True); th.start()
        window_start=now()
        if a.mode=="smoke": requests=smoke(a.port,out,a.request_timeout,a.smoke_prompt,a.smoke_n_predict)
        else:
            po=out/"probe"; cp=subprocess.run([sys.executable,str(a.probe),"--port",str(a.port),"--out",str(po),"--timeout",str(a.request_timeout)],text=True,capture_output=True)
            (out/"probe.stdout").write_text(cp.stdout); (out/"probe.stderr").write_text(cp.stderr)
            if cp.returncode: raise RuntimeError(f"probe failed rc={cp.returncode}")
            requests=json.loads((po/"requests.json").read_text())
        window_end=now(); stop.set(); th.join(timeout=a.sample_interval+2)
        stop_owned(proc); log.close()
        events=parse_events(out/"server.log"); logtxt=(out/"server.log").read_text(errors="replace")
        summary={"schema":"glm53-run-observation-v1","mode":a.mode,"completed_at_utc":now(),"requests":requests,"verification_events":events,
                 "request_window":{"start_utc":window_start,"end_utc":window_end,"contention_ref":"contention.jsonl"},
                 "aggregate_counter_caveat":"draft_n minus draft_n_accepted is unaccepted; only parsed completed verification events support rejection claims",
                 "protocol_caveat":"timestamps and in-window samples establish overlap; smoke is not P-BENCH-4 and full remains observation absent all ratified host/instrument gates"}
        dump(out/"run-observation.json",summary)
        if a.mode=="full":
            lines=(out/"contention.jsonl").read_text().splitlines(); mtp=plan["contract"]["native_mtp_enabled"]
            uptime=float(Path("/proc/uptime").read_text().split()[0]); uptime_note=f"host uptime at evidence assembly: {uptime:.0f}s" + (" (>1 week; noncanonical for ratified comparison)" if uptime>604800 else "")
            libs=json.loads((out/"loaded-libraries.json").read_text())["libraries"]
            server_impl=next((x["sha256"] for x in libs if Path(x["resolved"]).name.startswith("libllama-server-impl")),None)
            arm={"schema":"epyc.glm53.server_arm.v1","arm_id":a.arm_id,"date":now()[:10],"metric":"accepted_output_tokens_per_second","metric_direction":"higher_better","protocol_id":"P-BENCH-4","reps":5,"evidence_status":"observation","argv":argv,"environment":env,
                 "binary":json.loads((out/"loaded-libraries.json").read_text()),"candidate":json.loads((out/"source-identity.json").read_text()),"model":json.loads((out/"model-identity.json").read_text()),
                 "instrument":json.loads((out/"instrument-identity.json").read_text()),
                 "host":{"hostname":socket.gethostname(),"uptime_seconds":uptime},"host_caveats":[summary["protocol_caveat"],uptime_note],"context_recipe":plan.get("derivation",{}),"in_window_sample_refs":[f"contention.jsonl:{i+1}" for i in range(len(lines))],"requests":requests,
                 "native_mtp":{"spec_type":"draft-mtp" if mtp else "none","draft_graph_dispatch_observed":bool(mtp and MTP_CONTEXT in logtxt and events),"dispatch_basis":[MTP_CONTEXT,"per-verification accepted A/D draft tokens"] if mtp else [],"trace_environment":env.get("LLAMA_TRACE"),"trace_proof":{"info_event_refs":[e["log_ref"] for e in events],"server_impl_sha256":server_impl if mtp else None},"verification_events":events},
                 "contention":{"sample_count":len(lines),"region":"0-95","raw_ref":"contention.jsonl"}}
            dump(out/"arm.json",arm)
        return 0
    finally:
        stop.set()
        if th: th.join(timeout=2)
        if proc and proc.poll() is None:
            stop_owned(proc)
        if proc: dump(out/"server-exit.json",{"pid":proc.pid,"returncode":proc.returncode,"confirmed_absent":not Path(f"/proc/{proc.pid}").exists(),"ended_at_utc":now()})
        if not log.closed: log.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--plan",type=Path,required=True); p.add_argument("--out",type=Path); p.add_argument("--source-tree",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--mode",choices=("smoke","screen","full"),default="smoke"); p.add_argument("--arm-id",default="glm53-mtp"); p.add_argument("--port",type=int,default=18497); p.add_argument("--probe",type=Path,default=Path(__file__).parents[1]/"runtime/probe_server.py"); p.add_argument("--plan-generator",type=Path,default=Path(__file__).parents[1]/"serve/generate_serve_plan.py"); p.add_argument("--validator",type=Path,default=Path(__file__).parents[1]/"runtime/validate_glm53.py"); p.add_argument("--linkage",type=Path,default=Path("/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh")); p.add_argument("--region-lock",type=Path,default=Path("/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock")); p.add_argument("--startup-timeout",type=int,default=1800); p.add_argument("--request-timeout",type=int,default=3600); p.add_argument("--sample-interval",type=float,default=2); p.add_argument("--smoke-prompt",default="Explain a CPU cache in one sentence."); p.add_argument("--smoke-n-predict",type=int,default=32); p.add_argument("--preflight-only",action="store_true"); p.add_argument("--inside-lock",action="store_true",help=argparse.SUPPRESS); a=p.parse_args()
    checked=preflight(a)
    if a.preflight_only: print(json.dumps(checked,indent=2)); return 0
    if a.out is None: p.error("--out is required unless --preflight-only")
    if a.inside_lock: return inside(a)
    cmd=[str(a.region_lock),"run","--cpu-list","0-95","--role","glm53-validation","--timeout-s","0","--tag",f"glm53:{a.mode}","--",sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--inside-lock"]
    print(json.dumps({"locked_exec":cmd,"notice":"acquires physical region lock before server launch"},indent=2),flush=True)
    return subprocess.call(cmd)
if __name__=="__main__": raise SystemExit(main())
