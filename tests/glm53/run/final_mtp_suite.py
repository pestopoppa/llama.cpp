#!/usr/bin/env python3
"""Run all final native-MTP workloads on one server load."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
CANON=HERE/"canonical_workload_client.py";QUALITY=HERE/"quality_workload_client.py"
DECODE=HERE.parent/"runtime"/"probe_server.py"
MULTI=HERE/"multiprompt512_client.py"
def run(script,out,a,extra=()):
    cp=subprocess.run([sys.executable,str(script),"--port",str(a.port),"--out",str(out),"--timeout",str(a.timeout),*map(str,extra)],text=True,capture_output=True)
    (a.out/f"{out.name}.stdout").write_text(cp.stdout);(a.out/f"{out.name}.stderr").write_text(cp.stderr)
    if cp.returncode:raise RuntimeError(f"{script.name} failed rc={cp.returncode}")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--port",type=int,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--timeout",type=int,default=3600);ap.add_argument("--scorer-root",type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    sources={p.name:{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in (CANON,QUALITY,DECODE,MULTI)}
    (a.out/"instrument-provenance.json").write_text(json.dumps(sources,indent=2,sort_keys=True)+"\n")
    run(CANON,a.out/"canonical24",a);run(QUALITY,a.out/"quality5",a,("--scorer-root",a.scorer_root));run(DECODE,a.out/"decode512x5",a);run(MULTI,a.out/"multiprompt512x2",a)
    combined=[]
    for phase in ("canonical24","quality5","decode512x5","multiprompt512x2"):
        combined.extend([{"suite_phase":phase,**r} for r in json.loads((a.out/phase/"requests.json").read_text())])
    (a.out/"requests.json").write_text(json.dumps(combined,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"canonical24":24,"quality5":5,"decode512x5":5,"multiprompt512":2}))
if __name__=="__main__":raise SystemExit(main())
