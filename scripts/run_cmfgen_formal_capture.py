#!/usr/bin/env python3
"""동일 StageA59의 두 주파수 광선 상태만 포획하고 인구 갱신 전에 종료한다."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from run_cmfgen_lambda_probe import sha,stage,scan_logs,stop_child


def run(seed,run_root,native,build,baseline):
    started=time.monotonic()
    manifest=json.loads((build/"formal-build.json").read_text())
    if manifest["rebuilt_unmodified_sha256"]!="f2b9afcc064037a413bf206047a6b7c813882dc40f894436e5de704a32b232f1":
        raise ValueError("formal의 원래 부모 신원 불일치")
    for name,expected in manifest["output_sha256"].items():
        if sha(Path(name))!=expected:raise ValueError("formal build 변경")
    old=stage(seed,run_root,native/"exe/cmfgen_dev.exe",native)
    previous=json.loads(baseline.read_text())
    if old["seed_sha256"]!=previous["seed_sha256"]:raise ValueError("formal 출발 입력 불일치")
    env=os.environ.copy();env["LUMINA_MOM_CAPTURE_MODE"]="";env["LUMINA_FORMAL_CAPTURE_MODE"]="CAPTURE_FORMAL"
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    record=dict(scope="23731→23732 실제 formal 광선 입력/출력; 물리 갱신 금지",seed=str(seed),run=str(run_root),
        build=str(build),build_manifest_sha256=sha(build/"formal-build.json"),baseline_sha256=sha(baseline),
        parent_input_identity_equal=True,resources="CPU16/16GiB/GPU0 wall600s child500s retry0",
        scheduler_cluster=os.environ.get("SLURM_CLUSTER_NAME"),final_goal="NOT_JUDGED")
    with (run_root/"seq_logs/formal-inputs.json").open("x") as f:json.dump(record,f,ensure_ascii=False,indent=2)
    cursors={};reason=None;launch=time.monotonic()
    with (run_root/"seq_logs/cmfgen.stdout").open("xb") as out,(run_root/"seq_logs/cmfgen.stderr").open("xb") as err:
        proc=subprocess.Popen(["srun","--cpu-bind=cores",str(build/"cmfgen_formal_capture.exe")],cwd=run_root,
            env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        while proc.poll() is None:
            remaining=500-(time.monotonic()-launch)
            if remaining<=0:reason="child500s";stop_child(proc);break
            try:proc.wait(timeout=min(60,remaining))
            except subprocess.TimeoutExpired:
                reason=scan_logs(run_root,cursors)
                if reason:stop_child(proc);break
    from analyze_cmfgen_formal_capture import formal_state
    phases=[];files=[]
    for index in (23731,23732):
        for jit in range(21):
            prefix=run_root/("FORMAL_%05d_%02d"%(index,jit))
            if not Path(str(prefix)+"_in.bin").exists():continue
            for suffix in ("_in.bin","_out.bin","_rays_in.bin","_rays_out.bin","_geometry.bin"):
                path=Path(str(prefix)+suffix)
                if not path.is_file():raise ValueError("formal 부분 포획: "+str(path))
                files.append(path)
            state=formal_state(Path(str(prefix)+"_out.bin"),manifest["ray_inventory"])
            if state["phase"][:3]!=(60,index,jit):raise ValueError("formal phase 불일치")
            phases.append(state["phase"])
    if sum(p.stat().st_size for p in files)>256*1024**2:raise ValueError("formal 포획 출력256MiB 초과")
    complete=proc.returncode==96 and (60,23732,0,0,1) in phases and any(p[1]==23731 for p in phases)
    result=dict(returncode=proc.returncode,phases=phases,complete=complete,
        stop_reason="formal_saved_before_MOM_and_population" if complete else reason or "no_complete_formal_capture",
        elapsed_child_seconds=time.monotonic()-launch,elapsed_total_seconds=time.monotonic()-started,
        capture_bytes=sum(p.stat().st_size for p in files),evidence_sha256={str(p):sha(p) for p in files},final_goal="NOT_JUDGED")
    with (run_root/"seq_logs/formal-runtime.json").open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
    return 0 if complete else 1


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("seed","run","native","build","baseline"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();raise SystemExit(run(a.seed,a.run,a.native,a.build,a.baseline))
