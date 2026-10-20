#!/usr/bin/env python3
"""첫 실제 주파수부터 affine 수리 이력을 연결하고 첫 실패/인구 갱신 전에 종료한다."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from run_cmfgen_lambda_probe import sha,stage,scan_logs,stop_child


FAULT_NAMES=("MOM_CAPTURE.bin","MOM_KERNEL.bin","MOM_PREVIOUS.bin","MOM_FAULT.bin","AFFINE_INPUT.bin",
             "FORMAL_BAD_in.bin","FORMAL_BAD_out.bin","FORMAL_BAD_rays_in.bin","FORMAL_BAD_rays_out.bin","FORMAL_BAD_geometry.bin",
             "FORMAL_EVENT_in.bin","FORMAL_EVENT_out.bin","FORMAL_EVENT_rays_in.bin","FORMAL_EVENT_rays_out.bin","FORMAL_EVENT_geometry.bin","FLOOP_STATE.bin")


def sweep_coverage(path):
    count=0;last=None;flags={};worst=None
    if not path.exists():return dict(count=0,last=None,flags=flags,worst_J_difference=worst)
    if path.stat().st_size>64*1024**2:raise ValueError("주파수 요약64MiB 초과")
    with path.open() as f:
        for row in csv.DictReader(f):
            count+=1
            if int(row["outer"])!=60 or int(row["index"])!=count:raise ValueError("주파수 coverage 불일치")
            last={k:float(v) if k not in ("outer","index","jit","mom_iterations","code","edge_support_min_location","J_difference_worst_depth") else int(v) for k,v in row.items()}
            code=int(row["code"]);flags[str(code)]=flags.get(str(code),0)+1
            if 'J_difference' in last and (worst is None or last['J_difference']>worst['J_difference']):worst=last.copy()
    return dict(count=count,last=last,flags=flags,worst_J_difference=worst)


def run(seed,root,native,build,baseline):
    started=time.monotonic();manifest=json.loads((build/"sweep-build.json").read_text())
    event_context=bool(manifest.get("capture_event",False));nonfatal_psd=bool(manifest.get('nonfatal_psd',False))
    event_only=event_context and not nonfatal_psd;child_seconds=500 if event_only else 2400
    if manifest["rebuilt_unmodified_sha256"]!="f2b9afcc064037a413bf206047a6b7c813882dc40f894436e5de704a32b232f1":
        raise ValueError("원래 부모 실행파일 불일치")
    for name,expected in manifest["output_sha256"].items():
        if sha(Path(name))!=expected:raise ValueError("whole build 변경: "+name)
    old=stage(seed,root,native/"exe/cmfgen_dev.exe",native)
    if old["seed_sha256"]!=json.loads(baseline.read_text())["seed_sha256"]:raise ValueError("whole 출발 입력 불일치")
    env=os.environ.copy()
    env.update(LUMINA_MOM_CAPTURE_MODE="CAPTURE_FIRST",LUMINA_FORMAL_CAPTURE_MODE="",LUMINA_WHOLE_MODE="1")
    if env.get("OMP_NUM_THREADS")!="16":raise ValueError("OMP16 필수")
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    record=dict(scope="outer60 첫 주파수부터197291개 실제 이력 연결; 인구/T 갱신 전 종료",seed=str(seed),run=str(root),
        build=str(build),build_manifest_sha256=sha(build/"sweep-build.json"),baseline_sha256=sha(baseline),
        parent_input_identity_equal=True,resources="CPU16/OMP16/16GiB/GPU0 "+("parent600s child500s" if event_only else "parent2700s child2400s")+" retry0, no node pin",
        capture_event_context=event_context,nonfatal_PSD_observation=nonfatal_psd,
        expected_frequencies=197291,scheduler_cluster=os.environ.get("SLURM_CLUSTER_NAME"),final_goal="NOT_JUDGED")
    with (root/"seq_logs/whole-inputs.json").open("x") as f:json.dump(record,f,ensure_ascii=False,indent=2)
    reason=None;cursors={};launch=time.monotonic()
    with (root/"seq_logs/cmfgen.stdout").open("xb") as out,(root/"seq_logs/cmfgen.stderr").open("xb") as err:
        proc=subprocess.Popen(["srun","--cpu-bind=cores",str(build/"cmfgen_sweep.exe")],cwd=root,env=env,
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        while proc.poll() is None:
            remaining=child_seconds-(time.monotonic()-launch)
            if remaining<=0:reason="child%ss"%child_seconds;stop_child(proc);break
            try:proc.wait(timeout=min(60,remaining))
            except subprocess.TimeoutExpired:
                reason=scan_logs(root,cursors)
                if reason:stop_child(proc);break
    coverage=sweep_coverage(root/"WHOLE_SWEEP.csv")
    files=[root/name for name in FAULT_NAMES if (root/name).is_file()]
    snapshot_bytes=sum(p.stat().st_size for p in files)
    if snapshot_bytes>256*1024**2:raise ValueError("최초 실패 snapshot256MiB 초과")
    complete=proc.returncode==97 and (root/"WHOLE_COMPLETE.txt").is_file() and coverage["count"]==197291
    if complete:
        counters=(root/"WHOLE_COMPLETE.txt").read_text().splitlines()[0].split()
        complete=(counters[:3]==["60","197291","197291"] and not files and
                  all(int(code) in ((0,3,4,8) if nonfatal_psd else (0,)) for code in coverage['flags']))
    fault=proc.returncode in (95,98) and ((root/"MOM_FAULT.bin").is_file() or (root/"FORMAL_BAD_in.bin").is_file())
    for name in ("WHOLE_SWEEP.csv","WHOLE_COMPLETE.txt","WHOLE_CALL_SUMMARY.txt","WHOLE_CAUSE.txt","MOM_ACCEL_SUMMARY.txt"):
        if (root/name).is_file():files.append(root/name)
    result=dict(returncode=proc.returncode,diagnostic_complete=bool(complete),first_failure_preserved=bool(fault),
        event_context_complete=bool(event_context and fault and all((root/name).is_file() for name in FAULT_NAMES[-6:])),
        stop_reason="pre_population_sweep_complete" if complete else "first_observed_failure" if fault else reason or "incomplete",
        elapsed_child_seconds=time.monotonic()-launch,elapsed_total_seconds=time.monotonic()-started,
        snapshot_bytes=snapshot_bytes,coverage=coverage,evidence_sha256={str(p):sha(p) for p in files},
        closure_pass="NOT_JUDGED",final_goal="NOT_JUDGED")
    with (root/"seq_logs/whole-runtime.json").open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
    return 0 if complete else 1


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("seed","run","native","build","baseline"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();raise SystemExit(run(a.seed,a.run,a.native,a.build,a.baseline))
