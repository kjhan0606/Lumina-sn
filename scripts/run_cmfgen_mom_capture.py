#!/usr/bin/env python3
"""StageA59의 정확한 입력으로 최초60회 MOM 실패 호출만 한 번 포획한다."""
import argparse
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import time

from run_cmfgen_lambda_probe import sha, stage, stop_child, scan_logs, control


def capture_header(path,allow_initial=False,*,expected_outer=60):
    if type(expected_outer) is not int or not 1<=expected_outer<=2147483647:
        raise ValueError("명시한 포획 외부 반복 오류")
    if not path.is_file() or not 56 <= path.stat().st_size <= 256*1024*1024:
        raise ValueError("포획 상태 누락/크기 초과")
    with path.open("rb") as stream:
        raw=stream.read(56)
    if raw[:16].rstrip()!=b"LUMINAMOMCAPV1" or struct.unpack_from("=4i",raw,16)!=(8,64,32,32):
        raise ValueError("포획 kind/형식 불일치")
    outer,index,jit,first,new,nd=struct.unpack_from("=6i",raw,32)
    if outer!=expected_outer or index<1 or jit<0 or first not in ((0,1) if allow_initial else (0,)) or new not in (0,1) or nd!=90:
        raise ValueError("포획 호출 phase 불일치")
    return dict(outer_iteration=outer,frequency_index=index,J_IT_COUNTER=jit,
                FIRST_FREQ=bool(first),NEW_FREQ=bool(new),ND_SM=nd)


def verify_build(build):
    record=json.loads((build/"capture-build.json").read_text())
    if record.get("capture_target") is not None:
        raise ValueError("60회 전용 실행기에 새 포획 선택을 전달할 수 없음; 정확한 재시작/자원 범위가 필요")
    if record["rebuilt_unmodified_sha256"]!="f2b9afcc064037a413bf206047a6b7c813882dc40f894436e5de704a32b232f1":
        raise ValueError("원래 산술의 부모 빌드 미확인")
    for name,expected in record["output_sha256"].items():
        if sha(Path(name))!=expected:
            raise ValueError("포획 build 변경: "+name)
    return record


def run(seed,run_root,native,build,baseline,first_repair=False):
    started=time.monotonic()
    build_record=verify_build(build)
    if bool(build_record.get("first_repair",False))!=first_repair:
        raise ValueError("포획 목적과 build가 다름")
    previous=json.loads(baseline.read_text())
    old=stage(seed,run_root,native/"exe/cmfgen_dev.exe",native)
    if old["seed_sha256"]!=previous["seed_sha256"]:
        raise ValueError("실패 런과 출발 입력이 다름")
    # NUM_ITS=1은 LST_ITERATION을 바꾼다. 기존4회 제어를 그대로 유지하고
    # 포획/별도 source 종료점으로 첫 SOLVE_FOR_POPS 전에 끝낸다.
    if control((run_root/"IN_ITS").read_text(),"NUM_ITS")!="4":
        raise ValueError("원래 first-iteration 경로 변경")
    if (run_root/"VADAT").read_bytes()!=(seed/"VADAT").read_bytes():
        raise ValueError("원래 물리 입력 변경")
    env=os.environ.copy()
    env["LUMINA_MOM_CAPTURE_MODE"]="CAPTURE_FIRST" if first_repair else "CAPTURE"
    # native EDDFACTOR/BAMAT 자체는256MiB보다 크다. 새 포획물의 제한은
    # Fortran snapshot224MiB+trace32MiB에서 별도로 강제한다.
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    record=dict(scope="원래60회 phase의 MOM 실패 포획; 인구/온도 갱신 금지",
        run=str(run_root),seed=str(seed),build=str(build),build_manifest_sha256=sha(build/"capture-build.json"),
        failure_baseline_sha256=sha(baseline),parent_input_identity_equal=True,
        IN_ITS_sha256=sha(run_root/"IN_ITS"),VADAT_sha256=sha(run_root/"VADAT"),
        resources="CPU16 OMP16 RAM%sGiB GPU0 wall1200s; child cap1080s, retry0" % (16 if first_repair else 200),
        first_repair=first_repair,scheduler_cluster=os.environ.get("SLURM_CLUSTER_NAME"),
        final_goal="NOT_JUDGED")
    with (run_root/"seq_logs/mom-capture-inputs.json").open("x") as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2);stream.write("\n")
    cursors={};reason=None;launch=time.monotonic()
    with (run_root/"seq_logs/cmfgen.stdout").open("xb") as out,(run_root/"seq_logs/cmfgen.stderr").open("xb") as err:
        proc=subprocess.Popen(["srun","--cpu-bind=cores",str(build/"cmfgen_capture.exe")],cwd=run_root,
                              env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        while proc.poll() is None:
            remaining=1080-(time.monotonic()-launch)
            if remaining<=0:
                reason="child_cap1080s";stop_child(proc);break
            try:
                proc.wait(timeout=min(60,remaining))
            except subprocess.TimeoutExpired:
                reason=scan_logs(run_root,cursors)
                if reason:
                    stop_child(proc);break
    phase=None
    if (run_root/"MOM_CAPTURE.bin").exists():
        phase=capture_header(run_root/"MOM_CAPTURE.bin",allow_initial=first_repair)
        reason="capture_saved_before_population_update"
    else:
        reason=reason or scan_logs(run_root,cursors) or "no_failure_capture"
    event=None
    if first_repair and phase is not None:
        from analyze_cmfgen_mom_capture import first_event
        event=first_event(run_root/"MOM_FAULT.bin")
        if event["phase"]!=(phase["outer_iteration"],phase["frequency_index"],phase["J_IT_COUNTER"],int(phase["FIRST_FREQ"]),int(phase["NEW_FREQ"])):
            raise ValueError("최초 사건과 호출 snapshot phase가 다름")
        event={k:v for k,v in event.items() if k not in ("matrix","raw","hraw","slope")}
    names=("MOM_CAPTURE.bin","MOM_TRACE.bin","MOM_KERNEL.bin","MOM_PREVIOUS.bin","MOM_FAULT.bin") if first_repair else ("MOM_CAPTURE.bin","MOM_TRACE.bin")
    files=[run_root/name for name in names if (run_root/name).exists()]
    if sum(p.stat().st_size for p in files)>256*1024*1024:
        raise ValueError("추가 포획 출력 합계256MiB 초과")
    result=dict(returncode=proc.returncode,stop_reason=reason,phase=phase,event=event,
        elapsed_child_seconds=time.monotonic()-launch,elapsed_total_seconds=time.monotonic()-started,
        evidence_sha256={str(p):sha(p) for p in files},final_goal="NOT_JUDGED")
    with (run_root/"seq_logs/mom-capture-runtime.json").open("x") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write("\n")
    print(json.dumps(result,ensure_ascii=False))
    return 0 if phase is not None and proc.returncode==(95 if first_repair else 93) else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("seed","run","native","build","baseline"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--first-repair",action="store_true")
    a=parser.parse_args()
    raise SystemExit(run(a.seed.resolve(),a.run.resolve(),a.native.resolve(),a.build.resolve(),a.baseline.resolve(),a.first_repair))
