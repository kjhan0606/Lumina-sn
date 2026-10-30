#!/usr/bin/env python3
"""StageA59의 정확한 입력으로 최초60회 MOM 실패 호출만 한 번 포획한다."""
import argparse
import json
import os
from pathlib import Path
import resource
import re
import struct
import subprocess
import time

from run_cmfgen_lambda_probe import sha, stage, stop_child, scan_logs, control


FAILURE_TARGET = dict(frequency_native=6.10732968, outer_iteration=78,
                      frequency_absolute_tolerance=5.e-9)
BLUE_ENTRY_TARGET = dict(frequency_native=6.107497416899375, outer_iteration=78,
                         frequency_absolute_tolerance=5.e-9)
FAILURE_SECONDS = 13680


def failure_profile(args,environment=None):
    """승인된 추가 단회만 허용한다. 일반 continuation 시간 범위는 바꾸지 않는다."""
    fields=[getattr(args,n,None) for n in ('capture_build','capture_build_sha256',
            'capture_failure_inputs','capture_failure_inputs_sha256')]
    if not any(v is not None for v in fields):return False
    if not all(v is not None for v in fields):raise ValueError('포획 입력/해시 네 인수 모두 필요')
    env=os.environ if environment is None else environment
    if (type(args.memory_gib) is not int or args.memory_gib!=128 or
        type(args.seconds) is not int or args.seconds!=FAILURE_SECONDS or
        type(args.max_iterations) is not int or args.max_iterations!=16 or
        env.get('OMP_NUM_THREADS')!='16' or env.get('SLURM_CPUS_PER_TASK')!='16' or
        env.get('SLURM_MEM_PER_NODE')!='131072' or not env.get('SLURM_JOB_ID')):
        raise ValueError('포획은 CPU16/OMP16/128GiB·13680초·기존 NUM_ITS16 전용')
    return True


def failure_build(path,expected_sha,parent_sha,reference_sha):
    path=Path(path).resolve();report=path/'capture-build.json'
    if not re.fullmatch('[0-9a-f]{64}',expected_sha or '') or sha(report)!=expected_sha:
        raise ValueError('포획 build 보고서 신원 불일치')
    data=json.loads(report.read_text());parent=Path(data['parent'])/'parent-build-identity.json'
    entry_only=data.get('entry_only',False)
    if type(entry_only) is not bool:
        raise ValueError('포획 진입 모드 형식 오류')
    expected_target=BLUE_ENTRY_TARGET if entry_only else FAILURE_TARGET
    if (data.get('capture_target')!=expected_target or data.get('first_repair') is not False or
        data.get('rebuilt_unmodified_sha256')!=parent_sha or sha(parent)!=data['parent_manifest_sha256']):
        raise ValueError('포획 주파수/반복/교정 부모 불일치')
    provenance=json.loads(parent.read_text())
    if (provenance.get('reference_build_sha256')!=reference_sha or
        provenance.get('reference_executable_sha256')!=parent_sha):
        raise ValueError('포획 부모의 원 교정 build 계보 불일치')
    for group in ('input_sha256','output_sha256'):
        if not data.get(group) or any(sha(Path(p))!=h for p,h in data[group].items()):
            raise ValueError('포획 build 입력/출력 변경')
    executable=path/'cmfgen_capture.exe';inventory=path/'state_inventory.json'
    if any(str(p) not in data['output_sha256'] for p in (executable,inventory)):
        raise ValueError('포획 실행파일/상태 목록 누락')
    return dict(directory=str(path),report_sha256=expected_sha,
        executable=str(executable),executable_sha256=data['output_sha256'][str(executable)],
        inventory=str(inventory),target=expected_target,entry_only=entry_only,parent_executable_sha256=parent_sha)


def failure_inputs(path,expected_sha,args,identity,parent_sha):
    path=Path(path).resolve()
    if not re.fullmatch('[0-9a-f]{64}',expected_sha or '') or sha(path)!=expected_sha:
        raise ValueError('기존 실패 입력 보고서 신원 불일치')
    data=json.loads(path.read_text())
    if (data.get('first_iteration')!=74 or data.get('maximum_additional_iterations')!=16 or
        data.get('source_70_identity')!=identity or data.get('executable_sha256')!=parent_sha or
        data.get('build_report_sha256')!=args.build_report_sha256 or
        data.get('parent_cmfg_result_sha256')!=sha(args.seed_run.resolve()/'result.json') or
        data.get('source_iteration_reference_sha256')!=sha(args.ion_reference.resolve())):
        raise ValueError('기존74→78 실패의 입력/실행 계보 불일치')
    return data


def failure_evidence(root,context,returncode):
    """완전 snapshot/trace를 확인한다. 물리 수렴이나 전체모델 성공은 아니다."""
    from analyze_cmfgen_mom_capture import capture,trace,summary
    root=Path(root);snapshot=root/'MOM_CAPTURE.bin';history=root/'MOM_TRACE.bin'
    if context.get('entry_only',False):
        if returncode!=96 or not snapshot.is_file() or snapshot.stat().st_size>224*1024**2:
            raise ValueError('예상 STOP96 또는 이전 주파수 진입 snapshot이 없음')
        header=capture_header(snapshot,expected_outer=78)
        saved=capture(snapshot,json.loads(Path(context['inventory']).read_text()))
        target=context['target']
        if abs(saved['args']['FREQ']-target['frequency_native'])>target['frequency_absolute_tolerance']:
            raise ValueError('이전 주파수 진입 snapshot 신원 불일치')
        return dict(phase=header,frequency_native=saved['args']['FREQ'],
                    sha256={snapshot.name:sha(snapshot)},status='BLUE_ENTRY_NOT_PHYSICAL_CONVERGENCE')
    if returncode!=93 or not snapshot.is_file() or not history.is_file():
        raise ValueError('예상 STOP93 또는 실패 포획물이 없음')
    if snapshot.stat().st_size>224*1024**2 or history.stat().st_size>32*1024**2:
        raise ValueError('포획 snapshot/trace 크기 상한 초과')
    header=capture_header(snapshot,expected_outer=78)
    saved=capture(snapshot,json.loads(Path(context['inventory']).read_text()))
    calls=trace(history)
    if (not calls or tuple(saved['phase'])!=tuple(calls[-1]['phase']) or
        abs(saved['args']['FREQ']-FAILURE_TARGET['frequency_native'])>5.e-9 or
        calls[-1]['frequency']!=saved['args']['FREQ']):
        raise ValueError('포획과 trace의 주파수/호출 phase 불일치')
    rows=summary(calls[-1],saved['args']['INNER_BND_METH'])
    if (len(rows)!=100 or rows[-1]['iteration']!=100 or
        not isinstance(rows[-1]['MAX_ER'],(int,float)) or rows[-1]['MAX_ER']<1.e-8):
        raise ValueError('100회 실패 trace가 완전하지 않음')
    return dict(phase=header,frequency_native=saved['args']['FREQ'],
        last_iteration=rows[-1],sha256={p.name:sha(p) for p in (snapshot,history)},
        status='CAPTURED_NOT_PHYSICAL_CONVERGENCE')


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
