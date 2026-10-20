#!/usr/bin/env python3
"""완성 J60을 고정하고 native의 전체 원소 SE/ne를 고정T에서 네 번 갱신한다.

스펙트럼/flux/heat 출력은 대용 H를 쓰므로 물리 판정에 사용하지 않는다.
원래 실행/원자자료/장/인구는 보존하며 외부 C 인구를 주입하지 않는다.
"""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time

import numpy as np

from configure_cmfgen_stageA import set_control
from run_cmfgen_lambda_probe import stage,sha,control,stop_child
from run_cmfgen_formal_primary import fields,validate_edd
from ptf_gated_metrics import ScrtempReader,read_dimensions

BAD=re.compile(r'^\s*(?:\*+\s*)?(?:Error\b|Fatal\b|Fortran runtime error\b|Program received signal\b)|'
               r'\b(?:NaN|[+-]?Infinity)\b',re.I|re.M)
LOCAL=('atomic_local/SUL/III/19apr23/phot_data_A','atomic_local/COB/II/19apr23/phot_data_A')


def controls(vadat,in_its):
    expected=dict(FIX_T='T',FIX_T_AUTO='F',FIX_NE='F',DO_DDT='F',DO_NG='F',TRT_NON_TE='F',
        USE_FIXED_J='F',WRITE_RATES='T',WRITE_JH='T',INC_GRID='F')
    for key,value in expected.items():
        if control(vadat,key)!=value:raise ValueError('봉인 제어 불일치: '+key)
    for key in ('DO_LAM_AUTO','DO_T_AUTO'):
        if control(in_its,key)!='F':raise ValueError('자동 모드 전환 금지: '+key)
    return set_control(vadat,'USE_FIXED_J','T'),set_control(set_control(in_its,'DO_LAM_IT','T'),'NUM_ITS','4')


def output_bytes(root):
    total=0
    for folder in (root,root/'seq_logs'):
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False):total+=entry.stat(follow_symlinks=False).st_size
                elif folder==root and entry.is_dir(follow_symlinks=False) and entry.name not in ('seq_logs','atomic_local'):
                    raise ValueError('예상 밖 출력 하위 디렉터리: '+entry.name)
    for name in LOCAL:
        path=root/name
        if path.is_file():total+=path.stat().st_size
    return total


def scan(root,cursors):
    reason=None
    for name in ('OUTGEN','seq_logs/cmfgen.stdout','seq_logs/cmfgen.stderr'):
        path=root/name
        if not path.exists():continue
        offset,tail=cursors.get(name,(0,''))
        with path.open(errors='replace') as stream:
            stream.seek(offset);new=stream.read();offset=stream.tell()
        match=BAD.search(tail+new)
        if match:reason='native_solver_error:'+name+':'+match.group(0)
        cursors[name]=(offset,(tail+new)[-256:])
    if output_bytes(root)>16*1024**3:reason='root_size_cap_16GiB'
    return reason


def prepare(seed,field,root,native):
    if root.exists():raise ValueError('기존 계산 보존: '+str(root))
    # 모든 수치/대용량 검증은 offline 계산 노드 또는 Slurm 안에서 실행한다.
    runtime=json.loads((field/'seq_logs/whole-runtime.json').read_text())
    if not runtime.get('diagnostic_complete') or runtime.get('returncode')!=97:
        raise ValueError('완료된 J60만 사용')
    f=field/'FORMAL_PRIMARY_FIELDS.bin'
    if sha(f)!=runtime['evidence_sha256'].get(str(f)):raise ValueError('공통장 원본 변경')
    if sha(field/'SCRTEMP')!=sha(seed/'SCRTEMP'):raise ValueError('물질59가 아닌 장')
    g,a=fields(f);validate_edd(field,a)
    if a['field'].shape!=(197291,8,90):raise ValueError('새 장 coverage 불일치')
    next_vadat,next_in=controls((seed/'VADAT').read_text(),(seed/'IN_ITS').read_text())
    executable=native/'exe/cmfgen_dev.exe'
    record=stage(seed,root,executable,native,write_record=False)
    for name in ('EDDFACTOR','EDDFACTOR_INFO'):shutil.copy2(field/name,root/name)
    (root/'VADAT').write_text(next_vadat);(root/'IN_ITS').write_text(next_in)
    # stage()의 과거 Lambda 기록으로 실행 의미/자원을 잘못 표기하지 않는다.
    record.update(scope=__doc__,changed_controls=dict(USE_FIXED_J='F -> T',DO_LAM_IT='F -> T',NUM_ITS='19 -> 4'),
        preserved_controls='VADAT는 USE_FIXED_J만 변경; DO_LAM_AUTO=F/DO_T_AUTO=F 및 나머지 물리 제어 보존',
        field=str(field),field_sha256=sha(f),fixed_field_inputs={name:sha(root/name) for name in ('EDDFACTOR','EDDFACTOR_INFO')},
        effective_controls=dict(USE_FIXED_J='T',DO_LAM_IT='T',NUM_ITS='4',FIX_T='T',FIX_NE='F',
            DO_DDT='F',DO_NG='F',DO_LAM_AUTO='F',DO_T_AUTO='F',INC_GRID='F'),
        effective_input_paths={name:str(root/name) for name in ('VADAT','IN_ITS','SCRTEMP','EDDFACTOR')},
        input_controls_sha256={name:sha(root/name) for name in ('VADAT','IN_ITS')},
        resources='CPU16/OMP16/128GiB/GPU0 parent2700s child2400s retry0 no node pin',
        max_seconds=2400,file_cap_bytes=8*1024**3,root_cap_bytes=16*1024**3,
        useful_probe_criteria=None,physical_acceptance='전체 SE/전하 잔차와 일치 상태의 C 비교가 필요; 네 번 완료 자체가 수렴 아님',
        invalid_physics_outputs=['OBSFLUX','flux/luminosity/heat/energy outputs using native fixed-J dummy H'],
        frozen_field_not_joint_convergence=True,final_goal='NOT_JUDGED',
        runner_sha256=sha(Path(__file__)),source_dependencies={str(Path(__file__).with_name(name)):sha(Path(__file__).with_name(name))
            for name in ('run_cmfgen_lambda_probe.py','run_cmfgen_formal_primary.py','ptf_gated_metrics.py','configure_cmfgen_stageA.py')})
    if output_bytes(root)>16*1024**3:raise ValueError('초기 복제본이 전체16GiB 상한 초과')
    (root/'seq_logs/fixed-field-inputs.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record


def run(seed,field,root,native):
    if os.environ.get('OMP_NUM_THREADS')!='16' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Slurm CPU16/OMP16 내부에서만 실행')
    record=prepare(seed,field,root,native)
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    cursors={};reason=None;started=time.monotonic()
    with (root/'seq_logs/cmfgen.stdout').open('xb') as stdout,(root/'seq_logs/cmfgen.stderr').open('xb') as stderr:
        proc=subprocess.Popen(['srun','--cpu-bind=cores',record['binary']['path']],cwd=root,
            stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        while proc.poll() is None:
            remaining=2400-(time.monotonic()-started)
            if remaining<=0:reason='child2400s';stop_child(proc);break
            try:proc.wait(timeout=min(60,remaining))
            except subprocess.TimeoutExpired:
                reason=scan(root,cursors)
                if reason:stop_child(proc);break
    reason=reason or scan(root,cursors)
    unchanged=all(sha(root/name)==h for name,h in record['fixed_field_inputs'].items())
    seed_unchanged=all(sha(Path(name))==h for name,h in record['seed_sha256'].items())
    nd,nt=read_dimensions(root);reader=ScrtempReader(root/'SCRTEMP',nd,nt)
    current=reader.state(reader.last_it);initial=ScrtempReader(seed/'SCRTEMP',nd,nt).state(59)
    point=(root/'POINT1').read_text().splitlines()[1].split()
    temperature_unchanged=bool(np.array_equal(current[-1],initial[-1]))
    completed=proc.returncode==0 and not reason and unchanged and seed_unchanged and temperature_unchanged and reader.last_it==63 and point[:2]==['63','63']
    result=dict(returncode=proc.returncode,stop_reason=reason,last_completed_record=reader.last_it,
        four_native_updates_complete=completed,elapsed_seconds=time.monotonic()-started,
        fixed_J_unchanged=unchanged,seed_unchanged=seed_unchanged,T_unchanged=temperature_unchanged,
        n_e_relative_change=float(np.max(abs(current[-2]/initial[-2]-1))),
        positive_ne=bool(np.all(current[-2]>0)),populations_nonnegative=bool(np.all(current[:-2]>=0)),
        root_bytes=output_bytes(root),conditional_SE_convergence='NOT_JUDGED',energy_pass='NOT_APPLICABLE_DUMMY_H',
        joint_convergence=False,final_goal='NOT_JUDGED')
    (root/'seq_logs/fixed-field-runtime.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False));return 0 if completed else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('seed','field','run','native'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();raise SystemExit(run(*(getattr(args,k).resolve() for k in ('seed','field','run','native'))))
