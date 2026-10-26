#!/usr/bin/env python3
"""물질70 참조 복사장만 CROSS=T로 재계산한다. 물질/스펙트럼 실행은 없다.

기존 feedback의 같은 입력 복제와 pre-pop STOP97 실행파일을 재사용한다.
원 참조와 초기 EDD를 보존하고 VADAT의 CROSS만 바꾼다.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

import run_cmfgen_field_feedback as F
from run_cmfgen_formal_primary import coverage, fields, validate_edd
from run_cmfgen_lambda_probe import control, sha


def direct_vadat(text):
    for key,value in (('CROSS','F'),('USE_FIXED_J','F'),('FIX_T','T'),('INC_GRID','F')):
        if control(text,key)!=value:raise ValueError('직접평가 참조의 기준 제어 불일치: '+key)
    if float(control(text,'V_CROSS').replace('D','E'))!=750.:
        raise ValueError('다른 continuum 재사용 기준')
    return F.set_control(text,'CROSS','T')


def provenance(reference):
    reference=Path(reference).resolve()
    inputs=json.loads((reference/'seq_logs/feedback-inputs.json').read_text())
    runtime=json.loads((reference/'seq_logs/feedback-runtime.json').read_text())
    stage=json.loads((reference/'field/seq_logs/feedback-stage.json').read_text())
    if (not runtime['field_complete'] or stage['input_state_record']!=70 or
            stage['next_native_iteration']!=71 or inputs['source_state']['record']!=70 or
            runtime['radiation_produced_from_record']!=70):
        raise ValueError('완료된 동일 물질70 참조가 아님')
    source=Path(inputs['source']);build=Path(inputs['build'])
    manifest_path=build/'feedback-build.json'
    if sha(manifest_path)!=inputs['build_manifest_sha256']:
        raise ValueError('참조 build 정체성 불일치')
    manifest=json.loads(manifest_path.read_text())
    if manifest['expected_iteration']!=71 or manifest['pre_population_stop']!=97:
        raise ValueError('pre-pop 물질70 참조 실행파일이 아님')
    for name,digest in {**manifest['input_sha256'],**manifest['output_sha256']}.items():
        if sha(Path(name))!=digest:raise ValueError('봉인 native 실행파일/의존 입력 변경: '+name)
    for name in F.RESTART_OVERLAY:
        digest=sha(source/name)
        if digest!=stage['restart_sha256'][name] or digest!=inputs['source_sha256'][str(source/name)]:
            raise ValueError('원래 물질/초기 복사장 변경: '+name)
    for name,digest in stage['controls_sha256'].items():
        if sha(reference/'field'/name)!=digest:raise ValueError('원래 수송 제어 변경: '+name)
    return inputs,stage,manifest


def prepare(reference,root,wall_seconds=1800):
    reference,root=Path(reference).resolve(),Path(root).resolve()
    if wall_seconds not in (1800,7200):
        raise ValueError('승인되지 않은 직접평가 시간 한도')
    if root.exists():raise FileExistsError('기존 직접평가 계산 보존')
    inputs,old,manifest=provenance(reference)
    root.mkdir();(root/'seq_logs').mkdir()
    field=root/'field'
    copied=F.clone_material(Path(inputs['original']),Path(inputs['source']),field,
        Path(manifest['native']),False,70,'1.0D6')
    for name in ('VADAT','IN_ITS'):
        if (field/name).read_bytes()!=(reference/'field'/name).read_bytes():
            raise ValueError('CROSS 외의 초기 수송 제어 차이: '+name)
    if copied['restart_sha256']!=old['restart_sha256']:
        raise ValueError('원 참조와 다른 초기 물질/EDD')
    vadat=field/'VADAT';before=vadat.read_text();after=direct_vadat(before)
    vadat.write_text(after)
    # DET_MAIN_CONT_FREQ의 CROSS=T 분기는 CONT_FREQ를 다시 쓰지 않는다.
    # 복제된 과거 대응표를 새 direct 계산의 출력으로 오인하지 않게 보존한다.
    stale=field/'CONT_FREQ'
    if stale.exists():stale.rename(field/'CONT_FREQ.cached_input')
    record=dict(schema='lumina-direct-continuum-reference-v1',state='PREPARED',
        reference=str(reference),source=inputs['source'],build=inputs['build'],
        build_manifest_sha256=inputs['build_manifest_sha256'],
        executable=str(Path(inputs['build'])/'cmfgen_feedback_field.exe'),
        baseline_field_sha256=json.loads((reference/'seq_logs/feedback-runtime.json').read_text())['field_evidence_sha256']['FORMAL_PRIMARY_FIELDS.bin'],
        input_state_record=70,next_native_iteration=71,expected_depths=90,expected_frequencies=197291,
        changed_controls=dict(CROSS='F -> T'),V_CROSS_unchanged=control(after,'V_CROSS'),
        input_restart_sha256=copied['restart_sha256'],
        input_control_sha256={n:sha(field/n) for n in ('VADAT','IN_ITS')},
        original_inputs_sha256={str(reference/'seq_logs'/n):sha(reference/'seq_logs'/n)
            for n in ('feedback-inputs.json','feedback-runtime.json')},
        runner_sha256={n:sha(Path(__file__).with_name(n)) for n in (
            'run_cmfgen_direct_continuum.py','run_cmfgen_field_feedback.py',
            'run_cmfgen_formal_primary.py','run_cmfgen_lambda_probe.py','configure_cmfgen_stageA.py',
            'prepare_cmfgen_fixed_field_owner.py','ptf_gated_metrics.py','cmfgen_step_precision.py',
            'run_cmfgen_fixed_field.py')},
        resources=dict(CPU=16,OMP=16,memory_GiB=32,wall_seconds=wall_seconds,GPU=0,retries=0),
        physical_scope='같은 물질의 continuum 직접평가 formal 장; 물질/SE/RE/observer 갱신 아님',
        material_updates=0,observer_calls=0,final_goal='NOT_MET')
    (root/'direct-inputs.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record


def run(reference,root,wall_seconds=1800):
    if wall_seconds not in (1800,7200):
        raise ValueError('승인되지 않은 직접평가 시간 한도')
    if (os.environ.get('OMP_NUM_THREADS')!='16' or not os.environ.get('SLURM_JOB_ID') or
            os.environ.get('SLURM_CPUS_PER_TASK')!='16' or os.environ.get('SLURM_MEM_PER_NODE')!='32768'):
        raise ValueError('승인된 Slurm CPU16/OMP16/32GiB에서만 실제 실행')
    started=time.monotonic();root=Path(root).resolve()
    inputs=prepare(reference,root,wall_seconds);field=root/'field'
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    env=dict(os.environ,LUMINA_FORMAL_PRIMARY='1',LUMINA_FORMAL_PRIMARY_WHOLE='1',
        LUMINA_FORMAL_CELL_BUDGET='',LUMINA_WHOLE_MODE='',LUMINA_MOM_CAPTURE_MODE='',
        LUMINA_FORMAL_CAPTURE_MODE='',LUMINA_NATIVE_SE_WITNESS='')
    # 복제/검증 시간을 포함하고 종료·출력 검증 여유를 남긴다.
    reserve=80 if wall_seconds==1800 else 200
    execution=F.child(root,field,Path(inputs['executable']),env,started+wall_seconds-reserve)
    result=dict(schema='lumina-direct-continuum-reference-result-v1',state='INCOMPLETE',
        execution=execution,field_complete=False,material_updates=0,observer_calls=0,
        physical_RT_executions=int(execution['returncode'] is not None),final_goal='NOT_MET')
    try:
        if execution['returncode']!=97 or execution['stop_reason']:
            raise ValueError('직접평가 장의 pre-pop 정상 완료 아님')
        marker=(field/'FORMAL_PRIMARY_COMPLETE.txt').read_text().splitlines()[0].split()
        if marker[:3]!=['71','197291','197291']:raise ValueError('직접평가 장 세대/coverage 불일치')
        cov=coverage(field/'FORMAL_PRIMARY_SWEEP.csv')
        if cov['count']!=197291:raise ValueError('부분 주파수 장')
        if control((field/'MODEL').read_text(),'CROSS')!='T':raise ValueError('native가 CROSS=T를 소비하지 않음')
        if sha(field/'SCRTEMP')!=inputs['input_restart_sha256']['SCRTEMP']:
            raise ValueError('pre-pop 물질이 변경됨')
        g,a=fields(field/'FORMAL_PRIMARY_FIELDS.bin')
        g0,a0=fields(Path(reference)/'field/FORMAL_PRIMARY_FIELDS.bin')
        try:
            if not np.array_equal(g,g0) or not np.array_equal(a['freq'],a0['freq']):
                raise ValueError('CROSS 변경 이외의 기하/주파수 격자 변경')
            validate_edd(field,a)
        finally:a._mmap.close();a0._mmap.close()
        result.update(state='FIELD_COMPLETE',field_complete=True,coverage=cov,
            material_byte_unchanged=True,geometry_and_frequency_unchanged=True,
            source_fixed_point_scope='주파수별 source/J/H 잔차 확인; SE/RE 공동수렴 아님')
    except (ValueError,OSError) as error:result['validation_error']=str(error)
    if execution['stop_reason']=='shared_child_budget_2400s':
        result['execution']['stop_reason']=f'direct_reference_{wall_seconds-reserve}s_including_preparation'
    evidence=('FORMAL_PRIMARY_FIELDS.bin','FORMAL_PRIMARY_SWEEP.csv','FORMAL_PRIMARY_COMPLETE.txt',
              'EDDFACTOR','EDDFACTOR_INFO','MODEL','VADAT','IN_ITS')
    result['evidence_sha256']={n:sha(field/n) for n in evidence if (field/n).is_file()}
    result['elapsed_total_seconds']=time.monotonic()-started
    (root/'direct-runtime.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result['field_complete'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--execute',action='store_true',help='승인된 Slurm 실제 복사장 단회 실행')
    parser.add_argument('--wall-seconds',type=int,default=1800)
    args=parser.parse_args()
    if args.execute:raise SystemExit(run(args.reference,args.out,args.wall_seconds))
    print(json.dumps(prepare(args.reference,args.out,args.wall_seconds),ensure_ascii=False))
