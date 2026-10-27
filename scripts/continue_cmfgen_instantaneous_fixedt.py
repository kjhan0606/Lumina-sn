#!/usr/bin/env python3
"""미수렴 fixed-T CMFGEN 상태를 보존 복제본에서 추가 반복한다."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time

from ptf_gated_metrics import (ScrtempReader, read_dimensions, validate_rvtj,
                               parse_outgen)
from prepare_cmfgen_fixed_field_owner import point_records
from run_cmfgen_field_feedback import clone_material, verify_material
from run_cmfgen_fixed_field import output_bytes, scan
from run_cmfgen_instantaneous_fixedt import (ba_policy_violation, check_resources,
                                             configure, state_summary)
from run_cmfgen_lambda_probe import control, sha, stop_child


def verify_seed(seed_run, source70, original, build, native, build_report_sha, ion_reference):
    seed_run, source70, original = map(Path.resolve, (seed_run, source70, original))
    material = seed_run/'material'
    result = json.loads((seed_run/'result.json').read_text())
    inputs = json.loads((seed_run/'inputs.json').read_text())
    reference = json.loads(Path(ion_reference).read_text())
    reference_hashes = reference.get('input_sha256', {})
    if (inputs['source70'] != str(source70) or
            inputs['inherited_stage']['original_physical_seed'] != str(original) or
            Path(inputs['full_build']).resolve() != Path(build).resolve() or
            inputs['build_report_sha256'] != build_report_sha or
            result.get('four_responses_complete') is not True or
            result.get('last_completed_record') != 74 or
            result.get('final_RVTJ', {}).get('complete') is not True or
            not result.get('source_unchanged') or not result.get('controls_unchanged') or
            not result.get('executable_unchanged') or
            Path(reference.get('rundir','')).resolve() != material.resolve() or
            reference.get('iteration_range',{}).get('last') != 74 or
            any(name not in reference_hashes or sha(material/name) != reference_hashes[name]
                for name in ('SCRTEMP','RVTJ','POPSIL','POPSUL')) or
            any(sha(material/name) != expected for name,expected in inputs['controls_sha256'].items())):
        raise ValueError('완료된 동일 봉인 74상태가 아님')
    state70, identity70, _ = verify_material(source70, original, 70)
    if inputs.get('input_identity') != identity70:
        raise ValueError('원70 입력 identity 계보 불일치')
    nd, nt = read_dimensions(material)
    reader = ScrtempReader(material/'SCRTEMP', nd, nt)
    if (reader.last_it != 74 or point_records(material) != dict(POINT1=74,POINT2=74) or
            not (reader.state(70) == state70).all()):
        raise ValueError('70→74 상태 계보 또는 완료 pointer 불일치')
    if not validate_rvtj(reader, material/'RVTJ')['complete']:
        raise ValueError('74상태와 RVTJ 불일치')
    v, it = (material/'VADAT').read_text(), (material/'IN_ITS').read_text()
    expected = dict(FIX_T='T', FIX_T_AUTO='F', FIX_NE='F', USE_FIXED_J='F',
                    COMP_BA='T', N_FIX_BA='0', DO_DDT='F', DO_NG='F')
    if any(control(v, key) != value for key, value in expected.items()):
        raise ValueError('74상태 물리 또는 fresh-BA 제어 변경')
    if control(it, 'DO_LAM_AUTO') != 'F' or control(it, 'DO_T_AUTO') != 'F':
        raise ValueError('자동 Lambda/온도 전환 방지 제어 변경')
    manifest = json.loads((Path(build)/'result.json').read_text())
    if (sha(Path(build)/'result.json') != build_report_sha or
            sha(Path(manifest['variants']['instantaneous_decay']['executable'])) !=
            manifest['variants']['instantaneous_decay']['executable_sha256'] or
            Path(native).resolve() != Path(inputs['inherited_stage']['native_binary']['path']).resolve().parents[1] or
            inputs['executable']['executable_sha256'] != manifest['variants']['instantaneous_decay']['executable_sha256']):
        raise ValueError('CMFGEN 실행파일/build 신원 불일치')
    state = reader.state(74)
    if not (state[-1] == state70[-1]).all():
        raise ValueError('원70 대비 온도 변경')
    return state, identity70, inputs


def validate_final_generation(material, last_record):
    points=point_records(material)
    if points != dict(POINT1=last_record,POINT2=last_record):
        raise ValueError('POINT1/POINT2/SCRTEMP record 세대 불일치')
    return points


def run(args):
    source_commit = args.source_commit
    if not re.fullmatch(r'[0-9a-f]{40}', source_commit):
        raise ValueError('source commit 형식 오류')
    from run_cmfgen_mom_capture import failure_profile,failure_build,failure_inputs,failure_evidence
    capturing=failure_profile(args)
    if not capturing:check_resources(args.memory_gib, seconds=args.seconds)
    if type(args.max_iterations) is not int or not 1 <= args.max_iterations <= 32:
        raise ValueError('추가 반복 상한은1~32회')
    state, identity, _ = verify_seed(args.seed_run, args.source70, args.original,
        args.build, args.native, args.build_report_sha256, args.ion_reference)
    manifest = json.loads((Path(args.build).resolve()/'result.json').read_text())
    variant=manifest['variants']['instantaneous_decay']
    context=(failure_build(args.capture_build,args.capture_build_sha256,
             variant['executable_sha256'],args.build_report_sha256) if capturing else None)
    failed=(failure_inputs(args.capture_failure_inputs,args.capture_failure_inputs_sha256,
            args,identity,variant['executable_sha256']) if capturing else None)
    out = args.out.resolve()
    if out.exists(): raise FileExistsError('기존 continuation 출력 보존')
    out.mkdir()
    material = out/'material'; cursors = {}; reason = None
    seed_material = args.seed_run.resolve()/'material'
    source_hashes = {str(seed_material/n): sha(seed_material/n)
                     for n in ('SCRTEMP','POINT1','POINT2','RVTJ','STEQ_VALS',
                               'EDDFACTOR','EDDFACTOR_INFO','CUR_MODEL_DATA')}
    if capturing and source_hashes!=failed['source_state_sha256']:
        raise ValueError('기존 실패와 다른74재시작 상태')
    inherited = clone_material(args.original.resolve(), seed_material, material,
        args.native.resolve(), False, source_record=74, max_lam='1.0D6')
    vadat, in_its = configure((material/'VADAT').read_text(),
        (material/'IN_ITS').read_text(), max_iterations=args.max_iterations)
    if (material/'IT_SPECIFIER').exists():
        raise ValueError('재시작 세대/온도 제어를 덮어쓰는 IT_SPECIFIER 금지')
    (material/'VADAT').write_text(vadat); (material/'IN_ITS').write_text(in_its)
    controls_sha = {n:sha(material/n) for n in ('VADAT','IN_ITS')}
    if capturing and controls_sha!=failed['controls_sha256']:
        raise ValueError('기존 실패의 NUM_ITS/물리/재시작 제어와 다름')
    executable=Path(context['executable'] if capturing else variant['executable'])
    executable_sha=context['executable_sha256'] if capturing else variant['executable_sha256']
    record = dict(status='STARTED', first_iteration=74,
        maximum_additional_iterations=args.max_iterations, final_goal='NOT_MET',
        released_temperature=False, anchor_status='NOT_AVAILABLE', source_commit=source_commit,
        source_70_identity=identity, parent_cmfg_result=str(args.seed_run.resolve()/'result.json'),
        parent_cmfg_result_sha256=sha(args.seed_run.resolve()/'result.json'),
        source_state_sha256=source_hashes, build_report_sha256=args.build_report_sha256,
        source_iteration_reference=str(Path(args.ion_reference).resolve()),
        source_iteration_reference_sha256=sha(Path(args.ion_reference).resolve()),
        inherited_stage=inherited,
        executable_sha256=executable_sha, controls_sha256=controls_sha,
        driver_source_sha256={str(Path(__file__).resolve()):sha(Path(__file__).resolve()),
            str(Path(__file__).with_name('run_cmfgen_instantaneous_fixedt.py').resolve()):
                sha(Path(__file__).with_name('run_cmfgen_instantaneous_fixedt.py').resolve()),
            str(Path(__file__).with_name('run_cmfgen_field_feedback.py').resolve()):
                sha(Path(__file__).with_name('run_cmfgen_field_feedback.py').resolve())},
        resource_envelope=dict(CPUs=16,OMP=16,memory_GiB=args.memory_gib,
            GPU=0,internal_seconds=args.seconds,retry=0),
        interpretation='완료 반복과 Si/S 변화 측정. 반복 상한 도달도 수렴 인증이 아님')
    if capturing:
        record.update(capture_build=context,capture_failure_inputs=str(args.capture_failure_inputs.resolve()),
            capture_failure_inputs_sha256=args.capture_failure_inputs_sha256,
            capture_stop_before_population_outer=78,
            interpretation='74상태에서 기존 NUM_ITS16을 보존하되78에서 포획/인구 갱신 전 중단. 앵커 계산 아님')
        for name in ('run_cmfgen_mom_capture.py','analyze_cmfgen_mom_capture.py','prepare_cmfgen_mom_capture.py'):
            path=Path(__file__).with_name(name).resolve()
            record['driver_source_sha256'][str(path)]=sha(path)
    (out/'inputs.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    if output_bytes(material) > 16*1024**3: raise ValueError('restart 복제본16GiB 초과')
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    env={k:v for k,v in os.environ.items() if not k.startswith('LUMINA_')}
    if capturing:env['LUMINA_MOM_CAPTURE_MODE']='CAPTURE'
    started=time.monotonic(); deadline=started+args.seconds
    with (material/'seq_logs/cmfgen.stdout').open('xb') as stdout, \
         (material/'seq_logs/cmfgen.stderr').open('xb') as stderr:
        child=subprocess.Popen(['srun','--cpu-bind=cores',str(executable)],cwd=material,
            env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        while child.poll() is None:
            remaining=deadline-time.monotonic()
            if remaining <= 0:
                reason=f'child_budget_{args.seconds}s'; stop_child(child); break
            try: child.wait(timeout=min(60,remaining))
            except subprocess.TimeoutExpired:
                log=material/'OUTGEN'
                reason=scan(material,cursors) or ba_policy_violation(log.read_text() if log.exists() else '')
                if reason: stop_child(child); break
    reason=reason or scan(material,cursors)
    log=material/'OUTGEN'
    reason=reason or ba_policy_violation(log.read_text() if log.exists() else '')
    record.update(returncode=child.returncode,stop_reason=reason,elapsed_seconds=time.monotonic()-started)
    try:
        summary=state_summary(material,state,first=74,max_iterations=args.max_iterations)
        record.update(summary)
        final_pointer=validate_final_generation(material,record['last_completed_record'])
        record['final_point_records']=final_pointer
        text=(material/'OUTGEN').read_text()
        raw=parse_outgen(material/'OUTGEN')
        record['raw_logged_corrections']={i:vars(row) for i,row in raw.items()}
        record['BA_computed_count']=len(re.findall(r'^\s*BA matrix computed\s*$',text,re.M))
        record['BA_not_computed_count']=text.count('BA matrix **NOT** computed')
        record['source_unchanged']=all(sha(Path(p))==h for p,h in source_hashes.items())
        record['controls_unchanged']=all(sha(material/n)==h for n,h in controls_sha.items())
        record['executable_unchanged']=sha(executable)==executable_sha
        record['completed_responses']=record['last_completed_record']-74
        record['run_complete']=(child.returncode==0 and reason is None and
            record['completed_responses']>0 and record['BA_not_computed_count']==0 and
            record['BA_computed_count']>=record['completed_responses'] and
            record['source_unchanged'] and record['controls_unchanged'] and
            record['executable_unchanged'] and record['final_RVTJ'] is not None and
            record['final_RVTJ']['complete'])
    except (ValueError,OSError,RuntimeError) as error:
        record['evidence_error']=str(error); record['run_complete']=False
    if capturing:
        record['run_complete']=False
        record['capture_complete']=False
        record['status']='TARGET_NOT_CAPTURED_PRESERVED'
        try:
            if (reason is not None or not all(sha(Path(p))==h for p,h in source_hashes.items()) or
                not all(sha(material/n)==h for n,h in controls_sha.items()) or sha(executable)!=executable_sha or
                sha(args.capture_failure_inputs)!=args.capture_failure_inputs_sha256):
                raise ValueError('포획 실행 오류 또는 봉인 원본/제어/실행파일 변경')
            failure_build(args.capture_build,args.capture_build_sha256,
                          variant['executable_sha256'],args.build_report_sha256)
            record['capture']=failure_evidence(material,context,child.returncode)
            record['capture_complete']=True
            record['status']='CAPTURED_NOT_PHYSICAL_CONVERGENCE'
            record['stop_reason']='moment_failure_captured_before_population_update'
        except (ValueError,OSError,RuntimeError) as error:
            record['capture_error']=str(error)
    (out/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='iterations'},ensure_ascii=False))
    return (0 if record['capture_complete'] else 1) if capturing else (0 if record['run_complete'] else 1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original','source70','seed-run','native','build','out'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--ion-reference',type=Path,required=True)
    p.add_argument('--source-commit',required=True); p.add_argument('--build-report-sha256',required=True)
    p.add_argument('--memory-gib',type=int,choices=(32,128),required=True)
    p.add_argument('--seconds',type=int,choices=(6600,13680,42600),required=True)
    p.add_argument('--max-iterations',type=int,default=16)
    p.add_argument('--capture-build',type=Path)
    p.add_argument('--capture-build-sha256')
    p.add_argument('--capture-failure-inputs',type=Path)
    p.add_argument('--capture-failure-inputs-sha256')
    raise SystemExit(run(p.parse_args()))
