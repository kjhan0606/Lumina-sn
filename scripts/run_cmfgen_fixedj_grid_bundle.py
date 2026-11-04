#!/usr/bin/env python3
"""승인된 h=0.5/0.25 동일-state native 포획 두 건의 격리 준비·순차 실행."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from prepare_cmfgen_fixedj_heat import refined_grid_inputs, patched
import run_cmfgen_fixedj_heat_capture as capture
from run_cmfgen_lambda_probe import INPUTS, RESTART
from run_cmfgen_field_feedback import RESTART_OVERLAY


def observer_only_build(baseline_build, replacement):
    """원 물리 객체/소스는 보존하고 관측 스펙트럼 생략만 허용한다. 빌드/실행하지 않는다."""
    old=json.loads((baseline_build/'fixedj-heat-build.json').read_text())
    new=json.loads((replacement/'fixedj-heat-build.json').read_text())
    if (old['schema']!='lumina-native-fixedj-integral-re-build-v1' or
        new['schema']!=old['schema'] or
        new.get('observer_spectrum')!='NOT_EVALUATED_NO_OBSFLUX' or
        new['decay_implementation']!=old['decay_implementation']):
        raise ValueError('관측 스펙트럼만 제외한 동일 계보 빌드가 필요함')
    without_driver=lambda m:{p:d for p,d in m['input_sha256'].items()
                             if Path(p).name!='prepare_cmfgen_fixedj_heat.py'}
    if without_driver(old)!=without_driver(new):
        raise ValueError('교체 빌드의 물리 소스/객체/라이브러리가 변경됨')
    candidates=[Path(p) for p in old['input_sha256'] if Path(p).name=='cmfgen_sub.f']
    if len(candidates)!=1 or capture.sha(candidates[0])!=old['input_sha256'][str(candidates[0])]:
        raise ValueError('봉인 원 caller 소스 신원이 다름')
    original=candidates[0].read_text()
    for manifest,text in [(old,patched(original)),(new,patched(original,skip_observer_spectrum=True))]:
        if hashlib.sha256(text.encode()).hexdigest()!=manifest['output_sha256']['cmfgen_sub.f']:
            raise ValueError('관측 경로 외의 생성 소스 차이가 있음')
    for name in ('cmfgen_sub.f','cmfgen_fixedj_heat.exe'):
        if capture.sha(replacement/name)!=new['output_sha256'][name]:
            raise ValueError('교체 빌드 산출물 신원이 다름: '+name)
    return new


def completed_bridge_reference(root):
    """완료된 h05의 네 물리 출력을 봉인한다. 미완료/부분 출력은 기준이 될 수 없다."""
    path=root/'native/seq_logs/fixedj-heat-runtime.json'
    report=json.loads(path.read_text())
    if (report.get('status')!='CAPTURED' or report.get('srun_exit_code')!=96 or
        report.get('captured_state_exact') is not True or
        float(capture.control((root/'native/VADAT').read_text(),'FRAC_DOP').replace('D','E'))!=.5):
        raise ValueError('정상 완료한 h05 포획만 교체 기준으로 사용함')
    expected={name:report['outputs'][name]['sha256'] for name in capture.OUTPUT_NAMES}
    for name,digest in expected.items():
        if capture.sha(root/'native'/name)!=digest:
            raise ValueError('기준 h05 물리 출력 변경: '+name)
    return dict(root=str(root),runtime_sha256=capture.sha(path),output_sha256=expected,
                build_sha256=report['build_sha256'],fixed_j_sha256=report['fixed_j_sha256'],
                accepted_response_sha256=report['accepted_response_sha256'])


def compare_bridge(root,reference):
    # 산술·물리 정책 변경 없이 두 관측 출력 구간만 제거했으므로 우선 정확 재현을 요구한다.
    # 불일치 시 자동 허용오차 완화/후속 실행 없이 보존·분석한다.
    expected=reference['output_sha256']
    if set(expected)!=set(capture.OUTPUT_NAMES):
        raise ValueError('h05 대조 기준의 필수 물리 출력 누락')
    actual={name:capture.sha(root/'native'/name) for name in capture.OUTPUT_NAMES}
    different=[name for name in expected if expected[name]!=actual[name]]
    if different:
        raise ValueError('교체 빌드 h05 물리 출력 불일치: '+', '.join(different))
    return dict(status='EXACT_PHYSICAL_OUTPUT_REPRODUCTION',outputs=actual,
                final_goal='NOT_MET')


def stage(baseline, output, *, replacement_build=None, bridge_reference=None):
    baseline=baseline.resolve(strict=True);output=output.resolve()
    if (replacement_build is None)!=(bridge_reference is None):
        raise ValueError('교체 빌드와 완료 h05 대조 기준은 함께 지정해야 함')
    if output.exists():
        raise FileExistsError('새 묶음 경로가 필요함; 기존 자료 보존')
    if shutil.disk_usage(output.parent).free < 32*1024**3:
        raise ValueError('32GiB 출력 여유 공간 부족')
    history=json.loads((baseline/'native/seq_logs/feedback-stage.json').read_text())
    material=Path(history['material_source'])
    original=Path(history['original_physical_seed'])
    if any(p==output or p in output.parents for p in (baseline,material,original)):
        raise ValueError('원본 트리 안에 출력 금지')
    sources={name:original/name for name in INPUTS+RESTART if name!='setup_links.sh'}
    sources.update({name:material/name for name in RESTART_OVERLAY})
    sources.update({name:baseline/'native'/name for name in
        ('VADAT','IN_ITS','MODEL_SPEC','EDDFACTOR','EDDFACTOR_INFO','NATIVE_HEAT_INPUT.bin')})
    identities={path:capture.sha(path) for path in sources.values()}
    for name,digest in history['restart_sha256'].items():
        if capture.sha(material/name)!=digest:
            raise ValueError('원 포획 출발 checkpoint 변경: '+name)
    for name,item in history['linked_inputs'].items():
        target=Path(item['target'])
        if capture.sha(target)!=item['sha256']:
            raise ValueError('원자/보조 입력 변경: '+name)
        identities[target]=item['sha256']
    texts=[refined_grid_inputs((baseline/'native/VADAT').read_text(),
           (baseline/'native/MODEL_SPEC').read_text(),frac_dop=h) for h in (.5,.25)]
    build=json.loads((baseline/'build/fixedj-heat-build.json').read_text())
    exe=baseline/'build/cmfgen_fixedj_heat.exe'
    if capture.sha(exe)!=build['output_sha256'][exe.name]:
        raise ValueError('기준 실행파일 신원 변경')
    build_root=baseline/'build';bridge=None
    if replacement_build is not None:
        build_root=replacement_build.resolve(strict=True)
        bridge_root=bridge_reference.resolve(strict=True)
        if any(p==output or p in output.parents for p in (build_root,bridge_root)):
            raise ValueError('빌드/보존 기준 트리 안에 출력 금지')
        build=observer_only_build(baseline/'build',build_root)
        bridge=completed_bridge_reference(bridge_root)
        bridge['replacement_build']=str(build_root)
        if bridge['build_sha256']!=capture.sha(exe):
            raise ValueError('기준 h05는 원 실행파일의 완료 포획이어야 함')
        for name in ('EDDFACTOR','EDDFACTOR_INFO','NATIVE_HEAT_INPUT.bin'):
            if capture.sha(bridge_root/'native'/name)!=capture.sha(sources[name]):
                raise ValueError('기준 h05와 준비 상태/J가 다름: '+name)
        for name,key in [('VADAT','vadat'),('MODEL_SPEC','model_spec')]:
            if (bridge_root/'native'/name).read_text()!=texts[0][key]:
                raise ValueError('기준 h05와 준비 격자/control이 다름: '+name)
    for name in ('cmfgen_fixedj_heat.exe','fixedj-heat-build.json'):
        identities[build_root/name]=capture.sha(build_root/name)
    output.mkdir()
    attempts=[]
    for name,h,changes in zip(('h05','h025'),(.5,.25),texts):
        root=output/name
        for folder in ('native','inputs','build'):
            (root/folder).mkdir(parents=True)
        for filename,path in sources.items():
            if filename not in ('VADAT','MODEL_SPEC'):
                shutil.copy2(path,root/'native'/filename)
        (root/'native/VADAT').write_text(changes['vadat'])
        (root/'native/MODEL_SPEC').write_text(changes['model_spec'])
        for filename,item in history['linked_inputs'].items():
            (root/'native'/filename).symlink_to(Path(item['target']).resolve(strict=True))
        for filename in ('heat-state-input.json','fixedj-field-input.json'):
            shutil.copy2(baseline/'inputs'/filename,root/'inputs'/filename)
        for filename in ('cmfgen_fixedj_heat.exe','fixedj-heat-build.json'):
            shutil.copy2(build_root/filename,root/'build'/filename)
        # 원래 준비기의 stale-output/신원 확인을 그대로 재사용한다. 실행하지 않는다.
        checked=capture.validate_inputs(root)
        if bridge is not None and name=='h05' and (
            bridge['fixed_j_sha256']!=checked['fixed_j_sha256'] or
            bridge['accepted_response_sha256']!=checked['state_manifest']['accepted_response_sha256']):
            raise ValueError('기준 h05 포획과 준비 물질/J 신원이 다름')
        attempts.append(dict(root=name,frac_dop=h,input_sha256={
            f:capture.sha(root/'native'/f) for f in sources}))
    for path,digest in identities.items():
        if capture.sha(path)!=digest:
            raise ValueError('격리 준비 중 원본 변경: '+path.name)
    drivers=('run_cmfgen_fixedj_grid_bundle.py','run_cmfgen_fixedj_heat_capture.py',
             'prepare_cmfgen_fixedj_heat.py')
    report=dict(schema='lumina-fixedj-grid-bundle-v1',baseline=str(baseline),
        capture_build_sha256=build['output_sha256']['cmfgen_fixedj_heat.exe'],
        attempts=attempts,resources=dict(cpus=16,memory_MiB=32768,wall_seconds=21600,
        captures=2,retries=0,gpus=0),source_sha256={n:capture.sha(Path(__file__).with_name(n)) for n in drivers},
        source_input_sha256=[dict(path=str(p),sha256=h) for p,h in identities.items()],
        new_transport=False,material_update=False,final_goal='NOT_MET')
    if bridge is not None:report['observer_build_bridge']=bridge
    (output/'bundle-inputs.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def run_bundle(root, capture_runner=None, clock=time.monotonic):
    if capture_runner is None:
        capture_runner=capture.run
    root=root.resolve(strict=True)
    record=json.loads((root/'bundle-inputs.json').read_text())
    if (record['schema']!='lumina-fixedj-grid-bundle-v1' or
        record['resources']!=dict(cpus=16,memory_MiB=32768,wall_seconds=21600,captures=2,retries=0,gpus=0) or
        [(a['root'],a['frac_dop']) for a in record['attempts']]!=[('h05',.5),('h025',.25)]):
        raise ValueError('승인된 두 포획 순서/자원과 다름')
    if (os.environ.get('SLURM_CPUS_PER_TASK')!='16' or
        os.environ.get('SLURM_MEM_PER_NODE')!='32768' or not os.environ.get('SLURM_JOB_ID')):
        raise ValueError('Grammar CPU16/32GiB Slurm 할당이 필요함')
    for name,digest in record['source_sha256'].items():
        if capture.sha(Path(__file__).with_name(name))!=digest:
            raise ValueError('준비한 운전자 소스 변경: '+name)
    runtime=root/'bundle-runtime.json'
    result=dict(status='RUNNING',job_id=os.environ['SLURM_JOB_ID'],captures=[],final_goal='NOT_MET')
    with runtime.open('x') as stream:
        json.dump(result,stream)
    started=clock()
    try:
        for i,attempt in enumerate(record['attempts']):
            if i and clock()-started>21600-5400:
                raise RuntimeError('두 번째 포획에 최소90분 여유가 없어 시작하지 않음')
            for name,digest in attempt['input_sha256'].items():
                if capture.sha(root/attempt['root']/'native'/name)!=digest:
                    raise ValueError('준비 입력 변경: '+name)
            if (record.get('capture_build_sha256') and
                capture.sha(root/attempt['root']/'build/cmfgen_fixedj_heat.exe')!=record['capture_build_sha256']):
                raise ValueError('준비한 포획 실행파일 변경')
            code=capture_runner(root/attempt['root'])
            result['captures'].append(dict(root=attempt['root'],frac_dop=attempt['frac_dop'],returncode=code))
            if code:
                raise RuntimeError('포획 실패; 후속/재시도 없음')
            if i==0 and 'observer_build_bridge' in record:
                result['observer_build_bridge']=compare_bridge(root/attempt['root'],record['observer_build_bridge'])
            runtime.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        result['status']='CAPTURED_BOTH_NOT_PHYSICAL_EQUIVALENCE'
    except Exception as error:
        result.update(status='STOPPED',failure=f'{type(error).__name__}: {error}')
    result['wall_seconds']=clock()-started
    runtime.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['status']=='CAPTURED_BOTH_NOT_PHYSICAL_EQUIVALENCE' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--bundle-root',type=Path)
    parser.add_argument('--replacement-build',type=Path)
    parser.add_argument('--bridge-reference',type=Path)
    args=parser.parse_args()
    if args.bundle_root:
        if any((args.baseline,args.output,args.replacement_build,args.bridge_reference)):
            parser.error('실행에는 --bundle-root만 지정')
        raise SystemExit(run_bundle(args.bundle_root))
    if not args.baseline or not args.output:
        parser.error('준비에는 --baseline/--output 필요')
    report=stage(args.baseline,args.output,replacement_build=args.replacement_build,
                 bridge_reference=args.bridge_reference)
    print(json.dumps(dict(prepared=args.output.name,attempts=[a['root'] for a in report['attempts']],
                         final_goal='NOT_MET'),ensure_ascii=False))
