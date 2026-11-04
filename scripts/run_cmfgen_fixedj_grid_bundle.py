#!/usr/bin/env python3
"""승인된 h=0.5/0.25 동일-state native 포획 두 건의 격리 준비·순차 실행."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from prepare_cmfgen_fixedj_heat import refined_grid_inputs
import run_cmfgen_fixedj_heat_capture as capture
from run_cmfgen_lambda_probe import INPUTS, RESTART
from run_cmfgen_field_feedback import RESTART_OVERLAY


def stage(baseline, output):
    baseline=baseline.resolve(strict=True);output=output.resolve()
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
            shutil.copy2(baseline/'build'/filename,root/'build'/filename)
        # 원래 준비기의 stale-output/신원 확인을 그대로 재사용한다. 실행하지 않는다.
        capture.validate_inputs(root)
        attempts.append(dict(root=name,frac_dop=h,input_sha256={
            f:capture.sha(root/'native'/f) for f in sources}))
    for path,digest in identities.items():
        if capture.sha(path)!=digest:
            raise ValueError('격리 준비 중 원본 변경: '+path.name)
    drivers=('run_cmfgen_fixedj_grid_bundle.py','run_cmfgen_fixedj_heat_capture.py',
             'prepare_cmfgen_fixedj_heat.py')
    report=dict(schema='lumina-fixedj-grid-bundle-v1',baseline=str(baseline),
        attempts=attempts,resources=dict(cpus=16,memory_MiB=32768,wall_seconds=21600,
        captures=2,retries=0,gpus=0),source_sha256={n:capture.sha(Path(__file__).with_name(n)) for n in drivers},
        source_input_sha256=[dict(path=str(p),sha256=h) for p,h in identities.items()],
        new_transport=False,material_update=False,final_goal='NOT_MET')
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
            code=capture_runner(root/attempt['root'])
            result['captures'].append(dict(root=attempt['root'],frac_dop=attempt['frac_dop'],returncode=code))
            if code:
                raise RuntimeError('포획 실패; 후속/재시도 없음')
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
    args=parser.parse_args()
    if args.bundle_root:
        raise SystemExit(run_bundle(args.bundle_root))
    if not args.baseline or not args.output:
        parser.error('준비에는 --baseline/--output 필요')
    report=stage(args.baseline,args.output)
    print(json.dumps(dict(prepared=args.output.name,attempts=[a['root'] for a in report['attempts']],
                         final_goal='NOT_MET'),ensure_ascii=False))
