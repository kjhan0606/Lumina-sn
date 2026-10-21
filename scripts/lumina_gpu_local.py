#!/usr/bin/env python3
"""봉인 native90 두 구간 GPU 대조. 완성장/SE/RE·반복은 실행하지 않는다."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

import lumina_local_micro as M
from lumina_compiled_input import CompiledInput
from lumina_gpu_driver import Runtime,GpuDriver,Work
from lumina_nodal_driver import CompiledDriverPlan

ORIGIN=Path('/gpfs/kjhan/cmfgen_runs/lumina_native90_pair_20261021-e7DQZB/native-contract.json')
ORIGIN_SHA='0fe9718063d6d1ba46a1dbb32e4ccace4430f8d4f7040ef0be86ee3a025ac598'
REFERENCE=Path('/gpfs/kjhan/cmfgen_runs/lumina_cached_actual_20261021-2dCTQS/micro-result.json')
REFERENCE_SHA='44283d6c8584225cd7be731a40997b764c9f7d7aa4a7e824162e9bc6f7aea785'
BASELINE=Path('/gpfs/kjhan/cmfgen_runs/lumina_native90_basis_20261021-Tg3IOw/interval0-completed.npy')
BASELINE_SHA='be8c99316b242b8e8ea2b394b461e0b65ee0171219abb5f342e4fa276153a55a'
LIMITS=dict(cpus=4,host_GiB=16,device_GiB=16,cache_GiB=1,GPU=1,device_index=0,
    prep_window_seconds=1800,local_internal_seconds=900,interval_seconds=60,local_calls=2,
    local_full_vectors_total=512,cost_target_seconds=10,output_GiB=10,automatic_retries=0,
    actual_model_F_calls=0,SE_RE_updates=0)
SCHEMA='lumina-gpu-native-local-v1'


def prepare(root,*,extra_files=(),interval_indices=(0,13003),comparison_context=None):
    root=Path(root).resolve();out=root/'gpu-local-plan.json'
    if out.exists():raise FileExistsError('기존 GPU 계획 보존')
    interval_indices=tuple(interval_indices)
    if not interval_indices or len(set(interval_indices))!=len(interval_indices) or any(type(i) is not int or i not in (0,13003) for i in interval_indices):
        raise ValueError('원래 승인된0/13003의 중복 없는 부분집합만 허용')
    if len(interval_indices)!=LIMITS['local_calls']:raise ValueError('봉인 구간 수/호출 한도 불일치')
    for path,digest in ((ORIGIN,ORIGIN_SHA),(REFERENCE,REFERENCE_SHA),(BASELINE,BASELINE_SHA)):
        if M.N.sha(path)!=digest:raise ValueError('기존 입력/공간/전ν 참조 신원 변경')
    original=json.loads(ORIGIN.read_text());ref=json.loads(REFERENCE.read_text())
    reference=next(x for x in ref['results'] if x['specification']['index']==13003)
    if reference['spatial_reference']['self_mesh_status']!='MET':raise ValueError('자기 미세화 참조 미충족')
    with M.N.open_sealed_native(original['config'],original['preparation'],original['preparation_sha256']) as source:
        if source.identity!=original['source_identity'] or (source.ns,source.nf)!=(90,1115020):raise ValueError('native90 입력 변경')
        geometry=M.open_geometry(original['preparation']);plan=CompiledDriverPlan(geometry,source,panels=1,block_width=256)
        if reference['specification']['step']!=plan.arrays['steps'][13003].tolist():raise ValueError('공간 참조 기하 변경')
        record=dict(schema=SCHEMA,origin=str(ORIGIN),origin_sha256=ORIGIN_SHA,
            config=original['config'],preparation=original['preparation'],preparation_sha256=original['preparation_sha256'],
            source_identity=source.identity,input_sha256=original['input_sha256'],limits=LIMITS,
            cpu_library=str(root/'driver.so'),gpu_library=str(root/'history_cuda.so'),
            baseline=str(BASELINE),baseline_sha256=BASELINE_SHA,reference=str(REFERENCE),reference_sha256=REFERENCE_SHA,
            fine_columns=reference['spatial_reference']['columns'],fine_values=reference['spatial_reference']['fine'],
            intervals=[dict(index=i,step=plan.arrays['steps'][i].tolist()) for i in interval_indices],
            own_input='EXPLICIT_J0_I0_ONLY_NOT_A_COMPLETE_FIELD',accepted_physical_field=False,final_goal='NOT_MET')
    if comparison_context is not None:record['comparison_context']=comparison_context
    files=[root/'driver.so',root/'history_cuda.so',root/'compile.sh',root/'actual.sh']
    files.extend(Path(p) if Path(p).is_absolute() else root/p for p in extra_files)
    files.extend(p for folder in ('src','scripts','tests') for p in sorted((root/folder).iterdir()) if p.is_file())
    record['sealed_sha256']={str(path):M.N.sha(path) for path in files}
    with out.open('x') as f:json.dump(record,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(M.N.sha(out),flush=True)


def run(path,digest):
    path=Path(path).resolve();root=path.parent
    if M.N.sha(path)!=digest:raise ValueError('GPU 실행 계획 변경')
    c=json.loads(path.read_text())
    if c['schema']!=SCHEMA or c['limits']!=LIMITS:raise ValueError('GPU 실행 범위 변경')
    for group in ('sealed_sha256','input_sha256'):
        if any(M.N.sha(Path(p))!=h for p,h in c[group].items()):raise ValueError('GPU 봉인 입력/코드 변경')
    for key in ('origin','baseline','reference'):
        if M.N.sha(Path(c[key]))!=c[key+'_sha256']:raise ValueError('GPU 참조 신원 변경')
    if (root/'gpu-local.json').exists():raise FileExistsError('기존 GPU 결과 보존')
    with (root/'gpu-local.started').open('x') as f:f.write(digest+'\n')
    start=time.monotonic();record=dict(status='STARTED',plan_sha256=digest,results=[],actual_model_F_calls=0,
        SE_RE_updates=0,accepted_physical_field=False,final_goal='NOT_MET');work=None;driver=None
    def publish():
        record.update(elapsed_seconds=time.monotonic()-start,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if work is not None:record.update(parallel_work=work.used,work_counters=work.counts,
            carried_parallel_work=LIMITS.get('carried_parallel_work',0),
            combined_parallel_work=LIMITS.get('carried_parallel_work',0)+work.used)
        temporary=root/'gpu-local.partial';temporary.write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False))
        temporary.replace(root/'gpu-local.json')
    publish();code=1
    try:
        with M.N.open_sealed_native(c['config'],c['preparation'],c['preparation_sha256']) as source:
            if source.identity!=c['source_identity']:raise ValueError('실제 계수 입력 변경')
            g=M.open_geometry(c['preparation']);plan=CompiledDriverPlan(g,source,panels=1,block_width=256)
            field=M.ZeroTrial(source)
            with CompiledInput(c['cpu_library'],g,source,field,slab_width=128,cache_rows=128,profiles=1024) as provider:
                provider.bind_zero_trial();record['dense_preparation']=provider.prepare_dense();publish()
                smi=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],
                    check=True,capture_output=True,text=True,timeout=15).stdout.strip()
                free,util=[int(x.strip()) for x in smi.split(',')]
                if free<24*1024 or util>5:raise RuntimeError('GPU 부하/메모리 여유 부족')
                record['pre_gpu_memory_free_MiB']=free
                with Runtime(c['gpu_library'],free*1024**2) as runtime:
                    driver=GpuDriver(runtime,provider,c['cpu_library']);record['memory_before_local']=runtime.report();publish()
                    total_work=LIMITS['local_full_vectors_total']*source.nf;carried=LIMITS.get('carried_parallel_work',0)
                    if type(carried) is not int or carried<0 or carried>=total_work:raise ValueError('이월 작업량/원 예산 불일치')
                    work=Work(total_work-carried,start+LIMITS['local_internal_seconds'])
                    for spec in c['intervals']:
                        index=spec['index'];record['phase']='INTERVAL_'+str(index);publish()
                        if plan.arrays['steps'][index].tolist()!=spec['step']:raise ValueError('봉인 구간 변경')
                        runtime.zero(driver.ray);t=time.monotonic()
                        report=driver.interval(plan,index,work,max_seconds=LIMITS['interval_seconds'])
                        output=runtime.download(driver.ray,source.nf);seconds=time.monotonic()-t
                        entry=dict(index=index,completion_seconds=seconds,report=report)
                        if index==0:
                            baseline=np.load(c['baseline'],allow_pickle=False)
                            gap=float(abs(output-baseline).max()/abs(baseline).max())
                            entry.update(baseline_max_norm_difference=gap,accuracy_met=gap<=3e-12)
                        else:
                            actual=output[c['fine_columns']];fine=np.array(c['fine_values'])
                            gap=float(abs(actual-fine).max()/abs(fine).max())
                            entry.update(fine_columns=c['fine_columns'],fine_values=fine.tolist(),actual_values=actual.tolist(),
                                fine_max_norm_difference=gap,accuracy_met=gap<=1e-4)
                        out=root/f'gpu-interval-{index}.npy'
                        with out.open('xb') as f:np.save(f,output,allow_pickle=False)
                        entry['output']=dict(path=str(out),sha256=M.N.sha(out));entry['cost_met']=seconds<=10.
                        record['results'].append(entry);record['memory_after_local']=runtime.report();publish()
                        if not entry['accuracy_met'] or not entry['cost_met']:raise RuntimeError('실제 GPU 구간 정확도/10초 비용 미달')
                    driver.close();driver=None
                    record['memory_end']=runtime.report()
                record['status']='GPU_LOCAL_MET_NOT_FIELD';code=0
    except Exception as error:
        record.update(status='GPU_LOCAL_NOT_MET',error=type(error).__name__+': '+str(error))
        if driver is not None and hasattr(driver,'last_completed_batch'):
            record['last_completed_batch']=driver.last_completed_batch
    finally:
        record['phase']='finished';publish()
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'));parser.add_argument('path');parser.add_argument('--sha256')
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.path)
    else:raise SystemExit(run(args.path,args.sha256))
