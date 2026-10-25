#!/usr/bin/env python3
"""같은 봉인 입력·바이너리에서 주파수 묶음 완결 순서로 원13003을 실행한다."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

import lumina_local_micro as M
from lumina_compiled_input import CompiledInput
from lumina_gpu_compact import Runtime
from lumina_gpu_driver import Work
from lumina_gpu_hybrid import SlabLocalHybridDriver
from lumina_gpu_local import fine_comparison
from lumina_nodal_driver import CompiledDriverPlan

BASE = Path('/gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json')
BASE_SHA = 'f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab'


def run(root, seconds=720):
    if type(seconds) is not int or not 1 <= seconds <= 720:
        raise ValueError('원 구간 상한12분 이내의 양의 실행시간 필요')
    root = Path(root).resolve()
    output = root/'slab-local.json'
    if output.exists(): raise FileExistsError('기존 결과 보존')
    if M.N.sha(BASE) != BASE_SHA: raise ValueError('봉인 계획 변경')
    c = json.loads(BASE.read_text())
    for key in ('cpu_library','gpu_library'):
        if M.N.sha(c[key]) != c['sealed_sha256'][c[key]]:
            raise ValueError('기존 바이너리 신원 변경')
    start = time.monotonic()
    record = dict(status='STARTED', final_goal='NOT_MET', actual_model_F_calls=0,
        SE_RE_updates=0, input_plan_sha256=BASE_SHA, interval_index=13003,
        own_input='J0_I0_ZERO', completed_interval=False,
        limits=dict(CPU=1, GPU=1, host_GiB=16, device_GiB=16, workspace_GiB=1,
                    interval_seconds=seconds, parent_seconds=900, automatic_retries=0),
        input_hash_verification='원 봉인 및 open_sealed_native 계승; 대형 계수 본문 전체 재해시 안 함')
    driver = work = None

    def publish():
        record.update(elapsed_seconds=time.monotonic()-start,
                      peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if work is not None:
            record.update(parallel_work=work.used, work_counters=work.counts)
        temp = output.with_suffix('.partial')
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
        temp.replace(output)

    publish()
    try:
        with M.N.open_sealed_native(c['config'],c['preparation'],c['preparation_sha256']) as source:
            if source.identity != c['source_identity']: raise ValueError('원 계수 신원 변경')
            geometry = M.open_geometry(c['preparation'])
            plan = CompiledDriverPlan(geometry,source,panels=1,block_width=256)
            if plan.arrays['steps'][13003].tolist() != c['intervals'][0]['step']:
                raise ValueError('원13003 기하 변경')
            with CompiledInput(c['cpu_library'],geometry,source,M.ZeroTrial(source),
                               slab_width=128,cache_rows=128,profiles=1024) as provider:
                provider.bind_zero_trial()
                record['dense_preparation'] = provider.prepare_dense()
                smi = subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
                free = int(smi.stdout.strip())
                if free < 24*1024: raise RuntimeError('기존 장치 메모리 여유 기준 미달')
                library = root/'history_cuda.so'
                record['gpu_library_sha256'] = M.N.sha(library)
                record['gpu_library_scope'] = '같은 C 원천의 sm80 빌드; 이전 sm89 바이너리와 구분'
                with Runtime(library,free*1024**2) as runtime:
                    driver = SlabLocalHybridDriver(runtime,provider,c['cpu_library'],block_width=256)
                    try:
                        runtime.zero(driver.ray)
                        work = Work(c['limits']['local_full_vectors_total']*source.nf,start+seconds+30)
                        report = driver.interval(plan,13003,work,max_seconds=seconds)
                        values = runtime.download(driver.ray,source.nf)
                        actual = values[c['fine_columns']]
                        record.update(completed_interval=True, report=report,
                            fine_columns=c['fine_columns'],actual_values=actual.tolist(),
                            fine_values=c['fine_values'],
                            **fine_comparison(actual,np.array(c['fine_values']),each_output=True))
                        with (root/'interval-13003.npy').open('xb') as stream:
                            np.save(stream,values,allow_pickle=False)
                        record['output_sha256'] = M.N.sha(root/'interval-13003.npy')
                        record['status'] = 'INTERVAL_COMPLETE_NOT_FIELD'
                    finally:
                        record['last_completed_batch'] = driver.last_completed_batch
                        record['integration'] = driver.integration_report()
                        driver.close()
                        record['memory_end'] = runtime.report()
    except Exception as exc:
        record.update(status='INTERVAL_INCOMPLETE',error=type(exc).__name__+': '+str(exc))
    publish()
    return 0 if record['completed_interval'] and record['accuracy_met'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root')
    parser.add_argument('--seconds',type=int,default=720)
    args = parser.parse_args()
    raise SystemExit(run(args.root,args.seconds))
