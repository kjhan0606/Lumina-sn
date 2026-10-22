#!/usr/bin/env python3
"""원 실제 coarse의 고정 표본/제한 타일 비용 진단. 전체 수송/물리 수용은 하지 않는다."""
import argparse
from contextlib import contextmanager
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

from lumina_gpu_driver import Leaf,Location,check
from lumina_gpu_linear import LinearDriver,Piece
from lumina_gpu_compact import Runtime,CompactDriver,CompactMap,MAP_BYTES
from lumina_linear_native import NativeMap,Positive

BASELINE=Path('/gpfs/kjhan/cmfgen_runs/lumina_compact_native_doubletime_20261022-nd9sC6/gpu-local-plan.json')
BASELINE_SHA='9d978eecfea7c0c4318b73886d90bfdee99d38c81f626e600e385a2e1dfcc945'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def publish(path,record):
    record['peak_rss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False))
    temporary.replace(path)


def samples(nf):
    return sorted(set([round(k*(nf-1)/31) for k in range(32)]+[624712,624968,625223]))


def scaled(raw):
    p=raw.positive();return dict(m=p.m,e=p.e)


@contextmanager
def inputs(root,record):
    import lumina_local_micro as M
    from lumina_compiled_input import CompiledInput
    from lumina_nodal_driver import CompiledDriverPlan
    if sha(BASELINE)!=BASELINE_SHA:raise ValueError('원 실제 입력 계획 변경')
    old=json.loads(BASELINE.read_text())
    for group in ('input_sha256','sealed_sha256'):
        if any(sha(p)!=h for p,h in old[group].items()):raise ValueError('원 입력/고정 경로 변경')
    with M.N.open_sealed_native(old['config'],old['preparation'],old['preparation_sha256']) as source:
        if source.identity!=old['source_identity'] or source.nf!=1115020:raise ValueError('원 native 신원 변경')
        g=M.open_geometry(old['preparation']);plan=CompiledDriverPlan(g,source,panels=1,block_width=256)
        if plan.arrays['steps'][13003].tolist()!=old['intervals'][0]['step']:raise ValueError('원 기하 변경')
        with CompiledInput(old['cpu_library'],g,source,M.ZeroTrial(source),slab_width=128,cache_rows=128,profiles=1024) as provider:
            provider.bind_zero_trial();record['dense_preparation']=provider.prepare_dense()
            helper=LinearDriver.__new__(LinearDriver);helper.provider=provider
            cpu=ct.CDLL(old['cpu_library']);helper.geom=cpu.cmf_nodal_driver_geometry
            helper.geom.argtypes=[ct.c_void_p,ct.c_int]+[ct.c_void_p]*5+[ct.c_double,ct.c_int]+[ct.c_void_p]*6
            helper.geom.restype=ct.c_int
            geometry=helper.geometry(plan,13003,[Leaf(0.,1.)])
            loc,length,da,db=helper.piece(*geometry[:3],0,0)
            ends=(Location*2)(loc[0],loc[2]);view=provider.borrow_dense_view()
            # 장치 대조에서는 같은 아키텍처로 봉인한 현재 라이브러리의 host
            # 계획 함수를 쓴다. CPU-only 준비에는 원 계획 바이너리를 재사용한다.
            planner=root/'diagnostic.so'
            if not planner.is_file():planner=Path(old['gpu_library'])
            record['planning_library_sha256']=sha(planner)
            lib=ct.CDLL(str(planner));f=lib.lumina_linear_cuda_plan
            f.argtypes=[ct.c_void_p]*2+[ct.c_double]*4+[ct.c_int]*3+[ct.c_void_p];f.restype=ct.c_int
            piece=Piece();check(f(ct.byref(view),ends,length,source.dlognu,da,db,0,source.nf,2,ct.byref(piece)))
            record.update(source_identity=source.identity,interval=13003,coarse_only=True,full_frequency=False,
                          piece=dict(length=length,da=da,db=db,offset=piece.offset,span=piece.span,
                                     base=piece.base,nf=piece.nf,leaves=piece.leaves,source_leaves=piece.source_leaves))
            yield provider,old,piece,ends


def cpu(root):
    target=root/'cpu-cost.json'
    if target.exists():raise FileExistsError('기존 CPU 진단 보존')
    record=dict(status='STARTED',baseline_sha256=BASELINE_SHA,rows=[],GPU_seconds=0,final_goal='NOT_MET')
    started=time.monotonic();publish(target,record)
    try:
        lib=ct.CDLL(str(root/'cpu-probe.so'));f=lib.native_cost_probe
        f.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p];f.restype=ct.c_int
        record['library_sha256']=sha(root/'cpu-probe.so')
        with inputs(root,record) as (provider,_old,piece,_ends):
            record['columns']=samples(provider.nf);publish(target,record)
            for column in record['columns']:
                if time.monotonic()-started>270:raise TimeoutError('CPU 진단 내부270초')
                raw=NativeMap();stats=np.zeros(4,dtype='u8');times=np.full(2,-1.)
                record['phase']='COLUMN_'+str(column);publish(target,record);t=time.monotonic()
                code=f(ct.byref(piece),column,ct.byref(raw),stats.ctypes.data,times.ctypes.data)
                row=dict(column=column,code=code,wall_seconds=time.monotonic()-t,
                         build_seconds=None if times[0]<0 else float(times[0]),
                         query_seconds=None if times[1]<0 else float(times[1]),
                         reader_total=int(stats[0]),reader_build=int(stats[1]),reader_query=int(stats[0]-stats[1]),
                         tree_bytes=int(stats[2]),anchor=int(stats[3]))
                if code==0:
                    row.update(t=scaled(raw.transmission),source=scaled(raw.source),error=scaled(raw.source_error),
                               panels=raw.panels,accepted=raw.accepted,split=raw.split,pruned=raw.pruned,depth=raw.max_depth)
                else:row.update(panels=None,source=None,error=None)
                record['rows'].append(row);publish(target,record)
            record['status']='CPU_SAMPLE_FINISHED_WITH_MAP_FAILURES' if any(x['code'] for x in record['rows']) else 'CPU_SAMPLE_FINISHED_NO_MAP_FAILURES'
        return 0
    except Exception as error:
        record.update(status='CPU_SAMPLE_STOPPED',error=type(error).__name__+': '+str(error));return 1
    finally:
        record['elapsed_seconds']=time.monotonic()-started;publish(target,record)


def less_equal(a,b):
    return a.m==0 or (b.m!=0 and (a.e<b.e or (a.e==b.e and a.m<=b.m)))


def compare(raw,expected):
    actual_t=raw.transmission.positive();actual_s=raw.source.positive();actual_e=raw.source_error.positive()
    wanted=lambda name:Positive(expected[name]['m'],expected[name]['e'])
    t,s,e=[wanted(k) for k in ('t','source','error')]
    return (less_equal(actual_t.difference(t),t*3e-10) and less_equal(actual_s.difference(s),s*3e-10) and
            less_equal(actual_s.difference(s),actual_e+e) and less_equal(actual_e,actual_s*1e-8))


def gpu(root):
    target=root/'gpu-cost.json'
    if target.exists():raise FileExistsError('기존 GPU 진단 보존')
    baseline=json.loads((root/'cpu-cost.json').read_text())
    if baseline['status'] not in ('CPU_SAMPLE_FINISHED_NO_MAP_FAILURES','CPU_SAMPLE_FINISHED_WITH_MAP_FAILURES'):
        raise ValueError('완료된 고정 CPU 표본 필요')
    record=dict(status='STARTED',baseline_sha256=BASELINE_SHA,cpu_evidence_sha256=sha(root/'cpu-cost.json'),
                stages=[],comparisons=[],query_columns=0,performance_qualified=False,final_goal='NOT_MET')
    started=time.monotonic();publish(target,record);driver=None
    try:
        row=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],
            check=True,capture_output=True,text=True,timeout=15).stdout.strip().split(',')
        free,util=[int(v) for v in row]
        if free<24*1024:raise MemoryError('원 free24GiB 조건 미충족')
        record.update(precheck_free_MiB=free,precheck_utilization=util,library_sha256=sha(root/'diagnostic.so'))
        with inputs(root,record) as (provider,old,_host,ends):
            if record['piece']!=baseline['piece']:raise ValueError('원 CPU coarse와 host 계획 범위 불일치')
            with Runtime(root/'diagnostic.so',free*1024**2) as runtime:
                driver=CompactDriver(runtime,provider,old['cpu_library'])
                p=Piece();part=record['piece']
                check(runtime.lib.lumina_linear_cuda_plan(ct.byref(driver.view),ends,part['length'],provider._source.dlognu,
                    part['da'],part['db'],0,provider.nf,2,ct.byref(p)))
                allocation=driver.arena.take(provider.nf*MAP_BYTES)
                if allocation is None:raise MemoryError('고정 전역 map arena 한도')
                offset,maps=allocation
                record['memory_before']=runtime.report();publish(target,record)
                def timed(label,call):
                    if time.monotonic()-started>135:raise TimeoutError('장치 진단 내부135초')
                    record['phase']=label;publish(target,record);t=time.monotonic()
                    check(call());runtime.sync()
                    record['stages'].append(dict(phase=label,seconds=time.monotonic()-t));publish(target,record)
                timed('TREE_BUILD',lambda:runtime.lib.lumina_linear_cuda_build(ct.byref(p),driver.tree.address,runtime.status))
                f=runtime.lib.lumina_linear_cuda_query_range
                f.argtypes=[ct.c_void_p]*2+[ct.c_double]+[ct.c_int]*3+[ct.c_void_p]*2;f.restype=ct.c_int
                expected={x['column']:x for x in baseline['rows']}
                def query(first,count):
                    if record['query_columns']+count>1024:raise ValueError('진단 query 합계1024열 한도')
                    record['query_columns']+=count
                    timed(f'QUERY_{first}_{count}',lambda:f(ct.byref(p),driver.tree.address,1e-8,100000,first,count,maps.address,runtime.status))
                    raw=runtime.download(maps.part(first*MAP_BYTES,count*MAP_BYTES),(count,MAP_BYTES),'u1')
                    for column in sorted(k for k in expected if first<=k<first+count):
                        e=expected[column]
                        if e['code']!=0:continue
                        value=CompactMap.from_buffer_copy(raw[column-first].tobytes());met=compare(value,e)
                        record['comparisons'].append(dict(column=column,met=met,first=first,count=count,
                            actual_t=scaled(value.transmission),actual_source=scaled(value.source),actual_error=scaled(value.source_error)))
                        if not met:raise ValueError('같은 coarse CPU/GPU 수치 기준 미달')
                    publish(target,record)
                for column in baseline['columns']:query(column,1)
                for first,count in [(0,32),(624704,32),(624712,256),(624712,512)]:query(first,count)
                driver.arena.release(offset);driver.close();driver=None
                record['memory_end']=runtime.report()
        record['status']='LIMITED_GPU_COST_DIAGNOSIS_FINISHED_NOT_FIELD';return 0
    except Exception as error:
        record.update(status='LIMITED_GPU_COST_DIAGNOSIS_STOPPED',error=type(error).__name__+': '+str(error));return 1
    finally:
        record['elapsed_seconds']=time.monotonic()-started;publish(target,record)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu']);p.add_argument('root',type=Path);args=p.parse_args()
    root=args.root.resolve()
    raise SystemExit(cpu(root) if args.mode=='cpu' else gpu(root))
