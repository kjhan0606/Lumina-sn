#!/usr/bin/env python3
"""원13003의 coarse/두 half에서 고정35열 실패를 CPU로 재현한다."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import time

import numpy as np
import lumina_native_cost_diag as D
from lumina_gpu_driver import Leaf,Location,check
from lumina_gpu_linear import LinearDriver,Piece

OLD=Path('/gpfs/kjhan/cmfgen_runs/lumina_endpoint_cache_fixed_20261022-ClLEqp/cpu-probe.so')
NEW=Path('/gpfs/kjhan/cmfgen_runs/lumina_native_bin_budget_20261022-KdVw2W/cpu-probe.so')


def run(root,columns=None,*,global_coordinates=False,new_library=NEW,compare_original=False):
    root=Path(root).resolve();target=root/'cpu-three-pieces.json'
    if target.exists():raise FileExistsError('기존 세 조각 재현 보존')
    record=dict(status='STARTED',rows=[],GPU_seconds=0,final_goal='NOT_MET',
        coordinate_mode='ORIGINAL_GLOBAL_TREE' if global_coordinates else 'ANCHORED_LOCAL_TREE',
        compare_original=compare_original,
        purpose='같은 원13003 coarse/left/right의 고정35열. 전주파수 완성장이나 물리수용은 아니다.',
        source_sha256=D.sha(__file__),libraries={str(p):D.sha(p) for p in (OLD,new_library)})
    start=time.monotonic();D.publish(target,record)
    try:
        with D.inputs(root,record) as (provider,old,_coarse,_ends):
            columns=D.samples(provider.nf) if columns is None else columns
            if not 1<=len(columns)<=35 or len(set(columns))!=len(columns) or any(
                type(c) is not int or not 0<=c<provider.nf for c in columns):raise ValueError('최대35개 유효 출력열 필요')
            record['selected_columns']=columns
            import lumina_local_micro as M
            from lumina_nodal_driver import CompiledDriverPlan
            g=M.open_geometry(old['preparation']);plan=CompiledDriverPlan(g,provider._source,panels=1,block_width=256)
            helper=LinearDriver.__new__(LinearDriver);helper.provider=provider
            helper.geom=ct.CDLL(old['cpu_library']).cmf_nodal_driver_geometry
            helper.geom.argtypes=[ct.c_void_p,ct.c_int]+[ct.c_void_p]*5+[ct.c_double,ct.c_int]+[ct.c_void_p]*6
            helper.geom.restype=ct.c_int;geometry=helper.geometry(plan,13003,[Leaf(0.,1.)])
            planner=ct.CDLL(old['gpu_library']).lumina_linear_cuda_plan
            planner.argtypes=[ct.c_void_p]*2+[ct.c_double]*4+[ct.c_int]*3+[ct.c_void_p];planner.restype=ct.c_int
            functions={}
            for name,path in (('original',OLD),('native_bin',new_library)):
                f=getattr(ct.CDLL(str(path)),'native_cost_reference' if global_coordinates else 'native_cost_probe')
                f.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p];f.restype=ct.c_int
                functions[name]=f
            for k in range(3):
                loc,length,da,db=helper.piece(*geometry[:3],0,k);ends=(Location*2)(loc[0],loc[2]);piece=Piece()
                view=provider.borrow_dense_view()
                check(planner(ct.byref(view),ends,length,provider._source.dlognu,da,db,0,provider.nf,2,ct.byref(piece)))
                for column in columns:
                    if time.monotonic()-start>270:raise TimeoutError('세 조각 CPU 내부270초')
                    row=dict(piece=k,column=column,length=length,da=da,db=db,span=piece.span,offset=piece.offset)
                    new_map=None
                    # 먼저 새 후보를 조사하고 실패한 동일 위치만 원 적분기로 대조한다.
                    for name in ('native_bin','original'):
                        if name=='original' and row['native_bin']['code']==0 and not compare_original:continue
                        raw=D.NativeMap();stats=np.zeros(4,dtype='u8');times=np.full(2,-1.)
                        code=functions[name](ct.byref(piece),column,ct.byref(raw),stats.ctypes.data,times.ctypes.data)
                        item=dict(code=code,build_seconds=float(times[0]),query_seconds=float(times[1]))
                        if code==0:
                            item.update(t=D.scaled(raw.transmission),source=D.scaled(raw.source),error=D.scaled(raw.source_error),panels=raw.panels)
                            if name=='native_bin':new_map=raw
                        row[name]=item
                    if compare_original:row['comparison_met']=bool(new_map is not None and
                        row['original']['code']==0 and D.compare(new_map,row['original']))
                    record['rows'].append(row);D.publish(target,record)
            record['status']='CPU_THREE_PIECE_SAMPLE_FINISHED'
            record['new_failures']=sum(row['native_bin']['code']!=0 for row in record['rows'])
            return 0
    except Exception as error:
        record.update(status='CPU_THREE_PIECE_SAMPLE_STOPPED',error=type(error).__name__+': '+str(error));return 1
    finally:
        record['elapsed_seconds']=time.monotonic()-start;D.publish(target,record)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--column',type=int,action='append')
    p.add_argument('--global-coordinates',action='store_true')
    p.add_argument('--new-library',type=Path,default=NEW)
    p.add_argument('--compare-original',action='store_true')
    args=p.parse_args();raise SystemExit(run(args.root,args.column,global_coordinates=args.global_coordinates,
                                          new_library=args.new_library,compare_original=args.compare_original))
