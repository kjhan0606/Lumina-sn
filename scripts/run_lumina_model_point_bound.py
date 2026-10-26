#!/usr/bin/env python3
"""동결 자체 J의 원래점 하나에서 전ν bound PHOT 율을 제한 시간 안에 생산한다.

새 수송/CMFGEN formal/물질 갱신은 없다. 한도 종료는 완료 route만 보존하고
전체 snapshot은 INCOMPLETE다. 자동 재시도나 시간 연장은 없다.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

import cmfgen_model_continuum as B
import lumina_native_columns as N
import lumina_native90_rates as R
from lumina_model_point_rates import ModelPointBoundRates


def run(args):
    start=time.monotonic();deadline=start+260.;out=Path(args.out)
    if out.exists():raise FileExistsError('기존 전ν율 결과 보존')
    if os.environ.get('SLURM_CPUS_PER_TASK')!='1' or os.environ.get('SLURM_MEM_PER_NODE')!='4096':
        raise ValueError('CPU1/4GiB Slurm 직접시험 필요')
    out.mkdir()
    source_files=[Path(__file__),Path(__file__).with_name('lumina_model_point_rates.py'),
                  Path(B.__file__),Path(N.__file__),Path(R.__file__),Path(args.library)]
    identities={str(p):B.sha(p) for p in source_files}
    (out/'source-before.json').write_text(json.dumps(identities,indent=2)+'\n')
    base_path=Path(args.base)
    if B.sha(base_path)!='f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab':
        raise ValueError('다른 native 계획')
    base=json.loads(base_path.read_text());c=json.loads(Path(base['config']).read_text())
    plan=json.loads(Path(c['plan']).read_text())
    report_path=Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
    if B.sha(report_path)!=c['reuse_sources']['boundary_sources']['report_sha256']:
        raise ValueError('계수 report 신원 변경')
    report=json.loads(report_path.read_text());node=49
    with N.open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as source:
        rows,models,routes,state,do_dis,full_check,smoothing=R.material_state(c,plan,source,report)
        nu=np.array(source.frequency,copy=True)
        with N.FrozenNativeField(args.j,source,source.radius,source.frequency) as field:
            chunks=[]
            for begin in range(0,len(nu),65536):
                indices=np.arange(begin,min(len(nu),begin+65536),dtype='i8')
                chunks.append(field.read(np.array([node]),indices)[0])
            j=np.concatenate(chunks)
        source_identity=dict(source.identity)
    widths=np.zeros_like(nu);delta=np.diff(nu)
    widths[:-1]+=.5*delta;widths[1:]+=.5*delta
    producer=ModelPointBoundRates(routes,rows,models,state,node,Path(args.library).resolve())
    values=np.full((len(routes),6),np.nan);done=np.zeros(len(routes),dtype=bool)
    reason=None;active_route=0;visited=0
    try:
        for index,r in enumerate(producer.routes):
            active_route=index
            if time.monotonic()>=deadline:raise TimeoutError('260s including preparation')
            # 원 PHOT의 정확히0인 지지 밖만 생략한다. 실제 격자/가중치는 불변.
            support=min((entry[2] for entry in r.models),default=np.inf)
            begin=int(np.searchsorted(nu,support,side='left'))
            total=np.zeros(6,dtype=np.longdouble)
            for lo in range(begin,len(nu),16384):
                if time.monotonic()>=deadline:raise TimeoutError('260s including preparation')
                hi=min(len(nu),lo+16384)
                total+=producer.contract_route_block(index,nu[lo:hi],widths[lo:hi],j[lo:hi])
                visited+=hi-lo
            values[index]=total.astype('f8')
            if not np.isfinite(values[index]).all():raise ValueError('전ν율 수치 범위 초과')
            done[index]=True
            if (index+1)%250==0:
                print(json.dumps(dict(routes_complete=int(done.sum()),routes_expected=len(routes),
                    elapsed_seconds=time.monotonic()-start)),flush=True)
    except TimeoutError as error:reason=str(error)
    except Exception as error:reason=type(error).__name__+': '+str(error)
    np.save(out/'unit-rates.npy',values);np.save(out/'routes-complete.npy',done)
    np.savez(out/'roster.npz',lower=np.array([r.lower for r in producer.routes]),
        target=np.array([r.target for r in producer.routes]),Z=np.array([r.Z for r in producer.routes]),
        stage=np.array([r.stage for r in producer.routes]),probability=np.array([r.probability for r in producer.routes]),
        threshold=np.array([r.threshold_Hz for r in producer.routes]),saha=producer.saha,sigma_edge=producer.sigma_edge)
    (out/'numeric-completion.json').write_text(json.dumps(dict(routes_complete=int(done.sum()),
        routes_expected=len(routes),stop_reason=reason,wall_seconds=time.monotonic()-start),indent=2)+'\n')
    source_unchanged=all(B.sha(Path(p))==digest for p,digest in identities.items())
    complete=bool(done.all() and reason is None and source_unchanged)
    result=dict(state='COMPLETE_BOUND_ONLY' if complete else 'INCOMPLETE',
        native_node=node,frequencies=len(nu),routes_expected=len(routes),routes_complete=int(done.sum()),
        active_route=active_route,visited_route_frequency_values=visited,stop_reason=reason,
        source_unchanged=source_unchanged,source_sha256=identities,input_identity=source_identity,
        full_population_check=full_check,pseudo_included=False,BB_included=False,
        complete_bound_frequency_integral=complete,SE_RE_solution=False,
        actual_new_transport=0,material_updates=0,wall_seconds=time.monotonic()-start,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('state','routes_complete','routes_expected','stop_reason','wall_seconds')},ensure_ascii=False),flush=True)
    return 0 if complete else 2


def recover_saved(out):
    """수치 배열을 바꾸거나 재계산하지 않고 불완전 handoff의 범위만 기록한다."""
    out=Path(out);report=out/'recovery.json'
    if report.exists():raise FileExistsError('기존 복구 기록 보존')
    values=np.load(out/'unit-rates.npy',allow_pickle=False)
    done=np.load(out/'routes-complete.npy',allow_pickle=False)
    with np.load(out/'roster.npz',allow_pickle=False) as roster:
        if values.shape!=(len(done),6) or done.dtype!=np.dtype(bool) or any(
            len(roster[key])!=len(done) for key in roster.files):raise ValueError('부분 출력 차원 손상')
        if not np.isfinite(values[done]).all() or np.any(values[done,:3]<0):
            raise ValueError('완료 표지의 비유한/음수 율')
        ions=[]
        for Z,stage in sorted(set(zip(roster['Z'].tolist(),roster['stage'].tolist()))):
            mask=(roster['Z']==Z)&(roster['stage']==stage)
            ions.append(dict(Z=Z,stage=stage,expected=int(mask.sum()),completed=int(done[mask].sum())))
    result=dict(state='INCOMPLETE_RECOVERED_NUMERIC_ONLY',routes_expected=len(done),
        routes_complete=int(done.sum()),by_ion=ions,original_stop_reason='UNKNOWN_FINAL_METADATA_FAILED',
        final_metadata_error='B.sha received str instead of Path',
        start_manifest_available=(out/'source-before.json').exists(),
        source_identity_at_start='NOT_RECOVERABLE_WITHOUT_START_MANIFEST',
        numerical_reexecution=0,numerical_files_modified=0,
        artifact_sha256={name:B.sha(out/name) for name in ('unit-rates.npy','routes-complete.npy','roster.npz')},
        complete_bound_frequency_integral=False,SE_RE_solution=False,final_goal='NOT_MET')
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(state=result['state'],routes_complete=result['routes_complete'],routes_expected=len(done))))
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base')
    p.add_argument('--j');p.add_argument('--library');p.add_argument('--out',required=True)
    p.add_argument('--recover-saved',action='store_true');args=p.parse_args()
    if args.recover_saved:raise SystemExit(recover_saved(args.out))
    if not all((args.base,args.j,args.library)):p.error('--base/--j/--library 필요')
    raise SystemExit(run(args))
