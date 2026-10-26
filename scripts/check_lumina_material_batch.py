#!/usr/bin/env python3
"""실제27이온 대표 PHOT 경로·전90점에서 다점 BF와 기존 단일점 적분을 대조한다.

전경로/전원소 SE·전하·RE 또는 새 수송의 완료 시험은 아니다.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import cmfgen_model_continuum as B
import lumina_native_columns as N
import lumina_native90_rates as R
from lumina_model_material_rates import ModelMaterialRates
from lumina_model_point_rates import ModelPointBoundRates
from lumina_model_point_reference import ReferencePointRates
from compare_native_prrr_common_grid import continuum_mapping
from run_cmfgen_formal_primary import fields


def run(args):
    if os.environ.get('SLURM_CPUS_PER_TASK')!='1' or os.environ.get('SLURM_MEM_PER_NODE')!='4096':
        raise ValueError('CPU1·4GiB Slurm 직접시험 전용')
    started=time.monotonic();out=args.out
    if out.exists():
        raise FileExistsError('기존 다점 시험 보존')
    out.mkdir()
    project=Path(__file__).resolve().parents[1]
    base_path=Path('/gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json')
    if B.sha(base_path)!='f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab':
        raise ValueError('원 물질70 입력 계획 변경')
    base=json.loads(base_path.read_text());c=json.loads(Path(base['config']).read_text())
    plan=json.loads(Path(c['plan']).read_text())
    report_path=Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
    if B.sha(report_path)!=c['reuse_sources']['boundary_sources']['report_sha256']:
        raise ValueError('원 계수 보고서 신원 변경')
    field_path=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    native=Path('/gpfs/kjhan/cmfgen_runs/lumina_fixedj_exp70_20261026-phmMnx/run/material')
    paths=[Path(__file__),Path(args.library),base_path,Path(base['config']),Path(c['plan']),report_path,
           native/'CONT_FREQ',field_path,*[project/name for name in
           ('src/bf_rate_jnu.c','src/bf_rate_jnu.h','scripts/lumina_model_material_rates.py',
            'scripts/lumina_model_point_rates.py','scripts/lumina_model_point_reference.py')]]
    identities={str(p):B.sha(p) for p in paths}
    if identities[str(field_path)]!=plan['reference_field_sha256']:
        raise ValueError('원 물질과 같은 J70 아님')
    (out/'inputs.json').write_text(json.dumps(dict(input_sha256=identities,scope=__doc__,
        resources='CPU1/4GiB/10min/retry0; 대표 경로 적분/단일점 대조; 신규 수송/native0'),indent=2)+'\n')
    geometry,field=fields(field_path)
    try:
        with N.open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as source:
            rows,models,routes,state,do_dis,full_check,smoothing=R.material_state(c,plan,source,json.loads(report_path.read_text()))
            if source.identity!=base['source_identity'] or not np.array_equal(geometry[0]*1.e10,source.radius):
                raise ValueError('원90점 물질/복사장 반경 오류')
        ordered=sorted(routes,key=lambda r:(r.lower,r.phot_grid,r.target_superlevel))
        indices=[]
        for key in sorted(models):
            candidates=[i for i,r in enumerate(ordered) if (r.Z,r.stage)==key and r.phot_grid==0]
            if not candidates:
                raise ValueError('활성 이온의 primary PHOT 누락')
            indices.append(candidates[0])
        for key in ((14,2),(16,2)):
            candidates=[i for i,r in enumerate(ordered) if (r.Z,r.stage)==key and r.phot_grid==0 and
                        float(r.sigma(np.array([r.threshold_Hz*(1+1.e-14)]))[0])>0]
            if not candidates:
                raise ValueError('용해 대표의 양의 primary 단면 없음')
            indices.append(min(candidates,key=lambda i:ordered[i].threshold_Hz))
        indices=sorted(set(indices));selected=[ordered[i] for i in indices]
        nu=np.array(field['freq'][::-1]*1.e15)
        j=np.array(field['field'][::-1,0,:].T,order='C')
        evaluation=continuum_mapping(native/'CONT_FREQ',nu)
        width=np.zeros_like(nu);width[:-1]+=.5*np.diff(nu);width[1:]+=.5*np.diff(nu)
        if (len(models),j.shape)!=(27,(90,197291)):
            raise ValueError('실제27이온·전90점·원 J70 격자 아님')
        provider=ModelMaterialRates(selected,rows,models,args.library,nu,width,j,
            radiation_generation=70,evaluation=evaluation,dissolution_enabled=do_dis)
        trial=provider.prepare(state,material_generation=1)
        begin=time.monotonic()
        actual=provider.integrate(trial,material_generation=1,radiation_generation=70,route_block=32)
        batch_seconds=time.monotonic()-begin
        np.savez(out/'original-batch.npz',bound=actual[0],total=actual[1],route_indices=indices)
        print(json.dumps(dict(phase='BATCH_COMPLETE',routes=len(selected),points=90,seconds=batch_seconds)),flush=True)
        expected=[np.zeros_like(actual[0]),np.zeros_like(actual[1])]
        begin=time.monotonic()
        for node in range(90):
            point=ModelPointBoundRates(selected,rows,models,trial.state,node,args.library)
            scalar=ReferencePointRates(point)
            for r in range(len(selected)):
                values=scalar.contract(r,nu,width,j[node],evaluation,dissolution_enabled=do_dis)
                for accumulator,value in zip(expected,values):
                    accumulator[r,node]=value
            if node%15==14:
                print(json.dumps(dict(phase='SCALAR_CHECK',points_complete=node+1)),flush=True)
        scalar_seconds=time.monotonic()-begin
        comparisons=[]
        for name,a,e in zip(('bound','bound_plus_pseudo'),actual,expected):
            if np.any((a==0)!=(e==0)) or not np.allclose(a,e,rtol=2.e-11,atol=0.):
                raise ValueError('실물90점 배치/단일점 적분의 support 또는 성분 불일치')
            delta=np.divide(abs(a-e),abs(e),out=np.zeros_like(a),where=e!=0)
            comparisons.append(dict(name=name,values=a.size,max_relative=float(delta.max()),
                worst_index=list(map(int,np.unravel_index(np.argmax(delta),delta.shape)))))
        changed=dict(state)
        changed['superlevel']=state['superlevel'].copy()
        changed['superlevel'][:,49]*=1.03
        refreshed=provider.prepare(changed,material_generation=2)
        if np.array_equal(trial.state['occupation'][:,49],refreshed.state['occupation'][:,49]):
            raise ValueError('양이온 밀도 변경이 occupation에 반영되지 않음')
        try:
            provider.integrate(trial,material_generation=2,radiation_generation=70)
        except ValueError:
            stale_rejected=True
        else:
            raise ValueError('다른 trial 세대에서 구 율 소비 허용')
        # 새 trial 전체 적분을 중복하지 않고 같은 국소 블록으로 실제 율 갱신을 확인한다.
        eligible=[i for i in range(len(selected)) if provider.sigma_edge[i]>0 and
            nu[0]<provider.edge[i]<nu[-1] and
            trial.milne_prefactor[i,49]!=refreshed.milne_prefactor[i,49]]
        if not eligible:
            raise ValueError('선택 경로에 양이온 밀도 의존 역률 없음')
        lower=min(eligible,key=lambda i:selected[i].threshold_Hz)
        route=provider.routes[lower]
        start=max(0,int(np.searchsorted(nu,route.threshold_Hz))-8)
        end=min(len(nu),start+32,int(np.searchsorted(nu,nu[start]+32./float(trial.inverse_kT_Hz.max()),side='right')))
        before=provider.contract_block(trial,start,end,[lower],material_generation=1,radiation_generation=70)
        after=provider.contract_block(refreshed,start,end,[lower],material_generation=2,radiation_generation=70)
        if not np.array_equal(before[0][:,:,0],after[0][:,:,0]) or np.array_equal(before[1][:,49],after[1][:,49]):
            raise ValueError('고정 J의 bound Γ 불변/동적 역률 소비 오류')
        if any(B.sha(Path(p))!=h for p,h in identities.items()):
            raise ValueError('실물 다점 직접시험 중 입력/소스 변경')
        result=dict(schema='lumina-material-batch-direct-test-v1',state='COMPLETE_REPRESENTATIVE_ROUTES_90_POINTS',
            input_sha256=identities,source_identity=base['source_identity'],input_material=70,
            registered_field_signature=provider.field_signature,frequency_points=len(nu),material_points=90,
            model_ions=len(models),model_levels=len(rows),all_model_routes=len(routes),tested_routes=len(selected),
            tested_route_indices=indices,comparisons=comparisons,batch_seconds=batch_seconds,
            scalar_seconds=scalar_seconds,wall_seconds=time.monotonic()-started,
            trial_signatures=[trial.material_signature,refreshed.material_signature],
            Npositive_change_rebuilds_occupation=True,stale_trial_rejected=stale_rejected,
            source_unchanged=True,full_route_SE_charge_RE=False,public_material_updates=0,
            new_transport=0,new_native=0,final_goal='NOT_MET')
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(state=result['state'],batch_seconds=batch_seconds,scalar_seconds=scalar_seconds)),flush=True)
    finally:
        field._mmap.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
