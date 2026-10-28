#!/usr/bin/env python3
"""같은 F2 물질·격자에서 원장과 F2장의 한 이온 BF 흐름을 비교한다."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import cmfgen_model_continuum as B
from lumina_f2_node_rates import reconstruct
from lumina_model_material_rates import ModelMaterialRates
from run_cmfgen_formal_primary import fields
from run_lumina_full_material_response import BASE


def run(node,z,stage,out):
    start=time.monotonic();out=Path(out)
    if out.exists():raise FileExistsError('기존 실행 보존')
    out.mkdir();r=reconstruct(node);p=r['provider'];trial=r['trial'];roster=r['roster']
    base=json.loads(BASE.read_text());config=json.loads(Path(base['config']).read_text())
    plan=json.loads(Path(config['plan']).read_text());path=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if B.sha(path)!=plan['reference_field_sha256']:raise ValueError('CMFGEN 원장 변경')
    geometry,original=fields(path)
    try:
        old_nu=np.asarray(original['freq'][::-1])*1e15;old_j=np.asarray(original['field'][::-1,0,node])
        nu=p.frequency
        if nu[0]<old_nu[0] or nu[-1]>old_nu[-1] or np.any(np.diff(old_nu)<=0):raise ValueError('원장 주파수 범위 밖 보간')
        if not np.array_equal(geometry[0]*1e10,r['state']['physical'][:,0]):raise ValueError('원장과 F2 물질 반경 불일치')
        old_interpolated=np.interp(np.log(nu),np.log(old_nu),old_j)
    finally:original._mmap.close()
    old=ModelMaterialRates(p.routes,p.rows,p.models,r['library'],nu,p.width,old_interpolated[None,:],
        radiation_generation=1,dissolution_enabled=p.dissolved)
    old_trial=old.prepare({k:trial.state[k] for k in ('columns','superlevel','temperature','ne')},material_generation=1)
    np.testing.assert_allclose(old_trial.fraction,trial.fraction,rtol=2e-14,atol=0)
    np.testing.assert_allclose(old_trial.saha,trial.saha,rtol=2e-14,atol=0)
    ids=np.flatnonzero((roster['Z']==z)&(roster['stage']==stage))
    if not len(ids):raise ValueError('대상 이온 BF 경로 없음')
    _,old_total=old.integrate(old_trial,material_generation=1,radiation_generation=1,route_indices=ids)
    f2_total=r['total'][ids]
    root=Path('/gpfs/kjhan/cmfgen_runs/lumina_F2_response_lageunha_20261028-bBbs8U/run')
    report=json.loads((root/'result.json').read_text());response=Path(report['selected_response']['root'])/'response.npz'
    with np.load(response) as f:columns=f['columns'];population=f['population'][:,node];fraction=f['fraction'][:,node]
    index={tuple(map(int,c)):i for i,c in enumerate(columns)}
    level=np.array([index[(int(x['atomic_number']),int(x['ion_number']),int(x['super_level']))] for x in r['rows']])
    lo,up=roster['lower'][ids],roster['target'][ids]
    upper=np.array([level[u] if u>=0 else index[(int(roster['Z'][i]),int(roster['stage'][i]),-1)] for i,u in zip(ids,up)])
    lower_pop=population[level[lo]]*fraction[lo];upper_pop=population[upper];prob=roster['probability'][ids]
    ion_density=population[(columns[:,0]==z)&(columns[:,1]==stage-1)].sum()
    def flows(values):
        values=values[:,0,:];return np.array([np.sum(prob*lower_pop*values[:,0],dtype=np.longdouble),
            np.sum(prob*upper_pop*values[:,1],dtype=np.longdouble),np.sum(prob*upper_pop*values[:,2],dtype=np.longdouble)],dtype='f8')
    f2=flows(f2_total);cmf=flows(old_total)
    balance=Path(report['selected_response']['root'])/'balance.npz'
    if B.sha(balance)!=report['selected_response']['balance_sha256']:raise ValueError('F2 흐름 기준 변경')
    with np.load(balance) as f:
        labels=f['bf_ion_Z_stage'];ix=np.flatnonzero((labels[:,0]==z)&(labels[:,1]==stage))
        if len(ix)!=1:raise ValueError('F2 저장 이온 주소 오류')
        stored=f['bf_ion_number_flows'][ix[0],:3,node]
    np.testing.assert_allclose(f2,stored,rtol=3e-12,atol=0)
    result=dict(state='COMPLETE_SAME_MATERIAL_GRID_FIELD_BF_COMPARISON',node=node,Z=z,stage=stage,
        routes=len(ids),ion_density=float(ion_density),F2_flows=f2.tolist(),CMF70_interpolated_flows=cmf.tolist(),
        F2_photo_rate_per_ion=float(f2[0]/ion_density),CMF70_photo_rate_per_ion=float(cmf[0]/ion_density),
        field_sha256=r['field_sha256'],CMF70_field_sha256=plan['reference_field_sha256'],
        response_sha256=r['response_sha256'],interpolation='linear J vs log frequency, no extrapolation',
        native_CMFGEN_grid_quadrature=False,new_transport=0,new_SE=0,seconds=time.monotonic()-start,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',type=int,required=True);p.add_argument('--element',type=int,choices=(14,16),required=True)
    p.add_argument('--stage',type=int,default=3);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.node,a.element,a.stage,a.out)
