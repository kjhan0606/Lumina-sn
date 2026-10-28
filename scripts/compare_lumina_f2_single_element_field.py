#!/usr/bin/env python3
"""같은 F2 물질·격자에서 원장/F2장의 단일 원소 정상 이온분포를 비교한다."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from lumina_f2_node_rates import reconstruct
from lumina_model_collision_source import ModelCollisionSource
from lumina_model_linear_lines import LinearModelLines
from lumina_model_material_bridge import MaterialBridge
from lumina_model_material_rates import ModelMaterialRates
from run_cmfgen_formal_primary import fields
from run_lumina_full_material_response import BASE, BB_RESULT, PARAMETERS

def run(node,z,cache_result,out):
    start=time.monotonic();out=Path(out)
    if out.exists():raise FileExistsError('기존 결과 보존')
    out.mkdir();r=reconstruct(node,cache_result);p=r['provider'];trial=r['trial'];roster=r['roster']
    base=json.loads(BASE.read_text());config=json.loads(Path(base['config']).read_text());plan=json.loads(Path(config['plan']).read_text())
    path=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if B.sha(path)!=plan['reference_field_sha256']:raise ValueError('원장 변경')
    geometry,original=fields(path)
    try:
        old_nu=np.asarray(original['freq'][::-1])*1e15;old_j=np.asarray(original['field'][::-1,0,node]);nu=p.frequency
        if nu[0]<old_nu[0] or nu[-1]>old_nu[-1] or not np.array_equal(geometry[0]*1e10,r['state']['physical'][:,0]):
            raise ValueError('원장 범위/반경 불일치')
        interp=np.interp(np.log(nu),np.log(old_nu),old_j)
    finally:original._mmap.close()
    provider=ModelMaterialRates(p.routes,p.rows,p.models,r['library'],nu,p.width,interp[None,:],radiation_generation=1,dissolution_enabled=p.dissolved)
    old_trial=provider.prepare({k:trial.state[k] for k in ('columns','superlevel','temperature','ne')},material_generation=1)
    np.testing.assert_array_equal(old_trial.saha,trial.saha);np.testing.assert_array_equal(old_trial.fraction,trial.fraction)
    rid=np.flatnonzero(roster['Z']==z)
    _,old_total=provider.integrate(old_trial,material_generation=1,radiation_generation=1,route_indices=rid)
    np.savez(out/'old-field-bf.npz',ids=rid,total=old_total)
    root=Path('/gpfs/kjhan/cmfgen_runs/lumina_F2_response_lageunha_20261028-bBbs8U/run')
    report=json.loads((root/'result.json').read_text());response=Path(report['selected_response']['root'])/'response.npz'
    with np.load(response) as f:pop=f['population'];columns=f['columns'];ne=f['ne'];occupation=f['occupation'];fraction=f['fraction']
    state={k:r['state'][k] for k in ('columns','superlevel','temperature')};state.update(ne=ne,occupation=occupation)
    if B.sha(BB_RESULT)!='b95b16e3401068bc8239116a87c872f392e970ec5780b7ce04d32cf51d1e9ec9':raise ValueError('선 보고서 변경')
    bb=json.loads(BB_RESULT.read_text());line_path=BB_RESULT.parent/'line-roster.npz'
    if B.sha(line_path)!=bb['output_sha256']['line-roster.npz']:raise ValueError('선 명부 변경')
    with np.load(line_path) as f:lines={k:f[k] for k in f.files}
    zs=np.array([int(x['atomic_number']) for x in r['rows']]);ids=np.flatnonzero(zs[lines['lower']]==z);sel={k:v[ids] for k,v in lines.items()}
    line_library=root.parent/'build/lines.so'
    if B.sha(line_library)!=report['input_sha256'][str(line_library)]:raise ValueError('F2 선율 코드 변경')
    linear=LinearModelLines(sel,S.fixed_profile(r['runroot']/'MODEL'),line_library,nu,interp[:,None],radiation_generation=1)
    values=linear.contract(0,len(ids),radiation_generation=1)
    if not np.isin(values['status'],[1,2]).all():raise ValueError('원장 선율 범위 누락')
    saved=root/'jbar.npy'
    if B.sha(saved)!=report['output_sha256']['jbar.npy']:raise ValueError('F2 선율 변경')
    f2_j=np.load(saved,mmap_mode='r')[ids,node];A=sel['A'];bul=A*B.C**2/(2*B.H*sel['nu0']**3)
    saha=np.zeros((len(roster['lower']),90));saha[:,node]=trial.saha[:,0]
    source=ModelCollisionSource(r['runroot'],r['rows']);result=[]
    for name,j,rate in (('F2',f2_j,r['total'][:,0,:]),('CMF70',values['jbar'][:,0],old_total[:,0,:])):
        rates=np.zeros((len(roster['lower']),90,6))
        if name=='F2':rates[:,node,:]=rate
        else:rates[rid,node,:]=rate
        candidate=np.zeros(state['superlevel'].shape,dtype='f8')
        with MaterialBridge(r['library'],r['material'].parent/'atomic',r['material'],root/'occupation.bin',PARAMETERS) as bridge:
            bridge.prepare(state,fraction,saha,roster,rates,trial_id=2,radiation_generation=1)
            bridge.begin_element(z,node,sel,bul/sel['g_ratio']*j,bul*j,A,np.ones(len(ids),dtype='u1'))
            for key in sorted(k for k in source.models if k[0]==z):
                ion=source.load(*key)
                for lo,up,omega,_ in ion.blocks([state['temperature'][node]]):bridge.push(z,node,lo,up,omega[0])
                ion.verify_identity()
            ledger=bridge.finish(z,node,candidate)
        select=columns[:,0]==z;actual=candidate[select,node];reference=pop[select,node];stage=columns[select,1]
        if name=='F2':np.testing.assert_allclose(actual,reference,rtol=2e-8,atol=1e-30)
        result.append(dict(field=name,stage_fractions={str(int(q)):float(actual[stage==q].sum()/actual.sum()) for q in np.unique(stage)},
                           max_component_relative_F2=float(np.max(np.abs(actual-reference)/np.maximum(reference,1e-300))),
                           residual=float(ledger[6])))
    output=dict(state='COMPLETE_SAME_F2_MATERIAL_GRID_TWO_FIELD_SE',node=node,Z=z,results=result,
        CMF70_sha256=plan['reference_field_sha256'],F2_sha256=r['field_sha256'],F2_response_sha256=r['response_sha256'],
        old_field_BF_sha256=B.sha(out/'old-field-bf.npz'),cache_result_sha256=B.sha(cache_result),
        interpolation='linear J vs log frequency no extrapolation',seconds=time.monotonic()-start,
        scope='single element fixed T/ne; not charge/RE/transport convergence',final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print(json.dumps(output))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--node',type=int,required=True);a.add_argument('--element',type=int,choices=(14,16),required=True)
    a.add_argument('--cache-result',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    q=a.parse_args();run(q.node,q.element,q.cache_result,q.out)
