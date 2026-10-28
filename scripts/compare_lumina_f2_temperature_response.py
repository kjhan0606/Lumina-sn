#!/usr/bin/env python3
"""F2 자체장·ne 고정에서 한 점 한 원소의 T 변화에 대한 실제 SE 응답."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import cmfgen_model_continuum as B
from lumina_f2_node_rates import reconstruct
from lumina_model_collision_source import ModelCollisionSource
from lumina_model_material_bridge import MaterialBridge
from run_lumina_full_material_response import BB_RESULT, PARAMETERS

def run(node,z,temperature_factor,cache_result,out,changed_bf=None,changed_bf_sha256=None):
    start=time.monotonic();out=Path(out)
    if out.exists():raise FileExistsError('기존 결과 보존')
    if temperature_factor not in (0.99,1.01):raise ValueError('고정 소형 온도 섭동만 허용')
    out.mkdir();r=reconstruct(node,cache_result);p=r['provider'];base_trial=r['trial'];roster=r['roster']
    q={k:base_trial.state[k] for k in ('columns','superlevel','temperature','ne')}
    q['temperature']=q['temperature']*temperature_factor
    changed=p.prepare(q,material_generation=3)
    rid=np.flatnonzero(roster['Z']==z)
    if (changed_bf is None)!=(changed_bf_sha256 is None):raise ValueError('부분 BF 경로와 SHA가 함께 필요')
    if changed_bf is None:
        _,new_total=p.integrate(changed,material_generation=3,radiation_generation=1,route_indices=rid)
    else:
        if B.sha(changed_bf)!=changed_bf_sha256:raise ValueError('보존 BF SHA 변경')
        with np.load(changed_bf) as f:
            np.testing.assert_array_equal(f['ids'],rid);np.testing.assert_array_equal(f['saha'],changed.saha)
            np.testing.assert_array_equal(f['fraction'],changed.fraction)
            np.testing.assert_array_equal(f['occupation'],changed.state['occupation'])
            new_total=f['total'].copy()
        if new_total.shape!=(len(rid),1,6) or not np.isfinite(new_total).all() or np.any(new_total[:,:,:3]<0):
            raise ValueError('보존된 부분 BF 형상/값 오류')
    np.savez(out/'changed-BF.npz',ids=rid,total=new_total,saha=changed.saha,fraction=changed.fraction,
             occupation=changed.state['occupation'])
    root=Path('/gpfs/kjhan/cmfgen_runs/lumina_F2_response_lageunha_20261028-bBbs8U/run')
    report=json.loads((root/'result.json').read_text());response=Path(report['selected_response']['root'])/'response.npz'
    with np.load(response) as f:pop=f['population'];columns=f['columns'];ne=f['ne'];occupation=f['occupation'];fraction=f['fraction']
    if B.sha(BB_RESULT)!='b95b16e3401068bc8239116a87c872f392e970ec5780b7ce04d32cf51d1e9ec9':raise ValueError('선 보고서 변경')
    bb=json.loads(BB_RESULT.read_text());line_path=BB_RESULT.parent/'line-roster.npz'
    if B.sha(line_path)!=bb['output_sha256']['line-roster.npz']:raise ValueError('선 명부 변경')
    with np.load(line_path) as f:lines={k:f[k] for k in f.files}
    zs=np.array([int(x['atomic_number']) for x in r['rows']]);ids=np.flatnonzero(zs[lines['lower']]==z);sel={k:v[ids] for k,v in lines.items()}
    saved=root/'jbar.npy'
    if B.sha(saved)!=report['output_sha256']['jbar.npy']:raise ValueError('F2 선율 변경')
    J=np.load(saved,mmap_mode='r')[ids,node];A=sel['A'];bul=A*B.C**2/(2*B.H*sel['nu0']**3)
    source=ModelCollisionSource(r['runroot'],r['rows']);output=[]
    for name,trial,rate in (('F2',base_trial,r['total'][:,0,:]),('temperature_changed',changed,new_total[:,0,:])):
        temperature=np.array(r['state']['temperature'],copy=True);occ=occupation.copy();frac=fraction.copy()
        if name!='F2':
            temperature[node]=trial.state['temperature'][0];occ[:,node]=trial.state['occupation'][:,0];frac[:,node]=trial.fraction[:,0]
        state=dict(columns=r['state']['columns'],superlevel=r['state']['superlevel'],temperature=temperature,ne=ne,occupation=occ)
        saha=np.zeros((len(roster['lower']),90));saha[:,node]=trial.saha[:,0]
        rates=np.zeros((len(roster['lower']),90,6))
        if name=='F2':rates[:,node,:]=rate
        else:rates[rid,node,:]=rate
        candidate=np.zeros(state['superlevel'].shape,dtype='f8')
        with MaterialBridge(r['library'],r['material'].parent/'atomic',r['material'],root/'occupation.bin',PARAMETERS) as bridge:
            bridge.prepare(state,frac,saha,roster,rates,trial_id=2,radiation_generation=1)
            bridge.begin_element(z,node,sel,bul/sel['g_ratio']*J,bul*J,A,np.ones(len(ids),dtype='u1'))
            for key in sorted(k for k in source.models if k[0]==z):
                ion=source.load(*key)
                for lo,up,omega,_ in ion.blocks([temperature[node]]):bridge.push(z,node,lo,up,omega[0])
                ion.verify_identity()
            ledger=bridge.finish(z,node,candidate)
        select=columns[:,0]==z;actual=candidate[select,node];reference=pop[select,node];stage=columns[select,1]
        if name=='F2':np.testing.assert_allclose(actual,reference,rtol=2e-8,atol=1e-30)
        output.append(dict(case=name,temperature_K=float(temperature[node]),
            stage_fractions={str(int(q)):float(actual[stage==q].sum()/actual.sum()) for q in np.unique(stage)},
            BF_photon_heat=float(ledger[0]),BB_photon_heat=float(ledger[1]),
            BB_collisional_heat=float(ledger[4]),stationary_residual=float(ledger[6])))
    result=dict(state='COMPLETE_F2_LOCAL_TEMPERATURE_SENSITIVITY',node=node,Z=z,temperature_factor=temperature_factor,
        results=output,changed_BF_sha256=B.sha(out/'changed-BF.npz'),F2_response_sha256=r['response_sha256'],
        F2_field_sha256=r['field_sha256'],cache_result_sha256=B.sha(cache_result),seconds=time.monotonic()-start,
        reused_changed_BF_sha256=changed_bf_sha256,
        scope='single element fixed J/ne, not joint thermal or charge equilibrium',final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',type=int,required=True);p.add_argument('--element',type=int,choices=(14,16),required=True)
    p.add_argument('--factor',type=float,required=True);p.add_argument('--cache-result',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--changed-bf',type=Path);p.add_argument('--changed-bf-sha256')
    a=p.parse_args();run(a.node,a.element,a.factor,a.cache_result,a.out,a.changed_bf,a.changed_bf_sha256)
