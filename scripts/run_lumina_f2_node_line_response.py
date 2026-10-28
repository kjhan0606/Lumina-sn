#!/usr/bin/env python3
"""F2 동일 T/ne/J의 단일 원소 정상해에서 선 적분 방식의 영향을 구한다."""
import argparse
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from lumina_f2_node_rates import reconstruct
from lumina_model_collision_source import ModelCollisionSource
from lumina_model_material_bridge import MaterialBridge
from lumina_model_point_lines import ModelPointLines
from lumina_new_field_response import trapezoid_weights
from run_lumina_full_material_response import BB_RESULT, PARAMETERS


def run(node,z,point_library,out):
    out=Path(out)
    if out.exists():raise FileExistsError('기존 결과 보존')
    out.mkdir()
    r=reconstruct(node)
    np.savez(out/'node-bf.npz',bound=r['bound'],total=r['total'],saha=r['trial'].saha,
             fraction=r['trial'].fraction,occupation=r['trial'].state['occupation'])
    root=Path('/gpfs/kjhan/cmfgen_runs/lumina_F2_response_lageunha_20261028-bBbs8U/run')
    report=json.loads((root/'result.json').read_text()); response=Path(report['selected_response']['root'])/'response.npz'
    with np.load(response) as f:
        population=f['population'];columns=f['columns'];ne=f['ne'];occupation=f['occupation'];fraction=f['fraction']
    state={k:r['state'][k] for k in ('columns','superlevel','temperature')}
    state.update(ne=ne,occupation=occupation)
    ns=90;nr=len(r['roster']['lower']);saha=np.zeros((nr,ns));saha[:,node]=r['trial'].saha[:,0]
    rates=np.zeros((nr,ns,6));rates[:,node,:]=r['total'][:,0,:]
    if B.sha(BB_RESULT)!='b95b16e3401068bc8239116a87c872f392e970ec5780b7ce04d32cf51d1e9ec9':raise ValueError('선 명부 보고서 변경')
    old=json.loads(BB_RESULT.read_text());roster_path=BB_RESULT.parent/'line-roster.npz'
    if B.sha(roster_path)!=old['output_sha256']['line-roster.npz']:raise ValueError('선 명부 변경')
    with np.load(roster_path) as f:lines={k:f[k] for k in f.files}
    zs=np.array([int(x['atomic_number']) for x in r['rows']]);ids=np.flatnonzero(zs[lines['lower']]==z)
    sel={k:v[ids] for k,v in lines.items()};nu0=sel['nu0'];A=sel['A'];bul=A*B.C**2/(2*B.H*nu0**3)
    jbar_path=root/'jbar.npy'
    if B.sha(jbar_path)!=report['output_sha256']['jbar.npy']:raise ValueError('F2 선율 변경')
    jbar=np.load(jbar_path,mmap_mode='r')[ids,node]
    meta=json.loads(Path('/gpfs/kjhan/cmfgen_runs/lumina_accepted_sources_20261027-b5eCNv/run/boundary/preparation.json').read_text())
    nu=np.load(meta['grid']);J=np.load(report['field']['J']['path'],mmap_mode='r')[node]
    profile=S.fixed_profile(r['runroot']/'MODEL')
    producer=SimpleNamespace(lines=sel,profile=profile,dll=ctypes.CDLL(str(point_library)))
    producer.function=producer.dll.lumina_model_point_gaussian_jbar;producer.ptr=ctypes.POINTER(ctypes.c_double)
    producer.function.argtypes=[ctypes.c_size_t,ctypes.c_size_t]+[producer.ptr]*4+[ctypes.c_double]*2+[producer.ptr]*2+[ctypes.POINTER(ctypes.c_uint8)]
    producer.function.restype=ctypes.c_int
    point=ModelPointLines.contract(producer,nu,trapezoid_weights(nu),J)
    if not np.isin(point['status'],[1,2]).all():raise ValueError('점 선율 미완료')
    source=ModelCollisionSource(r['runroot'],r['rows']);results=[]
    for name,values,emission in (('F2_linear',jbar,A),('point',point['jbar'],A*point['profile_mass'])):
        candidate=np.zeros(state['superlevel'].shape,dtype='f8',order='C')
        with MaterialBridge(r['library'],r['material'].parent/'atomic',r['material'],root/'occupation.bin',PARAMETERS) as bridge:
            bridge.prepare(state,fraction,saha,r['roster'],rates,trial_id=2,radiation_generation=1)
            bridge.begin_element(z,node,sel,bul/sel['g_ratio']*values,bul*values,emission,np.ones(len(ids),dtype='u1'))
            for key in sorted(k for k in source.models if k[0]==z):
                ion=source.load(*key)
                for lo,up,omega,_ in ion.blocks([state['temperature'][node]]):bridge.push(z,node,lo,up,omega[0])
                ion.verify_identity()
            ledger=bridge.finish(z,node,candidate)
        select=columns[:,0]==z;actual=candidate[select,node];reference=population[select,node]
        if name=='F2_linear':
            if not np.allclose(actual,reference,rtol=2e-8,atol=1e-30):raise ValueError('저장 F2 정상해 재현 실패')
        total=actual.sum();stage=columns[select,1]
        results.append(dict(name=name,stage_fractions={str(int(q)):float(actual[stage==q].sum()/total) for q in np.unique(stage)},
            maximum_component_relative_to_F2=float(np.max(np.abs(actual-reference)/np.maximum(reference,1e-300))),ledger=ledger.tolist()))
    result=dict(state='COMPLETE_F2_SINGLE_ELEMENT_LINE_SENSITIVITY',node=node,Z=z,results=results,
        bound_reused=r['reused'],bound_direct=r['direct'],field_sha256=r['field_sha256'],response_sha256=r['response_sha256'],
        node_bf_sha256=B.sha(out/'node-bf.npz'),
        scope='fixed F2 T/ne/J, one element only; not new charge/RE/transport',final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',type=int,required=True);p.add_argument('--element',type=int,choices=(14,16),required=True)
    p.add_argument('--point-library',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.node,a.element,a.point_library,a.out)
