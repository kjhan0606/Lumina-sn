#!/usr/bin/env python3
"""보존된 F1/F2 열장부를 읽고 주요 점의 열채널과 잔차를 비교한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import cmfgen_model_continuum as B

def load(root):
    path=Path(root)/'result.json';r=json.loads(path.read_text())
    if r['state']!='COMPLETE_FINITE_NEW_FIELD_RESPONSE_NOT_JOINT_CONVERGENCE':raise ValueError('완료 물질응답 필요')
    selected=r['selected_response'];response=Path(selected['root'])/'response.npz';balance=Path(selected['root'])/'balance.npz'
    profiles=Path(root)/'physical-profiles.npz'
    for path,digest in ((response,selected['response_sha256']),(balance,selected['balance_sha256']),
                        (profiles,r['output_sha256']['physical-profiles.npz'])):
        if B.sha(path)!=digest:raise ValueError('완료 열장부 입력 변경: '+path.name)
    with np.load(response) as f:population=f['population'];temperature=f['temperature'];ne=f['ne']
    with np.load(balance) as f:b={k:f[k] for k in f.files}
    with np.load(profiles) as f:p={k:f[k] for k in f.files}
    if not np.array_equal(temperature,p['new_response_temperature']) or not np.array_equal(ne,p['new_response_ne']):
        raise ValueError('응답 T/ne 주소 불일치')
    return r,population,b,p

def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError('기존 결과 보존')
    base=Path('/gpfs/kjhan/cmfgen_runs');f1=base/'lumina_new_J_response_1224468_20261027/run'
    f2=base/'lumina_F2_response_lageunha_20261028-bBbs8U/run'
    if B.sha(f2/'result.json')!='b6d411ff28971492280275d2201e9d84a65315c0ad2613f04444e7b270b5fbea':raise ValueError('F2 보고서 변경')
    a,pa,ba,qa=load(f1);b,pb,bb,qb=load(f2)
    if a['source_identity']!=b['source_identity'] or not np.array_equal(qa['accepted_temperature'],qb['accepted_temperature']):
        raise ValueError('F1/F2 물질 조건 불일치')
    entries=[]
    for i in sorted({3,30,33,49,89,int(np.argmax(abs(bb['physical_stored_net'])))}):
        item=dict(node=i,CMFGEN70_temperature=float(qb['CMFGEN70_temperature'][i]),
                  fixed_temperature=float(qb['new_response_temperature'][i]))
        for name,pop,data in (('F1',pa,ba),('F2',pb,bb)):
            bf=float(data['bf'][0,i]-data['bf'][1,i]);line=float(data['bb'][0,i]-data['bb'][1,i])
            ff=float(data['ff_all_positive_ions'][0,i]-data['ff_all_positive_ions'][1,i])
            ad=float(data['physical_adiabatic'][i]);decay=float(data['stored_decay'][i]);net=float(data['physical_stored_net'][i])
            other=net-(bf+line+ff-ad+decay);energy=float(data['physical_internal_energy'][i]*pop[:,i].sum())
            item[name]=dict(net=net,bf_net=bf,bb_net=line,ff_net=ff,adiabatic_cooling=ad,stored_decay=decay,
                other_external_inferred=other,gross_normalized=float(data['physical_stored_gross_normalized'][i]),
                channel_normalized=float(data['physical_stored_channel_normalized'][i]),
                internal_energy_density=energy,frozen_energy_over_abs_net_seconds=energy/abs(net) if net else None)
        entries.append(item)
    out.mkdir();result=dict(state='COMPLETE_SAVED_F1_F2_THERMAL_LEDGER',entries=entries,
        field1_sha256=a['field']['J']['sha256'],field2_sha256=b['field']['J']['sha256'],
        note='energy/absolute net is a frozen-state scale, not a time-dependent cooling solution; T remains fixed',
        new_transport=0,new_SE=0,new_RE=0,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
