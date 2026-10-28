#!/usr/bin/env python3
"""같은 F2 점에서 Si/S 양방향 온도 응답을 합치되 나머지 채널은 고정한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import cmfgen_model_continuum as B

def run(cross_root,out):
    cross_root=Path(cross_root);out=Path(out)
    if out.exists():raise FileExistsError('기존 결과 보존')
    base=Path('/gpfs/kjhan/cmfgen_runs');paths={
        (30,14,.99):base/'lumina_f2_si31_tminus_recover_20261028-fJoYaq/result/result.json',
        (30,14,1.01):base/'lumina_f2_si31_tplus_recover_20261028-5iADtq/result/result.json',
        (33,16,.99):base/'lumina_f2_s34_tminus_retry_20261028-wUJBdC/result/result.json',
        (33,16,1.01):base/'lumina_f2_s34_tplus_20261028-oJBDBP/result/result.json',
        (30,16,.99):cross_root/'s31_minus/result.json',(30,16,1.01):cross_root/'s31_plus/result.json',
        (33,14,.99):cross_root/'si34_minus/result.json',(33,14,1.01):cross_root/'si34_plus/result.json'}
    field='fc66fc815d7e8c67028c269944e01227611497f7bdfa129bff863a18c1f69308'
    response='9e0d973610bbfb820d168c375981a545663db486365c25a0a644b09f9d594312';data={}
    for key,path in paths.items():
        value=json.loads(path.read_text())
        if (value['state']!='COMPLETE_F2_LOCAL_TEMPERATURE_SENSITIVITY' or
            (value['node'],value['Z'],value['temperature_factor'])!=key or
            value['F2_field_sha256']!=field or value['F2_response_sha256']!=response or
            B.sha(path.parent/'changed-BF.npz')!=value['changed_BF_sha256']):
            raise ValueError('온도 응답 세대/자료 불일치')
        data[key]=value
    f2=base/'lumina_F2_response_lageunha_20261028-bBbs8U/run/result.json'
    if B.sha(f2)!='b6d411ff28971492280275d2201e9d84a65315c0ad2613f04444e7b270b5fbea':raise ValueError('F2 결과 변경')
    report=json.loads(f2.read_text());balance=Path(report['selected_response']['root'])/'balance.npz'
    if B.sha(balance)!=report['selected_response']['balance_sha256']:raise ValueError('F2 열장부 변경')
    with np.load(balance) as f:net=f['physical_stored_net'].copy()
    records=[]
    for node in (30,33):
        record=dict(node=node,F2_total_stored_net=float(net[node]),cases=[])
        for factor in (.99,1.01):
            entries=[data[(node,z,factor)]['results'] for z in (14,16)]
            if not all(x[0]['case']=='F2' and x[1]['case']=='temperature_changed' for x in entries):raise ValueError('온도 사례 오류')
            temperature=[x[1]['temperature_K'] for x in entries]
            if temperature[0]!=temperature[1]:raise ValueError('원소별 섭동 온도 불일치')
            delta_BB=sum(x[1]['BB_photon_heat']-x[0]['BB_photon_heat'] for x in entries)
            delta_BF=sum(x[1]['BF_photon_heat']-x[0]['BF_photon_heat'] for x in entries)
            record['cases'].append(dict(factor=factor,temperature_K=temperature[0],
                Si_III=entries[0][1]['stage_fractions']['2'],S_III=entries[1][1]['stage_fractions']['2'],
                delta_SiS_BB=delta_BB,delta_SiS_BF=delta_BF,
                frozen_other_terms_net=float(net[node]+delta_BB+delta_BF)))
        records.append(record)
    out.mkdir();result=dict(state='COMPLETE_LOCAL_TWO_ELEMENT_TEMPERATURE_COMPARISON',records=records,
        input_sha256={str(path):B.sha(path) for path in paths.values()},F2_field_sha256=field,F2_response_sha256=response,
        scope='Si/S changed; other elements, FF, adiabatic, external, electron density and radiation frozen; not thermal equilibrium',
        new_BF=0,new_SE=0,new_transport=0,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cross-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.cross_root,a.out)
