#!/usr/bin/env python3
"""완성 F1/F2·CMFGEN 장과 수용 계수에서 원인 대역의 반경 경향을 판독한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import cmfgen_model_continuum as B
from lumina_native_columns import NativeSourceColumns
from prepare_lumina_outer_input import read_native
from run_cmfgen_formal_primary import fields
from run_lumina_full_material_response import BASE


def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError('기존 결과 보존')
    root=Path('/gpfs/kjhan/cmfgen_runs')
    f1=root/'lumina_new_J_response_1224468_20261027/run/result.json'
    f2=root/'lumina_F2_response_lageunha_20261028-bBbs8U/run/result.json'
    if B.sha(f2)!='b6d411ff28971492280275d2201e9d84a65315c0ad2613f04444e7b270b5fbea':raise ValueError('F2 보고 변경')
    a=json.loads(f1.read_text());b=json.loads(f2.read_text())
    if a['source_identity']!=b['source_identity']:raise ValueError('F1/F2 물질·계수 불일치')
    grid=Path('/gpfs/kjhan/cmfgen_runs/lumina_full_sr_verified_20261020-yb8k9y/grid/frequency.npy')
    jf1=Path(a['field']['J']['path']);jf2=Path(b['field']['J']['path'])
    for p,h in ((grid,b['source_identity']['grid_sha256']),(jf1,a['field']['J']['sha256']),(jf2,b['field']['J']['sha256'])):
        if B.sha(p)!=h:raise ValueError('입력 변경: '+p.name)
    base=json.loads(BASE.read_text());c=json.loads(Path(base['config']).read_text());plan=json.loads(Path(c['plan']).read_text())
    old=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if B.sha(old)!=plan['reference_field_sha256']:raise ValueError('원장 변경')
    prep=Path('/gpfs/kjhan/cmfgen_runs/lumina_accepted_sources_20261027-b5eCNv/run/boundary/preparation.json')
    if B.sha(prep)!=b['input_sha256'][str(prep)]:raise ValueError('수용 준비 변경')
    meta=json.loads(prep.read_text());material=Path(meta['material'])
    if B.sha(material)!=meta['input_sha256'][str(material)]:raise ValueError('수용 물질 변경')
    physical,epoch=read_native(material);radius=physical[:,0]
    geometry,ref=fields(old);nu=np.load(grid,mmap_mode='r');one=np.load(jf1,mmap_mode='r');two=np.load(jf2,mmap_mode='r')
    try:
        if not np.array_equal(geometry[0]*1e10,radius) or epoch!=b['source_identity']['epoch_seconds']:
            raise ValueError('반경·시점 불일치')
        old_nu=np.asarray(ref['freq'][::-1])*1e15
        bands=[]
        for lo,hi in ((550.,650.),(1500.,2500.)):
            lower=B.C*1e8/hi;upper=B.C*1e8/lo
            mask=(nu>lower)&(nu<upper);freq=np.r_[lower,nu[mask],upper];x=np.log(freq);indices=np.flatnonzero(mask)
            vals=[]
            for node in range(90):
                oldj=np.interp(x,np.log(old_nu),np.asarray(ref['field'][::-1,0,node]))
                arrays=[oldj]
                for source in (one,two):
                    data=source[node];edge=np.interp([lower,upper],nu,data)
                    arrays.append(np.r_[edge[0],data[indices],edge[1]])
                v=[float(np.trapezoid(j/freq,freq)) for j in arrays]
                if min(v)<=0:raise ValueError('대역 복사장 비양수')
                vals.append(dict(node=node,CMF=v[0],F1=v[1],F2=v[2],F2_over_CMF=v[2]/v[0],F2_over_F1=v[2]/v[1]))
            bands.append(dict(wavelength_A=[lo,hi],values=vals))
        coefficients=[];source_path=Path(meta['source_binary'])
        # 본문은 생산 당시 봉인 SHA를 계승하고 필요한 열만 읽는다.
        if meta['source_binary_sha256']!=b['source_identity']['source_sha256']:raise ValueError('수송 계수 계보 불일치')
        with NativeSourceColumns(source_path,nu,radius,physical[:,1],physical[:,2]) as source:
            for wavelength,node in ((600.,33),(1800.,30)):
                target=B.C*1e8/wavelength;idx=int(np.argmin(abs(nu-target)))
                chi,es,eta=source.read(np.arange(90),np.array([idx]))
                delta=np.diff(-radius);tau=np.r_[0.,np.cumsum(.5*(chi[:-1,0]+chi[1:,0])*delta)]
                coefficients.append(dict(wavelength_requested_A=wavelength,wavelength_actual_A=float(B.C*1e8/nu[idx]),
                    node=node,chi=float(chi[node,0]),electron_scattering=float(es[node,0]),true_emissivity=float(eta[node,0]),
                    radial_signed_optical_depth_from_outer=float(tau[node]),
                    meaning='signed radial extinction integral, not ray effective depth; selected accepted source not CMFGEN coefficient'))
    finally:
        ref._mmap.close();nu._mmap.close();one._mmap.close();two._mmap.close()
    out.mkdir();result=dict(state='COMPLETE_EXISTING_FIELDS_BAND_RADIAL_ANALYSIS',bands=bands,coefficients=coefficients,
        full_source_hash_verification='INHERITED_SEALED_PREPARATION_NOT_REHASHED',new_transport=0,new_SE=0,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(state=result['state'],bands=[dict(wavelength_A=x['wavelength_A'],
        samples=[x['values'][i] for i in (0,20,30,33,49,60,75,89)]) for x in bands],coefficients=coefficients)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
