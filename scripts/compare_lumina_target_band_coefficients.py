#!/usr/bin/env python3
"""같은 원래점의 원 CMFGEN formal 및 두 물질 세대의 수송 계수를 대조한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import cmfgen_model_continuum as B
from compare_lumina_native_formal_coefficients import formal
from lumina_native_columns import NativeSourceColumns,open_sealed_native
from prepare_lumina_outer_input import read_native
from run_lumina_full_material_response import BASE

def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError('기존 결과 보존')
    base=json.loads(BASE.read_text());config=json.loads(Path(base['config']).read_text());plan=json.loads(Path(config['plan']).read_text())
    original=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    geometry,frequency,ref=formal(original,plan['reference_field_sha256'],[30,33])
    prep=Path('/gpfs/kjhan/cmfgen_runs/lumina_accepted_sources_20261027-b5eCNv/run/boundary/preparation.json')
    report_path=Path('/gpfs/kjhan/cmfgen_runs/lumina_F2_response_lageunha_20261028-bBbs8U/run/result.json')
    if B.sha(report_path)!='b6d411ff28971492280275d2201e9d84a65315c0ad2613f04444e7b270b5fbea':raise ValueError('F2 결과 변경')
    report=json.loads(report_path.read_text())
    if B.sha(prep)!=report['input_sha256'][str(prep)]:raise ValueError('F2 준비 변경')
    meta=json.loads(prep.read_text());material=Path(meta['material']);grid=Path(meta['grid'])
    for path in (material,grid):
        if B.sha(path)!=meta['input_sha256'][str(path)]:raise ValueError('수용 물질/격자 변경')
    physical,epoch=read_native(material);nu=np.load(grid,mmap_mode='r')
    if not np.array_equal(geometry[0]*1e10,physical[:,0]) or epoch!=meta['epoch_seconds']:
        raise ValueError('CMFGEN과 수용 반경 불일치')
    if meta['source_binary_sha256']!=report['source_identity']['source_sha256']:raise ValueError('수용 계수 계보 오류')
    rows=[]
    with open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as old, \
         NativeSourceColumns(meta['source_binary'],nu,physical[:,0],physical[:,1],physical[:,2]) as new:
        if not np.array_equal(old.frequency,nu) or not np.array_equal(old.radius,physical[:,0]):raise ValueError('두 계수 격자/반경 불일치')
        for node,lo,hi in ((33,550.,650.),(30,1500.,2500.)):
            wave=B.C*1e8/frequency;sel=(wave>=lo)&(wave<hi);target=frequency[sel][::-1]
            if len(target)<2:raise ValueError('대역 표본 부족')
            first=max(0,int(np.searchsorted(nu,target[0]))-1);last=min(len(nu),int(np.searchsorted(nu,target[-1]))+1)
            idx=np.arange(first,last,dtype='i8')
            if len(idx)>131072:raise ValueError('요청 대역 크기 초과')
            old_values=old.read(np.array([node]),idx);new_values=new.read(np.array([node]),idx)
            native=ref[:,(30,33).index(node),sel][:,::-1]
            metrics={}
            for k,name in ((0,'chi'),(1,'scattering'),(2,'true_emissivity')):
                reference=native[{0:0,1:2,2:1}[k]]
                oldp=np.interp(np.log(target),np.log(nu[idx]),old_values[k][0]);newp=np.interp(np.log(target),np.log(nu[idx]),new_values[k][0])
                scale=float(np.trapezoid(abs(reference),target))
                if not scale>0:raise ValueError('대역 기준 적분 0')
                def measure(a):return dict(signed=float(np.trapezoid(a-reference,target)/scale),
                    absolute=float(np.trapezoid(abs(a-reference),target)/scale))
                metrics[name]=dict(material70=measure(oldp),accepted_F2_material=measure(newp))
            rows.append(dict(node=node,wavelength_A=[lo,hi],native_points=len(target),metrics=metrics,
                accepted_temperature=float(physical[node,1])))
        source_bytes=dict(original=old.coefficient_bytes_read,accepted=new.coefficient_bytes_read)
    nu._mmap.close();out.mkdir()
    result=dict(state='COMPLETE_TARGET_NODAL_COEFFICIENT_COMPARISON',comparisons=rows,coefficient_bytes_read=source_bytes,
        formal_sha256=plan['reference_field_sha256'],old_source_sha256=base['source_identity']['source_sha256'],
        accepted_source_sha256=meta['source_binary_sha256'],source_hash_verification='INHERITED_SEALS_NOT_REHASHED',
        limits='different material states and frequency grids; nodal coefficient mismatch is not a transport or ionization error attribution',
        new_transport=0,new_SE=0,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
