#!/usr/bin/env python3
"""동일 격자·반경에서 원70→F2 수용 물질의 source 평면 변화를 분해한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import cmfgen_model_continuum as B
from lumina_native_columns import NativeSourceColumns,open_sealed_native
from prepare_lumina_outer_input import read_native
from run_lumina_full_material_response import BASE

def components(source,node,first,last):
    out=[]
    for plane in range(7):
        offset=source.base+8*((plane*source.ns+node)*source.nf+first)
        raw=source._read(offset,8*(last-first))
        out.append(np.frombuffer(raw,dtype='<f8').copy())
    values=np.asarray(out)
    if not np.isfinite(values).all() or np.any(values[[1,2,3,5,6]]<0):raise ValueError('source 평면 손상')
    return values

def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError('기존 결과 보존')
    base=json.loads(BASE.read_text());prep=Path('/gpfs/kjhan/cmfgen_runs/lumina_accepted_sources_20261027-b5eCNv/run/boundary/preparation.json')
    result=Path('/gpfs/kjhan/cmfgen_runs/lumina_F2_response_lageunha_20261028-bBbs8U/run/result.json')
    if B.sha(result)!='b6d411ff28971492280275d2201e9d84a65315c0ad2613f04444e7b270b5fbea':raise ValueError('F2 결과 변경')
    report=json.loads(result.read_text())
    if B.sha(prep)!=report['input_sha256'][str(prep)]:raise ValueError('F2 준비 변경')
    meta=json.loads(prep.read_text());material=Path(meta['material']);grid=Path(meta['grid'])
    for path in (material,grid):
        if B.sha(path)!=meta['input_sha256'][str(path)]:raise ValueError('물질/격자 변경')
    physical,epoch=read_native(material);nu=np.load(grid,mmap_mode='r')
    if meta['source_binary_sha256']!=report['source_identity']['source_sha256'] or epoch!=meta['epoch_seconds']:
        raise ValueError('수용 source 계보 오류')
    rows=[];names=('bf_chi','bf_eta','ff_chi','ff_eta','bb_chi','bb_eta','es_chi')
    with open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as old, \
         NativeSourceColumns(meta['source_binary'],nu,physical[:,0],physical[:,1],physical[:,2]) as new:
        if not np.array_equal(old.frequency,nu) or not np.array_equal(old.radius,physical[:,0]):raise ValueError('격자·반경 변경')
        for node,lo,hi in ((33,550.,650.),(30,1500.,2500.)):
            first=int(np.searchsorted(nu,B.C*1e8/hi));last=int(np.searchsorted(nu,B.C*1e8/lo,side='right'))
            before=components(old,node,first,last);after=components(new,node,first,last);frequency=nu[first:last]
            for source,values in ((old,before),(new,after)):
                chi,es,eta=source.read(np.array([node]),np.arange(first,last,dtype='i8'))
                np.testing.assert_array_equal((values[6]+(values[0]+values[2]))+values[4],chi[0])
                np.testing.assert_array_equal((values[1]+values[3])+values[5],eta[0])
                np.testing.assert_array_equal(values[6],es[0])
            old_chi=(before[6]+(before[0]+before[2]))+before[4];old_eta=(before[1]+before[3])+before[5]
            scales={'chi':float(np.trapezoid(abs(old_chi),frequency)),'eta':float(np.trapezoid(abs(old_eta),frequency))}
            if min(scales.values())<=0:raise ValueError('기준 적분 0')
            terms={}
            for k,name in enumerate(names):
                scale=scales['eta' if name.endswith('eta') else 'chi']
                terms[name]=dict(original=float(np.trapezoid(before[k],frequency)/scale),
                    accepted=float(np.trapezoid(after[k],frequency)/scale),
                    change=float(np.trapezoid(after[k]-before[k],frequency)/scale),
                    absolute_change=float(np.trapezoid(abs(after[k]-before[k]),frequency)/scale))
            rows.append(dict(node=node,wavelength_A=[lo,hi],samples=last-first,terms=terms,
                temperature_K=float(physical[node,1]),ne=float(physical[node,2])))
        byte_counts={'original':old.coefficient_bytes_read,'accepted':new.coefficient_bytes_read}
    nu._mmap.close();out.mkdir();result=dict(state='COMPLETE_SAME_GRID_SOURCE_COMPONENT_CHANGE',rows=rows,
        source_bytes=byte_counts,original_source_sha256=base['source_identity']['source_sha256'],
        accepted_source_sha256=meta['source_binary_sha256'],source_hash_verification='INHERITED_SEALS_NOT_REHASHED',
        caveat='different thermodynamic and population states; component integrals are not F2 field causal attribution',
        new_transport=0,new_SE=0,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
