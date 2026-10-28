#!/usr/bin/env python3
"""완료 F2 장의 Si/S 선에서 선형 적분과 격자점 사다리꼴을 대조한다."""
import argparse
import csv
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from lumina_model_linear_lines import LinearModelLines
from lumina_model_point_lines import ModelPointLines
from lumina_new_field_response import trapezoid_weights
from run_lumina_full_material_response import BB_RESULT

def run(point_library,out):
    root=Path('/gpfs/kjhan/cmfgen_runs/lumina_F2_response_lageunha_20261028-bBbs8U/run')
    report_path=root/'result.json';out=Path(out)
    if out.exists():raise FileExistsError('기존 결과 보존')
    if B.sha(report_path)!='b6d411ff28971492280275d2201e9d84a65315c0ad2613f04444e7b270b5fbea':raise ValueError('F2 결과 변경')
    report=json.loads(report_path.read_text())
    if not report['charge_converged'] or report['frequencies']!=1115020:raise ValueError('완료 F2 필요')
    prep=Path('/gpfs/kjhan/cmfgen_runs/lumina_accepted_sources_20261027-b5eCNv/run/boundary/preparation.json')
    if B.sha(prep)!=report['input_sha256'][str(prep)]:raise ValueError('F2 준비 입력 변경')
    meta=json.loads(prep.read_text());grid=Path(meta['grid']);field=Path(report['field']['J']['path'])
    if B.sha(BB_RESULT)!='b95b16e3401068bc8239116a87c872f392e970ec5780b7ce04d32cf51d1e9ec9':raise ValueError('원 선 roster 보고서 변경')
    bb=json.loads(BB_RESULT.read_text());roster=BB_RESULT.parent/'line-roster.npz';saved=root/'jbar.npy'
    level_paths=[Path(p) for p in report['input_sha256'] if p.endswith('/atomic/levels.csv')]
    if len(level_paths)!=1:raise ValueError('원자 준위 주소 불명확')
    levels=level_paths[0]
    library=root.parent/'build/lines.so'
    response=Path(report['selected_response']['root'])/'response.npz'
    for p,digest in ((grid,report['source_identity']['grid_sha256']),(field,report['field']['J']['sha256']),
       (saved,report['output_sha256']['jbar.npy']),(roster,bb['output_sha256']['line-roster.npz']),
       (library,report['input_sha256'][str(library)]),(levels,report['input_sha256'][str(levels)]),
       (response,report['selected_response']['response_sha256'])):
        if B.sha(p)!=digest:raise ValueError('입력 변경: '+p.name)
    with np.load(roster) as f:lines={k:f[k] for k in f.files}
    rows=list(csv.DictReader(levels.open()));zs=np.array([int(r['atomic_number']) for r in rows])
    with np.load(response) as f:columns=f['columns'];population=f['population'];fraction=f['fraction']
    index={tuple(map(int,c)):i for i,c in enumerate(columns)}
    level_columns=np.array([index[(int(r['atomic_number']),int(r['ion_number']),int(r['super_level']))] for r in rows])
    ids=np.flatnonzero(np.isin(zs[lines['lower']],[14,16]));sel={k:v[ids] for k,v in lines.items()}
    nu=np.load(grid);J=np.load(field,mmap_mode='r');old=np.load(saved,mmap_mode='r')
    model_paths=[Path(p) for p in meta['input_sha256'] if p.endswith('/MODEL')]
    if len(model_paths)!=1 or B.sha(model_paths[0])!=meta['input_sha256'][str(model_paths[0])]:raise ValueError('모형 신원 오류')
    profile=S.fixed_profile(model_paths[0]);width=trapezoid_weights(nu);results=[]
    for node in (30,33):
        j=np.array(J[node]);lin=LinearModelLines(sel,profile,library,nu,j[:,None],radiation_generation=1)
        linear=lin.contract(0,len(ids),radiation_generation=1);measured=linear['jbar'][:,0]
        np.testing.assert_allclose(measured,old[ids,node],rtol=2e-14,atol=0)
        producer=SimpleNamespace(lines=sel,profile=profile,dll=ctypes.CDLL(str(point_library)))
        producer.function=producer.dll.lumina_model_point_gaussian_jbar
        producer.ptr=ctypes.POINTER(ctypes.c_double)
        producer.function.argtypes=[ctypes.c_size_t,ctypes.c_size_t]+[producer.ptr]*4+[ctypes.c_double]*2+[producer.ptr]*2+[ctypes.POINTER(ctypes.c_uint8)]
        producer.function.restype=ctypes.c_int
        point=ModelPointLines.contract(producer,nu,width,j)
        lo,up=sel['lower'],sel['upper'];lc,uc=level_columns[lo],level_columns[up]
        bul=sel['A']*B.C**2/(2*B.H*sel['nu0']**3)
        lower=population[lc,node]*fraction[lo,node];upper=population[uc,node]*fraction[up,node]
        delta_j=point['jbar']-measured
        delta_net=(lower/sel['g_ratio']-upper)*bul*delta_j-upper*sel['A']*(point['profile_mass']-1.)
        forward=lower*bul/sel['g_ratio']*measured
        reverse=upper*(sel['A']+bul*measured)
        for z in (14,16):
            choose=zs[sel['lower']]==z;valid=choose&np.isin(point['status'],[1,2])&(measured>0)
            rel=np.abs(point['jbar'][valid]-measured[valid])/measured[valid]
            mass=point['profile_mass'][valid]
            inter=valid&(lc!=uc);delta=delta_net[inter];gross=forward[inter]+reverse[inter]
            row_delta=np.zeros(len(columns));np.add.at(row_delta,lc[inter],-delta);np.add.at(row_delta,uc[inter],delta)
            selected_cols=columns[:,0]==z
            if gross.sum()<=0 or abs(row_delta[selected_cols].sum())>1e-12*max(np.abs(delta).sum(),1.):
                raise ValueError('선 변화의 원소 내 입자 보존 오류')
            results.append(dict(node=node,Z=z,total=int(choose.sum()),valid=int(valid.sum()),
                out_of_grid=int(np.count_nonzero(choose&(point['status']==3))),
                unsampled=int(np.count_nonzero(choose&(point['status']==4))),
                abs_relative_percentiles=np.percentile(rel,[50,90,99,100]).tolist(),
                profile_mass_percentiles=np.percentile(mass,[0,10,50,90,100]).tolist(),
                perturbation=dict(inter_superlevel_lines=int(inter.sum()),
                    abs_line_net_change_sum=float(np.abs(delta).sum()),line_gross=float(gross.sum()),
                    relative_abs_line_change=float(np.abs(delta).sum()/gross.sum()),
                    max_abs_superlevel_residual_change=float(np.max(np.abs(row_delta[selected_cols]))),
                    total_signed_residual_change=float(row_delta[selected_cols].sum()))))
    out.mkdir();result=dict(state='COMPLETE_SAME_F2_GRID_POINT_VS_LINEAR',results=results,
        meaning='point quadrature on production grid is not native original integer line intervals',
        field_sha256=report['field']['J']['sha256'],grid_sha256=report['source_identity']['grid_sha256'],
        F2_jbar_sha256=report['output_sha256']['jbar.npy'],point_library_sha256=B.sha(point_library),
        F2_response_sha256=report['selected_response']['response_sha256'],
        new_transport=0,new_SE_solve=0,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point-library',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.point_library,a.out)
