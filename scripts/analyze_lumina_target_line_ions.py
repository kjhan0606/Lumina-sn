#!/usr/bin/env python3
"""실제 선 명부와 두 FL 인구에서 목표 대역 선 계수 변화를 이온별로 합산한다."""
import argparse
import csv
import ctypes as C
import json
from pathlib import Path
import struct
import numpy as np
import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from analyze_lumina_target_source_components import components
from lumina_native_columns import NativeSourceColumns,open_sealed_native
from prepare_lumina_outer_input import read_native
from run_lumina_full_material_response import BASE,BB_RESULT

def full(path,nl,ns):
    blob=path.read_bytes();tag,n,m=struct.unpack_from('<8sII',blob)
    if (tag,n,m)!=(b'LMATFL01',nl,ns) or len(blob)!=16+8*nl*ns:raise ValueError('FL 인구 신원/형상 오류')
    p=np.frombuffer(blob,dtype='<f8',offset=16).reshape(nl,ns).copy()
    if not np.isfinite(p).all() or np.any(p<0):raise ValueError('FL 인구 비유한/음수')
    return p

def run(library,out):
    out=Path(out)
    if out.exists():raise FileExistsError('기존 결과 보존')
    base=json.loads(BASE.read_text());config=json.loads(Path(base['config']).read_text())
    prep=Path('/gpfs/kjhan/cmfgen_runs/lumina_accepted_sources_20261027-b5eCNv/run/boundary/preparation.json')
    result=Path('/gpfs/kjhan/cmfgen_runs/lumina_F2_response_lageunha_20261028-bBbs8U/run/result.json')
    if B.sha(result)!='b6d411ff28971492280275d2201e9d84a65315c0ad2613f04444e7b270b5fbea':raise ValueError('F2 결과 변경')
    report=json.loads(result.read_text())
    if B.sha(prep)!=report['input_sha256'][str(prep)] or B.sha(BB_RESULT)!='b95b16e3401068bc8239116a87c872f392e970ec5780b7ce04d32cf51d1e9ec9':
        raise ValueError('준비/선 명부 보고서 변경')
    meta=json.loads(prep.read_text());bb=json.loads(BB_RESULT.read_text());line_path=BB_RESULT.parent/'line-roster.npz'
    old_pop=Path(config['native_full_population']);new_pop=Path(meta['full_population']);material=Path(meta['material']);grid=Path(meta['grid'])
    levels=Path(config['seed'])/'atomic/levels.csv'
    for path,digest in ((old_pop,base['input_sha256'][str(old_pop)]),(new_pop,meta['input_sha256'][str(new_pop)]),
      (material,meta['input_sha256'][str(material)]),(grid,meta['input_sha256'][str(grid)]),
      (levels,report['input_sha256'][str(levels)]),(line_path,bb['output_sha256']['line-roster.npz'])):
        if B.sha(path)!=digest:raise ValueError('입력 변경: '+path.name)
    model_paths=[Path(p) for p in meta['input_sha256'] if p.endswith('/MODEL')]
    if len(model_paths)!=1 or B.sha(model_paths[0])!=meta['input_sha256'][str(model_paths[0])]:raise ValueError('모형 변경')
    rows=list(csv.DictReader(levels.open()));zs=np.array([int(v['atomic_number']) for v in rows]);stages=np.array([int(v['ion_number'])+1 for v in rows])
    with np.load(line_path) as f:lines={k:f[k] for k in f.files}
    oldp=full(old_pop,len(rows),90);newp=full(new_pop,len(rows),90);physical,_=read_native(material)
    nu=np.load(grid,mmap_mode='r');profile=S.fixed_profile(model_paths[0]);width=profile['support_fraction']
    ptr=C.POINTER(C.c_double);bptr=C.POINTER(C.c_uint8);dll=C.CDLL(str(library));fn=dll.lumina_model_point_gaussian_jbar
    fn.argtypes=[C.c_size_t,C.c_size_t]+[ptr]*4+[C.c_double]*2+[ptr]*2+[bptr];fn.restype=C.c_int
    dtype=lambda x:np.ascontiguousarray(x,dtype='f8');results=[]
    with open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as old, \
        NativeSourceColumns(meta['source_binary'],nu,physical[:,0],physical[:,1],physical[:,2]) as new:
        if not np.array_equal(old.frequency,nu) or not np.array_equal(old.radius,physical[:,0]):raise ValueError('격자/반경 불일치')
        for node,wlo,whi in ((33,550.,650.),(30,1500.,2500.)):
            first=int(np.searchsorted(nu,B.C*1e8/whi));last=int(np.searchsorted(nu,B.C*1e8/wlo,side='right'))
            selected=np.flatnonzero((lines['nu0']*(1+width)>=nu[first])&(lines['nu0']*(1-width)<=nu[last-1]))
            center=dtype(lines['nu0'][selected]);weights=np.zeros(len(nu));delta=np.diff(nu)
            weights[:-1]+=.5*delta;weights[1:]+=.5*delta
            one=np.zeros(len(nu));one[first:last]=1.;sub=np.asarray(nu[first:last]);d=np.diff(sub)
            one[first]=(.5*d[0])/weights[first];one[last-1]=(.5*d[-1])/weights[last-1]
            mass=np.empty(len(selected));j=np.empty(len(selected));status=np.empty(len(selected),dtype='u1')
            if fn(len(nu),len(center),*[dtype(v).ctypes.data_as(ptr) for v in (nu,weights,one,center)],
                  profile['width_fraction'],width,j.ctypes.data_as(ptr),mass.ctypes.data_as(ptr),status.ctypes.data_as(bptr)):
                raise ValueError('선 대역 이산 적분 실패')
            if not np.isin(status,[1,2]).all() or np.any(j<0) or np.any(j>mass*(1+2e-15)):
                raise ValueError('선별 적분 형상/상태 오류')
            mass=j
            lower=lines['lower'][selected];upper=lines['upper'][selected];f=lines['f'][selected];A=lines['A'][selected];ratio=lines['g_ratio'][selected]
            if not np.array_equal(zs[lower],zs[upper]) or not np.array_equal(stages[lower],stages[upper]):raise ValueError('선 이온 주소 오류')
            opacity=np.pi*B.atomic.E_CGS**2/(B.ME*B.C);emission=B.H/(4*np.pi)
            c0=opacity*f*(oldp[lower,node]-ratio*oldp[upper,node])*mass
            c1=opacity*f*(newp[lower,node]-ratio*newp[upper,node])*mass
            e0=emission*center*A*oldp[upper,node]*mass;e1=emission*center*A*newp[upper,node]*mass
            integrals=[]
            for src,cval,eval in ((old,c0,e0),(new,c1,e1)):
                planes=components(src,node,first,last);target=nu[first:last]
                refchi=float(np.trapezoid(planes[4],target));refeta=float(np.trapezoid(planes[5],target))
                np.testing.assert_allclose([np.sum(cval,dtype=np.longdouble),np.sum(eval,dtype=np.longdouble)],
                                           [refchi,refeta],rtol=3e-10,atol=1e-250)
                integrals.append(dict(chi=refchi,eta=refeta))
            groups=[]
            for z,stage in sorted(set(zip(zs[lower],stages[lower]))):
                mask=(zs[lower]==z)&(stages[lower]==stage)
                groups.append(dict(Z=int(z),stage=int(stage),lines=int(mask.sum()),
                    delta_chi=float(np.sum((c1-c0)[mask],dtype=np.longdouble)),
                    delta_eta=float(np.sum((e1-e0)[mask],dtype=np.longdouble)),
                    abs_line_delta_chi=float(np.sum(abs(c1-c0)[mask],dtype=np.longdouble))))
            def describe(i):
                lo,up=int(lower[i]),int(upper[i])
                ion=(zs==zs[lo])&(stages==stages[lo]);ion_old=float(oldp[ion,node].sum());ion_new=float(newp[ion,node].sum())
                if min(ion_old,ion_new)<=0:raise ValueError('이온 총인구 없음')
                return dict(Z=int(zs[lo]),stage=int(stages[lo]),wavelength_A=float(B.C*1e8/center[i]),
                    lower=rows[lo]['configuration'],upper=rows[up]['configuration'],oscillator_strength=float(f[i]),
                    lower_population_old=float(oldp[lo,node]),lower_population_new=float(newp[lo,node]),
                    upper_population_old=float(oldp[up,node]),upper_population_new=float(newp[up,node]),
                    ion_population_old=ion_old,ion_population_new=ion_new,
                    lower_fraction_old=float(oldp[lo,node]/ion_old),lower_fraction_new=float(newp[lo,node]/ion_new),
                    band_profile_mass=float(mass[i]),delta_chi=float(c1[i]-c0[i]),delta_eta=float(e1[i]-e0[i]))
            top_lines_chi=[describe(int(i)) for i in np.argsort(abs(c1-c0))[-10:][::-1]]
            top_lines_eta=[describe(int(i)) for i in np.argsort(abs(e1-e0))[-10:][::-1]]
            results.append(dict(node=node,wavelength_A=[wlo,whi],selected_lines=len(selected),line_integrals=integrals,
                groups=groups,top_chi=sorted(groups,key=lambda q:abs(q['delta_chi']),reverse=True)[:8],
                top_eta=sorted(groups,key=lambda q:abs(q['delta_eta']),reverse=True)[:8],
                top_lines_chi=top_lines_chi,top_lines_eta=top_lines_eta))
    nu._mmap.close();out.mkdir();data=dict(state='COMPLETE_EXACT_DISCRETE_LINE_ION_DECOMPOSITION',results=results,
        old_population_sha256=base['input_sha256'][str(old_pop)],accepted_population_sha256=meta['input_sha256'][str(new_pop)],
        source_hash_verification='INHERITED_SEALS_NOT_REHASHED',library_sha256=B.sha(library),new_transport=0,new_SE=0,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');print(json.dumps(data))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.library,a.out)
