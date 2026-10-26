#!/usr/bin/env python3
"""실제 native49 물질/동결 J를 C 점값 소비율과 대조한다.

33개 선택 열의 전역 가중치만 합한다. 전대역율·SE/RE 해·스펙트럼 판정 아님.
단위 upper-SL 인구로 inverse를 만들며 실제 미소 인구로 나누지 않는다.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cmfgen_model_continuum as B
import lumina_native_columns as N
import lumina_native90_rates as R


class Rates(C.Structure):
    _fields_=[(key,C.c_double) for key in ('photo','spontaneous','stimulated',
        'heat_photo','heat_spontaneous','heat_stimulated')]


def run(base_path,j_path,library):
    base_path,j_path,library=map(Path,(base_path,j_path,library))
    if B.sha(base_path)!='f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab':
        raise ValueError('봉인 native 입력 계획 불일치')
    base=json.loads(base_path.read_text());c=json.loads(Path(base['config']).read_text())
    plan=json.loads(Path(c['plan']).read_text())
    report_path=Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
    if B.sha(report_path)!=c['reuse_sources']['boundary_sources']['report_sha256']:
        raise ValueError('봉인 생산 계수 report 불일치')
    report=json.loads(report_path.read_text())
    lib=C.CDLL(str(library.resolve()));fn=lib.bf_rate_point_contract
    ptr=C.POINTER(C.c_double)
    fn.argtypes=[C.c_size_t]+[ptr]*6+[C.c_double,C.POINTER(Rates)]
    fn.restype=C.c_int
    with N.open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as source:
        rows,models,routes,state,do_dis,check,smoothing=R.material_state(c,plan,source,report)
        indices=np.unique(np.linspace(0,source.nf-1,33).astype('i8'))
        full_nu=source.frequency;nu=np.ascontiguousarray(full_nu[indices]);node=49
        widths=np.zeros_like(nu)
        left=indices>0;right=indices<len(full_nu)-1
        widths[left]+=.5*(full_nu[indices[left]]-full_nu[indices[left]-1])
        widths[right]+=.5*(full_nu[indices[right]+1]-full_nu[indices[right]])
        with N.FrozenNativeField(j_path,source,source.radius,source.frequency) as field:
            j=np.ascontiguousarray(field.read(np.array([node]),indices)[0])
        single=dict(state)
        for key in ('full','occupation','superlevel'):
            single[key]=state[key][:,node:node+1]
        for key in ('temperature','ne'):
            single[key]=state[key][node:node+1]
        members={}
        for l,row in enumerate(rows):
            key=(int(row['atomic_number']),int(row['ion_number']),int(row['super_level']))
            members.setdefault(key,[]).append(l)
        fractions=np.zeros(len(rows));t=float(single['temperature'][0]);ne=float(single['ne'][0])
        for levels in members.values():
            energy=np.array([float(rows[l]['energy_eV']) for l in levels])
            g=np.array([float(rows[l]['g']) for l in levels])
            w=g*np.exp(-(energy-energy.min())*B.EV/(B.KB*t))*single['occupation'][levels,0]
            fractions[levels]=w/w.sum()
        selected=[]
        for key in ((14,2),(14,3),(16,2),(16,3)):
            candidates=[r for r in routes if (r.Z,r.stage)==key and r.target>=0]
            score=lambda r: float((r.sigma(nu)*j*widths/(B.H*nu)).sum())
            primary=max(candidates,key=score)
            excited=[r for r in candidates if r.target_superlevel>0]
            selected.append(primary)
            if excited:
                alternate=max(excited,key=score)
                if alternate is not primary:selected.append(alternate)
        results=[]
        photon=4*np.pi*widths/(B.H*nu)
        for route in selected:
            lower,upper=route.lower,route.target
            sigma=route.sigma(nu)
            effective_upper=fractions[upper]*single['occupation'][lower]/single['occupation'][upper]
            coef=B.milne_route(nu,sigma,np.ones(1),effective_upper,
                single['temperature'],single['ne'],float(rows[lower]['g']),
                float(rows[upper]['g']),route.threshold_Hz,1.)
            arrays=[np.ascontiguousarray(x,dtype='f8') for x in
                    (nu,widths,j,coef[0][0],coef[1][0],coef[2][0])]
            result=Rates()
            rc=fn(len(nu),*[a.ctypes.data_as(ptr) for a in arrays],route.threshold_Hz,C.byref(result))
            if rc:raise ValueError('실제 점값 C 적분 실패')
            physical=B.continuum_block([route],rows,single,nu)
            forward=result.photo*single['full'][lower,0]*route.probability
            target_population=single['superlevel'][single['sl_index'][route.Z,route.stage,route.target_superlevel],0]
            inverse=(result.spontaneous+result.stimulated)*target_population*route.probability
            want_forward=float((physical['chi_abs'][0]*j*photon).sum())
            want_inverse=float(((physical['chi_stim'][0]*j+physical['eta_spont'][0])*photon).sum())
            if not np.isclose(forward,want_forward,rtol=3.e-12,atol=1.e-290):
                raise ValueError('C 점값 광흡수/생산 PHOT 불일치')
            if not np.isclose(inverse,want_inverse,rtol=3.e-12,atol=1.e-290):
                raise ValueError('C 점값 역재결합/생산 Milne 불일치')
            # J=0에서도 단위 upper 입자의 자발 재결합은 소멸하지 않는다.
            dark=np.zeros_like(j);arrays[2]=dark;zero=Rates()
            if fn(len(nu),*[a.ctypes.data_as(ptr) for a in arrays],route.threshold_Hz,C.byref(zero)):
                raise ValueError('J=0 점값 C 적분 실패')
            if zero.photo!=0 or zero.stimulated!=0 or zero.spontaneous!=result.spontaneous:
                raise ValueError('J=0 역재결합 소실')
            results.append(dict(Z=route.Z,stage=route.stage,lower=lower,upper=upper,
                target_superlevel=route.target_superlevel,probability=route.probability,
                photo=forward,photo_python=want_forward,reverse=inverse,reverse_python=want_inverse,
                per_upper_spontaneous=result.spontaneous))
        return dict(state='PASS',scope='native49/33선택열의 bound PHOT/Milne C 소비 대조',
            native_node=node,samples=len(indices),indices=indices.tolist(),routes=results,
            source_identity=source.identity,full_population_check=check,
            library_sha256=B.sha(library),script_sha256=B.sha(Path(__file__)),
            J_sample_sha256=hashlib.sha256(j.tobytes()).hexdigest(),
            complete_radiation_integral=False,material_updates=0,new_transport=0,
            pseudo_continuum_tested=False,final_goal='NOT_MET')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True)
    p.add_argument('--j',required=True);p.add_argument('--library',required=True)
    p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError('기존 시험 결과 보존')
    result=run(args.base,args.j,args.library)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('BF_POINT_NATIVE_TEST PASS routes='+str(len(result['routes'])))
