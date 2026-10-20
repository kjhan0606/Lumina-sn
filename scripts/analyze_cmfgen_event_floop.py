#!/usr/bin/env python3
"""실제 한 주파수의 첫/종료 JIT를 같은 좌표·절대량·원식으로 비교한다."""
import argparse
import csv
import ctypes
import json
import struct
from pathlib import Path

import numpy as np

from analyze_cmfgen_mom_capture import capture,first_event
from analyze_cmfgen_formal_capture import formal_state,geometry,rays,compensated_moments
from analyze_cmfgen_initial_closure import angular_support
from analyze_cmfgen_sweep_context import floop_state
from prepare_cmfgen_mom_capture import sha
from solve_cmfgen_mom_kernel import completed_operator,first_repair_equation,linmon


def analyze(run,replay,build,output):
    if output.exists():raise ValueError('기존 실제 F-loop 결과 보존')
    inv=json.loads((build/'capture_first_v1/state_inventory.json').read_text())
    start=first_event(run/'MOM_FAULT.bin');control=floop_state(run/'FLOOP_STATE.bin')
    boundary=struct.unpack_from('=7d',(run/'AFFINE_INPUT.bin').read_bytes(),36)
    gp=run/'FORMAL_EVENT_geometry.bin';rp=run/'FORMAL_EVENT_rays_out.bin'
    exact,bounds=compensated_moments(geometry(gp),rays(rp));reduction={}
    for label,path in (('OMP16_capture',run/'FORMAL_EVENT_out.bin'),('CPU1_replay',replay/'first_replay_formal_out.bin')):
        saved=formal_state(path,inv['ray'])
        observed=np.array([saved['ray'][k+'NU_STORE']['values'] for k in 'JHKN'])
        error=abs(observed-exact)
        if np.any((bounds==0)&(error!=0)):raise ValueError('formal 모멘트 지지 불일치')
        fraction=np.divide(error,bounds,out=np.zeros_like(error),where=bounds>0)
        if np.max(fraction)>1:raise ValueError('formal 모멘트 IEEE 합산 상한 초과')
        reduction[label]=float(np.max(fraction))
    lib=ctypes.CDLL(str(build/'capture_first_v1/analysis_v2/mon_mid.so'))
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    lib.native_mid.argtypes=[ctypes.c_int,ptr,ptr,ptr]
    rows=list(csv.DictReader((replay/'FLOOP_REPLAY.csv').open()));cases=[];paths=[]
    for row in rows:
        jit=int(row['jit']);mom=replay/('JIT_%02d_MOM.bin'%jit);fpath=replay/('JIT_%02d_FORMAL.bin'%jit)
        s=capture(mom,inv);fs=formal_state(fpath,inv['ray']);j=s['modules']['jrel'];q=s['modules']['ray']
        v=lambda g,k:g[k]['values']
        r=v(j,'R');n=len(r);x=v(j,'GAM_RSQJNU_SAVE');h=v(j,'GAM_RSQHNU_SAVE')[:-1]
        mid=np.zeros(n-1);lib.native_mid(n,r,np.ascontiguousarray(x),mid)
        km=v(q,'KMID_ON_J')[:-1]*(x[:-1]+x[1:]);nm=v(q,'NMID_ON_J')[:-1]*(x[:-1]+x[1:])
        norm=v(j,'GAM_REL')*r*r;formal=norm*v(q,'JNU_STORE')
        diff=abs(x-formal)/np.maximum(abs(x),abs(formal));worst=int(np.argmax(diff))
        hin=q['HBC']*v(q,'JNU_STORE')[0]-v(q,'HNU_STORE')[0]
        nin=q['NBC']*v(q,'JNU_STORE')[0]-v(q,'NNU_STORE')[0]
        grad=v(j,'BETA')[0]*norm[0]*hin*v(j,'Q')[0]/v(j,'DTAU_H')[0] if s['args']['INCL_ADVEC_TERMS'] else 0.
        flux=grad/(v(j,'P_H')[0]+v(j,'W')[0])
        A=completed_operator(s)
        A[3,0]+=norm[0]*(v(j,'P_H')[0]*hin+v(j,'DELTA')[0]*(hin-boundary[2]+(v(j,'GAM_REL_SQ')[0]*(v(j,'SIGMA')[0]+1)-1)*(nin-boundary[3])))+grad
        A[3,1]-=flux
        base,b,adv=first_repair_equation(s,dict(ND=n,matrix=A,slope=v(q,'dlnGRSQJdlnR')))
        numer=abs(base@x+adv*linmon(x,r)-b);denom=abs(base)@abs(x)+abs(adv*linmon(x,r))+abs(b)
        res=float(np.max(np.divide(numer,denom,out=np.zeros(n),where=denom>0)))
        cases.append(dict(jit=jit,phase=s['phase'],F_change=float(row['F_change']),native_F_cap=int(row['native_F_cap']),
            positive=bool(np.all(x>0)),nonlinear_residual=res,angular_support=angular_support(mid,h,km,nm),J_difference_max=float(diff.max()),
            worst_depth_1based=worst+1,J_at_worst=float(x[worst]/norm[worst]),J_formal_at_worst=float(v(q,'JNU_STORE')[worst]),
            J_max=float(np.max(x/norm)),first_MOM_J_equal=bool(np.array_equal(x,start['raw'])) if jit==start['phase'][2] else None,
            first_MOM_H_equal=bool(np.array_equal(v(j,'GAM_RSQHNU_SAVE'),start['hraw'])) if jit==start['phase'][2] else None,
            formal_arg_raw_J_equal=bool(np.array_equal(fs['args']['JNU'],v(q,'JNU_STORE')))))
        paths.extend((mom,fpath))
    report=dict(scope='실제 최초 사건 뒤 같은 주파수의 원래 F-loop 종료; 전주파수/인구/물리 동질성 판정 아님',
        control=dict(phase=control['phase'],ACC_EDD_FAC=control['tolerance'],compute_eddfac=control['compute_eddfac'],coherent_es=control['coherent_es']),
        geometry_equal=(replay/'replay_geometry.bin').read_bytes()==(run/'FORMAL_EVENT_geometry.bin').read_bytes(),
        first_rays_equal=(replay/'first_replay_rays_out.bin').read_bytes()==(run/'FORMAL_EVENT_rays_out.bin').read_bytes(),
        first_formal_moments_roundoff_bound_fraction=reduction,
        cases=cases,last_jit=cases[-1]['jit'],last_jit_PSD_nonnegative=cases[-1]['angular_support']['minimum']>=0,
        evidence_sha256={str(p):sha(p) for p in paths+[run/'FLOOP_STATE.bin',run/'MOM_FAULT.bin',gp,rp,run/'FORMAL_EVENT_out.bin',replay/'first_replay_formal_out.bin',replay/'FLOOP_REPLAY.csv',Path(__file__)]},final_goal='NOT_JUDGED')
    with output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='evidence_sha256'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','replay','build','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();analyze(a.run,a.replay,a.build,a.output)
