#!/usr/bin/env python3
"""실제 연결 실패의 절대량·독립해·안쪽 경계 반응을 구분한다. 수리 채택용이 아니다."""
import argparse
import ctypes
import json
import struct
from pathlib import Path

import numpy as np
from scipy.optimize import root

from analyze_cmfgen_mom_capture import capture,first_event
from analyze_cmfgen_initial_closure import angular_support
from prepare_cmfgen_mom_capture import sha
from solve_cmfgen_mom_kernel import first_repair_equation,high_precision_linear,linmon,linmon_matrix,relative_distance,tri_matrix


def analyze(run,replay,build,output):
    if output.exists():raise ValueError('기존 sweep 사건 분석 보존')
    inv=json.loads((build/'capture_first_v1/state_inventory.json').read_text())
    s=capture(run/'MOM_KERNEL.bin',inv);e=first_event(run/'MOM_FAULT.bin')
    a=s['args'];j=s['modules']['jrel'];q=s['modules']['ray']
    if a['INIT'] or a['N_TYPE'].strip()!='N_ON_J':raise ValueError('이 분석은 실제 비초기 N_ON_J 호출용')
    v=lambda group,k:group[k]['values']
    r=v(j,'R');n=len(r);x=e['raw'];H=e['hraw'][:-1]
    base,b,adv=first_repair_equation(s,e)
    D,branches=linmon_matrix(x,r);precise,_=high_precision_linear(base+adv[:,None]*D,b)
    lib=ctypes.CDLL(str(build/'capture_first_v1/analysis_v2/mon_mid.so'))
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    lib.native_mid.argtypes=[ctypes.c_int,ptr,ptr,ptr]
    def mid(y):
        out=np.zeros(n-1);lib.native_mid(n,r,np.ascontiguousarray(y),out);return out
    k=v(q,'KMID_ON_J')[:-1];nn=v(q,'NMID_ON_J')[:-1]
    formal=v(j,'GAM_REL')*r*r*v(q,'JNU_STORE')
    error=abs(x-formal)/np.maximum(abs(x),abs(formal));where=int(np.argmax(error))
    hminus=q['HMIN_IB']/q['JMIN_IB'];constant=q['HPLUS_IB']+hminus*q['JPLUS_IB']
    factor=v(j,'GAM_REL')[-1]*r[-1]**2
    oldH=a['DBB']/3/v(j,'CHI')[-1]
    incident=struct.unpack_from('=7d',(run/'AFFINE_INPUT.bin').read_bytes(),36)[0]
    flux_source=v(j,'BETA')[0]*v(j,'GAM_REL')[0]*r[0]**2*incident*v(j,'Q')[0]/v(j,'DTAU_H')[0]/(v(j,'P_H')[0]+v(j,'W')[0]) if a['INCL_ADVEC_TERMS'] else 0.
    def flux(y):
        h=v(j,'HU')[:-1]*y[1:]-v(j,'HL')[:-1]*y[:-1]+v(j,'HS')[:-1]*v(j,'GAM_RSQHNU_PREV')[:-1]
        h+=v(j,'EPS_PREV')[:-1]*(v(j,'GAM_RSQJNU_PREV')[:-1]+v(j,'GAM_RSQJNU_PREV')[1:])-v(j,'EPS')[:-1]*(y[:-1]+y[1:])
        h[0]+=flux_source
        return h
    def residual(M,rhs,y):
        numer=abs(M@y+adv*linmon(y,r)-rhs);denom=abs(M)@abs(y)+abs(adv*linmon(y,r))+abs(rhs)
        return float(np.max(np.divide(numer,denom,out=np.zeros(n),where=denom>0)))
    attempts=[]
    for name in ('unchanged','inner_half_boundary_row_only'):
        M=base.copy();rhs=b.copy()
        if name!='unchanged':M[-1,-1]+=hminus;rhs[-1]=factor*constant
        row_scale=abs(M)@abs(x)+abs(adv*linmon(x,r))+abs(rhs)
        answer=root(lambda y:(M@(y*x)+adv*linmon(y*x,r)-rhs)/row_scale,np.ones(n),
            jac=lambda y:(M+adv[:,None]*linmon_matrix(y*x,r)[0])*x[None,:]/row_scale[:,None],method='hybr',options=dict(xtol=1e-11,maxfev=300))
        y=answer.x*x
        item=dict(name=name,reported_success=bool(answer.success),positive=bool(np.all(y>0)),residual=residual(M,rhs,y),
            J_difference_max=float(np.max(abs(y-formal)/np.maximum(abs(y),abs(formal)))),
            J_inner_over_formal=float(y[-1]/formal[-1]),J_at_old_worst_over_formal=float(y[where]/formal[where]),
            H_inner_from_half=float(constant-hminus*y[-1]/factor))
        if np.all(y>0):item['angular_support']=angular_support(mid(y),flux(y),k*(y[:-1]+y[1:]),nn*(y[:-1]+y[1:]))
        attempts.append(item)
    paths=[run/'MOM_CAPTURE.bin',run/'MOM_KERNEL.bin',run/'MOM_FAULT.bin',run/'AFFINE_INPUT.bin',Path(__file__)]
    result=dict(scope='2089 첫 JIT의 원식/실제절대량 및 안쪽 경계 row-only 반응; 새 주파수 이력·채택 수리 아님',
        phase=e['phase'],iterations=e['iteration'],frequency_Hz=e['frequency']*1e15,
        native_vs_mp80_active_branch=relative_distance(x,precise),mp80_residual=residual(base,b,precise),same_mp80_branches=branches==linmon_matrix(precise,r)[1],
        angular_support=angular_support(mid(x),H,k*(x[:-1]+x[1:]),nn*(x[:-1]+x[1:])),
        absolute=dict(J_max=float(np.max(x/(v(j,'GAM_REL')*r*r))),J_formal_max=float(np.max(v(q,'JNU_STORE'))),
            worst_depth_1based=where+1,J_at_worst=float(x[where]/v(j,'GAM_REL')[where]/r[where]**2),J_formal_at_worst=float(v(q,'JNU_STORE')[where]),
            J_at_inner=float(x[-1]/factor),J_formal_at_inner=float(v(q,'JNU_STORE')[-1])),
        inner_boundary=dict(moment_H_diffusion=float(oldH),formal_H=float(q['HNU_AT_IB']),formal_Jplus=float(q['JPLUS_IB']),formal_Jminus=float(q['JMIN_IB']),
            formal_Hplus=float(q['HPLUS_IB']),formal_Hminus_magnitude=float(q['HMIN_IB']),hminus=float(hminus),constant=float(constant),
            half_J_sum_relative=float((q['JPLUS_IB']+q['JMIN_IB'])/v(q,'JNU_STORE')[-1]-1),
            half_H_difference_relative=float((q['HPLUS_IB']-q['HMIN_IB'])/q['HNU_AT_IB']-1)),
        row_only_response=attempts,replay_entry_equal=(run/'MOM_CAPTURE.bin').read_bytes()==(replay/'MOM_CAPTURE.bin').read_bytes(),
        replay_kernel_equal=(run/'MOM_KERNEL.bin').read_bytes()==(replay/'MOM_KERNEL.bin').read_bytes(),replay_event_equal=(run/'MOM_FAULT.bin').read_bytes()==(replay/'MOM_FAULT.bin').read_bytes(),
        evidence_sha256={str(p):sha(p) for p in paths},limitations='row-only변경은경계 공간gradient/마지막 내부flux/Q를다시조립하지않음; 생산수리로채택불가',final_goal='NOT_JUDGED')
    with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','replay','build','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();analyze(a.run,a.replay,a.build,a.output)
