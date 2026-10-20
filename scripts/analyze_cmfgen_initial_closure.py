#!/usr/bin/env python3
"""INIT의 비소비 영계수와 실제 물리 모멘트를 구분한다. 생산 결과 평가는 아니다."""
import argparse
import ctypes
import json
import struct
from pathlib import Path

import numpy as np

from analyze_cmfgen_mom_capture import capture,first_event
from prepare_cmfgen_mom_capture import sha
from solve_cmfgen_mom_kernel import high_precision_linear,relative_distance,tri_matrix,first_repair_equation,linmon_matrix,linmon


def angular_support(J,H,K,N):
    result=[]
    for sign in (-1,1):
        a=np.array([[1+sign*H/J,H/J+sign*K/J],[H/J+sign*K/J,K/J+sign*N/J]]).transpose(2,0,1)
        result.append(np.linalg.eigvalsh(a)[:,0])
    values=np.minimum(*result)
    i=int(np.argmin(values))
    return dict(minimum=float(values[i]),edge_1based=i+1,normalized_JHKN=[1,float(H[i]/J[i]),float(K[i]/J[i]),float(N[i]/J[i])])


def analyze(run,replay,build,output):
    if output.exists():raise ValueError("기존 초기 closure 분석 보존")
    inventory=json.loads((build/'capture_first_v1/state_inventory.json').read_text())
    old=capture(run/'MOM_KERNEL.bin',inventory)
    recovered=capture(replay/'INITIAL_PHYSICAL_CLOSURE.bin',inventory)
    again=capture(replay/'MOM_KERNEL.bin',inventory)
    e=first_event(run/'MOM_FAULT.bin')
    if not old['args']['INIT'] or e['kind']!=4:raise ValueError("초기 호출 포획이 아님")
    j=old['modules']['jrel'];q=recovered['modules']['ray'];n=e['ND']
    v=lambda g,k:g[k]['values']
    lib=ctypes.CDLL(str(build/'capture_first_v1/analysis_v2/mon_mid.so'))
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    lib.native_mid.argtypes=[ctypes.c_int,ptr,ptr,ptr]
    x=e['raw'];h=e['hraw'][:-1];mid=np.zeros(n-1)
    lib.native_mid(n,v(j,'R'),x,mid)
    coef=x[:-1]+x[1:]
    cleared=angular_support(mid,h,v(old['modules']['ray'],'KMID_ON_J')[:-1]*coef,v(old['modules']['ray'],'NMID_ON_J')[:-1]*coef)
    physical=angular_support(mid,h,v(q,'KMID_ON_J')[:-1]*coef,v(q,'NMID_ON_J')[:-1]*coef)
    M=tri_matrix(e['matrix']);b=e['matrix'][3];precise,_=high_precision_linear(M,b)
    base,rhs,adv=first_repair_equation(old,e)
    D,branches=linmon_matrix(x,v(j,'R'))
    root,_=high_precision_linear(base+adv[:,None]*D,rhs)
    denom=abs(base)@abs(root)+abs(adv*linmon(root,v(j,'R')))+abs(rhs)
    root_res=float(np.max(np.divide(abs(base@root+adv*linmon(root,v(j,'R'))-rhs),denom,out=np.zeros(n),where=denom>0)))
    root_h=v(j,'HU')[:-1]*root[1:]-v(j,'HL')[:-1]*root[:-1]
    hinc=struct.unpack_from('=7d',(run/'AFFINE_INPUT.bin').read_bytes(),36)[0]
    if old['args']['INCL_ADVEC_TERMS']:
        root_h[0]+=v(j,'BETA')[0]*v(j,'GAM_REL')[0]*v(j,'R')[0]**2*hinc*v(j,'Q')[0]/v(j,'DTAU_H')[0]/(v(j,'P_H')[0]+v(j,'W')[0])
    root_mid=np.zeros(n-1);lib.native_mid(n,v(j,'R'),root,root_mid)
    root_support=angular_support(root_mid,root_h,v(q,'KMID_ON_J')[:-1]*(root[:-1]+root[1:]),v(q,'NMID_ON_J')[:-1]*(root[:-1]+root[1:]))
    differences=[]
    for group in ('jrel','ray'):
        for k,one in old['modules'][group].items():
            two=again['modules'][group][k]
            if isinstance(one,dict):equal=np.array_equal(one['values'],two['values'],equal_nan=True)
            else:equal=one==two
            if not equal:differences.append(group+'.'+k)
    paths=[run/'MOM_CAPTURE.bin',run/'MOM_KERNEL.bin',run/'MOM_FAULT.bin',replay/'MOM_FAULT.bin',replay/'INITIAL_PHYSICAL_CLOSURE.bin',Path(__file__)]
    report=dict(scope='첫 주파수 JIT0의 실제 closure와 INIT 비소비 영계수 분리; 수렴/물리 목표 판정 아님',
        phase=e['phase'],frequency_native=e['frequency'],iterations=e['iteration'],cleared_coefficients_support=cleared,
        recovered_physical_closure_support=physical,native_vs_mp80_frozen_linear=relative_distance(x,precise),
        native_vs_mp80_active_branch=relative_distance(x,root),mp80_active_branch_residual=root_res,
        same_mp80_branches=branches==linmon_matrix(root,v(j,'R'))[1],mp80_active_branch_support=root_support,
        replay_entry_equal=(run/'MOM_CAPTURE.bin').read_bytes()==(replay/'MOM_CAPTURE.bin').read_bytes(),
        replay_event_equal=(run/'MOM_FAULT.bin').read_bytes()==(replay/'MOM_FAULT.bin').read_bytes(),kernel_different_fields=differences,
        evidence_sha256={str(p):sha(p) for p in paths},final_goal='NOT_JUDGED')
    with output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='evidence_sha256'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','replay','build','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();analyze(a.run,a.replay,a.build,a.output)
