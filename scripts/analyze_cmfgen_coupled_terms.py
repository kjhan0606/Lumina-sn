#!/usr/bin/env python3
"""같은 공동 종료 packet의 주파수 이류/공간 flux 항을 실제 광선과 대조한다."""
import argparse
import ctypes
import json
from pathlib import Path
import struct

import numpy as np

from analyze_cmfgen_mom_capture import capture
from analyze_cmfgen_formal_capture import geometry,rays,angular_integral
from prepare_cmfgen_mom_capture import sha
from run_cmfgen_coupled_packet import matrix_packet
from solve_cmfgen_mom_kernel import first_repair_equation,linmon


def angular_blue_source(g,intensities):
    """REL_VARIABLES가 소비한 실제 B_P/M·Iprev의 0/1차 각적분."""
    result=np.zeros((2,g['ND']))
    for ip,(geo,ray) in enumerate(zip(g['rays'],intensities['rays'])):
        for i in range(min(g['ND'],g['NP']-ip)):
            link=geo['LNK'][i]-1
            for side in ('P','M'):
                term=geo['B_'+side][link]*ray['I_'+side][link]
                for order,label in enumerate(('J','H')):
                    result[order,i]+=term*g['weights'][label+'QW_'+side][i,ip]
    return result


def analyze(run_root,trial,build,out):
    if out.exists():raise ValueError('기존 항별 분석 보존')
    parent_record=json.loads((build/'sweep-build.json').read_text())
    inv=json.loads((Path(parent_record['mom_build'])/'state_inventory.json').read_text())
    state=capture(trial/'TRIAL_KERNEL.bin',inv);packet=matrix_packet(trial/'TRIAL_MATRIX.bin')
    j,q=state['modules']['jrel'],state['modules']['ray'];v=lambda k:j[k]['values']
    r=v('R');n=len(r);physical=v('GAM_REL')*r*r
    g=geometry(trial/'TRIAL_GEOMETRY.bin')
    blue_rays=rays(run_root/'FORMAL_EVENT_rays_in.bin');current_rays=rays(trial/'TRIAL_RAYS_OUT.bin')
    blue=sum(angular_integral(g,blue_rays));current=sum(angular_integral(g,current_rays))
    bs=angular_blue_source(g,blue_rays);cs=angular_blue_source(g,current_rays)
    lib=ctypes.CDLL(str(Path(parent_record['mom_build'])/'analysis_v2/mon_mid.so'))
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    lib.native_mid.argtypes=[ctypes.c_int,ptr,ptr,ptr]
    def middle(x):
        ans=np.zeros(n-1);lib.native_mid(n,r,np.ascontiguousarray(x),ans);return ans
    X=physical*current[0];H=middle(physical*current[1])
    JP=v('GAM_RSQJNU_PREV');HP=v('GAM_RSQHNU_PREV')[:-1]
    base,b,adv=first_repair_equation(state,packet)
    source=capture(trial/'TRIAL_ENTRY.bin',inv)['args']['JNU_SM']
    source_x=physical*source
    grad=adv*linmon(X,r)
    absorption=v('DTAUONQ')*X+grad-v('DTAUONQ')*v('COH_VEC')*source_x-v('DTAUONQ')*v('SOURCE')*r*r
    current0=v('PSI')*X;native_blue0=v('PSIPREV')*JP
    ray_blue0=v('DTAUONQ')*r*r/v('CHI_J')/state['args']['dLOG_NU']*bs[0]
    ray_current0=v('DTAUONQ')*r*r/v('CHI_J')/state['args']['dLOG_NU']*cs[0]
    ray_blue1=middle(r*r*bs[1])/(.5*(v('CHI_H')[:-1]+v('CHI_H')[1:]))/state['args']['dLOG_NU']
    ray_current1=middle(r*r*cs[1])/(.5*(v('CHI_H')[:-1]+v('CHI_H')[1:]))/state['args']['dLOG_NU']
    den1=v('P_H')[:-1]+v('W')[:-1]
    current1=v('W')[:-1]*H+den1*v('EPS')[:-1]*(X[:-1]+X[1:])
    native_blue1=den1*(v('HS')[:-1]*HP+v('EPS_PREV')[:-1]*(JP[:-1]+JP[1:]))
    bc=struct.unpack_from('=7d',(trial/'TRIAL_BOUNDARY.bin').read_bytes(),36)
    flux_const=v('BETA')[0]*physical[0]*bc[0]*v('Q')[0]/v('DTAU_H')[0]/den1[0]
    spatial1=den1*(v('HU')[:-1]*X[1:]-v('HL')[:-1]*X[:-1]);spatial1[0]+=den1[0]*flux_const
    Hpred=spatial1/den1+native_blue1/den1-v('EPS')[:-1]*(X[:-1]+X[1:])
    E0_moment=Hpred[:-1]-Hpred[1:]+absorption[1:-1]+current0[1:-1]-native_blue0[1:-1]
    matrix_f=base@X+adv*linmon(X,r)-b
    source_lag=v('DTAUONQ')*v('COH_VEC')*(X-source_x);source_lag[[0,-1]]=0
    denom=(abs(Hpred[:-1])+abs(Hpred[1:])+abs(absorption[1:-1])+abs(current0[1:-1])+abs(native_blue0[1:-1]))
    identity=float(np.max(abs(E0_moment-(matrix_f+source_lag)[1:-1])/denom))
    if identity>1e-12:raise ValueError('0차 원시식/소거식 재조립 불일치')
    def diff(a,z):
        return abs(a-z)/np.maximum(abs(a),abs(z))
    native0=(H[:-1]-H[1:]+absorption[1:-1]+current0[1:-1]-native_blue0[1:-1])
    ray0=(H[:-1]-H[1:]+absorption[1:-1]+current0[1:-1]-ray_blue0[1:-1])
    native1=v('P_H')[:-1]*H-spatial1+current1-native_blue1
    ray1=v('P_H')[:-1]*H-spatial1+current1-ray_blue1
    ray_pair0=H[:-1]-H[1:]+absorption[1:-1]+ray_current0[1:-1]-ray_blue0[1:-1]
    ray_pair1=v('P_H')[:-1]*H-spatial1+ray_current1-ray_blue1
    net0=ray_current0[1:-1]-ray_blue0[1:-1]
    net1=ray_current1-ray_blue1
    physical0=abs(H[:-1]-H[1:])+abs(absorption[1:-1])+abs(net0)
    physical1=abs(v('P_H')[:-1]*H)+abs(spatial1)+abs(net1)
    denom0=(abs(H[:-1])+abs(H[1:])+abs(absorption[1:-1])+abs(current0[1:-1])+abs(ray_blue0[1:-1])+abs(native_blue0[1:-1]))
    denom1=abs(v('P_H')[:-1]*H)+abs(spatial1)+abs(current1)+abs(native_blue1)+abs(ray_blue1)
    summary=dict(
        primitive_matrix_identity=identity,
        zeroth_frequency_current_ray_vs_moment_max=float(np.max(diff(current0[1:-1],ray_current0[1:-1]))),
        first_frequency_current_ray_vs_moment_max=float(np.max(diff(current1,ray_current1))),
        zeroth_primitive_native_blue_max=float(np.max(abs(native0)/denom0)),
        zeroth_primitive_ray_blue_max=float(np.max(abs(ray0)/denom0)),
        first_primitive_native_blue_max=float(np.max(abs(native1)/denom1)),
        first_primitive_ray_blue_max=float(np.max(abs(ray1)/denom1)),
        ray_pair_zeroth_net_balance_max=float(np.max(np.divide(abs(ray_pair0),physical0,out=np.zeros(n-2),where=physical0>0))),
        ray_pair_first_net_balance_max=float(np.max(np.divide(abs(ray_pair1),physical1,out=np.zeros(n-1),where=physical1>0))))
    samples=[]
    for depth in (28,35,36,37,49,61):
        i=depth-1
        samples.append(dict(depth_1based=depth,Jray=float(current[0,i]),zeroth_space_H_div=float(H[i-1]-H[i]),
            zeroth_absorption_source_advection=float(absorption[i]),zeroth_current=float(current0[i]),
            zeroth_blue_moment=float(native_blue0[i]),zeroth_blue_ray=float(ray_blue0[i]),zeroth_current_ray=float(ray_current0[i]),
            first_space_gradient=float(spatial1[i]),first_local_H=float(v('P_H')[i]*H[i]),first_current=float(current1[i]),
            first_blue_moment=float(native_blue1[i]),first_blue_ray=float(ray_blue1[i]),first_current_ray=float(ray_current1[i])))
        samples[-1].update(ray_pair_zeroth_net_balance=float(abs(ray_pair0[i-1])/physical0[i-1]),
            ray_pair_first_net_balance=float(abs(ray_pair1[i])/physical1[i]))
    paths=[trial/'TRIAL_MATRIX.bin',trial/'TRIAL_KERNEL.bin',trial/'TRIAL_RAYS_OUT.bin',run_root/'FORMAL_EVENT_rays_in.bin',Path(__file__)]
    report=dict(scope='같은 공동 종료 packet의 source-matched 원시식과 실제 B·I 각적분; 생산 수리/물리합격 아님',
        trial=str(trial),summary=summary,samples=samples,input_sha256={str(p):sha(p) for p in paths},final_goal='NOT_JUDGED')
    with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(summary=summary,samples=samples),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','trial','build','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();analyze(a.run,a.trial,a.build,a.out)
