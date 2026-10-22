#!/usr/bin/env python3
"""실제 715Å packet에서 봉인 Bézier SC의 국소·전체 공간광선을 독립 재현한다.

청색 이전 강도·원 χ/η·기하·현재 상류 강도를 고정한다. 전 대역의
연산자 기여율이나 새 물리 계산은 아니다.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze_cmfgen_center_packet import INVENTORY
from analyze_cmfgen_formal_capture import formal_state, geometry, rays
from prepare_cmfgen_mom_capture import sha


def e_coeff(tau):
    if tau>0.01:
        ee=np.exp(-tau)
        e0=1.-ee
        e1=tau-e0
        e2=tau*tau-2.*e1
    else:
        e2=tau**3*(1.-.25*tau*(1.-.2*tau*(1.-tau/6.*(1.-tau/7.))))/3.
        e1=.5*(tau*tau-e2)
        e0=tau-e1
        ee=1.-e0
    return ee,e0,e1,e2


def bezier_source(sup,target,far,tau,neighbor):
    if tau<=0:
        return 0.
    if tau<=.5:
        coefficients=[]
        for i in range(13):
            factorial=math.factorial(i)
            coefficients.append(np.array([
                1/(factorial*(i+3)),
                2/(factorial*(i+2)*(i+3)),
                2/(factorial*(i+1)*(i+2)*(i+3))],dtype='f8'))
        weights=coefficients[12]
        for i in range(11,-1,-1):
            weights=-tau*weights+coefficients[i]
        weights=tau*weights
    else:
        inverse=1/tau
        ee=np.exp(-tau)
        weights=np.array([
            2*inverse*inverse-ee*(1+2*inverse+2*inverse*inverse),
            2*inverse-4*inverse*inverse+ee*(2*inverse+4*inverse*inverse),
            1-2*inverse+2*inverse*inverse-2*ee*inverse*inverse])
    if neighbor<=0:
        control=.5*sup+.5*target
    else:
        if tau>=neighbor:
            ratio=neighbor/tau
            w=1/(1+ratio)
        else:
            ratio=tau/neighbor
            w=ratio/(1+ratio)
        base=.5*(1-w)*sup+.5*(1+w)*target
        term=.5*w*(far-target)
        lo=min(sup,target);hi=max(sup,target)
        if term==0:
            control=min(hi,max(lo,base))
        else:
            distance=max(0.,base-lo) if term>0 else max(0.,hi-base)
            if tau>neighbor:
                ratio=neighbor/tau
                if abs(term)>=distance*ratio:
                    distance=np.finfo(float).max
                else:
                    distance=abs(term)/ratio
            else:
                distance=abs(term)*(tau/neighbor)
            if term>0:
                control=lo if distance>=base-lo else base-distance
            else:
                control=hi if distance>=hi-base else base+distance
    return float(np.dot(weights,[sup,control,target]))


def local_sc(chi,eta,previous,geometry_ray,current,k,dlognu,direction,mode='bezier'):
    """0-based 내부점 k의 원 CMFGEN 한 단계; 상류 현재 강도만 사용한다."""
    n=len(current)
    inside=(2<=k<=n-2) if direction=='P' else (1<=k<=n-2)
    if not inside or not 0<dlognu or len(chi)<n or len(eta)<n:
        raise ValueError('3점 내부 stencil과 유한 dlogν 필요')
    b=geometry_ray['B_'+direction][:n]
    nu_dnu=1./dlognu
    tau_chi=chi[:n]+(nu_dnu+3.)*b
    source=(eta[:n]+nu_dnu*b*previous[:n])/tau_chi
    s=geometry_ray['S_'+direction][:n]
    gap=(s[:-1]-s[1:]) if direction=='P' else (s[1:]-s[:-1])
    dtau=.5*gap*(tau_chi[:-1]+tau_chi[1:])
    if not np.isfinite(source[k-1:k+2]).all() or not np.isfinite(dtau[k-1:k+1]).all():
        raise ValueError('원 SC 국소 입력 비유한')
    if direction=='P':
        near,far=dtau[k],dtau[k-1]
        ee,e0,e1,e2=e_coeff(near)
        if mode=='bezier':
            return current[k+1]*ee+bezier_source(source[k+1],source[k],
                                                  source[k-1],near,far)
        if mode!='parabolic':
            raise ValueError('SC source 모드 오류')
        alpha=(e2-near*e1)/(far*(near+far))
        beta=((near+far)*e1-e2)/(far*near)
        gamma=e0+(e2-(far+2.*near)*e1)/(near*(near+far))
        contribution=alpha*source[k-1]+beta*source[k]+gamma*source[k+1]
        if contribution<0:
            extrap=source[k]+far*(source[k]-source[k+1])/near
            contribution=alpha*extrap+beta*source[k]+gamma*source[k+1]
        return current[k+1]*ee+contribution
    if direction=='M':
        near,far=dtau[k-1],dtau[k]
        ee,e0,e1,e2=e_coeff(near)
        if mode=='bezier':
            return current[k-1]*ee+bezier_source(source[k-1],source[k],
                                                  source[k+1],near,far)
        if mode!='parabolic':
            raise ValueError('SC source 모드 오류')
        alpha=e0+(e2-(far+2.*near)*e1)/(near*(far+near))
        beta=((far+near)*e1-e2)/(near*far)
        gamma=(e2-near*e1)/(far*(far+near))
        contribution=alpha*source[k-1]+beta*source[k]+gamma*source[k+1]
        if contribution<0:
            extrap=source[k]+far*(source[k]-source[k-1])/near
            contribution=alpha*source[k-1]+beta*source[k]+gamma*extrap
        return current[k-1]*ee+contribution
    raise ValueError('방향 P/M 필요')


def extended_material(args,g):
    """원 formal의 ND_ADD 외층 χ/η 외삽(삽입 광선점 없는 경우)."""
    if g['RAY_POINTS_INSERTED'] or g['ND_EXT']!=g['ND']+g['ND_ADD']:
        raise ValueError('지원되지 않는 원 광선 기하')
    chi=np.asarray(args['CHI'],dtype='f8')
    eta=np.asarray(args['ETA'],dtype='f8')
    esec=np.asarray(args['ESEC'],dtype='f8')
    radius=np.asarray(args['R'],dtype='f8')
    ext_radius=np.asarray(g['R_EXT'],dtype='f8')
    add=g['ND_ADD'];lo=g['IDMIN']-1;hi=g['IDMAX']-1
    chi_ext=np.empty(g['ND_EXT']);eta_ext=np.empty_like(chi_ext)
    chi_ext[add:]=chi;eta_ext[add:]=eta
    if add:
        ratio=radius[lo]/ext_radius[:add]
        denominator=np.log(radius[lo]/radius[hi])
        e_power=max(2.,np.log(esec[hi]/esec[lo])/denominator)
        if chi[lo]<=esec[lo] or chi[hi]<=esec[hi]:
            chi_ext[:add]=chi[lo]*ratio**e_power
        else:
            power=max(2.,np.log((chi[hi]-esec[hi])/(chi[lo]-esec[lo]))/denominator)
            chi_ext[:add]=(chi[lo]-esec[lo])*ratio**power+esec[lo]*ratio**e_power
        eta_power=max(3.5,np.log(eta[hi]/eta[lo])/denominator)
        eta_ext[:add]=np.maximum(eta[lo]*ratio**eta_power,1e-280)
    if not np.isfinite(chi_ext).all() or not np.isfinite(eta_ext).all():
        raise ValueError('외층 물질 외삽 비유한')
    return chi_ext,eta_ext


def replay_ray(args,geom,previous,chi,eta,ip):
    """봉인 Bézier·사다리꼴 tau·DIFFUSION 경계의 실제 한 광선 호출."""
    n=len(geom['R']);dlognu=float(args['dLOG_NU'])
    chi=chi[:n];eta=eta[:n]
    im=np.zeros(n);plus=np.zeros(n)
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        for k in range(1,n-1):
            im[k]=local_sc(chi,eta,previous['I_M'],geom,im,k,dlognu,'M')
        b=geom['B_M'][:n];alpha=1/dlognu
        effective=chi+(alpha+3)*b
        source=(eta+alpha*b*previous['I_M'][:n])/effective
        tau=.5*(geom['S_M'][n-1]-geom['S_M'][n-2])*(effective[n-1]+effective[n-2])
        ee,e0,e1,_=e_coeff(tau)
        im[n-1]=im[n-2]*ee+(e0-e1/tau)*source[n-2]+(e1/tau)*source[n-1]
        if ip<=args['NC']:
            if args['INNER_BND_METH'].strip()!='DIFFUSION':
                raise ValueError('봉인 광선 경계가 DIFFUSION이 아님')
            plus[n-1]=args['B_PLANCK']+geom['MU_P'][n-1]*args['DBB']/args['CHI'][-1]
        else:
            plus[n-1]=im[n-1]
        for k in range(n-2,1,-1):
            plus[k]=local_sc(chi,eta,previous['I_P'],geom,plus,k,dlognu,'P')
        b=geom['B_P'][:n]
        effective=chi+(alpha+3)*b
        source=(eta+alpha*b*previous['I_P'][:n])/effective
        for k in (1,0):
            tau=.5*(geom['S_P'][k]-geom['S_P'][k+1])*(effective[k]+effective[k+1])
            ee,e0,e1,_=e_coeff(tau)
            plus[k]=plus[k+1]*ee+(e1/tau)*source[k]+(e0-e1/tau)*source[k+1]
    if not np.isfinite(im).all() or not np.isfinite(plus).all():
        raise ValueError('독립 전체 광선 해 비유한')
    return plus,im


def check(packet, out=None, full_ray=False, all_jit=False):
    if out is not None and out.exists():
        raise FileExistsError('기존 packet 대조 결과 보존')
    inventory=json.loads(INVENTORY.read_text())['ray_inventory']
    cases=[]
    phases=tuple((index,jit) for index in (44360,44361)
                 for jit in ((0,1,2) if all_jit else (2,)))
    for index,jit in phases:
        prefix=packet/f'FORMAL_{index}_{jit:02d}'
        state=formal_state(Path(str(prefix)+'_out.bin'),inventory)
        g=geometry(Path(str(prefix)+'_geometry.bin'))
        previous=rays(Path(str(prefix)+'_rays_in.bin'))
        actual=rays(Path(str(prefix)+'_rays_out.bin'))
        if g['RAY_POINTS_INSERTED'] or state['args']['INITIALIZE']:
            raise ValueError('이 검사는 삽입 없는 실제 packet만 지원')
        chi=np.full(g['ND_EXT'],np.nan)
        eta=np.full(g['ND_EXT'],np.nan)
        chi[g['ND_ADD']:]=np.asarray(state['args']['CHI'])
        eta[g['ND_ADD']:]=np.asarray(state['args']['ETA'])
        dlognu=float(state['args']['dLOG_NU'])
        errors=[];relative=[];parabolic=[];count=0;details=[]
        for ip,(geom,prior,now) in enumerate(zip(g['rays'],previous['rays'],actual['rays'])):
            for native_id in (50,51):
                if ip>=g['NP']-native_id+1:
                    continue
                k=int(geom['LNK'][native_id-1])-1
                if not 2<=k<=len(now['I_P'])-2:
                    continue
                for direction in ('P','M'):
                    with np.errstate(invalid='ignore',divide='ignore'):
                        wanted=local_sc(chi,eta,prior['I_'+direction],geom,
                                        now['I_'+direction],k,dlognu,direction)
                        old=local_sc(chi,eta,prior['I_'+direction],geom,
                                     now['I_'+direction],k,dlognu,direction,'parabolic')
                    seen=float(now['I_'+direction][k])
                    error=abs(wanted-seen)
                    errors.append(error)
                    relative.append(error/max(abs(wanted),abs(seen),1e-30))
                    parabolic.append(abs(old-seen)/max(abs(old),abs(seen),1e-30))
                    details.append((parabolic[-1],ip+1,native_id,direction,k+1,old,seen))
                    count+=1
        if count==0:
            raise ValueError('국소 SC 내부 광선 표본 없음')
        if max(relative)>1e-12:
            raise ValueError('봉인 Bézier SC 국소 광선 재현 실패')
        worst=max(details)
        case=dict(index=index,jit=jit,native_indices_one_based=[50,51],
                          compared_ray_directions=count,
                          max_absolute_error=max(errors),max_scaled_error=max(relative),
                          median_scaled_error=float(np.median(relative)),
                          parabolic_max_scaled_error=max(parabolic),
                          parabolic_median_scaled_error=float(np.median(parabolic)),
                          parabolic_worst=dict(scaled_error=float(worst[0]),ip=worst[1],
                                               id=worst[2],direction=worst[3],
                                               ray_point=worst[4],
                                               parabolic=float(worst[5]),actual=float(worst[6])))
        if full_ray:
            chi_ext,eta_ext=extended_material(state['args'],g)
            ray_errors=[];ray_relative=[]
            for ip,(geom,prior,now) in enumerate(zip(g['rays'],previous['rays'],actual['rays']),1):
                calculated=replay_ray(state['args'],geom,prior,chi_ext,eta_ext,ip)
                for direction,value in zip(('I_P','I_M'),calculated):
                    error=np.abs(value-now[direction])
                    ray_errors.append(float(np.max(error)))
                    ray_relative.append(float(np.max(error/np.maximum(
                        np.maximum(abs(value),abs(now[direction])),1e-30))))
            case['full_ray_max_absolute_error']=max(ray_errors)
            case['full_ray_max_scaled_error']=max(ray_relative)
            if case['full_ray_max_scaled_error']>1e-10:
                raise ValueError('봉인 Bézier 전체 광선 재현 실패')
        cases.append(case)
    evidence=[Path(str(packet/f'FORMAL_{index}_{jit:02d}')+suffix)
              for index,jit in phases
              for suffix in ('_in.bin','_out.bin','_geometry.bin',
                             '_rays_in.bin','_rays_out.bin')]
    report=dict(scope=__doc__,cases=cases,producer_sha256=sha(Path(__file__)),
                evidence_sha256={str(path):sha(path) for path in evidence},
                final_goal='NOT_MET')
    if out is not None:
        with out.open('x') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(result=str(out) if out is not None else None,
                          phases=len(cases),
                          local_max_scaled_error=max(x['max_scaled_error'] for x in cases),
                          full_ray_max_scaled_error=(max(x['full_ray_max_scaled_error'] for x in cases)
                                                     if full_ray else None),
                          parabolic_max_scaled_error=max(x['parabolic_max_scaled_error'] for x in cases)),
                     ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--full-ray',action='store_true')
    parser.add_argument('--all-jit',action='store_true')
    args=parser.parse_args()
    check(args.packet.resolve(),args.out.resolve() if args.out else None,
          args.full_ray,args.all_jit)
