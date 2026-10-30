"""전90점 공통 T/ne Newton의 깊이 결합 잔차와 Jacobian 조립.

원소 SE 제거 뒤 국소 물질응답의 ±logT/±logne 배치4개를 소비한다.
AD는 실제 전체 프로필에서 다시 계산하므로 이웃 결합을 제거하지 않는다.
이 내부 FD 대응은 native E/Jv의 독립 증거가 아니며 수렴을 인증하지 않는다.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaterialPacket:
    temperature:np.ndarray
    ne:np.ndarray
    nuclei:np.ndarray
    internal_energy:np.ndarray
    local_heat:np.ndarray
    charge:np.ndarray
    thermal_scale:np.ndarray

    def validate(self):
        arrays=[np.asarray(getattr(self,k),dtype='f8') for k in self.__dataclass_fields__]
        n=len(arrays[0])
        if (n<2 or any(a.shape!=(n,) for a in arrays) or not np.isfinite(arrays).all() or
            any(np.any(a<=0) for a in (arrays[0],arrays[1],arrays[2],arrays[6])) or
            np.any(arrays[3]<0)):
            raise ValueError('공통 T/ne 물질응답 프로필 오류')
        return n


def packet(result,energy,policy):
    if policy not in ('physical_stored','physical_instantaneous_scenario',
                       'native_stored','native_instantaneous_scenario'):
        raise ValueError('명시한 에너지 정책 필요')
    family=policy.split('_',1)[0];ad=energy[family+'_adiabatic'];net=energy[policy+'_net']
    scale=.5*(energy[policy+'_gross']+abs(net)) # max(전체 가열,전체 냉각), 임의 floor 없음
    # 열잔차의 AD와 같은 합산 순서를 유지하고 다른 C 집계의 최하위 bit를 섞지 않는다.
    nuclei=result['population'].sum(axis=0)
    p=MaterialPacket(result['temperature'],result['ne'],nuclei,
        energy[family+'_internal_energy'],net+ad,result['charge_residual'],scale)
    p.validate();return p


def residual(value,adiabatic,charge_scale,thermal_scale):
    n=value.validate();c=np.asarray(charge_scale);h=np.asarray(thermal_scale)
    if c.shape!=(n,) or h.shape!=(n,) or not np.isfinite([c,h]).all() or np.any(c<=0) or np.any(h<=0):
        raise ValueError('공통 T/ne 잔차 행 scaling 오류')
    ad=np.asarray(adiabatic(value.temperature,value.ne,value.nuclei,value.internal_energy))
    if ad.shape!=(n,) or not np.isfinite(ad).all():raise ValueError('전체 깊이 단열 프로필 오류')
    return np.r_[value.charge/c,(value.local_heat-ad)/h]


def jacobian(base,plus_T,minus_T,plus_ne,minus_ne,step,adiabatic,*,charge_scale=None,thermal_scale=None):
    """열/전하의 국소 계수와 깊이 AD의 전 행을 합친 [charge,RE] × [logT,logne].

    국소 물질응답은 각 점의 T/ne/J/N만에 의존하는 기존 선형 SE 제거 계약이다.
    CE·비국소 인구 의존 과정을 켜면 이 배치 조립 계약을 사용할 수 없다.
    """
    n=base.validate()
    if not np.isfinite(step) or step<=0:raise ValueError('centered log step 오류')
    c=base.ne if charge_scale is None else np.asarray(charge_scale)
    h=base.thermal_scale if thermal_scale is None else np.asarray(thermal_scale)
    residual(base,adiabatic,c,h) # 행 scaling·기하 소비를 실제로 확인한다.
    matrix=np.zeros((2*n,2*n))
    for variable,(plus,minus) in enumerate(((plus_T,minus_T),(plus_ne,minus_ne))):
        if plus.validate()!=n or minus.validate()!=n:raise ValueError('centered 응답 점 주소 불일치')
        moving='temperature' if variable==0 else 'ne';fixed='ne' if variable==0 else 'temperature'
        for sample,sign in ((plus,1),(minus,-1)):
            if (not np.array_equal(getattr(sample,fixed),getattr(base,fixed)) or
                not np.allclose(getattr(sample,moving),getattr(base,moving)*np.exp(sign*step),rtol=8*np.finfo('f8').eps,atol=0)):
                raise ValueError('centered ±logT/±logne 물질응답 신원 불일치')
            if not np.allclose(sample.nuclei,base.nuclei,rtol=1.e-12,atol=0):
                raise ValueError('같은 원소총수가 아닌 centered 응답')
        dc=(plus.charge-minus.charge)/(2*step)
        dh=(plus.local_heat-minus.local_heat)/(2*step)
        for column in range(n):
            ad=[]
            for sample in (plus,minus):
                profiles=[]
                for name in ('temperature','ne','nuclei','internal_energy'):
                    values=np.array(getattr(base,name),copy=True)
                    values[column]=getattr(sample,name)[column];profiles.append(values)
                ad.append(np.asarray(adiabatic(*profiles)))
            derivative=(ad[0]-ad[1])/(2*step)
            if derivative.shape!=(n,) or not np.isfinite(derivative).all():
                raise ValueError('단열 민감도 오류')
            j=variable*n+column
            matrix[column,j]=dc[column]/c[column]
            matrix[n:,j]=-derivative/h
            matrix[n+column,j]+=dh[column]/h[column]
    return matrix


def newton_direction(matrix,scaled_residual):
    matrix=np.asarray(matrix);f=np.asarray(scaled_residual)
    if (f.ndim!=1 or matrix.shape!=(len(f),len(f)) or not np.isfinite(matrix).all() or not np.isfinite(f).all()):
        raise ValueError('공통 Newton 선형계 오류')
    direction=np.linalg.solve(matrix,-f)
    if not np.isfinite(direction).all():raise ValueError('공통 Newton 방향 비유한')
    return direction


def solve_coupled(evaluate,base,adiabatic,*,step=1.e-4,numerical_tolerance=1.e-10,
                  max_evaluations=12,on_evaluation=None,max_log_step=.1,on_direction=None,
                  project_candidate=None,secant_after_accept=False):
    """동적 물질 생산 callback을 쓰는 전체 깊이 감쇠 Newton.

    evaluate(T,ne)는 모든 동적 율과 SE를 갱신한 MaterialPacket을 반환한다.
    project_candidate는 같은 T에서만 전하를 닫는다. 추가 실제평가도 같은
    총예산을 소비하며 실패하면 후보를 수용하지 않는다. 기본 경로는 보존한다.
    iteration/관측 한도는 미수렴으로 반환하며 마지막 수용 상태를 보존한다.
    numerical_converged는 이 고정장 계의 내부 잔차만 뜻한다.
    """
    base.validate()
    if (type(max_evaluations) is not int or max_evaluations<0 or
        not np.isfinite(numerical_tolerance) or numerical_tolerance<=0 or
        not np.isfinite(max_log_step) or max_log_step<=0 or
        type(secant_after_accept) is not bool):
        raise ValueError('공동 Newton 비용/수치 목표 오류')
    c=base.ne.copy();h=base.thermal_scale.copy();history=[];used=0;iterations=0;entries={}
    secant_matrix=None
    def call(t,ne,phase):
        nonlocal used
        value=evaluate(t,ne);value.validate();used+=1
        if not np.array_equal(value.temperature,t) or not np.array_equal(value.ne,ne):
            raise ValueError('동적 평가가 요청 T/ne를 소비하지 않음')
        entry=dict(evaluation=used,phase=phase,
            scaled_max=float(abs(residual(value,adiabatic,c,h)).max()))
        history.append(entry)
        entries[id(value)]=(value,entry)
        if on_evaluation is not None:on_evaluation(value,entry)
        return value
    while True:
        f=residual(base,adiabatic,c,h);norm=float(abs(f).max())
        if norm<=numerical_tolerance:
            return dict(numerical_converged=True,packet=base,history=history,iterations=iterations)
        if max_evaluations-used<(1 if secant_matrix is not None else 5):break
        t=base.temperature;ne=base.ne
        using_secant=secant_matrix is not None
        if using_secant:
            j=secant_matrix
        else:
            probes=(call(t*np.exp(step),ne,'T+'),call(t*np.exp(-step),ne,'T-'),
                    call(t,ne*np.exp(step),'ne+'),call(t,ne*np.exp(-step),'ne-'))
            j=jacobian(base,*probes,step,adiabatic,charge_scale=c,thermal_scale=h)
        try:
            direction=newton_direction(j,f)
        except np.linalg.LinAlgError:
            if using_secant and max_evaluations-used>=5:
                secant_matrix=None
                continue
            raise
        n=len(t);x=np.r_[np.log(t),np.log(ne)]
        if on_direction is not None:
            if secant_after_accept:
                on_direction(base,j,f,direction,c,h,'secant' if using_secant else 'centered')
            else:
                on_direction(base,j,f,direction,c,h)
        # 전체 방향에 같은 감쇠를 적용한다. 개별 T/ne의 물리값을 clamp하지 않는다.
        magnitude=float(abs(direction).max())
        accepted=False;alpha=min(1.,max_log_step/magnitude) if magnitude>0 else 1.
        while used<max_evaluations and alpha>=2.**-20:
            with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                trial=np.exp(x+alpha*direction)
            if not np.isfinite(trial).all() or np.any(trial<=0):alpha*=.5;continue
            candidate=call(trial[:n],trial[n:],'line_search')
            if project_candidate is not None:
                raw=candidate
                def at_fixed_T(ne):
                    if used>=max_evaluations:raise RuntimeError('공동 열보정 실제평가 한도 초과')
                    return call(raw.temperature,ne,'charge_projection_ne')
                candidate=project_candidate(raw,at_fixed_T,max_evaluations-used)
                if candidate is None:
                    entry=entries[id(raw)][1]
                    entry.update(alpha=alpha,accepted=False,charge_solved=False)
                    if on_evaluation is not None:on_evaluation(raw,entry)
                    alpha*=.5;continue
                candidate.validate()
                if (not np.array_equal(candidate.temperature,raw.temperature) or
                    id(candidate) not in entries or
                    not np.allclose(candidate.nuclei,base.nuclei,rtol=1.e-12,atol=0)):
                    raise ValueError('전하 보정은 실제 평가된 동일 T/핵수 후보만 반환해야 함')
            merit=float(abs(residual(candidate,adiabatic,c,h)).max())
            accepted=bool(merit<=numerical_tolerance or merit<norm*(1.-1.e-4*alpha))
            entry=entries[id(candidate)][1]
            entry.update(alpha=alpha,accepted=accepted,
                predicted_scaled_max=float(abs(f+alpha*(j@direction)).max()),scaled_start_max=norm)
            if project_candidate is not None:
                entry.update(phase='charge_projected_line',charge_solved=True)
                if on_evaluation is not None:on_evaluation(candidate,entry)
            if accepted:
                if secant_after_accept:
                    # 실제 재평가된 잔차만으로 갱신한다. 원 centered J는 첫
                    # 반복과 secant 실패 시 복원하며, 물리 후보를 보간하지 않는다.
                    displacement=np.r_[np.log(candidate.temperature),np.log(candidate.ne)]-x
                    squared=float(displacement@displacement)
                    if squared>np.finfo('f8').tiny:
                        change=residual(candidate,adiabatic,c,h)-f
                        updated=j+np.outer(change-j@displacement,displacement)/squared
                        secant_matrix=updated if np.isfinite(updated).all() else None
                    else:
                        secant_matrix=None
                base=candidate;iterations+=1;break
            alpha*=.5
        if not accepted:
            if using_secant and max_evaluations-used>=5:
                secant_matrix=None
                continue
            break
    return dict(numerical_converged=False,packet=base,history=history,iterations=iterations)
