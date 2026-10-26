"""봉인한 온도 제안 하나의 전하를 닫은 뒤 전체 깊이 열잔차를 판단한다.

새 미분이나 온도 line search는 만들지 않는다. 같은 T의 실제 SE 응답을
내부 secant가 소비하며, 한도 소진/전하 비수렴이면 원 수용 상태를 보존한다.
"""
import numpy as np

from lumina_common_charge import solve
from lumina_coupled_material import residual


def proposal(base,adiabatic,basis,alpha,*,max_log_step=.1):
    n=base.validate()
    if (not np.isfinite(alpha) or not 0<alpha<=1 or
        not np.isfinite(max_log_step) or max_log_step<=0):
        raise ValueError('전하 투영의 온도 제안 비율 오류')
    if (not np.array_equal(base.temperature,basis['temperature']) or
        not np.array_equal(base.ne,basis['ne'])):
        raise ValueError('전하 투영의 봉인 기준 T/ne 변경')
    c,h=basis['charge_scale'],basis['thermal_scale'];f=residual(base,adiabatic,c,h)
    if not np.array_equal(f,basis['scaled_residual']):
        raise ValueError('전하 투영의 기준 잔차/열정책 변경')
    j=np.asarray(basis['jacobian']);d=np.asarray(basis['direction'])
    if j.shape!=(2*n,2*n) or d.shape!=(2*n,) or not np.isfinite(j).all() or not np.isfinite(d).all():
        raise ValueError('전하 투영의 보존 선형화 오류')
    error=np.linalg.norm(j@d+f,np.inf)
    budget=64*(2*n)*np.finfo('f8').eps*(np.linalg.norm(j,np.inf)*np.linalg.norm(d,np.inf)+np.linalg.norm(f,np.inf))
    if error>budget:raise ValueError('전하 투영의 보존 방향이 같은 선형계를 풀지 않음')
    magnitude=float(abs(d).max());first=min(1.,max_log_step/magnitude) if magnitude>0 else 1.
    if alpha>first:raise ValueError('전하 투영의 온도 보폭 한도 초과')
    with np.errstate(over='raise',invalid='raise',under='raise'):
        values=np.exp(np.r_[np.log(base.temperature),np.log(base.ne)]+alpha*d)
    return values[:n],values[n:]


def trial(evaluate,base,adiabatic,basis,alpha,lower,upper,*,max_evaluations=8,
          charge_tolerance=1.e-10,numerical_tolerance=1.e-10,on_evaluation=None):
    if (type(max_evaluations) is not int or max_evaluations<1 or
        not np.isfinite([charge_tolerance,numerical_tolerance]).all() or
        charge_tolerance<=0 or numerical_tolerance<=0):
        raise ValueError('전하 투영 비용/수치 목표 오류')
    temperature,initial_ne=proposal(base,adiabatic,basis,alpha)
    lo,hi=[np.asarray(a,dtype='f8') for a in (lower,upper)]
    if (lo.shape!=initial_ne.shape or hi.shape!=initial_ne.shape or
        not np.isfinite([lo,hi]).all() or np.any(lo<=0) or np.any(hi<=lo) or
        np.any(initial_ne<lo) or np.any(initial_ne>hi)):
        raise ValueError('전하 투영의 원소 수밀도 경계 오류')
    used=0;charge_history=[]
    def actual(ne):
        nonlocal used
        if used>=max_evaluations:raise RuntimeError('전하 투영 실제평가 한도 초과')
        value=evaluate(temperature,ne);used+=1;value.validate()
        if (not np.array_equal(value.temperature,temperature) or not np.array_equal(value.ne,ne) or
            not np.allclose(value.nuclei,base.nuclei,rtol=1.e-12,atol=0)):
            raise ValueError('전하 투영 실제 후보의 T/ne/원소총수 불일치')
        return dict(packet=value,charge_residual=value.charge)
    first=actual(initial_ne);candidate=first['packet']
    initial=dict(evaluation=1,phase='charge_projection_initial',
                 charge_relative=float(abs(candidate.charge/candidate.ne).max()))
    if on_evaluation is not None:on_evaluation(candidate,initial)
    converged=initial['charge_relative']<=charge_tolerance
    if not converged and used<max_evaluations:
        def record(data,entry):
            item={**entry,'evaluation':entry['evaluation']+1}
            charge_history.append(item)
            if on_evaluation is not None:
                on_evaluation(data['packet'],dict(evaluation=item['evaluation'],phase='charge_projection_ne',
                    charge_relative=item['charge_relative'],charge_accepted=item['accepted']))
        inner=solve(actual,initial_ne,candidate.charge,lo,hi,tolerance=charge_tolerance,
                    max_evaluations=max_evaluations-1,on_evaluation=record)
        converged=inner['converged']
        if inner['payload'] is not None:candidate=inner['payload']['packet']
    c,h=basis['charge_scale'],basis['thermal_scale']
    start=float(abs(residual(base,adiabatic,c,h)).max())
    current=float(abs(residual(candidate,adiabatic,c,h)).max())
    accepted=bool(converged and (current<=numerical_tolerance or current<start*(1.-1.e-4*alpha)))
    entry=dict(evaluation=used,phase='charge_projected_line',alpha=alpha,accepted=accepted,
        charge_solved=bool(converged),charge_relative=float(abs(candidate.charge/candidate.ne).max()),
        scaled_start_max=start,scaled_max=current)
    if on_evaluation is not None:on_evaluation(candidate,entry)
    return dict(numerical_converged=accepted and current<=numerical_tolerance,
        packet=candidate if accepted else base,iterations=int(accepted),history=[entry],
        charge_history=charge_history,evaluations_used=used)
