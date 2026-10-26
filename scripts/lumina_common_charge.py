"""고정 T의 공통 log(ne) 감쇠 secant. 깊이 결합 T/RE 해법의 내부 초기화다.

각 평가에서 전원소 SE와 파생 율을 다시 만든다. 전하의 단조성/유일근을
가정하지 않고 실제 전체 잔차 감소로만 후보를 수용한다. 작은 적용 step이나
반복 횟수를 수렴으로 바꾸지 않는다. 실패/비수렴은 마지막 수용 후보를 반환.
"""
import numpy as np


def solve(evaluate,initial,initial_charge,lower,upper,*,tolerance=1.e-10,
          max_evaluations=8,on_evaluation=None):
    ne=np.array(initial,dtype='f8',copy=True);charge=np.array(initial_charge,dtype='f8',copy=True)
    lo,hi=[np.asarray(a,dtype='f8') for a in (lower,upper)]
    if (ne.ndim!=1 or not len(ne) or any(a.shape!=ne.shape for a in (charge,lo,hi)) or
        not np.isfinite([ne,charge,lo,hi]).all() or np.any(lo<=0) or np.any(hi<=lo) or
        np.any(ne<lo) or np.any(ne>hi) or tolerance<=0 or max_evaluations<1):
        raise ValueError('공통 전하 초기화/수밀도 경계 오류')
    scale=ne.copy();x=np.log(ne);f=charge/scale;previous=None;previous_f=None;payload=None
    history=[];used=0;norm=float(np.max(abs(charge/ne)))
    if norm<=tolerance:return dict(converged=True,ne=ne,charge=charge,payload=None,history=history)
    proposal=np.log((ne+charge)/ne) if np.all(ne+charge>0) else -f
    while used<max_evaluations:
        if previous is not None:
            dx=x-previous;df=f-previous_f
            valid=(df!=0)&(dx!=0)&np.isfinite(df)&np.isfinite(dx)
            step=proposal.copy();step[valid]=-f[valid]*dx[valid]/df[valid]
        else:step=proposal.copy()
        if not np.isfinite(step).all():raise ValueError('공통 전하 secant 비유한')
        alpha=1.;accepted=False
        while used<max_evaluations and alpha>=2.**-20:
            nx=x+alpha*step
            if (np.any(nx<np.log(lo)) or np.any(nx>np.log(hi)) or not np.isfinite(nx).all()):
                alpha*=.5;continue
            trial_ne=np.exp(nx);data=evaluate(trial_ne);used+=1
            new_charge=np.asarray(data['charge_residual']);new_norm=float(np.max(abs(new_charge/trial_ne)))
            if new_charge.shape!=ne.shape or not np.isfinite(new_charge).all():
                raise ValueError('공통 전하 응답 형상/비유한 오류')
            accepted=bool(new_norm<=tolerance or new_norm<norm*(1.-1.e-4*alpha))
            record=dict(evaluation=used,alpha=alpha,accepted=accepted,charge_relative=new_norm)
            history.append(record)
            if on_evaluation is not None:on_evaluation(data,record)
            if accepted:
                previous=x;previous_f=f;x=nx;ne=trial_ne;charge=new_charge
                f=charge/scale;norm=new_norm;payload=data
                proposal=np.log((ne+charge)/ne) if np.all(ne+charge>0) else -f
                break
            alpha*=.5
        if not accepted:break
        if norm<=tolerance:return dict(converged=True,ne=ne,charge=charge,payload=payload,history=history)
    return dict(converged=False,ne=ne,charge=charge,payload=payload,history=history)
