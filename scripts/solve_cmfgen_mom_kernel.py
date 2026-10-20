#!/usr/bin/env python3
"""보존된 단일 MOM 호출의 원래 이산식에서 양의 근을 독립 확인한다."""
import argparse
import json
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from scipy.linalg import solve
from scipy.optimize import root,least_squares

from analyze_cmfgen_mom_capture import capture,trace,number,residual,first_event
from prepare_cmfgen_mom_capture import sha


def linmon(x,r):
    """native subs/nordtau.f DERIVCHI(...,'LINMON')의 같은 산술/분기."""
    h=np.diff(r)
    if len(x)<3 or len(r)!=len(x) or not (np.all(h>0) or np.all(h<0)):
        raise ValueError("LINMON 격자 불일치")
    s=np.diff(x)/h
    d=np.empty(len(x))
    d[0]=s[0]+(s[0]-s[1])*h[0]/(h[0]+h[1])
    d[1:-1]=(s[:-1]*h[1:]+s[1:]*h[:-1])/(h[:-1]+h[1:])
    d[-1]=s[-1]+(s[-1]-s[-2])*h[-1]/(h[-2]+h[-1])
    sign=lambda a:np.copysign(1.,a)
    d[0]=(sign(s[0])+sign(d[0]))*min(abs(s[0]),.5*abs(d[0]))
    d[1:-1]=(sign(s[:-1])+sign(s[1:]))*np.minimum(np.minimum(abs(s[:-1]),abs(s[1:])),.5*abs(d[1:-1]))
    d[-1]=(sign(s[-1])+sign(d[-1]))*min(abs(s[-1]),.5*abs(d[-1]))
    return d


def linmon_matrix(x,r):
    """동일 limiter의 현재 선형 분기. 경계의3점 의존성도 보존한다."""
    n=len(x);h=np.diff(r);s=np.diff(x)/h
    sec=np.diff(np.eye(n),axis=0)/h[:,None]
    P=np.empty((n,n))
    P[0]=sec[0]+(sec[0]-sec[1])*h[0]/(h[0]+h[1])
    P[1:-1]=(sec[:-1]*h[1:,None]+sec[1:]*h[:-1,None])/(h[:-1]+h[1:])[:,None]
    P[-1]=sec[-1]+(sec[-1]-sec[-2])*h[-1]/(h[-2]+h[-1])
    p=np.empty(n)
    p[0]=s[0]+(s[0]-s[1])*h[0]/(h[0]+h[1])
    p[1:-1]=(s[:-1]*h[1:]+s[1:]*h[:-1])/(h[:-1]+h[1:])
    p[-1]=s[-1]+(s[-1]-s[-2])*h[-1]/(h[-2]+h[-1])
    D=np.zeros((n,n));branches=[]
    for i in range(n):
        if i in (0,n-1):
            k=0 if i==0 else n-2
            sign=np.copysign(1.,s[k])+np.copysign(1.,p[i])
            if abs(s[k])<=.5*abs(p[i]):
                D[i]=sign*np.copysign(1.,s[k])*sec[k];label="end_secant"
            else:
                D[i]=sign*.5*np.copysign(1.,p[i])*P[i];label="end_parabola"
        else:
            sign=np.copysign(1.,s[i-1])+np.copysign(1.,s[i])
            k=int(np.argmin([abs(s[i-1]),abs(s[i]),.5*abs(p[i])]))
            if k<2:
                j=i-1+k;D[i]=sign*np.copysign(1.,s[j])*sec[j];label="left" if k==0 else "right"
            else:
                D[i]=sign*.5*np.copysign(1.,p[i])*P[i];label="parabola"
        branches.append(label if sign else "extremum_zero")
    return D,branches


def tri_matrix(A):
    a,b,c,_=A
    return np.diag(b)+np.diag(a[1:],-1)+np.diag(c[:-1],1)


def high_precision_linear(M,b,dps=80):
    with mp.workdps(dps):
        a=mp.matrix([[mp.mpf(float(v)) for v in row] for row in M])
        rhs=mp.matrix([mp.mpf(float(v)) for v in b])
        x=mp.lu_solve(a,rhs)
        return np.array([float(v) for v in x]),[mp.nstr(v,dps) for v in x]


def relative_distance(x,y):
    if np.any((y==0)&(x!=0)):return float("inf")
    return float(np.max(np.divide(abs(x-y),abs(y),out=np.zeros_like(x),where=y!=0)))


def affine_boundary_from_half(plus,minus):
    """동일 실제 광선의 두 스트림에서 H=hJ−Hinc, N=nJ−Ninc를 만든다."""
    plus,minus=np.asarray(plus),np.asarray(minus)
    if (plus.shape!=(4,) or minus.shape!=(4,) or not np.all(np.isfinite([plus,minus]))
            or plus[0]<=0 or minus[0]<0):
        raise ValueError("입사/출사 half 모멘트의 지지 오류")
    h,n=plus[1]/plus[0],plus[3]/plus[0]
    return dict(h=float(h),n=float(n),H_incid=float(h*minus[0]-minus[1]),
                N_incid=float(n*minus[0]-minus[3]))


def affine_spatial_terms(state,current):
    """고정 Q에서 경계의 βH 계수·상수와 첫 내부 flux의 일관된 치환."""
    if not state["args"]["INCL_ADVEC_TERMS"]:return (0.,0.,0.,0.)
    j=state["modules"]["jrel"];v=lambda k:j[k]["values"]
    scale=v("Q")[0]/v("DTAU_H")[0]
    delta=(v("BETA")[0]*current["h"]-v("VdHdR_TERM")[0])*scale
    incident=v("BETA")[0]*v("GAM_REL")[0]*v("R")[0]**2*current["H_incid"]*scale
    denominator=v("P_H")[0]+v("W")[0]
    return delta,incident,delta/denominator,incident/denominator


def affine_outer_system(state,base,rhs,current,previous,jprev,spatial=False):
    """경계 H/N을 affine으로 치환한다. 나머지 frozen VEF/물리항은 유지한다."""
    j,q=state["modules"]["jrel"],state["modules"]["ray"]
    v=lambda k:j[k]["values"]
    delta=v("DELTA")[0];g=v("GAM_REL_SQ")[0]*(v("SIGMA")[0]+1)
    factor=v("GAM_REL")[0]*v("R")[0]**2
    def psi(bc,k):return delta*(bc["h"]+(g-1)*bc["n"]+g*v("BETA")[0]*k)
    matrix=base.copy();b=rhs.copy()
    matrix[0,0]+=psi(current,q["K_ON_J"]["values"][0])-v("PSI")[0]+(current["h"]-q["HBC"])*v("P_H")[0]
    offset=factor*(v("P_H")[0]*current["H_incid"]+delta*(
        current["H_incid"]-previous["H_incid"]+(g-1)*(current["N_incid"]-previous["N_incid"])))
    b[0]=psi(previous,q["K_ON_J_PREV"]["values"][0])*jprev[0]+offset
    if spatial:
        delta,incident,flux_delta,flux_incident=affine_spatial_terms(state,current)
        matrix[0,0]+=delta;b[0]+=incident
        matrix[1,0]-=flux_delta;b[1]-=flux_incident
    return matrix,b


def affine_outer_primitive(state,x,jprev,current,previous,spatial=False):
    """H/N 치환 전 경계식. 소거식과 별개로 부호·이전 입사항을 검산한다."""
    j,q=state["modules"]["jrel"],state["modules"]["ray"]
    v=lambda k:j[k]["values"]
    factor=v("GAM_REL")[0]*v("R")[0]**2
    H=current["h"]*x[0]-factor*current["H_incid"]
    N=current["n"]*x[0]-factor*current["N_incid"]
    Hp=previous["h"]*jprev[0]-factor*previous["H_incid"]
    Np=previous["n"]*jprev[0]-factor*previous["N_incid"]
    k=q["K_ON_J"]["values"];kp=q["K_ON_J_PREV"]["values"]
    gradient=((k[0]+v("VdHdR_TERM")[0])*v("Q")[0]*x[0]
              -(k[1]+v("VdHdR_TERM")[1])*v("Q")[1]*x[1])/v("DTAU_H")[0]
    if spatial:
        delta,incident,_,_=affine_spatial_terms(state,current)
        gradient+=delta*x[0]-incident
    g=v("GAM_REL_SQ")[0]*(v("SIGMA")[0]+1)
    spectral=v("DELTA")[0]*(H-Hp+(g-1)*(N-Np)+g*v("BETA")[0]*(k[0]*x[0]-kp[0]*jprev[0]))
    return float(gradient+v("P_H")[0]*H+spectral)


def completed_operator(state):
    """완료 호출의 SAVE 경계와 마지막 실제 계수로 원래 삼중대각식을 재조립한다."""
    j,q=state["modules"]["jrel"],state["modules"]["ray"]
    v=lambda g,k:g[k]["values"]
    n=j["ND"];A=np.zeros((4,n));x=v(j,"GAM_RSQJNU_SAVE");H=v(j,"GAM_RSQHNU_SAVE")
    jp,hp=v(j,"GAM_RSQJNU_PREV"),v(j,"GAM_RSQHNU_PREV")
    if not np.array_equal(x,v(j,"GAM_RSQJNU")):
        raise ValueError("이전 반환 J와 SAVE가 다름")
    A[0,1:-1]=-v(j,"HL")[:-2]-v(j,"EPS")[:-2]
    A[2,1:-1]=-v(j,"HU")[1:-1]+v(j,"EPS")[1:-1]
    A[1,1:-1]=(v(j,"DTAUONQ")*(v(j,"P_J")-v(j,"COH_VEC"))+v(j,"PSI")+v(j,"HL"))[1:-1]
    A[1,1:-1]+=v(j,"HU")[:-2]-v(j,"EPS")[:-2]+v(j,"EPS")[1:-1]
    A[3,1:-1]=(v(j,"DTAUONQ")*v(j,"SOURCE")*v(j,"R")**2+v(j,"PSIPREV")*jp)[1:-1]
    A[3,1:-1]+=-v(j,"HS")[:-2]*hp[:-2]+v(j,"HS")[1:-1]*hp[1:-1]
    A[3,1:-1]+=-v(j,"EPS_PREV")[:-2]*(jp[:-2]+jp[1:-1])+v(j,"EPS_PREV")[1:-1]*(jp[1:-1]+jp[2:])
    k,dv,Q,dt=v(q,"K_ON_J_SAVE"),v(j,"VdHdR_TERM"),v(j,"Q"),v(j,"DTAU_H")
    A[2,0]=-(k[1]+dv[1])*Q[1]/dt[0]
    A[1,0]=(k[0]+dv[0])*Q[0]/dt[0]+v(j,"PSI")[0]+q["HBC_SAVE"]*v(j,"P_H")[0]
    A[3,0]=v(j,"PSIPREV")[0]*jp[0]
    A[0,-1]=-Q[-2]*(k[-2]+dv[-2])/dt[-2]
    A[1,-1]=(k[-1]+dv[-1])/dt[-2]
    A[3,-1]=v(j,"GAM_REL")[-1]*q["IN_HBC_SAVE"]
    return A


def previous_state_audit(state):
    """직전 SAVE의 원식과 H 관계를 검사한다. 현재 raw STORE와 섞지 않는다."""
    j,q=state["modules"]["jrel"],state["modules"]["ray"]
    v=lambda g,k:g[k]["values"]
    n=j["ND"];x=v(j,"GAM_RSQJNU_SAVE");H=v(j,"GAM_RSQHNU_SAVE")
    jp,hp=v(j,"GAM_RSQJNU_PREV"),v(j,"GAM_RSQHNU_PREV")
    A=completed_operator(state)
    fresh=1+v(j,"CON_DELTA")/v(j,"CHI_J")*v(j,"R")*linmon(x,v(j,"R"))/x
    original_residual=residual(A,x)
    A[1,1:-1]+=v(j,"DTAUONQ")[1:-1]*(fresh-v(j,"P_J"))[1:-1]
    matrix=tri_matrix(A)
    error=abs(matrix@x-A[3]);denominator=abs(matrix)@abs(x)+abs(A[3])
    ratios=np.divide(error,denominator,out=np.zeros(n),where=denominator>0)
    raw_H=v(j,"HU")[:-1]*x[1:]-v(j,"HL")[:-1]*x[:-1]+v(j,"HS")[:-1]*hp[:-1]
    raw_H+=v(j,"EPS_PREV")[:-1]*(jp[:-1]+jp[1:])-v(j,"EPS")[:-1]*(x[:-1]+x[1:])
    changed=abs(raw_H-H[:-1])>1e-10*np.maximum(abs(raw_H),abs(H[:-1]))
    return dict(frequency_native_units=j["FREQ_SAVE"],stored_operator_residual=original_residual,
        fresh_nonlinear_residual=residual(A,x),worst_depth_1based=int(np.argmax(ratios))+1,
        depths_above_1e_minus8_1based=(np.flatnonzero(ratios>1e-8)+1).tolist(),
        returned_H_replaced_edges_1based=(np.flatnonzero(changed)+1).tolist(),
        raw_H_exceeds_native_MAX_VAL_edges_1based=(np.flatnonzero(abs(raw_H)>np.maximum(x[:-1],x[1:]))+1).tolist(),
        meaning=("직전 입력 field도 방정식을 만족하는 정상해라는 가정은 성립하지 않음; 더 앞선 원인은 미확정"
                 if residual(A,x)>1e-8 else "이전 상태의 재평가 잔차는 작음; 최종 앵커 수렴의 증거는 아님"))


def first_repair_equation(state,event):
    """최초 사건의 실제 사용 기울기를 분리한다. 해나 율에 floor를 넣지 않는다."""
    a=state["args"];j=state["modules"]["jrel"]
    v=lambda k:j[k]["values"]
    r=v("R");n=len(r)
    if (a["INNER_BND_METH"].strip()!="DIFFUSION" or a["OUTER_BND_METH"].strip()!="HONJ"
            or n!=a["ND_SM"] or n!=event["ND"] or np.any(r!=a["R_SM"])):
        raise ValueError("최초 사건의 경계/격자 미지원")
    alpha=v("DTAUONQ")*v("CON_DELTA")/v("CHI_J") if a["INCL_ADVEC_TERMS"] else np.zeros(n)
    alpha[[0,-1]]=0
    fixed=event["matrix"].copy()
    fixed[1]-=alpha*event["slope"]
    return tri_matrix(fixed),fixed[3],alpha*r


def analyze_first_repair(original,build,output):
    """첫 대체 전 원식을 조사한다.100회 trace·양수 첫 iterate를 가정하지 않는다."""
    started=time.monotonic()
    if output.exists():raise ValueError("기존 최초 사건 분석 보존")
    inventory=json.loads((build/"state_inventory.json").read_text())
    state=capture(original/"MOM_KERNEL.bin",inventory)
    event=first_event(original/"MOM_FAULT.bin")
    if state["phase"]!=event["phase"] or state["args"]["FREQ"]!=event["frequency"]:
        raise ValueError("최초 사건과 kernel 신원 불일치")
    a=state["args"];j=state["modules"]["jrel"];q=state["modules"]["ray"]
    v=lambda k:j[k]["values"]
    base,rhs,adv=first_repair_equation(state,event)
    R=v("R");n=len(R);jp=v("GAM_RSQJNU_PREV");hp=v("GAM_RSQHNU_PREV")
    def flux(x):
        return (v("HU")[:-1]*x[1:]-v("HL")[:-1]*x[:-1]+v("HS")[:-1]*hp[:-1]
                +v("EPS_PREV")[:-1]*(jp[:-1]+jp[1:])-v("EPS")[:-1]*(x[:-1]+x[1:]))
    def F(x):return base@x+adv*linmon(x,R)-rhs
    def jacobian(x):return base+adv[:,None]*linmon_matrix(x,R)[0]
    def scaled(x):
        den=abs(base)@abs(x)+abs(adv*linmon(x,R))+abs(rhs)
        return float(np.max(np.divide(abs(F(x)),den,out=np.zeros(n),where=den>0)))
    terms={"local_emission":v("DTAUONQ")*v("SOURCE")*R*R,"previous_H":np.zeros(n),
           "previous_J":v("PSIPREV")*jp,"previous_N_lower":np.zeros(n),"previous_N_upper":np.zeros(n)}
    terms["previous_H"][1:-1]=-v("HS")[:-2]*hp[:-2]+v("HS")[1:-1]*hp[1:-1]
    terms["previous_N_lower"][1:-1]=-v("EPS_PREV")[:-2]*(jp[:-2]+jp[1:-1])
    terms["previous_N_upper"][1:-1]=v("EPS_PREV")[1:-1]*(jp[1:-1]+jp[2:])
    den=sum(abs(x) for x in terms.values())[1:-1]
    err=abs(sum(terms.values())[1:-1]-rhs[1:-1])
    if np.any((den==0)&(err!=0)):raise ValueError("최초 사건의 우변 지지 불일치")
    rhs_error=float(np.max(np.divide(err,den,out=np.zeros_like(err),where=den>0)))
    bc=np.array([v("PSIPREV")[0]*jp[0],v("GAM_REL")[-1]*a["DBB"]*R[-1]**2/3/v("CHI")[-1]])
    den=abs(bc)+abs(rhs[[0,-1]])
    boundary_error=float(np.max(np.divide(abs(bc-rhs[[0,-1]]),den,out=np.zeros(2),where=den>0)))
    if rhs_error>1e-12 or boundary_error>1e-12:raise ValueError("최초 사건의 우변/경계 재조립 불일치")
    M=tri_matrix(event["matrix"]);precise,_=high_precision_linear(M,rhs)
    formal=v("GAM_REL")*R*R*q["JNU_STORE"]["values"]
    # 단위 정규화일 뿐 해의 양수화가 아니다. 원래 formal 입력 자체가 부적절하면 거부한다.
    if np.any(~np.isfinite(formal)) or np.any(formal<=0):
        raise ValueError("최초 사건의 자연 formal 초기값이 양수가 아님")
    scale=formal
    row_scale=abs(base)@scale+abs(adv*linmon(scale,R))+abs(rhs)
    if np.any(row_scale<=0):raise ValueError("최초 사건 식의 정규화0")
    fun=lambda y:F(y*scale)/row_scale
    jac=lambda y:jacobian(y*scale)*scale[None,:]/row_scale[:,None]
    answers=[]
    for name,start in (("formal",formal),("native_at_event",event["raw"])):
        answer=root(fun,start/scale,jac=jac,method="hybr",options=dict(xtol=1e-11,maxfev=500))
        answers.append((name,answer,answer.x*scale))
    if not any(np.all(x>0) and scaled(x)<=1e-10 for _,_,x in answers):
        answer=least_squares(fun,np.ones(n),jac=jac,bounds=(0,np.inf),xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=500)
        answers.append(("positive_formal",answer,answer.x*scale))
    results=[]
    for name,answer,x in answers:
        h=flux(x);limit=np.maximum(x[:-1],x[1:])
        item=dict(initial=name,reported_success=bool(answer.success),nfev=int(answer.nfev),
            positive=bool(np.all(x>0)),equation_residual=number(scaled(x)),solution=x.tolist(),
            negative_depths_1based=(np.flatnonzero(x<0)+1).tolist(),
            H_MAX_VAL_edges_1based=(np.flatnonzero(abs(h)>limit)+1).tolist())
        if item["positive"] and item["equation_residual"]<=1e-10:
            D,branches=linmon_matrix(x,R);mpx,_=high_precision_linear(base+adv[:,None]*D,rhs)
            item.update(mp80_relative=number(relative_distance(x,mpx)),mp80_positive=bool(np.all(mpx>0)),
                mp80_equation_residual=number(scaled(mpx)),same_mp80_branches=branches==linmon_matrix(mpx,R)[1])
        results.append(item)
    previous=original/"MOM_PREVIOUS.bin"
    previous_audit=None
    if previous.exists():
        p=capture(previous,inventory)
        if (p["phase"][0]!=event["phase"][0] or p["args"]["FREQ"]<=event["frequency"]
                or not np.array_equal(jp,p["modules"]["jrel"]["GAM_RSQJNU_SAVE"]["values"])
                or not np.array_equal(hp,p["modules"]["jrel"]["GAM_RSQHNU_SAVE"]["values"])):
            raise ValueError("최초 사건의 PREV와 직전 완료 호출 불일치")
        previous_audit=previous_state_audit(p)
    bad=max(event["bad_1based"]-1,0)
    report=dict(scope="최초 대체 전 원래 MOM 이산식; 전 주파수/분포 수렴 아님",
        event={k:v for k,v in event.items() if k not in ("matrix","raw","hraw","slope")},
        native_linear_residual=residual(event["matrix"],event["raw"]),
        native_vs_mp80_relative=number(relative_distance(event["raw"],precise)),
        native_nonlinear_residual=number(scaled(event["raw"])),
        rhs_reconstruction_relative=rhs_error,boundary_rhs_relative=boundary_error,
        negative_rhs_depths_1based=(np.flatnonzero(rhs<0)+1).tolist(),
        event_depth_rhs_terms={k:float(x[bad]) for k,x in terms.items()},
        H_formula_relative=number(relative_distance(flux(event["raw"]),event["hraw"][:-1])) if event["kind"]==2 else None,
        previous_completed_audit=previous_audit,
        root_checks=results,positive_root_verified=any(x["positive"] and x["equation_residual"]<=1e-10
            and x.get("mp80_relative",1)<=1e-8 and x.get("mp80_positive",False)
            and x.get("mp80_equation_residual",1)<=1e-10 and x.get("same_mp80_branches",False)
            and not x["H_MAX_VAL_edges_1based"] for x in results),
        elapsed_seconds=time.monotonic()-started,
        evidence_sha256={str(p):sha(p) for p in [original/"MOM_KERNEL.bin",original/"MOM_FAULT.bin",
            original/"MOM_CAPTURE.bin",build/"state_inventory.json",build/"capture-build.json",Path(__file__)]+([previous] if previous.exists() else [])},
        final_goal="NOT_JUDGED")
    with output.open("x") as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("root_checks","evidence_sha256")},ensure_ascii=False))
    for item in results:print(json.dumps({k:v for k,v in item.items() if k!="solution"},ensure_ascii=False))


def solve_kernel(kernel,original,build,native,output):
    started=time.monotonic()
    if output.exists():raise ValueError("기존 원식 분석 보존")
    inventory=json.loads((build/"state_inventory.json").read_text())
    state=capture(kernel/"MOM_KERNEL.bin",inventory)
    before=capture(original/"MOM_CAPTURE.bin",inventory)
    call=trace(original/"MOM_TRACE.bin")[-1]
    cases=call["iterations"]
    a=state["args"];j=state["modules"]["jrel"];q=state["modules"]["ray"]
    vec=lambda group,name:group[name]["values"]
    R=vec(j,"R");n=len(R)
    if (a["INNER_BND_METH"].strip()!="DIFFUSION" or a["OUTER_BND_METH"].strip()!="HONJ"
            or n!=a["ND_SM"] or n!=len(vec(q,"JNU_STORE")) or np.any(R!=a["R_SM"])):
        raise ValueError("이 포획 이외의 경계/격자에 kernel을 적용하지 않음")
    alpha=vec(j,"DTAUONQ")*vec(j,"CON_DELTA")/vec(j,"CHI_J")
    alpha[[0,-1]]=0  # 원래 두 경계행은 P_J를 사용하지 않는다.
    fixed=cases[1]["TRI0"].copy()
    fixed[1]-=alpha*cases[1]["ERR0"][1][2]
    base=tri_matrix(fixed);rhs=fixed[3];adv=alpha*R
    diagonal_error=0.;other_error=0.;slope_error=0.
    for i,data in cases.items():
        A=data["TRI0"];slope=data["ERR0"][1][2]
        predicted=fixed[1]+alpha*slope
        denominator=abs(A[1])+abs(predicted)
        ratio=np.divide(abs(A[1]-predicted),denominator,out=np.zeros(n),where=denominator>0)
        diagonal_error=max(diagonal_error,float(ratio.max()))
        other_error=max(other_error,float(np.max(abs(A[[0,2,3]]-fixed[[0,2,3]]))))
        if i+1 in cases:
            damping=.8 if data["ERR0"][0]<.01 else .1
            field=data["ERR0"][1][1]
            predicted=damping*R*linmon(field,R)/field+(1-damping)*slope
            expected=cases[i+1]["ERR0"][1][2]
            scale=np.maximum(abs(expected),abs(predicted))
            mismatch=np.divide(abs(predicted-expected),scale,out=np.zeros(n),where=scale>0)
            slope_error=max(slope_error,float(mismatch.max()))
    if diagonal_error>1e-12 or other_error!=0 or slope_error>1e-11:
        raise ValueError("재구성식이 native 반복을 재현하지 못함")
    if a["FREQ"]>=before["modules"]["jrel"]["FREQ_SAVE"]:
        raise ValueError("청색측 이전 주파수 정체성 불일치")
    if not np.array_equal(vec(j,"GAM_RSQJNU_PREV"),vec(before["modules"]["jrel"],"GAM_RSQJNU_SAVE")):
        raise ValueError("원래 PREV/SAVE 선택 불일치")
    # 각 경계와 음수 우변을 원래 항에서 독립 재조립한다.
    Hprev=vec(j,"GAM_RSQHNU_PREV");Jprev=vec(j,"GAM_RSQJNU_PREV")
    terms={"local_emission":vec(j,"DTAUONQ")*vec(j,"SOURCE")*R*R,
           "previous_H":np.zeros(n),"previous_J":vec(j,"PSIPREV")*Jprev,
           "previous_N_lower":np.zeros(n),"previous_N_upper":np.zeros(n)}
    terms["previous_H"][1:-1]=-vec(j,"HS")[:-2]*Hprev[:-2]+vec(j,"HS")[1:-1]*Hprev[1:-1]
    terms["previous_N_lower"][1:-1]=-vec(j,"EPS_PREV")[:-2]*(Jprev[:-2]+Jprev[1:-1])
    terms["previous_N_upper"][1:-1]=vec(j,"EPS_PREV")[1:-1]*(Jprev[1:-1]+Jprev[2:])
    rebuilt=sum(terms.values())
    rhs_error=float(np.max(abs(rebuilt[1:-1]-rhs[1:-1])/(sum(abs(v) for v in terms.values())[1:-1])))
    boundary_rhs=np.array([vec(j,"PSIPREV")[0]*Jprev[0],
        vec(j,"GAM_REL")[-1]*a["DBB"]*R[-1]*R[-1]/3/vec(j,"CHI")[-1]])
    boundary_error=float(np.max(abs(boundary_rhs-rhs[[0,-1]])/np.maximum(abs(boundary_rhs),abs(rhs[[0,-1]]))))
    if rhs_error>1e-12 or boundary_error>1e-12:raise ValueError("우변/경계 재조립 불일치")
    linear=[]
    first_negative=min(i for i,d in cases.items() if np.any(d["RAW0"]<0))
    for i in (first_negative,100):
        M=tri_matrix(cases[i]["TRI0"]);b=cases[i]["TRI0"][3]
        pivoted=solve(M,b);precise,strings=high_precision_linear(M,b)
        linear.append(dict(iteration=i,kappa_2=float(np.linalg.cond(M)),
            native_vs_mp_relative=relative_distance(cases[i]["RAW0"],precise),
            pivoted_vs_mp_relative=relative_distance(pivoted,precise),
            negative_depths_1based=(np.flatnonzero(precise<0)+1).tolist(),mp80_solution=strings))
    def F(x):return base@x+adv*linmon(x,R)-rhs
    def J(x):return base+adv[:,None]*linmon_matrix(x,R)[0]
    def scaled_residual(x):
        denominator=abs(base)@abs(x)+abs(adv*linmon(x,R))+abs(rhs)
        return float(np.max(np.divide(abs(F(x)),denominator,out=np.zeros(n),where=denominator>0)))
    scale=cases[1]["RAW0"].copy()
    formal=vec(j,"GAM_REL")*R*R*vec(q,"JNU_STORE")
    if np.any(scale<=0) or np.any(formal<=0):raise ValueError("선택한 자연 초기값이 양수가 아님")
    row_scale=abs(base)@scale+abs(adv*linmon(scale,R))+abs(rhs)
    if np.any(row_scale<=0):raise ValueError("식의 정규화0")
    fun=lambda y:F(y*scale)/row_scale
    jac=lambda y:J(y*scale)*scale[None,:]/row_scale[:,None]
    solutions=[]
    for name,start in (("native_first",scale),("formal",formal)):
        answer=root(fun,start/scale,jac=jac,method="hybr",options=dict(xtol=1e-11,maxfev=500))
        solutions.append((name,answer,answer.x*scale))
    if not any(np.all(x>0) and scaled_residual(x)<=1e-10 for _,_,x in solutions):
        answer=least_squares(fun,np.ones(n),jac=jac,bounds=(0,np.inf),xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=500)
        solutions.append(("positive_native_first",answer,answer.x*scale))
    results=[]
    for name,answer,x in solutions:
        item=dict(initial=name,reported_success=bool(answer.success),nfev=int(answer.nfev),
                  positive=bool(np.all(x>0)),equation_residual=number(scaled_residual(x)),solution=x.tolist(),
                  negative_depths_1based=(np.flatnonzero(x<0)+1).tolist(),minimum_solution=float(np.min(x)),
                  message=str(answer.message))
        if item["positive"] and item["equation_residual"]<=1e-10:
            D,branches=linmon_matrix(x,R);M=base+adv[:,None]*D
            precise,strings=high_precision_linear(M,rhs)
            flux=vec(j,"HU")[:-1]*x[1:]-vec(j,"HL")[:-1]*x[:-1]+vec(j,"HS")[:-1]*Hprev[:-1]
            flux+=vec(j,"EPS_PREV")[:-1]*(Jprev[:-1]+Jprev[1:])-vec(j,"EPS")[:-1]*(x[:-1]+x[1:])
            bad=abs(flux)>np.maximum(x[:-1],x[1:])
            item.update(active_branch_mp80_relative=relative_distance(x,precise),mp80_positive=bool(np.all(precise>0)),
                        mp80_equation_residual=number(scaled_residual(precise)),active_branches=branches,
                        same_mp80_branches=branches==linmon_matrix(precise,R)[1],mp80_solution=strings,
                        H_MAX_VAL_would_fire=int(np.sum(bad)))
        results.append(item)
    report=dict(scope="보존한 MOM 한 호출의 원래 비선형 이산식; 생산 수리/전 주파수 수렴 아님",
        diagonal_reconstruction_relative=diagonal_error,other_coefficients_max_abs_change=other_error,
        native_slope_update_relative=slope_error,rhs_reconstruction_relative=rhs_error,boundary_rhs_relative=boundary_error,
        negative_rhs_depths_1based=(np.flatnonzero(rhs<0)+1).tolist(),
        depth6_rhs_terms={k:float(v[5]) for k,v in terms.items()},linear_checks=linear,root_checks=results,
        previous_frequency_audit=previous_state_audit(before),
        elapsed_seconds=time.monotonic()-started,
        positive_root_verified=any(r["positive"] and r["equation_residual"]<=1e-10 and
            r.get("active_branch_mp80_relative",1)<=1e-8 and r.get("mp80_positive",False) and
            r.get("mp80_equation_residual",1)<=1e-10 and r.get("same_mp80_branches",False) and
            r.get("H_MAX_VAL_would_fire",1)==0 for r in results),
        evidence_sha256={str(p):sha(p) for p in (kernel/"MOM_KERNEL.bin",original/"MOM_CAPTURE.bin",
            original/"MOM_TRACE.bin",build/"state_inventory.json",native/"subs/nordtau.f",Path(__file__))},final_goal="NOT_JUDGED")
    with output.open("x") as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("linear_checks","root_checks","evidence_sha256")},ensure_ascii=False))
    for check in results:print(json.dumps({k:v for k,v in check.items() if k not in ("solution","mp80_solution","active_branches")},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("original","build","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("kernel","native"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--first-repair",action="store_true")
    a=parser.parse_args()
    if a.first_repair:
        analyze_first_repair(a.original,a.build,a.output)
    elif a.kernel is None or a.native is None:
        parser.error("기존 kernel 분석에는 --kernel과 --native가 필요함")
    else:
        solve_kernel(a.kernel,a.original,a.build,a.native,a.output)
