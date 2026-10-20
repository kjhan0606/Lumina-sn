#!/usr/bin/env python3
"""두 주파수 formal 포획의 실제 광선·기하·인자와 모멘트를 판독한다."""
import json
import math
from pathlib import Path

import numpy as np

from analyze_cmfgen_mom_capture import Reader
from prepare_cmfgen_formal_capture import ARGS
from prepare_cmfgen_mom_capture import sha


def module_state(r,inventory):
    result={}
    for f in inventory:
        count=1;bounds=None
        if f["alloc"]:
            present=r.integers()[0]
            if present not in (0,1):raise ValueError("formal allocation flag 오류")
            if not present:
                result[f["name"]]=None;continue
            bounds=r.integers(2);count=bounds[1]-bounds[0]+1
            if not 0<=count<=2000000:raise ValueError("formal array 크기 오류")
        elif f["fixed"]:raise ValueError("formal 미지원 fixed 필드")
        values=r.reals(count) if f["type"].startswith("REAL") else np.array(r.integers(count))
        result[f["name"]]=dict(bounds=bounds,values=values) if f["alloc"] else values[0].item()
    return result


def formal_state(path,inventory,allow_initial=False):
    r=Reader(path.read_bytes())
    if r.take(16).rstrip()!=b"LUMINAFORMALV1" or r.integers(4)!=(8,64,32,32):raise ValueError("formal kind/tag 오류")
    phase=r.integers(5);nc,np_,nd=r.integers(3)
    if not (3<=nd<=10000 and nd<=np_<=500 and 1<=nc<=np_):raise ValueError("formal 차원 오류")
    a=dict(NC=nc,NP=np_,ND=nd)
    for name,kind,dim in ARGS:
        if kind=="I":continue
        if kind=="C":
            n=r.integers()[0]
            if not 1<=n<=200:raise ValueError("formal 문자열 크기 오류")
            value=r.take(n).decode("ascii")
        elif kind=="L":
            value=r.integers()[0]
            if value not in (0,1):raise ValueError("formal logical 오류")
            value=bool(value)
        else:
            value=r.reals(a[dim] if dim else 1)
            if not dim:value=float(value[0])
        a[name]=value
    if (a["INITIALIZE"] and not allow_initial) or a["INNER_BND_METH"].strip()!="DIFFUSION":raise ValueError("formal 포획 경계/초기화 미지원")
    state=module_state(r,inventory)
    if r.take(16).rstrip()!=b"LUMINAFORMALEND1" or r.offset!=len(r.raw):raise ValueError("formal 끝/후행 오류")
    return dict(phase=phase,args=a,ray=state)


def rays(path):
    r=Reader(path.read_bytes())
    if r.take(16).rstrip()!=b"LUMINARAYS1":raise ValueError("광선 tag 오류")
    nc,np_,nd=r.integers(3)
    if not (3<=nd<=10000 and nd<=np_<=500 and 1<=nc<=np_):raise ValueError("광선 차원 오류")
    data=[]
    for _ in range(np_):
        n=r.integers()[0]
        if not 1<=n<=2000000:raise ValueError("광선 길이 오류")
        data.append(dict(I_P=r.reals(n),I_M=r.reals(n)))
    if r.take(16).rstrip()!=b"LUMINARAYSEND1" or r.offset!=len(r.raw):raise ValueError("광선 끝/후행 오류")
    return dict(NC=nc,NP=np_,ND=nd,rays=data)


def geometry(path):
    r=Reader(path.read_bytes())
    if r.take(16).rstrip()!=b"LUMINAGEOMETRY1":raise ValueError("기하 tag 오류")
    nc,np_,nd,ne,na,imin,imax,inserted=r.integers(8)
    if not (3<=nd<=10000 and nd<=np_<=500 and 1<=nc<=np_ and ne==nd+na and inserted in (0,1)):
        raise ValueError("기하 차원 오류")
    result=dict(NC=nc,NP=np_,ND=nd,ND_EXT=ne,ND_ADD=na,IDMIN=imin,IDMAX=imax,RAY_POINTS_INSERTED=bool(inserted))
    for name in ("R_EXT","V_EXT","VDOP_VEC_EXT","SIGMA_EXT"):result[name]=r.reals(ne)
    result["rays"]=[]
    for _ in range(np_):
        n=r.integers()[0]
        if not 1<=n<=2000000:raise ValueError("기하 광선 길이 오류")
        d={k:r.reals(n) for k in ("R","S_P","S_M","MU_P","MU_M","B_P","B_M")}
        d["LNK"]=np.array(r.integers(nd));result["rays"].append(d)
    names=("JQW_P","HQW_P","KQW_P","NQW_P","JQW_M","HQW_M","KQW_M","NQW_M")
    result["weights"]={k:np.zeros((nd,np_)) for k in names}
    for i in range(nd):
        for k in names:result["weights"][k][i,:np_-i]=r.reals(np_-i)
    if r.take(16).rstrip()!=b"LUMINAGEOMEND1" or r.offset!=len(r.raw):raise ValueError("기하 끝/후행 오류")
    return result


def angular_integral(g,rs):
    if any(g[k]!=rs[k] for k in ("NC","NP","ND")):raise ValueError("광선/기하 차원 불일치")
    plus=np.zeros((4,g["ND"]));minus=np.zeros_like(plus)
    for ip,(geo,intensity) in enumerate(zip(g["rays"],rs["rays"])):
        for i in range(min(g["ND"],g["NP"]-ip)):
            link=geo["LNK"][i]-1
            if not 0<=link<len(intensity["I_P"]):raise ValueError("광선 LNK 오류")
            for k,letter in enumerate("JHKN"):
                plus[k,i]+=intensity["I_P"][link]*g["weights"][letter+"QW_P"][i,ip]
                minus[k,i]+=intensity["I_M"][link]*g["weights"][letter+"QW_M"][i,ip]
    return plus,minus


def compensated_moments(g,rs):
    """실제 사용 강도·가중치의 보상 합과 순서 변경의 IEEE 반올림 상한."""
    values=np.zeros((4,g["ND"]));bounds=np.zeros_like(values)
    for i in range(g["ND"]):
        for k,letter in enumerate("JHKN"):
            terms=[];magnitudes=[]
            for ip in range(g["NP"]-i):
                link=g["rays"][ip]["LNK"][i]-1
                a=float(rs["rays"][ip]["I_P"][link]*g["weights"][letter+"QW_P"][i,ip])
                b=float(rs["rays"][ip]["I_M"][link]*g["weights"][letter+"QW_M"][i,ip])
                terms.append(a+b);magnitudes.extend((abs(a),abs(b)))
            values[k,i]=math.fsum(terms)
            gamma=(3*(g["NP"]-i)+8)*np.finfo(float).eps
            bounds[k,i]=gamma/(1-gamma)*math.fsum(magnitudes)
    return values,bounds


def compare_replay(run,replay,build,output):
    if output.exists():raise ValueError("기존 formal 비교 보존")
    manifest=json.loads((build/"formal-build.json").read_text())
    result=[];evidence=[]
    for phase in ("23731_00","23731_01","23732_00"):
        prefix=run/("FORMAL_"+phase);out=replay/phase
        gp=Path(str(prefix)+"_geometry.bin");rp=Path(str(prefix)+"_rays_out.bin")
        g=geometry(gp);rs=rays(rp)
        geometry_equal=gp.read_bytes()==(out/"replay_geometry.bin").read_bytes()
        rays_equal=rp.read_bytes()==(out/"replay_rays_out.bin").read_bytes()
        if not geometry_equal or not rays_equal:raise ValueError("실제 formal 기하/광선 바이트 재현 실패: "+phase)
        original=formal_state(Path(str(prefix)+"_out.bin"),manifest["ray_inventory"])
        copied=formal_state(out/"replay_out.bin",manifest["ray_inventory"])
        exact,bound=compensated_moments(g,rs)
        roundoff={}
        for label,state in (("original",original),("replay",copied)):
            observed=np.array([state["ray"][k+"NU_STORE"]["values"] for k in "JHKN"])
            error=abs(observed-exact)
            if np.any((bound==0)&(error!=0)):raise ValueError("formal 모멘트 지지 불일치")
            ratio=np.divide(error,bound,out=np.zeros_like(error),where=bound>0)
            if np.max(ratio)>1:raise ValueError("formal 모멘트 오차가 IEEE 합산 상한 초과")
            roundoff[label]=float(np.max(ratio))
        plus,minus=angular_integral(g,rs)
        item=dict(phase=original["phase"],geometry_byte_equal=geometry_equal,rays_byte_equal=rays_equal,
                  maximum_roundoff_bound_fraction=roundoff,
                  minimum_ray_intensity=float(min(np.min(v) for ray in rs["rays"] for v in ray.values())),
                  negative_ray_values=sum(int(np.sum(v<0)) for ray in rs["rays"] for v in ray.values()),
                  outer_half_plus=plus[:,0].tolist(),outer_half_minus=minus[:,0].tolist())
        result.append(item)
        evidence += [gp,rp,out/"replay_geometry.bin",out/"replay_rays_out.bin",out/"replay_out.bin"]
    report=dict(scope="실제 세 formal 호출의 기하/광선/병렬 모멘트 재현; 수렴 판정 아님",cases=result,
                evidence_sha256={str(p):sha(p) for p in evidence+[build/"formal-build.json",Path(__file__)]},final_goal="NOT_JUDGED")
    with output.open("x") as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


def compare_split(run,replay,build,output):
    """확장층의 실제 입사 강도에서 재시작한 모델 내부 적분만 대조한다."""
    if output.exists():raise ValueError("기존 표면 전달 비교 보존")
    inventory=json.loads((build/"formal-build.json").read_text())["ray_inventory"]
    cases=[];evidence=[]
    for phase in ("23731_00","23731_01","23732_00"):
        prefix=run/("FORMAL_"+phase);out=replay/("split_"+phase)
        gp=Path(str(prefix)+"_geometry.bin");rp=Path(str(prefix)+"_rays_out.bin")
        g=geometry(gp);reference=rays(rp);copied=rays(out/"replay_rays_out.bin")
        if gp.read_bytes()!=(out/"replay_geometry.bin").read_bytes():raise ValueError("표면 전달의 기하 불일치")
        for ray,original,new in zip(g["rays"],reference["rays"],copied["rays"]):
            start=ray["LNK"][0]-1
            if not 0<=start<len(original["I_P"]):raise ValueError("표면 광선 시작점 오류")
            for name in ("I_P","I_M"):
                if not np.array_equal(original[name][start:],new[name][start:]):
                    raise ValueError("실제 입사 경계에서 안쪽 광선 출력 불일치")
        exact,bound=compensated_moments(g,reference)
        state=formal_state(out/"replay_out.bin",inventory)
        moments=np.array([state["ray"][k+"NU_STORE"]["values"] for k in "JHKN"])
        error=abs(moments-exact)
        if np.any((bound==0)&(error!=0)):raise ValueError("표면 전달 모멘트 지지 오류")
        ratio=np.divide(error,bound,out=np.zeros_like(error),where=bound>0)
        if np.max(ratio)>1:raise ValueError("표면 전달 모멘트 오차 상한 초과")
        cases.append(dict(phase=phase,interior_ray_values_byte_equal=True,
            exterior_inward_values_excluded=True,maximum_roundoff_bound_fraction=float(ratio.max())))
        evidence += [gp,rp,out/"replay_rays_out.bin",out/"replay_out.bin"]
    report=dict(scope="실제 외부 확장→표면 입사 전달의 동일 내부 해; 다음 주파수 전 이력 재현은 아님",
                cases=cases,evidence_sha256={str(p):sha(p) for p in evidence+[Path(__file__),build/"formal-build.json"]},
                final_goal="NOT_JUDGED")
    with output.open("x") as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(cases,ensure_ascii=False))
