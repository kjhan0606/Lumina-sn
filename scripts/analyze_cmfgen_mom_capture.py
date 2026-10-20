#!/usr/bin/env python3
"""포획된 native MOM의 동일 입력·내부 반복·선형계/대체 후 잔차를 판독한다."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from prepare_cmfgen_mom_capture import ARGUMENTS, sha


class Reader:
    def __init__(self, raw):
        self.raw,self.offset=raw,0

    def take(self,n):
        if n<0 or self.offset+n>len(self.raw):
            raise ValueError("MOM 이진 자료 절단")
        result=self.raw[self.offset:self.offset+n]
        self.offset+=n
        return result

    def integers(self,n=1):
        return struct.unpack("="+str(n)+"i",self.take(4*n))

    def reals(self,n=1):
        return np.frombuffer(self.take(8*n),dtype="=f8").copy()


def capture(path,inventory):
    r=Reader(path.read_bytes())
    if r.take(16).rstrip()!=b"LUMINAMOMCAPV1" or r.integers(4)!=(8,64,32,32):
        raise ValueError("native capture 형식/kind 불일치")
    phase=r.integers(5)
    nd=r.integers()[0]
    if not 2<=nd<=2000000:
        raise ValueError("capture ND 범위")
    args={"ND_SM":nd}
    for name,kind,array in ARGUMENTS:
        if name=="ND_SM":
            continue
        if kind=="C":
            length=r.integers()[0]
            if not 1<=length<=200:
                raise ValueError("capture 문자열 길이")
            args[name]=r.take(length).decode("ascii")
        elif kind=="L":
            args[name]=bool(r.integers()[0])
        else:
            value=r.reals(nd if array else 1)
            args[name]=value if array else float(value[0])
    modules={}
    for group in ("jrel","ray"):
        fields={}
        for f in inventory[group]:
            count=1
            bounds=None
            if f["alloc"]:
                present=r.integers()[0]
                if present not in (0,1):
                    raise ValueError("capture allocation flag")
                if not present:
                    fields[f["name"]]=None
                    continue
                bounds=r.integers(2)
                count=bounds[1]-bounds[0]+1
                if not 0<=count<=2000000:
                    raise ValueError("capture array 크기")
            elif f["fixed"]:
                if f["fixed"]!="N_ERR_MAX":
                    raise ValueError("미지원 fixed 크기")
                count=1000
            values=r.reals(count) if f["type"].startswith("REAL") else np.array(r.integers(count))
            fields[f["name"]]=dict(bounds=bounds,values=values) if f["alloc"] or f["fixed"] else values[0].item()
        modules[group]=fields
    if r.take(16).rstrip()!=b"LUMINAMOMENDV1" or r.offset!=len(r.raw):
        raise ValueError("capture 끝/후행 자료 불일치")
    return dict(phase=phase,args=args,modules=modules)


def first_event(path):
    r=Reader(path.read_bytes())
    if r.take(16).rstrip()!=b"LUMINAMOMFAULT1" or r.integers(4)!=(8,64,32,32):
        raise ValueError("최초 사건 형식/kind 불일치")
    phase=r.integers(5)
    kind,iteration,bad,n,pending=r.integers(5)
    if kind not in (1,2,3,4) or not 1<=iteration<=100 or not 3<=n<=2000000 or pending not in (0,1):
        raise ValueError("최초 사건 종류/차원 불일치")
    if (kind==3 and bad!=0) or (kind!=3 and not 1<=bad<=n-(kind==2)):
        raise ValueError("최초 사건 위치 불일치")
    frequency=float(r.reals()[0])
    matrix=r.reals(4*n).reshape((4,n),order="F")
    raw,hraw,slope=(r.reals(n) for _ in range(3))
    if r.take(16).rstrip()!=b"LUMINAMOMFEND1" or r.offset!=len(r.raw):
        raise ValueError("최초 사건 끝/후행 자료 불일치")
    return dict(phase=phase,kind=kind,iteration=iteration,bad_1based=bad,ND=n,
                initial_pending=bool(pending),frequency=frequency,matrix=matrix,raw=raw,hraw=hraw,slope=slope)


def trace(path):
    r=Reader(path.read_bytes())
    calls=[]
    while r.offset<len(r.raw):
        start=r.offset
        tag=r.take(4)
        if tag==b"CALL":
            phase=r.integers(5)
            frequency,previous=r.reals(2)
            nd=r.integers()[0]
            calls.append(dict(phase=phase,frequency=float(frequency),previous_frequency=float(previous),ND_SM=nd,
                              iterations={},records=[r.raw[start:r.offset]]))
            continue
        if not calls or tag not in (b"TRI0",b"RAW0",b"ERR0"):
            raise ValueError("MOM trace 순서/태그 불일치")
        iteration,n=r.integers(2)
        if not 1<=iteration<=100 or not 2<=n<=2000000:
            raise ValueError("MOM trace 차원/반복 불일치")
        current=calls[-1]["iterations"].setdefault(iteration,dict(ND=n))
        if current["ND"]!=n or tag.decode() in current:
            raise ValueError("MOM trace 반복 중복/차원 변경")
        if tag==b"TRI0":
            current["TRI0"]=r.reals(4*n).reshape(4,n)
        elif tag==b"RAW0":
            current["RAW0"]=r.reals(n)
        else:
            current["ERR0"]=(float(r.reals()[0]),r.reals(3*n).reshape(3,n))
        calls[-1]["records"].append(r.raw[start:r.offset])
    return calls


def residual(A,x):
    lower,diagonal,upper,rhs=A.astype(np.longdouble)
    x=x.astype(np.longdouble)
    lhs=diagonal*x
    scale=np.abs(rhs)+np.abs(lhs)
    lhs[1:]+=lower[1:]*x[:-1]
    lhs[:-1]+=upper[:-1]*x[1:]
    scale[1:]+=np.abs(lower[1:]*x[:-1])
    scale[:-1]+=np.abs(upper[:-1]*x[1:])
    error=np.abs(lhs-rhs)
    ratio=np.zeros(len(x),dtype=np.longdouble)
    np.divide(error,scale,out=ratio,where=scale>0)
    ratio[(scale==0)&(error!=0)]=np.inf
    return float(np.max(ratio))


def number(value):
    return float(value) if math.isfinite(float(value)) else str(float(value))


def summary(call,inner):
    result=[]
    if sorted(call["iterations"])!=list(range(1,max(call["iterations"],default=0)+1)):
        raise ValueError("native 내부 반복 누락")
    for i,data in sorted(call["iterations"].items()):
        if not all(k in data for k in ("TRI0","RAW0","ERR0")):
            raise ValueError("불완전 native 반복")
        error,(old,new,slope)=data["ERR0"]
        raw=data["RAW0"]
        row=dict(iteration=i,MAX_ER=number(error),negative_before_replacement=int(np.sum(raw<0)),
                 replaced_rows=int(np.count_nonzero(raw!=new)),raw_min=number(np.min(raw)),
                 after_min=number(np.min(new)),log_derivative_min=number(np.min(slope)),
                 log_derivative_max=number(np.max(slope)))
        if inner.strip()=="DIFFUSION":
            row.update(linear_residual_before=number(residual(data["TRI0"],raw)),
                       linear_residual_after_replacement=number(residual(data["TRI0"],new)))
        result.append(row)
    return result


def analyze(run,replay,build,output):
    if output.exists():
        raise ValueError("기존 MOM 분석 보존")
    inventory=json.loads((build/"state_inventory.json").read_text())
    inp=capture(run/"MOM_CAPTURE.bin",inventory)
    original=trace(run/"MOM_TRACE.bin")
    repeated=trace(replay/"MOM_TRACE.bin")
    if not original or len(repeated)!=1:
        raise ValueError("실패 호출 선택 불가")
    a,b=original[-1],repeated[0]
    if a["phase"]!=inp["phase"] or a["frequency"]!=inp["args"]["FREQ"]:
        raise ValueError("capture/trace 호출 정체성 불일치")
    rows=summary(a,inp["args"]["INNER_BND_METH"])
    again=summary(b,inp["args"]["INNER_BND_METH"])
    failure=bool(len(rows)==100 and float(rows[-1]["MAX_ER"])>=1e-8)
    replay_failure=bool(len(again)==100 and float(again[-1]["MAX_ER"])>=1e-8)
    files=[run/"MOM_CAPTURE.bin",run/"MOM_TRACE.bin",replay/"MOM_CAPTURE.bin",replay/"MOM_TRACE.bin",
           build/"capture-build.json",build/"state_inventory.json",Path(__file__)]
    controls={name:inp["args"][name] for name in ("INIT","COHERENT","INCL_ADVEC_TERMS","INCL_REL_TERMS",
               "INNER_BND_METH","OUTER_BND_METH","METHOD","N_TYPE","XM_CHK_OPTION","J_CHK_OPTION","H_CHK_OPTION")}
    report=dict(scope="동일 native MOM 한 호출의 실패 재현/선형계 잔차; 최종 물리 동질성 아님",
        phase=inp["phase"],frequency_native_units=a["frequency"],previous_frequency_native_units=a["previous_frequency"],
        controls=controls,capture_failure=failure,replay_failure=replay_failure,
        input_snapshot_byte_equal=sha(run/"MOM_CAPTURE.bin")==sha(replay/"MOM_CAPTURE.bin"),
        failing_call_trace_byte_equal=a["records"]==b["records"],original_target_call_count=len(original),
        iterations=rows,replay_iterations=again,
        evidence_sha256={str(p):sha(p) for p in files},final_goal="NOT_JUDGED")
    with output.open("x") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("iterations","replay_iterations","evidence_sha256")},ensure_ascii=False))
    print(json.dumps(dict(first=rows[0],last=rows[-1]),ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","replay","build","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    analyze(args.run,args.replay,args.build,args.output)
