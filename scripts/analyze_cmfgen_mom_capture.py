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

MOM_ITERATION_LIMIT = 100
MOM_CHANGE_TOLERANCE = 1.0e-8


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
        if not 1<=iteration<=MOM_ITERATION_LIMIT or not 2<=n<=2000000:
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


def native_iteration_outcome(args,rows):
    """Classify the native loop using its actual advection/change/cap branches."""
    if not rows:
        raise ValueError("native MOM 반복 결과가 비어 있음")
    count=len(rows)
    limit_reached=count==MOM_ITERATION_LIMIT
    advection=bool(args.get("INCL_ADVEC_TERMS",False))
    max_error=None
    change_met=None
    if not advection:
        classification="ADVECTION_DISABLED_NONITERATIVE"
    else:
        max_error=float(rows[-1]["MAX_ER"])
        change_met=math.isfinite(max_error) and max_error<MOM_CHANGE_TOLERANCE
        if change_met and limit_reached:
            classification="CHANGE_TOLERANCE_MET_ON_ITERATION_LIMIT"
        elif change_met:
            classification="CHANGE_TOLERANCE_MET"
        elif limit_reached:
            classification="ITERATION_LIMIT_FORCED_NONCONVERGED_EXIT"
        else:
            classification="TRACE_ENDED_WITHOUT_STOP_CRITERION"
    return dict(advection_terms_enabled=advection,iteration_count=count,
        iteration_limit=MOM_ITERATION_LIMIT,iteration_limit_reached=limit_reached,
        final_MAX_ER=(number(max_error) if max_error is not None else None),
        relative_change_tolerance=(MOM_CHANGE_TOLERANCE if advection else None),
        change_criterion_met=change_met,classification=classification)


def _state_value(group,name):
    value=group[name]
    if isinstance(value,dict):
        if value.get("values") is None:
            raise ValueError("MOM 분석 배열 미할당: "+name)
        return np.asarray(value["values"])
    return np.asarray(value)


def entry_replay_summary(entry,final,call,target):
    """Summarize one replay of an entry-only snapshot; it is not a captured failure."""
    phase=entry["phase"]
    frequency=float(call["frequency"])
    if (len(phase)!=5 or phase!=call["phase"] or final["phase"]!=phase or
        int(phase[0])!=int(target["outer_iteration"]) or
        abs(frequency-float(target["frequency_native"]))>
            float(target["frequency_absolute_tolerance"]) or
        float(entry["args"]["FREQ"])!=frequency or
        float(_state_value(final["modules"]["jrel"],"FREQ_SAVE"))!=frequency):
        raise ValueError("entry/replay MOM phase 또는 주파수 불일치")

    rows=summary(call,entry["args"]["INNER_BND_METH"])
    if not rows:
        raise ValueError("entry replay의 native MOM 반복 기록이 비어 있음")
    iteration_outcome=native_iteration_outcome(entry["args"],rows)

    ray0=entry["modules"]["ray"]
    ray1=final["modules"]["ray"]
    jrel1=final["modules"]["jrel"]
    formal_names=("JNU_STORE","HNU_STORE","KNU_STORE","NNU_STORE")
    formal=[_state_value(ray0,name).astype(float,copy=False) for name in formal_names]
    if any(a.ndim!=1 or a.size!=formal[0].size for a in formal):
        raise ValueError("formal J/H/K/N 차원 불일치")
    if (int(_state_value(ray0,"HN_DEF_ON_NODES"))!=1 or
        int(_state_value(ray0,"ND_STORE"))!=formal[0].size):
        formal_support={"available":False,"reason":"formal H/N 노드 배치 불일치"}
    elif (not all(np.all(np.isfinite(a)) for a in formal) or
          np.any(formal[0]<=0.0)):
        formal_support={"available":False,"reason":"formal 모멘트가 비유한 또는 J<=0"}
    else:
        from analyze_cmfgen_initial_closure import angular_support
        formal_support={"available":True,**angular_support(*formal)}

    returned_j=_state_value(jrel1,"GAM_RSQJNU_SAVE").astype(float,copy=False)
    returned_h_all=_state_value(jrel1,"GAM_RSQHNU_SAVE").astype(float,copy=False)
    kmid_all=_state_value(ray1,"KMID_ON_J_SAVE").astype(float,copy=False)
    if returned_j.ndim!=1 or returned_j.size<2:
        raise ValueError("반환 MOM J 차원 불일치")
    n=returned_j.size
    if returned_h_all.shape!=(n,) or kmid_all.shape!=(n,):
        raise ValueError("반환 MOM J/H/K 경계 차원 불일치")
    returned_h=returned_h_all[:-1]
    kmid=kmid_all[:-1]
    returned_k=kmid*(returned_j[:-1]+returned_j[1:])
    denom=np.maximum(returned_j[:-1],returned_j[1:])*returned_k
    ratio=np.full(n-1,np.inf)
    good=np.isfinite(denom)&np.isfinite(returned_h)&(denom>0.0)
    np.divide(returned_h*returned_h,denom,out=ratio,where=good)
    ratio[(denom==0.0)&(returned_h==0.0)]=0.0
    invalid=(~np.isfinite(denom)|~np.isfinite(returned_h)|
             (returned_j[:-1]<0.0)|(returned_j[1:]<0.0)|(returned_k<0.0))
    violations=invalid|(ratio>1.0)
    last_iteration=max(call["iterations"])
    trace_final_j=np.asarray(call["iterations"][last_iteration]["ERR0"][1][1],dtype=float)
    if trace_final_j.shape!=returned_j.shape:
        raise ValueError("trace와 최종 MOM J 차원 불일치")
    returned_support=dict(
        condition="H^2 <= max(J_left,J_right)*K; conservative necessary condition",
        edge_count=n-1,max_ratio=number(np.max(ratio)),
        violating_edges_1based=(np.flatnonzero(violations)+1).tolist(),
        invalid_edges_1based=(np.flatnonzero(invalid)+1).tolist(),
        final_J_matches_last_trace=bool(np.array_equal(returned_j,trace_final_j)))
    formal_preserved={name:bool(np.array_equal(
        _state_value(ray0,name),_state_value(ray1,name))) for name in formal_names}
    return dict(phase=phase,frequency_native_units=frequency,target=target,
        replay_iterations=rows,native_iteration_outcome=iteration_outcome,
        formal_angular_support=formal_support,
        returned_mom_conservative_support=returned_support,
        formal_JHKN_preserved=formal_preserved,
        final_goal="NOT_JUDGED")


def verify_entry_replay_build(build):
    """Verify the targeted trace-flux replay build without the legacy runner gate."""
    build=Path(build).resolve()
    manifest=build/"capture-build.json"
    record=json.loads(manifest.read_text())
    target=record.get("capture_target")
    if (not isinstance(target,dict) or record.get("trace_flux") is not True or
        record.get("entry_only") is not False or record.get("first_repair") is not False):
        raise ValueError("반환 상태 저장용 target trace-flux build 필요")
    try:
        frequency=float(target["frequency_native"])
        outer=target["outer_iteration"]
        tolerance=float(target["frequency_absolute_tolerance"])
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError("trace-flux build target 형식 오류") from error
    if (not math.isfinite(frequency) or frequency<=0 or type(outer) is not int or outer<1 or
        not math.isfinite(tolerance) or tolerance<=0):
        raise ValueError("trace-flux build target 범위 오류")

    for group in ("input_sha256","output_sha256"):
        files=record.get(group)
        if not isinstance(files,dict) or not files:
            raise ValueError("trace-flux build 해시 목록 누락: "+group)
        for name,expected in files.items():
            path=Path(name)
            if not path.is_file() or sha(path)!=expected:
                raise ValueError("trace-flux build 파일/해시 불일치: "+name)

    parent_manifest=Path(record["parent"])/"parent-build-identity.json"
    if sha(parent_manifest)!=record.get("parent_manifest_sha256"):
        raise ValueError("trace-flux 부모 build identity 불일치")
    generator=Path(__file__).with_name("prepare_cmfgen_mom_capture.py")
    if sha(generator)!=record.get("generator_sha256"):
        raise ValueError("trace-flux build 생성기 신원 불일치")
    outputs={str(Path(name).resolve()) for name in record["output_sha256"]}
    required={str((build/name).resolve()) for name in ("replay_mom.exe","state_inventory.json")}
    if not required.issubset(outputs):
        raise ValueError("trace-flux replay 실행파일/상태 목록 누락")
    return record


def analyze_entry_only(run,replay,build,output):
    """Analyze a pre-MOM entry capture with exactly one instrumented replay."""
    if output.exists():
        raise ValueError("기존 entry replay 분석 보존")
    build=Path(build).resolve()
    build_record=verify_entry_replay_build(build)
    target=build_record["capture_target"]
    entry_path=run/"MOM_CAPTURE.bin"
    run_trace=run/"MOM_TRACE.bin"
    trace_path=replay/"MOM_TRACE.bin"
    final_path=replay/"MOM_FINAL.bin"
    h_trace=replay/"MOM_H_TRACE.bin"
    if run_trace.exists() and run_trace.stat().st_size:
        raise ValueError("entry-only 원 실행이 MOM 호출까지 진행함")
    if not final_path.is_file() or not h_trace.is_file() or h_trace.stat().st_size==0:
        raise ValueError("trace-flux replay의 H trace/final snapshot 누락; REPLAY mode 환경 확인")
    inventory_path=build/"state_inventory.json"
    inventory=json.loads(inventory_path.read_text())
    entry=capture(entry_path,inventory)
    final=capture(final_path,inventory)
    calls=trace(trace_path)
    if len(calls)!=1:
        raise ValueError("entry replay는 정확히 MOM 호출 한 건이어야 함")
    result=entry_replay_summary(entry,final,calls[0],target)
    parent_manifest=Path(build_record["parent"])/"parent-build-identity.json"
    files=[entry_path,trace_path,final_path,h_trace,build/"capture-build.json",
           inventory_path,build/"replay_mom.exe",parent_manifest,
           Path(__file__),Path(__file__).with_name("prepare_cmfgen_mom_capture.py")]
    report=dict(scope="entry-only capture -> 단일 MOM replay; 물리 수렴/CMFGEN 동질성 아님",
        entry_trace_absent=not run_trace.exists() or run_trace.stat().st_size==0,
        entry_capture_sha256=sha(entry_path),
        replay_evidence_sha256={str(p):sha(p) for p in files if p!=entry_path},
        **result)
    with output.open("x") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k!="replay_iterations"},ensure_ascii=False))
    print(json.dumps(dict(first=result["replay_iterations"][0],
                          last=result["replay_iterations"][-1]),ensure_ascii=False))
    return report


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
    original_outcome=native_iteration_outcome(inp["args"],rows)
    replay_outcome=native_iteration_outcome(inp["args"],again)
    failure=original_outcome["classification"]=="ITERATION_LIMIT_FORCED_NONCONVERGED_EXIT"
    replay_failure=replay_outcome["classification"]=="ITERATION_LIMIT_FORCED_NONCONVERGED_EXIT"
    files=[run/"MOM_CAPTURE.bin",run/"MOM_TRACE.bin",replay/"MOM_CAPTURE.bin",replay/"MOM_TRACE.bin",
           build/"capture-build.json",build/"state_inventory.json",Path(__file__)]
    controls={name:inp["args"][name] for name in ("INIT","COHERENT","INCL_ADVEC_TERMS","INCL_REL_TERMS",
               "INNER_BND_METH","OUTER_BND_METH","METHOD","N_TYPE","XM_CHK_OPTION","J_CHK_OPTION","H_CHK_OPTION")}
    report=dict(scope="동일 native MOM 한 호출의 실패 재현/선형계 잔차; 최종 물리 동질성 아님",
        phase=inp["phase"],frequency_native_units=a["frequency"],previous_frequency_native_units=a["previous_frequency"],
        controls=controls,capture_failure=failure,replay_failure=replay_failure,
        capture_native_iteration_outcome=original_outcome,
        replay_native_iteration_outcome=replay_outcome,
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
    parser.add_argument("--entry-only",action="store_true",
                        help="MOM 호출 전 포획 + 단일 trace-flux replay를 분석")
    args=parser.parse_args()
    if args.entry_only:
        analyze_entry_only(args.run,args.replay,args.build,args.output)
    else:
        analyze(args.run,args.replay,args.build,args.output)
