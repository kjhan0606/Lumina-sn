#!/usr/bin/env python3
"""고정된 양의 전이율의 nominal 인구 해를 외향 반올림 구간으로 인증한다.

생산 rank/조건수 채택 판정은 바꾸지 않는다. 원자자료 오차/공동수렴/참조 진실의
인증이 아니라 고정 binary64 off-diagonal 율에 대한 수치 전방오차의 경계다.
"""
import argparse
import csv
import ctypes as C
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from prepare_cmfgen_mom_capture import sha
from summarize_cmfgen_einstein_response import load_case


def load_library(path):
    lib=C.CDLL(str(path))
    lib.generator_interval_precision.restype=C.c_int
    lib.generator_interval_environment.restype=C.c_int
    if lib.generator_interval_precision()!=np.finfo(np.longdouble).nmant+1:
        raise ValueError('C/NumPy long double 정밀도 불일치')
    ptr=C.c_void_p
    lib.generator_interval_stationary.argtypes=[ptr,C.c_size_t,ptr,ptr]
    lib.generator_interval_stationary.restype=C.c_int
    lib.generator_interval_group_sums.argtypes=[ptr,ptr,C.c_size_t,ptr,C.c_size_t,ptr,ptr]
    lib.generator_interval_group_sums.restype=C.c_int
    lib.generator_interval_candidate_error.argtypes=[ptr,ptr,ptr,C.c_size_t,C.c_double,ptr,ptr]
    lib.generator_interval_candidate_error.restype=C.c_int
    lib.generator_interval_certify_candidate.argtypes=[ptr,C.c_size_t,ptr,C.c_double,ptr,C.c_size_t,ptr,ptr,ptr,ptr]
    lib.generator_interval_certify_candidate.restype=C.c_int
    return lib


def enclose(lib,matrix,groups=None):
    matrix=np.asfortranarray(matrix,dtype=float)
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError('정방 유한 generator 필요')
    n=len(matrix);low=np.zeros(n,dtype=np.longdouble);high=np.zeros_like(low)
    status=lib.generator_interval_stationary(matrix.ctypes.data,n,low.ctypes.data,high.ctypes.data)
    if status:
        raise ValueError('구간 GTH 인증 실패 status='+str(status))
    if groups is None:return low,high
    groups=np.asarray(groups,dtype=np.int32)
    if groups.shape!=(n,) or np.any(groups<0):raise ValueError('인구 그룹 오류')
    ng=int(groups.max())+1;glow=np.zeros(ng,dtype=np.longdouble);ghigh=np.zeros_like(glow)
    status=lib.generator_interval_group_sums(low.ctypes.data,high.ctypes.data,n,
        groups.ctypes.data,ng,glow.ctypes.data,ghigh.ctypes.data)
    if status:raise ValueError('그룹 구간 인증 실패')
    return low,high,glow,ghigh


def candidate_error(lib,low,high,pop,total):
    pop=np.ascontiguousarray(pop,dtype=float)
    absolute=np.zeros(len(pop),dtype=np.longdouble);relative=np.zeros_like(absolute)
    status=lib.generator_interval_candidate_error(low.ctypes.data,high.ctypes.data,pop.ctypes.data,len(pop),total,
        absolute.ctypes.data,relative.ctypes.data)
    if status:raise ValueError('실제 density의 전방오차 경계 실패 status='+str(status))
    return absolute,relative


def certify(root,out,cases):
    if out.exists():raise ValueError('기존 수치 전방오차 인증 보존')
    out.mkdir();source=Path(__file__).with_name('certify_generator_interval.c')
    header=Path(__file__).resolve().parents[1]/'src/population_generator_interval.h'
    shutil.copy2(header,out/header.name)
    shutil.copy2(source,out/source.name);shutil.copy2(__file__,out/Path(__file__).name)
    library=out/'interval.so'
    flags=['gcc','-shared','-fPIC','-O2','-fno-fast-math','-ffp-contract=off','-frounding-math',
           str(out/source.name),'-lm','-o',str(library)]
    subprocess.run(flags,check=True,capture_output=True,timeout=30)
    lib=load_library(library);results=[]
    for Z,depth in cases:
        started=time.monotonic()
        identity,planes,pop,files=load_case(root,depth,Z)
        matrix=sum(planes.values());groups=[int(r['spectroscopic_stage'])-2 for r in identity]
        low,high,glow,ghigh=enclose(lib,matrix,groups)
        boundary=next(csv.DictReader(files['boundary_mass'].open()))
        total=float(boundary['n_element']);nominal=pop.astype(np.longdouble)/np.longdouble(total)
        absolute,relative=candidate_error(lib,low,high,pop,total)
        binary=out/f'z{Z}_depth{depth}_intervals.npz'
        np.savez(binary,lower=low,upper=high,stage_lower=glow,stage_upper=ghigh,
                 C_nominal_probability=nominal,groups=np.array(groups),
                 probability_error_upper=absolute,relative_error_upper=relative)
        diagnostic=Path(str(files['matrix_raw']).replace('matrix_raw.csv','diagnostics.csv'))
        old={r['key']:r['value'] for r in csv.DictReader(diagnostic.open())}
        stages=[]
        for i in range(len(glow)):
            stages.append(dict(stage=i+2,
                lower_approx=np.format_float_scientific(glow[i],precision=25,unique=False),
                upper_approx=np.format_float_scientific(ghigh[i],precision=25,unique=False),
                interval_width_approx=float(ghigh[i]-glow[i])))
        results.append(dict(Z=Z,depth_1based=depth,N=len(matrix),seconds=time.monotonic()-started,
            exact_input_rate_enclosure=True,old_raw_rank=int(old['raw_rank']),old_kappa=float(old['kappa_2']),
            maximum_probability_width_approx=float(np.max(high-low)),
            C_nominal_max_abs_error_bound_approx=float(np.max(np.maximum(abs(nominal-low),abs(nominal-high)))),
            rigorous_probability_error_upper_display=float(absolute.max()),
            rigorous_relative_error_upper_display=float(relative.max()),
            stages=stages,authoritative_binary=str(binary),binary_sha256=sha(binary),
            input_sha256={str(p):sha(p) for p in [*files.values(),diagnostic]}))
    report=dict(scope=__doc__,cases=results,long_double_mantissa_bits=lib.generator_interval_precision(),
        rounding_environment_status=lib.generator_interval_environment(),x87_extended_precision_checked=True,
        arithmetic='각 양의 합/곱/몫을 long double nearest 저장 후 nextafterl로 외향 확장; 정확한0만 생략',
        generator='생산 GTH와 같은 binary64 off-diagonal 전이율; 대각은 그 정확한 유출률 합으로 정의',
        bounds='NPZ longdouble 배열이 권위 경계다. JSON 십진 표시는 근삿값이며 원자율 불확실성을 포함하지 않는다.',
        production_adoption_changed=False,rank_kappa_criteria_changed=False,final_goal='NOT_JUDGED',
        compiler_command=flags,producer_sha256={str(p):sha(p) for p in (source,header,Path(__file__),library)})
    (out/'interval-certification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cases=results,production_adoption_changed=False),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',default='14:28,14:61')
    a=p.parse_args();certify(a.root.resolve(),a.out.resolve(),[tuple(map(int,s.split(':'))) for s in a.cases.split(',')])
