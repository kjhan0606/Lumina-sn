#!/usr/bin/env python3
"""원 주파수 재구성의 작은 비음수 다항식 민감도만 조사한다. 수송은 없다."""
import argparse
import ctypes as ct
from fractions import Fraction as F
import hashlib
import json
from math import comb
import os
from pathlib import Path
import subprocess
import time

import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def exact_basis():
    basis=[]
    for j in range(8):
        power=[F(1)];denominator=1
        for k in range(8):
            if k==j:continue
            new=[F(0)]*(len(power)+1)
            for m,v in enumerate(power):new[m]-=(k-3)*v;new[m+1]+=v
            power=new;denominator*=j-k
        basis.append([sum(F(comb(k,m),comb(7,m))*power[m]/denominator for m in range(k+1)) for k in range(8)])
    return basis


def exact_limited(values,basis):
    values=[F(float(x)) for x in values]
    b=[sum(basis[j][k]*values[j] for j in range(8)) for k in range(8)]
    b[0]=values[3];b[7]=values[4]
    low=[(1-F(k,7))*values[3]+F(k,7)*values[4] for k in range(8)]
    theta=min([F(1)]+[low[k]/(low[k]-b[k]) for k in range(8) if b[k]<0])
    b=[theta*x+(1-theta)*y for x,y in zip(b,low)]
    for degree in range(7,0,-1):b=[(b[k]+b[k+1])/2 for k in range(degree)]
    return float(b[0]),float(theta)


def run(root):
    root=Path(root).resolve();target=root/'condition.json';library=root/'condition.so'
    if target.exists() or library.exists():raise FileExistsError('기존 재구성 조사 보존')
    started=time.monotonic();paths=[Path(__file__),ROOT/'src/cmf_gpu_reconstruction.hpp',
        ROOT/'src/cmf_ordered_maps.hpp',ROOT/'tests/reconstruction_condition_probe.cpp']
    identities={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    command=[os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off','-Wall','-Wextra','-Werror',
        '-fPIC','-shared','-I'+str(ROOT/'src'),str(paths[-1]),'-o',str(library)]
    subprocess.run(command,check=True,capture_output=True,text=True,timeout=60)
    fun=ct.CDLL(str(library)).reconstruction_condition_probe
    a=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');i=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    fun.argtypes=[ct.c_int,a,a,a,a,i];fun.restype=ct.c_int
    basis=exact_basis();rows=[];groups=[]
    for power in (-500,0,500):
        scale=2.**power;x=np.arange(8,dtype='f8')-3;nominal=(x*(1-x))**2*scale
        cases=[('nominal',nominal)]
        for j in range(8):
            for direction in (-1,1):
                for steps in (1,2,4,8):
                    values=nominal.copy()
                    for _ in range(steps):values[j]=np.nextafter(values[j],np.inf if direction>0 else -np.inf)
                    if np.any(values<0):continue
                    cases.append((f'j{j}_{direction:+d}_{steps}ulp',values))
        first=len(rows)
        for name,values in cases:
            out=np.full(1,-17.);controls=np.full(8,-17.);theta=np.full(1,-17.);flags=np.full(2,-17,dtype='i4')
            code=fun(8,values,out,controls,theta,flags)
            if code:raise ValueError('원 재구성 CPU probe 실패: '+str(code))
            exact,exact_theta=exact_limited(values,basis)
            delta=float(abs(values-nominal).max())
            rows.append(dict(power=power,case=name,input=values.tolist(),perturbation=delta,
                output=float(out[0]),normalized_output=float(out[0]/scale),controls=controls.tolist(),theta=float(theta[0]),
                limited=bool(flags[0]),fallback=bool(flags[1]),exact_limited_output=exact,
                exact_theta=exact_theta,amplification_vs_nominal=(None if delta==0 else abs(float(out[0])-rows[first]['output'])/delta)))
        group=rows[first:];groups.append(dict(power=power,min_output=min(r['normalized_output'] for r in group),
            max_output=max(r['normalized_output'] for r in group),
            largest_amplification=max(r['amplification_vs_nominal'] or 0. for r in group)))
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in identities.items()):raise ValueError('조사 도중 소스 변경')
    record=dict(status='CPU_RECONSTRUCTION_CONDITION_OBSERVED_NOT_FIELD',input_sha256=identities,
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),groups=groups,rows=rows,
        scope='합성 p=x²(1-x)²의8점/미소ULP변화 조사. 실제native90 발생빈도·현재13003 원인·J/PHOT 오차·CMFGEN 불일치 원인 판정은 아니다.',
        actual_model_F_calls=0,GPU_seconds=0,final_goal='NOT_MET',elapsed_seconds=time.monotonic()-started)
    with target.open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=record['status'],groups=groups),ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();run(args.root)
