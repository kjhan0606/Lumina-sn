"""봉인된737열 일반 χ 사례만의 CPU 정확도/비용 대조. 생산 실행기가 아니다."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROFILE=Path('/gpfs/kjhan/cmfgen_runs/lumina_B_native_case_20261022-GSf2Dk/failed-profile.npz')
PROFILE_HASH='1b65ed7c06be54bf128274ae9117fc29dcd6a8e8978c27f8149d8851c73d99d2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_reference(values,parameters,order):
    length,shift,span,delta,incident,_=parameters
    nf=values.shape[1]
    edges=np.unique([0.,1.]+[float(u) for u in (np.arange(nf)-shift)/span if 0.<u<1.])
    x,w=np.polynomial.legendre.leggauss(order);t=(x+1)/2;w=w/2
    def field(u,k):
        u=np.asarray(u);coordinate=shift+span*u
        a=np.interp(coordinate,np.arange(nf),values[k])
        b=np.interp(coordinate,np.arange(nf),values[k+1])
        return u*a+(1-u)*b
    tau=0.;source=0.
    for a,b in zip(edges[:-1],edges[1:]):
        du=b-a;H=length*du
        k0,km,k1=[float(field(u,0)) for u in (a,(a+b)/2,b)]
        c2=2*(k1-2*km+k0);c1=k1-k0-c2
        depth=tau+H*(k0*t+c1*t*t/2+c2*t*t*t/3)+3*delta*du*t
        source+=H*np.sum(w*field(a+du*t,2)*np.exp(-depth))
        tau+=H*(k0+c1/2+c2/3)+3*delta*du
    return float(incident*np.exp(-tau)+source)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-root',required=True,type=Path);args=ap.parse_args()
    if socket.gethostname().lower()!='lageunha':
        raise RuntimeError('이 CPU 시험은 확인된 Lageunha에서만 실행한다')
    root=args.output_root.resolve()
    if not root.is_dir() or (root/'case.json').exists() or (root/'measure.so').exists():
        raise RuntimeError('새 분리 출력 root가 필요하다')
    if digest(PROFILE)!=PROFILE_HASH:
        raise RuntimeError('고정 사례 identity가 다르다')
    record={'schema':'lumina-source-measure-fixed737-v1','scope':'CPU 고정 사례만; GPU 성능·전체장 ETA·CMFGEN J 일치가 아니다',
            'profile':str(PROFILE),'profile_sha256':PROFILE_HASH,'rows':[],
            'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in
                [ROOT/'src/cmf_source_measure.hpp',ROOT/'src/cmf_characteristic_linear.hpp',
                 ROOT/'src/cmf_linear_moments.hpp',ROOT/'src/cmf_ordered_maps.hpp',
                 ROOT/'tests/source_measure_probe.cpp',Path(__file__).resolve()]}}
    command=[os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off','-Wall','-Wextra','-Werror',
             '-fPIC','-shared','-I'+str(ROOT/'src'),str(ROOT/'tests/source_measure_probe.cpp'),'-o',str(root/'measure.so')]
    subprocess.run(command,check=True,timeout=60)
    record['library_sha256']=digest(root/'measure.so')
    lib=ct.CDLL(str(root/'measure.so'));fun=lib.source_measure_cost
    p=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    e=np.ctypeslib.ndpointer(dtype='i8',flags='C_CONTIGUOUS')
    i=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    u=np.ctypeslib.ndpointer(dtype='u8',flags='C_CONTIGUOUS')
    fun.argtypes=[ct.c_int,ct.c_int,p,p,ct.c_int,p,e,i,p,u];fun.restype=ct.c_int
    with np.load(PROFILE) as data:
        values=np.ascontiguousarray(data['values']);parameters=np.ascontiguousarray(data['parameters'])
    record['parameters']=parameters.tolist()
    started=time.perf_counter()
    refs=[independent_reference(values,parameters,order) for order in (96,192)]
    record['reference96'],record['reference192']=refs
    record['reference_self_relative_gap']=abs(refs[0]/refs[1]-1)
    record['reference_seconds']=time.perf_counter()-started
    good=record['reference_self_relative_gap']<1e-12
    for block in (1,4,16,32):
        m=np.full(7,-17.);ex=np.full(7,-17,dtype='i8');counts=np.full(5,-17,dtype='i4')
        timings=np.zeros(2);stats=np.zeros(4,dtype='u8')
        code=fun(values.shape[1],block,values,parameters,100000,m,ex,counts,timings,stats)
        row={'block':block,'status':code}
        if code==0:
            decoded=np.ldexp(m,ex.astype('i4'));value,error=decoded[0],decoded[3]
            row.update(value=float(value),error=float(error),relative_gap=abs(float(value/refs[1]-1)),
                       build_seconds=float(timings[0]),transport_seconds=float(timings[1]),
                       counts=dict(zip(['panels','accepted','split','pruned','max_depth'],map(int,counts))),
                       reader_calls=int(stats[0]),build_reader_calls=int(stats[1]),query_reader_calls=int(stats[0]-stats[1]),
                       tree_bytes=int(stats[2]),source_node_bytes=int(stats[3]),
                       error_contains_reference=bool(abs(value-refs[1])<=error),
                       own_relative_budget_met=bool(error<=parameters[5]*value))
            good=good and row['error_contains_reference'] and row['own_relative_budget_met']
        else:
            good=False
        record['rows'].append(row)
    record['status']='CPU_FIXED737_ACCURACY_MET_NOT_PRODUCTION' if good else 'CPU_FIXED737_NOT_MET'
    with (root/'case.json').open('x') as f:
        json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),flush=True)
    return 0 if good else 1


if __name__=='__main__':
    raise SystemExit(main())
