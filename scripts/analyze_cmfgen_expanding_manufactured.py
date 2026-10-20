#!/usr/bin/env python3
"""팽창 구의 독립 연속해와 실제 native angular 결과·에너지 장부를 비교한다."""
import json
from pathlib import Path
import struct

import numpy as np

from analyze_cmfgen_formal_energy import ledger


def exact_moments(r, a=.025, order=256):
    # beta=a*r, chi=eta/nu^3=1. ds=gamma*(1-a*z) dz를 해석 적분한다.
    x,w=np.polynomial.legendre.leggauss(order);result=[]
    for radius in r:
        beta=a*radius;mu_obs_core=np.sqrt(1-1/radius**2)
        mu_core=(mu_obs_core-beta)/(1-beta*mu_obs_core)
        value=np.zeros(4)
        for low,high in ((-1.,-beta),(-beta,mu_core),(mu_core,1.)):
            if high<=low:continue
            mu=low+(x+1)*(high-low)/2
            mu_obs=(mu+beta)/(1+beta*mu);z=radius*mu_obs
            p2=radius**2*(1-mu_obs**2);outer_z=np.sqrt(4-p2);q=np.sqrt(1-a*a*p2)
            path=(np.arcsin(a*z/q)+np.arcsin(a*outer_z/q))/a
            path+=a*(4-radius**2)/(np.sqrt(1-a*a*radius**2)+np.sqrt(1-4*a*a))
            intensity=-np.expm1(-path);intensity[mu>mu_core]=1
            for k in range(4):value[k]+=np.dot(w,mu**k*intensity)*(high-low)/4
        result.append(value)
    return np.array(result)


def analyze(root):
    path=root/'EXPANDING_TRANSFER.bin'
    with path.open('rb') as f:
        n,nf,nc,np_=struct.unpack('<4i',f.read(16))
        geometry=np.frombuffer(f.read(4*n*8),dtype='<f8').reshape(4,n).copy()
        p=np.frombuffer(f.read(np_*8),dtype='<f8').copy()
    dtype=np.dtype([('freq','<f8'),('moments','<f8',(4,n))])
    offset=16+(4*n+np_)*8
    if path.stat().st_size!=offset+nf*dtype.itemsize:raise ValueError('제조해 부분 출력')
    data=np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=(nf,))
    frequency=data['freq']
    weights=np.empty(nf);weights[0]=(frequency[0]-frequency[1])/2
    weights[-1]=(frequency[-2]-frequency[-1])/2;weights[1:-1]=(frequency[:-2]-frequency[2:])/2
    dtype=np.dtype([('freq','<f8'),('weight','<f8'),('field','<f8',(8,n))])
    records=np.zeros(nf,dtype=dtype);records['freq']=frequency;records['weight']=weights*1e15
    records['field'][:,:4,:]=data['moments'];records['field'][:,4,:]=1
    records['field'][:,5,:]=frequency[:,None]**3;records['field'][:,7,:]=data['moments'][:,0,:]
    summary,_,_=ledger(geometry,records)
    r=geometry[0];exact=exact_moments(r)
    final=data['moments'][-1].T/frequency[-1]**3
    errors=np.max(abs(final-exact),axis=0)
    radial=np.loadtxt(root/'EXPANDING_RADIAL_COORDINATE.dat');a=.025
    primitive=(np.arcsin(a*r)+np.sqrt(1-a*a*r*r))/a
    radial_error=float(np.max(abs((radial[:,1]-radial[-1,1])-(primitive-primitive[-1]))))
    return dict(n=n,n_frequency_intervals=nf-1,n_core=nc,final_moment_max_absolute_error=dict(zip(('J','H','K','N'),map(float,errors))),
                radial_characteristic_max_absolute_error=radial_error,energy=summary,
                scope='구형 팽창 제조해만; 실제90격자/SiS/앵커 판정 아님',final_goal='NOT_JUDGED')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    a=p.parse_args();print(json.dumps(analyze(a.root),ensure_ascii=False,indent=2))
