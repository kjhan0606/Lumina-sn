"""실제 첫 장/물질의 BF 흐름과 원본 CMFGEN PRRR_SL_V6 비교.

동일 pack sigma/GL8 적분을 공유하므로 원본 PHOT/주파수격자/전체 CMFGEN
동등성 시험이 아니다. 목적 준위 departure 및 실제 입자 흐름을 검사한다.
"""
import csv
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np

base=Path('/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results')
prior=base/'toy-material-425642'
response=base/'saved-field-response-20261017.vNeZQn'
h,k,c,me=6.62607015e-27,1.380649e-16,2.99792458e10,9.1093837015e-28
buf=np.memmap(prior/'iterate_input.bin',dtype='u1',mode='r')
ns,ni,nl,nelem,nf,gen,it=struct.unpack_from('<7Q',buf,72)
epoch,lo,dl=struct.unpack_from('<3d',buf,128)
assert ns==113 and it==0 and epoch==172800
start=152+4*(nl+2*ni);shell=39
T=struct.unpack_from('<d',buf,start+8*(3*ns+shell))[0]
ne=struct.unpack_from('<d',buf,start+8*((4+nelem)*ns+shell))[0]
J=np.frombuffer(buf,'<f8',ns*nf,start+8*ns*(5+nelem+ni+nl)).reshape(ns,nf)[shell]
assert abs(T-9461.6)<1e-8 and abs(ne/299562954966.39233-1)<1e-12
atom=list(csv.DictReader((prior/'input/levels.csv').open()))
pack=(prior/'input/bf_channels.bin').read_bytes()
assert hashlib.sha256(pack).hexdigest()=='656329de676b540a08beb1e6652e839785f799428a40b31fd2774ce44d8af0cf'
magic,version,nlevels,nr,nb,blo,bhi=struct.unpack_from('<IIiiidd',pack)
assert magic==0x43424643 and version==2 and nlevels==len(atom)
pos=36;off=np.frombuffer(pack,'<i4',nlevels+1,pos);pos+=4*(nlevels+1)
upper=np.frombuffer(pack,'<i4',nr,pos);pos+=4*nr
threshold=np.frombuffer(pack,'<f8',nr,pos);pos+=8*nr
sigma=np.frombuffer(pack,'<f8',nr*nb,pos).reshape(nr,nb)
edges=np.ascontiguousarray(blo*np.exp(np.arange(nb+1)*math.log(bhi/blo)/nb));edges[-1]=bhi
width=np.diff(edges);fe=lo*np.exp(np.arange(nf+1)*dl)
assert fe[0]<=blo and fe[-1]>=bhi
union=np.unique(np.r_[edges,fe[(fe>blo)&(fe<bhi)]])
left,right=union[:-1],union[1:];mid=.5*(left+right)
cb=np.searchsorted(edges,mid,side='right')-1;fb=np.searchsorted(fe,mid,side='right')-1
Jc=np.ascontiguousarray(np.bincount(cb,weights=J[fb]*(right-left),minlength=nb)/width)
assert np.max(abs(np.bincount(cb,weights=right-left,minlength=nb)/width-1))<1e-12

lib=ct.CDLL('./comparison.so');ptr=ct.POINTER(ct.c_double)
def p(x):
    assert x.dtype==np.float64 and (x.flags.c_contiguous or x.flags.f_contiguous)
    return x.ctypes.data_as(ptr)
lib.bf_vector.argtypes=[ct.c_int,ptr,ct.c_double,ptr]+[ct.c_double]*4+[ptr]*3
lib.bf_vector.restype=ct.c_int
lib.prrr_bridge.argtypes=[ct.c_int]+[ptr]*9;lib.prrr_bridge.restype=None
gx,gw=np.polynomial.legendre.leggauss(8)
lam=ne*(h*h/(2*math.pi*me*k*T))**1.5
results={}
for Z,expected in ((14,250),(16,322)):
    raw=(response/f'matrix_Z{Z}.bin').read_bytes();n=struct.unpack_from('<Q',raw,8)[0]
    x=np.frombuffer(raw,'<f8',n,24+8*n*n)
    meta={int(r['global_level']):r for r in csv.DictReader((response/f'levels_Z{Z}.csv').open())}
    pop={i:float(x[int(r['matrix_index'])])*float(r['within']) for i,r in meta.items()}
    ground=next(i for i,r in meta.items() if int(r['charge'])==2 and float(r['energy_eV'])==0)
    gg=float(atom[ground]['g']);nref=pop[ground]
    cases={}
    for label,parent_factor,field_factor in (('actual',1.,1.),('upper_departure_0.1',.1,1.),('upper_departure_10',10.,1.),('J_times_2',1.,2.)):
        err=0.;count=0;zero=0;totals=np.zeros(4);worst=None
        for lower,row in meta.items():
            if int(row['charge'])!=1:continue
            gl=float(atom[lower]['g']);nlower=pop[lower]
            for r in range(off[lower],off[lower+1]):
                u=int(upper[r]);assert u in meta and int(meta[u]['charge'])==2
                gu=float(atom[u]['g']);eu=float(atom[u]['energy_eV'])*1.602176634e-12
                th=float(threshold[r]);su=np.ascontiguousarray(sigma[r]);field=Jc*field_factor
                cup=np.empty(nb);cdn=np.empty(nb)
                assert lib.bf_vector(nb,p(edges),th,p(su),T,ne,gl,gu,p(field),p(cup),p(cdn))==0
                # CMFGEN HNST is relative to the actual ground-parent population.
                # LOG_DIST carries relative LTE parent populations, independently of actual DI.
                hnst=nref*lam*gl/(2*gg)*math.exp((h*th-eu)/(k*T))
                nupper=pop[u]*parent_factor
                di=np.empty((2,nb),order='F');di[0]=nref;di[1]=nupper
                ld=np.empty((2,nb),order='F');ld[0]=math.log(gg);ld[1]=math.log(gu)-eu/(k*T)
                hn=np.full(nb,nlower);hs=np.full(nb,hnst)
                active=(edges[1:]>th)&(su>0)
                w=np.zeros(nb);jp=np.zeros(nb);jr=np.zeros(nb)
                a=np.maximum(edges[:-1][active],th);b=edges[1:][active]
                freq=.5*(a[:,None]+b[:,None])+.5*(b-a)[:,None]*gx
                # Same full-bin averaged sigma and partial-bin GL8 as production.
                # Original CMFGEN integral uses Hz quadrature / frequency in 1e15 Hz.
                w[active]=4*math.pi/h/1e15*su[active]
                jp[active]=width[active]*np.sum(.5*gw*field[active,None]/(freq/1e15),axis=1)
                jr[active]=width[active]*np.sum(.5*gw*(2*h*freq**3/c**2+field[active,None])*
                                               np.exp(-h*freq/(k*T))/(freq/1e15),axis=1)
                pr=np.empty(nb);rr=np.empty(nb)
                lib.prrr_bridge(nb,*[p(v) for v in (w,hn,hs,di,ld,jp,jr,pr,rr)])
                expected_rates=np.array([nlower*cup.sum(),nupper*cdn.sum()])
                original_rates=np.array([pr.sum(),rr.sum()])
                assert np.all(np.isfinite(original_rates)) and np.all(original_rates>=0)
                scale=np.maximum(expected_rates,original_rates)
                delta=np.divide(abs(expected_rates-original_rates),scale,out=np.zeros(2),where=scale>0)
                if float(delta.max())>err:err=float(delta.max());worst=dict(lower=lower,channel=int(r),upper=u)
                assert np.all((expected_rates==0)==(original_rates==0))
                zero+=int(np.all(scale==0));count+=1;totals+=np.r_[expected_rates,original_rates]
        assert count==expected and err<2e-11
        cases[label]=dict(channels=count,zero_channels=zero,max_relative_difference=err,worst=worst,
                         lumina_forward_reverse=totals[:2].tolist(),cmfgen_forward_reverse=totals[2:].tolist())
    results[Z]=cases
out=dict(scope='same sigma/GL8/current shell39 J; original PRRR_SL_V6 departure and flow assembly only',
         T=T,ne=ne,epoch=epoch,results=results,new_RTE=False,production_changed=False,goal_achieved=False)
with Path('summary.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
