"""같은 저장장/원자료/열·비열 행렬에서 BF 장의 bin 평균화만 분리한다.
생산 수정이나 새 RTE가 아니다. 기존 범위의 반복4, 셸39 조건부 응답이다.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np

base=Path('/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results')
prior=base/'toy-material-425144'
matrix_root=base/'nt-line-response-20261016.3ZVCFq'
h,k,c,me=6.62607015e-27,1.380649e-16,2.99792458e10,9.1093837015e-28

def solve(A,anchor):
    M=np.array(A,dtype=np.longdouble,copy=True)
    p=anchor;M[p]=1
    b=np.zeros(len(M),dtype=np.longdouble);b[p]=1
    scale=np.max(abs(M),axis=1);M/=scale[:,None];b/=scale
    for j in range(len(M)-1):
        p=j+int(abs(M[j:,j]).argmax());assert M[p,j]!=0
        if p!=j:M[[j,p]]=M[[p,j]];b[[j,p]]=b[[p,j]]
        f=M[j+1:,j]/M[j,j]
        M[j+1:,j+1:]-=f[:,None]*M[j,j+1:];b[j+1:]-=f*b[j];M[j+1:,j]=0
    x=np.empty(len(M),dtype=np.longdouble)
    for j in range(len(M)-1,-1,-1):x[j]=(b[j]-M[j,j+1:]@x[j+1:])/M[j,j]
    assert np.all(x>=0) and abs(x.sum()-1)<1e-12
    return x

raw=(prior/'iterate_input.bin').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='244806e4ab10c0b1d821735713cca8c0110e927194210688333bc9dfb58f54c3'
ns,ni,nl,nelem,nf,gen,it=struct.unpack_from('<7Q',raw,72)
time,lo,dl=struct.unpack_from('<3d',raw,128);assert ns==113 and it==4
start=152+4*(nl+2*ni)
T=struct.unpack_from('<d',raw,start+8*(3*ns+39))[0]
ne=struct.unpack_from('<d',raw,start+8*((4+nelem)*ns+39))[0]
J=np.frombuffer(raw,'<f8',ns*nf,start+8*ns*(5+nelem+ni+nl)).reshape(ns,nf)[39]
atom=list(csv.DictReader((prior/'input/levels.csv').open()))
bf=(prior/'input/bf_channels.bin').read_bytes()
assert hashlib.sha256(bf).hexdigest()=='656329de676b540a08beb1e6652e839785f799428a40b31fd2774ce44d8af0cf'
magic,version,nlevels,nr,nb,blo,bhi=struct.unpack_from('<IIiiidd',bf)
assert magic==0x43424643 and version==2 and nlevels==len(atom)
pos=36;off=np.frombuffer(bf,'<i4',nlevels+1,pos);pos+=4*(nlevels+1)
upper=np.frombuffer(bf,'<i4',nr,pos);pos+=4*nr
threshold=np.frombuffer(bf,'<f8',nr,pos);pos+=8*nr
sigma=np.frombuffer(bf,'<f8',nr*nb,pos).reshape(nr,nb);pos+=8*nr*nb
upper_g=np.frombuffer(bf,'<f8',nr,pos);lower=np.repeat(np.arange(nlevels),np.diff(off))
edges=blo*np.exp(np.arange(nb+1)*math.log(bhi/blo)/nb);edges[-1]=bhi;bw=np.diff(edges)
fe=lo*np.exp(np.arange(nf+1)*dl);assert fe[0]<=blo and fe[-1]>=bhi
union=np.unique(np.r_[edges,fe[(fe>blo)&(fe<bhi)]])
left,right=union[:-1],union[1:];mid=.5*(left+right)
cb=np.searchsorted(edges,mid,side='right')-1;fb=np.searchsorted(fe,mid,side='right')-1
Jc=np.bincount(cb,weights=J[fb]*(right-left),minlength=nb)/bw
assert np.max(abs(np.bincount(cb,weights=right-left,minlength=nb)/bw-1))<1e-12

def integrated(temperature,order):
    gx,gw=np.polynomial.legendre.leggauss(order);a=h/(k*temperature)
    def segments(l,r,field,reference):
        nu=.5*(l[:,None]+r[:,None])+.5*(r-l)[:,None]*gx
        up=field*np.log1p((r-l)/l)/h
        down=field*.5*(r-l)*np.sum(gw*np.exp(-a*(nu-reference[:,None]))/(h*nu),axis=1)
        return up,down
    pf,rf=segments(left,right,J[fb],edges[cb])
    pc,rc=segments(left,right,Jc[cb],edges[cb])
    allbins=[np.bincount(cb,weights=q,minlength=nb) for q in (pf,rf,pc,rc)]
    def channel(r,bins):
        arrays=[q[bins].copy() for q in allbins]
        first=bins[0];th=threshold[r]
        if edges[first]<th:
            take=(cb==first)&(right>th)
            l=np.maximum(left[take],th);rr=right[take]
            for ix,field in ((0,J[fb[take]]),(2,Jc[cb[take]])):
                p,d=segments(l,rr,field,np.full(len(l),edges[first]))
                arrays[ix][0]=p.sum();arrays[ix+1][0]=d.sum()
        width=edges[bins+1]-np.maximum(edges[bins],th)
        assert np.all(width>0)
        mult=4*math.pi*sigma[r,bins]*bw[bins]/width
        reference=np.exp(-a*(edges[bins]-th))
        return [float(np.sum(mult*q*(reference if i%2 else 1))) for i,q in enumerate(arrays)]
    return channel

states={}
for Z in (14,16):
    buf=(matrix_root/f'matrix_Z{Z}.bin').read_bytes();n=struct.unpack_from('<Q',buf,8)[0]
    total=struct.unpack_from('<d',buf,16)[0]
    A=np.frombuffer(buf,'<f8',n*n,24).reshape(n,n,order='F').astype(np.longdouble)
    saved=np.frombuffer(buf,'<f8',n,24+8*n*n)/total;stage=np.full(n,-1);meta={}
    for r in csv.DictReader((matrix_root/f'levels_Z{Z}.csv').open()):
        gl=int(r['global_level']);i=int(r['matrix_index']);stage[i]=int(r['charge']);meta[gl]=(i,float(r['within']))
    stage[-1]=max(stage)+1
    states[Z]=dict(A=A,saved=saved,stage=stage,meta=meta)

def add(M,l,u,ru,rd):
    M[u,l]+=ru;M[l,l]-=ru;M[l,u]+=rd;M[u,u]-=rd

results={};gx,gw=np.polynomial.legendre.leggauss(8);gw*=.5
for order in (2,4):
    rate=integrated(T,order)
    delta={Z:np.zeros_like(s['A']) for Z,s in states.items()}
    control={Z:np.zeros_like(s['A']) for Z,s in states.items()}
    channels={Z:0 for Z in states};control_error=0
    for r,gl in enumerate(lower):
        Z=int(atom[gl]['atomic_number'])
        if Z not in states:continue
        s=states[Z];l,wl=s['meta'][int(gl)];gu=int(upper[r])
        if gu>=0:u,wu=s['meta'][gu];ug=float(atom[gu]['g'])
        else:u=len(s['A'])-1;wu=1.;ug=upper_g[r]
        bins=np.flatnonzero(sigma[r]>0)
        if not len(bins):continue
        channels[Z]+=1;th=threshold[r]
        lhs=np.maximum(edges[bins],th);rhs=edges[bins+1]
        nu=.5*(lhs[:,None]+rhs[:,None])+.5*(rhs-lhs)[:,None]*gx
        inv=ne*(h*h/(2*math.pi*me*k*T))**1.5*float(atom[gl]['g'])/(2*ug)
        factor=4*math.pi*sigma[r,bins]*bw[bins]
        old_up=float(np.sum(factor*Jc[bins]*np.sum(gw/(h*nu),axis=1)))
        old_down=float(np.sum(factor*Jc[bins]*np.sum(gw*np.exp(-h*(nu-th)/(k*T))/(h*nu),axis=1)))
        fine_up,fine_down,flat_up,flat_down=rate(r,bins)
        for v,w in ((flat_up,old_up),(flat_down,old_down)):
            if w>1e-200:control_error=max(control_error,abs(v/w-1))
        add(delta[Z],l,u,(fine_up-old_up)*wl,(fine_down-old_down)*inv*wu)
        add(control[Z],l,u,(flat_up-old_up)*wl,(flat_down-old_down)*inv*wu)
    assert control_error<1e-9
    values={}
    for Z,s in states.items():
        ii=s['stage']==1;iii=s['stage']==2
        ratio=lambda x:float(x[iii].sum()/x[ii].sum())
        anchor=int(np.argmax(s['saved']))
        baseline=ratio(solve(s['A'],anchor));expected=ratio(s['saved']);assert abs(baseline/expected-1)<1e-8
        flat=ratio(solve(s['A']+control[Z],anchor));assert abs(flat/baseline-1)<1e-8
        fine=ratio(solve(s['A']+delta[Z],anchor))
        values[Z]=dict(channels=channels[Z],coarse_ratio=baseline,fine_ratio=fine,
                       relative_change=fine/baseline-1,constant_bin_control_relative=flat/baseline-1)
    results[order]=dict(GL8_constant_bin_rate_control_max=control_error,ions=values)
for Z in states:assert abs(results[2]['ions'][Z]['fine_ratio']/results[4]['ions'][Z]['fine_ratio']-1)<1e-8
output=dict(scope='native iteration4 shell39; original BB domain; fixed T/ne/NT; BF J representation only; no RTE',
            T=T,ne=ne,orders=results,production_changed=False,goal_achieved=False)
with Path('summary.json').open('x') as f:json.dump(output,f,indent=2)
print(json.dumps(output,indent=2))
