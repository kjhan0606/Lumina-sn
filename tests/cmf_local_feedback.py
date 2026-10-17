"""고정J 물질행렬에 국소 BB 복사장 미분을 더하는 진단. 생산해가 아니다.
BF/충돌/T/ne/다른 원소는 동결한다. 불투명하며 기존 국소 소스와 J가
가까운 구간만 사용하고, 선택 범위를 바꾸어 근사 의존성을 드러낸다.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import numpy as np
from scipy import sparse

root=Path(sys.argv[1]).resolve()
with_bf='--bf' in sys.argv[2:]
with_nt='--nt' in sys.argv[2:]
with_advection='--advection' in sys.argv[2:]
with_einstein='--einstein' in sys.argv[2:]
if with_advection:assert with_bf and with_nt
if with_einstein:assert with_bf and with_nt
base=root.parent
prior=base/'toy-material-425144'
thermal_root=base/'line-response-20261016.IaZLH0'
line_root=root if with_nt else thermal_root
fine_root=base/'fine-material-20261016.z5VQYD'
def solve(A,b):
    M=np.array(A,dtype=np.longdouble,copy=True)
    v=np.array(b,dtype=np.longdouble,copy=True)
    vector=v.ndim==1
    if vector:v=v[:,None]
    for j in range(len(M)-1):
        p=j+int(abs(M[j:,j]).argmax());assert M[p,j]!=0
        if p!=j:M[[j,p]]=M[[p,j]];v[[j,p]]=v[[p,j]]
        factor=M[j+1:,j]/M[j,j]
        M[j+1:,j+1:]-=factor[:,None]*M[j,j+1:]
        v[j+1:]-=factor[:,None]*v[j];M[j+1:,j]=0
    x=np.empty_like(v)
    for j in range(len(M)-1,-1,-1):
        assert M[j,j]!=0
        x[j]=(v[j]-M[j,j+1:]@x[j+1:])/M[j,j]
    return x[:,0] if vector else x

raw=(prior/'iterate_input.bin').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='244806e4ab10c0b1d821735713cca8c0110e927194210688333bc9dfb58f54c3'
ns,ni,nlev,nelem,nf,gen,iteration=struct.unpack_from('<7Q',raw,72)
t,nu_lo,dlog=struct.unpack_from('<3d',raw,128)
assert ns==113 and iteration==4
maps=np.frombuffer(raw,'<i4',nlev+2*ni,152)
start=152+4*(nlev+2*ni)
ions=np.frombuffer(raw,'<f8',ni*ns,start+8*ns*(5+nelem)).reshape(ni,ns)
pop=np.frombuffer(raw,'<f8',nlev*ns,start+8*ns*(5+nelem+ni)).reshape(nlev,ns)
global_pop=np.empty_like(pop);global_pop[maps[:nlev]]=pop
lookup={(int(maps[nlev+i]),int(maps[nlev+ni+i])):i for i in range(ni)}
atom=list(csv.DictReader((prior/'input/levels.csv').open()))
fine=(fine_root/'fine_shell39.bin').read_bytes()
assert hashlib.sha256(fine).hexdigest()=='830be58c508e4868677bbe395ac0a050844af16844eea50b6106bea6d1761e1e'
assert struct.unpack_from('<Q',fine,8)[0]==nf
nu,chi,es,eta,J=np.frombuffer(fine,'<f8',5*nf,16).reshape(5,nf)
h,c=6.62607015e-27,2.99792458e10
blocks=[];pop0=[];pops=[];stages=[];groups=[];lines=[];totals={};meta={};offset=0
for Z in (14,16):
    buf=(line_root/f'matrix_Z{Z}.bin').read_bytes()
    N=struct.unpack_from('<Q',buf,8)[0];total=struct.unpack_from('<d',buf,16)[0]
    matrix=np.frombuffer(buf,'<f8',N*N,24).reshape(N,N,order='F').copy()
    x=np.frombuffer(buf,'<f8',N,24+8*N*N)/total
    stage=np.full(N,-1);x0=np.zeros(N);within=np.zeros(N)
    for row in csv.DictReader((line_root/f'levels_Z{Z}.csv').open()):
        i=int(row['matrix_index']);gl=int(row['global_level']);stage[i]=int(row['charge'])
        x0[i]+=global_pop[gl,39]/total;within[i]+=float(row['within'])
        meta[gl]=(offset+i,float(row['within']),total)
    assert np.count_nonzero(stage<0)==1 and stage[-1]<0
    stage[-1]=max(stage)+1;x0[-1]=ions[lookup[Z,int(stage[-1])],39]/total
    assert abs(x0.sum()-1)<1e-10 and np.max(abs(within[:-1]-1))<1e-10
    for row in csv.DictReader((line_root/f'lines_Z{Z}.csv').open()):
        r={key:float(value) for key,value in row.items()}
        for key in ('line','lower_global','upper_global'):r[key]=int(r[key])
        r['l']=offset+int(r['lower_matrix']);r['u']=offset+int(r['upper_matrix']);r['total']=total
        lines.append(r)
    blocks.append(matrix);pop0.append(x0);pops.append(x);stages.append(stage)
    groups.append((Z,offset,offset+N));totals[Z]=total;offset+=N
A=sparse.block_diag([sparse.csr_matrix(b) for b in blocks]).toarray().astype(np.longdouble)
x0=np.concatenate(pop0);stationary=np.concatenate(pops);stage=np.concatenate(stages);N=len(x0)
nt_check=None
if with_nt:
    thermal=[]
    for Z in (14,16):
        for kind in ('levels','lines'):
            assert (root/f'{kind}_Z{Z}.csv').read_bytes()==(thermal_root/f'{kind}_Z{Z}.csv').read_bytes()
        buf=(thermal_root/f'matrix_Z{Z}.bin').read_bytes();n=struct.unpack_from('<Q',buf,8)[0]
        thermal.append(sparse.csr_matrix(np.frombuffer(buf,'<f8',n*n,24).reshape(n,n,order='F')))
    old=sparse.block_diag(thermal).toarray().astype(np.longdouble)
    ntmatrix=np.zeros_like(A);counts=np.zeros_like(A,dtype=int)
    terminals={Z:hi-1 for Z,lo,hi in groups};routes=0
    for row in csv.DictReader((root/'nonthermal_rates.csv').open()):
        Z=int(row['Z']);gl=int(row['lower_global']);gu=int(row['upper_global'])
        l,w,_=meta[gl];u=meta[gu][0] if gu>=0 else terminals[Z]
        if l==u:continue
        rate=np.longdouble(row['rate_per_particle'])*np.longdouble(w)
        ntmatrix[u,l]+=rate;ntmatrix[l,l]-=rate;counts[u,l]+=1;counts[l,l]+=1;routes+=1
    discrepancy=A-old-ntmatrix
    # 같은 열적 항에 양의 전이율을 순차 합산할 때의 누적 반올림 상계.
    eps=np.finfo(float).eps;bound=(counts+4)*eps*(abs(A)+abs(old)+abs(ntmatrix))
    assert np.all(abs(discrepancy)<=bound)
    nt_check={'routes_between_superlevels':routes,'matrix_difference_max_abs':float(max(abs(discrepancy).ravel())),
              'max_NT_matrix_entry':float(max(abs(ntmatrix).ravel())),
              'same_J_lines_and_level_mapping':True,'matrix_difference_matches_recorded_NT_with_roundoff_bound':True}
    print('ACTUAL_NT_MATRIX_CHECK',json.dumps(nt_check),flush=True)
pivots=[lo+int(np.argmax(x0[lo:hi])) for Z,lo,hi in groups]
assert len(lines)==16536
by_id={r['line']:r for r in lines}
for i,row in enumerate(csv.DictReader((prior/'input/line_list.csv').open())):
    if i not in by_id:continue
    r=by_id[i];assert float(row['nu'])==r['nu']
    r['kappa']=2.6540281e-2*float(row['f_lu'])*float(row['wavelength_cm'])*r['nu']/c
assert all('kappa' in r for r in lines)
for row in csv.DictReader((fine_root/'lines.csv').open()):
    r=by_id[int(row['line'])];gl,gu=r['lower_global'],r['upper_global']
    ratio=float(atom[gl]['g'])/float(atom[gu]['g'])
    reconstructed=r['kappa']*(global_pop[gl,39]-ratio*global_pop[gu,39])
    assert abs(reconstructed/float(row['integrated_chi'])-1)<1e-10

pr=[];pc=[];pv=[];cr=[];cc=[];cv=[];er=[];ec=[];ev=[]
selected=np.array([r['line'] in (644532,432305,536032,642035,267406,430299,269281,405266) for r in lines])
source_owner=[]
for q,r in enumerate(lines):
    f=r['nu'];width=f*5e6/c
    lo=int(np.searchsorted(nu,f-4*width));hi=int(np.searchsorted(nu,f+4*width,side='right'))
    indices=np.arange(lo,hi);phi=np.exp(-((nu[indices]-f)/width)**2)/(math.sqrt(math.pi)*width)
    weights=phi*nu[indices]*dlog;weights/=weights.sum()
    assert abs(float(weights@J[indices])/r['Jbar']-1)<1e-10
    pr.extend([q]*len(indices));pc.extend(indices);pv.extend(weights)
    gl,gu=r['lower_global'],r['upper_global'];g_ratio=float(atom[gl]['g'])/float(atom[gu]['g'])
    for column,value in ((r['l'],r['kappa']*r['w_lower']*r['total']),
                         (r['u'],-r['kappa']*g_ratio*r['w_upper']*r['total'])):
        cr.extend(indices);cc.extend([column]*len(indices));cv.extend(value*phi)
        source_owner.extend([q]*len(indices))
    er.extend(indices);ec.extend([r['u']]*len(indices))
    ev.extend(h*f/(4*math.pi)*r['A_ul']*r['w_upper']*r['total']*phi)
P=sparse.csr_matrix((pv,(pr,pc)),shape=(len(lines),nf))
Dchi=sparse.coo_matrix((cv,(cr,cc)),shape=(nf,N))
Deta=sparse.coo_matrix((ev,(er,ec)),shape=(nf,N))
coefficient_correction=None
if with_einstein:
    # 동일 인구의 소광만 Einstein B와 일치시킨 조건부 대조.
    # 생산 입력/율계는 변경하지 않으며 공간 수송을 대체하지 않는다.
    coefficient_ratios=np.array([h*r['nu']*r['B_lu']/(4*math.pi*r['kappa']) for r in lines])
    assert np.all(np.isfinite(coefficient_ratios)) and np.all(coefficient_ratios>0)
    correction_values=np.asarray(cv)*(coefficient_ratios[np.asarray(source_owner)]-1)
    coefficient_correction=sparse.coo_matrix((correction_values,(cr,cc)),shape=(nf,N)).tocsr()@x0
lower=np.array([r['l'] for r in lines]);upper=np.array([r['u'] for r in lines])
blu=np.array([r['B_lu']*r['w_lower'] for r in lines]);bul=np.array([r['B_ul']*r['w_upper'] for r in lines])
inc=sparse.csr_matrix((np.r_[-np.ones(len(lines)),np.ones(len(lines))],
    (np.r_[lower,upper],np.r_[np.arange(len(lines)),np.arange(len(lines))])),shape=(N,len(lines)))
flow=blu*x0[lower]-bul*x0[upper]
R=inc@sparse.diags(flow)@P
if with_bf:
    bfraw=(prior/'input/bf_channels.bin').read_bytes()
    assert hashlib.sha256(bfraw).hexdigest()=='656329de676b540a08beb1e6652e839785f799428a40b31fd2774ce44d8af0cf'
    magic,version,nl,nr,nb,blo,bhi=struct.unpack_from('<IIiiidd',bfraw)
    assert magic==0x43424643 and version==2 and nl==len(atom)
    pos=36;offsets=np.frombuffer(bfraw,'<i4',nl+1,pos);pos+=4*(nl+1)
    up=np.frombuffer(bfraw,'<i4',nr,pos);pos+=4*nr
    threshold=np.frombuffer(bfraw,'<f8',nr,pos);pos+=8*nr
    sigma=np.frombuffer(bfraw,'<f8',nr*nb,pos).reshape(nr,nb);pos+=8*nr*nb
    upper_g=np.frombuffer(bfraw,'<f8',nr,pos)
    low=np.repeat(np.arange(nl),np.diff(offsets))
    edges=blo*np.exp(np.arange(nb+1)*math.log(bhi/blo)/nb);edges[-1]=bhi
    centers=np.sqrt(edges[:-1]*edges[1:])
    fe=nu_lo*np.exp(np.arange(nf+1)*dlog)
    union=np.unique(np.r_[edges,fe[(fe>blo)&(fe<bhi)]])
    left,right=union[:-1],union[1:];mid=.5*(left+right)
    cb=np.searchsorted(edges,mid,side='right')-1;fb=np.searchsorted(fe,mid,side='right')-1
    W=sparse.csr_matrix(((right-left)/np.diff(edges)[cb],(cb,fb)),shape=(nb,nf))
    assert max(abs(W@np.ones(nf)-1))<1e-12
    lc=np.log(centers);coordinate=np.log(nu)
    bindex=np.searchsorted(lc,coordinate)-1;bindex=np.clip(bindex,0,nb-2)
    fraction=np.clip((coordinate-lc[bindex])/(lc[bindex+1]-lc[bindex]),0,1)
    # 위 경계 연장은 생산 log-nu 보간의 동일 표현이며 물리값 clamp가 아니다.
    I=sparse.csr_matrix((np.r_[1-fraction,fraction],(np.r_[np.arange(nf),np.arange(nf)],np.r_[bindex,bindex+1])),shape=(nf,nb))
    jcoarse=W@J;Dcb=np.zeros((nb,N));Deb=np.zeros((nb,N));Rbf=np.zeros((N,nb));Aradbf=np.zeros((N,N))
    T=np.frombuffer(raw,'<f8',ns,start+8*ns*3)[39]
    ne=np.frombuffer(raw,'<f8',ns,start+8*ns*(4+nelem))[39]
    gx,gw=np.polynomial.legendre.leggauss(8);gw*=.5;k=1.380649e-16;me=9.1093837015e-28
    terminal={Z:hi-1 for Z,lo,hi in groups}
    channels=0
    for r in range(nr):
        gl=int(low[r])
        if gl not in meta:continue
        Z=int(atom[gl]['atomic_number']);l,wl,total=meta[gl]
        if up[r]>=0:u,wu,_=meta[int(up[r])];gu=float(atom[int(up[r])]['g'])
        else:u=terminal[Z];wu=1.;gu=upper_g[r]
        bins=np.flatnonzero(sigma[r]>0)
        if not len(bins):continue
        channels+=1;sg=sigma[r,bins]
        left0=np.maximum(edges[bins],threshold[r]);right0=edges[bins+1]
        assert np.all(right0>left0)
        fq=.5*(left0[:,None]+right0[:,None])+.5*(right0-left0)[:,None]*gx
        inv=ne*(h*h/(2*math.pi*me*k*T))**1.5*float(atom[gl]['g'])/(2*gu)*np.exp(-h*(fq-threshold[r])/(k*T))
        spont=2*h*fq**3/c**2
        Dcb[bins,l]+=sg*wl*total
        Dcb[bins,u]-=sg*np.sum(inv*gw,axis=1)*wu*total
        Deb[bins,u]+=sg*np.sum(inv*spont*gw,axis=1)*wu*total
        photo=4*math.pi*sg*np.diff(edges)[bins]*np.sum(gw/(h*fq),axis=1)*wl
        stim=4*math.pi*sg*np.diff(edges)[bins]*np.sum(gw*inv/(h*fq),axis=1)*wu
        net=photo*x0[l]-stim*x0[u]
        Rbf[u,bins]+=net;Rbf[l,bins]-=net
        pu=photo@jcoarse[bins];pd=stim@jcoarse[bins]
        Aradbf[u,l]+=pu;Aradbf[l,l]-=pu;Aradbf[l,u]+=pd;Aradbf[u,u]-=pd
    # 고정 ne/T에서도 FF의 이온 전하제곱 합은 개체수에 따라 달라진다.
    ff=3.6926e8/math.sqrt(T)*ne/centers**3*(-np.expm1(-h*centers/(k*T)))
    y=h*centers/(k*T)
    bplanck=2*h*centers**3/c**2*np.exp(-y)/(-np.expm1(-y))
    charge_weight=np.zeros(N)
    for Z,lo,hi in groups:charge_weight[lo:hi]=stage[lo:hi]**2*totals[Z]
    Dcb+=ff[:,None]*charge_weight;Deb+=(ff*bplanck)[:,None]*charge_weight
    print('BF_CHANNELS',channels,flush=True)
dr=864000000000.;chi_abs=chi-es
advection_terms={}
if with_advection:
    # 정상 Euler 수송에 남는 (v/c)*dJ/dr. 기존 미수렴 저장장의 할선이며
    # 새로운 공간 수송을 대신하지 않는다. 선택된 국소 불투명 구간에만 적용한다.
    ri=np.frombuffer(raw,'<f8',ns,start)
    ro=np.frombuffer(raw,'<f8',ns,start+8*ns)
    radius=(ri+ro)/2
    stored_J=np.frombuffer(raw,'<f8',ns*nf,start+8*ns*(5+nelem+ni+nlev)).reshape(ns,nf)
    assert np.array_equal(stored_J[39],J)
    beta=radius[39]/(t*c)
    advection_terms={
        'left':beta*(stored_J[39]-stored_J[38])/(radius[39]-radius[38]),
        'center':beta*(stored_J[40]-stored_J[38])/(radius[40]-radius[38]),
        'right':beta*(stored_J[40]-stored_J[39])/(radius[40]-radius[39])}
    assert all(np.all(np.isfinite(v)) for v in advection_terms.values())
local_depth=chi_abs*min(dr,c*t*dlog)
local_error=abs((eta+es*J)/(chi*J)-1)
def constrained(mat,rhs):
    m=mat.copy();b=np.array(rhs,dtype=np.longdouble,copy=True)
    for pivot,(Z,lo,hi) in zip(pivots,groups):
        m[pivot,:]=0;m[pivot,lo:hi]=1;b[pivot]=0
    scales=np.max(abs(m),axis=1)
    result=solve(m/scales[:,None],b/scales if b.ndim==1 else b/scales[:,None])
    residual=np.max(abs(m@result-b)/(scales if b.ndim==1 else scales[:,None]))
    return result,float(residual)
f0=A@x0
d0,r0=constrained(A,-f0)
assert max(abs(d0-(stationary-x0)))<1e-8
def ratios(x):
    return {str(Z):float(x[lo:hi][stage[lo:hi]==2].sum()/x[lo:hi][stage[lo:hi]==1].sum()) for Z,lo,hi in groups}
results={'scope':'local BB-only Jacobian approximation, BF/T/ne/collisions/other elements frozen, no RTE',
 'baseline_ratio':ratios(x0),'fixed_J_ratio':ratios(x0+d0),'fixed_J_residual':r0,'cases':[]}
results['nonthermal']='actual_recorded_deposition_rates_frozen_at_native_candidate' if with_nt else 'zero_control'
if with_nt:results['nonthermal_matrix_check']=nt_check
if with_bf:
    results['scope']='local BB/BF/FF source and BB/BF rate Jacobian, fixed T/ne/other elements, no spatial RTE'
    expected=json.loads((base/'response-components-20261016.EeuAzT/summary.json').read_text())
    trial_ratios=[]
    for amplitude in (-.01,.01):
        At=A+amplitude*Aradbf
        dx,res=constrained(At,-At@x0);trial_ratios.append(ratios(x0+dx))
    bfcheck={str(Z):math.log(trial_ratios[1][str(Z)]/trial_ratios[0][str(Z)])/math.log(1.01/.99) for Z in (14,16)}
    if not with_nt:
        assert max(abs(bfcheck[str(Z)]-expected['dln_ratio_dln_J_by_consumer']['BF'][str(Z)]) for Z in (14,16))<1e-6
    results['BF_finite_difference_with_current_matrix' if with_nt else 'BF_full_C_finite_difference_reproduced']=bfcheck
    print('BF_RESPONSE_CURRENT_MATRIX' if with_nt else 'BF_RESPONSE_REPRODUCED',bfcheck,flush=True)
cases=[('all',1e4,1e-5),('all',1e2,1e-3)] if with_bf else [('eight',1e4,1e-5),('all',1e4,1e-5),('all',1e2,1e-3)]
for source_set,depth,error in cases:
    mask=(local_depth>depth)&(local_error<error)&np.isfinite(local_error)
    if source_set=='eight':
        choose=selected[np.array(source_owner)]
        dc=sparse.csr_matrix((np.asarray(cv)[choose],(np.asarray(cr)[choose],np.asarray(cc)[choose])),shape=(nf,N))
        choose_eta=selected[np.repeat(np.arange(len(lines)),np.diff(P.indptr))]
        de=sparse.csr_matrix((np.asarray(ev)[choose_eta],(np.asarray(er)[choose_eta],np.asarray(ec)[choose_eta])),shape=(nf,N))
    else:dc=Dchi.tocsr();de=Deta.tocsr()
    inverse=np.zeros(nf);inverse[mask]=1/chi_abs[mask]
    D=sparse.diags(inverse)@(de-sparse.diags(J)@dc)
    C=(R@D).toarray().astype(np.longdouble)
    if with_bf:
        L0=sparse.diags(inverse);LJ=sparse.diags(inverse*J)
        response0=np.asarray((R@L0@I).toarray())+Rbf@(W@L0@I).toarray()
        responseJ=np.asarray((R@LJ@I).toarray())+Rbf@(W@LJ@I).toarray()
        C+=Rbf@(W@D).toarray()+response0@Deb-responseJ@Dcb
    direction=x0*np.random.default_rng(20261016).normal(size=N)
    for Z,lo,hi in groups:direction[lo:hi]-=x0[lo:hi]*sum(direction[lo:hi])
    step=1e-4
    shifts=[]
    for sign in (-1,1):
        dx=sign*step*direction;x=x0+dx
        chi_change=dc@dx;eta_change=de@dx
        if with_bf:chi_change+=I@(Dcb@dx);eta_change+=I@(Deb@dx)
        denominator=chi_abs+chi_change
        assert np.all(denominator[mask]>0)
        dJ=np.zeros(nf);dJ[mask]=(eta_change[mask]-J[mask]*chi_change[mask])/denominator[mask]
        shift=inc@((blu*x[lower]-bul*x[upper])*(P@dJ))
        if with_bf:shift+=Rbf@(W@dJ)
        shifts.append(shift)
    finite=(shifts[1]-shifts[0])/(2*step);analytic=C@direction
    derivative_error=float(max(abs(finite-analytic))/max(abs(analytic)))
    assert derivative_error<1e-6
    delta,residual=constrained(A+C,-f0)
    feedback,feedback_residual=constrained(A,-C)
    eigen,vectors=np.linalg.eig(np.asarray(feedback,dtype=float))
    order=np.argsort(abs(eigen))[::-1]
    positive=bool(np.all(x0+delta>=0))
    item={'sources':source_set,'minimum_local_depth':depth,'maximum_baseline_local_error':error,
      'active_bins':int(sum(mask)),'differential_check_relative_error':derivative_error,
      'linear_step_nonnegative':positive,'minimum_linear_population':float(min(x0+delta)),
      'linear_step_ratio':ratios(x0+delta) if positive else None,
      'linear_residual':residual,'feedback_residual':feedback_residual,
      'feedback_spectral_radius':float(max(abs(eigen))),
      'dominant_feedback_eigenvalues':[[float(eigen[i].real),float(eigen[i].imag)] for i in order[:5]],
      'dominant_mode_stage_absolute_weights':{},
      'dln_ratio_linear_step':{},'fixed_J_dln_ratio_linear_step':{}}
    for Z,lo,hi in groups:
        mode=abs(vectors[:,order[0]]);mode/=mode.sum()
        item['dominant_mode_stage_absolute_weights'][str(Z)]={str(q):float(mode[lo:hi][stage[lo:hi]==q].sum()) for q in sorted(set(stage[lo:hi]))}
        ii=stage[lo:hi]==1;iii=stage[lo:hi]==2
        item['dln_ratio_linear_step'][str(Z)]=float(delta[lo:hi][iii].sum()/x0[lo:hi][iii].sum()-delta[lo:hi][ii].sum()/x0[lo:hi][ii].sum())
        item['fixed_J_dln_ratio_linear_step'][str(Z)]=float(d0[lo:hi][iii].sum()/x0[lo:hi][iii].sum()-d0[lo:hi][ii].sum()/x0[lo:hi][ii].sum())
    if with_advection:
        item['advection_local_forcing']={}
        for stencil,term in advection_terms.items():
            # L_old(delta J)=-v/c*dJ/dr. 국소 응답에만 -term/(chi-es)를 쓴다.
            forcing_J=-inverse*term
            forcing=R@forcing_J+Rbf@(W@forcing_J)
            trial,trial_residual=constrained(A+C,-f0-forcing)
            extra=trial-delta
            event={'linear_residual':trial_residual,
                   'linear_step_nonnegative':bool(np.all(x0+trial>=0)),
                   'minimum_linear_population':float(min(x0+trial)),
                   'incremental_dln_ratio':{},'forced_bins':int(sum(mask)),
                   'maximum_forcing_J_over_J':float(max(abs(forcing_J[mask]/J[mask])))}
            for Z,lo,hi in groups:
                assert abs(extra[lo:hi].sum())<1e-10
                ii=stage[lo:hi]==1;iii=stage[lo:hi]==2
                event['incremental_dln_ratio'][str(Z)]=float(
                    extra[lo:hi][iii].sum()/x0[lo:hi][iii].sum()-
                    extra[lo:hi][ii].sum()/x0[lo:hi][ii].sum())
            item['advection_local_forcing'][stencil]=event
    if with_einstein:
        # 원래 국소 선형 연산자에 계수 변화의 강제항만 추가한다.
        # 고정물질 국소식의 정확 차분도 함께 표시하여 1차 근사 크기를 점검.
        forcing_J=-inverse*J*coefficient_correction
        exact_J=np.zeros(nf)
        corrected_denominator=chi_abs+coefficient_correction
        assert np.all(corrected_denominator[mask]>0)
        exact_J[mask]=-J[mask]*coefficient_correction[mask]/corrected_denominator[mask]
        forcing=R@forcing_J+Rbf@(W@forcing_J)
        trial,trial_residual=constrained(A+C,-f0-forcing)
        fixed,fixed_residual=constrained(A,-forcing)
        extra=trial-delta
        event={'scope':'Si/S BB opacity consistency; first-order local coupled response; no spatial RTE',
          'coefficient_ratio_range':[float(min(coefficient_ratios)),float(max(coefficient_ratios))],
          'linear_residual':trial_residual,'fixed_J_response_residual':fixed_residual,
          'linear_step_nonnegative':bool(np.all(x0+trial>=0)),
          'minimum_linear_population':float(min(x0+trial)),
          'maximum_forcing_J_over_J':float(max(abs(forcing_J[mask]/J[mask]))),
          'local_exact_vs_linear_relative_difference':float(max(abs(exact_J-forcing_J))/max(abs(forcing_J))),
          'incremental_dln_ratio':{},'without_feedback_dln_ratio':{}}
        for Z,lo,hi in groups:
            assert abs(extra[lo:hi].sum())<1e-10
            ii=stage[lo:hi]==1;iii=stage[lo:hi]==2
            event['incremental_dln_ratio'][str(Z)]=float(
                extra[lo:hi][iii].sum()/x0[lo:hi][iii].sum()-extra[lo:hi][ii].sum()/x0[lo:hi][ii].sum())
            event['without_feedback_dln_ratio'][str(Z)]=float(
                fixed[lo:hi][iii].sum()/x0[lo:hi][iii].sum()-fixed[lo:hi][ii].sum()/x0[lo:hi][ii].sum())
        item['einstein_coefficient_local_forcing']=event
    results['cases'].append(item)
    print(json.dumps(item),flush=True)
if with_advection:
    results['advection_scope']='steady Eulerian v/c*dJ/dr, old J secants; local thick mask only; no spatial solve, no new equilibrium'
with (root/('summary_bf.json' if with_bf else 'summary.json')).open('x') as out:json.dump(results,out,indent=2)
