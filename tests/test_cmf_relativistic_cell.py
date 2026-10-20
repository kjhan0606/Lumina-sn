"""작은 full-SR 셀 formal을 독립 lab 적분/방향별 각도 oracle과 대조한다."""
import ctypes as ct
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
PTR=ct.POINTER(ct.c_double)


class Boundary(ct.Structure):
    _fields_=[('kind',ct.c_int),('temperature',ct.c_double),('scale',ct.c_double),
              ('source',PTR),('ratio',PTR),('gradient',PTR)]


@pytest.fixture(scope='module')
def core(tmp_path_factory):
    path=tmp_path_factory.mktemp('rel-cell')/'core.so'
    subprocess.run([os.environ.get('CC','cc'),'-O2','-Wall','-Wextra','-Werror',
        '-std=c11','-fPIC','-shared','-fopenmp','-I'+str(ROOT/'src'),
        *[str(ROOT/'src'/name) for name in ('cmf_exact_sliding.c',
          'cmf_relativistic_characteristic.c','cmf_error_envelope.c')],
        '-lm','-o',str(path)],check=True,timeout=60,capture_output=True)
    lib=ct.CDLL(str(path));f=lib.cmf_exact_cell_solve_relativistic_raw_sources
    f.argtypes=[ct.c_int,ct.c_int,ct.c_double,PTR,PTR,PTR,ct.c_double,
        ct.POINTER(Boundary),PTR,PTR,PTR,PTR,PTR,ct.c_size_t,
        ct.c_int,ct.c_double,ct.c_int,ct.c_void_p]
    f.restype=ct.c_int
    return f


def sample(static=False,scatter=False):
    ri=np.array([2e14,3e14,4e14]);ro=np.array([3e14,4e14,6e14])
    nf=24;dlog=.01;nu=1e14*np.exp(dlog*np.arange(nf))
    chi=np.tile(np.linspace(2e-15,8e-16,nf),(3,1));chi[1,8:11]=-3e-16
    es=np.full_like(chi,2e-16 if scatter else 0.)
    eta=np.tile(np.linspace(1e-22,2e-22,nf),(3,1))
    return dict(ri=ri,ro=ro,nu=nu,dlog=dlog,chi=chi,es=es,eta=eta,
        epoch=1e30 if static else ro[-1]/(.12*2.99792458e10),
        temperature=14000.,gradient=np.linspace(2e-8,4e-8,nf))


def solve(core,p,mode=3,envelope=True):
    def ptr(a):return a.ctypes.data_as(PTR)
    ns,nf=p['chi'].shape;j=np.zeros((ns,nf));error=np.zeros_like(j)
    bc=Boundary(3,p['temperature'],0.,None,None,ptr(p['gradient']))
    status=core(ns,nf,p['dlog'],ptr(p['nu']),ptr(p['ri']),ptr(p['ro']),p['epoch'],
        ct.byref(bc),ptr(p['chi']),ptr(p['es']),ptr(p['eta']),ptr(j),
        ptr(error) if envelope else None,3,200,1e-12,mode,None)
    assert status==0
    assert np.isfinite(j).all() and np.isfinite(error).all() and (error>=0).all()
    return j,error


NODES,WEIGHTS=np.polynomial.legendre.leggauss(48)


def geometry(impact,z0,z1,epoch):
    # C의 해석 적분을 호출하거나 복제하지 않고 D를 직접 Gauss 적분한다.
    A=2.99792458e10*epoch;z=.5*(z0+z1)+.5*(z1-z0)*NODES
    D=(1-z/A)/np.sqrt(1-(impact*impact+z*z)/(A*A))
    length=.5*(z1-z0)*np.dot(WEIGHTS,D)
    long=np.longdouble;AA=long(2.99792458e10)*long(epoch)
    def logd(zz):
        zz=long(zz);bb=(long(impact)**2+zz*zz)/(AA*AA)
        return -.5*np.log1p(-bb)+np.log1p(-zz/AA)
    return length,float(logd(z0)-logd(z1))


def propagate(up,chi,eta,length,drift,dlog):
    if length==0:return up.copy()
    shift=drift/dlog;b=drift/length;kappa=chi+3*b
    result=np.empty_like(up);nf=len(up)
    # 각 주파수 경계를 명시적으로 순회하는 O(shift) 독립 formal.
    for f in range(nf):
        x=f+shift;intensity=np.interp(x,np.arange(nf),up)
        cuts=[x]+[m+.5 for m in range(int(np.floor(x-.5)),-1,-1) if f<m+.5<x]+[float(f)]
        for high,low in zip(cuts[:-1],cuts[1:]):
            fraction=(high-low)/shift if shift else 1.
            ds=length*fraction
            index=min(nf-1,max(0,int(np.floor(.5*(high+low)+.5))))
            tau=kappa[index]*ds;t=np.exp(-tau)
            e=ds if tau==0 else ds*(-np.expm1(-tau)/tau)
            intensity=t*intensity+e*eta[index]
        result[f]=intensity
    return result


def formal(p,j,wrong_angles=False):
    ns,nf=j.shape;ri,ro=p['ri'],p['ro'];mid=.5*(ri+ro);A=2.99792458e10*p['epoch']
    rays=np.r_[ri[0]*np.arange(16)/16,ri[0],mid]
    samples=[[] for _ in range(ns)];source=p['eta']+p['es']*j
    B=2*6.62607015e-27*p['nu']**3/2.99792458e10**2/np.expm1(
        6.62607015e-27*p['nu']/(1.380649e-16*p['temperature']))
    for impact in rays:
        cells=[s for s in range(ns-1,-1,-1) if ro[s]>impact]
        edge=np.zeros(nf);incoming={};outgoing={}
        for direction in range(2):
            if direction and impact<ri[0]:
                w=np.sqrt((ri[0]-impact)*(ri[0]+impact))/ri[0];beta=ri[0]/A
                edge=B+(w-beta)/(1-beta*w)*p['gradient']
            for s in (reversed(cells) if direction else cells):
                zhi=np.sqrt((ro[s]-impact)*(ro[s]+impact))
                zm=np.sqrt((mid[s]-impact)*(mid[s]+impact)) if mid[s]>impact else 0.
                zlo=np.sqrt((ri[s]-impact)*(ri[s]+impact)) if ri[s]>impact else 0.
                z=(zlo,zm,zhi) if direction else (-zhi,-zm,-zlo)
                for half in range(2):
                    length,drift=geometry(impact,z[half],z[half+1],p['epoch'])
                    if drift/p['dlog']<1e-12:
                        # shift를 f에 더할 수 없는 정적 극한은 독립적인 직접 적분.
                        tau=p['chi'][s]*length;t=np.exp(-tau)
                        e=np.full(nf,length)
                        np.divide(-np.expm1(-tau),p['chi'][s],out=e,where=tau!=0)
                        edge=t*edge+e*source[s]
                    else:edge=propagate(edge,p['chi'][s],source[s],length,drift,p['dlog'])
                    if half==0:(outgoing if direction else incoming)[s]=edge.copy()
        for s in cells:
            if impact<=mid[s]:
                u=np.sqrt((mid[s]-impact)*(mid[s]+impact))/mid[s];beta=mid[s]/A
                plus=(u-beta)/(1-beta*u);minus=(-u-beta)/(1+beta*u)
                if wrong_angles:plus,minus=u,-u
                samples[s].append((plus,minus,outgoing[s],incoming[s]))
    result=np.zeros_like(j)
    for s,values in enumerate(samples):
        values.sort(key=lambda v:v[0])
        for a,b in zip(values[:-1],values[1:]):
            result[s]+=.25*(b[0]-a[0])*(a[2]+b[2])+.25*(a[1]-b[1])*(a[3]+b[3])
    return result


@pytest.mark.parametrize('static',[False,True])
def test_full_sr_actual_cells_match_independent_lab_formal(core,static):
    p=sample(static=static);j,error=solve(core,p)
    expected=formal(p,np.zeros_like(j))
    np.testing.assert_allclose(j,expected,rtol=2e-12,atol=1e-27)
    direct,_=solve(core,p,mode=2,envelope=False)
    np.testing.assert_allclose(j,direct,rtol=2e-14,atol=0)
    if not static:
        wrong=formal(p,np.zeros_like(j),wrong_angles=True)
        assert np.max(np.abs(wrong-j)/j)>.005


def test_scattering_fixed_point_and_error_use_same_full_sr_formal(core):
    p=sample(scatter=True);j,error=solve(core,p)
    expected=np.zeros_like(j)
    for _ in range(100):
        new=formal(p,expected)
        if np.max(np.abs(new-expected)/np.maximum(new,1e-300))<1e-13:break
        expected=new
    else:raise AssertionError('독립 산란 반복 미수렴')
    np.testing.assert_allclose(j,new,rtol=3e-12,atol=1e-27)
    assert np.max(error)>0
