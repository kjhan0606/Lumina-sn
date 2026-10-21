"""공간선형 B의 중심화 모멘트와 보이지 않는 좁은 원천의 오차 상계를 대조한다."""
import ctypes as ct
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def moments(tmp_path_factory):
    path=tmp_path_factory.mktemp('linear-moments')/'moments.so'
    proc=subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off','-Wall','-Wextra','-Werror',
        '-fPIC','-shared','-I'+str(ROOT/'src'),str(ROOT/'tests/linear_moments_probe.cpp'),'-o',str(path)],
        capture_output=True,text=True,timeout=60)
    assert proc.returncode==0,proc.stderr
    fun=ct.CDLL(str(path)).lumina_linear_moments_probe
    p=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    fun.argtypes=[ct.c_int,ct.c_int,p,ct.c_int,ip,p,p,p];fun.restype=ct.c_int
    return fun


def run(fun,values,intervals,paths,block=32):
    values=np.ascontiguousarray(values,dtype='f8');intervals=np.array(intervals,dtype='f8')
    bins=np.ascontiguousarray(np.floor(intervals),dtype='i4');fractions=np.ascontiguousarray(intervals-bins,dtype='f8')
    paths=np.ascontiguousarray(paths,dtype='f8');output=np.full((len(intervals),34),-17.)
    code=fun(values.shape[1],block,values,len(intervals),bins,fractions,paths,output)
    return code,output


def reference(values,begin,end,length,u0,u1):
    nf=values.shape[1];edges=np.r_[begin,np.arange(np.floor(begin)+1,np.ceil(end)),end]
    x,w=np.polynomial.legendre.leggauss(8);p=np.zeros(4);q=np.zeros(4)
    for a,b in zip(edges[:-1],edges[1:]):
        if b<=a:continue
        at=.5*((b-a)*x+a+b);weight=.5*(b-a)*w
        sample=np.array([np.interp(at,np.arange(nf),row) for row in values]);t=(at-begin)/(end-begin)
        p+=(sample*(weight*(1-t))).sum(axis=1)/(end-begin)
        q+=(sample*(weight*t)).sum(axis=1)/(end-begin)
    result=[]
    for k in (0,2):
        result.append(length*(u1-u0)*(u0*p[k]+u1*q[k]+(1-u0)*p[k+1]+(1-u1)*q[k+1]))
    return p,q,result


@pytest.mark.parametrize('nf',[2,7,33,257])
@pytest.mark.parametrize('block',[1,4,32])
def test_partial_bins_non_power_of_two_and_blue_extension_match_independent_gauss(moments,nf,block):
    rng=np.random.default_rng(1022+nf);values=rng.uniform(-.8,.8,(4,nf));values[2:]=abs(values[2:])
    intervals=[(.125,.875),(0.,nf-1.),(.375,nf+2.75),(nf-1.,nf+1.25),(nf+.125,nf+4.)]
    if nf>7:intervals.append((3.25,nf-2.375))
    paths=[(2.,.13,.91)]*len(intervals);code,out=run(moments,values,intervals,paths,block);assert code==0
    for row,(a,b),path in zip(out,intervals,paths):
        p,q,(tau,mass)=reference(values,a,b,*path)
        assert abs(row[0]-tau)<=row[1]+3e-15
        assert abs(row[2]-mass)<=row[3]+3e-15
        assert row[4]==pytest.approx(b-a,rel=2e-15)
        for k in range(4):
            np.testing.assert_allclose(row[6+7*k:8+7*k],[p[k],q[k]],rtol=2e-12,atol=2e-15)


def test_spatial_frequency_cross_term_is_integrated_not_frozen_at_midpoint(moments):
    x=np.arange(33);values=np.array([.3+.01*x,.2+.001*x,.04+.02*x,.13+.003*x])
    start=.375;span=12.25;L=2.;code,out=run(moments,values,[(start,start+span)],[(L,0.,1.)]);assert code==0
    c0=.2+.001*start;c1=.3+.01*start-c0+.001*span;c2=(.01-.001)*span
    exact=L*(c0+c1/2+c2/3)
    frozen=L*(.5*(.3+.2)+.5*(.01+.001)*(start+span/2))
    assert out[0,0]==pytest.approx(exact,rel=2e-14)
    assert abs(out[0,0]-frozen)==pytest.approx(L*(.01-.001)*span/12,rel=2e-13)


def test_narrow_eta_mass_and_slope_bound_survive_zero_endpoints_and_missed_gauss_nodes(moments):
    nf=513;values=np.zeros((4,nf));values[2:,256]=1.
    code,out=run(moments,values,[(0.,512.)],[(1.,0.,1.)]);assert code==0
    assert out[0,2]==pytest.approx(1/512,rel=1e-14)
    nodes,_=np.polynomial.legendre.leggauss(16)
    assert np.all(np.interp(256*(nodes+1),np.arange(nf),values[2])==0.)
    assert values[2,0]==values[2,-1]==0.
    assert out[0,6+7*2+3]==1. and out[0,6+7*2+6]>=1.


def test_source_and_chi_keep_exact_zeros_and_signed_integrals(moments):
    values=np.zeros((4,9));values[0]=-2.;values[1]=-4.
    code,out=run(moments,values,[(1.25,7.75)],[(3.,0.,1.)]);assert code==0
    assert out[0,0]==pytest.approx(-9.) and out[0,2]==out[0,3]==0.


def test_normalized_moments_avoid_huge_width_position_products(moments):
    values=np.full((4,1025),1e290)
    code,out=run(moments,values,[(0.,1024.)],[(1e-290,.1,.9)]);assert code==0
    assert out[0,0]==pytest.approx(.8,rel=2e-14) and out[0,2]==pytest.approx(.8,rel=2e-14)


def test_subnormal_rounding_uncertainty_is_not_claimed_as_zero(moments):
    values=np.full((4,5),np.nextafter(0.,1.))
    code,out=run(moments,values,[(0.,4.)],[(1e300,0.,1.)]);assert code==0
    assert out[0,3]>0 and out[0,1]>0


@pytest.mark.parametrize('bad',['eta','chi','length','fraction','range'])
def test_invalid_input_does_not_publish_partial_queries(moments,bad):
    values=np.ones((4,9));intervals=[(0.,1.),(2.,7.)];paths=[(1.,0.,1.)]*2
    if bad=='eta':values[3,5]=-1.
    elif bad=='chi':values[0,3]=np.nan
    elif bad=='length':paths[1]=(-1.,0.,1.)
    elif bad=='fraction':intervals[1]=(np.nan,7.)
    else:intervals[1]=(3.,2.)
    # NaN은 Python의 좌표 변환 자체에서 표현 불가 정수로 바뀌어 C가 거부한다.
    with np.errstate(invalid='ignore'):code,out=run(moments,values,intervals,paths)
    assert code!=0;np.testing.assert_array_equal(out,np.full((2,34),-17.))
