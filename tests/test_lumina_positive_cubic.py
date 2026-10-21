"""B의 양의 cubic 지수 원천 적분을 독립 구적/고정밀 극한과 대조한다."""
import ctypes as ct
from decimal import Decimal,localcontext
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def cubic(tmp_path_factory):
    path=tmp_path_factory.mktemp('positive-cubic')/'cubic.so'
    proc=subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off','-Wall','-Wextra','-Werror',
        '-fPIC','-shared','-I'+str(ROOT/'src'),str(ROOT/'tests/positive_cubic_probe.cpp'),'-o',str(path)],
        capture_output=True,text=True,timeout=60)
    assert proc.returncode==0,proc.stderr
    f=ct.CDLL(str(path)).lumina_positive_cubic_probe
    p=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    q=np.ctypeslib.ndpointer(dtype='i8',flags='C_CONTIGUOUS');i=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    f.argtypes=[ct.c_int,p,p,p,p,ct.c_double,p,q,i];f.restype=ct.c_int
    return f


def run(fun,length,source,phase,logs,relative=1e-10):
    length,source,phase,logs=[np.ascontiguousarray(v,dtype='f8') for v in (length,source,phase,logs)]
    n=len(length);m=np.full((n,2),-17.);e=np.full((n,2),-17,dtype='i8');d=np.full(n,-17,dtype='i4')
    code=fun(n,length,source,phase,logs,relative,m,e,d)
    return code,m,e,d


def test_signed_cubic_phase_and_positive_source_match_independent_gauss128(cubic):
    rng=np.random.default_rng(1022);n=160;phase=rng.uniform(-.24,.24,(n,4));source=rng.uniform(0.,2.,(n,3))
    source[0]=0.;length=rng.uniform(.1,3.,n);logs=rng.uniform(-10.,10.,n)
    code,m,e,d=run(cubic,length,source,phase,logs);assert code==0
    x,w=np.polynomial.legendre.leggauss(128);t=.5*(x+1);weight=.5*w
    for k in range(n):
        eta=(1-t)**2*source[k,0]+2*t*(1-t)*source[k,1]+t*t*source[k,2]
        phi=(1-t)**3*phase[k,0]+3*t*(1-t)**2*phase[k,1]+3*t*t*(1-t)*phase[k,2]+t**3*phase[k,3]
        expected=length[k]*np.sum(weight*eta*np.exp(logs[k]+phi))
        value=np.ldexp(m[k,0],int(e[k,0]));error=np.ldexp(m[k,1],int(e[k,1]))
        assert value>=0 and error>=0
        assert abs(value-expected)<=error+8e-15*abs(expected)
        assert error<=1.01e-10*value or value==error==0.
        assert 0<=d[k]<=24


@pytest.mark.parametrize('length,source,log_scale',[(1e-200,1e-200,500.),(1e300,1e300,-1000.),
    (1e-300,1e-300,0.),(1e300,float(np.nextafter(0.,1.)),500.),(1.,1.,1000.)])
def test_scaled_integral_keeps_underflow_and_overflow_until_final_conversion(cubic,length,source,log_scale):
    code,m,e,d=run(cubic,[length],[[source]*3],[[0.]*4],[log_scale]);assert code==0
    with localcontext() as context:
        context.prec=100;D=Decimal.from_float
        actual=D(m[0,0])*(Decimal(2)**int(e[0,0]));error=D(m[0,1])*(Decimal(2)**int(e[0,1]))
        expected=D(length)*D(source)*D(log_scale).exp()
        assert actual>0 and abs(actual-expected)<=error
        assert abs(actual/expected-1)<Decimal('3e-13')


@pytest.mark.parametrize('bad',['source','phase','length','range','budget','log'])
def test_unsupported_or_invalid_inputs_do_not_publish_partial_results(cubic,bad):
    length=[1.,1.];source=np.ones((2,3));phase=np.zeros((2,4));logs=[0.,0.];relative=1e-10
    if bad=='source':source[1,2]=-1.
    elif bad=='phase':phase[1,0]=np.nan
    elif bad=='length':length[1]=-1.
    elif bad=='range':phase[1,2]=1.
    elif bad=='budget':relative=1e-16;phase[1]=[-.25,.25,-.25,.25]
    else:logs[1]=np.inf
    code,m,e,d=run(cubic,length,source,phase,logs,relative)
    assert code!=0
    np.testing.assert_array_equal(m,np.full((2,2),-17.));np.testing.assert_array_equal(e,np.full((2,2),-17));np.testing.assert_array_equal(d,[-17,-17])
