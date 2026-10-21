"""GPU 선택 경로의 순서/스케일 합성을 CPU의 독립 고정밀 산술로 대조한다."""
import ctypes as ct
from decimal import Decimal, localcontext
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def ordered(tmp_path_factory):
    library=tmp_path_factory.mktemp('ordered')/'ordered.so'
    subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off',
        '-Wall','-Wextra','-Werror','-fPIC','-shared','-I'+str(ROOT/'src'),
        str(ROOT/'tests/ordered_maps_probe.cpp'),'-o',str(library)],
        check=True,capture_output=True,timeout=60)
    fun=ct.CDLL(str(library)).lumina_ordered_probe
    p=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    ip=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    fun.argtypes=[ct.c_int,p,p,ct.c_int,ip,ip,p,p];fun.restype=ct.c_int
    return fun


def reference(values,logs,incident):
    with localcontext() as context:
        context.prec=100
        D=Decimal.from_float
        d,t,aa,pa,ab,mb=map(Decimal,(0,1,0,0,0,0))
        for v,log_t in zip(values,logs):
            rd,rt,ra,rp,rb,rm=D(float(v[0])),D(float(log_t)).exp(),*[D(float(x)) for x in v[2:]]
            d,t,aa,pa,ab,mb=d+rd,rt*t,rt*aa+ra,rt*(pa+rd*aa)+rp,rt*ab+rb,rt*mb+rm+d*rb
        return np.array([float(x) for x in (d,t,aa,pa,ab,mb,t*D(float(incident))+pa+mb)])


def run(fun,values,logs,ranges,incident):
    values=np.ascontiguousarray(values,dtype='f8');logs=np.ascontiguousarray(logs,dtype='f8')
    ranges=np.asarray(ranges,dtype='i4');begin=np.ascontiguousarray(ranges[:,0]);end=np.ascontiguousarray(ranges[:,1])
    initial=np.ascontiguousarray(incident,dtype='f8');out=np.full((len(ranges),7),-17.,dtype='f8')
    return fun(len(values),values,logs,len(ranges),begin,end,initial,out),out


@pytest.mark.parametrize('n',[1,2,3,7,17,33])
def test_every_subrange_keeps_temporal_order_and_positive_time_moments(ordered,n):
    rng=np.random.default_rng(1021+n);values=rng.uniform(.01,1.,(n,6));values[:,0]/=n
    logs=rng.uniform(-3.,1.,n);values[:,1]=np.exp(logs)
    ranges=[(lo,hi) for lo in range(n+1) for hi in range(lo,n+1)]
    initial=rng.uniform(.01,2.,len(ranges));code,actual=run(ordered,values,logs,ranges,initial)
    assert code==0
    expected=np.stack([reference(values[a:b],logs[a:b],i) for (a,b),i in zip(ranges,initial)])
    np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=0)


def test_early_transmission_underflow_does_not_erase_representable_incident(ordered):
    values=np.zeros((3,6));values[:,0]=1/3;logs=np.array([-500.,-500.,500.]);values[:,1]=np.exp(logs)
    code,actual=run(ordered,values,logs,[(0,3)],[1e300]);assert code==0
    expected=reference(values,logs,1e300)
    assert expected[6]>1e80 and actual[0,6]>1e80
    np.testing.assert_allclose(actual[0],expected,rtol=3e-12,atol=0)
    assert math.exp(-500)*math.exp(-500)==0.


def test_small_source_survives_attenuation_then_gain(ordered):
    values=np.zeros((4,6));values[:,0]=.25;logs=np.array([0.,-500.,-500.,500.]);values[:,1]=np.exp(logs)
    values[0,3]=1e300
    code,actual=run(ordered,values,logs,[(0,4)],[0.]);assert code==0
    np.testing.assert_allclose(actual[0],reference(values,logs,0.),rtol=3e-12,atol=0)
    assert actual[0,6]>1e80


@pytest.mark.parametrize('failure',['source','log','radius','range','incident','unsupported_scale'])
def test_invalid_or_unsupported_queries_do_not_publish_partial_results(ordered,failure):
    values=np.ones((3,6));values[:,0]=.1;logs=np.zeros(3);ranges=[(0,1),(0,3)];initial=[1.,1.]
    if failure=='source':values[2,3]=-1.
    elif failure=='log':logs[2]=math.nan
    elif failure=='radius':values[2,0]=math.inf
    elif failure=='range':ranges[1]=(0,4)
    elif failure=='incident':initial[1]=math.nan
    else:logs[2]=-1e100
    code,out=run(ordered,values,logs,ranges,initial)
    assert code!=0
    np.testing.assert_array_equal(out,np.full((2,7),-17.))
