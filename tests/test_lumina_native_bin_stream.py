"""원 logν-index 보간의 한-bin 적분/순서 합성을 독립 적분과 대조한다."""
import ctypes as ct
from decimal import Decimal,localcontext
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from test_lumina_characteristic_linear import run,decoded,reference

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    path=tmp_path_factory.mktemp('native-bin-stream')/'probe.so'
    proc=subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off',
        '-Wall','-Wextra','-Werror','-fPIC','-shared','-I'+str(ROOT/'src'),
        str(ROOT/'tests/native_bin_stream_probe.cpp'),'-o',str(path)],capture_output=True,text=True,timeout=60)
    assert proc.returncode==0,proc.stderr
    fun=ct.CDLL(str(path)).native_bin_stream_probe
    p=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    fun.argtypes=[ct.c_int,ct.c_int,p,p,ct.c_int,p,
        np.ctypeslib.ndpointer(dtype='i8',flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')];fun.restype=ct.c_int
    return fun


@pytest.mark.parametrize('shift,span',[(.217,0.),(.217,7.45),(2.31,18.7),(19.,3.),(.125,126.5)])
def test_native_knots_signed_cross_terms_partial_blue_and_fullsr(kernel,shift,span):
    nf=129 if span>100 else 17;rng=np.random.default_rng(102240)
    values=np.vstack([rng.uniform(-.8,1.,(2,nf)),rng.uniform(.01,2.,(2,nf))])
    args=dict(shift=shift,span=span,length=1.7,delta=.021,incident=.13)
    fields=decoded(run(kernel,values,relative=1e-8,**args));expected=reference(values,**args)
    assert abs(fields[0]-expected)<=fields[3]+2e-13*abs(expected)
    assert fields[3]<=1e-8*fields[0] and fields[3]>=sum(fields[4:])*(1-3e-14)


@pytest.mark.parametrize('chi',[0.,1.,1000.,1e8,1e10,-30.])
def test_static_limit_and_thick_bin_subdivision_with_decimal_reference(kernel,chi):
    values=np.array([[chi]*9,[chi]*9,[2.]*9,[2.]*9])
    result=run(kernel,values,span=6.8,shift=.37,delta=.013,incident=.7,relative=1e-8)
    fields=decoded(result)
    with localcontext() as c:
        c.prec=90;D=Decimal.from_float;depth=D(chi)+3*D(.013)
        expected=D(.7)*(-depth).exp()+D(2.)*(1-(-depth).exp())/depth
        actual=D(result[1][0])*Decimal(2)**int(result[2][0])
        error=D(result[1][3])*Decimal(2)**int(result[2][3])
        assert abs(actual-expected)<=error
    assert fields[3]<=1e-8*fields[0]
    if chi>=1000.:assert result[3][3]>0 and result[3][2]>0


@pytest.mark.parametrize('length,eta,chi',[(1e-200,1e-200,-500e200),
    (1e300,1e300,1000e-300),(1e300,float(np.nextafter(0.,1.)),-500e-300),(1.,1.,-1000.)])
def test_independent_scales_keep_recoverable_source_and_gain(kernel,length,eta,chi):
    values=np.array([[chi]*3,[chi]*3,[eta]*3,[eta]*3])
    result=run(kernel,values,length=length,span=1.25,relative=1e-7)
    assert result[0]==0,result[0]
    with localcontext() as c:
        c.prec=100;D=Decimal.from_float;depth=D(length)*D(chi)
        expected=D(length)*D(eta)*(1-(-depth).exp())/depth
        actual=D(result[1][0])*Decimal(2)**int(result[2][0]);error=D(result[1][3])*Decimal(2)**int(result[2][3])
        assert actual>0 and abs(actual-expected)<=error and error<=Decimal('1e-7')*actual


def test_narrow_emission_and_downstream_order_are_not_reversed(kernel):
    values=np.zeros((4,129));values[2:,79]=[3.,7.];values[:2,110]=1e6
    args=dict(shift=.125,span=126.5,length=2.1)
    fields=decoded(run(kernel,values,relative=1e-8,**args))
    values[:2]=0.;expected=reference(values,**args)
    assert fields[0]>0 and abs(fields[0]-expected)<=fields[3]+2e-13*expected


def test_weak_individual_source_has_no_unrelated_bright_scale(kernel):
    rng=np.random.default_rng(102241);values=np.vstack([rng.uniform(-.3,.7,(2,17)),rng.uniform(.2,2.,(2,17))])
    args=dict(shift=.37,span=16.3,delta=.018,length=1.2)
    bright=decoded(run(kernel,values,relative=1e-8,**args));values[2:]*=1e-80
    weak=decoded(run(kernel,values,relative=1e-8,**args));expected=reference(values,**args)
    assert weak[0]>0 and abs(weak[0]-expected)<=weak[3]+2e-13*expected
    assert abs(weak[0]/(1e-80*bright[0])-1)<1e-8


def test_varying_subnormal_before_large_length_is_not_erased(kernel):
    tiny=np.nextafter(0.,1.);v=np.zeros((4,5))
    v[2]=np.array([1,0,2,0,1])*tiny;v[3]=np.array([0,1,0,2,0])*tiny
    fields=decoded(run(kernel,v,span=3.7,shift=.1,length=1e300,relative=1e-8))
    normalized=v.copy();normalized[2:]/=tiny
    expected=reference(normalized,span=3.7,shift=.1)*(tiny*1e300)
    assert fields[0]>0 and abs(fields[0]-expected)<=fields[3]+2e-13*expected


def test_signed_tau_cancellation_and_true_zero(kernel):
    values=np.ones((4,129));values[:2]=4000.*(-1.)**np.arange(129)
    args=dict(shift=.125,span=126.5,length=1.3)
    fields=decoded(run(kernel,values,relative=1e-8,**args));expected=reference(values,**args)
    assert abs(fields[0]-expected)<=fields[3]+2e-12*expected
    values[2:]=0.;np.testing.assert_array_equal(decoded(run(kernel,values,relative=1e-8,**args)),np.zeros(7))


def test_random_native_log_index_profiles_against_independent_integral(kernel):
    rng=np.random.default_rng(102242)
    for _ in range(40):
        nf=int(rng.integers(4,35));v=np.vstack([rng.uniform(-3,5,(2,nf)),rng.uniform(0,3,(2,nf))])
        args=dict(length=float(rng.uniform(.2,2)),shift=float(rng.uniform(0,nf)),
            span=float(rng.uniform(0,2*nf)),delta=float(rng.uniform(0,.03)),incident=float(rng.uniform(0,1)))
        fields=decoded(run(kernel,v,relative=1e-8,**args));expected=reference(v,**args)
        assert abs(fields[0]-expected)<=fields[3]+2e-13*abs(expected)


@pytest.mark.parametrize('source_scale',[1.,1e-80])
def test_opaque_upstream_bins_share_only_the_same_outputs_error_budget(kernel,source_scale):
    v=np.ones((4,129));v[:2]=1000.
    v[2:,50:]*=1e100;v[2:]*=source_scale
    # 50번 이후 원천은 큰 하류 광학깊이로 억제된다. 밝은 다른 출력은 없다.
    args=dict(shift=.125,span=126.5,length=1.3)
    fields=decoded(run(kernel,v,relative=1e-8,**args));expected=reference(v,**args)
    assert fields[0]>0 and abs(fields[0]-expected)<=fields[3]+2e-13*expected
    assert fields[3]<=fields[0]*1e-8


@pytest.mark.parametrize('bad',['eta','length','span','tolerance','work'])
def test_failure_never_publishes_partial_physical_values(kernel,bad):
    values=np.ones((4,17));args={}
    if bad=='eta':values[2,5]=-1.
    elif bad=='length':args['length']=-1.
    elif bad=='span':args['span']=-1.
    elif bad=='tolerance':args['relative']=np.nan
    else:args.update(span=14.3,limit=1)
    code,m,e,counts=run(kernel,values,**args)
    assert code!=0
    np.testing.assert_array_equal(m,np.full(7,-17.));np.testing.assert_array_equal(e,np.full(7,-17))
    np.testing.assert_array_equal(counts,np.full(5,-17))
