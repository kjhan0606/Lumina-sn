"""일반 χ 원천 모멘트 후보와 독립 native-bin 적분을 직접 대조한다."""
import ctypes as ct
from decimal import Decimal,localcontext
from math import comb
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from test_lumina_characteristic_linear import run,decoded,reference

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module',params=[False,True],ids=['original-upper','envelope-upper'])
def source_measure(tmp_path_factory,request):
    path=tmp_path_factory.mktemp('source-measure')/'measure.so'
    proc=subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off',
        '-Wall','-Wextra','-Werror','-fPIC','-shared',*(['-DCMF_GPU_ENVELOPE_UPPER'] if request.param else []),'-I'+str(ROOT/'src'),
        str(ROOT/'tests/source_measure_probe.cpp'),'-o',str(path)],capture_output=True,text=True,timeout=60)
    assert proc.returncode==0,proc.stderr
    lib=ct.CDLL(str(path));f=lib.source_measure_probe
    p=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    e=np.ctypeslib.ndpointer(dtype='i8',flags='C_CONTIGUOUS')
    i=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    f.argtypes=[ct.c_int,ct.c_int,p,p,ct.c_int,p,e,i];f.restype=ct.c_int
    f._library=lib
    return f


@pytest.mark.parametrize('length,u0,u1,hi0,hi1',[
    (1.,0.,1.,0.,0.),(1.3,.125,.875,2.,7.),(1e300,.25,.75,1e300,1e-300),
    (1e-300,.25,.75,1e-300,2e-300),(1e300,0.,1.,np.nextafter(0.,1.),0.),
    (1e300,0.,2.**-60,1e-300,1e-200),(1.3,1.-2.**-52,1.,7.,2.),
    (1e300,.25,.75,1e-300,1e300)])
def test_envelope_is_above_high_precision_pointwise_bound_without_intermediate_loss(source_measure,length,u0,u1,hi0,hi1):
    f=source_measure._library.source_envelope_upper_probe
    f.argtypes=[ct.c_double]*5+[ct.c_void_p]*2;f.restype=ct.c_int
    m=ct.c_double(-17.);e=ct.c_int64(-17)
    assert f(length,u0,u1,hi0,hi1,ct.byref(m),ct.byref(e))==0
    with localcontext() as context:
        context.prec=100;D=Decimal.from_float
        end=lambda u:D(u)*D(hi0)+(1-D(u))*D(hi1)
        wanted=D(length)*(D(u1)-D(u0))*max(end(u0),end(u1))
        actual=D(m.value)*Decimal(2)**e.value
        assert actual>=wanted


@pytest.mark.parametrize('args',[(0.,0.,1.,1.,1.),(1.,.8,.2,1.,1.),(1.,0.,1.,-1.,1.),
                                  (1.,0.,1.,float('nan'),1.)])
def test_envelope_failure_preserves_output(source_measure,args):
    f=source_measure._library.source_envelope_upper_probe
    f.argtypes=[ct.c_double]*5+[ct.c_void_p]*2;f.restype=ct.c_int
    m=ct.c_double(-17.);e=ct.c_int64(-17)
    assert f(*args,ct.byref(m),ct.byref(e))==1
    assert m.value==-17. and e.value==-17


@pytest.mark.parametrize('block',[1,4,32])
def test_native_knots_signed_space_frequency_blue_and_static(source_measure,block):
    rng=np.random.default_rng(102211)
    for shift,span in [(.217,0.),(.217,7.45),(2.31,18.7),(19.,3.)]:
        values=np.vstack([rng.uniform(-.8,1.,(2,17)),rng.uniform(.01,2.,(2,17))])
        args=dict(length=1.7,shift=shift,span=span,delta=.021,incident=.13)
        result=run(source_measure,values,relative=1e-8,block=block,**args);fields=decoded(result)
        expected=reference(values,**args)
        assert abs(fields[0]-expected)<=fields[3]+2e-13*abs(expected)
        assert fields[3]<=1e-8*fields[0]


@pytest.mark.parametrize('chi',[0.,1.,1000.,-30.])
def test_static_absorption_gain_fullsr_and_arithmetic_error(source_measure,chi):
    values=np.array([[chi]*9,[chi]*9,[2.]*9,[2.]*9])
    result=run(source_measure,values,span=6.8,shift=.37,delta=.013,incident=.7,relative=1e-8)
    fields=decoded(result)
    with localcontext() as context:
        context.prec=90;D=Decimal.from_float;depth=D(chi)+3*D(.013)
        expected=D(.7)*(-depth).exp()+D(2.)*(1-(-depth).exp())/depth
        actual=D(result[1][0])*Decimal(2)**int(result[2][0])
        error=D(result[1][3])*Decimal(2)**int(result[2][3])
        assert abs(actual-expected)<=error
    assert fields[3]<=1e-8*fields[0]


def test_no_eta_secant_splits_when_all_panel_endpoints_miss_emission(source_measure):
    values=np.zeros((4,129));values[2:,79]=[3.,7.]
    args=dict(shift=.125,span=126.5,length=2.1)
    result=run(source_measure,values,relative=1e-8,**args);fields=decoded(result)
    expected=reference(values,**args)
    assert fields[0]>0 and abs(fields[0]-expected)<=fields[3]+2e-14*expected
    assert result[3][0]==1 and result[3][2]==0


def test_narrow_chi_and_deep_absorber_after_exit_emission(source_measure):
    values=np.zeros((4,129));values[2:,79]=[3.,7.];values[:2,110]=1e6
    args=dict(shift=.125,span=126.5,length=2.1)
    result=run(source_measure,values,relative=1e-8,**args);fields=decoded(result)
    values[:2]=0.;expected=reference(values,**args)
    assert fields[0]>0 and abs(fields[0]-expected)<=fields[3]+2e-13*expected


def test_narrow_absorption_and_individual_weak_source_budget(source_measure):
    values=np.zeros((4,65));values[2:]=1.;values[:2,3]=[120.,80.]
    args=dict(shift=.125,span=63.5,length=1.3,incident=0.)
    bright=decoded(run(source_measure,values,relative=1e-8,**args))
    values[2:]*=1e-80
    weak=decoded(run(source_measure,values,relative=1e-8,**args));expected=reference(values,**args)
    assert weak[0]>0 and weak[3]<=1e-8*weak[0]
    assert abs(weak[0]-expected)<=weak[3]+2e-13*expected
    assert abs(weak[0]/(1e-80*bright[0])-1)<1e-8


@pytest.mark.parametrize('length,eta,chi',[(1e-200,1e-200,-500e200),
    (1e300,1e300,1000e-300),(1e300,float(np.nextafter(0.,1.)),-500e-300),(1.,1.,-1000.)])
def test_positive_scales_do_not_require_representable_intermediate_products(source_measure,length,eta,chi):
    values=np.array([[chi]*3,[chi]*3,[eta]*3,[eta]*3])
    result=run(source_measure,values,length=length,span=1.25,relative=1e-7)
    assert result[0]==0,result[0]
    with localcontext() as context:
        context.prec=100;D=Decimal.from_float;depth=D(length)*D(chi)
        expected=D(length)*D(eta)*(1-(-depth).exp())/depth
        actual=D(result[1][0])*Decimal(2)**int(result[2][0])
        error=D(result[1][3])*Decimal(2)**int(result[2][3])
        assert actual>0 and abs(actual-expected)<=error
        assert error<=Decimal('1e-7')*actual


def test_varying_subnormal_recovered_and_true_zero_preserved(source_measure):
    tiny=np.nextafter(0.,1.)
    values=np.zeros((4,5));values[2]=np.array([1,0,2,0,1])*tiny;values[3]=np.array([0,1,0,2,0])*tiny
    fields=decoded(run(source_measure,values,span=3.7,shift=.1,length=1e300,relative=1e-8))
    normalized=values.copy();normalized[2:]/=tiny
    # 독립 기준도 tiny를 먼저 곱해 subnormal 반올림으로 훼손하지 않는다.
    expected=reference(normalized,span=3.7,shift=.1)*(tiny*1e300)
    assert fields[0]>0 and abs(fields[0]-expected)<=fields[3]+2e-13*expected
    values[2:]=0.;values[:2]=-1.
    np.testing.assert_array_equal(decoded(run(source_measure,values,span=3.7,shift=.1)),np.zeros(7))


def test_random_general_opacity_not_only_transparent_fast_path(source_measure):
    rng=np.random.default_rng(102212)
    for _ in range(20):
        nf=int(rng.integers(4,23))
        values=np.vstack([rng.uniform(-2,3,(2,nf)),rng.uniform(0,3,(2,nf))])
        args=dict(length=float(rng.uniform(.2,2)),shift=float(rng.uniform(0,nf)),
            span=float(rng.uniform(0,1.5*nf)),delta=float(rng.uniform(0,.03)),incident=float(rng.uniform(0,1)))
        result=run(source_measure,values,relative=1e-8,**args);fields=decoded(result);expected=reference(values,**args)
        assert abs(fields[0]-expected)<=fields[3]+2e-13*abs(expected)
        assert fields[3]<=1e-8*fields[0]


@pytest.mark.parametrize('bad',['eta','length','span','tolerance','work'])
def test_failure_never_publishes_partial_result(source_measure,bad):
    values=np.ones((4,17));args={}
    if bad=='eta':values[2,5]=-1.
    elif bad=='length':args['length']=-1.
    elif bad=='span':args['span']=-1.
    elif bad=='tolerance':args['relative']=np.nan
    else:values[:2]=1000.;args['limit']=1
    code,m,e,counts=run(source_measure,values,**args)
    assert code!=0
    np.testing.assert_array_equal(m,np.full(7,-17.));np.testing.assert_array_equal(e,np.full(7,-17))
    np.testing.assert_array_equal(counts,np.full(5,-17))


@pytest.mark.parametrize('width',[1,2,4,7,16,31,32,33,64,255,256])
def test_positive_projection_preserves_exact_constant_linear_and_native_knots(source_measure,width):
    lib=source_measure._library
    p=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    fun=lib.source_projection_weights;fun.argtypes=[ct.c_int,p,ct.c_int];fun.restype=ct.c_int
    weights=np.zeros((width+1,26))
    assert fun(width,weights,weights.size)==0
    assert np.min(weights)>0
    np.testing.assert_allclose(weights.sum(axis=0),np.full(26,1/26),rtol=4e-14,atol=0)
    np.testing.assert_allclose(np.linspace(0,1,width+1)@weights,
                               (np.arange(26)+1)/(26*27),rtol=4e-14,atol=0)
    # 독립 native-bin Gauss16: 이 행렬의 생성/병합 재귀를 사용하지 않는다.
    rng=np.random.default_rng(102220+width);eta=rng.uniform(0,2,width+1)
    x,w=np.polynomial.legendre.leggauss(16);v=(x+1)/2;w=w/(2*width)
    reference=np.zeros(26)
    for j in range(width):
        t=(j+v)/width;source=(1-v)*eta[j]+v*eta[j+1]
        for k in range(26):
            reference[k]+=np.sum(w*comb(25,k)*t**k*(1-t)**(25-k)*source)
    np.testing.assert_allclose(eta@weights,reference,rtol=4e-13,atol=0)


@pytest.mark.parametrize('width',[1,7,32,64,256])
def test_projected_and_sequential_positive_moments_agree_including_subnormal(source_measure,width):
    lib=source_measure._library
    p=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    e=np.ctypeslib.ndpointer(dtype='i8',flags='C_CONTIGUOUS')
    f=lib.source_projection_moments;f.argtypes=[ct.c_int,p,ct.c_int,p,e];f.restype=ct.c_int
    rng=np.random.default_rng(102221+width)
    for eta_scale in (1.,1e-200,1e200,np.nextafter(0.,1.)):
        values=np.zeros((4,width+1));values[2:]=rng.integers(0,4,(2,width+1))*eta_scale
        results=[]
        for direct in (0,1):
            m=np.full(52,-17.);ex=np.full(52,-17,dtype='i8')
            assert f(width,values,direct,m,ex)==0
            with localcontext() as context:
                context.prec=90
                results.append([Decimal.from_float(a)*Decimal(2)**int(b) for a,b in zip(m,ex)])
        for a,b in zip(*results):
            assert abs(a-b)<=Decimal('4e-13')*max(abs(a),abs(b))


def test_projection_failure_preserves_output_canary(source_measure):
    lib=source_measure._library;p=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    e=np.ctypeslib.ndpointer(dtype='i8',flags='C_CONTIGUOUS')
    f=lib.source_projection_moments;f.argtypes=[ct.c_int,p,ct.c_int,p,e];f.restype=ct.c_int
    values=np.ones((4,9));values[2,4]=-1
    m=np.full(52,-17.);ex=np.full(52,-17,dtype='i8')
    assert f(8,values,1,m,ex)!=0
    np.testing.assert_array_equal(m,np.full(52,-17.));np.testing.assert_array_equal(ex,np.full(52,-17))
