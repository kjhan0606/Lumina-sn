"""원본 Fortran 미분식/주파수 항과 새 국소 계수의 직접 대조. 전체 수송 아님."""
import ctypes as ct
from decimal import Decimal, localcontext
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
PTR=ct.POINTER(ct.c_double)


class Characteristic(ct.Structure):
    _fields_=[(name,ct.c_double) for name in ('gamma','dr_ds','dmu_ds','b')]


class Frequency(ct.Structure):
    _fields_=[(name,ct.c_double) for name in ('chi','eta','alpha')]


class Segment(ct.Structure):
    _fields_=[(name,ct.c_double) for name in ('length','drift','mu_from','mu_to')]


@pytest.fixture(scope='module')
def libraries(tmp_path_factory):
    out=tmp_path_factory.mktemp('rel-characteristic')
    subprocess.run([os.environ.get('CC','cc'),'-O2','-std=c11','-Wall','-Wextra','-Werror',
        '-fPIC','-shared','-I'+str(ROOT/'src'),str(ROOT/'src/cmf_relativistic_characteristic.c'),
        '-lm','-o',str(out/'characteristic.so')],check=True,timeout=45,capture_output=True)
    native=Path(os.environ.get('CMFGEN_SOURCE_ROOT','/gpfs/kjhan/cmfgen_src/cur_cmf'))
    for relative in ('tools/set_kind_module.f','plane/derivs.f','plane/derivr.f','plane/rel_variables.f'):
        shutil.copy2(native/relative,out/Path(relative).name)
    subprocess.run(['gfortran','-O2','-fPIC','-shared','-ffixed-line-length-none',
        'set_kind_module.f','derivs.f','derivr.f','rel_variables.f','-o','native.so'],
        cwd=out,check=True,timeout=45,capture_output=True)
    c=ct.CDLL(str(out/'characteristic.so'));f=ct.CDLL(str(out/'native.so'))
    c.cmf_rel_characteristic_coefficients.argtypes=[ct.c_double]*4+[ct.POINTER(Characteristic)]
    c.cmf_rel_characteristic_coefficients.restype=ct.c_int
    c.cmf_rel_frequency_raw_terms.argtypes=[ct.c_double]*5+[ct.POINTER(Frequency)]
    c.cmf_rel_frequency_raw_terms.restype=ct.c_int
    c.cmf_rel_homologous_segment.argtypes=[ct.c_double]*4+[ct.POINTER(Segment)]
    c.cmf_rel_homologous_segment.restype=ct.c_int
    for name in ('derivs_','derivr_'):
        getattr(f,name).argtypes=[PTR]*6;getattr(f,name).restype=None
    f.rel_variables_.argtypes=[ct.POINTER(ct.c_int)]+[PTR]*7;f.rel_variables_.restype=None
    return c,f


def coefficients(c,r,mu,beta,gradient):
    result=Characteristic()
    assert c.cmf_rel_characteristic_coefficients(r,mu,beta,gradient,ct.byref(result))==0
    return result


def native_derivative(f,name,r,mu,beta,gradient):
    x=ct.c_double(r if name=='derivr_' else 0.)
    state=(ct.c_double*2)(r,mu);answer=(ct.c_double*2)()
    b,g,gamma=ct.c_double(beta),ct.c_double(gradient),ct.c_double(1/np.sqrt(1-beta*beta))
    getattr(f,name)(ct.byref(x),state,answer,ct.byref(b),ct.byref(g),ct.byref(gamma))
    return np.array(answer)


def test_original_derivs_and_derivr_match(libraries):
    c,f=libraries
    for r in (1e12,2e14,6e15):
        for beta in (0.,.01,.12,.4):
            for mu in (-1.,-.7,-beta,.2,1.):
                for gradient in (0.,beta/r,-.3/r):
                    value=coefficients(c,r,mu,beta,gradient)
                    expected=native_derivative(f,'derivs_',r,mu,beta,gradient)
                    np.testing.assert_allclose([value.dr_ds,value.dmu_ds],expected,rtol=3e-15,atol=1e-30)
                    if mu+beta!=0:
                        radial=native_derivative(f,'derivr_',r,mu,beta,gradient)
                        np.testing.assert_allclose([1/value.dr_ds,value.dmu_ds/value.dr_ds],
                            radial,rtol=3e-15,atol=1e-30)


def test_static_and_relativistic_turning_limits(libraries):
    c,_=libraries;r=2e14
    for mu in (-1.,-.5,0.,.7,1.):
        value=coefficients(c,r,mu,0.,0.)
        assert value.dr_ds==mu and value.b==0
        assert value.dmu_ds==pytest.approx((1-mu*mu)/r,rel=2e-15)
    for beta in (.01,.12,.4):
        value=coefficients(c,r,-beta,beta,beta/r)
        assert value.dr_ds==0
        assert value.dmu_ds==pytest.approx((1-beta*beta)**1.5/r,rel=3e-15)
        assert value.b==pytest.approx(beta/(value.gamma*r),rel=3e-15)


def test_radial_homologous_doppler_gradient(libraries):
    c,_=libraries;r=6e15;beta=.12;gradient=beta/r
    for mu in (-1.,1.):
        value=coefficients(c,r,mu,beta,gradient)
        assert value.dmu_ds==0
        assert value.b==pytest.approx(value.gamma**3*(1+mu*beta)*gradient,rel=3e-15)
        # ν_cmf/ν_lab=1/[γ(1+βμ)]의 정확한 radial 미분.
        assert -value.b/value.dr_ds==pytest.approx(-mu*gradient/(1-beta*beta),rel=3e-15)


def test_native_radius_unit_conversion(libraries):
    c,f=libraries;r=2e14;mu=.3;beta=.12;gradient=beta/r
    value=coefficients(c,r,mu,beta,gradient)
    # 실제 native R/s 단위10^10cm를 입력한 뒤 물리적 cm^-1로 환산한다.
    expected=native_derivative(f,'derivs_',r*1e-10,mu,beta,gradient*1e10)
    np.testing.assert_allclose([value.dr_ds,value.dmu_ds],expected*np.array([1.,1e-10]),rtol=3e-15)
    q=1e4;chi=1e-14;eta=2e-22;I=1e-7;terms=Frequency()
    assert c.cmf_rel_frequency_raw_terms(chi,eta,q,value.b,I,ct.byref(terms))==0
    n=ct.c_int(1);chi0,eta0,q0,b0,I0=map(ct.c_double,(chi*1e10,eta*1e10,q,value.b*1e10,I))
    out_chi,out_s=ct.c_double(),ct.c_double()
    f.rel_variables_(ct.byref(n),ct.byref(chi0),ct.byref(eta0),ct.byref(q0),ct.byref(b0),
        ct.byref(I0),ct.byref(out_chi),ct.byref(out_s))
    assert terms.chi==pytest.approx(out_chi.value*1e-10,rel=3e-15)
    assert terms.eta==pytest.approx(out_chi.value*out_s.value*1e-10,rel=3e-15)


def test_lab_impact_and_frequency_differential_invariants(libraries):
    c,_=libraries;r=2e14
    for beta in (.01,.12,.4):
        for mu in (-.8,-beta,.3,.8):
            gradient=beta/r;v=coefficients(c,r,mu,beta,gradient)
            impact_change=v.dr_ds/r-v.b-mu*v.dmu_ds/(1-mu*mu)
            assert abs(impact_change)<2e-15*(abs(v.dr_ds/r)+abs(v.b)+abs(v.dmu_ds))
            expected=v.gamma**2*beta*gradient*v.dr_ds+(
                gradient*mu*v.dr_ds+beta*v.dmu_ds)/(1+beta*mu)
            assert v.b==pytest.approx(expected,rel=4e-15)


def test_native_rel_variables_match_raw_frequency_terms(libraries):
    c,f=libraries
    for b in (0.,1e-17,3e-16):
        for q in (0.,1.,1e4):
            for chi in (-1e-15,0.,1e-14):
                if chi+(q+3)*b==0:continue # 원본의 η/χ 특이점은 아래 raw 시험으로 구분한다.
                eta,I=2e-22,3e-7;value=Frequency()
                assert c.cmf_rel_frequency_raw_terms(chi,eta,q,b,I,ct.byref(value))==0
                n=ct.c_int(1);chi0,eta0,q0,b0,I0=map(ct.c_double,(chi,eta,q,b,I))
                out_chi,out_s=ct.c_double(),ct.c_double()
                f.rel_variables_(ct.byref(n),ct.byref(chi0),ct.byref(eta0),ct.byref(q0),
                    ct.byref(b0),ct.byref(I0),ct.byref(out_chi),ct.byref(out_s))
                assert value.chi==pytest.approx(out_chi.value,rel=2e-15,abs=0)
                assert value.eta==pytest.approx(out_chi.value*out_s.value,rel=3e-15,abs=0)


def test_zero_signed_extinction_and_red_upwind_are_not_repaired(libraries):
    c,_=libraries;value=Frequency(7,8,9);b=1e-16;q=10.
    assert c.cmf_rel_frequency_raw_terms(-(q+3)*b,1e-22,q,b,1e-7,ct.byref(value))==0
    assert value.chi==0 and value.eta>0
    assert c.cmf_rel_frequency_raw_terms(-1e-14,1e-22,0,0,0,ct.byref(value))==0
    assert value.chi==-1e-14 and value.eta==1e-22
    before=bytes(value)
    assert c.cmf_rel_frequency_raw_terms(1e-14,1e-22,q,-b,1e-7,ct.byref(value))==3
    assert bytes(value)==before


@pytest.mark.parametrize('args',[(0.,0.,.1,0.),(1.,1.01,.1,0.),(1.,0.,1.,0.),(1.,0.,.1,np.nan)])
def test_invalid_coefficients_do_not_publish(libraries,args):
    c,_=libraries;value=Characteristic(1,2,3,4);before=bytes(value)
    assert c.cmf_rel_characteristic_coefficients(*args,ct.byref(value))==1
    assert bytes(value)==before


def test_homologous_segment_path_and_doppler_integrals(libraries):
    c,_=libraries;nodes,weights=np.polynomial.legendre.leggauss(64)
    for impact,z0,z1 in ((0.,2e14,4e14),(2e14,-5e15,-1e14),
                         (2e14,-1e14,1e14),(2e14,5e15,5e15+1e8)):
        for epoch in (1683072.,1e30):
            value=Segment()
            assert c.cmf_rel_homologous_segment(impact,z0,z1,epoch,ct.byref(value))==0
            A=2.99792458e10*epoch;z=.5*(z0+z1)+.5*(z1-z0)*nodes
            beta=np.hypot(impact,z)/A;D=(1-z/A)/np.sqrt(1-beta*beta)
            expected=.5*(z1-z0)*np.dot(weights,D)
            assert value.length==pytest.approx(expected,rel=3e-15)
            # 얇은 구간의 두 끝점 로그를 빼면 longdouble에도 상쇄가 남는다.
            # 허용오차를 늘리지 않고 독립적인 80자리 끝점 값으로 대조한다.
            with localcontext() as context:
                context.prec=80
                AA=Decimal.from_float(2.99792458e10)*Decimal.from_float(epoch)
                pp=Decimal.from_float(impact)
                def logD(zz):
                    zz=Decimal.from_float(zz)
                    beta2=(pp*pp+zz*zz)/(AA*AA)
                    return -(1-beta2).ln()/2+(1-zz/AA).ln()
                expected_drift=float(logD(z0)-logD(z1))
            assert value.drift==pytest.approx(expected_drift,rel=2e-13,abs=1e-300)
            assert value.drift>0 and value.length>0
            for zz,mu in ((z0,value.mu_from),(z1,value.mu_to)):
                rr=np.hypot(impact,zz);bb=rr/A;w=zz/rr
                assert mu==pytest.approx((w-bb)/(1-bb*w),rel=3e-15,abs=1e-16)


def test_segment_turning_and_invalid_input(libraries):
    c,_=libraries;p=2e14;epoch=1683072.;v=Segment()
    assert c.cmf_rel_homologous_segment(p,0,1e12,epoch,ct.byref(v))==0
    assert v.mu_from==-p/(2.99792458e10*epoch)
    before=bytes(v)
    for args in ((-p,0.,1e12,epoch),(p,1e12,0.,epoch),(p,0.,1e12,0.),(1e20,0.,1e12,epoch)):
        assert c.cmf_rel_homologous_segment(*args,ct.byref(v))==1
        assert bytes(v)==before
