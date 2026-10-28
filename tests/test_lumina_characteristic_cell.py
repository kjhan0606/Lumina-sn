"""실제 이동ν 셀의 생산 Reader·D 미분·독립 ODE·공용 chain 일치를 확인한다."""
from types import SimpleNamespace

import numpy as np
import pytest

from compare_lumina_characteristic_cell import Sampler,Cell,reference,build
from lumina_fullsr_frozen import FrozenKernel
from lumina_fullsr_sc import SCALED


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    return build(tmp_path_factory.mktemp('characteristic-cell'))[0]


def sampler(library,chi=.7,eta=.3):
    frequency=np.exp(np.linspace(-1.,1.,17));shape=(2,17)
    return Sampler(library,frequency,np.full(shape,chi),np.full(shape,.1),
                   np.full(shape,eta),np.zeros(shape))


def cell(library,*,moving=False,chi=.7,eta=.3):
    velocity=np.array([.3,.2])*2.99792458e10 if moving else np.zeros(2)
    g=SimpleNamespace(radius=np.array([3.,2.]),velocity=velocity)
    return Cell(sampler(library,chi,eta),g,2.,3.,1.,1.)


def test_same_reader_assembles_ES_J_before_frequency_interpolation(library):
    frequency=np.exp(np.array([-.2,0.,.2]));es=np.tile([.1,.3,.5],(2,1))
    j=np.tile([1.,3.,5.],(2,1));sample=Sampler(library,frequency,np.ones((2,3)),es,np.zeros((2,3)),j)
    chi,eta=sample.at(.5,np.exp(-.1))
    assert chi[0]==pytest.approx(1.) and eta[0]==pytest.approx(.5)
    assert eta[0]!=pytest.approx(.2*2.)


def test_sampler_preserves_native_frequency_endpoints_and_rejects_extrapolation(library):
    sample=sampler(library)
    chi,eta=sample.at([0.,1.],sample.frequency[[0,-1]])
    np.testing.assert_array_equal(chi,[.7,.7]);np.testing.assert_array_equal(eta,[.3,.3])
    with pytest.raises(ValueError):sample.at(.5,sample.frequency[-1]*1.01)
    with pytest.raises(ValueError):sample.at(1.1,1.)


def test_radial_weights_preserve_inner_outer_order_and_product_before_frequency(library):
    frequency=np.exp(np.array([-.1,.1]))
    plane=lambda values:np.repeat(np.asarray(values,dtype='f8')[:,None],2,axis=1)
    sample=Sampler(library,frequency,plane([.2,.8]),plane([.1,.4]),
                   plane([.3,.5]),plane([1.,4.]))
    chi,eta=sample.at([0.,.5,1.],1.)
    np.testing.assert_allclose(chi,[.8,.5,.2],rtol=1.e-14)
    np.testing.assert_allclose(eta,[2.1,.4+.25*2.5,.4],rtol=1.e-14)
    assert eta[1]!=pytest.approx((2.1+.4)/2)


def test_analytic_doppler_gradient_matches_independent_finite_difference(library):
    c=cell(library,moving=True);x=.37;epsilon=1.e-5
    finite=(np.log(c.point(x+epsilon)[2])-np.log(c.point(x-epsilon)[2]))/(2*epsilon)/(c.z1-c.z0)
    assert c.point(x)[3]==pytest.approx(finite,rel=1.e-8)


def test_static_gray_ODE_and_chain_match_closed_form(library):
    c=cell(library);incoming=.4;exact=.4*np.exp(-.7*(c.z1-c.z0))+.3/.7*(-np.expm1(-.7*(c.z1-c.z0)))
    actual,_=reference(c,incoming,1.e-10)
    assert actual==pytest.approx(exact,rel=1.e-11)
    initial=np.zeros(1,dtype=SCALED);initial['mantissa'],initial['exponent']=np.frexp(incoming)
    code,result,_,_=FrozenKernel(library).chain(*c.mesh(1),initial)
    assert code==0
    assert float(np.ldexp(result['mantissa'][0],int(result['exponent'][0])))==pytest.approx(exact,rel=2.e-14)


def test_moving_vacuum_preserves_lorentz_intensity_scaling(library):
    c=cell(library,moving=True,chi=0.,eta=0.);incoming=.4
    expected=incoming*(c.point(1.)[2]/c.point(0.)[2])**3
    actual,_=reference(c,incoming,1.e-10)
    assert actual==pytest.approx(expected,rel=2.e-11)
    initial=np.zeros(1,dtype=SCALED);initial['mantissa'],initial['exponent']=np.frexp(incoming)
    for pieces in (1,4,16):
        code,result,_,_=FrozenKernel(library).chain(*c.mesh(pieces),initial)
        assert code==0
        assert float(np.ldexp(result['mantissa'][0],int(result['exponent'][0])))==pytest.approx(expected,rel=4.e-14)


def test_refinement_keeps_identical_native_samples(library):
    c=cell(library,moving=True)
    coarse=c.mesh(4);fine=c.mesh(8)
    np.testing.assert_array_equal(coarse[2],fine[2][::2])
    np.testing.assert_array_equal(coarse[3],fine[3][::2])
    assert np.sum(coarse[1])==pytest.approx(np.sum(fine[1]),rel=2.e-14)
