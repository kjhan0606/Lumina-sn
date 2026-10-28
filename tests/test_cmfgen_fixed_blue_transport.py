"""고정-blue 식의 생산 적분 코어 재사용과 입력·물리 판정 경계를 확인한다."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from compare_cmfgen_fixed_blue_transport import build_kernel, differences, compare, transport


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    return build_kernel(tmp_path_factory.mktemp('fixed-blue-kernel'))[0]


@pytest.mark.parametrize('chi', [0., .7, -1.2, 1200.])
def test_grey_constant_source_matches_closed_form(kernel, chi):
    b, blue, eta, alpha, length, incoming = .13, .7, .2, 4., .8, .4
    extinction = chi+(alpha+3)*b
    emission = eta+alpha*b*blue
    tau = extinction*length
    exact = incoming*np.exp(-tau)+emission*length*(-np.expm1(-tau))/tau if tau else incoming+emission*length
    result = kernel.leg([length], [chi]*2, [eta]*2, [b]*2, [blue]*2, alpha, incoming, 1)
    np.testing.assert_allclose(result, [incoming, exact], rtol=3e-13)


def test_linear_blue_product_converges_to_independent_integral(kernel):
    chi, eta, b, blue = [np.array(a) for a in ([.2, 1.1], [.3, .8], [.07, .3], [.9, .2])]
    alpha, length, incoming = 9., 1.2, .4
    effective = chi+(alpha+3)*b
    transmission = np.exp(-.5*length*effective.sum())
    def integrand(x):
        e = (1-x)*eta[0]+x*eta[1]
        drift = (1-x)*b[0]+x*b[1]
        previous = (1-x)*blue[0]+x*blue[1]
        remaining = length*(1-x)*(effective[1]+.5*(1-x)*(effective[0]-effective[1]))
        return length*(e+alpha*drift*previous)*np.exp(-remaining)
    exact = incoming*transmission+quad(integrand, 0, 1, epsabs=1e-13, epsrel=1e-13)[0]
    errors = []
    for pieces in (1, 4, 16, 64):
        result = kernel.leg([length], chi, eta, b, blue, alpha, incoming, pieces)[-1]
        errors.append(abs(result-exact))
    assert all(a > 8*b for a, b in zip(errors[:-1], errors[1:])), errors
    assert errors[-1] < 2e-5


@pytest.mark.parametrize('invalid', ['length', 'eta', 'b', 'blue', 'alpha', 'incoming'])
def test_invalid_inputs_fail_without_changing_inputs(kernel, invalid):
    values = dict(length=np.array([1.]), chi=np.array([.2, .3]), eta=np.array([.1, .2]),
        b=np.array([.1, .2]), blue=np.array([.3, .4]), alpha=2., incident=.1, pieces=1)
    if invalid in ('alpha', 'incoming'):
        values['incident' if invalid == 'incoming' else invalid] = -1.
    else:
        values[invalid][0] = -1.
    saved = {k: a.copy() for k, a in values.items() if isinstance(a, np.ndarray)}
    with pytest.raises(ValueError): kernel.leg(**values)
    for k, a in saved.items(): np.testing.assert_array_equal(values[k], a)


def test_signed_flux_uses_local_J_not_flux_denominator():
    a = np.array([[1., 2.], [-.4, 0.], [.2, 1.], [-.1, 0.]])
    b = a.copy(); b[1] += .1
    np.testing.assert_allclose(differences(a, b)[1], [.1, .05])


def test_existing_output_is_preserved(tmp_path):
    marker = tmp_path/'original.json'; marker.write_text('{"original":true}')
    with pytest.raises(FileExistsError): compare(tmp_path, tmp_path, tmp_path, tmp_path)
    assert json.loads(marker.read_text()) == {'original': True}


def test_zero_length_tangent_does_not_need_a_spatial_cell():
    class NoCell:
        def leg(self, *args): raise AssertionError('영길이 접선에서 cell 호출')
    import time
    ray = dict(R=np.array([1.]), LNK=np.array([1]))
    weights = {k+'QW_'+s: np.array([[.5]]) for k in 'JHKN' for s in 'PM'}
    g = dict(ND=1, NP=1, NC=0, rays=[ray], weights=weights)
    actual = dict(rays=[dict(I_M=np.array([.7]))]); blue = dict(rays=[{}])
    a = dict(R=np.array([1.]), CHI=np.array([.2]), ETA=np.array([.1]))
    result, stat = transport(NoCell(), a, g, blue, actual, 1, time.monotonic()+1)
    np.testing.assert_allclose(result, .7)
    assert stat['directional_cells'] == 0


def test_core_boundary_is_fixed_but_tangent_is_continuous(kernel):
    import time
    radius = np.array([2., 1.]); geometries = []
    for n in (2, 2, 1):
        geometries.append(dict(R=radius[:n], LNK=np.arange(1, n+1),
            S_M=np.arange(n, dtype=float), S_P=-np.arange(n, dtype=float),
            B_M=np.zeros(n), B_P=np.zeros(n)))
    weights = {k+'QW_'+s: np.ones((2, 3)) for k in 'JHKN' for s in 'PM'}
    g = dict(ND=2, NP=3, NC=1, rays=geometries, weights=weights)
    a = dict(R=radius, CHI=np.full(2, .2), ETA=np.zeros(2), dLOG_NU=.01)
    actual = dict(rays=[dict(I_M=np.full(n, .7), I_P=np.full(n, .9)) for n in (2, 2, 1)])
    blue = dict(rays=[dict(I_M=np.zeros(n), I_P=np.zeros(n)) for n in (2, 2, 1)])
    result, stat = transport(kernel, a, g, blue, actual, 4, time.monotonic()+2)
    expected_outer = .7+.9*np.exp(-.2)+.7+.7*np.exp(-.4)+1.4
    expected_inner = .7*np.exp(-.2)+.9+2*.7*np.exp(-.2)
    np.testing.assert_allclose(result, [[expected_outer, expected_inner]]*4, rtol=2e-14)
    assert stat['directional_cells'] == 4
