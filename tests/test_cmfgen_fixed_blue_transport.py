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
import compare_cmfgen_fixed_blue_transport as comparison


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


def test_extended_domain_difference_vanishes_with_the_same_incoming(kernel):
    import time
    radius = np.array([2., 1.]); exterior = np.array([3., 2., 1.])
    geometries = []
    for n in (3, 2):
        geometries.append(dict(R=exterior[:n], LNK=np.array([2, 3]),
            S_M=np.arange(n, dtype=float), S_P=-np.arange(n, dtype=float),
            B_M=np.zeros(n), B_P=np.zeros(n)))
    weights = {k+'QW_'+s: np.ones((2, 2)) for k in 'JHKN' for s in 'PM'}
    g = dict(ND=2, NP=2, NC=1, ND_EXT=3, ND_ADD=1, IDMIN=1, IDMAX=2,
             R_EXT=exterior, RAY_POINTS_INSERTED=False, rays=geometries, weights=weights)
    a = dict(R=radius, CHI=np.full(2, .2), ETA=np.full(2, .1),
             ESEC=np.full(2, .05), dLOG_NU=.01)
    actual = dict(rays=[dict(I_M=np.full(n, .7), I_P=np.full(n, .9)) for n in (3, 2)])
    blue = dict(rays=[dict(I_M=np.zeros(n), I_P=np.zeros(n)) for n in (3, 2)])
    internal, _ = transport(kernel, a, g, blue, actual, 4, time.monotonic()+2)
    extended, stat = transport(kernel, a, g, blue, actual, 4, time.monotonic()+2, 'extended')
    assert np.max(abs(internal-extended)) > .01
    chi, eta = comparison.extended_material(a, g)
    incoming = kernel.leg([1.], chi[:2], eta[:2], [0., 0.], [0., 0.], 100., .7, 4)[-1]
    for ray in actual['rays']: ray['I_M'][1] = incoming
    matched, _ = transport(kernel, a, g, blue, actual, 4, time.monotonic()+2)
    np.testing.assert_allclose(matched, extended, rtol=3e-14)
    assert stat['directional_cells'] == 6


@pytest.fixture
def formal_packet(tmp_path, monkeypatch):
    build = tmp_path/'build'; build.mkdir()
    (build/'formal-build.json').write_text('{"ray_inventory":[]}')
    a = {name: np.array([.2, .3]) for name in ('CHI', 'ETA', 'ESEC', 'R', 'V', 'SIGMA')}
    a.update(FREQ=4.2, dLOG_NU=.01, NC=1, NP=2, ND=2)
    entry = dict(phase=(71, 44360, 2, 0, 0), args=a)
    final = dict(phase=entry['phase'], args={k: v.copy() if isinstance(v, np.ndarray) else v for k, v in a.items()})
    monkeypatch.setattr(comparison, 'formal_state', lambda path, _: entry if path.name.endswith('_in.bin') else final)
    monkeypatch.setattr(comparison, 'geometry', lambda _: dict(NC=1, NP=2, ND=2, RAY_POINTS_INSERTED=False))
    monkeypatch.setattr(comparison, 'rays', lambda _: dict(NC=1, NP=2, ND=2))
    return tmp_path, build, entry, final


def test_saved_total_eta_does_not_claim_a_source_J_decomposition(formal_packet):
    root, build, _, _ = formal_packet
    *_, identity = comparison.load_packet(root, build, None, 'FORMAL_44360_02')
    assert identity == {'kind': 'SAVED_FORMAL_TOTAL_ETA', 'decomposition_verified': False}


@pytest.mark.parametrize('bad', ['phase', 'ETA', 'CHI', 'FREQ', 'dimension', 'prefix'])
def test_formal_packet_mismatch_is_not_silently_accepted(formal_packet, monkeypatch, bad):
    root, build, entry, final = formal_packet
    prefix = 'FORMAL_44360_02'
    if bad == 'phase': final['phase'] = (71, 44361, 2, 0, 0)
    elif bad == 'prefix': prefix = '../FORMAL_44360_02'
    elif bad == 'dimension':
        monkeypatch.setattr(comparison, 'rays', lambda _: dict(NC=1, NP=3, ND=2))
    elif bad == 'FREQ': final['args']['FREQ'] += .01
    else: final['args'][bad][0] += .01
    with pytest.raises(ValueError): comparison.load_packet(root, build, None, prefix)


def test_exported_full_ray_state_reproduces_moments_without_mutating_inputs(kernel):
    import time
    r = np.array([2., 1.])
    ray = dict(R=r, LNK=np.array([1, 2]), S_M=np.array([0., 1.]),
        S_P=np.array([0., -1.]), B_M=np.zeros(2), B_P=np.zeros(2))
    weights = {k+'QW_'+s: np.ones((2, 2)) for k in 'JHKN' for s in 'PM'}
    g = dict(ND=2, NP=2, NC=2, ND_EXT=2, ND_ADD=0, IDMIN=1, IDMAX=2,
             R_EXT=r, RAY_POINTS_INSERTED=False, rays=[ray, ray], weights=weights)
    a = dict(R=r, CHI=np.full(2, .2), ETA=np.full(2, .1), ESEC=np.full(2, .05), dLOG_NU=.01)
    actual = dict(rays=[dict(I_M=np.full(2, .7), I_P=np.full(2, .9)) for _ in range(2)])
    previous = dict(rays=[dict(I_M=np.zeros(2), I_P=np.zeros(2)) for _ in range(2)])
    returned = []
    result, _ = transport(kernel, a, g, previous, actual, 4, time.monotonic()+2,
                           'extended', ray_output=returned)
    rebuilt = sum(comparison.angular_integral(g, dict(NC=2, NP=2, ND=2, rays=returned)))
    np.testing.assert_array_equal(result, rebuilt)
    returned[0]['I_P'][0] = 999.
    np.testing.assert_array_equal(actual['rays'][0]['I_P'], .9)
    with pytest.raises(ValueError):
        transport(kernel, a, g, previous, actual, 4, time.monotonic()+2, ray_output=[])


def test_pair_rejects_a_broken_saved_blue_chain_before_building(tmp_path, monkeypatch):
    geom = tmp_path/'geometry.bin'; geom.write_bytes(b'geometry')
    current = tmp_path/'current.bin'; current.write_bytes(b'current')
    wrong = tmp_path/'blue.bin'; wrong.write_bytes(b'wrong')
    def packet(root, build, inv, prefix):
        is_first = prefix == 'FORMAL_44360_02'
        paths = [geom, geom, geom, wrong, current if is_first else wrong]
        return dict(phase=(71, 44360 if is_first else 44361, 2, 0, 0)), {}, {}, {}, {}, paths, {}
    monkeypatch.setattr(comparison, 'load_packet', packet)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    out = tmp_path/'must-not-exist'
    with pytest.raises(ValueError, match='current→blue'):
        comparison.advance_pair(tmp_path, tmp_path, out)
    assert not out.exists()
