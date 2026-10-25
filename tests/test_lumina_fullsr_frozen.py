"""중점 동결의 순서·fullSR 불변량·gain과 양의 주파수 보간을 독립식에 대조한다."""
import ctypes as ct
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from lumina_fullsr_frozen import FrozenKernel
from lumina_fullsr_sc import Location, SCALED
from test_lumina_fullsr_sc import manufactured


@pytest.fixture(scope='module')
def frozen(tmp_path_factory):
    path = tmp_path_factory.mktemp('fullsr-frozen')/'frozen.so'
    built = subprocess.run([os.environ.get('CXX', 'c++'), '-O3', '-std=c++17', '-ffp-contract=off',
        '-Wall', '-Wextra', '-Werror', '-fopenmp', '-fPIC', '-shared',
        str(ROOT/'src/cmf_fullsr_sc.cpp'), str(ROOT/'src/cmf_fullsr_frozen.cpp'), '-o', str(path)],
        capture_output=True, text=True, timeout=90)
    assert built.returncode == 0, built.stderr
    return FrozenKernel(path)


def positive(values):
    result = np.zeros(np.shape(values), dtype=SCALED)
    result['mantissa'], result['exponent'] = np.frexp(values)
    return result


def values(array): return np.ldexp(array['mantissa'], array['exponent'].astype('i4'))


@pytest.mark.parametrize('delta', [0., .0037, .08, .0837, .8, 2.])
@pytest.mark.parametrize('chi', [0., .7, -3.])
def test_grey_transfer_and_blue_extension_match_closed_form(frozen, delta, chi):
    _, provider, _, _ = manufactured(nf=65, eta=.3, chi=chi)
    loc = Location(1, 1, 0, .3); incident = positive(np.full(65, .4))
    code, result, report = frozen.cell(provider, loc, 1., delta, incident)
    assert code == 0, report
    tau = chi+3*delta
    expected = .4*math.exp(-tau)+.3*(-math.expm1(-tau))/tau if tau else .7
    # 일정 I 보간은 U의 곡률 때문에 유한 격자에서 근사다. 정적/연장 끝은 정확하다.
    np.testing.assert_allclose(values(result)[-1], expected, rtol=3e-14)
    if delta == 0: np.testing.assert_allclose(values(result), expected, rtol=3e-14)
    else: assert np.max(abs(values(result)/expected-1)) < 2e-4


@pytest.mark.parametrize('reconstruction', [0, 1])
def test_vacuum_preserves_i_over_nu_cubed(frozen, reconstruction):
    _, provider, _, _ = manufactured(nf=129, eta=0., chi=0.)
    h = provider._source.dlognu; delta = .2437
    spectrum = .2*np.exp(3*h*np.arange(129))
    code, result, report = frozen.cell(provider, Location(1, 1, 0, .3), 1., delta,
                                      positive(spectrum), reconstruction=reconstruction)
    assert code == 0, report
    np.testing.assert_allclose(values(result)[:100], spectrum[:100], rtol=4e-14)


def test_underflow_remains_recoverable_across_spatial_cells(frozen):
    _, provider, _, _ = manufactured(nf=17, eta=0., chi=1200.)
    loc = Location(1, 1, 0, .3)
    code, tiny, _ = frozen.cell(provider, loc, 1., 0., positive(np.ones(17)))
    assert code == 0 and np.all(tiny['exponent'] < -1074) and np.all(tiny['mantissa'] > 0)
    provider.keep[0][0] = -1200.
    code, result, report = frozen.cell(provider, loc, 1., 0., tiny)
    assert code == 0, report
    np.testing.assert_allclose(values(result), 1., rtol=4e-13)


def test_positive_u_cubic_reduces_smooth_frequency_interpolation_error(frozen):
    errors = []
    for h in (.02, .01, .005):
        _, provider, _, _ = manufactured(nf=129, eta=0., chi=0.)
        provider._source.dlognu = h; delta = 7.37*h
        exact = .4*math.exp(-3*delta); pair = []
        for reconstruction in (0, 1):
            code, result, report = frozen.cell(provider, Location(1, 1, 0, .3), 1., delta,
                positive(np.full(129, .4)), reconstruction=reconstruction)
            assert code == 0, report
            pair.append(abs(values(result)[32]/exact-1))
        assert pair[1] < .02*pair[0]
        errors.append(pair[1])
    assert all(.09 < b/a < .17 for a, b in zip(errors, errors[1:]))


@pytest.mark.parametrize('reconstruction', [0, 1])
def test_nonuniform_spectra_match_independent_ordered_bin_propagation(frozen, reconstruction):
    _, provider, _, _ = manufactured(nf=65, eta=0., chi=0.)
    nf = 65; chi = .2+.1*np.sin(np.arange(nf)*.7); eta = .3+.1*np.cos(np.arange(nf)*.3)
    provider.keep[0][0] = chi; provider.keep[0][2] = eta
    # 정수 bin 이동에서는 입사 보간이 없어 비가환 이동창 자체를 검증한다.
    h = provider._source.dlognu; delta = 8*h; incoming = positive(np.linspace(.2, .9, nf))
    code, result, report = frozen.cell(provider, Location(1, 1, 0, .3), 1., delta, incoming,
                                     reconstruction=reconstruction)
    assert code == 0, report
    expected = []
    for j in range(nf):
        top = min(j+8, nf-1); value = values(incoming)[top]
        pieces = []
        if j+8 > nf-1: pieces.append(((j+8-(nf-1))/8, chi[-1], chi[-1], eta[-1], eta[-1]))
        for k in range(top-1, j-1, -1): pieces.append((1/8, chi[k+1], chi[k], eta[k+1], eta[k]))
        for length, cb, cr, eb, er in pieces:
            tau = .5*(cb+cr)*length+3*delta*length
            t = math.exp(-tau); w0 = (1-(1+tau)*t)/tau**2; w1 = (tau-1+t)/tau**2
            value = t*value+length*(w0*eb+w1*er)
        expected.append(value)
    np.testing.assert_allclose(values(result), expected, rtol=5e-13)
    assert report['primitive_maps'] < 2*nf and report['compositions'] < 10*nf


def test_positive_cubic_has_no_negative_control_or_intensity_floor(frozen):
    _, provider, _, _ = manufactured(nf=129, eta=0., chi=0.)
    spectrum = np.tile([0., 1e-300, 0., 1., 2., 1., 0., 0.], 17)[:129]
    code, result, report = frozen.cell(provider, Location(1, 1, 0, .3), 1., .0037, positive(spectrum))
    assert code == 0, report
    assert np.all(result['mantissa'] >= 0) and np.all(np.isfinite(result['mantissa']))
    assert np.min(values(result)) == 0 and np.max(values(result)) <= 2.


def test_failed_call_preserves_output_and_shifted_range_uses_valid_blue_input(frozen):
    _, provider, _, _ = manufactured(nf=65)
    initial = positive(np.full(65, .3)); out = positive(np.full(65, 99.)); before = out.copy()
    code, out, report = frozen.cell(provider, Location(1, 1, 0, .3), 1., .02, initial,
                                   output=out, max_seconds=1e-12)
    assert code == 4 and report['published_outputs'] == 0
    np.testing.assert_array_equal(out, before)
    initial['mantissa'][:20] = -1.
    code, out, report = frozen.cell(provider, Location(1, 1, 0, .3), 1., .02, initial, output=out, last=20)
    assert code == 0, report
    np.testing.assert_array_equal(out[:20], before[:20])


def test_partial_interval_red_halo_matches_whole_grid(frozen):
    import time
    from compare_lumina_fullsr_sc_interval import frozen_interval
    _, provider, _, _ = manufactured(nf=65, chi=.4)
    provider.nf = 65
    provider.keep[0][2] = 1.+.3*np.sin(np.arange(65)*.7)
    n = 5; lengths = np.full((n, 3), 1/n); deltas = np.tile([.0037/n, .0037/(2*n), 0.], (n, 1))
    loc = (Location*(5*n))(*[Location(1, 1, 0, .3) for _ in range(5*n)])
    ray = positive(np.zeros(65))
    for k in range(n):
        code, ray, report = frozen.cell(provider, loc[5*k+2], 1/n, .0037/n, ray)
        assert code == 0, report
    code, actual, report = frozen_interval(frozen, provider, lengths, deltas, loc, 1,
                                          [20, 21, 22], time.monotonic()+120, 1)
    assert code == 0, report
    np.testing.assert_array_equal(actual, values(ray)[[20, 21, 22]])


@pytest.mark.parametrize('q0,q1', [(0., 1.), (3., 20.), (20., 3.), (-3., -20.),
    (-20., -3.), (-1000., 1000.), (1000., -1000.), (0., -1200.), (-1200., 0.),
    (1000., 1000.), (-1200., -1200.)])
def test_linear_opacity_source_matches_independent_gauss_integral(frozen, q0, q1):
    kernel = FrozenKernel(frozen.library._name, linear_chi=True)
    _, provider, _, _ = manufactured(nf=65, eta=0., chi=0.)
    h = provider._source.dlognu; delta = h
    provider.keep[0][0, :, 0] = q1-3*delta; provider.keep[0][0, :, 1] = q0-3*delta
    provider.keep[0][2, :, 0] = .8; provider.keep[0][2, :, 1] = .3
    code, result, report = kernel.cell(provider, Location(1, 1, 0, .3), 1., delta, positive(np.zeros(65)))
    assert code == 0, report
    x, weights = np.polynomial.legendre.leggauss(256); y = .5*(1+x)
    phase = -q1*y-.5*(q0-q1)*y*y
    points = [0., 1.]
    if q0 > 0 and q1 < 0: points.append(-q1/(q0-q1))
    peak = max(-q1*t-.5*(q0-q1)*t*t for t in points)
    normalized = .5*np.dot(weights, (.8*(1-y)+.3*y)*np.exp(phase-peak))
    log_exact = peak+math.log(normalized)
    log_actual = math.log(float(result[0]['mantissa']))+int(result[0]['exponent'])*math.log(2.)
    assert abs(log_actual-log_exact) < 5e-10


def test_linear_opacity_bin_subdivision_does_not_change_the_formal(frozen):
    kernel = FrozenKernel(frozen.library._name, linear_chi=True)
    _, provider, _, _ = manufactured(nf=65)
    provider.keep[0][0] = .2+5*np.cos(np.arange(65)*.37)
    provider.keep[0][2] = .3+.1*np.sin(np.arange(65)*.23)
    incoming = positive(.2+.1*np.cos(np.arange(65)*.17))
    outputs = []
    for n in (1, 4):
        code, result, report = kernel.cell(provider, Location(1, 1, 0, .3), 1., .0837,
                                         incoming, bin_subdivisions=n)
        assert code == 0, report
        outputs.append(values(result))
    np.testing.assert_allclose(*outputs, rtol=2e-11, atol=0.)


def test_linear_opacity_recovers_a_tiny_source_after_large_gain(frozen):
    kernel = FrozenKernel(frozen.library._name, linear_chi=True)
    _, provider, _, _ = manufactured(nf=17, eta=1e-300, chi=-1e53)
    code, result, report = kernel.cell(provider, Location(1, 1, 0, .3), 1e-50, 0., positive(np.zeros(17)))
    assert code == 0, report
    q = -1e53*1e-50
    log_exact = math.log(1e-300)+math.log(1e-50)-q-math.log(-q)
    log_actual = math.log(float(result[0]['mantissa']))+int(result[0]['exponent'])*math.log(2.)
    assert abs(log_actual-log_exact) < 5e-10
