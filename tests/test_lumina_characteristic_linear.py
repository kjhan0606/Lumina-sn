"""공간×주파수 재구성 자체를 독립 bin별 적분/고정밀 극한과 비교한다."""
import ctypes as ct
from decimal import Decimal, localcontext
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def characteristic(tmp_path_factory):
    path = tmp_path_factory.mktemp('characteristic-linear')/'linear.so'
    proc = subprocess.run([os.environ.get('CXX', 'c++'), '-O2', '-std=c++17', '-ffp-contract=off',
        '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared', '-I'+str(ROOT/'src'),
        str(ROOT/'tests/characteristic_linear_probe.cpp'), '-o', str(path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    f = ct.CDLL(str(path)).lumina_characteristic_linear_probe
    p = np.ctypeslib.ndpointer(dtype='f8', flags='C_CONTIGUOUS')
    e = np.ctypeslib.ndpointer(dtype='i8', flags='C_CONTIGUOUS')
    i = np.ctypeslib.ndpointer(dtype='i4', flags='C_CONTIGUOUS')
    f.argtypes = [ct.c_int, ct.c_int, p, p, ct.c_int, p, e, i]; f.restype = ct.c_int
    return f


def run(fun, values, *, length=1., shift=0., span=0., delta=0., incident=0., relative=1e-7, block=4, limit=100000):
    values = np.ascontiguousarray(values, dtype='f8')
    parameters = np.array([length, shift, span, delta, incident, relative], dtype='f8')
    m = np.full(7, -17.); e = np.full(7, -17, dtype='i8'); counts = np.full(5, -17, dtype='i4')
    code = fun(values.shape[1], block, values, parameters, limit, m, e, counts)
    return code, m, e, counts


def decoded(result):
    code, m, e, counts = result
    assert code == 0, (code, counts)
    return np.ldexp(m, e.astype('i4'))


def reference(values, *, length=1., shift=0., span=0., delta=0., incident=0., order=96):
    """네이티브 bin마다 χ quadratic을 독립 구성하고 Gauss96으로 ηexp(-τ)를 적분한다."""
    nf = values.shape[1]
    edges = [0., 1.]
    if span:
        edges += [float(u) for u in (np.arange(nf)-shift)/span if 0. < u < 1.]
    edges = np.unique(edges)
    xg, wg = np.polynomial.legendre.leggauss(order); t = .5*(1+xg); wg = .5*wg
    def field(u, k):
        x = shift+span*np.asarray(u)
        a = np.interp(x, np.arange(nf), values[k]); b = np.interp(x, np.arange(nf), values[k+1])
        return np.asarray(u)*a+(1-np.asarray(u))*b
    tau = 0.; source = 0.
    for a, b in zip(edges[:-1], edges[1:]):
        du = b-a; H = length*du
        k0, km, k1 = [float(field(u, 0)) for u in (a, .5*(a+b), b)]
        c2 = 2.*(k1-2*km+k0); c1 = k1-k0-c2
        depth = tau+H*(k0*t+c1*t*t/2+c2*t*t*t/3)+3*delta*du*t
        source += H*np.sum(wg*field(a+du*t, 2)*np.exp(-depth))
        tau += H*(k0+c1/2+c2/3)+3*delta*du
    return incident*np.exp(-tau)+source


@pytest.mark.parametrize('block', [1, 4, 32])
def test_signed_space_frequency_cross_term_blue_and_fractional_endpoints(characteristic, block):
    rng = np.random.default_rng(102201)
    for shift, span in [(.217, 0.), (.217, 7.45), (2.31, 18.7), (19., 3.)]:
        values = np.vstack([rng.uniform(-.8, 1., (2, 17)), rng.uniform(.01, 2., (2, 17))])
        args = dict(length=1.7, shift=shift, span=span, delta=.021, incident=.13)
        result = run(characteristic, values, block=block, **args); fields = decoded(result)
        expected = reference(values, **args)
        assert fields[0] >= 0 and fields[3] <= 1e-7*fields[0]
        assert abs(fields[0]-expected) <= fields[3]+2e-13*abs(expected)
        assert fields[3] >= sum(fields[4:])*(1-3e-14)
        assert result[3][0] < 12000


@pytest.mark.parametrize('chi', [0., 1., 1000., -30.])
def test_static_strong_absorption_gain_and_fullsr_term(characteristic, chi):
    values = np.array([[chi]*9, [chi]*9, [2.]*9, [2.]*9])
    result = run(characteristic, values, span=6.8, shift=.37, delta=.013, incident=.7)
    fields = decoded(result)
    with localcontext() as context:
        context.prec = 90; D = Decimal.from_float; optical = D(chi)+3*D(.013)
        trans = (-optical).exp(); expected = D(.7)*trans+D(2.)*(1-trans)/optical
        actual = D(result[1][0])*Decimal(2)**int(result[2][0])
        error = D(result[1][3])*Decimal(2)**int(result[2][3])
        assert abs(actual-expected) <= error
    assert fields[3] <= 1e-7*fields[0]
    if chi == 1000.:
        assert result[3][3] > 0  # 제거한 꼬리는 전량 오차에 포함한다.
        assert result[3][0] < 250


def test_narrow_emission_is_not_lost_when_whole_panel_endpoints_are_zero(characteristic):
    values = np.zeros((4, 129)); values[2:, 79] = [3., 7.]
    result = run(characteristic, values, shift=.125, span=126.5, length=2.1, relative=1e-6)
    fields = decoded(result); expected = reference(values, shift=.125, span=126.5, length=2.1)
    assert fields[0] > 0 and abs(fields[0]-expected) <= fields[3]+2e-14*expected
    assert abs(fields[0]/expected-1.) < .01
    assert result[3][1] > 0 and result[3][2] > 0


def test_deep_absorber_does_not_destroy_exit_side_narrow_emission_budget(characteristic):
    values = np.zeros((4, 129)); values[2:, 79] = [3., 7.]; values[:2, 110] = 1e6
    # backward u에서 방출선 뒤의 더 깊은 흡수선은 이 원천을 감쇠시키지 않는다.
    result = run(characteristic, values, shift=.125, span=126.5, length=2.1, relative=1e-6)
    fields = decoded(result)
    values[:2] = 0.; expected = reference(values, shift=.125, span=126.5, length=2.1)
    assert fields[0] > 0 and abs(fields[0]-expected) <= fields[3]+2e-14*expected
    assert fields[3] <= 1e-6*fields[0]


def test_cumulative_tau_roundoff_uses_output_budget_not_an_infinitesimal_panel_budget(characteristic):
    values = np.ones((4, 129)); values[:2] = 4000.*(-1.)**np.arange(129)
    args = dict(shift=.125, span=126.5, length=1.3)
    result = run(characteristic, values, relative=1e-8, **args); fields = decoded(result)
    expected = reference(values, **args)
    assert abs(fields[0]-expected) <= fields[3]+2e-12*expected
    assert fields[3] <= 1e-8*fields[0] and result[3][4] < 50


def test_narrow_absorption_in_front_of_constant_source(characteristic):
    values = np.zeros((4, 65)); values[2:] = 1.; values[:2, 3] = [120., 80.]
    args = dict(shift=.125, span=63.5, length=1.3, incident=.2)
    result = run(characteristic, values, **args); fields = decoded(result); expected = reference(values, **args)
    assert abs(fields[0]-expected) <= fields[3]+2e-13*expected
    assert abs(fields[0]/expected-1.) < .01


def test_weak_source_is_not_scaled_by_a_bright_unrelated_frequency(characteristic):
    rng = np.random.default_rng(102202)
    values = np.vstack([rng.uniform(-.3, .7, (2, 9)), rng.uniform(.2, 2., (2, 9))])
    args = dict(shift=.37, span=6.9, delta=.018, length=1.2)
    bright = decoded(run(characteristic, values, **args)); values[2:] *= 1e-80
    weak = decoded(run(characteristic, values, **args))
    expected = reference(values, **args)
    assert weak[0] > 0 and weak[3] <= 1e-7*weak[0]
    assert abs(weak[0]-expected) <= weak[3]+2e-13*expected
    assert abs(weak[0]/(1e-80*bright[0])-1.) < 2e-7


@pytest.mark.parametrize('length,eta,chi', [(1e-200, 1e-200, -500e200), (1e300, 1e300, 1000e-300),
    (1e300, float(np.nextafter(0., 1.)), -500e-300), (1., 1., -1000.)])
def test_scaled_source_and_transmission_do_not_require_ieee_observable_range(characteristic, length, eta, chi):
    values = np.array([[chi]*3, [chi]*3, [eta]*3, [eta]*3])
    result = run(characteristic, values, length=length, span=1.25, relative=1e-6)
    assert result[0] == 0, result[0]
    with localcontext() as context:
        context.prec = 100; D = Decimal.from_float
        depth = D(length)*D(chi); expected = D(length)*D(eta)*(1-(-depth).exp())/depth
        actual = D(result[1][0])*Decimal(2)**int(result[2][0])
        error = D(result[1][3])*Decimal(2)**int(result[2][3])
        assert actual > 0 and abs(actual-expected) <= error
        assert error <= Decimal('1e-6')*actual


def test_true_zero_source_and_incident_remain_exact_zero(characteristic):
    values = np.zeros((4, 9)); values[:2] = -1.
    result = run(characteristic, values, shift=.35, span=7.3)
    np.testing.assert_array_equal(decoded(result), np.zeros(7))


def test_zero_opacity_linear_space_source_has_the_exact_static_limit(characteristic):
    values = np.zeros((4, 7)); values[2] = 2.; values[3] = 6.
    result = run(characteristic, values, span=0., shift=.375, length=1.75, incident=.25)
    fields = decoded(result)
    assert abs(fields[0]-7.25) <= fields[3] and fields[3] < 1e-7*7.25
    assert result[3][0] == 1 and result[3][1] == 1


def test_independent_randomized_opacity_source_and_offset_cases(characteristic):
    rng = np.random.default_rng(102203)
    for _ in range(40):
        nf = int(rng.integers(4, 23))
        values = np.vstack([rng.uniform(-2., 3., (2, nf)), rng.uniform(0., 3., (2, nf))])
        args = dict(length=float(rng.uniform(.2, 2.)), shift=float(rng.uniform(0., nf)),
            span=float(rng.uniform(0., 1.5*nf)), delta=float(rng.uniform(0., .03)),
            incident=float(rng.uniform(0., 1.)))
        result = run(characteristic, values, **args); fields = decoded(result); expected = reference(values, **args)
        assert abs(fields[0]-expected) <= fields[3]+2e-13*abs(expected)
        assert fields[3] <= 1e-7*fields[0]


@pytest.mark.parametrize('bad', ['eta', 'length', 'span', 'tolerance', 'work', 'subnormal_variation'])
def test_failed_inputs_or_insufficient_work_do_not_publish_partial_results(characteristic, bad):
    values = np.ones((4, 17)); args = {}
    if bad == 'eta': values[2, 5] = -1.
    elif bad == 'length': args['length'] = -1.
    elif bad == 'span': args['span'] = -1.
    elif bad == 'tolerance': args['relative'] = np.nan
    elif bad == 'work': values[0:2] = 1000.; args['limit'] = 1
    else: values[2:] = np.nextafter(0., 1.); values[2:, 0] = 0.; args['span'] = .75
    code, m, e, counts = run(characteristic, values, **args)
    assert code != 0
    np.testing.assert_array_equal(m, np.full(7, -17.)); np.testing.assert_array_equal(e, np.full(7, -17))
    np.testing.assert_array_equal(counts, np.full(5, -17))
