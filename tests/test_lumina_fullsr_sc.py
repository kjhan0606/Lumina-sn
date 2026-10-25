"""독립 고정밀 적분·주파수 상태·실패 원자성·스레드 재현 시험."""
import ctypes as ct
from decimal import Decimal, localcontext
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from lumina_compiled_input import DenseView
from lumina_fullsr_sc import Kernel, Location, SCALED


class Positive(ct.Structure):
    _fields_ = [('mantissa', ct.c_double), ('exponent', ct.c_int64)]


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    path = tmp_path_factory.mktemp('fullsr-sc')/'sc.so'
    built = subprocess.run([os.environ.get('CXX', 'c++'), '-O3', '-std=c++17', '-ffp-contract=off',
        '-Wall', '-Wextra', '-Werror', '-fopenmp', '-fPIC', '-shared',
        str(ROOT/'src/cmf_fullsr_sc.cpp'), '-o', str(path)], capture_output=True, text=True, timeout=90)
    assert built.returncode == 0, built.stderr
    result = Kernel(path)
    result.step = result.library.cmf_fullsr_sc_step
    result.step.argtypes = [ct.c_double]*3+[ct.c_int]+[ct.c_double]*4+[ct.c_void_p]*2
    result.step.restype = ct.c_int
    return result


def scaled(x): return Positive(*math.frexp(x))


def decimal_positive(x): return Decimal(x.mantissa)*Decimal(2)**x.exponent


def call_step(kernel, chi, eta=(.3, .8), incident=.4, blue=(.7, .9), length=1., drift=0., h=.01, top=0):
    state = (Positive*3)(scaled(incident), scaled(blue[0]), scaled(blue[1]))
    result = Positive(.75, 99)
    code = kernel.step(length, drift, h, top, *chi, *eta, state, ct.byref(result))
    return code, result


@pytest.mark.parametrize('tau', [0., 1e-10, -1e-10, .099, -.099, .1, -.1, 1., -1., 1000., -1000.])
def test_signed_optical_depth_against_independent_decimal_integral(kernel, tau):
    code, actual = call_step(kernel, (tau, tau))
    assert code == 0
    with localcontext() as ctx:
        ctx.prec = 100
        t = Decimal(tau); transmission = (-t).exp()
        w0 = (1-(1+t)*transmission)/t**2 if tau else Decimal('.5')
        w1 = (t-1+transmission)/t**2 if tau else Decimal('.5')
        expected = transmission*Decimal(.4)+w0*Decimal(.3)+w1*Decimal(.8)
        assert abs(decimal_positive(actual)/expected-1) < Decimal('4e-13')


def test_zero_effective_opacity_and_invariant_vacuum(kernel):
    code, actual = call_step(kernel, (-1., -1.), eta=(.3, .8), drift=.01,
                             blue=(.7, .9), h=.01)
    expected = .4+.55+math.exp(-.03)*.8
    assert code == 0 and math.isclose(float(decimal_positive(actual)), expected, rel_tol=2e-15)
    # 일정 U=I/nu^3의 청색 상태를 넣으면 공간에 따라 I가 변하지 않는다.
    code, actual = call_step(kernel, (0., 0.), eta=(0., 0.), incident=.4,
                             blue=(.4*math.exp(.03),)*2, drift=3., h=.01)
    assert code == 0 and math.isclose(float(decimal_positive(actual)), .4, rel_tol=3e-15)


def test_underflow_and_gain_are_not_clipped(kernel):
    state = (Positive*3)(scaled(1.), scaled(0.), scaled(0.)); tiny = Positive(); recovered = Positive()
    assert kernel.step(1., 0., .01, 0, 1200., 1200., 0., 0., state, ct.byref(tiny)) == 0
    assert tiny.exponent < -1074 and tiny.mantissa > 0
    state[0] = tiny
    assert kernel.step(1., 0., .01, 0, -1200., -1200., 0., 0., state, ct.byref(recovered)) == 0
    assert math.isclose(float(decimal_positive(recovered)), 1., rel_tol=4e-13)
    code, actual = call_step(kernel, (-1e53,)*2, eta=(1e-300,)*2,
                             incident=0., blue=(0., 0.), length=1e-50)
    assert code == 0 and actual.mantissa > 0
    with localcontext() as ctx:
        ctx.prec = 80
        tau = Decimal(-1e53)*Decimal(1e-50)
        expected = Decimal(1e-300)*Decimal(1e-50)*(1-(-tau).exp())/tau
        assert abs(decimal_positive(actual)/expected-1) < Decimal('1e-12')


def test_spatial_sc_refines_to_independent_variable_opacity_integral(kernel):
    nodes, weights = np.polynomial.legendre.leggauss(64)
    s = .5*(nodes+1.)
    # 최고주파수 일반식의 연속해. chi=.2+1.4s, eta=.3+.7s, 전체 drift=.09.
    attenuation = np.exp(-.2*(1-s)-.7*(1-s*s)-.27*(1-s))
    exact = .4*math.exp(-1.17)+.5*np.dot(weights, (.3+.7*s)*attenuation)
    errors = []
    for n in (4, 8, 16, 32):
        state = (Positive*3)(scaled(.4), scaled(0.), scaled(0.)); output = Positive()
        for k in range(n):
            a, b = k/n, (k+1)/n
            assert kernel.step(1/n, .09/n, .01, 1, .2+1.4*a, .2+1.4*b,
                               .3+.7*a, .3+.7*b, state, ct.byref(output)) == 0
            state[0] = output
        errors.append(abs(float(decimal_positive(output))-exact))
    assert all(.23 < b/a < .27 for a, b in zip(errors, errors[1:]))


def test_frequency_upwind_refines_to_fullsr_vacuum_solution(kernel):
    plan, provider, bc, grad = manufactured(nf=65, drift=0., eta=0., chi=0.)
    n = 128; nv = n+2
    plan.npath, plan.nvertex, plan.nevent = 1, nv, 1
    events = np.zeros(nv+1, dtype='i4'); events[-1] = 1
    plan.arrays = dict(path_ptr=np.array([0, nv], dtype='i4'),
        kind=np.array([1, 2]+[0]*n, dtype='i4'),
        length=np.array([0., 0.]+[1/n]*n), drift=np.array([0., 0.]+[.02/n]*n),
        core_mu=np.zeros(nv), event_ptr=events, event_node=np.array([0], dtype='i4'),
        event_weights=np.array([[1., 0., .25]]))
    plan.locations = (Location*nv)(*[Location(1, 1, 0, .5) for _ in range(nv)])
    bc[:] = .7; grad[:] = 0.
    errors = []
    for subdivisions in (1, 2, 4):
        _, _, escape, _ = solve(kernel, plan, provider, bc, grad, subdiv=subdivisions)
        errors.append(abs(escape[0, 0]-.7*math.exp(-.06)))
    assert all(.48 < b/a < .52 for a, b in zip(errors, errors[1:]))


@pytest.mark.parametrize('bad', ['eta', 'drift', 'length', 'incident'])
def test_bad_step_preserves_output(kernel, bad):
    state = (Positive*3)(scaled(.5), scaled(0.), scaled(0.)); result = Positive(.75, 99)
    eta, drift, length = 1., 0., 1.
    if bad == 'eta': eta = -1.
    if bad == 'drift': drift = -1.
    if bad == 'length': length = 0.
    if bad == 'incident': state[0].mantissa = -.5
    before = bytes(result)
    assert kernel.step(length, drift, .01, 0, 1., 1., eta, eta, state, ct.byref(result)) != 0
    assert bytes(result) == before


def manufactured(nf=33, drift=.02, eta=.3, chi=.4):
    ns, ne, npath = 2, 2, 2
    canonical = np.zeros((3, ns, nf)); canonical[0] = chi; canonical[1] = .01; canonical[2] = eta
    outer = np.zeros((nf, 2*ne+1)); j = np.zeros((ns, nf)); radius = np.array([2., 1.])
    arrays = canonical, outer, j, radius, radius
    ptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_double))
    view = DenseView(ns, nf, ne, 0, 1, *[ptr(a) for a in arrays])
    provider = SimpleNamespace(_source=SimpleNamespace(dlognu=.01), borrow_dense_view=lambda: view)
    # 외부 영입사 -> 공간 -> 별도 core 경계 -> 공간. core 양쪽 청색 상태는 별개다.
    a = dict(path_ptr=np.array([0, 4, 8], dtype='i4'), kind=np.array([1, 0, 2, 0]*2, dtype='i4'),
        length=np.array([0., .7, 0., 1.]*2), drift=np.array([0., drift, 0., drift]*2),
        core_mu=np.array([0., 0., .5, 0.]*2), event_ptr=np.arange(9, dtype='i4'),
        event_node=np.array([0, 0, 1, 1]*2, dtype='i4'),
        event_weights=np.tile(np.array([[.25, -.1, .05]]), (8, 1)))
    plan = SimpleNamespace(provider=provider, npath=npath, nvertex=8, nevent=8, arrays=a,
        locations=(Location*8)(*[Location(1, 1, 0, .5) for _ in range(8)]))
    # 배열 수명은 시험이 끝날 때까지 유지한다.
    provider.keep = arrays
    return plan, provider, np.full(nf, .7), np.full(nf, .2)


def solve(kernel, plan, provider, bc, grad, tile=7, threads=1, subdiv=1):
    nf = len(bc); blue = np.zeros(plan.nvertex, dtype=SCALED)
    moments = np.full((3, 2, nf), -99.); escape = np.full((plan.npath, nf), -99.)
    code, report = kernel.advance(plan, provider, bc, grad, blue, moments, escape=escape,
        tile_width=tile, threads=threads, frequency_subdivisions=subdiv)
    assert code == 0, report
    return blue, moments, escape, report


@pytest.mark.parametrize('subdiv', [1, 3])
def test_tile_and_thread_boundaries_preserve_full_blue_solution(kernel, subdiv):
    args = manufactured()
    first = solve(kernel, *args, tile=33, subdiv=subdiv)
    for tile, threads in ((1, 1), (7, 2), (16, 2)):
        value = solve(kernel, *args, tile=tile, threads=threads, subdiv=subdiv)
        for a, b in zip(value[:3], first[:3]): np.testing.assert_array_equal(a, b)
    assert first[3]['completed_frequencies'] == 33 and first[3]['next_frequency'] == -1


def test_static_grey_solution_core_reset_and_tiny_blue_top_source(kernel):
    plan, provider, bc, grad = manufactured(drift=0.)
    _, moments, escape, _ = solve(kernel, plan, provider, bc, grad)
    expected = .8*math.exp(-.4)+.3*(-math.expm1(-.4))/.4
    np.testing.assert_allclose(escape, expected, rtol=2e-15)
    assert np.all(moments[0] >= 0) and np.all(moments[2] >= 0) and np.all(moments[1] <= 0)
    plan, provider, bc, grad = manufactured(eta=1e-290, chi=0.)
    bc[:] = 0.; grad[:] = 0.
    _, _, escape, _ = solve(kernel, plan, provider, bc, grad)
    expected = 1e-290*(-math.expm1(-.06))/.06
    assert np.all(escape[:, -1] > 0)
    np.testing.assert_allclose(escape[:, -1], expected, rtol=2e-15, atol=0.)


def test_partial_failure_and_resume_match_fresh_full_solution(kernel):
    plan, provider, bc, grad = manufactured(nf=17)
    good = solve(kernel, plan, provider, bc, grad)
    blue = np.zeros(plan.nvertex, dtype=SCALED); moments = np.full((3, 2, 17), -99.)
    escape = np.full((plan.npath, 17), -99.)
    provider.keep[0][2, :, 8] = -1.
    code, report = kernel.advance(plan, provider, bc, grad, blue, moments, escape=escape, tile_width=4, threads=2)
    assert code == 2 and report['completed_frequencies'] == 8 and report['next_frequency'] == 8
    assert np.all(moments[:, :, :9] == -99.) and np.all(escape[:, :9] == -99.)
    provider.keep[0][2, :, 8] = .3
    code, report = kernel.advance(plan, provider, bc, grad, blue, moments, escape=escape,
                                 first=8, tile_width=3, threads=2)
    assert code == 0
    for a, b in zip((blue, moments, escape), good[:3]): np.testing.assert_array_equal(a, b)


def test_deadline_does_not_publish_a_partial_tile(kernel):
    plan, provider, bc, grad = manufactured()
    blue = np.zeros(plan.nvertex, dtype=SCALED); moments = np.full((3, 2, 33), -99.)
    code, report = kernel.advance(plan, provider, bc, grad, blue, moments, max_seconds=1e-12)
    assert code == 4 and report['completed_frequencies'] == 0
    assert np.all(moments == -99.) and np.all(blue['mantissa'] == 0)


@pytest.fixture(scope='module')
def geometry_library(tmp_path_factory):
    path = tmp_path_factory.mktemp('fullsr-geometry')/'geometry.so'
    built = subprocess.run([os.environ.get('CC', 'cc'), '-O2', '-std=c11', '-ffp-contract=off',
        '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared', *[str(ROOT/'src'/name) for name in
        ('cmf_frequency_history.c', 'cmf_nodal_input.c', 'cmf_nodal_history.c', 'cmf_nodal_driver.c')],
        '-lm', '-o', str(path)], capture_output=True, text=True, timeout=90)
    assert built.returncode == 0, built.stderr
    return path


def test_original_geometry_events_and_new_frozen_source_are_connected(kernel, geometry_library, tmp_path):
    from lumina_compiled_input import CompiledInput
    from lumina_fullsr_sc import SCPlan
    from lumina_local_micro import ZeroTrial
    from lumina_nodal_driver import CompiledDriverPlan
    from test_lumina_compiled_input import packet
    g, source, _ = packet(tmp_path, physical=True)
    with source, CompiledInput(geometry_library, g, source, ZeroTrial(source),
                               slab_width=8, profiles=128) as provider:
        provider.bind_zero_trial(); provider.prepare_dense()
        original = CompiledDriverPlan(g, source, panels=1, block_width=8)
        plan = SCPlan(original, provider, geometry_library, spatial_subdivisions=2)
        a = plan.arrays
        assert plan.nvertex == 2*original.half_segments+plan.npath+int(original.arrays['paths'][:, 4].sum())
        assert a['length'].sum() > 0 and np.all(a['drift'] >= 0)
        expected_weights = original.arrays['event_weights'].sum(axis=0)+original.arrays['core_weights'].sum(axis=0)
        np.testing.assert_allclose(a['event_weights'].sum(axis=0), expected_weights, rtol=2e-15, atol=1e-14)
        bc = np.full(source.nf, 1e-4); grad = np.zeros(source.nf)
        blue = np.zeros(plan.nvertex, dtype=SCALED); moments = np.full((3, source.ns, source.nf), -99.)
        code, report = kernel.advance(plan, provider, bc, grad, blue, moments, tile_width=3, threads=2)
        assert code == 0, report
        assert np.all(np.isfinite(moments)) and np.all(moments[[0, 2]] >= 0)
        assert np.all(moments[1]**2 <= moments[0]*moments[2]*(1+2e-14))
        # 제공자를 새로 결합하면 정적 계수는 보존하고 새 산란 source를 실제로 소비한다.
        class Field:
            ns, nf, radius, frequency, closed = source.ns, source.nf, source.radius, source.frequency, False
            values = np.full((source.ns, source.nf), 1e-3)
            identity = dict(role='SYNTHETIC_OWN_FIELD')
        field = Field(); field.values.flags.writeable = False
        provider._field = field; provider.bind_view(); provider.prepare_dense()
        second = np.empty_like(moments); blue[:] = 0
        code, report = kernel.advance(plan, provider, bc, grad, blue, second, tile_width=8, threads=1)
        assert code == 0, report
        assert np.max(second[0]-moments[0]) > 0
