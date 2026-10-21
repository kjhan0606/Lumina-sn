"""B native provider·구간 합성·공간 오차 추정을 독립식과 대조한다."""
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
from lumina_linear_native import CPositive, Location, Map, NativeKernel, NativeMap, Positive, SpatialFailure, spatial_interval
from test_lumina_characteristic_linear import reference


class View(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in ('ns', 'nf', 'ne', 'jmode')]+[('generation', ct.c_uint64)]+[
        (name, ct.POINTER(ct.c_double)) for name in ('canonical', 'outer', 'j', 'radius', 'external_radius')]


@pytest.fixture(scope='module',params=['B','SOURCE_MEASURE'])
def native(tmp_path_factory,request):
    path = tmp_path_factory.mktemp('linear-native')/'native.so'
    source='cmf_measure_native.cpp' if request.param=='SOURCE_MEASURE' else 'cmf_linear_native.cpp'
    proc = subprocess.run([os.environ.get('CXX', 'c++'), '-O2', '-std=c++17', '-ffp-contract=off',
        '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared', '-I'+str(ROOT/'src'), str(ROOT/'src'/source),
        str(ROOT/'tests/linear_native_probe.cpp'), '-o', str(path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    kernel = NativeKernel(path,source_measure=request.param=='SOURCE_MEASURE')
    kernel.lib.lumina_linear_provider_probe.argtypes = [ct.POINTER(View), ct.POINTER(Location), ct.c_int,
        ct.c_int, ct.POINTER(ct.c_double)]
    kernel.lib.lumina_linear_provider_probe.restype = ct.c_int
    return kernel


def inputs(nf=65, jmode=1):
    rng = np.random.default_rng(102204); ns, ne = 4, 3
    canonical = np.ascontiguousarray(np.stack([rng.uniform(-.2, .5, (ns, nf)),
        rng.uniform(.2, 2., (ns, nf)), rng.uniform(.01, 1., (ns, nf))]))
    j = np.ascontiguousarray(rng.uniform(.01, .8, (ns, nf)))
    outer = np.zeros((nf, 2*ne+1)); outer[:, :ne] = np.log([.1, .2, .3])
    outer[:, ne:2*ne] = np.log([.03, .08, .11]); outer[:, 2*ne] = -1.
    radius = np.array([4., 3., 2., 1.]); external = np.array([6., 5., 4.])
    arrays = canonical, outer, j, radius, external
    ptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_double))
    view = View(ns, nf, ne, jmode, 7, *[ptr(a) for a in arrays])
    return view, arrays


def endpoint(view, arrays, loc, column):
    c, outer, j, _, _ = arrays; f, l, w = loc.first, loc.last, loc.weight
    if loc.mode == 2:
        v = outer[column]
        return v[2*view.ne]*np.exp((1-w)*v[f]+w*v[l]), np.exp((1-w)*v[view.ne+f]+w*v[view.ne+l])
    jm = j if view.jmode else np.zeros_like(j)
    if loc.mode == 0: return c[0, f, column], c[2, f, column]+c[1, f, column]*jm[f, column]
    return (1-w)*c[0, f, column]+w*c[0, l, column], (1-w)*c[2, f, column]+w*c[2, l, column]+\
        ((1-w)*c[1, f, column]+w*c[1, l, column])*((1-w)*jm[f, column]+w*jm[l, column])


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('jmode', [0, 1])
def test_native_assembly_matches_original_order_before_frequency_interpolation(native, mode, jmode):
    view, arrays = inputs(jmode=jmode); loc = Location(mode, 3 if mode == 0 else 2, 3 if mode == 0 else 1, 0. if mode == 0 else .37)
    locations = (Location*2)(loc, loc)
    for column in (0, 17, 64):
        output = (ct.c_double*2)(-17., -17.)
        assert native.lib.lumina_linear_provider_probe(ct.byref(view), locations, 0, column, output) == 0
        np.testing.assert_allclose(list(output), endpoint(view, arrays, loc, column), rtol=3e-15, atol=1e-15)
    if mode == 1 and jmode:
        c, _, j, _, _ = arrays; f, l, w = loc.first, loc.last, loc.weight
        wrong = (1-w)*(c[2, f]+c[1, f]*j[f])+w*(c[2, l]+c[1, l]*j[l])
        correct = np.array([endpoint(view, arrays, loc, x)[1] for x in range(view.nf)])
        assert np.max(abs(wrong-correct)) > .01  # 잘못된 총η 선보간과 구분한다.


@pytest.mark.parametrize('column,shift,span', [(0, .37, 14.3), (31, 3.125, 17.4), (63, .37, 8.1), (64, .25, 3.), (17, .375, 0.)])
def test_native_window_map_matches_independent_bin_integral(native, column, shift, span):
    view, arrays = inputs(); locs = [Location(1, 3, 2, .23), Location(1, 2, 1, .72)]
    h = .001; L = 1.1; da, db = (shift+span)*h, shift*h
    result, detail = native.map(view, locs, L, h, da, db, column, relative=1e-7)
    ca, ea = np.array([endpoint(view, arrays, locs[0], i) for i in range(view.nf)]).T
    cb, eb = np.array([endpoint(view, arrays, locs[1], i) for i in range(view.nf)]).T
    values = np.array([ca, cb, ea, eb]); expected = reference(values, length=L, shift=column+shift, span=span, delta=da-db)
    assert abs(result.source.value()-expected) <= result.source_error.value()+2e-13*expected
    assert detail.built_bins <= 64
    incident = Positive.of(.31); actual, error = result.apply(incident)
    expected = reference(values, length=L, shift=column+shift, span=span, delta=da-db, incident=.31)
    assert abs(actual.value()-expected) <= error.value()+2e-13*expected


@pytest.mark.parametrize('bad', ['generation', 'location', 'column', 'eta', 'ES', 'budget'])
def test_native_failures_preserve_output_canary(native, bad):
    view, arrays = inputs(); locs = (Location*2)(Location(1, 3, 2, .3), Location(1, 2, 1, .7))
    generation, column, relative = 7, 3, 1e-7
    if bad == 'generation': generation = 8
    elif bad == 'location': locs[1].first = 19
    elif bad == 'column': column = 65
    elif bad == 'eta': arrays[0][2, :, :] = -3.
    elif bad == 'ES': arrays[0][1, :, :] = 0.; view.jmode = 0
    else: relative = 1e-18
    output = NativeMap(); ct.memset(ct.byref(output), 0xA5, ct.sizeof(output)); before = bytes(output)
    code = native.fun(ct.byref(view), generation, locs, 1., .001, .014, .001, column, 32, relative, 100000, ct.byref(output))
    assert code != 0 and bytes(output) == before


def test_scaled_positive_composition_keeps_recoverable_underflow():
    tiny = Positive(.75, -1600); huge = Positive(.75, 1600)
    assert (tiny*huge).value() == .5625
    a = Map(tiny, Positive(), Positive(), 0.); b = Map(huge, Positive(), Positive(), 0.)
    value, error = a.then(b).apply(Positive.of(2.))
    assert value.value() == 1.125 and error.value() > 0
    assert CPositive(.5, 1).positive().value() == 1.


def test_spatial_adaptation_distinguishes_nonlinear_geometry_from_local_integration_error():
    def maps(a, b):
        def part(x, y):
            # η(s)=1+8s²의 endpoint 공간선형 근사. χ=0은 정확한 정적 적분이다.
            source = (y-x)*((1+8*x*x)+(1+8*y*y))/2
            return Map(Positive.of(1.), Positive.of(source), Positive.of(source*1e-12), 1e-14)
        m = .5*(a+b); return part(a, b), part(a, m), part(m, b)
    value, report = spatial_interval(maps, relative=1e-5)
    assert abs(value.value()-(1+8/3)) <= report['estimated_total_error']
    assert report['numerical_error'] < report['geometry_estimate'] and report['leaves'] > 1
    assert not report['error_certificate']


def test_spatial_failed_completion_does_not_return_a_partial_field():
    def maps(a, b):
        zero = Map(Positive.of(1.), Positive(), Positive(), 0.)
        source = Map(Positive.of(1.), Positive.of(1.), Positive(), 0.)
        return zero, source, source
    with pytest.raises(SpatialFailure) as e: spatial_interval(maps, max_leaves=1)
    assert e.value.last_completed['leaves'] == 1


@pytest.mark.parametrize('bad', [math.nan, 0., -1., .1])
def test_spatial_invalid_relative_budget_is_rejected(bad):
    with pytest.raises(ValueError): spatial_interval(lambda a, b: None, relative=bad)
