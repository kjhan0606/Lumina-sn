"""관측자 변환/외곽 광선/단위의 독립 해석 시험. 실제 모델 수용은 아니다."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import lumina_observer_spectrum as O


def assemble(nu, p, core, radius, doppler, intensities):
    spectrum = O.ObserverSpectrum(nu, p, np.asarray(core, dtype=bool), radius, doppler)
    for path, intensity in enumerate(intensities):
        spectrum.add(path, intensity)
    return spectrum.finish()


def test_constant_disk_has_absolute_luminosity_not_flux_or_comoving_H():
    nu = np.geomspace(1e14, 1e16, 9); radius = 3e15; intensity = 7e-8
    result = assemble(nu, np.array([0., .5, 1.])*radius, [True, False, False], radius, np.ones(3),
                      np.full((3, 9), intensity))
    np.testing.assert_allclose(result['luminosity_erg_s_Hz'], 4*math.pi**2*radius**2*intensity, rtol=3e-16)
    assert result['frame'] == 'OBSERVER_LAB' and not result['accepted_physical_spectrum']
    assert result['final_goal'] == 'NOT_MET' and not result['error_certificate']


def test_core_jump_uses_inner_limit_then_atmospheric_limit_despite_execution_order():
    nu = np.geomspace(1e14, 1e16, 8); radius = 4.
    # 실제 내부 각도 계획의 실행 순서는 같은 p에서 대기→core다.
    p = np.array([0., .25, .5, .5, .8, 1.])*radius
    core = np.array([True, True, False, True, False, False])
    values = np.where(core, 3., .7)
    result = assemble(nu, p, core, radius, np.ones(6), np.repeat(values[:, None], 8, axis=1))
    wanted = 4*math.pi**2*radius**2*(3.*.25+.7*.75)
    np.testing.assert_allclose(result['luminosity_erg_s_Hz'], wanted, rtol=3e-16)


def test_lab_angular_linear_disk_converges_to_analytic_integral():
    nu = np.geomspace(1e14, 1e16, 8); errors = []
    for n in (17, 65, 257, 1025):
        q = np.linspace(0., 1., n); values = 2.+.7*np.sqrt(1.-q)
        result = assemble(nu, np.sqrt(q), np.zeros(n, dtype=bool), 1., np.ones(n),
                          np.repeat(values[:, None], 8, axis=1))
        actual = result['luminosity_erg_s_Hz'][0]/(4*math.pi**2)
        errors.append(abs(actual/(2.+2*.7/3)-1))
    assert errors[-1] < 1e-5 and errors[-1] < errors[0]/100


@pytest.mark.parametrize('alpha', [0, 3])
def test_relativistic_frequency_and_cubic_intensity_transform(alpha):
    beta = .2; p = np.array([0., math.sqrt(.75), 1.]); mu = np.sqrt(1-p*p)
    d = (1-beta*mu)/math.sqrt((1-beta)*(1+beta)); errors = []
    weights = O.disk_weights(p, np.zeros(3, dtype=bool), 1.)
    for n in (2049, 8193):
        nu = np.geomspace(1e14, 1e16, n); intensity = (nu/nu[0])**alpha
        result = assemble(nu, p, np.zeros(3, dtype=bool), 1., d, np.repeat(intensity[None, :], 3, axis=0))
        wanted = 4*math.pi**2*(result['frequency_Hz']/nu[0])**alpha*np.dot(weights, d**(alpha-3))
        errors.append(float(np.max(abs(result['luminosity_erg_s_Hz']/wanted-1))))
        assert result['frequency_Hz'][0]*d.min() >= nu[0]
        assert result['frequency_Hz'][-1]*d.max() <= nu[-1]
    if alpha == 0:
        assert max(errors) < 1e-15
    else:
        assert errors[-1] < 5e-7 and errors[-1] < errors[0]/10


def test_angstrom_jacobian_and_band_integral_preserve_energy():
    nu = np.geomspace(1e14, 1e16, 8193)
    result = assemble(nu, [0., 1.], [True, False], 1., np.ones(2), np.ones((2, len(nu))))
    lnu = result['luminosity_erg_s_Hz']; lam = result['wavelength_A']; llam = result['luminosity_erg_s_A']
    np.testing.assert_allclose(llam*lam*lam/(O.A.C*1e8), lnu, rtol=8*np.finfo('f8').eps)
    a = np.sum(.5*(lnu[1:]+lnu[:-1])*np.diff(nu))
    b = np.sum(-.5*(llam[1:]+llam[:-1])*np.diff(lam))
    assert abs(a/b-1) < 2e-7


@pytest.mark.parametrize('p,core', [([.1, 1.], [False, False]), ([0., .7], [True, False]),
    ([0., .5, .5, 1.], [True, False, False, False])])
def test_missing_disk_or_invalid_duplicate_is_rejected(p, core):
    with pytest.raises(ValueError):
        O.disk_weights(p, np.asarray(core), 1.)


def test_incomplete_bad_or_repeated_ray_does_not_silently_publish():
    nu = np.geomspace(1e14, 1e16, 8)
    s = O.ObserverSpectrum(nu, [0., 1.], np.array([True, False]), 1., [1., 1.])
    with pytest.raises(ValueError, match='미완료'):
        s.finish()
    s.add(0, np.ones(8))
    with pytest.raises(ValueError, match='순서'):
        s.add(0, np.ones(8))
    with pytest.raises(ValueError, match='강도'):
        s.add(1, np.full(8, np.nan))
    with pytest.raises(ValueError, match='미완료'):
        s.finish()
    with pytest.raises(ValueError, match='공통'):
        O.ObserverSpectrum(nu, [0., 1.], np.array([True, False]), 1., [1e-9, 1e9])


@pytest.fixture(scope='module')
def observer_plan():
    r = np.geomspace(4e14, 1e14, 24)
    g = O.A.geometry(r, r/(1e7), np.zeros(24), rext_factor=1.5)
    source = SimpleNamespace(ns=24, nf=8, radius=r, frequency=np.geomspace(1e14, 1e16, 8), dlognu=math.log(100)/7)
    return O.ObserverDriverPlan(g, source, block_width=4)


def test_observer_plan_covers_added_annulus_and_zero_length_tangent(observer_plan):
    p = observer_plan
    assert p.observer_impact[0] == 0. and p.observer_impact[-1] == p.outer_radius
    assert np.sum(p.observer_impact > p.source.radius[0]) == p.g.added
    assert not p.paths[-1][2] and not p.paths[-1][3]
    assert p.arrays['paths'][-1, 1] == p.arrays['paths'][-1, 3] == 0
    assert np.max(p.arrays['event_node']) < p.source.ns
    assert len(p.arrays['event_ptr']) == len(p.arrays['steps'])+1
    assert p.half_segments == len(p.arrays['steps'])
    weights = O.disk_weights(p.observer_impact, p.observer_core, p.outer_radius)
    assert weights.sum() == pytest.approx(1., rel=1e-15)
    assert weights[p.observer_impact > p.source.radius[0]].sum() > 0
    beta = p.g.velocity[0]/O.A.C
    np.testing.assert_allclose(p.observer_doppler[[0, -1]],
        [(1-beta)/math.sqrt(1-beta*beta), 1/math.sqrt(1-beta*beta)], rtol=3e-16)


def test_disk_refinement_preserves_material_and_inserts_new_tangent_turning_points(observer_plan):
    base = observer_plan
    p = O.ObserverDriverPlan(base.g, base.source, block_width=4, disk_subdivisions=2)
    assert p.source is base.source and len(p.paths) > len(base.paths)
    np.testing.assert_array_equal(p.arrays['radius'], base.arrays['radius'])
    original = set(base.observer_impact)
    checked = 0
    for impact, core, minus, plus in p.paths:
        if impact in original or core:
            continue
        assert minus[-1].z == 0.
        assert p._endpoints[id(plus[0])][0] == 0.
        checked += 1
    assert checked > 0
    assert O.disk_weights(p.observer_impact, p.observer_core, p.outer_radius).sum() == pytest.approx(1.)
    with pytest.raises(ValueError, match='세분화'):
        O.ObserverDriverPlan(base.g, base.source, disk_subdivisions=True)


def test_streaming_uses_same_driver_and_existing_scratch_and_never_returns_partial(observer_plan, monkeypatch):
    # 실제 GPU가 아닌 수송 호출 연결의 시험 대역이다.
    p = O.ObserverDriverPlan.__new__(O.ObserverDriverPlan)
    p.source = observer_plan.source; p.driver_block_width = 4; p.outer_radius = 1.
    p.observer_impact = np.array([0., .5, 1.]); p.observer_core = np.array([True, False, False])
    p.observer_doppler = np.ones(3)
    p.arrays = dict(paths=np.array([[0, 1, 1, 1, 1], [2, 1, 3, 1, 0], [4, 0, 4, 0, 0]]),
                    steps=np.zeros((4, 5)), core_mu=np.array([1., 0., 0.]))
    class Buffer:
        def __init__(self): self.address = self; self.values = np.zeros(8)
    class Runtime:
        status = object()
        def zero(self, buffer): buffer.values[:] = 0
        def upload(self, buffer, value): buffer.values[:] = value
        def sync(self): pass
        def download(self, buffer, n): return buffer.values.copy()
    def boundary(n, b, grad, mu, ray, status):
        ray.values[:] = b.values+mu*grad.values; return 0
    d = SimpleNamespace(nf=8, width=4, provider=SimpleNamespace(_source=p.source), bound=lambda:None,
        rt=Runtime(), ray=Buffer(), characteristic=Buffer(), candidate=Buffer(), interval_limit_seconds=60.,
        lib=SimpleNamespace(lumina_history_cuda_boundary=boundary), generation=3, input_identity={'material':'same'})
    calls = []
    def interval(plan, index, work, max_seconds):
        assert plan is p; calls.append(index); d.ray.values += 1.
    d.interval = interval
    result = O.run_observer(d, p, np.full(8, 3.), np.zeros(8))
    assert calls == [0, 1, 2, 3] and result['physical_intervals'] == 4
    np.testing.assert_allclose(result['luminosity_erg_s_Hz'], 4*math.pi**2*1.5, rtol=3e-16)
    assert result['input_identity'] == d.input_identity and result['source_generation'] == 3
    def fail(*args, **kwargs): raise RuntimeError('미완료 수송')
    d.interval = fail
    with pytest.raises(RuntimeError, match='미완료'):
        O.run_observer(d, p, np.ones(8), np.zeros(8))
