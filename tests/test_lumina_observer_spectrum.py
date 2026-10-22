"""관측자 변환/외곽 광선/단위의 독립 해석 시험. 실제 모델 수용은 아니다."""
import math
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import lumina_observer_spectrum as O
from test_lumina_characteristic_linear import characteristic, run as solve_characteristic, decoded


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


def cpu_adapter(p):
    """장치 메모리/경계 주입만 CPU 배열로 대신한다. 구간 수송은 호출자가 지정한다."""
    class Buffer:
        def __init__(self): self.address = self; self.values = np.zeros(p.source.nf)
    class Runtime:
        status = object()
        def zero(self, buffer): buffer.values[:] = 0
        def upload(self, buffer, value): buffer.values[:] = value
        def sync(self): pass
        def download(self, buffer, n): return buffer.values.copy()
    def boundary(n, b, grad, mu, ray, status):
        ray.values[:] = b.values+mu*grad.values; return 0
    return SimpleNamespace(nf=p.source.nf, width=4, provider=SimpleNamespace(_source=p.source), bound=lambda:None,
        rt=Runtime(), ray=Buffer(), characteristic=Buffer(), candidate=Buffer(), interval_limit_seconds=60.,
        lib=SimpleNamespace(lumina_history_cuda_boundary=boundary), generation=3, input_identity={'material':'same'})


def test_streaming_uses_same_driver_and_existing_scratch_and_never_returns_partial(observer_plan):
    # 실제 GPU가 아닌 수송 호출 연결의 시험 대역이다.
    p = O.ObserverDriverPlan.__new__(O.ObserverDriverPlan)
    p.source = observer_plan.source; p.driver_block_width = 4; p.outer_radius = 1.
    p.observer_impact = np.array([0., .5, 1.]); p.observer_core = np.array([True, False, False])
    p.observer_doppler = np.ones(3)
    p.arrays = dict(paths=np.array([[0, 1, 1, 1, 1], [2, 1, 3, 1, 0], [4, 0, 4, 0, 0]]),
                    steps=np.zeros((4, 5)), core_mu=np.array([1., 0., 0.]))
    d = cpu_adapter(p)
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


@pytest.mark.parametrize('optical_radius', [0., .7, 3.])
def test_real_characteristic_kernel_preserves_emitting_annulus_and_core_occultation(
        observer_plan, characteristic, optical_radius, tmp_path):
    # 정상·정적·구간 일정 회색 매질의 정확해다. 실제 원자/외삽 폐합 시험은 아니다.
    g = replace(observer_plan.g, velocity=np.zeros_like(observer_plan.g.velocity))
    p = O.ObserverDriverPlan(g, observer_plan.source, block_width=4)
    d = cpu_adapter(p); radius = p.outer_radius; core = float(p.boundaries[0])
    chi = optical_radius/radius; eta = 2e-8/radius
    values = np.array([[chi, chi], [chi, chi], [eta, eta], [eta, eta]])
    calls = 0; emergent = []
    def interval(plan, index, work, max_seconds):
        nonlocal calls
        work.take('map_queries', d.nf)
        impact, za, zb, ra, rb = p.arrays['steps'][index]
        length, drift = O.A.segment_geometry(g, impact, za, zb, start_radius=ra, end_radius=rb)
        assert drift == 0.
        np.testing.assert_array_equal(d.ray.values, np.full(d.nf, d.ray.values[0]))
        answer = solve_characteristic(characteristic, values, length=length,
                                      incident=float(d.ray.values[0]), relative=1e-10)
        d.ray.values[:] = decoded(answer)[0]; calls += 1
    d.interval = interval
    result = O.run_observer(d, p, np.zeros(d.nf), np.zeros(d.nf), seconds=120.,
                            progress=lambda _report:emergent.append(float(d.ray.values[0])))
    assert calls == p.half_segments == result['physical_intervals']
    impact = p.observer_impact
    z_outer = np.sqrt((radius-impact)*(radius+impact))
    lengths = 2*z_outer
    inside = p.observer_core
    lengths[inside] = z_outer[inside]-np.sqrt((core-impact[inside])*(core+impact[inside]))
    wanted = eta*lengths if chi == 0 else eta*(-np.expm1(-chi*lengths))/chi
    np.testing.assert_allclose(emergent, wanted, rtol=2e-9, atol=0.)
    # 독립적인 각 광선 정확해를 같은 명시적 disk 구적에 넣은 값과 비교한다.
    weights = O.disk_weights(impact, inside, radius)
    exact_discrete = 4*math.pi**2*radius**2*np.dot(weights, wanted)
    np.testing.assert_allclose(result['luminosity_erg_s_Hz'], exact_discrete, rtol=2e-9, atol=0.)
    outer = impact > p.source.radius[0]
    assert np.any(np.asarray(emergent)[outer] > 0.)
    missing_annulus = 4*math.pi**2*radius**2*np.dot(weights[~outer], wanted[~outer])
    assert missing_annulus < .95*exact_discrete  # 외곽 고리를 빼는 오류는5% 이상이다.
    assert not result['accepted_physical_spectrum'] and not result['error_certificate']
    nonzero = wanted > 0
    ray_error = float(np.max(abs(np.asarray(emergent)[nonzero]/wanted[nonzero]-1)))
    rows = [dict(disk_subdivisions=1, paths=len(p.paths), C_characteristic_calls=calls,
                 luminosity_erg_s_Hz=float(result['luminosity_erg_s_Hz'][0]))]
    continuum = None
    if chi == 0:
        # 일정 체적 방출과 차가운 불투명 core의 정확한 부피/차폐 적분이다.
        continuum = (8*math.pi**2*eta/3)*(radius**3+(radius*radius-core*core)**1.5-core**3)
        rows[0]['relative_to_continuum'] = abs(rows[0]['luminosity_erg_s_Hz']/continuum-1)
        for subdivisions in (4, 16):
            p = O.ObserverDriverPlan(g, observer_plan.source, block_width=4, disk_subdivisions=subdivisions)
            d = cpu_adapter(p); d.interval = interval; calls = 0
            result = O.run_observer(d, p, np.zeros(d.nf), np.zeros(d.nf), seconds=120.)
            rows.append(dict(disk_subdivisions=subdivisions, paths=len(p.paths), C_characteristic_calls=calls,
                luminosity_erg_s_Hz=float(result['luminosity_erg_s_Hz'][0]),
                relative_to_continuum=float(abs(result['luminosity_erg_s_Hz'][0]/continuum-1))))
    record = dict(optical_radius=optical_radius, native_nodes=24, frequencies=d.nf,
        maximum_each_ray_relative=ray_error, continuum_luminosity_erg_s_Hz=continuum, rows=rows,
        excluded_annulus_fraction=float(1-missing_annulus/exact_discrete), GPU_calls=0,
        native_model_F_calls=0, final_goal='NOT_MET',
        scope='정적 회색 제조 매질·실제 C++ 특성선 커널. CMFGEN 원자자료/실제 외곽 폐합/자기일관 해의 시험이 아니다.')
    with (tmp_path/'grey-observer-transport.json').open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if chi == 0:
        assert all(a['relative_to_continuum'] > b['relative_to_continuum'] for a, b in zip(rows[:-1], rows[1:]))
        assert rows[-1]['relative_to_continuum'] <= 1e-4
