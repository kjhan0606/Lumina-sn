"""물리 구간당 한 번만 점값을 재구성하는 소형 특성선 후보.

첫 주파수 상류 차분은 사용하지 않는다. 내부 적분 패널에서는 관측 주파수의
불변량 V를 유지한다. 실제90점/생산 연결이나 최종 물리 정확성 승인이 아니다.
"""
import ctypes
from functools import lru_cache
from math import comb

import numpy as np

import lumina_outer_atmosphere as A
from lumina_linear_frequency import SmallLinearFrequencyTransport


@lru_cache(maxsize=12)
def _bernstein_matrix(offset, points):
    """국소 점값을 해당 단위 구간의 Bernstein 계수로 바꾼다."""
    degree = points-1; nodes = offset+np.arange(points, dtype='f8')
    matrix = np.empty((points, points))
    for j in range(points):
        other = np.delete(nodes, j)
        power = np.polynomial.polynomial.polyfromroots(other)/np.prod(nodes[j]-other)
        for k in range(points):
            matrix[k, j] = sum(comb(k, m)/comb(degree, m)*power[m] for m in range(k+1))
    matrix.flags.writeable = False
    return matrix


@lru_cache(maxsize=16)
def _layout(nf, whole, points):
    left = np.arange(nf-whole-1)+whole
    first = np.minimum(np.maximum(left-(points//2-1), 0), nf-points)
    matrices = np.stack([_bernstein_matrix(int(offset), points) for offset in first-left])
    for values in (left, first, matrices): values.flags.writeable = False
    return left, first, matrices


def _positive_polynomial(data, left, right, fraction, matrices):
    """공유 점값 다항식 평가. 호출자는 공간/전역ν 인덱스를 따로 관리한다."""
    points = data.shape[1]
    base = left[:, None]
    bernstein = base+np.einsum('ijk,ik->ij', matrices, data-base)
    low = base+np.arange(points)[None, :]/(points-1)*(right-left)[:, None]
    bernstein[:, 0] = left; bernstein[:, -1] = right
    if not np.isfinite(bernstein).all(): raise ValueError('재구성 계수 수치 범위 오류')
    ratios = np.ones_like(bernstein)
    np.divide(low, low-bernstein, out=ratios, where=bernstein < 0)
    theta = np.min(ratios, axis=1); mask = theta < 1
    limited = int(np.count_nonzero(mask))
    if np.any(mask):
        theta[mask] = np.nextafter(theta[mask], 0.)
        bernstein[mask] = theta[mask, None]*bernstein[mask]+(1-theta[mask, None])*low[mask]
        fallback = np.any(bernstein < 0, axis=1)
        bernstein[fallback] = low[fallback]
    for degree in range(points-1, 0, -1):
        bernstein[:, :degree] = (1-fraction)*bernstein[:, :degree]+fraction*bernstein[:, 1:degree+1]
    return bernstein[:, 0].copy(), limited


def reconstruct(incident, shift, *, points=8):
    """6점5차/8점7차 점값 재구성. 음수 방지는 다항식 계수 단계에서만 한다.

청색 끝 밖에서는 기존 Iν를 상수 연장한다. 자료를 bin 평균으로 보지 않는다.
출력 강도에 floor/clip을 적용하지 않는다.
"""
    values = np.asarray(incident, dtype='f8')
    if (type(points) is not int or points not in (6, 8) or
            values.ndim != 1 or not points <= len(values) <= 512 or
            not np.isfinite(values).all() or np.any(values < 0) or
            not np.isfinite(shift) or shift < 0):
        raise ValueError('소형 고차 점값 재구성 입력 오류')
    nf = len(values); result = np.full(nf, values[-1]); limited = 0
    if shift >= nf-1:
        return result, dict(reconstructions=1, limited_polynomials=0, reconstruction_points=points)
    whole = int(np.floor(shift)); fraction = float(shift-whole)
    if fraction == 0:
        result[:nf-whole] = values[whole:]
        return result, dict(reconstructions=1, limited_polynomials=0, reconstruction_points=points)
    left, first, matrices = _layout(nf, whole, points)
    data = values[first[:, None]+np.arange(points)[None, :]]
    result[:len(left)], limited = _positive_polynomial(data, values[left], values[left+1], fraction, matrices)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('재구성 다항식 수치 범위 오류')
    return result, dict(reconstructions=1, limited_polynomials=limited, reconstruction_points=points)


def load(path):
    library = ctypes.CDLL(str(path)); function = library.lumina_small_uncoupled_characteristic
    pointer = np.ctypeslib.ndpointer(dtype='f8', ndim=1, flags='C_CONTIGUOUS')
    function.argtypes = [ctypes.c_int, ctypes.c_double]+[pointer]*6
    function.restype = ctypes.c_int

    def segment(length, chi, eta_start, eta_end, incident):
        arrays = [np.ascontiguousarray(v, dtype='f8') for v in (eta_start, eta_end, incident)]
        nf = arrays[-1].size; opacity = np.asarray(chi, dtype='f8')
        if opacity.shape == (nf,): opacity = np.stack((opacity, opacity))
        if not 6 <= nf <= 512 or opacity.shape != (2, nf) or any(v.shape != (nf,) for v in arrays):
            raise ValueError('소형 비결합 특성선 차원 오류')
        first, last = (np.ascontiguousarray(v) for v in opacity)
        result = np.empty(nf)
        code = function(nf, length, first, last, *arrays, result)
        if code: raise ValueError('비결합 특성선 적분 실패: '+str(code))
        return result
    return segment


def propagate(segment, dlog, incident, geometry, coefficients, remaining_drift, *, panels=1):
    """한 물리 구간 안에서 V=I_local exp(−3δ)를 고정 주파수로 전파한다.

geometry(a,b)는 공변 길이/drift, coefficients(x)는 같은 동결 J의 χ/총η다.
계수의 주파수 보간은 선형이며 χ 부호/η 양수를 보존한다. 공통 교차 mesh는 없다.
"""
    initial = np.asarray(incident, dtype='f8')
    if (not np.isfinite(dlog) or dlog <= 0 or type(panels) is not int or
            not 1 <= panels <= 256 or not all(callable(v) for v in (segment, geometry, coefficients, remaining_drift))):
        raise ValueError('소형 물리 구간 입력/패널 오류')
    start_drift = float(remaining_drift(0.)); last_drift = float(remaining_drift(1.))
    if not np.isfinite(start_drift) or start_drift < 0 or last_drift != 0:
        raise ValueError('물리 구간의 관측 Doppler 범위 오류')
    value, info = reconstruct(initial, start_drift/dlog)
    value *= np.exp(-3*start_drift)
    coordinate = np.arange(len(initial), dtype='f8')

    def source(x):
        delta = float(remaining_drift(x)); chi, eta = (np.asarray(v, dtype='f8') for v in coefficients(x))
        if (not np.isfinite(delta) or delta < 0 or chi.shape != initial.shape or eta.shape != initial.shape or
                not np.isfinite(chi).all() or not np.isfinite(eta).all() or np.any(eta < 0)):
            raise ValueError('이동 주파수의 동결 계수/드리프트 오류')
        target = coordinate+delta/dlog
        return np.interp(target, coordinate, chi), np.interp(target, coordinate, eta)*np.exp(-3*delta)

    previous = source(0.)
    for index in range(panels):
        a, b = index/panels, (index+1)/panels
        length, drift = map(float, geometry(a, b))
        if (not np.isfinite(length) or length < 0 or not np.isfinite(drift) or drift < 0 or
                (length == 0 and drift != 0)):
            raise ValueError('물리 구간 기하 오류')
        current = source(b)
        value = segment(length, np.stack((previous[0], current[0])), previous[1], current[1], value)
        previous = current
    return value, dict(info, formal_steps=panels, interior_intensity_remappings=0,
        frequency_crossing_cuts=0, accepted_physical_field=False, final_goal='NOT_JUDGED')


def interval_geometry(g, impact, a, b, first_radius, last_radius, panels):
    """J/계수와 독립인 한 물리 구간의 정확한 원래 끝점/패널 기하."""
    x = [i/panels for i in range(panels+1)]
    z = [a+(b-a)*v for v in x]; z[0] = a; z[-1] = b
    radius = [float(np.hypot(impact, v)) for v in z]
    radius[0] = first_radius; radius[-1] = last_radius
    observer_doppler = A.doppler(g, impact, b, last_radius)
    native_radius = [None]*(panels+1)
    native_radius[0] = first_radius; native_radius[-1] = last_radius
    delta = [float(np.log(A.doppler(g, impact, v, r)/observer_doppler)) for v, r in zip(z, native_radius)]
    lengths = [A.segment_geometry(g, impact, z[i], z[i+1],
        start_radius=native_radius[i], end_radius=native_radius[i+1])[0] for i in range(panels)]
    if min(delta) < 0 or any(a < b for a, b in zip(delta[:-1], delta[1:])):
        raise ValueError('같은 특성선의 Doppler 단조성 오류')
    return dict(radius=dict(zip(x, radius)), delta=dict(zip(x, delta)),
        geometry={(x[i], x[i+1]):(float(lengths[i]), delta[i]-delta[i+1]) for i in range(panels)})


class SmallSemiLagrangianTransport(SmallLinearFrequencyTransport):
    """같은24×8 이상 합성 원래점 경로. 패널은 물리 구간 수를 바꾸지 않는다."""
    def __init__(self, *args, characteristic_segment, panels=1, **kwargs):
        if type(panels) is not int or not 1 <= panels <= 16:
            raise ValueError('소형 원래점 특성선 패널 범위 오류')
        self.characteristic_segment = characteristic_segment
        self.panels = panels; self._coordinates = {}; self._characteristics = {}
        self.reconstructions = self.limited_polynomials = 0
        super().__init__(*args, linear_segment=None, subdivisions=1, **kwargs)
        if self.nf < 8: raise ValueError('8점 재구성에는 최소8ν가 필요하다')

    def _path(self, impact, coordinates, radii, subdivisions):
        steps = super()._path(impact, coordinates, radii, subdivisions)
        for step, (a, b) in zip(steps, zip(coordinates[:-1], coordinates[1:])):
            self._coordinates[id(step)] = float(a), float(b)
        return steps

    def _characteristic(self, step, impact):
        key = id(step)
        if key in self._characteristics: return self._characteristics[key]
        a, b = self._coordinates[key]; first_radius, last_radius, _ = self._endpoints[key]
        data = interval_geometry(self.g, impact, a, b, first_radius, last_radius, self.panels)
        self._characteristics[key] = data  # 기하만 저장하며 χ/η/J/적응 장을 저장하지 않는다.
        return data

    def _sequence(self, steps, incident, own_j, log_chi, log_eta, samples, impact):
        intensity = incident
        for step in steps:
            plan = self._characteristic(step, impact)
            intensity, report = propagate(self.characteristic_segment, self.dlog, intensity,
                lambda a, b: plan['geometry'][a, b],
                lambda x: self._at(plan['radius'][x], own_j, log_chi, log_eta),
                lambda x: plan['delta'][x], panels=self.panels)
            self.formal_steps += report['formal_steps']
            self.reconstructions += report['reconstructions']
            self.limited_polynomials += report['limited_polynomials']
            for sample in step.observations:
                radius = self.sample_radii[sample]; beta = float(A.velocity_at(self.g, radius)/A.C)
                lab_mu = step.z/radius; mu = (lab_mu-beta)/(1-beta*lab_mu)
                samples[sample].append((mu, intensity.copy()))
        return intensity

    def apply(self, own_j, split_outer=False):
        self.reconstructions = self.limited_polynomials = 0
        result, report = super().apply(own_j, split_outer=split_outer)
        report.update(frequency_scheme='8-point seventh-degree point-value semi-Lagrangian',
            reconstructions=self.reconstructions, limited_polynomials=self.limited_polynomials,
            physical_intervals=self.half_segments, internal_panels=self.panels,
            interior_intensity_remappings=0, frequency_crossing_cuts=0,
            coefficient_frequency_rule='linear chi and total eta at the same characteristic frequency',
            geometry_cache_entries=len(self._characteristics))
        return result, report
