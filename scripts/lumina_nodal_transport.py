"""원래 점의 χ·진η·ES·자체 J를 보존하는 독립 소형 수송 후보.

생산 P0 API/ABI는 변경하지 않는다. 24노드·32ν 이하의 합성 시험 전용이며
기존 raw C 구간 함수를 호출한다. 선형 반경 재구성의 구간중점/도착점 후보다.
native SC 동일성, 두꺼운 변화 소스의 정확성, 실제 모델 실행 승인은 아니다.
"""
from dataclasses import replace

import numpy as np

import lumina_outer_atmosphere as A
import lumina_outer_transport as T


class NodalGeometry(T.CoupledGeometry):
    """자체 J 표본을 원래 점에 두며 공간 세분화마다 보간 위치를 갱신한다."""
    def __init__(self, g, subdivisions=1, coefficient_location='midpoint'):
        if (not 24 <= len(g.native_radius) <= 512 or type(subdivisions) is not int or
                not 1 <= subdivisions <= 32):
            raise ValueError('원래점 광선 계획 차원/세분화 오류')
        if coefficient_location not in ('midpoint', 'endpoint'):
            raise ValueError('명시한 내부 계수 평가 위치 필요')
        self.coefficient_location = coefficient_location
        self.g, self.n = g, len(g.native_radius)-1
        self.boundaries = g.native_radius[::-1]
        self.sample_radii = g.native_radius.copy()  # 원래 바깥→안쪽 순서.
        core = self.boundaries[0]
        self.impacts = np.unique(np.r_[core*np.arange(17)/16, self.sample_radii])
        self.paths = []
        for impact in self.impacts:
            cuts = np.unique(np.r_[g.radius, self.sample_radii])
            cuts = cuts[cuts >= max(core, impact)]
            z = np.sqrt((cuts-impact)*(cuts+impact))
            minus = self._path(impact, -z[::-1], cuts[::-1], subdivisions)
            plus = self._path(impact, z, cuts, subdivisions)
            # 경계 접선은 같은 μ의 두 한계를 가진다. 먼저 외측 경로,
            # 다음 core 경계 한계를 두어 각도 적분이 경계 jump를 잇지 않게 한다.
            # native의 유한 각도 stencil과 동일하다고 주장하지 않는다.
            branches = (False, True) if impact == core else (impact < core,)
            for is_core in branches:
                self.paths.append((float(impact), bool(is_core), minus, plus))
        self.half_segments = sum(len(a)+len(b) for _, _, a, b in self.paths)

    def _path(self, impact, coordinates, radii, subdivisions):
        steps = super()._path(impact, coordinates, radii, subdivisions)
        previous = float(coordinates[0])
        result = []
        for step in steps:
            if not step.exterior:
                if self.coefficient_location == 'endpoint' and step.observations:
                    radius = float(self.sample_radii[step.observations[0]])
                else:
                    z = step.z if self.coefficient_location == 'endpoint' else .5*(previous+step.z)
                    radius = float(np.hypot(impact, z))
                lo, hi = self.boundaries[step.index:step.index+2]
                weight = float((radius-lo)/(hi-lo))
                if not 0 <= weight <= 1:
                    raise ValueError('원래점 선형 재구성 위치 오류')
                step = replace(step, weight=weight)
            result.append(step)
            previous = step.z
        return tuple(result)


def interpolate_source(chi, eta_true, es, own_j, inner, outer, weight):
    """총η를 먼저 보간하지 않는다. ES(r)·J(r)는 일반적으로 이차다."""
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('원래점 보간 가중치 오류')
    mix = lambda value: (1-weight)*value[inner]+weight*value[outer]
    absorption = mix(chi)
    emission = mix(eta_true)+mix(es)*mix(own_j)
    if not np.isfinite(absorption).all() or not np.isfinite(emission).all() or np.any(emission < 0):
        raise ValueError('원래점 보간 계수 비유한/음의 방출률')
    return absorption, emission


class SmallNodalTransport(NodalGeometry):
    """실제 전 모델을 우회 실행하지 않는 소형 독립 경로. 참조 J 입력은 없다."""
    def __init__(self, g, frequency, chi, eta_true, es, temperature, gradient,
                 segment, subdivisions=1, coefficient_location='midpoint'):
        nu = np.asarray(frequency, dtype='f8')
        if (len(g.native_radius) != 24 or nu.ndim != 1 or not 2 <= len(nu) <= 32 or
                not np.isfinite(nu).all() or np.any(nu <= 0) or np.any(np.diff(nu) <= 0)):
            raise ValueError('24원래점·최대32ν 소형 시험 범위 필요')
        self.nu = nu.copy()
        self.nf = len(nu)
        self.dlog = float(np.log(nu[1]/nu[0]))
        if np.max(abs(np.log(nu[1:]/nu[:-1])-self.dlog)) > 64*np.finfo(float).eps:
            raise ValueError('raw 구간 함수에는 log-uniform ν 필요')
        self.chi, self.eta_true, self.es = [np.array(a, dtype='f8', copy=True) for a in (chi, eta_true, es)]
        shape = (len(g.native_radius), self.nf)
        if (any(a.shape != shape or not np.isfinite(a).all() for a in (self.chi, self.eta_true, self.es)) or
                np.any(self.eta_true < 0) or np.any(self.es <= 0)):
            raise ValueError('원래점 χ/진η/양의 ES 입력 오류')
        self.gradient = np.array(gradient, dtype='f8', copy=True)
        if (self.gradient.shape != (self.nf,) or not np.isfinite(self.gradient).all() or
                not np.isfinite(temperature) or temperature <= 0):
            raise ValueError('원래점 내부 확산 경계 입력 오류')
        h, kb = 6.62607015e-27, 1.380649e-16
        self.planck = 2*h*nu**3/A.C**2/np.expm1(h*nu/(kb*temperature))
        self.segment = segment
        super().__init__(g, subdivisions, coefficient_location)

    def _sequence(self, steps, incident, own_j, log_chi, log_eta, samples, impact):
        intensity = incident
        for step in steps:
            if step.exterior:
                i, w = step.index, step.weight
                chi = np.sign(self.chi[0])*np.exp((1-w)*log_chi[i]+w*log_chi[i+1])
                eta = np.exp(log_eta[i] if w == 0 else log_eta[i+1] if w == 1 else
                             (1-w)*log_eta[i]+w*log_eta[i+1])
            else:
                inner = self.n-step.index
                chi, eta = interpolate_source(self.chi, self.eta_true, self.es, own_j,
                                               inner, inner-1, step.weight)
            intensity = self.segment(self.dlog, step.length, step.drift, chi, eta, intensity)
            if not np.isfinite(intensity).all() or np.any(intensity < 0):
                raise ValueError('원래점 구간 강도 비유한/음수')
            for sample in step.observations:
                r = self.sample_radii[sample]
                beta = float(A.velocity_at(self.g, r)/A.C)
                lab_mu = step.z/r
                mu = (lab_mu-beta)/(1-beta*lab_mu)
                samples[sample].append((mu, intensity.copy()))
        return intensity

    def apply(self, own_j, split_outer=False):
        own_j = np.asarray(own_j)
        if (own_j.shape != self.chi.shape or not np.isfinite(own_j).all() or np.any(own_j < 0)):
            raise ValueError('자체 원래점 J 입력 오류')
        ext = A.coefficients(self.g, self.chi, self.es, self.eta_true, own_j)
        magnitude = abs(ext['chi'][:self.g.added+1][::-1])
        log_chi = np.zeros_like(magnitude)
        np.log(magnitude, out=log_chi, where=magnitude > 0)
        eta = ext['eta_total'][:self.g.added+1][::-1]
        log_eta = np.full_like(eta, -np.inf)
        np.log(eta, out=log_eta, where=eta > 0)
        samples = [[] for _ in self.sample_radii]
        core = self.boundaries[0]
        for impact, is_core, minus, plus in self.paths:
            incident = self._sequence(minus, np.zeros(self.nf), own_j, log_chi, log_eta, samples, impact)
            if is_core:
                lab_mu = float(np.sqrt((core-impact)*(core+impact))/core)
                beta = float(A.velocity_at(self.g, core)/A.C)
                mu = (lab_mu-beta)/(1-beta*lab_mu)
                incident = self.planck+mu*self.gradient
                if not np.isfinite(incident).all() or np.any(incident < 0):
                    raise ValueError('원래점 확산 경계 강도 비유한/음수')
                # 최내곽 관측은 incoming뿐 아니라 경계 재설정 직후 outgoing도 필요하다.
                samples[-1].append((mu, incident.copy()))
            self._sequence(plus, incident, own_j, log_chi, log_eta, samples, impact)
        angular = np.array([T.moments(values) for values in samples])
        return angular[:, 0], dict(moments=angular, half_segments=self.half_segments,
            native_nodes=len(self.sample_radii), auxiliary_field_unknowns=0,
            field_coordinate='original native radii, outer-to-inner point samples',
            coefficient_rule='linear-r chi/true-eta/ES/J; assemble ES*J after interpolation',
            interior_coefficient_location=self.coefficient_location,
            exterior_coefficient_location='legacy midpoint log interpolation',
            inner_grazing_rule='two-sided angular boundary limit; not claimed native finite-stencil equality',
            affine_error_certificate=False, accepted_physical_field=False, final_goal='NOT_JUDGED')


def solve(problem, initial, tolerance=1e-10, cap=80):
    """자체 J 반복과 실제 반환 잔차. 공간 오차를 인증하지 않는다."""
    if type(cap) is not int or not 1 <= cap <= 80 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('소형 원래점 반복 한도/잔차 기준 오류')
    current = np.array(initial, dtype='f8', copy=True)
    for iteration in range(1, cap+1):
        updated, _ = problem.apply(current)
        maximum = float(updated.max())
        change = float(abs(updated-current).max())
        ratio = change/maximum if maximum else (0. if change == 0 else float('inf'))
        if ratio <= tolerance:
            checked, info = problem.apply(updated)
            absolute = float(abs(checked-updated).max())
            residual = absolute/maximum if maximum else (0. if absolute == 0 else float('inf'))
            if residual <= tolerance:
                return updated, dict(info, iterations=iteration, residual_max=absolute,
                    residual_max_norm_ratio=residual, accepted_physical_field=False,
                    error_upper='NOT_PROVIDED_FOR_NONLINEAR_OPERATOR')
        current = updated
    raise ValueError('소형 원래점 반복 상한 소진; 자동 연장 없음')
