"""소형 주파수 상류 차분·끝점 선형 소스 후보. 기존 연산자와 다르다.

구간마다 청색→적색으로 결합하여 풀고 광선을 한 번 진행한다. 관측점별
경계 재적분/주파수 교차점 공통 분할은 없다. 생산 연결·실제 모델 승인이 아니다.
"""
import ctypes

import numpy as np

import lumina_outer_atmosphere as A
from lumina_nodal_transport import SmallNodalTransport, interpolate_source


def load(path):
    library = ctypes.CDLL(str(path))
    function = library.lumina_small_linear_opacity_frequency
    pointer = np.ctypeslib.ndpointer(dtype='f8', ndim=1, flags='C_CONTIGUOUS')
    function.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]+[pointer]*6
    function.restype = ctypes.c_int

    def segment(dlog, length, drift, chi, eta_start, eta_end, incident):
        arrays = [np.ascontiguousarray(v, dtype='f8') for v in (eta_start, eta_end, incident)]
        nf = arrays[-1].size
        opacity = np.asarray(chi, dtype='f8')
        if opacity.shape == (nf,):
            opacity = np.stack((opacity, opacity))
        if not 2 <= nf <= 512 or any(v.shape != (nf,) for v in arrays):
            raise ValueError('합성 주파수 결합 구간 차원 오류')
        if opacity.shape != (2, nf):
            raise ValueError('합성 선형 흡광계수 끝점 차원 오류')
        chi_start, chi_end = (np.ascontiguousarray(v) for v in opacity)
        output = np.empty(nf)
        code = function(nf, dlog, length, drift, chi_start, chi_end, *arrays, output)
        if code:
            raise ValueError('선형 소스 구간 입력/수치 범위 실패: '+str(code))
        return output
    return segment


class SmallLinearFrequencyTransport(SmallNodalTransport):
    """기존24×최대32ν 합성 광선·경계·각도 표본을 그대로 쓰는 별도 후보."""
    def __init__(self, *args, linear_segment, **kwargs):
        self.linear_segment = linear_segment
        self._endpoints = {}
        self._coefficient_cache = None
        self.formal_steps = 0
        super().__init__(*args, segment=None, **kwargs)

    def _path(self, impact, coordinates, radii, subdivisions):
        steps = super()._path(impact, coordinates, radii, subdivisions)
        at = 0
        for index, (first, last) in enumerate(zip(coordinates[:-1], coordinates[1:])):
            parts = np.linspace(first, last, subdivisions+1)
            for number, (a, b) in enumerate(zip(parts[:-1], parts[1:])):
                start = float(radii[index]) if number == 0 else float(np.hypot(impact, a))
                end = float(radii[index+1]) if number == subdivisions-1 else float(np.hypot(impact, b))
                self._endpoints[id(steps[at])] = start, end, float(np.hypot(impact, .5*(a+b)))
                at += 1
        return steps

    def _at(self, radius, own_j, log_chi, log_eta):
        if radius in self._coefficient_cache:
            return self._coefficient_cache[radius]
        if radius > self.g.native_radius[0]:
            ascending = self.g.radius[:self.g.added+1][::-1]
            index = int(np.searchsorted(ascending, radius))-1
            weight = float(np.log(radius/ascending[index])/np.log(ascending[index+1]/ascending[index]))
            result = (np.sign(self.chi[0])*np.exp((1-weight)*log_chi[index]+weight*log_chi[index+1]),
                np.exp(log_eta[index] if weight == 0 else log_eta[index+1] if weight == 1 else
                       (1-weight)*log_eta[index]+weight*log_eta[index+1]))
        elif radius == self.g.native_radius[-1]:
            result = self.chi[-1], self.eta_true[-1]+self.es[-1]*own_j[-1]
        else:
            ascending = self.g.native_radius[::-1]
            index = int(np.searchsorted(ascending, radius))-1
            lo, hi = ascending[index:index+2]; inner = self.n-index
            result = interpolate_source(self.chi, self.eta_true, self.es, own_j,
                inner, inner-1, float((radius-lo)/(hi-lo)))
        self._coefficient_cache[radius] = result
        return result

    def _sequence(self, steps, incident, own_j, log_chi, log_eta, samples, impact):
        intensity = incident
        for step in steps:
            first, last, _ = self._endpoints[id(step)]
            chi_start, eta_start = self._at(first, own_j, log_chi, log_eta)
            chi_end, eta_end = self._at(last, own_j, log_chi, log_eta)
            intensity = self.linear_segment(self.dlog, step.length, step.drift, np.stack((chi_start, chi_end)),
                eta_start, eta_end, intensity)
            self.formal_steps += 1
            for sample in step.observations:
                radius = self.sample_radii[sample]
                beta = float(A.velocity_at(self.g, radius)/A.C)
                lab_mu = step.z/radius
                mu = (lab_mu-beta)/(1-beta*lab_mu)
                samples[sample].append((mu, intensity.copy()))
        return intensity

    def apply(self, own_j, split_outer=False):
        if self._coefficient_cache is not None:
            raise ValueError('같은 합성 수송 객체 중복 실행 금지')
        self._coefficient_cache = {}; self.formal_steps = 0
        try:
            result, report = super().apply(own_j, split_outer=split_outer)
            report.update(frequency_scheme='first-order blue upwind; endpoint-linear coupled source',
                formal_steps=self.formal_steps, observer_boundary_reintegrations=0,
                frequency_crossing_cuts=0, source_integration='endpoint-linear chi and total eta; positive quadrature',
                raw_sliding_equivalence=False, native_SC_equivalence=False)
            return result, report
        finally:
            self._coefficient_cache = None
