"""명시적 소형 외부대기 결합 후보. 전체 생산 실행기나 오차 인증기가 아니다.

일반 속도장 구간은 공간적으로 세분화하고 실제 기존 raw positive formal을 쓴다.
변하는 χ/η/속도의 P0 구간 오차와 native SC/보간 차이는 아직 별도로 남는다.
"""
from dataclasses import dataclass

import numpy as np

import lumina_outer_atmosphere as A


@dataclass(frozen=True)
class Step:
    length: float
    drift: float
    exterior: bool
    index: int
    weight: float
    observations: tuple
    z: float


def moments(samples):
    """CMF μ에서 양의 선형 I의 J/H/K를 적분. clipping/재정규화 없음."""
    ordered = sorted(samples, key=lambda x: x[0])
    mu = np.array([s[0] for s in ordered])
    intensities = np.array([s[1] for s in ordered])
    if len(mu) < 2 or mu[0] != -1 or mu[-1] != 1:
        raise ValueError('전체 방향의 자체 각도 표본 누락')
    x, y = mu[:-1, None], mu[1:, None]
    first, last = intensities[:-1], intensities[1:]
    width = y-x
    j = np.sum(width*(first+last)/4, axis=0)
    h = np.sum(width*((2*x+y)*first+(x+2*y)*last)/12, axis=0)
    k = np.sum(width*((3*x*x+2*x*y+y*y)*first+(x*x+2*x*y+3*y*y)*last)/24, axis=0)
    return np.array((j, h, k))


class CoupledGeometry:
    """주파수장을 만들지 않는 공통 광선 계획. 생성만으로 수송 실행을 허가하지 않는다."""
    def __init__(self,g,subdivisions=1):
        if (not 24 <= len(g.native_radius) <= 512 or type(subdivisions) is not int or
                not 1 <= subdivisions <= 4):
            raise ValueError('외부대기 광선 계획 차원/세분화 범위 오류')
        self.g = g
        self.n = len(g.native_radius)-1
        self.boundaries = g.native_radius[::-1]
        self.centers = .5*(self.boundaries[:-1]+self.boundaries[1:])
        self.sample_radii = np.r_[self.centers,g.native_radius[list(g.reference_indices)]]
        core = self.boundaries[0]
        self.impacts = np.unique(np.r_[core*np.arange(17)/16,self.sample_radii])
        self.paths = []
        for impact in self.impacts:
            cuts = np.unique(np.r_[g.radius,self.sample_radii])
            cuts = cuts[cuts >= max(core,impact)]
            z = np.sqrt((cuts-impact)*(cuts+impact))
            core_ray = impact < core
            negative = self._path(impact,-z[::-1],cuts[::-1],subdivisions)
            positive = self._path(impact,z,cuts,subdivisions)
            self.paths.append((float(impact),core_ray,negative,positive))
        self.half_segments = sum(len(a)+len(b) for _,_,a,b in self.paths)

    def _path(self, impact, coordinates, radii, subdivisions):
        result = []
        for position,(start, end) in enumerate(zip(coordinates[:-1],coordinates[1:])):
            parts = np.linspace(start,end,subdivisions+1)
            for number,(a,b) in enumerate(zip(parts[:-1],parts[1:])):
                radius = float(np.hypot(impact,.5*(a+b)))
                exterior = radius > self.g.native_radius[0]
                if exterior:
                    ascending = self.g.radius[:self.g.added+1][::-1]
                    index = int(np.searchsorted(ascending,radius))-1
                    if not 0 <= index < len(ascending)-1:
                        raise ValueError('연장 구간 계수 보간 범위 오류')
                    weight = float(np.log(radius/ascending[index])/
                                   np.log(ascending[index+1]/ascending[index]))
                else:
                    index = int(np.searchsorted(self.boundaries,radius))-1
                    if not 0 <= index < self.n:
                        raise ValueError('원래 물질 구간 누락')
                    weight = 0.
                observations = []
                if number == subdivisions-1:
                    for i,r in enumerate(self.sample_radii):
                        if impact <= r and abs(end) == np.sqrt((r-impact)*(r+impact)):
                            observations.append(i)
                ds, drift = A.segment_geometry(self.g,float(impact),float(a),float(b),
                    start_radius=float(radii[position]) if number == 0 else None,
                    end_radius=float(radii[position+1]) if number == subdivisions-1 else None)
                result.append(Step(ds,drift,exterior,index,weight,tuple(observations),float(b)))
        return tuple(result)


class SmallCoupledTransport(CoupledGeometry):
    """24원래 노드 이하·32ν 이하의 직접시험 경로. 큰 입력을 우회 실행하지 않는다."""
    def __init__(self, g, frequency, cell_chi, cell_es, cell_eta,
                 node_chi, node_es, node_eta, temperature, gradient,
                 segment, subdivisions=1):
        self.g = g
        self.nu = np.array(frequency, dtype='f8', copy=True)
        self.n = len(g.native_radius)-1
        self.nf = len(self.nu)
        if (len(g.native_radius) != 24 or not 2 <= self.nf <= 32 or
                type(subdivisions) is not int or not 1 <= subdivisions <= 4 or
                not np.isfinite(self.nu).all() or np.any(self.nu <= 0) or
                not np.all(np.diff(self.nu) > 0)):
            raise ValueError('승인된 소형 결합 시험 범위 밖')
        self.dlog = float(np.log(self.nu[1]/self.nu[0]))
        if np.max(abs(np.log(self.nu[1:]/self.nu[:-1])-self.dlog)) > 64*np.finfo(float).eps:
            raise ValueError('기존 formal에 log-uniform ν 필요')
        self.cell_chi, self.cell_es, self.cell_eta = [np.array(a, dtype='f8', copy=True)
            for a in (cell_chi,cell_es,cell_eta)]
        self.node_chi, self.node_es, self.node_eta = [np.array(a, dtype='f8', copy=True)
            for a in (node_chi,node_es,node_eta)]
        for a in (self.cell_chi,self.cell_es,self.cell_eta):
            if a.shape != (self.n,self.nf) or not np.isfinite(a).all():
                raise ValueError('원래 물질 셀 계수 차원/유한성 오류')
        if np.any(self.cell_es < 0) or np.any(self.cell_eta < 0):
            raise ValueError('음의 산란/진방출률')
        for a in (self.node_chi,self.node_es,self.node_eta):
            if a.shape != (self.n+1,self.nf) or not np.isfinite(a).all():
                raise ValueError('외삽 기준 노드 계수 차원/유한성 오류')
        self.gradient = np.array(gradient, dtype='f8', copy=True)
        if (self.gradient.shape != (self.nf,) or not np.isfinite(self.gradient).all() or
                not np.isfinite(temperature) or temperature <= 0):
            raise ValueError('내부 확산 경계 입력 오류')
        h, kb = 6.62607015e-27, 1.380649e-16
        self.planck = 2*h*self.nu**3/A.C**2/np.expm1(h*self.nu/(kb*temperature))
        super().__init__(g,subdivisions)
        self.segment = segment

    def _sequence(self, steps, incident, cell_eta, log_ext_chi, log_ext_eta,
                  samples, impact):
        intensity = incident
        for step in steps:
            if step.exterior:
                i,w = step.index,step.weight
                chi = np.sign(self.node_chi[0])*np.exp((1-w)*log_ext_chi[i]+w*log_ext_chi[i+1])
                if w == 0:
                    eta = np.exp(log_ext_eta[i])
                elif w == 1:
                    eta = np.exp(log_ext_eta[i+1])
                else:
                    eta = np.exp((1-w)*log_ext_eta[i]+w*log_ext_eta[i+1])
            else:
                chi, eta = self.cell_chi[step.index], cell_eta[step.index]
            intensity = self.segment(self.dlog,step.length,step.drift,chi,eta,intensity)
            if not np.isfinite(intensity).all() or np.any(intensity < 0):
                raise ValueError('자체 구간 강도 비유한/음수')
            for sample in step.observations:
                r = self.sample_radii[sample]
                beta = float(A.velocity_at(self.g,r)/A.C)
                lab_mu = step.z/r
                mu = (lab_mu-beta)/(1-beta*lab_mu)
                samples[sample].append((mu,intensity.copy()))
        return intensity

    def apply(self, own_j, split_outer=False):
        own_j = np.asarray(own_j)
        if (own_j.shape != (self.n+2,self.nf) or not np.isfinite(own_j).all() or
                np.any(own_j < 0)):
            raise ValueError('자체 셀/보조 노드 J 오류')
        nodes = np.zeros_like(self.node_eta)
        for offset, index in enumerate(self.g.reference_indices):
            nodes[index] = own_j[self.n+offset]
        ext = A.coefficients(self.g,self.node_chi,self.node_es,self.node_eta,nodes)
        # 외부와 접합 노드만 log 보간한다. 내부의 실제 signed χ를 지우거나
        # 외부 보간에서 읽지 않을 내부 음수 χ 때문에 전체 문제를 거부하지 않는다.
        magnitude = abs(ext['chi'][:self.g.added+1][::-1])
        log_chi = np.zeros_like(magnitude)
        # χ_a=0이면 전체 연장 χ=0이다. log 저장소의0은 실제 χ floor가
        # 아니라 사용하지 않는 자리이며, 위의 부호0을 곱해 정확히0을 보존한다.
        np.log(magnitude,out=log_chi,where=magnitude>0)
        eta = ext['eta_total'][:self.g.added+1][::-1]
        log_eta = np.full_like(eta,-np.inf)
        np.log(eta,out=log_eta,where=eta>0)  # 0 자체를 보존하는 log 표현이며 floor가 아니다.
        cell_eta = self.cell_eta+self.cell_es*own_j[:self.n]
        samples = [[] for _ in self.sample_radii]
        incoming = []
        for impact,core,negative,positive in self.paths:
            initial = np.zeros(self.nf)
            if split_outer:
                joint = next((i for i,s in enumerate(negative) if not s.exterior),len(negative))
                initial = self._sequence(negative[:joint],initial,cell_eta,log_chi,log_eta,samples,impact)
                incoming.append(initial.copy())
                initial = self._sequence(negative[joint:],initial,cell_eta,log_chi,log_eta,samples,impact)
            else:
                initial = self._sequence(negative,initial,cell_eta,log_chi,log_eta,samples,impact)
            if core:
                r = self.boundaries[0]
                lab_mu = np.sqrt((r-impact)*(r+impact))/r
                beta = float(A.velocity_at(self.g,r)/A.C)
                mu = (lab_mu-beta)/(1-beta*lab_mu)
                initial = self.planck+mu*self.gradient
                if np.any(initial < 0) or not np.isfinite(initial).all():
                    raise ValueError('내부 경계의 실제 signed μ 강도 오류')
            if split_outer:
                joint = next((i for i,s in enumerate(positive) if s.exterior),len(positive))
                initial = self._sequence(positive[:joint],initial,cell_eta,log_chi,log_eta,samples,impact)
                self._sequence(positive[joint:],initial,cell_eta,log_chi,log_eta,samples,impact)
            else:
                self._sequence(positive,initial,cell_eta,log_chi,log_eta,samples,impact)
        angular = np.array([moments(a) for a in samples])
        return angular[:,0], dict(moments=angular,join_incoming=incoming,
            half_segments=self.half_segments,affine_error_certificate=False,
            coefficient_rule='piecewise constant cells; log-linear external node profiles; spatial refinement required')


def solve(problem, initial, tolerance=1e-10, cap=100):
    """소형 비선형 후보와 실제 새 전체 연산자의 잔차. 유일성/해오차 증명은 아니다."""
    current = np.array(initial,dtype='f8',copy=True)
    for iteration in range(1,cap+1):
        updated, info = problem.apply(current,split_outer=True)
        maximum = float(updated.max())
        change = float(abs(updated-current).max())
        ratio = change/maximum if maximum > 0 else (0. if change == 0 else np.inf)
        if ratio <= tolerance:
            checked, checked_info = problem.apply(updated,split_outer=True)
            residual = float(abs(checked-updated).max())
            norm = float(updated.max())
            relative = residual/norm if norm > 0 else (0. if residual == 0 else np.inf)
            if relative <= tolerance:
                return updated,dict(iterations=iteration,residual_max=residual,
                    residual_max_norm_ratio=relative,accepted_physical_field=False,
                    error_upper='NOT_PROVIDED_FOR_NONLINEAR_OPERATOR',**checked_info)
        current = updated
    raise ValueError('소형 결합 반복 상한 소진; 자동 연장/해법 전환 없음')
