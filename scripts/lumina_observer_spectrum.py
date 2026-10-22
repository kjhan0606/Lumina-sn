"""같은 동결 물질 수송의 출사 광선을 관측자계 절대 광도로 연결한다.

유한 외곽 밖은 진공인 정상 구대칭 스냅샷이다. 초기화는 수송을 실행하지 않는다.
내부 J/H/K를 스펙트럼으로 대체하지 않으며, 물질·장 공동수렴을 증명하지 않는다.
"""
import math
import time

import numpy as np

import lumina_outer_atmosphere as A
from lumina_gpu_driver import Work, check, check_host
from lumina_nodal_driver import CompiledDriverPlan


class ObserverDriverPlan(CompiledDriverPlan):
    """원 native 광선을 보존하며 외곽 고리와 선택적 disk 미세화를 추가한다."""
    def __init__(self, geometry, source, *, block_width=256, disk_subdivisions=1):
        if type(disk_subdivisions) is not int or disk_subdivisions not in (1, 2, 4, 8, 16):
            raise ValueError('관측자 disk 세분화는1/2/4/8/16 중 하나여야 함')
        super().__init__(geometry, source, panels=1, block_width=block_width)
        a = self.arrays
        paths = a['paths'].tolist(); steps = a['steps'].tolist()
        events = a['event_ptr'].tolist(); core_mu = a['core_mu'].tolist()
        core_weights = a['core_weights'].tolist()
        outer = float(geometry.radius[0]); original = {float(p[0]) for p in self.paths}
        knots = sorted(original | set(float(r) for r in geometry.radius))
        extra = set(knots)-original
        for left, right in zip(knots[:-1], knots[1:]):
            lo, hi = (left/outer)**2, (right/outer)**2
            for j in range(1, disk_subdivisions):
                impact = outer*math.sqrt(lo+(hi-lo)*j/disk_subdivisions)
                if not left < impact < right:
                    raise ValueError('관측자 disk 세분화 좌표의 표현 한도')
                extra.add(impact)
        for impact in sorted(extra):
            is_core = impact < self.boundaries[0]
            # 새 접선이 원래 반경점 사이에 있으면 z=0 전환점도 반드시 넣는다.
            minimum = max(float(self.boundaries[0]), impact)
            cuts = np.unique(np.r_[geometry.radius[geometry.radius >= minimum], minimum])
            z = np.sqrt((cuts-impact)*(cuts+impact))
            minus = self._path(impact, -z[::-1], cuts[::-1], 1)
            plus = self._path(impact, z, cuts, 1)
            entry = []
            for leg in (minus, plus):
                entry.extend((len(steps), len(leg)))
                for step in leg:
                    steps.append((impact, *self._endpoints[id(step)]))
                    # 추가 광선은 disk 적분용이며 내부 J/H/K stencil에는 더하지 않는다.
                    events.append(events[-1])
            self.paths.append((impact, bool(is_core), minus, plus))
            paths.append((*entry, int(is_core)))
            z_core = float(np.sqrt((minimum-impact)*(minimum+impact)))
            core_mu.append(self._mu(impact, z_core, source.ns-1) if is_core else 0.)
            core_weights.append([0., 0., 0.])
        for name, values, dtype in (('paths', paths, 'i4'), ('steps', steps, 'f8'),
                ('event_ptr', events, 'i4'), ('core_mu', core_mu, 'f8'), ('core_weights', core_weights, 'f8')):
            a[name] = np.asarray(values, dtype=dtype); a[name].flags.writeable = False
        self.half_segments = len(steps)
        self.outer_radius = float(geometry.radius[0])
        self.observer_impact = np.array([p[0] for p in self.paths])
        self.observer_core = np.array([p[1] for p in self.paths], dtype=bool)
        self.observer_doppler = np.array([float(A.doppler(geometry, p,
            float(np.sqrt((self.outer_radius-p)*(self.outer_radius+p))), self.outer_radius))
            for p in self.observer_impact])
        for values in (self.observer_impact, self.observer_core, self.observer_doppler):
            values.flags.writeable = False


def disk_weights(impact, core_branch, radius):
    """q=(p/R)^2 사다리꼴. 중복 core의 안쪽 한계가 먼저 와야 한다."""
    p = np.asarray(impact, dtype='f8'); core = np.asarray(core_branch)
    if (not math.isfinite(radius) or radius <= 0 or p.ndim != 1 or len(p) < 2 or
            core.shape != p.shape or core.dtype != np.dtype(bool) or
            not np.isfinite(p).all() or np.any(p < 0) or np.any(p > radius)):
        raise ValueError('관측자 disk 광선 범위 오류')
    order = np.lexsort((~core, p)); q = (p[order]/radius)**2
    if q[0] != 0. or q[-1] != 1.:
        raise ValueError('관측자 disk의 중심 또는 외곽 고리 누락')
    duplicate = np.flatnonzero(np.diff(q) == 0.)
    if (len(duplicate) > 1 or any(not core[order[j]] or core[order[j+1]] for j in duplicate)):
        raise ValueError('core 양쪽 한계가 아닌 중복 impact')
    weights = np.zeros(len(p)); width = .5*np.diff(q)
    weights[order[:-1]] += width; weights[order[1:]] += width
    return weights


class ObserverSpectrum:
    """광선 하나씩 조립한다. 부분 결과나 대역 외 외삽은 반환하지 않는다."""
    def __init__(self, frequency, impact, core_branch, radius, doppler):
        nu = np.array(frequency, dtype='f8', copy=True)
        d = np.array(doppler, dtype='f8', copy=True)
        self.weights = disk_weights(impact, core_branch, radius)
        if (nu.ndim != 1 or len(nu) < 2 or not np.isfinite(nu).all() or np.any(nu <= 0) or
                np.any(np.diff(nu) <= 0) or d.shape != self.weights.shape or
                not np.isfinite(d).all() or np.any(d <= 0)):
            raise ValueError('관측자 주파수/Doppler 범위 오류')
        # 보간 대상 주파수 자체로 검사해 대역 경계의 반올림도 외삽하지 않는다.
        mask = (nu*d.min() >= nu[0]) & (nu*d.max() <= nu[-1])
        self.frequency = nu[mask]
        if len(self.frequency) < 2:
            raise ValueError('모든 광선에 공통인 관측 주파수 대역이 없음')
        self.input_frequency = nu; self.log_frequency = np.log(nu)
        self.doppler = d; self.radius = float(radius)
        self.total = np.zeros(len(self.frequency)); self.correction = self.total.copy()
        self.next_path = 0; self.failed = False

    def add(self, path, intensity):
        if self.failed or type(path) is not int or path != self.next_path or path >= len(self.weights):
            raise ValueError('관측자 광선 순서 또는 중복 입력 오류')
        value = np.asarray(intensity)
        if (value.shape != self.input_frequency.shape or not np.isfinite(value).all() or np.any(value < 0)):
            self.failed = True; raise ValueError('공변계 출사 강도 오류')
        d = self.doppler[path]
        target = self.frequency*d
        if target[0] < self.input_frequency[0] or target[-1] > self.input_frequency[-1]:
            self.failed = True; raise ValueError('관측자 주파수 외삽 거부')
        # log I가 아니라 I를 log nu에 선형 보간한다. I/nu^3 불변량을 적용한다.
        sampled = np.interp(np.log(target), self.log_frequency, value)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            term = sampled/d**3*self.weights[path]
            part = term-self.correction; total = self.total+part
        if (not np.isfinite(total).all() or not np.isfinite(term).all() or
                np.any((sampled > 0) & (self.weights[path] > 0) & (term == 0))):
            self.failed = True; raise ValueError('관측자 강도의 유효 표현 범위 초과')
        self.correction = (total-self.total)-part; self.total = total
        self.next_path += 1

    def finish(self):
        if self.failed or self.next_path != len(self.weights):
            raise ValueError('미완료 관측자 광선을 스펙트럼으로 게시할 수 없음')
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            luminosity = self.total*(4*math.pi**2*self.radius**2)
            per_angstrom = luminosity*(self.frequency/A.C)*self.frequency*1e-8
        if (not np.isfinite(luminosity).all() or not np.isfinite(per_angstrom).all() or
                np.any((self.total > 0) & ((luminosity == 0) | (per_angstrom == 0)))):
            raise ValueError('관측자 광도 단위 변환 범위 초과')
        return dict(frequency_Hz=self.frequency.copy(), wavelength_A=A.C/self.frequency*1e8,
            luminosity_erg_s_Hz=luminosity, luminosity_erg_s_A=per_angstrom,
            paths=self.next_path, frame='OBSERVER_LAB',
            frequency_rule='I linear in log(nu), common covered band only; no extrapolation',
            disk_rule='trapezoid in (impact/R)^2; two-sided core limit',
            error_certificate=False, final_goal='NOT_MET', accepted_physical_spectrum=False)


def run_observer(driver, plan, planck, gradient, *, seconds=5400., interval_seconds=60.,
                 max_full_vectors=2000000, progress=None):
    """실제 수송을 호출하는 명시적 경로. 별도의 실행 계약/예산 아래에서만 호출한다."""
    driver.bound()
    if (not isinstance(plan, ObserverDriverPlan) or plan.source is not driver.provider._source or
            plan.driver_block_width != driver.width or not math.isfinite(seconds) or not 0 < seconds <= 5400. or
            type(max_full_vectors) is not int or not 1 <= max_full_vectors <= 2000000 or
            isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float)) or
            not math.isfinite(interval_seconds) or not 0 < interval_seconds <= getattr(driver, 'interval_limit_seconds', 5400.)):
        raise ValueError('관측자 수송 계획/한도 오류')
    boundary = [np.ascontiguousarray(v, dtype='f8') for v in (planck, gradient)]
    if (any(v.shape != (driver.nf,) or not np.isfinite(v).all() for v in boundary) or np.any(boundary[0] < 0)):
        raise ValueError('관측자 수송의 확산 경계 오류')
    check_host(driver.nf*64)
    spectrum = ObserverSpectrum(plan.source.frequency, plan.observer_impact, plan.observer_core,
                                plan.outer_radius, plan.observer_doppler)
    started = time.monotonic(); work = Work(max_full_vectors*driver.nf, started+seconds)
    work.counts['observer_frequency_columns'] = 0
    intervals = 0; a = plan.arrays
    for path, p in enumerate(a['paths']):
        driver.rt.zero(driver.ray)
        for leg in (0, 1):
            if leg == 1 and p[4]:
                work.take('boundary_columns', driver.nf)
                # 기존 광선 scratch를 경계 주입에만 재사용해 장치 버퍼를 늘리지 않는다.
                driver.rt.upload(driver.characteristic, boundary[0]); driver.rt.upload(driver.candidate, boundary[1])
                check(driver.lib.lumina_history_cuda_boundary(driver.nf, driver.characteristic.address,
                    driver.candidate.address, float(a['core_mu'][path]), driver.ray.address, driver.rt.status))
            begin, count = map(int, p[2*leg:2*leg+2])
            for index in range(begin, begin+count):
                work.check()
                driver.interval(plan, index, work, max_seconds=min(interval_seconds, work.deadline-time.monotonic()))
                intervals += 1
        driver.rt.sync(); work.take('observer_frequency_columns', len(spectrum.frequency))
        spectrum.add(path, driver.rt.download(driver.ray, driver.nf))
        if progress is not None:
            progress(dict(paths_completed=path+1, physical_intervals=intervals, elapsed_seconds=time.monotonic()-started))
    driver.bound(); work.check()
    if intervals != len(a['steps']):
        raise RuntimeError('관측자 물리 구간 누락')
    result = spectrum.finish(); work.check()
    result.update(source_generation=driver.generation, input_identity=driver.input_identity,
        physical_intervals=intervals, elapsed_seconds=time.monotonic()-started,
        parallel_work=work.used, work_counters=work.counts,
        scope='same frozen material/field observer solve, not self-consistent state acceptance')
    return result
