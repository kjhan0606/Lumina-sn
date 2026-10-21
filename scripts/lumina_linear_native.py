"""B의 native 입력 연결과 공간 오차 추정. import는 계산을 시작하지 않는다.

공간 오차는 같은 입사에서 전체/두 반쪽을 비교한 추정이며 인증 상계가 아니다.
구간 적분의 수치 오차와 혼동하지 않는다. 실행 봉인·PHOT 소비는 상위 책임이다.
"""
import ctypes as ct
from dataclasses import dataclass
import math
from pathlib import Path
import time


@dataclass(frozen=True)
class Positive:
    m: float = 0.
    e: int = 0

    def __post_init__(self):
        if not math.isfinite(self.m) or (self.m != 0 and not .5 <= self.m < 1.) or type(self.e) is not int or abs(self.e) >= 2**60:
            raise ValueError('양의 scale 표현 범위 오류')

    @classmethod
    def of(cls, value):
        if not math.isfinite(value) or value < 0:
            raise ValueError('비유한/음수 양의 상태')
        m, e = math.frexp(value)
        return cls(m, e)

    def __add__(self, other):
        if not isinstance(other, Positive): other = Positive.of(other)
        if not self.m: return other
        if not other.m: return self
        a, b = (self, other) if self.e >= other.e else (other, self)
        shift = b.e-a.e
        m, e = math.frexp(a.m+(0. if shift < -1074 else math.ldexp(b.m, shift)))
        return Positive(m, a.e+e)

    def __mul__(self, other):
        if not isinstance(other, Positive): other = Positive.of(other)
        if not self.m or not other.m: return Positive()
        m, e = math.frexp(self.m*other.m)
        return Positive(m, self.e+other.e+e)

    def __le__(self, other):
        if not self.m: return True
        if not other.m: return False
        return (self.e, self.m) <= (other.e, other.m)

    def difference(self, other):
        a, b = (other, self) if self <= other else (self, other)
        if not b.m: return a
        shift = b.e-a.e
        m, e = math.frexp(abs(a.m-(0. if shift < -1074 else math.ldexp(b.m, shift))))
        return Positive(m, a.e+e) if m else Positive()

    def value(self):
        if not self.m or self.e < -1074: return 0.
        if self.e > 1024: raise OverflowError('마지막 관측 변환 범위 초과')
        value = math.ldexp(self.m, self.e)
        if not math.isfinite(value): raise OverflowError('마지막 관측 변환 비유한')
        return value


EPS = float.fromhex('0x1p-52')


@dataclass(frozen=True)
class Map:
    t: Positive
    source: Positive
    source_error: Positive
    tau_error: float

    def __post_init__(self):
        if not math.isfinite(self.tau_error) or not 0 <= self.tau_error <= .25:
            raise ValueError('T 오차의 보증 가능한 범위 초과')

    @property
    def upper_t(self):
        if not math.isfinite(self.tau_error) or not 0 <= self.tau_error <= .25:
            raise ValueError('T 오차의 보증 가능한 범위 초과')
        return self.t*math.exp(self.tau_error)

    def apply(self, initial):
        response = self.t*initial
        value = response+self.source
        error = self.source_error+response*math.expm1(self.tau_error)+value*(32*EPS)
        return value, error

    def then(self, right):
        source = right.t*self.source+right.source
        error = right.upper_t*self.source_error+right.source_error+\
            (right.t*self.source)*math.expm1(right.tau_error)+source*(32*EPS)
        return Map(self.t*right.t, source, error, self.tau_error+right.tau_error+8*EPS)


class Location(ct.Structure):
    _fields_ = [('mode', ct.c_int), ('first', ct.c_int), ('last', ct.c_int), ('weight', ct.c_double)]


class CPositive(ct.Structure):
    _fields_ = [('m', ct.c_double), ('e', ct.c_int64)]
    def positive(self):
        if not math.isfinite(self.m) or (self.m != 0 and not .5 <= self.m < 1.) or abs(self.e) >= 2**60:
            raise ValueError('native B 양의 상태 ABI/값 오류')
        return Positive(self.m, self.e)


class NativeMap(ct.Structure):
    _fields_ = [(name, CPositive) for name in ('transmission', 'source', 'source_error', 'source_roundoff',
        'source_frequency', 'source_omitted')]+[('tau_error', ct.c_double)]+[
        (name, ct.c_uint64) for name in ('built_bins', 'tree_combinations')]+[
        (name, ct.c_int) for name in ('panels', 'accepted', 'split', 'pruned', 'max_depth')]


class NativeKernel:
    def __init__(self, library):
        self.path = Path(library).resolve(); self.lib = ct.CDLL(str(self.path))
        size = self.lib.cmf_linear_native_map_bytes; size.argtypes = []; size.restype = ct.c_size_t
        if size() != ct.sizeof(NativeMap): raise ValueError('native B 맵 ABI 크기 불일치')
        self.fun = self.lib.cmf_linear_native_map
        self.fun.argtypes = [ct.c_void_p, ct.c_uint64, ct.POINTER(Location)]+[ct.c_double]*4+[
            ct.c_int, ct.c_int, ct.c_double, ct.c_int, ct.POINTER(NativeMap)]
        self.fun.restype = ct.c_int

    def map(self, view, locations, length, h, da, db, column, *, relative=1e-8, max_panels=100000):
        if len(locations) != 2: raise ValueError('입구/출구 두 위치 필요')
        loc = (Location*2)(*locations); result = NativeMap()
        code = self.fun(ct.byref(view), int(view.generation), loc, length, h, da, db, column, 32,
            relative, max_panels, ct.byref(result))
        if code:
            piece = dict(code=code, column=column, length=length, h=h, da=da, db=db, relative=relative,
                locations=[dict(mode=p.mode, first=p.first, last=p.last, weight=p.weight) for p in locations])
            raise NativePieceFailure(piece)
        return Map(result.transmission.positive(), result.source.positive(), result.source_error.positive(),
                   result.tau_error), result


class NativePieceFailure(ValueError):
    def __init__(self, piece):
        super().__init__('native B 공간선형 조각 실패: '+str(piece['code'])); self.piece = piece


class SpatialFailure(RuntimeError):
    def __init__(self, message, last_completed):
        super().__init__(message); self.last_completed = last_completed


class NativeInterval:
    """기존 fullSR 기하 C API만 재사용한다. 과거 C 수송 연산자는 호출하지 않는다."""
    def __init__(self, kernel, provider, plan, index, *, max_calls=30000, max_panels=20000000):
        import numpy as np
        if plan.source is not provider._source or type(index) is not int or not 0 <= index < len(plan.arrays['steps']):
            raise ValueError('같은 native source의 실제 구간 필요')
        if provider.identity.get('field', {}).get('role', '').startswith('REFERENCE'):
            raise ValueError('참조 J를 자체 수송 입력으로 사용 금지')
        self.kernel, self.provider, self.plan, self.index = kernel, provider, plan, index
        self.view = provider.borrow_dense_view(); self.generation = int(self.view.generation)
        self.cpu = ct.CDLL(str(provider._binary)); self.geometry = self.cpu.cmf_nodal_driver_geometry
        self.geometry.argtypes = [ct.c_void_p, ct.c_int]+[ct.c_void_p]*5+[ct.c_double, ct.c_int]+[ct.c_void_p]*6
        self.geometry.restype = ct.c_int
        self.geometries = {}; self.maps = {}; self.calls = self.panels = self.bins = self.combinations = 0
        self.max_calls, self.max_panels = max_calls, max_panels
        self.np = np

    def check(self):
        if int(self.provider.borrow_dense_view().generation) != self.generation:
            raise ValueError('native J 생성번호가 바뀐 기하/계수 cache')

    def make_maps(self, column):
        if type(column) is not int or not 0 <= column < self.view.nf: raise ValueError('원래 주파수 열 범위')
        maps = {}  # 이 closure의 열만 사용해 교차 열 cache 오염을 막는다.
        def make(a, b):
            self.check(); key = (a, b); m = .5*(a+b)
            if key not in self.geometries:
                np = self.np; arrays = self.plan.arrays
                edges = np.array([[a, b]], dtype='f8'); lengths = np.empty(3); deltas = np.empty(3); radii = np.empty(5)
                locations = (Location*5)(); shift = ct.c_double()
                code = self.geometry(self.provider._ctx, len(arrays['radius']),
                    *[arrays[k].ctypes.data for k in ('radius', 'velocity', 'gauss_x', 'gauss_w')],
                    arrays['steps'][self.index].ctypes.data, self.provider._source.dlognu, 1, edges.ctypes.data,
                    lengths.ctypes.data, deltas.ctypes.data, radii.ctypes.data, ct.addressof(locations), ct.byref(shift))
                if code: raise ValueError('기존 fullSR 기하 API 실패: '+str(code))
                self.geometries[key] = lengths, deltas, locations
            lengths, deltas, locations = self.geometries[key]
            for i, (u, v, start, end) in enumerate(((a, b, 0, 2), (a, m, 0, 1), (m, b, 1, 2))):
                if (u, v) in maps: continue
                if self.calls >= self.max_calls: raise RuntimeError('native B 맵 호출 한도')
                try:
                    result, stats = self.kernel.map(self.view, (locations[2*start], locations[2*end]), float(lengths[i]),
                        self.provider._source.dlognu, float(deltas[start]), float(deltas[end]), column)
                except NativePieceFailure as error:
                    error.piece['segment'] = [u, v]
                    raise
                self.calls += 1; self.panels += stats.panels; self.bins += stats.built_bins
                self.combinations += stats.tree_combinations
                if self.panels > self.max_panels: raise RuntimeError('native B 내부 원천 panel 한도')
                maps[u, v] = result
            return maps[a, b], maps[a, m], maps[m, b]
        return make


def spatial_interval(make_maps, *, initial=Positive(), relative=1e-5, seconds=60., max_leaves=4096, max_depth=32):
    """make_maps(a,b)는 coarse/left/right 맵을 반환한다. 열마다 독립 수용한다.

    같은 관측 주파수 특성선을 끝까지 합성하며 소구간 사이의 입사 재보간은 없다.
    앞 구간 오차는 뒤의 T 상계로 전파한다. 성공 반환만 완성 구간으로 소비한다.
    """
    if not math.isfinite(relative) or not 0 < relative <= .01 or not math.isfinite(seconds) or not 0 < seconds <= 60.:
        raise ValueError('공간 적응 범위 오류')
    if type(max_leaves) is not int or not 1 <= max_leaves <= 4096 or type(max_depth) is not int or not 0 <= max_depth <= 32:
        raise ValueError('공간 leaf/깊이 한도 오류')
    started = time.monotonic(); leaves = [(0., 1., 0)]; rounds = 0; last = None
    def check():
        if time.monotonic()-started >= seconds: raise SpatialFailure('공간 구간 시간 한도', last)
    while True:
        check(); value = initial; numeric = Positive(); local = []; responses = []
        for a, b, depth in leaves:
            check(); coarse, left, right = make_maps(a, b); fine = left.then(right)
            cv, ce = coarse.apply(value); fv, fe = fine.apply(value)
            numeric = fine.upper_t*numeric+fe
            local.append(cv.difference(fv)+ce+fe+(cv+fv)*(16*EPS)); responses.append(fine.upper_t)
            value = fv
        check(); tail = Positive.of(1.); geometry = Positive(); scores = [Positive()]*len(leaves)
        for i in range(len(leaves)-1, -1, -1):
            scores[i] = local[i]*tail; geometry = geometry+scores[i]; tail = tail*responses[i]
        error = geometry+numeric; rounds += 1
        last = dict(round=rounds, leaves=len(leaves), value=value.value(), numerical_error=numeric.value(),
            geometry_estimate=geometry.value(), estimated_total_error=error.value(), tolerance=(value*relative).value(),
            elapsed_seconds=time.monotonic()-started, error_certificate=False,
            normalization='each output frequency; no unrelated global maximum or intensity floor')
        if error <= value*relative:
            return value, dict(last, status='SPATIAL_ESTIMATE_MET_NOT_FIELD', boundaries=[x[0] for x in leaves]+[1.])
        worst = max(scores, key=lambda x: (bool(x.m), x.e, x.m)); threshold = worst*.5
        selected = [threshold <= x for x in scores]
        if len(leaves)+sum(selected) > max_leaves: raise SpatialFailure('공간 leaf 한도', last)
        updated = []
        for choose, (a, b, depth) in zip(selected, leaves):
            if not choose: updated.append((a, b, depth)); continue
            mid = .5*(a+b)
            if depth >= max_depth or mid <= a or mid >= b: raise SpatialFailure('공간 깊이/정밀도 한도', last)
            updated.extend(((a, mid, depth+1), (mid, b, depth+1)))
        leaves = updated
