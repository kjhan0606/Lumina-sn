"""한 동결 구간/slab의 전파계수만 재사용한다. 이전 오차값은 재사용하지 않는다."""
import ctypes as ct

import numpy as np

from lumina_frequency_history import Stats
from lumina_nodal_history import NodalKernel


class CachedKernel(NodalKernel):
    def __init__(self, path):
        super().__init__(path)
        p = np.ctypeslib.ndpointer(dtype='f8', ndim=1, flags='C_CONTIGUOUS')
        ip = np.ctypeslib.ndpointer(dtype='u1', ndim=1, flags='C_CONTIGUOUS')
        self._maps = self.library.cmf_nodal_history_maps
        self._maps.argtypes = [ct.c_void_p]+[ct.c_int]*4+[ct.c_double]+[p]*4+[
            ip, ct.POINTER(Stats), ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint64)]
        self._maps.restype = ct.c_int
        self._replay = self.library.cmf_nodal_history_replay
        self._replay.argtypes = [ct.c_void_p]+[ct.c_int]*4+[ct.c_double]+[p]*3+[
            ct.POINTER(ct.POINTER(ct.c_double)), ip]+[p]*4+[ct.c_uint64, ct.POINTER(ct.c_uint64)]
        self._replay.restype = ct.c_int


def expected_calls(deltas, h):
    spans = np.stack((deltas[:, 0]-deltas[:, 2], deltas[:, 0]-deltas[:, 1],
                      deltas[:, 1]-deltas[:, 2]), axis=1)/h
    return 3*len(deltas)+int(np.count_nonzero((spans >= .75)&(spans <= 1.25)))


class CachedComparison:
    """객체는 한 물리 구간의 adaptive 호출 안에서만 만들고 폐기한다."""
    def __init__(self, kernel, provider, begin, width):
        kernel._check_provider(provider)
        if provider._bound is None or type(begin) is not int or type(width) is not int or not 1 <= width <= 512 or begin < 0 or begin+width > provider.nf:
            raise ValueError('재사용 구간/연결 범위 오류')
        self.kernel, self.provider, self.begin, self.width = kernel, provider, begin, width
        self.bindings = provider.report()['bindings']
        self.cache = {}
        self.built_leaves = self.replayed_leaf_columns = self.fallback_vectors = 0
        self.fallback_scalar_formals = self.unsafe_leaf_columns_built = 0
        self.peak_workspace_bytes = 0

    def evaluate(self, lengths, deltas, radii, initial, budget):
        from lumina_local_micro import MicroLimit
        self.kernel._check_provider(self.provider)
        if self.provider.report()['bindings'] != self.bindings:
            self.cache.clear()
            raise ValueError('동결 J 재결합 뒤 전파계수 재사용 금지')
        lengths, deltas, radii, initial = [np.ascontiguousarray(x, dtype='f8') for x in (lengths, deltas, radii, initial)]
        n, w, h = len(lengths), self.width, self.provider._source.dlognu
        if not 1 <= n <= 4096 or n*w > 2097152 or lengths.shape != (n, 3) or deltas.shape != (n, 3) or radii.shape != (n, 5) or initial.shape != (w,):
            raise ValueError('재사용 mesh/입사 차원 오류')
        if not all(np.isfinite(x).all() for x in (lengths, deltas, radii, initial)) or np.any(initial < 0):
            raise ValueError('재사용 mesh/입사 수치 오류')
        keys = [lengths[i].tobytes()+deltas[i].tobytes()+radii[i].tobytes() for i in range(n)]
        if len(set(keys)) != n:
            raise ValueError('재사용 mesh 중복 leaf')
        current = set(keys)
        self.cache = {k: v for k, v in self.cache.items() if k in current}
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        # maps와 작업/복사/오차 배열을 포함한다. 초과하면 부분 장을 반환하지 않는다.
        spans = np.stack((deltas[:, 0]-deltas[:, 2], deltas[:, 0]-deltas[:, 1], deltas[:, 1]-deltas[:, 2]), axis=1)/h
        overlap = (spans >= .75)&(spans <= 1.25)
        dimensions = 6+2*overlap.sum(axis=1)
        batch_size = 64
        estimate = (int(dimensions.sum())+12*min(batch_size, len(missing))+2*n+24)*w*8+n*2048+4*1024**2
        estimate += 2*n*((w+7)//8)+min(batch_size, len(missing))*w
        if estimate > 128*1024**2:
            raise MicroLimit('전파계수 재사용 작업공간128MiB 한도', dict(bytes=estimate))
        self.peak_workspace_bytes = max(self.peak_workspace_bytes, estimate)
        calls = checks = 0; stats = Stats()
        total_stats = {key: 0 for key, _ in Stats._fields_}
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start+batch_size]
            l, d, r = [np.ascontiguousarray(x[batch]) for x in (lengths, deltas, radii)]
            expected = expected_calls(d, h)
            budget.take(expected*w)
            maps = np.empty((len(batch), 12, w)); safe = np.empty((len(batch), w), dtype='u1')
            c, b = ct.c_uint64(), ct.c_uint64()
            code = self.kernel._maps(self.provider._ctx, self.provider.nf, self.begin, w, len(batch), h,
                l.ravel(), d.ravel(), r.ravel(), maps.ravel(), safe.ravel(), ct.byref(stats), ct.byref(c), ct.byref(b))
            if code: raise ValueError('전파계수 생성 실패: '+str(code))
            calls += c.value; checks += b.value
            if c.value != expected: raise ValueError('전파계수 실제 작업량 불일치')
            # leaf마다 소유 복사해 삭제한 부모의 큰 batch가 계속 살아 있지 않게 한다.
            for j, i in enumerate(batch):
                selected = [0, 1, 4, 5, 8, 9]
                for piece in range(3):
                    if overlap[i, piece]: selected.extend((4*piece+2, 4*piece+3))
                self.cache[keys[i]] = (maps[j, selected].copy(), np.packbits(safe[j], bitorder='little'))
                self.unsafe_leaf_columns_built += int(np.count_nonzero(safe[j] == 0))
            self.built_leaves += len(batch)
            for key, _ in Stats._fields_:
                value = int(getattr(stats, key))
                if key in ('queue_peak', 'queue_bytes'): total_stats[key] = max(total_stats[key], value)
                else: total_stats[key] += value
            del maps, safe, l, d, r
        budget.check()
        p = ct.POINTER(ct.c_double)
        pointers = (p*n)(*[self.cache[k][0].ctypes.data_as(p) for k in keys])
        safe = np.array([self.cache[k][1] for k in keys], dtype='u1').ravel()
        value = np.empty(w); errors = np.empty((n, w)); responses = np.empty_like(errors)
        fallbacks = (ct.c_uint64*3)()
        code = self.kernel._replay(self.provider._ctx, self.provider.nf, self.begin, w, n, h,
            lengths.ravel(), deltas.ravel(), radii.ravel(), pointers, safe, initial, value,
            errors.ravel(), responses.ravel(), budget.maximum-budget.used, fallbacks)
        budget.take(int(fallbacks[0]))
        self.fallback_scalar_formals += int(fallbacks[0])
        self.fallback_vectors += int(fallbacks[1])
        self.replayed_leaf_columns += n*w
        if code == 5: raise MicroLimit('재생 fallback의 전체 scalar formal 한도', kind='GLOBAL_WORK')
        if code: raise ValueError('전파계수 순차 재생 실패: '+str(code))
        return value, errors, responses, dict(total_stats,
            formal_vectors=calls+fallbacks[0]/w, scalar_formals=int(calls*w+fallbacks[0]),
            branch_checks=checks+fallbacks[2]/w,
            new_map_formal_vectors=int(calls), fallback_formal_vectors=fallbacks[0]/w,
            fallback_scalar_formals=int(fallbacks[0]), fallback_kernel_entries=int(fallbacks[1]),
            cached_leaves=len(self.cache), built_leaves=self.built_leaves,
            replayed_leaf_columns=self.replayed_leaf_columns,
            unsafe_leaf_columns_built=self.unsafe_leaf_columns_built,
            cache_workspace_peak_bytes=self.peak_workspace_bytes)
