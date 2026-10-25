"""전 주파수 이력을 C에서 합성하는 별도 후보 로더.

χ의 공간동결 오차는 남는다. 통과 bin을 빠짐없이 적분하는 것과 공간/최종
물리 정확성을 구별한다. 기존 생산 라이브러리를 자동 교체하거나 실행하지 않는다.
"""
import ctypes as ct

import numpy as np


class Map(ct.Structure):
    _fields_ = [(key, ct.c_double) for key in ('d', 't', 'aa', 'pa', 'ab', 'mb')]


class Stats(ct.Structure):
    _fields_ = [(key, ct.c_uint64) for key in ('full_bin_maps', 'partial_bin_maps', 'blue_maps',
        'compositions', 'pushes', 'transfers', 'queue_peak', 'queue_bytes',
        'quadrature_nodes', 'quadrature_panels', 'slabs')]


def values(mapping):
    return np.array([getattr(mapping, key) for key, _ in Map._fields_])


class Kernel:
    def __init__(self, path):
        self.library = ct.CDLL(str(path))
        ptr = np.ctypeslib.ndpointer(dtype='f8', ndim=1, flags='C_CONTIGUOUS')
        self._apply = self.library.cmf_frequency_history
        self._apply.argtypes = [ct.c_int]+[ct.c_double]*4+[ptr]*5+[ct.c_int]*2+[ct.POINTER(Stats)]
        self._apply.restype = ct.c_int
        self._map = self.library.cmf_history_map
        self._map.argtypes = [ct.c_double]*8+[ct.POINTER(Map), ct.POINTER(Stats)]
        self._map.restype = ct.c_int
        self._compose = self.library.cmf_history_compose
        self._compose.argtypes = [Map, Map, ct.POINTER(Map)]
        self._compose.restype = ct.c_int
        self._reconstruct = self.library.cmf_history_reconstruct
        self._reconstruct.argtypes = [ct.c_int, ct.c_double, ptr, ptr, ct.POINTER(ct.c_uint64)]
        self._reconstruct.restype = ct.c_int
        self._response = self.library.cmf_frequency_history_response
        self._response.argtypes = [ct.c_int]+[ct.c_double]*4+[ptr]*5+[ct.c_int]*2+[ptr, ct.POINTER(Stats)]
        self._response.restype = ct.c_int
        self._compare = self.library.cmf_frequency_history_compare
        self._compare.argtypes = [ct.c_int,ct.c_int,ct.c_double]+[ptr]*8+[ct.POINTER(Stats)]
        self._compare.restype = ct.c_int
        self._small = self.library.cmf_frequency_small_spatial
        self._small.argtypes = [ct.c_int]+[ct.c_double]*4+[ptr]*7+[ct.POINTER(Stats)]
        self._small.restype = ct.c_int
        self._hybrid = self.library.cmf_frequency_hybrid_compare
        self._hybrid.argtypes = [ct.c_int,ct.c_int,ct.c_double]+[ptr]*8+[ct.POINTER(Stats),ct.POINTER(ct.c_uint64),ct.POINTER(ct.c_uint64)]
        self._hybrid.restype = ct.c_int

    def reconstruct(self, initial, shift):
        initial = np.ascontiguousarray(initial, dtype='f8')
        if initial.ndim != 1 or not 8 <= len(initial) <= 2000000: raise ValueError('전역8점 재구성 차원 오류')
        value = np.empty_like(initial); limited = ct.c_uint64()
        code = self._reconstruct(len(initial), shift, initial, value, ct.byref(limited))
        if code: raise ValueError('전역8점 재구성 실패: '+str(code))
        return value, dict(reconstructions=1, limited_polynomials=int(limited.value))

    def physical(self, h, length, drift, chi, eta_a, eta_b, initial, **kwargs):
        if not np.isfinite(h) or h <= 0: raise ValueError('양의 logν 간격이 필요하다')
        initial, interpolation = self.reconstruct(initial, drift/h)
        value, report = self.apply(h, length, drift, 0., chi, eta_a, eta_b, initial, **kwargs)
        return value, dict(report, **interpolation)

    def compare(self, h, lengths, deltas, chi, eta, initial):
        lengths,deltas,chi,eta,initial = [np.ascontiguousarray(v,dtype='f8') for v in (lengths,deltas,chi,eta,initial)]
        nf=initial.size;count=len(lengths)
        if (initial.shape != (nf,) or not 1<=count<=4096 or not 1<=nf<=2000000 or count*nf>2097152 or
                lengths.shape!=(count,3) or deltas.shape!=(count,3) or chi.shape!=(count,3,nf) or eta.shape!=chi.shape):
            raise ValueError('작은 공간 적응 비교 차원 오류')
        out=np.empty(nf);errors=np.empty((count,nf));responses=np.empty_like(errors);stats=Stats()
        code=self._compare(count,nf,h,*[v.ravel() for v in (lengths,deltas,chi,eta,initial,out,errors,responses)],ct.byref(stats))
        if code: raise ValueError('주파수 이력 공간 비교 실패: '+str(code))
        return out,errors,responses,{key:int(getattr(stats,key)) for key,_ in Stats._fields_}

    def map(self, length, normalization, k0, k1, ea0, ea1, eb0, eb1):
        output = Map(); stats = Stats()
        code = self._map(length, normalization, k0, k1, ea0, ea1, eb0, eb1, ct.byref(output), ct.byref(stats))
        if code: raise ValueError('이력 거리 모멘트 입력/수치 범위 실패: '+str(code))
        return output

    def small(self,h,length,delta_a,delta_b,chi_a,chi_b,eta_a,eta_b,initial):
        arrays=[np.ascontiguousarray(v,dtype='f8') for v in (chi_a,chi_b,eta_a,eta_b,initial)]
        nf=arrays[-1].size
        if not 1<=nf<=2000000 or any(v.shape!=(nf,) for v in arrays):
            raise ValueError('작은 이동의 공간χ 차원 오류')
        out=np.empty(nf);response=np.empty(nf);stats=Stats()
        code=self._small(nf,h,length,delta_a,delta_b,*arrays,out,response,ct.byref(stats))
        if code:raise ValueError('작은 이동의 공간χ 적분 실패: '+str(code))
        return out,response,{key:int(getattr(stats,key)) for key,_ in Stats._fields_}

    def compare_hybrid(self,h,lengths,deltas,chi,eta,initial):
        lengths,deltas,chi,eta,initial=[np.ascontiguousarray(v,dtype='f8') for v in (lengths,deltas,chi,eta,initial)]
        nf=initial.size;count=len(lengths)
        if (initial.shape!=(nf,) or not 1<=count<=4096 or not 1<=nf<=2000000 or count*nf>2097152 or
                lengths.shape!=(count,3) or deltas.shape!=(count,3) or chi.shape!=(count,5,nf) or eta.shape!=(count,3,nf)):
            raise ValueError('작은 혼합 공간 적응 차원 오류')
        out=np.empty(nf);errors=np.empty((count,nf));responses=np.empty_like(errors);stats=Stats()
        calls=ct.c_uint64();checks=ct.c_uint64()
        code=self._hybrid(count,nf,h,*[v.ravel() for v in (lengths,deltas,chi,eta,initial,out,errors,responses)],
            ct.byref(stats),ct.byref(calls),ct.byref(checks))
        if code:raise ValueError('주파수 이력 혼합 비교 실패: '+str(code))
        report={key:int(getattr(stats,key)) for key,_ in Stats._fields_}
        return out,errors,responses,dict(report,formal_vectors=int(calls.value),branch_checks=int(checks.value))

    def compose(self, left, right):
        output = Map(); code = self._compose(left, right, ct.byref(output))
        if code: raise ValueError('이력 모멘트 합성 실패: '+str(code))
        return output

    def apply(self, h, length, delta_a, delta_b, chi, eta_a, eta_b, characteristic_intensity, *, slab_width=256, naive=False):
        arrays = [np.ascontiguousarray(v, dtype='f8') for v in (chi, eta_a, eta_b, characteristic_intensity)]
        nf = arrays[-1].size
        if (not 1 <= nf <= 2000000 or any(v.shape != (nf,) for v in arrays) or
                type(slab_width) is not int or not 1 <= slab_width <= 65536 or type(naive) is not bool):
            raise ValueError('전역 주파수 이력 차원 오류')
        output = np.empty(nf); stats = Stats()
        code = self._apply(nf, h, length, delta_a, delta_b, *arrays, output,
            slab_width, int(naive), ct.byref(stats))
        if code: raise ValueError('주파수 이력 입력/수치 범위 실패: '+str(code))
        report = {key:int(getattr(stats, key)) for key, _ in Stats._fields_}
        report.update(frequencies=nf, algorithm='naive direct bins' if naive else 'positive rolling moments',
            intensity_grid='already on the moving characteristic; no incident interpolation here',
            spatial_chi='frozen within this spatial subpiece; not a spatial error certificate',
            interior_intensity_remappings=0, accepted_physical_field=False, final_goal='NOT_JUDGED')
        return output, report
