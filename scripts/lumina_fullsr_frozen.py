"""중점 물질 동결·fullSR 특성선 이동창의 명시적 소형 연결. 기본 생산 경로는 바꾸지 않는다."""
import ctypes as ct
from pathlib import Path

import numpy as np

from lumina_fullsr_sc import SCALED

AFFINE = np.dtype([('transmission', SCALED), ('source', SCALED)], align=True)


class Stats(ct.Structure):
    _fields_ = [(k, ct.c_uint64) for k in ('primitive_maps', 'compositions', 'completed_outputs', 'heap_bytes')]+[
        ('elapsed_seconds', ct.c_double), ('status', ct.c_int), ('failed_column', ct.c_int)]


class FrozenKernel:
    def __init__(self, library, *, linear_chi=False):
        self.library = ct.CDLL(str(Path(library).resolve()))
        self.function = getattr(self.library, 'cmf_fullsr_frozen_linear_cell' if linear_chi else 'cmf_fullsr_frozen_cell')
        self.function.argtypes = [ct.c_void_p]*2+[ct.c_double]*3+[ct.c_int]*3+[
            ct.c_void_p]*2+[ct.c_double, ct.c_void_p]
        self.function.restype = ct.c_int
        self.begin_function = getattr(self.library, 'cmf_fullsr_frozen_begin', None)
        self.fixed_function = getattr(self.library, 'cmf_fullsr_frozen_fixed_cell', None)
        self.chain_function = getattr(self.library, 'cmf_fullsr_linear_chain', None)
        if self.begin_function is not None:
            self.begin_function.argtypes = [ct.c_int]+[ct.c_double]*2+[ct.c_int]*2+[
                ct.c_void_p]*2+[ct.c_double, ct.c_void_p]
            self.begin_function.restype = ct.c_int
        if self.fixed_function is not None:
            self.fixed_function.argtypes = [ct.c_void_p]*2+[ct.c_double]*4+[ct.c_int]*2+[
                ct.c_void_p]*2+[ct.c_double, ct.c_void_p]
            self.fixed_function.restype = ct.c_int
        if self.chain_function is not None:
            self.chain_function.argtypes = [ct.c_int]*2+[ct.c_void_p]*7+[ct.c_double]+[ct.c_void_p]*2
            self.chain_function.restype = ct.c_int

    @staticmethod
    def _buffers(nf, incident, output):
        if output is None: output = np.zeros(nf, dtype=SCALED)
        for a in (incident, output):
            if not isinstance(a, np.ndarray) or a.shape != (nf,) or a.dtype != SCALED or not a.flags.c_contiguous:
                raise ValueError('전 native 주파수의 연속 scaled 강도 필요')
        if not output.flags.writeable: raise ValueError('출력 쓰기 권한 필요')
        return output

    @staticmethod
    def _report(stats, code):
        report = {key: getattr(stats, key) for key, _ in Stats._fields_}
        report.update(published_outputs=stats.completed_outputs if code == 0 else 0,
                      accepted_physical_field=False, final_goal='NOT_MET')
        return report

    def begin(self, provider, delta, incident, *, output=None, reconstruction=1, last=0, max_seconds=120.):
        if self.begin_function is None: raise ValueError('고정 출구 프레임 입구 API가 없는 바이너리')
        view = provider.borrow_dense_view(); output = self._buffers(view.nf, incident, output); stats = Stats()
        code = self.begin_function(view.nf, provider._source.dlognu, delta, reconstruction, last,
            incident.ctypes.data, output.ctypes.data, max_seconds, ct.byref(stats))
        return code, output, self._report(stats, code)

    def chain(self, lengths, deltas, chi, eta, initial, *, max_seconds=120.):
        if self.chain_function is None: raise ValueError('실제 경로 계수 chain API가 없는 바이너리')
        lengths, deltas, chi, eta = [np.ascontiguousarray(a, dtype='f8') for a in (lengths, deltas, chi, eta)]
        nstep = len(lengths); nray = len(initial)
        if (not 1 <= nstep <= 65536 or not 1 <= nray <= 512 or nstep*nray > 2000000 or
                lengths.shape != (nstep,) or deltas.shape != (nstep,) or chi.shape != (nstep+1, nray) or
                eta.shape != chi.shape or initial.shape != (nray,) or initial.dtype != SCALED or
                not initial.flags.c_contiguous): raise ValueError('선형 chain 배열 크기/dtype 오류')
        output = np.zeros(nray, dtype=SCALED); maps = np.zeros((nstep, nray), dtype=AFFINE)
        stats = Stats(); failed = ct.c_int(-1)
        code = self.chain_function(nstep, nray, *[a.ctypes.data for a in (lengths, deltas, chi, eta, initial, output, maps)],
                                   max_seconds, ct.byref(stats), ct.byref(failed))
        report = self._report(stats, code); report['failed_step'] = failed.value
        return code, output, maps, report

    def fixed_cell(self, provider, location, length, da, db, incident, *, output=None,
                   bin_subdivisions=1, last=0, max_seconds=120.):
        if self.fixed_function is None: raise ValueError('고정 출구 프레임 cell API가 없는 바이너리')
        if 'REFERENCE' in getattr(getattr(provider, '_field', None), 'identity', {}).get('role', ''):
            raise ValueError('참조 J는 자체 수송 source로 사용하지 않는다')
        view = provider.borrow_dense_view(); output = self._buffers(view.nf, incident, output); stats = Stats()
        code = self.fixed_function(ct.byref(view), ct.byref(location), length, da, db, provider._source.dlognu,
            bin_subdivisions, last, incident.ctypes.data, output.ctypes.data, max_seconds, ct.byref(stats))
        return code, output, self._report(stats, code)

    def cell(self, provider, location, length, delta, incident, *, output=None,
             bin_subdivisions=1, reconstruction=1, last=0, max_seconds=120.):
        if 'REFERENCE' in getattr(getattr(provider, '_field', None), 'identity', {}).get('role', ''):
            raise ValueError('참조 J는 자체 수송 source로 사용하지 않는다')
        view = provider.borrow_dense_view()
        output = self._buffers(view.nf, incident, output)
        stats = Stats()
        code = self.function(ct.byref(view), ct.byref(location), length, delta, provider._source.dlognu,
            bin_subdivisions, reconstruction, last, incident.ctypes.data, output.ctypes.data,
            max_seconds, ct.byref(stats))
        return code, output, self._report(stats, code)
