"""기존 C 광선 기하로 실제 phase 절점을 만든다. 계수나 J는 준비하지 않는다."""
import ctypes as ct
from pathlib import Path
import time

import numpy as np

from lumina_compiled_input import CompiledInput
from lumina_fullsr_chain_gpu import LOCATION_DTYPE
from lumina_fullsr_sc import Location
from lumina_local_micro import ZeroTrial
from lumina_native_columns import sha
import lumina_outer_atmosphere as A


class PhaseGeometry:
    def __init__(self, library, geometry, source):
        self.identity = sha(Path(library))
        self.library = ct.CDLL(str(Path(library).resolve()))
        self.provider = CompiledInput(library, geometry, source, ZeroTrial(source), slab_width=8, cache_rows=4, profiles=8)
        self.radius = np.ascontiguousarray(geometry.radius, dtype='f8')
        self.velocity = np.ascontiguousarray(geometry.velocity, dtype='f8')
        self.gx, self.gw = A.gauss_rule(8); self.h = source.dlognu
        self.function = self.library.cmf_nodal_driver_phase_grid
        self.function.argtypes = [ct.c_void_p, ct.c_int]+[ct.c_void_p]*5+[
            ct.c_double, ct.c_int, ct.c_int]+[ct.c_void_p]*8
        self.function.restype = ct.c_int

    def grid(self, step, subdivisions=4):
        step = np.ascontiguousarray(step, dtype='f8')
        if step.shape != (5,) or type(subdivisions) is not int: raise ValueError('phase 기하 입력 크기/세분 오류')
        before = time.monotonic(); n = ct.c_int()
        args = [self.provider._ctx, len(self.radius)]+[a.ctypes.data for a in (self.radius, self.velocity, self.gx, self.gw, step)]+[
            self.h, subdivisions]
        code = self.function(*args, 0, ct.byref(n), *[None]*7)
        if code: raise ValueError('C phase 기하 크기 조회 실패: '+str(code))
        x, radius, phase = [np.empty(n.value+1, dtype='f8') for _ in range(3)]
        length, drift = [np.empty(n.value, dtype='f8') for _ in range(2)]
        locations = (Location*(n.value+1))(); error = ct.c_double()
        code = self.function(*args, n.value+1, ct.byref(n), x.ctypes.data, radius.ctypes.data,
            length.ctypes.data, drift.ctypes.data, phase.ctypes.data, ct.byref(locations), ct.byref(error))
        if code: raise ValueError('C phase 기하 생성 실패: '+str(code))
        return dict(x=x, radius=radius, length=length, drift=drift, phase=phase,
                    locations=np.frombuffer(locations, dtype=LOCATION_DTYPE),
                    max_phase_roundoff=error.value, elapsed_seconds=time.monotonic()-before)

    def close(self): self.provider.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
