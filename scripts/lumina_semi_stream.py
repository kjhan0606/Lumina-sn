"""전역ν 점값/8점 halo를 보존하는 제한 메모리 물리 구간 수송.

전ν×8² 캐시가 없다. 블록 끝은 물리 경계가 아니며, 실제 실행 권한은 별도다.
소형 C 적분을 명시적으로 받아 사용한다. 부분 실패를 완료 장으로 반환하지 않는다.
"""
import ctypes
from functools import lru_cache
import time

import numpy as np

from lumina_semi_lagrangian import _bernstein_matrix, _positive_polynomial


def partition(count, width=32):
    if type(count) is not int or not 8 <= count <= 2000000 or type(width) is not int or not 1 <= width <= 32:
        raise ValueError('전역ν/블록 차원 오류')
    for begin in range(0, count, width):
        yield np.arange(begin, min(begin+width, count), dtype=np.int64)


def _indices(values, count):
    indices = np.asarray(values)
    if (indices.ndim != 1 or not 1 <= indices.size <= 32 or indices.dtype.kind not in 'iu' or
            np.any(indices < 0) or np.any(indices >= count)):
        raise ValueError('출력 전역ν 인덱스 오류')
    return indices.astype(np.int64, copy=False)


def _read(incident, indices):
    selected, inverse = np.unique(indices, return_inverse=True)
    value = np.asarray(incident(selected), dtype='f8')
    if value.shape != selected.shape or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError('불변 입사 ray의 전역ν 요청 값 오류')
    return value[inverse]


@lru_cache(maxsize=1)
def _bank():
    bank = np.stack([_bernstein_matrix(offset, 8) for offset in range(-6, 1)])
    bank.flags.writeable = False
    return bank


def reconstruct_columns(count, indices, incident, shift):
    if type(count) is not int or not 8 <= count <= 2000000 or not callable(incident) or not np.isfinite(shift) or shift < 0:
        raise ValueError('전역ν 고차 재구성 입력 오류')
    indices = _indices(indices, count)
    if shift >= count-1:
        return np.full(len(indices), _read(incident, np.array([count-1]))[0]), 0
    whole = int(np.floor(shift)); fraction = float(shift-whole)
    left = np.minimum(indices+whole, count-1)
    result = _read(incident, left)
    active = left < count-1
    if fraction == 0 or not np.any(active): return result, 0
    local = left[active]; first = np.minimum(np.maximum(local-3, 0), count-8)
    stencil = first[:, None]+np.arange(8)[None, :]
    data = _read(incident, stencil.ravel()).reshape(stencil.shape)
    matrices = _bank()[first-local+6]
    value, limited = _positive_polynomial(data, data[np.arange(len(local)), local-first],
        data[np.arange(len(local)), local+1-first], fraction, matrices)
    result[active] = value
    if not np.isfinite(result).all() or np.any(result < 0): raise ValueError('전역ν 재구성 범위 오류')
    return result, limited


def load(path):
    library = ctypes.CDLL(str(path)); function = library.lumina_small_uncoupled_characteristic
    pointer = np.ctypeslib.ndpointer(dtype='f8', ndim=1, flags='C_CONTIGUOUS')
    function.argtypes = [ctypes.c_int, ctypes.c_double]+[pointer]*6
    function.restype = ctypes.c_int
    def segment(length, chi, eta0, eta1, initial):
        arrays = [np.ascontiguousarray(v, dtype='f8') for v in (eta0, eta1, initial)]
        nf = arrays[-1].size; opacity = np.asarray(chi, dtype='f8')
        if not 1 <= nf <= 32 or opacity.shape != (2, nf) or any(v.shape != (nf,) for v in arrays):
            raise ValueError('비결합 블록 적분 차원 오류')
        first, last = (np.ascontiguousarray(v) for v in opacity)
        output = np.empty(nf)
        code = function(nf, length, first, last, *arrays, output)
        if code: raise ValueError('비결합 블록 적분 실패: '+str(code))
        return output
    return segment


def coefficient_columns(count, dlog, position, delta, indices, coefficients):
    """실제 formal 없이도 검사할 수 있는 이동ν χ/총η 요청 연결."""
    indices = _indices(indices, count)
    if not np.isfinite(delta) or delta < 0 or not np.isfinite(dlog) or dlog <= 0:
        raise ValueError('이동 계수의 주파수 간격/드리프트 오류')
    target = indices.astype('f8')+delta/dlog
    if not np.isfinite(target).all(): raise ValueError('이동 계수의 전역 인덱스 수치 범위 초과')
    raw_left = np.floor(target)
    left = np.minimum(raw_left, count-1).astype(np.int64); right = np.minimum(left+1, count-1)
    fraction = target-raw_left
    first, last = [tuple(np.asarray(v, dtype='f8') for v in coefficients(position, part)) for part in (left, right)]
    for pair in (first, last):
        if len(pair) != 2 or any(v.shape != indices.shape or not np.isfinite(v).all() for v in pair) or np.any(pair[1] < 0):
            raise ValueError('이동 전역ν의 χ/총η 요청 오류')
    return (first[0]+fraction*(last[0]-first[0]),
            (first[1]+fraction*(last[1]-first[1]))*np.exp(-3*delta))


def propagate(segment, dlog, count, incident, geometry, coefficients, remaining_drift, *,
              panels=1, block_width=32, max_seconds=120.):
    """coefficients(x,global_indices)는 한 동결 세대의 χ/총η 요청 열만 반환한다."""
    if (type(count) is not int or not 8 <= count <= 2000000 or
            type(block_width) is not int or not 1 <= block_width <= 32 or
            not np.isfinite(dlog) or dlog <= 0 or type(panels) is not int or not 1 <= panels <= 256 or
            not np.isfinite(max_seconds) or not 0 < max_seconds <= 120 or
            not all(callable(v) for v in (segment, incident, geometry, coefficients, remaining_drift))):
        raise ValueError('전역ν 물리 구간 입력/상한 오류')
    parts = partition(count, block_width); started = time.monotonic()
    x = [i/panels for i in range(panels+1)]
    delta = [float(remaining_drift(v)) for v in x]
    if (not np.isfinite(delta).all() or min(delta) < 0 or delta[-1] != 0 or
            any(a < b for a, b in zip(delta[:-1], delta[1:]))):
        raise ValueError('공변 Doppler 경로 오류')
    lengths = []
    for i in range(panels):
        length, drift = map(float, geometry(x[i], x[i+1]))
        if (not np.isfinite(length) or length < 0 or not np.isfinite(drift) or drift < 0 or
                (length == 0 and drift != 0)):
            raise ValueError('공변 길이/주파수 이동 오류')
        lengths.append(length)
    output = np.empty(count); limited = blocks = completed = 0
    source = lambda position, value, indices: coefficient_columns(count, dlog, position, value, indices, coefficients)
    for indices in parts:
        if time.monotonic()-started > max_seconds: raise TimeoutError('물리 구간 소형 시간 상한')
        current, nlimited = reconstruct_columns(count, indices, incident, delta[0]/dlog)
        limited += nlimited; current *= np.exp(-3*delta[0])
        previous = source(x[0], delta[0], indices)
        for i in range(panels):
            if time.monotonic()-started > max_seconds: raise TimeoutError('물리 구간 소형 시간 상한')
            next_source = source(x[i+1], delta[i+1], indices)
            current = segment(lengths[i], np.stack((previous[0], next_source[0])), previous[1], next_source[1], current)
            previous = next_source
        output[indices] = current; blocks += 1; completed += len(indices)
    return output, dict(frequencies=count, completed_bins=completed, frequency_blocks=blocks,
        block_width=block_width, physical_interval_reconstructions=1, limited_polynomials=limited,
        formal_steps=panels*blocks, scalar_formal_steps=panels*count, interior_intensity_remappings=0,
        frequency_crossing_cuts=0, output_bytes=output.nbytes,
        reconstruction_matrix_bank_bytes=_bank().nbytes,
        reconstruction_workspace_budget_bytes=block_width*(8*8+5*8+12)*8,
        memory_scope='고정 수치 배열의 예산; 출력/제공자/Python/RSS와 구별',
        elapsed_seconds=time.monotonic()-started, accepted_physical_field=False, final_goal='NOT_JUDGED')
