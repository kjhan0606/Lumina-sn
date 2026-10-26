"""봉인된 bound BF 생산자와 분리된 native-SE 준위용해율 생산자."""
import ctypes as C

import numpy as np

import cmfgen_model_continuum as B
from lumina_model_point_rates import Rates


def contract_dissolution(point, frequency, quadrature_Hz, J, *, elements=(14, 16),
                         dissolution_enabled=False):
    """원 PHOT 확률을 아직 곱하지 않은 (route, 6) 단위율을 반환한다."""
    nu, width, j = [np.asarray(v, dtype='f8') for v in (frequency, quadrature_Hz, J)]
    if (nu.ndim != 1 or len(nu) < 2 or width.shape != nu.shape or j.shape != nu.shape or
        not np.isfinite(nu).all() or np.any(nu <= 0) or np.any(np.diff(nu) <= 0) or
        not np.isfinite(width).all() or np.any(width < 0) or
        not np.isfinite(j).all() or np.any(j < 0) or not elements):
        raise ValueError('준위용해의 원 주파수/복사장 입력 오류')
    if dissolution_enabled is not True:
        raise ValueError('원 물질에서 준위용해가 활성화되지 않음')
    function = point.library.bf_rate_point_contract
    function.argtypes = [C.c_size_t] + [point.ptr] * 6 + [C.c_double, C.POINTER(Rates)]
    function.restype = C.c_int
    b, a, x = B.dissolution_vectors(point.state)
    reference = B.ground_lte_reference(point.rows, point.models, point.state)
    node = point.node
    result = np.zeros((len(point.routes), 6))
    for i, route in enumerate(point.routes):
        if route.Z not in elements or not route.dissolution_primary:
            continue
        if route.target_energy_eV != 0 or not point.upper_population[i] > 0:
            raise ValueError('준위용해는 상위 바닥준위로만 연결')
        edge = route.threshold_Hz
        first = int(np.searchsorted(nu, max(edge - 3.289395e15 / 4, .8 * edge), side='right'))
        last = int(np.searchsorted(nu, edge, side='left'))
        if first >= last or point.sigma_edge[i] == 0:
            continue
        selected = nu[first:last]
        z = point.models[route.Z, route.stage]['z_screen']
        effective = np.sqrt(3.289395e15 * z * z / (edge - selected))
        k = np.minimum(1., 16 * effective / (1 + effective)**2 / 3)
        k = (k * z**3 / effective**4)**1.5
        first_term = 7.782 + x[node] * b[node] * k
        ydis = 1.091 * (x[node] + 4 * (z - 1) * a[node]) * b[node] * b[node]
        fraction = first_term / (first_term + ydis * k * k)
        sigma = np.where(fraction > 1.e-4, point.sigma_edge[i] * fraction, 0.)
        if not np.any(sigma):
            continue
        saha = reference[route.lower, node] / point.upper_population[i]
        with np.errstate(over='raise', invalid='raise'):
            stimulated = sigma * saha * np.exp(B.H * (edge - selected) / (B.KB * point.temperature))
            eta = stimulated * (2 * B.H * selected**3 / B.C**2)
        arrays = [np.ascontiguousarray(v, dtype='f8') for v in
                  (selected, width[first:last], j[first:last], sigma, stimulated, eta)]
        item = Rates()
        if function(len(selected), *[v.ctypes.data_as(point.ptr) for v in arrays], edge, C.byref(item)):
            raise ValueError(f'준위용해 C 점값 적분 실패: route {i}')
        result[i] = [getattr(item, key) for key, _ in item._fields_]
    if not np.isfinite(result).all() or np.any(result[:, :3] < 0):
        raise ValueError('준위용해율 수치 오류')
    return result
