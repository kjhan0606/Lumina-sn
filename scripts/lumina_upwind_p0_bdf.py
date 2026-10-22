"""소형 강성 P0 청색 상풍 연립식의 sparse BDF 기준해.

χ는 주파수마다 다르고 signed 값을 보존한다. 고정된 공간 구간의
상수 χ/η/b만 푼다. 원 CMFGEN의 공간 short-characteristic이나
완전한 광선·자기일관 수송 결과가 아니다.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags


def bdf_p0_segment(dlognu, length, drift, chi, eta, incident,
                   rtol=1e-8, atol=1e-15):
    chi = np.asarray(chi, dtype='f8')
    eta = np.asarray(eta, dtype='f8')
    incident = np.asarray(incident, dtype='f8')
    n = len(incident)
    if (n < 2 or n > 8192 or chi.shape != (n,) or eta.shape != (n,) or
            not np.isfinite([dlognu, length, drift, rtol, atol]).all() or
            dlognu <= 0 or length < 0 or drift < 0 or
            (length == 0 and drift != 0) or
            not 0 < rtol <= 1e-3 or not 0 < atol <= 1e-8 or
            not all(np.isfinite(a).all() for a in (chi, eta, incident)) or
            np.any(eta < 0) or np.any(incident < 0)):
        raise ValueError('소형 유한 P0 상풍 기준 입력 필요')
    if length == 0:
        return incident.copy(), dict(nfev=0, nlu=0)

    lam = drift / dlognu
    diagonal = -(chi * length + 3 * drift + lam)
    diagonal[-1] = -(chi[-1] * length + 3 * drift)
    matrix = diags((diagonal, np.full(n - 1, lam)), (0, 1),
                   shape=(n, n), format='csc')
    source = eta * length
    if not np.isfinite(matrix.data).all() or not np.isfinite(source).all():
        raise ValueError('이산 연산자 비유한')
    result = solve_ivp(lambda _, y: matrix @ y + source, (0., 1.), incident,
                       method='BDF', jac=matrix, rtol=rtol, atol=atol)
    if not result.success or not np.isfinite(result.y).all() or \
            np.any(result.y[:, -1] < 0):
        raise ValueError('강성 상풍 기준해 실패 또는 음수')
    return result.y[:, -1], dict(nfev=result.nfev, nlu=result.nlu)
