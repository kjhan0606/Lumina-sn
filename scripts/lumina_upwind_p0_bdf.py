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


def bdf_p0_segment_nonuniform(log_frequency, length, drift, chi, eta, incident,
                              rtol=1e-8, atol=1e-15):
    """증가하는 실제 logν 격자의 한 고정계수 P0 구간 기준해.

    청색 이웃 결합은 각 적색 빈 k의 실제 간격 x[k+1]-x[k]를 쓴다.
    이 함수는 CMFGEN의 Bézier 공간 SC나 전체 경로를 대체하지 않는다.
    """
    x = np.asarray(log_frequency, dtype='f8')
    chi = np.asarray(chi, dtype='f8')
    eta = np.asarray(eta, dtype='f8')
    incident = np.asarray(incident, dtype='f8')
    n = len(incident)
    if (n < 2 or n > 8192 or any(a.shape != (n,) for a in (x, chi, eta)) or
            not all(np.isfinite(a).all() for a in (x, chi, eta, incident)) or
            np.any(np.diff(x) <= 0) or np.any(eta < 0) or
            np.any(incident < 0) or
            not np.isfinite([length, drift, rtol, atol]).all() or
            length < 0 or drift < 0 or (length == 0 and drift != 0) or
            not 0 < rtol <= 1e-3 or not 0 < atol <= 1e-8):
        raise ValueError('소형 유한 비등간격 P0 상풍 기준 입력 필요')
    if length == 0:
        return incident.copy(), dict(nfev=0, nlu=0)
    coupling = drift / np.diff(x)
    diagonal = -(chi * length + 3 * drift)
    diagonal[:-1] -= coupling
    matrix = diags((diagonal, coupling), (0, 1), shape=(n, n), format='csc')
    source = eta * length
    if not np.isfinite(matrix.data).all() or not np.isfinite(source).all():
        raise ValueError('비등간격 이산 연산자 비유한')
    result = solve_ivp(lambda _, y: matrix @ y + source, (0., 1.), incident,
                       method='BDF', jac=matrix, rtol=rtol, atol=atol)
    if not result.success or not np.isfinite(result.y).all() or \
            np.any(result.y[:, -1] < 0):
        raise ValueError('비등간격 상풍 기준해 실패 또는 음수')
    return result.y[:, -1], dict(nfev=result.nfev, nlu=result.nlu)


def bdf_p0_segment_nonuniform_spatial_linear(log_frequency, length, drift,
                                              chi_inner, chi_outer,
                                              eta_inner, eta_outer, incident,
                                              rtol=1e-8, atol=1e-15):
    """실제 비등간격 격자에서 양 끝 χ/η를 광선 거리로 선형 보간한 P0.

    주파수 결합 b는 구간 평균 drift/length로 고정한다. 원 Bézier SC나
    실제 가변 b의 생산 계산을 뜻하지 않는다.
    """
    x = np.asarray(log_frequency, dtype='f8')
    incident = np.asarray(incident, dtype='f8')
    chi_inner, chi_outer, eta_inner, eta_outer = (
        np.asarray(value, dtype='f8') for value in
        (chi_inner, chi_outer, eta_inner, eta_outer))
    n = len(incident)
    arrays = (x, chi_inner, chi_outer, eta_inner, eta_outer, incident)
    if (n < 2 or n > 8192 or any(value.shape != (n,) for value in arrays) or
            not all(np.isfinite(value).all() for value in arrays) or
            np.any(np.diff(x) <= 0) or
            np.any(eta_inner < 0) or np.any(eta_outer < 0) or
            np.any(incident < 0) or
            not np.isfinite([length, drift, rtol, atol]).all() or
            length < 0 or drift < 0 or (length == 0 and drift != 0) or
            not 0 < rtol <= 1e-3 or not 0 < atol <= 1e-8):
        raise ValueError('공간 선형 비등간격 P0 입력 오류')
    if length == 0:
        return incident.copy(), dict(nfev=0, nlu=0)
    coupling = drift / np.diff(x)
    static_diagonal = np.zeros(n)
    static_diagonal[:-1] = coupling
    static_diagonal += 3 * drift

    def diagonal(t):
        chi = chi_inner + t * (chi_outer - chi_inner)
        return -(length * chi + static_diagonal)

    def derivative(t, value):
        result = diagonal(t) * value
        result[:-1] += coupling * value[1:]
        result += length * (eta_inner + t * (eta_outer - eta_inner))
        return result

    def jacobian(t, _):
        return diags((diagonal(t), coupling), (0, 1), shape=(n, n),
                     format='csc')

    result = solve_ivp(derivative, (0., 1.), incident, method='BDF',
                       jac=jacobian, rtol=rtol, atol=atol)
    if (not result.success or not np.isfinite(result.y).all() or
            np.any(result.y[:, -1] < 0)):
        raise ValueError('공간 선형 P0 강성해 실패 또는 음수')
    return result.y[:, -1], dict(nfev=result.nfev, nlu=result.nlu)
