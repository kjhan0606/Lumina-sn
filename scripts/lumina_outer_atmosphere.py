"""CMFGEN 외부대기 폐쇄관계의 자체 후보. 생산 수송에 아직 연결하지 않았다.

외부 총방출률은 native 기준 노드의 자체 J에 의존한다. 이를 국소 산란 계수로
다시 분해하거나 양의 affine K의 오차보증을 적용하지 않는다.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

C = 2.99792458e10
REFERENCE_SOURCE_SHA256 = 'eede8d1193f481d5f48002901b16d91e39ff0d073868121e56e78e3fc61faebd'
NATIVE_LENGTH_CM = 1e10
LEGACY_ETA_MINIMUM = 1e-280/NATIVE_LENGTH_CM  # native 길이 단위의 기존 처방을 cgs로 환산.


@dataclass(frozen=True)
class OuterGeometry:
    radius: np.ndarray  # 원래와 연장 노드 모두 큰 반경부터, cm
    velocity: np.ndarray  # cm/s
    sigma: np.ndarray
    added: int
    reference_indices: tuple
    native_radius: np.ndarray


def geometry(radius, velocity, sigma, rext_factor=0.):
    """원본 CMF_FORMAL_REL_V4의24노드/속도/외삽 기준점 선택을 같은 순서로 평가."""
    r, v, s = [np.array(a, dtype='f8', copy=True) for a in (radius, velocity, sigma)]
    if (r.ndim != 1 or len(r) < 24 or v.shape != r.shape or s.shape != r.shape or
            not all(np.isfinite(a).all() for a in (r, v, s)) or np.any(r <= 0) or
            not np.all(np.diff(r) < 0) or np.any(v <= 0) or np.any(v >= C) or
            not np.isfinite(rext_factor)):
        raise ValueError('외부대기 원래 노드/속도 입력 오류')
    n, added = len(r), 24
    if 1 < rext_factor < 10:
        maximum = rext_factor*r[0]
    elif v[-1]/1e5 < 10 and r[0]/r[-1] >= 9.99:
        maximum = 10*r[0]
    elif v[-1]/1e5 > 10 or v[0]/1e5 > 2e4:
        maximum = 1.5*r[0]
    else:
        maximum = min(10., np.sqrt(r[0]/r[-1]))*r[0]
    alpha = r[0]+(r[0]-r[1])
    if not maximum > alpha:
        raise ValueError('원본 외부격자 구성 범위 불일치')
    factor = np.exp(np.log(maximum/alpha)/(added-4))
    ext = np.empty(n+added); ext[added:] = r
    ext[0], ext[4] = maximum, maximum/factor
    ext[1] = ext[0]-.001*(ext[0]-ext[4])
    ext[2] = ext[0]-.1*(ext[0]-ext[4])
    ext[3] = ext[0]-.4*(ext[0]-ext[4])
    # 원소스는 위의5번째 값을 간격 산출에 쓴 후 이 루프에서 덮어쓴다.
    for i in range(4, added-1):
        ext[i] = ext[i-1]/factor
    ext[added-1] = alpha
    exponent = (s[0]+1)*(r[0]/r[-1]-1)
    terminal = v[0]/(1-r[-1]/r[0])**exponent
    ve = np.r_[terminal*(1-r[-1]/ext[:added])**exponent, v]
    se = np.r_[exponent/(ext[:added]/ext[-1]-1)-1, s]
    if (not np.isfinite(ve).all() or not np.isfinite(se).all() or
            np.any(ve <= 0) or np.any(ve >= C) or not np.all(np.diff(ext) < 0)):
        raise ValueError('연장 속도/격자의 유효 표현 범위 초과')
    ratio = ext[0]/r[0]
    if ratio < r[0]/r[n//3-1]:
        candidates = np.flatnonzero(r[0]/r > min(3., ratio))
        if not len(candidates):
            raise ValueError('원본 외삽 기준점 선택이 정의되지 않음')
        last = min(max(4, int(candidates[0])+1), n//6)-1
    else:
        last = n//6-1
    for a in (ext, ve, se, r):
        a.flags.writeable = False
    return OuterGeometry(ext, ve, se, added, (0, last), r)


def coefficients(g, chi, scattering, eta_true, own_node_j):
    """각 trial J에서 native 총η를 먼저 조립해 연장한다. 모든 ν를 유지한다."""
    chi, es, eta, j = [np.asarray(a, dtype='f8') for a in
                       (chi, scattering, eta_true, own_node_j)]
    if (chi.ndim != 2 or chi.shape[0] != len(g.native_radius) or
            any(a.shape != chi.shape for a in (es, eta, j)) or
            not all(np.isfinite(a).all() for a in (chi, es, eta, j)) or
            np.any(es <= 0) or np.any(eta < 0) or np.any(j < 0)):
        raise ValueError('외삽 계수/J 차원 또는 양의 입력 오류')
    total = eta+es*j
    a, b = g.reference_indices
    if not np.isfinite(total).all():
        raise ValueError('외삽 기준점의 총방출률 비유한')
    log_radius = np.log(g.native_radius[a]/g.native_radius[b])
    ratio = (g.native_radius[a]/g.radius[:g.added])[:, None]
    es_power = np.maximum(2., np.log(es[b]/es[a])/log_radius)
    special = (chi[a] <= es[a]) | (chi[b] <= es[b])
    ext_chi = np.empty((g.added, chi.shape[1]))
    ext_chi[:, special] = chi[a, special]*ratio**es_power[special]
    normal = ~special
    absorption_power = np.maximum(2., np.log((chi[b, normal]-es[b, normal])/
                                            (chi[a, normal]-es[a, normal]))/log_radius)
    ext_chi[:, normal] = ((chi[a, normal]-es[a, normal])*ratio**absorption_power +
                          es[a, normal]*ratio**es_power[normal])
    # 양의 domain은 원식, 0은 그 연속 한계다. a=0이면 외부 방출률은0이며
    # 기존 외부 처방만 남는다. b=0,a>0이면 지수 하한3.5가 작동한다.
    eta_power = np.full(chi.shape[1],3.5)
    active = (total[a] > 0) & (total[b] > 0)
    with np.errstate(over='ignore',under='ignore',divide='ignore'):
        ratios = total[b,active]/total[a,active]
        logs = np.log(ratios)
    extreme = ~np.isfinite(logs)
    logs[extreme] = np.log(total[b,active][extreme])-np.log(total[a,active][extreme])
    eta_power[active] = np.maximum(3.5,logs/log_radius)
    ext_eta = total[a]*ratio**eta_power
    ext_eta = np.maximum(LEGACY_ETA_MINIMUM, ext_eta)
    if not np.isfinite(ext_chi).all() or not np.isfinite(ext_eta).all():
        raise ValueError('연장 계수 비유한')
    return dict(chi=np.vstack((ext_chi, chi)), eta_total=np.vstack((ext_eta, total)),
        eta_power=eta_power, es_power=es_power, special_chi=special,
        eta_floor_scope='existing CMFGEN extension closure only; no intensity/population floor',
        scattering_added_locally_in_extension=False)


def velocity_at(g, radius):
    """참조 특성선과 같이 노드 사이의 속도를 선형 보간한다. 끝값 연장 금지."""
    r = np.asarray(radius)
    if np.any(r < g.radius[-1]) or np.any(r > g.radius[0]):
        raise ValueError('속도 표본이 연장 대기 밖에 있음')
    return np.interp(r, g.radius[::-1], g.velocity[::-1])


def doppler(g, impact, z, radius=None):
    if radius is None:
        r = np.hypot(impact, z)
    else:
        # sqrt(r²-p²)에서 만든 절점은 hypot 역변환 시1ULP 달라질 수 있다.
        # 임의 좌표를 clip하지 않고 원래 반경과 정확히 같은 z 구성인지 확인한다.
        r = float(radius)
        if (not np.isfinite(r) or r < impact or
                abs(z) != np.sqrt((r-impact)*(r+impact))):
            raise ValueError('원래 반경 절점과 실제 z 구성 불일치')
    beta = velocity_at(g, r)/C
    return (1-beta*np.asarray(z)/r)/np.sqrt((1-beta)*(1+beta))


def segment_geometry(g, impact, start_z, end_z, order=8, start_radius=None, end_radius=None):
    """일반 속도장의 ds=D dz와 endpoint drift. 계수 변화의 구간 오차는 별도다."""
    if (not np.isfinite(impact) or impact < 0 or not np.isfinite(start_z) or
            not np.isfinite(end_z) or end_z < start_z or order not in (8, 16)):
        raise ValueError('일반 속도장 구간 입력 오류')
    nodes, weights = gauss_rule(order)
    half = .5*(end_z-start_z)
    z = .5*(start_z+end_z)+half*nodes
    length = float(half*np.dot(weights, doppler(g, impact, z)))
    first = doppler(g, impact, start_z, start_radius)
    last = doppler(g, impact, end_z, end_radius)
    drift = float(np.log(first/last))
    if not np.isfinite(length) or not np.isfinite(drift) or length < 0 or drift < 0:
        raise ValueError('유효하지 않은 길이 또는 적색 upwind가 필요한 구간')
    return length, drift


@lru_cache(maxsize=2)
def gauss_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.flags.writeable = weights.flags.writeable = False
    return nodes, weights
