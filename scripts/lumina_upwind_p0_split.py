"""고정계수 이산 P0 상풍 연립식의 양성 보존 분할 후보 (비생산).

주파수마다 다른 χ와 η를 허용한다. 흡수/방출과 순수 상풍의
부분 흐름을 각각 정확히 적용하되, 두 흐름의 교환자는 근사한다.
CMFGEN 공간 short-characteristic도 생산 수송기도 아니다.
"""
import numpy as np
from scipy.stats import poisson


def split_p0_segment(dlognu, length, drift, chi, eta, incident, steps):
    chi = np.asarray(chi, dtype='f8')
    eta = np.asarray(eta, dtype='f8')
    incident = np.asarray(incident, dtype='f8')
    n = len(incident)
    if (n < 2 or chi.shape != (n,) or eta.shape != (n,) or
            type(steps) is not int or not 1 <= steps <= 16384 or
            not np.isfinite([dlognu, length, drift]).all() or
            dlognu <= 0 or length < 0 or drift < 0 or
            (length == 0 and drift != 0) or
            not all(np.isfinite(a).all() for a in (chi, eta, incident)) or
            np.any(eta < 0) or np.any(incident < 0)):
        raise ValueError('유한 고정계수 P0 상풍 입력 필요')
    if length == 0:
        return incident.copy()

    lam = drift / dlognu
    mean = lam / steps
    if mean > 32:
        raise ValueError('한 분할의 주파수 이동이 32빈을 넘음')
    # 양성·자기수렴만으로는 τ≫steps인 청색 빈의 짧은 누출을 보지 못한다.
    # 한 흡수 half-step이 그 시간척도를 건너뛰면 잘못된 0에 수렴하므로
    # 반드시 거부한다. 이 조건은 정확도 충분조건이 아니다.
    if np.max(np.abs(chi * length + 3 * drift)) > 2 * steps:
        raise ValueError('강한 χ의 청색 과도구간을 분할이 해상하지 못함')
    # D의 유한 청색 경계는 가장 높은 빈을 고정하고, 더 낮은 빈에는
    # Poisson survival tail만큼 공급한다. 그 밖의 잘린 질량 <1e-14.
    cutoff = min(n - 1, int(poisson.ppf(1 - 1e-14, mean)) + 1)
    probability = poisson.pmf(np.arange(cutoff + 1), mean)
    blue_tail = poisson.sf(np.arange(cutoff), mean)

    half = 1 / (2 * steps)
    rate = (chi * length + 3 * drift) * half
    with np.errstate(over='ignore', invalid='ignore'):
        attenuation = np.exp(-rate)
        phi = np.ones(n)
        active = rate != 0
        phi[active] = np.expm1(-rate[active]) / (-rate[active])
        emission = eta * length * half * phi
    if (not np.isfinite(attenuation).all() or
            not np.isfinite(emission).all() or
            np.any(attenuation < 0) or np.any(emission < 0)):
        raise ValueError('흡수/방출 부분흐름 수치 범위 초과')

    value = incident.copy()
    for _ in range(steps):
        value = attenuation * value + emission
        moved = np.zeros(n)
        moved[-1] = value[-1]
        for offset, weight in enumerate(probability):
            if offset >= n - 1:
                break
            moved[:n - 1 - offset] += weight * value[offset:n - 1]
        for distance, weight in enumerate(blue_tail, 1):
            moved[n - 1 - distance] += weight * value[-1]
        value = attenuation * moved + emission
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError('분할 상풍 강도 비유한 또는 음수')
    return value
