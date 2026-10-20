"""ES12.4와 binary64 연산 폭 안에서 실제 전체 갱신 적용만 확인한다.

수렴·남은 SE 잔차를 판정하지 않는다. 이전 A/B의 동일 정밀도 규약을 유지한다.
"""
import numpy as np


def applied_full_step(pre,after,correction):
    if not (pre.shape==after.shape==correction.shape):raise ValueError('갱신 상태 차원 불일치')
    if not all(np.isfinite(x).all() for x in (pre,after,correction)):return False
    x=abs(correction);radius=np.zeros_like(x);positive=x>0
    radius[positive]=.5*np.power(10.,np.floor(np.log10(x[positive]))-4)
    bound=abs(pre)*radius+8*np.finfo(float).eps*(abs(pre)+abs(pre*correction))
    return bool(np.all(abs(after-pre*(1-correction))<=bound))
