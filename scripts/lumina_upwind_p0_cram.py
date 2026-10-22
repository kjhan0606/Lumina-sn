"""기각된 CRAM16 음성 대조군; 물리 실행에 연결하지 않는다.

IPF CRAM16 수치 계수: OpenMC 공식 구현 및 Pusa(2012) 정정.
https://docs.openmc.org/en/stable/_modules/openmc/deplete/cram.html
https://arxiv.org/abs/1206.2880

음의 실수축 스칼라 근사는 실제 주파수 이동(~166.5빈)의 비정규
상풍 행렬에서 정확도를 보장하지 않는다. 직접 반례는 대응 시험에 둔다.
"""
import numpy as np


ALPHA=np.array([
    5.464930576870210e3-3.797983575308356e4j,
    9.045112476907548e1-1.115537522430261e3j,
    2.344818070467641e2-4.228020157070496e2j,
    9.453304067358312e1-2.951294291446048e2j,
    7.283792954673409e2-1.205646080220011e5j,
    3.648229059594851e1-1.155509621409682e2j,
    2.547321630156819e1-2.639500283021502e1j,
    2.394538338734709e1-5.650522971778156j],dtype='c16')
THETA=np.array([
    3.509103608414918+8.436198985884374j,
    5.948152268951177+3.587457362018322j,
    -5.264971343442647+16.22022147316793j,
    1.419375897185666+10.92536348449672j,
    6.416177699099435+1.194122393370139j,
    4.993174737717997+5.996881713603942j,
    -1.413928462488886+13.49772569889275j,
    -10.84391707869699+19.27744616718165j],dtype='c16')
ALPHA0=2.124853710495224e-16


def cram16_p0_segment(dlognu,length,drift,chi,eta,incident):
    """Bidiagonal+상수원 행렬에 직접 역대입; 수치오차는 별도 기준해로 평가."""
    chi=np.asarray(chi,dtype='f8');eta=np.asarray(eta,dtype='f8')
    incident=np.asarray(incident,dtype='f8');n=len(incident)
    if (n<2 or n>8192 or chi.shape!=(n,) or eta.shape!=(n,) or
            not np.isfinite([dlognu,length,drift]).all() or dlognu<=0 or
            length<0 or drift<0 or (length==0 and drift!=0) or
            not all(np.isfinite(a).all() for a in (chi,eta,incident)) or
            np.any(eta<0) or np.any(incident<0)):
        raise ValueError('소형 유한 P0 상풍 입력 필요')
    if length==0:
        return incident.copy()
    lam=drift/dlognu
    diag=-(chi*length+3*drift+lam)
    diag[-1]=-(chi[-1]*length+3*drift)
    source=eta*length
    y=np.r_[incident,1.]
    for alpha,theta in zip(ALPHA,THETA):
        solved=np.empty(n+1,dtype='c16')
        solved[-1]=-y[-1]/theta
        solved[n-1]=(y[n-1]-source[n-1]*solved[-1])/(diag[n-1]-theta)
        for k in range(n-2,-1,-1):
            solved[k]=(y[k]-lam*solved[k+1]-source[k]*solved[-1])/(diag[k]-theta)
        y+=2*np.real(alpha*solved)
    y*=ALPHA0
    if not np.isfinite(y).all() or np.any(y[:-1]<0):
        raise ValueError('CRAM16 후보 비유한 또는 음수')
    return y[:-1]
