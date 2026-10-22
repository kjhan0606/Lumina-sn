"""소형 고정계수 CMF 이산 상풍 연립식의 sparse 행렬지수 기준해.

큰 전주파수 수송기나 CMFGEN 3점 공간 SC가 아니다. 비교기 직접시험용.
"""
import numpy as np
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import expm_multiply


def exact_p0_segment(dlognu,length,drift,chi,eta,incident):
    chi=np.asarray(chi,dtype='f8')
    eta=np.asarray(eta,dtype='f8')
    incident=np.asarray(incident,dtype='f8')
    nf=len(incident)
    if (nf<2 or nf>8192 or chi.shape!=(nf,) or eta.shape!=(nf,) or
            not np.isfinite([dlognu,length,drift]).all() or dlognu<=0 or
            length<0 or drift<0 or (length==0 and drift!=0) or
            not all(np.isfinite(a).all() for a in (chi,eta,incident)) or
            np.any(eta<0) or np.any(incident<0)):
        raise ValueError('소형 유한 P0 이산 상풍 입력 필요')
    if length==0:
        return incident.copy()
    b=drift/length
    coupling=b/dlognu
    diagonal=-(chi+3*b+coupling)
    diagonal[-1]=-(chi[-1]+3*b)
    matrix=diags((diagonal,np.full(nf-1,coupling)),(0,1),
                 shape=(nf,nf),format='csr')
    augmented=bmat([[matrix,csr_matrix(eta[:,None])],
                    [None,csr_matrix((1,1))]],format='csr')
    result=expm_multiply(augmented*length,np.r_[incident,1.])[:nf]
    if not np.isfinite(result).all() or np.any(result<0):
        raise ValueError('이산 상풍 기준해 비유한 또는 음수')
    return result
