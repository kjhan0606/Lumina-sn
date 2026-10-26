"""native 공통격자의 BF 소비 연산자 비교용 단위율.

실제ν/J/전역 가중치는 유지하고 σ만 명시된 CONT_FREQ에서 평가한다.
생산 수송의 기본 단면 규약을 바꾸거나 cache 지수를 복제하지 않는다.
"""
import ctypes as C

import numpy as np

import cmfgen_model_continuum as B
from lumina_model_point_rates import Rates


def dissolution_sigma(point, index, evaluation, enabled):
    evaluation = np.asarray(evaluation,dtype='f8')
    result = np.zeros_like(evaluation)
    route = point.routes[index]
    if not enabled or not route.dissolution_primary:
        return result
    if route.target_energy_eV != 0:
        raise ValueError('native-SE 용해는 바닥 목표로만 연결')
    edge = route.threshold_Hz
    active = ((evaluation < edge)&(evaluation > .8*edge)&
              (evaluation > edge-3.289395e15/4))
    if not np.any(active) or point.sigma_edge[index] == 0:
        return result
    z = point.models[route.Z,route.stage]['z_screen']
    b,a,x = B.dissolution_vectors(point.state)
    node = point.node
    effective = np.sqrt(3.289395e15*z*z/(edge-evaluation[active]))
    k = np.minimum(1.,16*effective/(1+effective)**2/3)
    k = (k*z**3/effective**4)**1.5
    first = 7.782+x[node]*b[node]*k
    y = 1.091*(x[node]+4*(z-1)*a[node])*b[node]*b[node]
    fraction = first/(first+y*k*k)
    result[active] = np.where(fraction > 1e-4,point.sigma_edge[index]*fraction,0.)
    return result


class ReferencePointRates:
    def __init__(self, point):
        self.point = point
        self.function = point.library.bf_rate_point_contract
        self.function.argtypes = [C.c_size_t]+[point.ptr]*6+[C.c_double,C.POINTER(Rates)]
        self.function.restype = C.c_int

    def contract(self,index,frequency,quadrature_Hz,J,evaluation,*,dissolution_enabled):
        nu,width,j,evaluation = [np.ascontiguousarray(v,dtype='f8') for v in
                                 (frequency,quadrature_Hz,J,evaluation)]
        if (nu.ndim != 1 or not len(nu) or any(v.shape != nu.shape for v in (width,j,evaluation)) or
                not all(np.isfinite(v).all() for v in (nu,width,j,evaluation)) or
                np.any(nu <= 0) or np.any(evaluation <= 0) or np.any(width < 0) or
                np.any(j < 0) or np.any(np.diff(nu) <= 0)):
            raise ValueError('공통격자 실제ν/계수ν/J/가중치 오류')
        route = self.point.routes[index]
        bound = route.sigma(evaluation)
        pseudo = dissolution_sigma(self.point,index,evaluation,dissolution_enabled)
        outputs = []
        for sigma in (bound,bound+pseudo):
            induced = np.zeros_like(sigma)
            active = sigma > 0
            # σ의 평가ν를 재결합 열적 지수의 ν로 바꾸지 않는다.
            with np.errstate(over='raise',invalid='raise'):
                induced[active] = (sigma[active]*self.point.milne_prefactor[index]*
                    np.exp(-(nu[active]-route.threshold_Hz)*self.point.inverse_kT_Hz))
                eta = induced*(2*B.H*nu**3/B.C**2)
            arrays = [np.ascontiguousarray(v,dtype='f8') for v in (nu,width,j,sigma,induced,eta)]
            result = Rates()
            if self.function(len(nu),*[v.ctypes.data_as(self.point.ptr) for v in arrays],
                             route.threshold_Hz,C.byref(result)):
                raise ValueError('공통격자 C 점값 율 적분 실패')
            outputs.append(np.array([getattr(result,k) for k,_ in result._fields_]))
        return outputs


def reference_line_rates(producer,frequency,quadrature_Hz,J):
    """물리 A와 native의 미정규화 이산 자발항을 비교용으로 함께 보존한다.

    SET_LINE_OPAC의 LINE_QW와 ZNET_SIM의 적분은 자발항에도 profile mass를
    포함한다. Jbar를 mass로 나눠 새 정규화 규약으로 바꾸지 않는다.
    """
    values = producer.contract(frequency,quadrature_Hz,J)
    if not np.isin(values['status'],[1,2]).all():
        raise ValueError('같은 원 J에서 미완성 BB 프로파일 표본')
    values['spontaneous_native_discrete'] = values['spontaneous']*values['profile_mass']
    return values
