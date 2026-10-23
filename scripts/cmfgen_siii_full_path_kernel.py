"""고정 원 χ/η의 공간 세분 CMF 광선 커널. 자기일관 물질 갱신은 별도다."""
import math

import numpy as np
from numba import njit

from check_cmfgen_local_sc_packet import BEZIER_COEFFICIENTS


WEIGHT_POLY = np.stack(BEZIER_COEFFICIENTS)


@njit(cache=True)
def e_coeff(tau):
    if tau > .01:
        ee = math.exp(-tau)
        e0 = 1.-ee
        e1 = tau-e0
    else:
        e2 = tau**3*(1.-.25*tau*(1.-.2*tau*(1.-tau/6.*(1.-tau/7.))))/3.
        e1 = .5*(tau*tau-e2)
        e0 = tau-e1
        ee = 1.-e0
    return ee, e0, e1


@njit(cache=True)
def bezier_source(sup, target, far, tau, neighbor):
    if tau <= 0.:
        return 0.
    if tau <= .5:
        w0, w1, w2 = WEIGHT_POLY[12]
        for i in range(11, -1, -1):
            w0 = -tau*w0+WEIGHT_POLY[i, 0]
            w1 = -tau*w1+WEIGHT_POLY[i, 1]
            w2 = -tau*w2+WEIGHT_POLY[i, 2]
        w0 *= tau
        w1 *= tau
        w2 *= tau
    else:
        inverse = 1./tau
        ee = math.exp(-tau)
        w0 = 2*inverse*inverse-ee*(1+2*inverse+2*inverse*inverse)
        w1 = 2*inverse-4*inverse*inverse+ee*(2*inverse+4*inverse*inverse)
        w2 = 1-2*inverse+2*inverse*inverse-2*ee*inverse*inverse
    if neighbor <= 0.:
        control = .5*sup+.5*target
    else:
        if tau >= neighbor:
            ratio = neighbor/tau
            w = 1./(1.+ratio)
        else:
            ratio = tau/neighbor
            w = ratio/(1.+ratio)
        base = .5*(1.-w)*sup+.5*(1.+w)*target
        term = .5*w*(far-target)
        lo = min(sup, target)
        hi = max(sup, target)
        if term == 0.:
            control = min(hi, max(lo, base))
        else:
            distance = max(0., base-lo) if term > 0. else max(0., hi-base)
            if tau > neighbor:
                ratio = neighbor/tau
                if abs(term) >= distance*ratio:
                    distance = np.finfo(np.float64).max
                else:
                    distance = abs(term)/ratio
            else:
                distance = abs(term)*(tau/neighbor)
            if term > 0.:
                control = lo if distance >= base-lo else base-distance
            else:
                control = hi if distance >= hi-base else base+distance
    return w0*sup+w1*control+w2*target


def refine_nodes(coarse, subdivisions):
    coarse = np.asarray(coarse, dtype='f8')
    if (type(subdivisions) is not int or not 1 <= subdivisions <= 128 or
            coarse.ndim != 1 or len(coarse) < 4 or not np.isfinite(coarse).all()):
        raise ValueError('유한 native 광선과 1–128분할 필요')
    fine_index = np.arange((len(coarse)-1)*subdivisions+1)/subdivisions
    return np.interp(fine_index, np.arange(len(coarse)), coarse)


@njit(cache=True)
def sweep(chi, eta, s_m, s_p, b_m, b_p, prev_m, prev_p,
          dlognu, inner_plus, core_ray, subdivisions):
    """한 광선·한 formal 호출의 inward 뒤 outward SC를 세분 격자에서 푼다."""
    n = len(chi)
    m = (n-1)*subdivisions+1
    chi_f = np.empty(m)
    eta_f = np.empty(m)
    for k in range(m):
        j = min(k//subdivisions, n-2)
        fraction = (k-j*subdivisions)/subdivisions
        chi_f[k] = (1.-fraction)*chi[j]+fraction*chi[j+1]
        eta_f[k] = (1.-fraction)*eta[j]+fraction*eta[j+1]
    alpha = 1./dlognu
    em = chi_f+(alpha+3.)*b_m
    ep = chi_f+(alpha+3.)*b_p
    source_m = (eta_f+alpha*b_m*prev_m)/em
    source_p = (eta_f+alpha*b_p*prev_p)/ep
    tau_m = .5*(s_m[1:]-s_m[:-1])*(em[1:]+em[:-1])
    tau_p = .5*(s_p[:-1]-s_p[1:])*(ep[:-1]+ep[1:])
    minus = np.zeros(m)
    plus = np.zeros(m)
    for k in range(1, m-1):
        tau = tau_m[k-1]
        ee = e_coeff(tau)[0]
        minus[k] = (minus[k-1]*ee +
                    bezier_source(source_m[k-1], source_m[k],
                                  source_m[k+1], tau, tau_m[k]))
    tau = tau_m[m-2]
    ee, e0, e1 = e_coeff(tau)
    minus[m-1] = (minus[m-2]*ee + (e0-e1/tau)*source_m[m-2] +
                  (e1/tau)*source_m[m-1])
    plus[m-1] = inner_plus if core_ray else minus[m-1]
    for k in range(m-2, 2*subdivisions-1, -1):
        tau = tau_p[k]
        ee = e_coeff(tau)[0]
        plus[k] = (plus[k+1]*ee +
                   bezier_source(source_p[k+1], source_p[k],
                                 source_p[k-1], tau, tau_p[k-1]))
    for k in range(2*subdivisions-1, -1, -1):
        tau = tau_p[k]
        ee, e0, e1 = e_coeff(tau)
        plus[k] = (plus[k+1]*ee + (e1/tau)*source_p[k] +
                   (e0-e1/tau)*source_p[k+1])
    return plus, minus
