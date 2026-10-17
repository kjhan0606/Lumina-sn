#ifndef LUMINA_CMF_FREE_FREE_H
#define LUMINA_CMF_FREE_FREE_H
#include "lumina.h"
#include <math.h>

/* BF 표 바깥을 끝값으로 연장하지 않는다. 모든 BF 문턱보다 낮음을
 * 호출자가 확인한 영역에서만 이 자유–자유 연속체를 단독 사용한다.
 * 동일 모형의 gaunt=1과 실제 sum(q^2 n_ion), 안정적인 저주파 계산. */
static inline int cmf_free_free(double nu, double T, double ne, double z2ni,
                                double *chi, double *eta)
{
    if (!chi || !eta || !isfinite(nu) || nu <= 0 || !isfinite(T) || T <= 0 ||
        !isfinite(ne) || ne < 0 || !isfinite(z2ni) || z2ni < 0) return -1;
    double x = H_PLANCK * nu / (K_BOLTZMANN * T);
    double coeff = C_FF_OPACITY / sqrt(T) * ne * z2ni;
    *chi = coeff / (nu * nu * nu) * (-expm1(-x));
    /* chi*B의 해석적 소거식: 저주파에서도 0/0이나 큰 B를 만들지 않는다. */
    *eta = coeff * (2 * H_PLANCK / (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT)) * exp(-x);
    return isfinite(*chi) && *chi >= 0 && isfinite(*eta) && *eta >= 0 ? 0 : -1;
}
#endif
