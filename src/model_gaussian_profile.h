#ifndef LUMINA_MODEL_GAUSSIAN_PROFILE_H
#define LUMINA_MODEL_GAUSSIAN_PROFILE_H

#include <math.h>

/* 실제 source와 점값 Jbar가 같은 native Gaussian을 사용한다.
 * 유한 지지나 이산 적분값으로 재정규화하지 않는다. */
static inline double model_gaussian_phi(double nu,double nu0,double width_fraction)
{
    double width=nu0*width_fraction,x=(nu-nu0)/width;
    return exp(-x*x)/(1.77245385095516*width);
}
#endif
