#ifndef CMF_ANDERSON_H
#define CMF_ANDERSON_H
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* 고정된 formal 연산자의 다음 추정만 바꾼다. 수렴/오차 판정은 호출자의
 * 실제 G(x)-x로 해야 하며, 최종 출력은 외삽값이 아니라 G(x)다.
 * Walker--Ni (2011), AA(1)의 가중 최소제곱. 물리값 floor/clamp 없음. */
typedef struct {
    double *previous_g, *previous_r;
    double previous_norm;
    int have_previous, proposed;
    size_t proposals, rejected;
} CMFAnderson1;

static inline int cmf_anderson1_enabled(void)
{
    const char *value=getenv("LUMINA_CMF_ANDERSON1");
    return value && strcmp(value,"1")==0;
}

/* x는 입출력, g 및 두 작업 배열과는 겹치지 않아야 한다.
 * 비가속 다음값의 보존으로 실패한 추정을 취소할 수 있다. */
static inline void cmf_anderson1_next(CMFAnderson1 *a,size_t n,
    double *x,const double *g,double actual_relative_residual)
{
    if(a->proposed && (!isfinite(actual_relative_residual) ||
                      actual_relative_residual>a->previous_norm)) {
        memcpy(x,a->previous_g,n*sizeof(double));
        a->have_previous=0;a->proposed=0;a->rejected++;
        return;
    }
    long double numerator=0,denominator=0;
    if(a->have_previous)for(size_t i=0;i<n;i++) {
        double scale=fmax(fmax(fabs(x[i]),fabs(g[i])),
                          fmax(fabs(a->previous_g[i]),fabs(a->previous_r[i])));
        if(scale==0)continue;
        long double r=((long double)g[i]-x[i])/scale;
        long double dr=r-(long double)a->previous_r[i]/scale;
        numerator+=dr*r;denominator+=dr*dr;
    }
    double gamma=denominator>0?(double)(numerator/denominator):NAN;
    int use=a->have_previous && isfinite(gamma);
    if(use)for(size_t i=0;i<n;i++) {
        double candidate=g[i]-gamma*(g[i]-a->previous_g[i]);
        if(!(candidate>=0) || !isfinite(candidate)){use=0;a->rejected++;break;}
    }
    for(size_t i=0;i<n;i++) {
        double candidate=use?g[i]-gamma*(g[i]-a->previous_g[i]):g[i];
        a->previous_r[i]=g[i]-x[i];
        a->previous_g[i]=g[i];x[i]=candidate;
    }
    a->previous_norm=actual_relative_residual;
    a->have_previous=1;a->proposed=use;
    if(use)a->proposals++;
}
#endif
