#ifndef CMF_OBSERVER_H
#define CMF_OBSERVER_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* 같은 공변계 재료와 J를 관측자계로 변환하는 구형 형식해.
 * 재료는 로그 주파수 셀 안에서 상수. 모든 입력은 읽기 전용이다.
 * t_exp=0은 정적 시험, 양수는 homologous expansion [s]. */
typedef struct {
    int ns, nf;
    double nu_min, dlognu, t_exp, T_inner, inner_scale;
    const double *ri, *ro;
    const double *chi, *es, *source, *J; /* [ns*nf], source=eta_fixed/chi */
    int nobs, ncore, nouter, substeps;
    const double *nu_obs;
    int source_is_emissivity; /* 1: source=eta_fixed, 0: 예전 eta/chi 입력 */
} CMFObserverInput;

/* Lnu [erg/s/Hz]. 실패하면 caller의 Lnu는 바꾸지 않는다. */
int cmf_observer_cpu(const CMFObserverInput *in, double *Lnu);
int cmf_observer_gpu(const CMFObserverInput *in, double *Lnu);
int cmf_observer_validate(const CMFObserverInput *in);
int cmf_observer_quadrature(const CMFObserverInput *in, double *p, double *w);
int cmf_observer_reduce(const CMFObserverInput *in, const double *intensity,
                        const double *w, double *Lnu);
#ifdef __cplusplus
}
#endif
#endif
