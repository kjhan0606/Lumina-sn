#ifndef CMF_NONTHERMAL_BINDING_H
#define CMF_NONTHERMAL_BINDING_H
#include "cmf_nonthermal_excitation.h"
#ifdef __cplusplus
extern "C" {
#endif
/* 이름/에너지가 확인된 부분 원자 모형. excitation의 격자/단면적을 차용한다.
 * unresolved/missing이0이 아니면 완전한 비열 원자 모델이라고 주장할 수 없다. */
typedef struct {
    CMFNonthermalBoundModel view;
    CMFNonthermalTransition *transitions;
    CMFNonthermalRoute *channels;
    double *ion_sigma;
    int n_ion_channels,n_resolved_routes,n_unresolved_routes,n_missing_ions;
    double max_threshold_shift_eV;
    int cross_section_policy;
} CMFNonthermalResolvedModel;
enum { CMF_NT_SIGMA_SAME_EXCESS=0, CMF_NT_SIGMA_SOURCE_KINETIC=1 };
/* SOURCE_KINETIC: 원형 fit은 원래 입사 E에서 평가, 실제 target 에너지가
 * 부족한 충돌만 금지한다. event 손실은 실제 준위 차이이며 fit 계수는 바꾸지 않는다. */
int cmf_nonthermal_binding_build_policy(const AtomicData*,const CMFNonthermalAtomicData*,
    const CMFNonthermalExcitations*,const char *csv,int policy,CMFNonthermalResolvedModel*);
int cmf_nonthermal_binding_build(const AtomicData*,const CMFNonthermalAtomicData*,
    const CMFNonthermalExcitations*,const char *csv,CMFNonthermalResolvedModel*);
void cmf_nonthermal_binding_free(CMFNonthermalResolvedModel*);
#ifdef __cplusplus
}
#endif
#endif
