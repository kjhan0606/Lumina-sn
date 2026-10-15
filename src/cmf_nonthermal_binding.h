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
} CMFNonthermalResolvedModel;
int cmf_nonthermal_binding_build(const AtomicData*,const CMFNonthermalAtomicData*,
    const CMFNonthermalExcitations*,const char *csv,CMFNonthermalResolvedModel*);
void cmf_nonthermal_binding_free(CMFNonthermalResolvedModel*);
#ifdef __cplusplus
}
#endif
#endif
