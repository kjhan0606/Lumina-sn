#ifndef CMF_NONTHERMAL_EXCITATION_H
#define CMF_NONTHERMAL_EXCITATION_H
#include "cmf_nonthermal_provider.h"
#ifdef __cplusplus
extern "C" {
#endif
/* 실제 원자표의 여기 채널만 소유한다. 전리 채널을 대신하는 완전한 SN
 * 공급자가 아니다. bound view는 이 소유자가 살아있는 동안만 유효하다. */
typedef struct {
    CMFNonthermalBoundModel view;
    CMFNonthermalTransition *transitions;
    CMFNonthermalRoute *channels;
    double *energy,*sigma;
    int lower_limit,n_bethe,n_omega,n_subthermal;
    double minimum_threshold_eV;
} CMFNonthermalExcitations;
int cmf_nonthermal_excitation_build(const AtomicData*,const double *energy,int n_energy,
                                   int lower_limit,CMFNonthermalExcitations*);
void cmf_nonthermal_excitation_free(CMFNonthermalExcitations*);
#ifdef __cplusplus
}
#endif
#endif
