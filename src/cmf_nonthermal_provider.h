#ifndef CMF_NONTHERMAL_PROVIDER_H
#define CMF_NONTHERMAL_PROVIDER_H
#include "cmf_nonthermal.h"
#include "cmf_nonthermal_rates.h"
#ifdef __cplusplus
extern "C" {
#endif
/* 이미 실제 full-level/terminal에 연결한 불변 단면적 묶음. branch probability는
 * 각 단면적에 포함되어야 한다. 낮은 준위 인구는 저장하지 않고 매 후보에서 읽는다.
 * 모든 실제 여기/전리 경로를 선택하는 원자 loader는 이 인터페이스 밖에 있다. */
typedef struct {
    int n_levels,n_ions,n_energy,n_channels;
    const double *energy_eV;
    const CMFNonthermalTransition *transitions;
    const CMFNonthermalRoute *channels; /* population은 무시; 실제 lower 인구로 대체 */
} CMFNonthermalBoundModel;
int cmf_nonthermal_bound_evaluate(const void*,const AtomicData*,const NLTEConfig*,
                                const PlasmaState*,const GammaDeposition*,CMFNonthermalRates*);
int cmf_nonthermal_transition_energy(const AtomicData*,const CMFNonthermalTransition*,double*,int*);
#ifdef __cplusplus
}
#endif
#endif
