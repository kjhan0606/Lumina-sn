#ifndef CMF_GAMMA_SOURCE_H
#define CMF_GAMMA_SOURCE_H
#include "cmf_gamma_transport.h"
#include "cmf_ejecta.h"
#ifdef __cplusplus
extern "C" {
#endif
enum{CMF_GAMMA_MAX_LINES=64};
typedef struct {
    int n_lines,parent[CMF_GAMMA_MAX_LINES];
    double energy_MeV[CMF_GAMMA_MAX_LINES],probability[CMF_GAMMA_MAX_LINES];
    CMFDecaySourceData decay;
    double declared_total_MeV[2];
    double completion_scale[2];
} CMFGammaNuclearData;
/* 56Ni/56Co reaction 행 및 gamma appendix를 실제로 읽는다. 다른 사슬은
 * 포함하지 않는다. 헤더 Q와 합산 line+positron 차이를 숨기거나 보정하지 않는다. */
int cmf_gamma_nuclear_read(CMFGammaNuclearData*,const char *path);
/* 원본 NORM_LINES_ONLY 처방의 명시적 대조. 선 주파수/양전자 에너지는
 * 유지하고 누락 선 에너지를 기존 선 가중치에 배분한다. 관측자료 인자는 없다. */
int cmf_gamma_nuclear_complete_declared_Q(CMFGammaNuclearData*,const CMFGammaNuclearData*);
typedef struct {
    CMFGammaInput input;
    CMFEjectaEpoch epoch;
    double *storage;
} CMFToyGammaSource;
/* endpoint source. 원본의2일 조성에서 경과시간만 붕괴시킨다. nuclear가
 * source보다 오래 살아 있어야 한다(line energies 차용). 첫 사용 전0초기화. */
int cmf_toy_gamma_source_build(CMFToyGammaSource*,const CMFToyEjecta*,const CMFGammaNuclearData*,double epoch);
int cmf_toy_gamma_source_build_at(CMFToyGammaSource*,const CMFToyEjecta*,const CMFGammaNuclearData*,
                                double previous_epoch,double epoch,CMFDecaySourceMode);
void cmf_toy_gamma_source_free(CMFToyGammaSource*);
#ifdef __cplusplus
}
#endif
#endif
