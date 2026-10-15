#ifndef LUMINA_CMF_EJECTA_H
#define LUMINA_CMF_EJECTA_H
#include "lumina.h"
#ifdef __cplusplus
extern "C" {
#endif

enum { CMF_TOY_ELEMENTS=6 }; /* Ni56, Co56, Fe56, Ca, S, Si 순서 */
/* 초기 입력의 실제 시점에서의 핵수/g. t=0 Ni 열을 현재 Ni로 재해석하지
 * 않는다. 안정 IGE/다른 원소를 가진 모델은 이 toy 전용 reader의 범위 밖이다.
 * 첫 사용 전0초기화. 호출자가 핵/원자 질량을 명시한다. */
typedef struct {
    int n_shells,n_source_shells;
    double epoch_s,mass_amu[CMF_TOY_ELEMENTS];
    double model_halflife_s[2],model_decay_energy_MeV[2];
    double initial_ni56_mass_g,max_input_density_error,max_input_fraction_error;
    double *storage,*velocity_edges,*mass_g,*initial_temperature,*number_per_g;
} CMFToyEjecta;
void cmf_toy_ejecta_free(CMFToyEjecta*);
/* 연속 원본 셸을 묶는다. 질량/각 핵수/초기 적분 aT^4를 보존한다.
 * requested_shells=0이면 원본 전체. 출력 실패시 기존 객체를 보존한다. */
int cmf_toy_ejecta_read(CMFToyEjecta*,const char *path,int requested_shells,
                      const double mass_amu[CMF_TOY_ELEMENTS]);
/* 기존 경계를 유지하면서 원본 T최대/T최소 비로만 추가 분할한다.
 * ratio=0이면 기존 reader와 같다. 양의 ratio는1보다 커야 한다.
 * 공개 이온분포를 사용하지 않으며 max_shells 초과 시 실패한다. */
int cmf_toy_ejecta_read_refined(CMFToyEjecta*,const char *path,int requested_shells,
                      const double mass_amu[CMF_TOY_ELEMENTS],double ratio,int max_shells);

typedef struct {
    double lambda[2];
    double gamma_energy_erg[2],positron_kinetic_erg[2];
} CMFDecaySourceData;
typedef enum { CMF_DECAY_ENDPOINT=1,CMF_DECAY_INTERVAL_AVERAGE=2 } CMFDecaySourceMode;
typedef struct {
    int n_shells;
    double epoch_s,previous_epoch_s;
    CMFDecaySourceMode source_mode;
    double *storage,*density,*number_density,*gamma_emission,*positron_emission;
} CMFEjectaEpoch;
void cmf_ejecta_epoch_free(CMFEjectaEpoch*);
/* 복사/양전자 방출률만 계산한다. 감마선 침적/열화/비열율로 대체하지 않는다.
 * 원본 시점 이후의 경과시간만 붕괴시키며 팽창 희석은 한 번만 적용한다. */
int cmf_ejecta_epoch_build(CMFEjectaEpoch*,const CMFToyEjecta*,
    const CMFDecaySourceData*,double previous_epoch_s,double epoch_s,CMFDecaySourceMode);
/* 이미 같은 셸수로 할당한 geometry/rho/원소조성에 적용한다. 현재 Ni/Co/Fe
 * 질량은 명시한56 동위원소 질량이다. 온도/ne/준위인구/과거는 바꾸지 않는다.
 * 반환 후 후보 인구를 투영된 과거로 초기화하고 실제 물질식을 풀어야 한다. */
int cmf_ejecta_apply_epoch(const CMFToyEjecta*,const CMFEjectaEpoch*,
                         Geometry*,PlasmaState*,AtomicData*);
#ifdef __cplusplus
}
#endif
#endif
