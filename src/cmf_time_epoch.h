#ifndef LUMINA_CMF_TIME_EPOCH_H
#define LUMINA_CMF_TIME_EPOCH_H
#include "lumina_cmfgen.h"
#include "nlte_population_candidate.h"
#ifdef __cplusplus
extern "C" {
#endif

/* 한 물리 시점의 소유 snapshot. 첫 사용 전0초기화한다. 반복 후보와 실제
 * 과거를 혼동하지 않는다. */
typedef struct {
    NLTEPhysicalHistory material;
    CMFTimeRadiationHistory radiation;
    double *radiation_storage;
} CMFPhysicalTimeState;
void cmf_physical_time_state_free(CMFPhysicalTimeState *state);
/* 초기 이력 또는 수렴한 시점의 깊은복사. 자체로 수렴을 인증하지 않는다. */
int cmf_physical_time_state_capture(CMFPhysicalTimeState *state,
    const NLTEConfig*,const AtomicData*,const PlasmaState*,const CMFTimeRadiationHistory*);

typedef enum {
    CMF_EPOCH_OK=0,CMF_EPOCH_INVALID=1,CMF_EPOCH_ALLOCATION=2,
    CMF_EPOCH_RADIATION=3,CMF_EPOCH_MATERIAL=4,CMF_EPOCH_NOT_CONVERGED=5,
    CMF_EPOCH_HISTORY=6
} CMFTimeEpochStatus;
typedef struct {
    int iterations,radiation_calls,converged;
    double tolerance,temperature_update,electron_update,population_update,
           radiation_update,flux_update;
    uint64_t radiation_generation,population_generation;
} CMFTimeEpochReport;

/* cs는 현재 기하에서 초기화, time_trial은 실제 과거/경계를 지정해야 한다.
 * gamma는 현재 시점의 게시물이어야 한다. 밀도/조성/경계를 여기서 조작하지 않는다.
 * 현재 후보 객체들은 제자리 갱신된다. 실패 때 과거와 next는 보존되지만 현재
 * 후보는 마지막 반복 상태다. 호출자는 이를 다음 과거로 채택하면 안 된다.
 * 성공 때만 next를 교체한다. 관측 스펙트럼은 마지막 물질에서 준비하되 파일
 * 저장/다음 epoch의 기하와 초기조건 선택은 호출자 책임이다. tolerance는
 * 반복 수렴이지 이산 오차나 CMFGEN 동질성 보증이 아니다. */
CMFTimeEpochStatus cmfgen_solve_time_epoch(CMFGENState*,Geometry*,OpacityState*,BFOpacity*,
    PlasmaState*,NLTEConfig*,AtomicData*,GammaDeposition*,double T_inner,
    int iteration_cap,double tolerance,CMFPhysicalTimeState *next,CMFTimeEpochReport*);

/* 실제 정밀 재료/시간장/보존 재빈/R6. 정상장 해나 초기 predictor는 호출하지 않는다. */
int cmfgen_time_radiation_publish(CMFGENState*,Geometry*,OpacityState*,BFOpacity*,
    PlasmaState*,NLTEConfig*,AtomicData*,double T_inner,int observer);

/* 입력 온도 고정 정상 초기해. LUMINA_FIXED_TE_PROFILE은 입력 hydro T를
 * 담아야 하며 목표 시점의 T가 아니다. 성공은 초기 NLTE/복사장 공동 수렴만
 * 뜻한다. 성공 때만 같은 정밀 J와 실제 면 H/물질의 깊은복사 initial을
 * 교체한다. 실패하면 initial은 보존한다. 팽창 운전은 호출자 책임이다. */
typedef struct {
    int iterations,radiation_calls,converged;
    double tolerance,electron_update,population_update,radiation_update;
} CMFInitialNLTEReport;
CMFTimeEpochStatus cmfgen_solve_initial_nlte(CMFGENState*,Geometry*,OpacityState*,BFOpacity*,
    PlasmaState*,NLTEConfig*,AtomicData*,GammaDeposition*,double T_inner,
    int iteration_cap,double tolerance,CMFPhysicalTimeState *initial,CMFInitialNLTEReport*);
int cmfgen_initial_radiation_publish(CMFGENState*,Geometry*,OpacityState*,BFOpacity*,
    PlasmaState*,NLTEConfig*,AtomicData*,double T_inner,int observer);

/* R6와 물질 에너지 소비부가 동일한 시간장 의미를 확인한다. 정상장 오차
 * 인증을 요구하거나 시간장의 반복 차이를 오차 상계로 승격하지 않는다. */
static inline int cmf_time_line_field_qualified(const OpacityState *o,double old,double now){
    return o && old>0 && now>old && isfinite(now) &&
        o->jbar_line_det_operator==CMF_FINE_LINE_OPERATOR_COMMON_GAUSSIAN &&
        o->jbar_line_det_time_old==old && o->jbar_line_det_time_new==now &&
        o->jbar_line_det_time_equation_residual>=0 && o->jbar_line_det_time_equation_residual<=1e-10 &&
        o->jbar_line_det_exact_tolerance>0 && o->jbar_line_det_exact_tolerance<1 &&
        o->jbar_line_det_time_closure_update>=0 &&
        o->jbar_line_det_time_closure_update<o->jbar_line_det_exact_tolerance &&
        o->jbar_line_det_exact_residual>=0 && o->jbar_line_det_exact_residual<o->jbar_line_det_exact_tolerance &&
        o->jbar_line_det_exact_iterations>=2 && o->jbar_line_det_exact_iterations<=o->jbar_line_det_exact_iteration_cap &&
        o->jbar_line_det_error_upper && o->jbar_line_det_grid_n_bins>=2 &&
        o->jbar_line_det_grid_nu_min>0 && o->jbar_line_det_grid_nu_max>o->jbar_line_det_grid_nu_min &&
        o->jbar_line_det_profile_error_min>=0 && isfinite(o->jbar_line_det_profile_error_max) &&
        o->jbar_line_det_profile_error_max>=o->jbar_line_det_profile_error_min;
}
#ifdef __cplusplus
}
#endif
#endif
