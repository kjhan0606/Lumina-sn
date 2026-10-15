#ifndef LUMINA_CMF_TIME_FIELD_H
#define LUMINA_CMF_TIME_FIELD_H
#include "cmf_time_moment.h"
#include "cmf_exact_sliding.h"
#ifdef __cplusplus
extern "C" {
#endif

/* 형식해는 각도 closure만 제공한다. 시간 J를 정상 고정점으로 덮지 않는다. */
typedef int (*CMFTimeFormalApply)(void *context,const double *J,CMFFormalMoments *out);
typedef struct {
    int iterations,iteration_cap,converged;
    double tolerance,relative_update,closure_update,equation_residual;
    CMFTimeMomentStatus status; /* 호출자가 dt 부적합과 입력/물질 오류를 구분한다. */
} CMFTimeFieldReport;

/* p.eddington/outer_h/outer_f는 이 함수가 실제 형식해로 대체한다.
 * initial_J는 반복 초기값일 뿐이며 old_J/H가 실제 과거다. delta_J는 마지막
 * 반복 차이로 엄밀한 오차 상계가 아니다. 실패하면 모든 출력 배열을 보존한다. */
CMFTimeMomentStatus cmf_time_field_solve(const CMFTimeMomentInput *p,
    const double *initial_J,CMFTimeFormalApply formal,void *context,
    int iteration_cap,double tolerance,double *new_J,double *new_H,
    double *delta_J,CMFTimeFieldReport *report);

/* 원본 Lagrange/frequency 격자와 실제 과거를 빌린다. trial은 새 결과만 소유한다. */
typedef struct {
    int n_shells,n_bins;
    double epoch_s,nu_min,dlognu;
    const double *velocity_edges,*J,*H;
} CMFTimeRadiationHistory;
typedef struct {
    const CMFTimeRadiationHistory *old;
    const double *inner_H;
    double epoch_s;
    double *J,*H;
    CMFTimeFieldReport report;
} CMFTimeRadiationTrial;
void cmf_time_radiation_trial_free(CMFTimeRadiationTrial *trial);

#ifdef __cplusplus
}
#endif
#endif
