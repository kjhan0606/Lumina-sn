#ifndef LUMINA_CMF_OUTER_TRANSPORT_H
#define LUMINA_CMF_OUTER_TRANSPORT_H
#include <stddef.h>
#include <stdint.h>
#include "cmf_exact_sliding.h"
#define CMF_OUTER_REFERENCE_ETA_MIN_CGS (1e-280/1e10)

/* 명시적 외부대기 후보 API. 기존 생산 solver의 자동 대체 경로가 아니다.
 * 원래 셀은 반경 오름차순, 마지막 두 J는 native 외삽 기준 노드 표본이다.
 * 포인터 계수는 context 생존 중 불변/유효해야 한다. 계획은 내부에 복사한다.
 * 외삽의 비선형성을 보존하며 기존 affine K 오차 인증을 제공하지 않는다. */
typedef struct {
    double length,drift,weight;
    int32_t exterior,index,observation_begin,observation_count;
} CMFOuterStep;

typedef struct {
    int32_t sample;
    double mu;
} CMFOuterObservation;

typedef struct {
    int32_t minus_begin,minus_count,plus_begin,plus_count;
    int32_t core,turn_begin,turn_count;
    double inner_mu;
} CMFOuterRay;

typedef struct {
    int32_t cells,frequencies,rays,steps,observations,exterior_nodes;
    double dlognu,log_reference_radius_ratio;
    const CMFOuterStep *step;
    const CMFOuterRay *ray;
    const CMFOuterObservation *observation;
    const double *turning_mu; /* cells+2, 각 표본의 접선 μ=-β */
    const double *exterior_ratio; /* r_original_outer/r, 접합→최외곽 순서 */
    const double *cell_chi,*cell_es,*cell_eta;
    const double *node_chi,*node_es,*node_eta; /* 기준점2개×frequencies */
    const double *inner_planck,*inner_gradient;
    size_t work_budget_bytes;
} CMFOuterProblem;

typedef struct {
    CMFExactStatus status;
    int iterations,formal_calls;
    double change_max_norm_ratio,residual_max,residual_max_norm_ratio;
    size_t work_bytes;
    int componentwise_error_verified; /* 항상0: 비선형 해오차 상한 미제공 */
} CMFOuterReport;

typedef struct CMFOuterContext CMFOuterContext;
/* 기하와 차원만으로 계산하는 작업공간 상한. 계수/J는 읽거나 할당하지 않는다.
 * readonly 입력/호출자 출력/Python 메모리를 포함한 전체 RSS 상한은 아니다. */
CMFExactStatus cmf_outer_workspace_bytes(const CMFOuterProblem *problem,size_t *work_bytes);
CMFExactStatus cmf_outer_create(const CMFOuterProblem *problem,
    CMFOuterContext **result,size_t *work_bytes);
/* 명시 opt-in. 논리 worker와 승인 CPU 상한은1..16, moments는1(J) 또는3.
 * CPU1에서도 W2는 순차 실행한다. C 작업공간은 전체 RSS/실행 승인이 아니다. */
CMFExactStatus cmf_outer_weighted_workspace_bytes(const CMFOuterProblem *problem,
    int workers,int moments,int approved_cpus,size_t *work_bytes);
CMFExactStatus cmf_outer_create_weighted(const CMFOuterProblem *problem,
    int workers,int moments,int approved_cpus,CMFOuterContext **result,size_t *work_bytes);
/* 가장 최근 F의 실제 ray 실행 팀 크기. ray 실행 전에 실패했으면0이다. */
int cmf_outer_parallel_threads(const CMFOuterContext *context);
void cmf_outer_destroy(CMFOuterContext *context);
CMFExactStatus cmf_outer_apply(CMFOuterContext *context,const double *own_j,
    double *new_j,double *new_h,double *new_k);
CMFExactStatus cmf_outer_solve(CMFOuterContext *context,double *own_j,
    int iteration_cap,double tolerance,CMFOuterReport *report);
#endif
