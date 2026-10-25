#ifndef CMF_FULLSR_FROZEN_H
#define CMF_FULLSR_FROZEN_H
#include "cmf_fullsr_sc.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    uint64_t primitive_maps,compositions,completed_outputs,heap_bytes;
    double elapsed_seconds;
    int status,failed_column;
} CMFFrozenStats;
/* 공간 중점에서 물질과 b=delta/length를 동결한 fullSR 특성선 한 조각.
 * native bin 안은 양의 SC 맵으로 세분하고 순서 있는 이동창으로 합성한다.
 * 입사 U=I/nu^3는 reconstruction=0:선형,1:양의 단조 cubic으로 읽는다.
 * 최고nu 밖은 기존 상수 I/chi/eta 연장이다. [last,nf)만 완성하며 실패 때
 * 출력 전체를 보존한다. 이는 전공간 적응/자체장/물리 합격 API가 아니다. */
int cmf_fullsr_frozen_cell(const CMFNodalDenseView *view,const CMFNodalLocation *location,
    double length,double delta,double h,int bin_subdivisions,int reconstruction,int last,
    const CMFLinearPositive *incident,CMFLinearPositive *output,double max_seconds,CMFFrozenStats *stats);
/* 같은 공간 동결에서 bin 안의 선형 chi까지 적분하는 별도 선택이다.
 * 기존 양의 모멘트 코어를 재사용하고 평균 chi 선택은 비교용으로 보존한다. */
int cmf_fullsr_frozen_linear_cell(const CMFNodalDenseView *view,const CMFNodalLocation *location,
    double length,double delta,double h,int bin_subdivisions,int reconstruction,int last,
    const CMFLinearPositive *incident,CMFLinearPositive *output,double max_seconds,CMFFrozenStats *stats);
/* 원래 물리구간 입구에서 한 번만 출구 프레임 Q로 재구성한다. */
int cmf_fullsr_frozen_begin(int nf,double h,double delta,int reconstruction,int last,
    const CMFLinearPositive *incident,CMFLinearPositive *output,double max_seconds,CMFFrozenStats *stats);
/* Q_j는 같은 j 그대로 운반한다. source만 [j+db/h,j+da/h]를 조회한다.
 * T에는 이미3(da-db)가 있으므로 그 프레임 보정만 적용한다. */
int cmf_fullsr_frozen_fixed_cell(const CMFNodalDenseView *view,const CMFNodalLocation *location,
    double length,double da,double db,double h,int bin_subdivisions,int last,
    const CMFLinearPositive *incident,CMFLinearPositive *output,double max_seconds,CMFFrozenStats *stats);
typedef struct {CMFLinearPositive transmission,source;} CMFFrozenAffine;
/* 소수 특성선의 실제 r,nu 끝점 계수를 직접 진행한다. 공간 동결/재투영 없음.
 * chi/eta=(nstep+1,nray), maps=(nstep,nray). 선형 끝점 근사는 남는다. */
int cmf_fullsr_linear_chain(int nstep,int nray,const double *length,const double *delta,
    const double *chi,const double *eta,const CMFLinearPositive *initial,CMFLinearPositive *output,
    CMFFrozenAffine *maps,double max_seconds,CMFFrozenStats *stats,int *failed_step);
#ifdef __cplusplus
}
#endif
#endif
