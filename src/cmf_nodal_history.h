#ifndef CMF_NODAL_HISTORY_H
#define CMF_NODAL_HISTORY_H
#include "cmf_nodal_input.h"
#include "cmf_frequency_history.h"
/* radii=(입구,중간,출구). intensity는 전역 특성선 장의 begin:begin+count다.
 * small=0은 중간χ rolling 이력, small=1은 양끝 공간χ 적분이다. */
int cmf_nodal_history_range(CMFNodalInput *input,int global_nf,int begin,int count,int small,
    double h,double length,double delta_a,double delta_b,const double *radii,
    const double *intensity,double *out,double *response,CMFHistoryStats *stats);
int cmf_nodal_history_compare(CMFNodalInput *input,int global_nf,int begin,int width,int leaves,double h,
    const double *lengths,const double *deltas,const double *radii,const double *initial,
    double *out,double *errors,double *responses,CMFHistoryStats *stats,uint64_t *calls,uint64_t *checks);
int cmf_nodal_history_maps(CMFNodalInput *input,int global_nf,int begin,int width,int leaves,double h,
    const double *lengths,const double *deltas,const double *radii,double *maps,unsigned char *safe,
    CMFHistoryStats *stats,uint64_t *calls,uint64_t *checks);
/* maps는 leaf별 주소 배열이다. 먼저 primary 3쌍, 이어 overlap이 있는 piece의
 * alternate 쌍만 저장한다. safe=(leaf,ceil(width/8)) little-bit 순서다.
 * 안전하지 않은 연속 출력열만 실제 입사로 원래 비교를 수행한다.
 * fallback_stats=(실제scalar, kernel vector호출, 분기scalar)의3개 정수다. */
int cmf_nodal_history_replay(CMFNodalInput *input,int global_nf,int begin,int width,int leaves,double h,
    const double *lengths,const double *deltas,const double *radii,const double *const *maps,const unsigned char *safe,
    const double *initial,double *out,double *errors,double *responses,uint64_t max_fallback_scalar,uint64_t *fallback_stats);
#endif
