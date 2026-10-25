#ifndef CMF_FULLSR_CHAIN_H
#define CMF_FULLSR_CHAIN_H
#include "cmf_linear_native.h"
#ifdef __cplusplus
extern "C" {
#endif
int lumina_fullsr_chain_workspace(size_t *local);
int lumina_fullsr_chain_device_uuid(char *bytes);
int lumina_fullsr_chain_build(int steps,int rays,const double *length,const double *delta,
    const double *chi,const double *eta,void *maps,int *status);
int lumina_fullsr_chain_provider_build(const CMFNodalDenseView *view,int steps,int rays,
    const double *length,const double *delta,const CMFNodalLocation *location,const double *phase,
    const int *columns,void *maps,int *status);
int lumina_fullsr_chain_reduce(int rows,int rays,const void *input,void *output,int *status);
int lumina_fullsr_chain_apply(int rays,const void *maps,const CMFLinearPositive *initial,
    CMFLinearPositive *output,int *status);
/* 전 native 입사/출구는 별도 장치 배열이다. 원구간당 U 재구성은 한 번뿐이다. */
int lumina_fullsr_chain_stream(const CMFNodalDenseView *view,int steps,int first,int count,
    const double *length,const double *delta,const CMFNodalLocation *location,const double *phase,
    double h,const CMFLinearPositive *incident,CMFLinearPositive *output,int *status);
/* 관측/기존 경계 API에 연결하는 terminal 변환이다. 0/underflow/overflow 수를 누적한다.
 * counts[3]은 caller가 한 세대 시작/재개 때 초기화/복원한다. */
int lumina_fullsr_chain_encode(int nf,const double *input,CMFLinearPositive *output,int *status);
int lumina_fullsr_chain_decode(int nf,const CMFLinearPositive *input,double *output,uint64_t *counts,int *status);
#ifdef __cplusplus
}
#endif
#endif
