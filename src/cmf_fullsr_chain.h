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
#ifdef __cplusplus
}
#endif
#endif
