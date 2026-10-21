#ifndef CMF_LINEAR_NATIVE_H
#define CMF_LINEAR_NATIVE_H
#include "cmf_nodal_input.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {double mantissa;int64_t exponent;} CMFLinearPositive;
typedef struct {
    CMFLinearPositive transmission,source,source_error,source_roundoff,source_frequency,source_omitted;
    double tau_error;
    uint64_t built_bins,tree_combinations;
    int panels,accepted,split,pruned,max_depth;
} CMFLinearNativeMap;
size_t cmf_linear_native_map_bytes(void);
/* host 포인터의 한 공간선형 조각이다. 원래 열 하나의 이동 경로만 준비한다.
 * locations[2]=(입구,출구), delta는 원 물리구간 출구 기준이다.
 * 입력과 생성번호는 호출 동안 동결한다. source 오차는 기하 오차가 아니다.
 * 실패는 기존 out을 보존한다. T/S 모두 마지막 관측 변환까지 scale을 유지한다. */
int cmf_linear_native_map(const CMFNodalDenseView *view,uint64_t generation,const CMFNodalLocation *locations,
    double length,double h,double da,double db,int column,int block,double relative,int max_panels,CMFLinearNativeMap *out);
#ifdef __cplusplus
}
#endif
#endif
