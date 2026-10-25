#ifndef CMF_FULLSR_SC_H
#define CMF_FULLSR_SC_H
#include "cmf_nodal_input.h"
#include "cmf_linear_native.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    uint64_t completed_frequencies,space_updates,observation_updates,heap_bytes;
    double elapsed_seconds;
    int status,next_frequency,failed_path,failed_frequency,failed_vertex;
} CMFFullSRSCStats;
/* U=I_nu/nu^3 상풍식의 같은 주파수 I 단위 SC 한 조각.
 * state=(입사 I, 이전 청색의 왼쪽 I, 이전 청색의 오른쪽 I).
 * signed chi를 보존하며 source/chi 나눗셈을 하지 않는다. */
int cmf_fullsr_sc_step(double length,double drift,double h,int top,
    double chi0,double chi1,double eta0,double eta1,
    const CMFLinearPositive *state,CMFLinearPositive *output);
/* 같은 동결 source의 원래 주파수 first..last(내림차순)를 전진한다.
 * kind=1:외부 영입사,2:core 확산 경계,0:공간 전파.
 * blue는 바로 앞 청색 주파수의 전공간 상태이며 호출 사이에도 보존한다.
 * 완성 타일만 moments/escape에 저장하고 타일 실패 때 blue를 복구한다.
 * 반환4는 시간 한도다. stats.next_frequency부터 계속할 수 있다.
 * moments=(3,ns,nf), escape=(npath,nf)는 선택 출력이다. 전체장 수용은
 * 원격자 전부가 완료된 뒤 호출자가 별도로 판정한다. */
int cmf_fullsr_sc_advance(const CMFNodalDenseView *view,
    int npath,int nvertex,const int *path_ptr,const int *kind,
    const CMFNodalLocation *location,const double *length,const double *drift,const double *core_mu,
    int nevent,const int *event_ptr,const int *event_node,const double *event_weights,
    const double *planck,const double *gradient,double h,int frequency_subdivisions,
    int first,int last,int tile_width,int threads,double seconds,
    CMFLinearPositive *blue,double *moments,double *escape,CMFFullSRSCStats *stats);
#ifdef __cplusplus
}
#endif
#endif
