#ifndef CMF_HISTORY_CUDA_H
#define CMF_HISTORY_CUDA_H
#include <stddef.h>
#include "cmf_nodal_input.h"
#ifdef __cplusplus
extern "C" {
#endif
/* view의 포인터와 nodes/status/initial/maps/output은 장치 주소다.
 * view/locations 구조 자체는 host다. 메모리 소유권은 호출자에 있다.
 * status는 상위 실행기가 시작 전에0으로 만들며 오류는 호출 사이에 유지된다.
 * 반환0은 enqueue 성공뿐이다. 동기화+status 검사 전 출력 사용/출판 금지. */
size_t lumina_history_cuda_map_bytes(void);
int lumina_history_cuda_workspace(size_t *build_local,size_t *query_local,size_t *small_local);
int lumina_history_cuda_build(const CMFNodalDenseView *view,const CMFNodalLocation *locations,
    double h,double length,double da,double db,int low,int count,int leaves,void *nodes,int *status);
/* small=0: 주파수 history,1: 작은 공간 이동. maps는 순서 보존 Map[count].
 * initial/output은 둘 다NULL(맵 준비) 또는 둘 다유효(직접 적용)여야 한다.
 * 작은 이동은 nodes 불필요. 다른 경로의 tree는 [low,low+count) bins를 포함해야 한다. */
int lumina_history_cuda_query(const CMFNodalDenseView *view,const CMFNodalLocation *locations,
    double h,double length,double da,double db,int small,int tree_low,int tree_count,int leaves,
    const void *nodes,int begin,int count,const double *initial,double *output,void *maps,int *status);
size_t lumina_history_cuda_result_bytes(void);
int lumina_history_cuda_replay_workspace(size_t *forward_local,size_t *backward_local);
int lumina_history_cuda_aux_workspace(size_t *local);
/* primary/alternative는 장치 pointer[n]이며 각 대상은 coarse,left,right×width 맵이다.
 * masks[n]의 bit0/1/2가 겹침 분기를 표시한다. results[n*width]는 내부 작업공간.
 * scores[n], summary[2]=(전역오차,현재최대값)는 호출마다0으로 초기화한다.
 * stride는 맵 plane 간격이다. width<=stride<=2000000,width<=65536,n<=4096,
 * n*width<=2097152. 호출 순서/오류·출판 규약은 위와 같다. */
int lumina_history_cuda_replay(int n,int width,int stride,const void *const *primary,const void *const *alternative,
    const unsigned *masks,const double *initial,double *output,void *results,double *scores,double *summary,int *status);
/* 첫 공간 leaf의 전ν 맵을 재사용하여 slab별 오차/현재최대값을 환원한다. */
int lumina_history_cuda_initial(int nf,int block_width,const void *primary,const void *alternative,unsigned mask,
    const double *initial,double *output,double *errors,double *maxima,int *status);
int lumina_history_cuda_reconstruct(int nf,double shift,const double *initial,double *output,uint64_t *limited,int *status);
int lumina_history_cuda_boundary(int nf,const double *planck,const double *gradient,double mu,double *ray,int *status);
int lumina_history_cuda_accumulate(int ns,int nf,int observations,const int *nodes,const double *weights,
    const double *ray,double *moments,int *status);
#ifdef __cplusplus
}
#endif
#endif
