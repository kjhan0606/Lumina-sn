#ifndef CMF_LINEAR_CUDA_H
#define CMF_LINEAR_CUDA_H
#include "cmf_linear_native.h"
#ifdef __cplusplus
extern "C" {
#endif
/* 구조체는 host, 내부 포인터/nodes/maps/상태는 장치 주소다. 한 J 세대에 고정한다. */
typedef struct {
    CMFNodalDenseView view;CMFNodalLocation locations[2];
    double length,h,da,db,offset,span;
    /* source_leaves는 종전192B 구조의 끝 padding을 사용한다. 기존 변형은 같은 leaves다. */
    int begin,count,base,nf,block,leaves,bins,source_leaves;
} CMFLinearCudaPiece;
size_t lumina_linear_cuda_node_bytes(void);
size_t lumina_linear_cuda_map_bytes(void);
size_t lumina_linear_cuda_local_bytes(void);
int lumina_linear_cuda_workspace(size_t *local);
int lumina_linear_cuda_plan(const CMFNodalDenseView *view,const CMFNodalLocation *locations,
    double length,double h,double da,double db,int begin,int count,int block,CMFLinearCudaPiece *out);
int lumina_linear_cuda_build(const CMFLinearCudaPiece *piece,void *nodes,int *status);
int lumina_linear_cuda_query(const CMFLinearCudaPiece *piece,const void *nodes,double relative,
    int max_panels,void *maps,int *status);
#ifdef CMF_GPU_DIAGNOSTIC_RANGE
/* 진단 전용. 프로필 좌표는 그대로이고 maps는 원 piece.count 크기의 배열이다.
 * first/count 위치에만 쓰며 전체 출력으로 소비하지 않는다. */
int lumina_linear_cuda_query_range(const CMFLinearCudaPiece *piece,const void *nodes,double relative,
    int max_panels,int first,int count,void *maps,int *status);
#endif
/* 각 maps[row]는 coarse/left/right×stride. 오류는 시작 때부터 sticky다.
 * 반환0은 enqueue일 뿐이며 sync/status 전 출력 소비는 금지한다. */
int lumina_linear_cuda_replay(int n,int width,int stride,int block_width,double relative,
    const void *const *maps,const double *initial,double *output,void *results,
    double *scores,double *slab_ratios,void *frequency_errors,int *status);
/* 임의 양의 가중치의 기계적 축약이다. 생산 PHOT 계수 여부는 caller가 봉인한다.
 * pairs는 ceil(nf/4096)×2개의 scaled(value,error); 실제 광흡수율은 완성 J에서만 산출한다. */
int lumina_linear_cuda_weighted(int nf,const double *weights,const double *intensity,const void *errors,void *pairs,int *status);
/* 기존 입사 재구성/경계/각도 누적의 같은 API. C의 수송 연산자는 사용하지 않는다. */
int lumina_history_cuda_reconstruct(int nf,double shift,const double *initial,double *output,uint64_t *limited,int *status);
int lumina_history_cuda_boundary(int nf,const double *planck,const double *gradient,double mu,double *ray,int *status);
int lumina_history_cuda_accumulate(int ns,int nf,int observations,const int *nodes,const double *weights,
    const double *ray,double *moments,int *status);
#ifdef __cplusplus
}
#endif
#endif
