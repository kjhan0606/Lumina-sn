#ifndef CMF_FOREST_CUDA_H
#define CMF_FOREST_CUDA_H
#include <stddef.h>
#include <stdint.h>
#include "cmf_nodal_input.h"
#ifdef __cplusplus
extern "C" {
#endif
/* 독립 맵/트리 묶음이다. 포인터는 장치 주소, nodes_offset은 Map 원소 단위다.
 * output=Map[width], 작은 이동/무이력은 leaves=0이다. */
typedef struct {
    CMFNodalDenseView view;
    CMFNodalLocation locations[3];
    double h,length,da,db;
    int low,count,leaves,begin,width,small;
    uint64_t nodes_offset;
    void *output;
} CMFForestJob;
/* 한 독립 주파수 slab의 시간순 공간 replay. first는 pointer/score 원소 offset. */
typedef struct {int first,count,begin,width;} CMFForestReplay;
size_t lumina_forest_job_bytes(void);
size_t lumina_forest_replay_bytes(void);
int lumina_forest_workspace(size_t *local);
/* n<=128. jobs는 장치 배열이다. forest는 Σ2*leaves Map이다.
 * 무거운 grid는32768 threads 이하씩 발행한다. enqueue0은 완료가 아니다. */
int lumina_forest_cuda(int n,const CMFForestJob *jobs,int max_leaves,int max_width,void *forest,int *status);
int lumina_forest_replay_cuda(int n,const CMFForestReplay *jobs,int max_width,int slots,int slot_width,
    const void *const *primary,const void *const *alternative,const unsigned *masks,
    const double *initial,double *output,void *results,double *scores,double *summaries,int *status);
#ifdef __cplusplus
}
#endif
#endif
