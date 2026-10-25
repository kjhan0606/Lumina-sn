#ifndef CMF_NODAL_DRIVER_H
#define CMF_NODAL_DRIVER_H
#include "cmf_nodal_history.h"

typedef struct {
    uint64_t scalar_formals,kernel_entries,branch_scalar_formals,physical_intervals;
    uint64_t reconstructions,limited_polynomials,slab_evaluations,max_mesh_leaves,max_rounds;
    uint64_t driver_heap_peak_bytes,kernel_heap_peak_bound_bytes,failed_interval,failed_path;
    double elapsed_seconds,last_global_error,last_global_tolerance;
    int status;
} CMFNodalDriverStats;

/* 기존 호출자의 Stats 버퍼를 확대하지 않는다. 선택 API의 별도 출력이다. */
typedef struct {
    uint64_t cache_build_scalar_formals,cache_replay_leaf_columns,cache_fallback_scalar_formals;
    uint64_t cache_workspace_peak_bytes,cache_leaf_builds,cache_evictions,global_rechecks,cache_peak_active_slabs;
} CMFCachedDriverStats;

/* GPU 외곽 실행기용 같은 기하 계산. edges=(n,2), 출력 lengths/deltas=(n,3),
 * radii=(n,5), locations=(n,5), shift=물리구간 입구의 delta/h다.
 * 계수/수송/장 계산은 하지 않는다. 실패한 출력은 소비하지 않는다. */
int cmf_nodal_driver_geometry(CMFNodalInput *input,int nr,const double *radius,const double *velocity,
    const double *gauss_x,const double *gauss_w,const double *step,double h,int n,const double *edges,
    double *lengths,double *deltas,double *radii,CMFNodalLocation *locations,double *shift);

/* 같은 propagate의 한 실제 구간/전체ν 비교 전용. 전체장·관측 누적은 없다. */
int cmf_nodal_driver_interval(CMFNodalInput *input,int nr,
    const double *radius,const double *velocity,const double *gauss_x,const double *gauss_w,
    const double *step,double dlognu,int panels,int block_width,double max_seconds,
    uint64_t max_full_vectors,const double *initial,double *output,
    CMFNodalDriverStats *stats,CMFCachedDriverStats *cache_stats);

/* paths=(npath,5): minus begin/count,plus begin/count,core flag.
 * steps=(nstep,5): impact,z0,z1,r0,r1. 관측은 step CSR이다.
 * output=(3,ns,nf); 입력 동결 J와 별개이며 호출 전 0으로 둔다.
 * 반환0만 수용이며 오차 추정은 인증 상한이 아니다. */
int cmf_nodal_driver(CMFNodalInput *input,int ns,int nf,int nr,
    const double *radius,const double *velocity,const double *gauss_x,const double *gauss_w,
    int npath,const int *paths,const double *core_mu,const double *core_weights,
    int nstep,const double *steps,int nevent,const int *event_ptr,const int *event_node,const double *event_weights,
    const double *planck,const double *gradient,double dlognu,int panels,int block_width,
    double rtol,double atol,double max_seconds,uint64_t max_full_vectors,uint64_t max_interval_vectors,
    int max_leaves,int max_depth,double *output,CMFNodalDriverStats *stats);
int cmf_nodal_driver_cached(CMFNodalInput *input,int ns,int nf,int nr,
    const double *radius,const double *velocity,const double *gauss_x,const double *gauss_w,
    int npath,const int *paths,const double *core_mu,const double *core_weights,
    int nstep,const double *steps,int nevent,const int *event_ptr,const int *event_node,const double *event_weights,
    const double *planck,const double *gradient,double dlognu,int panels,int block_width,
    double rtol,double atol,double max_seconds,uint64_t max_full_vectors,uint64_t max_interval_vectors,
    int max_leaves,int max_depth,double *output,CMFNodalDriverStats *stats,CMFCachedDriverStats *cache_stats);
/* 봉인된 현재 native90 전격자 전용. 기존 소형 API의120초 상한은 유지한다. */
int cmf_nodal_driver_native90(CMFNodalInput *input,int ns,int nf,int nr,
    const double *radius,const double *velocity,const double *gauss_x,const double *gauss_w,
    int npath,const int *paths,const double *core_mu,const double *core_weights,
    int nstep,const double *steps,int nevent,const int *event_ptr,const int *event_node,const double *event_weights,
    const double *planck,const double *gradient,double dlognu,int panels,int block_width,
    double rtol,double atol,double max_seconds,uint64_t max_full_vectors,uint64_t max_interval_vectors,
    int max_leaves,int max_depth,double *output,CMFNodalDriverStats *stats,CMFCachedDriverStats *cache_stats);
#endif
