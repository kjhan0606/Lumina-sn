#ifndef CMF_NODAL_INPUT_H
#define CMF_NODAL_INPUT_H
#include <stdint.h>
typedef struct CMFNodalInput CMFNodalInput;
/* 계수/J와 독립인 반경 보간 위치. 같은 context의 반경 좌표에만 사용한다. */
typedef struct {int mode,first,last;double weight;} CMFNodalLocation;
typedef struct {
    uint64_t queries,cache_hits,cache_misses,source_bytes,source_preads;
    uint64_t profile_hits,profile_misses,bindings,field_bytes_copied;
    uint64_t static_cache_bytes,profile_cache_bytes,field_buffer_bytes,cache_peak_rows;
} CMFNodalInputStats;
typedef struct {
    uint64_t canonical_bytes,profile_bytes,native_accesses,profile_accesses;
    double preparation_seconds;
    int ready;
} CMFNodalDenseStats;
/* 읽기 전용 빌린 뷰다. context 파괴/J 재결합 후 다시 요청해야 한다.
 * canonical=(χ,ES,진η)×ns×nf, outer=nf×(2*ne+1)이다.
 * jmode=0은 명시적0,1은 전체 native J다. generation은 같은 context의 bind 횟수다. */
typedef struct {
    int ns,nf,ne,jmode;
    uint64_t generation;
    const double *canonical,*outer,*j,*radius,*external_radius;
} CMFNodalDenseView;
int cmf_nodal_input_dense_view(const CMFNodalInput *ctx,CMFNodalDenseView *view);
/* 선택 경로다. 원래 합산 순서의3평면과 현재 동결J의 외부profile을 준비한다.
 * 기존 Stats ABI/희소 기본 경로는 유지한다. J 재결합은 외부profile만 폐기한다. */
int cmf_nodal_input_prepare_dense(CMFNodalInput *ctx,double max_seconds);
int cmf_nodal_input_dense_stats(const CMFNodalInput *ctx,CMFNodalDenseStats *stats);
/* fd는 O_RDONLY다. 소유 dup를 만들며 source 본문 전체를 mmap하지 않는다. */
CMFNodalInput *cmf_nodal_input_create(int fd,int ns,int nf,int added,int reference_b,
    const double *native_radius,const double *radius,double legacy_eta_minimum,
    int slab_width,int cache_rows,int profiles,int *code);
void cmf_nodal_input_destroy(CMFNodalInput *ctx);
/* 한 F 안의 제한된 연속 J slab. 복사한 동결 값만 사용하며 이전 총η cache를 버린다. */
int cmf_nodal_input_bind(CMFNodalInput *ctx,int start,int count,const double *j);
/* 호출자가 수명과 읽기 전용을 보장하는 C-order native J. 소유/복사하지 않는다.
 * 전체 배열을 검사하지 않고 실제 조회한 값의 유한성/비음수만 검사한다. */
int cmf_nodal_input_bind_view(CMFNodalInput *ctx,const double *j);
/* 명시적 국소 J=0 시험만을 위한 모드. 전체 J 배열/자체해를 만들지 않는다. */
int cmf_nodal_input_bind_zero_trial(CMFNodalInput *ctx);
int cmf_nodal_input_raw(CMFNodalInput *ctx,int nr,const int *rows,int nc,const int *columns,
    double *chi,double *es,double *eta);
int cmf_nodal_input_sample(CMFNodalInput *ctx,double radius,double column,double *chi,double *eta);
int cmf_nodal_input_prepare(const CMFNodalInput *ctx,double radius,CMFNodalLocation *location);
int cmf_nodal_input_sample_prepared(CMFNodalInput *ctx,const CMFNodalLocation *location,
    double column,double *chi,double *eta);
int cmf_nodal_input_eval(CMFNodalInput *ctx,int nr,const double *radii,int nc,const double *columns,
    double *chi,double *eta);
int cmf_nodal_input_stats(const CMFNodalInput *ctx,CMFNodalInputStats *stats);
int cmf_nodal_input_shape(const CMFNodalInput *ctx,int *ns,int *nf);
#endif
