#ifndef CMF_FREQUENCY_HISTORY_H
#define CMF_FREQUENCY_HISTORY_H

#include <stdint.h>

/* 별도 후보 API. 이전 생산 ABI/실행기는 변경하지 않는다.
 * 거리 모멘트와 d는 같은 전체 소구간 길이로 정규화한다.
 * 물리 단위의 큰 거리 모멘트를 만든 뒤 나누는 중간 overflow를 피한다. */
typedef struct {
    double d, t, aa, pa, ab, mb;
} CMFHistoryMap;

typedef struct {
    uint64_t full_bin_maps, partial_bin_maps, blue_maps, compositions;
    uint64_t pushes, transfers, queue_peak, queue_bytes;
    uint64_t quadrature_nodes, quadrature_panels, slabs;
} CMFHistoryStats;
/* C 내부 제공자 전용. point는 호출자가 정한 반경 표본 번호이며 column은 전역
 * logν 점 인덱스다. Python 콜백을 주파수 루프에 넣기 위한 API가 아니다. */
typedef int (*CMFCoefficientProvider)(void *user,int point,double column,double *chi,double *eta);

int cmf_history_map(double length, double normalization,
    double k0, double k1, double ea0, double ea1, double eb0, double eb1,
    CMFHistoryMap *out, CMFHistoryStats *stats);
int cmf_history_compose(CMFHistoryMap left, CMFHistoryMap right, CMFHistoryMap *out);
int cmf_history_reconstruct(int nf,double shift,const double *input,double *out,uint64_t *limited);
/* CMF_EXPQUAD_ENGINE 선택 빌드만 제공. 실패 시 출력은 바꾸지 않는다. */
int cmf_history_moments_engine(double q0,double q1,double *out);
/* 선택 엔진의 같은 스레드 일반 모멘트 시도/성공만 센다. 기존 Stats ABI는 유지한다. */
void cmf_history_moments_engine_usage(uint64_t *attempts,uint64_t *accepted,int reset);

/* intensity는 이미 ν_i+delta_a에 놓인 I_local이다. 여기서 입사 재보간이나
 * exp(-3 delta_a)를 추가하지 않는다. χ/η는 logν 선형 점값이며 청색 끝값을 연장한다.
 * naive=1은 작은 독립 누적 대조 전용이며 production 경로가 아니다. */
int cmf_frequency_history(int nf, double h, double length, double delta_a, double delta_b,
    const double *chi, const double *eta_a, const double *eta_b,
    const double *intensity, double *out, int slab_width, int naive, CMFHistoryStats *stats);

int cmf_frequency_history_response(int nf,double h,double length,double delta_a,double delta_b,
    const double *chi,const double *eta_a,const double *eta_b,const double *intensity,
    double *out,int slab_width,int naive,double *response,CMFHistoryStats *stats);
int cmf_frequency_history_provider(int global_nf,int begin,int count,double h,double length,double delta_a,double delta_b,
    CMFCoefficientProvider provider,void *user,int chi_point,int eta_a_point,int eta_b_point,
    const double *intensity,double *out,double *response,CMFHistoryStats *stats);
/* 작은 공간 적응 비교용. lengths/deltas=(count,3), chi/eta=(count,3,nf).
 * 전ν×leaf 저장은 작은 검증에 한정한다. 실제 미세장은 별도 slab 연결이 필요하다. */
int cmf_frequency_history_compare(int count,int nf,double h,const double *lengths,
    const double *deltas,const double *chi,const double *eta,const double *initial,
    double *out,double *errors,double *responses,CMFHistoryStats *stats);

/* 작은 이동의 실제 logν 절점 분할. 각 조각의 특성선 끝점 χ/η를 선형 투영한다.
 * 공간-주파수의 쌍선형 보간 전체를 정확 적분한다고 주장하지 않는다.
 * 남은 공간/투영 오차는 전체 대 두 반구간 비교로 제어한다. */
int cmf_frequency_small_spatial(int nf,double h,double length,double delta_a,double delta_b,
    const double *chi_a,const double *chi_b,const double *eta_a,const double *eta_b,
    const double *initial,double *out,double *response,CMFHistoryStats *stats);
int cmf_frequency_small_provider(int global_nf,int begin,int count,double h,double length,double delta_a,double delta_b,
    CMFCoefficientProvider provider,void *user,int first_point,int last_point,
    const double *initial,double *out,double *response,CMFHistoryStats *stats);
/* chi=(count,5,nf): 입구/1/4/중간/3/4/출구. eta=(count,3,nf).
 * q≈1에서는 두 방법 차이도 오차에 더하고 실제 추가 formal 호출을 센다. */
int cmf_frequency_hybrid_compare(int count,int nf,double h,const double *lengths,
    const double *deltas,const double *chi,const double *eta,const double *initial,
    double *out,double *errors,double *responses,CMFHistoryStats *stats,
    uint64_t *formal_calls,uint64_t *branch_checks);
/* 제공자의 반경 표본 번호는 각 공간 leaf의5점이며 point=leaf*5+offset이다.
 * errors/responses는 leaf×출력 slab만 저장하며 전ν×leaf가 아니다. */
int cmf_frequency_hybrid_provider_compare(int global_nf,int begin,int width,int count,double h,
    const double *lengths,const double *deltas,CMFCoefficientProvider provider,void *user,
    const double *initial,double *out,double *errors,double *responses,CMFHistoryStats *stats,
    uint64_t *formal_calls,uint64_t *branch_checks);

/* 동결 계수의 leaf별 whole/left/right와 대체분기의 T,B. maps=(leaf,6,2,width).
 * B는 입사0에서 직접 얻고 T는 kernel 응답을 그대로 쓴다. safe=(leaf,width)다.
 * safe=0인 열은 실제입사로
 * 원래 비교를 실행해야 한다. 압축 전파가 극단값의 순차 계산을 바꾸지 않게 한다. */
int cmf_frequency_hybrid_provider_maps(int global_nf,int begin,int width,int count,double h,
    const double *lengths,const double *deltas,CMFCoefficientProvider provider,void *user,
    double *maps,unsigned char *safe,CMFHistoryStats *stats,uint64_t *formal_calls,uint64_t *branch_checks);

#endif
