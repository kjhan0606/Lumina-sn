#ifndef LUMINA_CMF_EXACT_SLIDING_H
#define LUMINA_CMF_EXACT_SLIDING_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

/* 음의 물질 소광은 곧 음의 광도가 아니다. 유한한 signed 감쇠와
 * eta>=0를 사용한다. 영소광 방출에는 emissivity API를 사용한다. */
static inline int cmf_exact_extinction_valid(double ct, double ce, double a)
{
    double effective = ct + 3.0*a;
    return fabs(ct)<=DBL_MAX && ce>=0 && ce<=DBL_MAX &&
           a>0 && a<=DBL_MAX && fabs(effective)<=DBL_MAX;
}

static inline double cmf_exact_source_scale(double ct,double a,double drift)
{ return fmax(drift,fabs(ct+3*a)); }

/* source=eta/(a_drift*max(1,|dt_bin|)). 모든 구간 방출 가중은 양수다.
 * 작은 tau는 expm1로, 큰 양의 dt는 예전 source 크기로 계산해 유지한다. */
#ifdef __CUDACC__
#define CMF_COEF_HD __host__ __device__
#else
#define CMF_COEF_HD
#endif
static CMF_COEF_HD inline double cmf_exact_emission_weight(double depth,double length)
{
    if(length==0)return 0;
    if(fabs(depth)>=1)return depth>0?-expm1(-depth*length):expm1(-depth*length);
    double tau=depth*length;
    return tau==0?length:length*(-expm1(-tau)/tau);
}
#undef CMF_COEF_HD
static inline int cmf_exact_material_source_valid(double ct, double sf)
{
    return fabs(ct)<=DBL_MAX && fabs(sf)<=DBL_MAX &&
           (ct<0 ? sf<=0 : sf>=0);
}
static inline int cmf_exact_fixed_valid(double ct,double fixed,int emissivity)
{ return emissivity ? fixed>=0 && fixed<=DBL_MAX : cmf_exact_material_source_valid(ct,fixed); }

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CMF_EXACT_MODE_SLIDING = 1,
    CMF_EXACT_MODE_DIRECT_REFERENCE = 2,
    /* Subtraction-free O(1)-amortized window composition.  This mode is kept
     * separate until direct-oracle and full production gates are sealed. */
    CMF_EXACT_MODE_POSITIVE_SLIDING = 3
} CMFExactMode;

typedef enum {
    CMF_EXACT_OK = 0,
    CMF_EXACT_INVALID_INPUT = 1,
    CMF_EXACT_ALLOCATION_FAILED = 2,
    CMF_EXACT_NONFINITE = 3,
    CMF_EXACT_NEGATIVE_RECURRENCE = 4,
    CMF_EXACT_NOT_CONVERGED = 5,
    CMF_EXACT_ERROR_ENVELOPE_FAILED = 6
} CMFExactStatus;

typedef struct {
    CMFExactStatus status;
    int mode;
    int iterations_used;
    int iteration_cap;
    double tolerance;
    double final_max_relative_change;
    double final_max_absolute_change;
    /* 양의 유효 소광일 때 지역 계수비로 구한 보수적 상계.
     * 증폭/영감쇠에서는 이 값은 무한대일 수 있으며 아래 실제 연산자
     * 상계를 사용한다. 지역비와 산란 연산자 노름을 혼동하지 않는다. */
    double max_scattering_ratio;
    /* 실제 양의 산란 연산자의 ||K||_infinity 상계. 지역 계수비와 구별. */
    double scattering_operator_norm_upper;
    /* 별도 양의 가중 노름 인증. 미사용/미인증은 INFINITY. */
    double scattering_operator_weighted_norm_upper;
    double fixed_point_absolute_error_bound;
    double max_characteristic_drift_bins;
    uint64_t negative_recurrence_count;
    double first_negative_recurrence;
    size_t n_rays;
    size_t segment_slots;
    int componentwise_error_envelope_verified;
    size_t componentwise_error_seed_attempts;
    size_t componentwise_error_refinement_iterations;
    double componentwise_residual_upper_max;
    double componentwise_error_upper_min;
    double componentwise_error_upper_max;
} CMFExactReport;

/* Exact homologous drifting-characteristic formal solution.
 *
 * Frequency bins are ascending in nu (red -> blue), with constant dlognu.
 * The production mode is CMF_EXACT_MODE_SLIDING, O(1) work per output bin.
 * CMF_EXACT_MODE_DIRECT_REFERENCE is an independent O(beta) oracle intended
 * only for small-grid selftests.  J is both the ALI initial state and output.
 * I_nu obeys dI/ds=eta-(chi_tot+3a)I along dln(nu)/ds=-a,
 * with a=1/(c*time_explosion). S_fixed=eta_fixed/chi_tot: the solver
 * applies chi_tot/(chi_tot+3a) to this material source, and
 * chi_es/(chi_tot+3a) to J. A finite S_fixed at chi_tot=0 emits nothing.
 */
CMFExactStatus cmf_exact_characteristic_solve(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J,
    int iteration_cap, double tolerance, CMFExactMode mode,
    CMFExactReport *report);

CMFExactStatus cmf_exact_characteristic_solve_with_envelope(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J, double *componentwise_error_upper,
    size_t envelope_refinement_iterations,
    int iteration_cap, double tolerance, CMFExactMode mode,
    CMFExactReport *report);

/* One affine F(J)=b+KJ application using the positive sliding owner.  lower
 * and upper are outward binary64 enclosures of the same operation returned in
 * nearest.  This diagnostic API is the small-grid oracle for the production
 * residual/envelope integration; it does not mutate input_J. */
CMFExactStatus cmf_exact_characteristic_apply_positive_bounds(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, const double *input_J,
    double *lower, double *nearest, double *upper);

const char *cmf_exact_status_name(CMFExactStatus status);

/* 한 번의 기존 형식해에서 얻은 각도 모멘트. 시간 의존 J를 대신하지 않으며
 * 변수 Eddington 해법의 K/J 및 외곽 H/J를 계산하는 입력이다.
 * 셸 중심: ns*nb, 외곽면: nb. J=0일 때 비를 임의로 만들지 않는다.
 * CPU 소형 연결 시험용이며 현재 ray*segment 저장 방식은 큰 격자용이 아니다. */
typedef struct {
    double *J, *H, *K;
    double *outer_J, *outer_H, *outer_K;
    /* 선택 출력: (ns+1)*nb의 실제 면 H. NULL이면 저장/계산하지 않는다.
     * 중심 r=0은 대칭으로0, 내부 코어는 지정 B와 유입 강도의 차이다.
     * 구조체를 반드시0 또는 전체 aggregate 초기화한다. */
    double *face_H;
} CMFFormalMoments;

CMFExactStatus cmf_exact_formal_moments(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, const double *input_J, CMFFormalMoments *moments);

/* 생산용 방출률 직접 입력. chi=0,eta>0 및 signed 유효 소광도 표현한다.
 * error_upper가 NULL이면 성분별 상계 없이 지정 mode의 형식해를 푼다.
 * 실패시 J/error_upper를 변경하지 않는다. */
CMFExactStatus cmf_exact_emissivity_solve(
    int ns,int nb,double dlog,const double *nu,const double *ri,const double *ro,
    double time,double temperature,double boundary_scale,const double *chi,
    const double *scatter,const double *eta,double *J,double *error_upper,
    size_t refinements,int iterations,double tolerance,CMFExactMode mode,CMFExactReport *report);
CMFExactStatus cmf_exact_formal_moments_emissivity(
    int ns,int nb,double dlog,const double *nu,const double *ri,const double *ro,
    double time,double temperature,double boundary_scale,const double *chi,
    const double *scatter,const double *eta,const double *J,CMFFormalMoments *moments);

#ifdef __cplusplus
}
#endif

#endif
