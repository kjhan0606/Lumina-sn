#ifndef LUMINA_CMF_EXACT_SLIDING_H
#define LUMINA_CMF_EXACT_SLIDING_H

#include <stddef.h>
#include <stdint.h>

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
    /* Conservative infinity-norm qualification.  The formal lambda operator
     * has row sum <=1, so max(chi_es/chi_tot) is an upper bound on the fixed-
     * point iteration norm.  A finite error bound is published only below 1. */
    double max_scattering_ratio;
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
    /* 실제 할당한 방향별 광선 표본 슬롯. segment_slots는 전체 기하다. */
    size_t ray_buffer_slots;
} CMFExactReport;

typedef enum {
    CMF_EXACT_INNER_BLACKBODY = 1,
    CMF_EXACT_INNER_DIFFUSION = 2,
    /* 실제 CMF_FORMAL_REL_V4: I+=B_nu(T_inner)+mu_cmf*(-dB/dr)/chi.
     * 기존 source+scattering*J 경계와 별개이며 외부 J 주입은 없다. */
    CMF_EXACT_INNER_PLANCK_DIFFUSION = 3
} CMFExactInnerKind;

/* 실제 내부 반경에서 정의하는 comoving 경계. DIFFUSION은
 * I+(mu,nu)=source_fixed + scattering_ratio*J_inner + mu*gradient_over_chi.
 * J_inner는 첫 셀 중심 J로 이산화한다. 외부 CMFGEN J를 주입하지 않는다.
 * source_fixed와 scattering_ratio는 실제 경계 물질의 eta_true/chi,
 * chi_es/chi다. BB로 대체하거나 첫 셀 온도로 경계 온도를 바꾸지 않는다. */
typedef struct {
    CMFExactInnerKind kind;
    double temperature;
    double blackbody_scale;
    const double *source_fixed;       /* [n_bins], DIFFUSION */
    const double *scattering_ratio;   /* [n_bins], DIFFUSION */
    const double *gradient_over_chi;  /* [n_bins], positive -dB/dr / chi */
} CMFExactInnerBoundary;

/* CMFGEN cmfgen_sub.f의 광도 지정 확산식:
 * G=-dT/dr = [3 L/(16 pi^2 r^2)] / Int[(dBnu/dT)/chi_nu]dnu.
 * 결과는 G*(dBnu/dT)/chi_nu다. 전체 선+연속 소광과 적분 격자를
 * 호출자가 공급한다. 이 함수 자체는 주파수/공간 이산화 오차를 증명하지 않는다. */
CMFExactStatus cmf_exact_diffusion_gradient(
    int n_bins, const double *nu, const double *dnu,
    double temperature, double luminosity, double radius,
    const double *chi, double *gradient_over_chi);

/* 새 물질 셀 경로. 모든 실제 r_inner/r_outer 구간을 통과하고 J는 산술
 * 반경 중심의 각도 평균이다(체적 평균 아님). 진공 외부 경계는 실제
 * r_outer[last], 내부 경계는 실제 r_inner[0]에 놓는다. 기존 nodal-like
 * characteristic API의 이산화와 별도로 보존한다. */
CMFExactStatus cmf_exact_cell_solve(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    const CMFExactInnerBoundary *boundary,
    const double *chi_tot, const double *chi_es, const double *S_fixed,
    double *J, double *componentwise_error_upper,
    size_t envelope_refinement_iterations,
    int iteration_cap, double tolerance, CMFExactMode mode,
    CMFExactReport *report);

/* signed χ와 비음수 η_true를 직접 받는 실제 셀 경로. η/χ를 만들지
 * 않으므로 χ=0 극한과 χ<0 증폭을 보존한다. 광선의 유한 증폭과 실제
 * 산란 K(1)의 상한을 사용하며 기존 max(χ_es/χ) 논리를 적용하지 않는다.
 * report.max_scattering_ratio는 이 경로에서 실제 K(1)의 최대 상한이다.
 * componentwise 인증은 같은 이산 연산의 반올림/반복 오차 인증이지,
 * 상대론·공간·주파수 이산화 오차 또는 CMFGEN 동등성 증명이 아니다. */
CMFExactStatus cmf_exact_cell_solve_raw_sources(
    int n_shells,int n_bins,double dlognu,const double *nu,
    const double *r_inner,const double *r_outer,double time_explosion,
    const CMFExactInnerBoundary *boundary,
    const double *chi_tot,const double *chi_es,const double *eta_true,
    double *J,double *componentwise_error_upper,
    size_t envelope_refinement_iterations,
    int iteration_cap,double tolerance,CMFExactMode mode,CMFExactReport *report);

/* 정상 homologous full-SR의 lab p,z 재매개화. 각 반셀의 정확한 ds와
 * Doppler 이동, 방향별 comoving mu 적분을 같은 formal/K/오차 연산에 쓴다.
 * 반셀 안의 물질과 b_eff는 P0 이산화다. blue 끝은 dI/dlognu=0 연장이고
 * 3b 소광은 유지한다. native implicit-upwind와 이산적으로 같다는 뜻이나
 * 시간미분/observer/전역 물질 수렴의 구현을 뜻하지 않는다. */
CMFExactStatus cmf_exact_cell_solve_relativistic_raw_sources(
    int n_shells,int n_bins,double dlognu,const double *nu,
    const double *r_inner,const double *r_outer,double time_explosion,
    const CMFExactInnerBoundary *boundary,
    const double *chi_tot,const double *chi_es,const double *eta_true,
    double *J,double *componentwise_error_upper,
    size_t envelope_refinement_iterations,
    int iteration_cap,double tolerance,CMFExactMode mode,CMFExactReport *report);

/* Exact homologous drifting-characteristic formal solution.
 *
 * Frequency bins are ascending in nu (red -> blue), with constant dlognu.
 * The production mode is CMF_EXACT_MODE_SLIDING, O(1) work per output bin.
 * CMF_EXACT_MODE_DIRECT_REFERENCE is an independent O(beta) oracle intended
 * only for small-grid selftests.  J is both the ALI initial state and output.
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

#ifdef __cplusplus
}
#endif

#endif
