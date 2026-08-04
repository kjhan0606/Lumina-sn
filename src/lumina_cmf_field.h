#ifndef LUMINA_CMF_FIELD_H
#define LUMINA_CMF_FIELD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LCMF_OK = 0,
    LCMF_EINVAL = -1,
    LCMF_ENOMEM = -2,
    LCMF_EGRID = -3,
    LCMF_ENEGATIVE = -4,
    LCMF_ENONFINITE = -5,
    LCMF_EUNSUPPORTED = -6,
    LCMF_ENOCONV = -7,
    LCMF_EIO = -8,
    LCMF_ESCHEMA = -9,
    LCMF_ECHECKSUM = -10,
    LCMF_EHOMOLOGY = -11,
    LCMF_ESIGNUNCERTAIN = -12
} LCMFStatus;

typedef enum {
    LCMF_BC_DIFFUSION = 1,
    LCMF_BC_IRRADIATION = 2
} LCMFInnerBC;

typedef enum {
    LCMF_SCAT_NONE = 0,
    LCMF_SCAT_COHERENT = 1,
    LCMF_SCAT_REDISTRIBUTION = 2
} LCMFScatterMode;

typedef double (*LCMFBoundaryFn)(void *ctx, double p_cm, double mu,
                                 double nu_hz);
typedef int (*LCMFRedistributionFn)(void *ctx, size_t radial_index,
                                    const double *nu_hz, size_t nnu,
                                    const double *J_nu,
                                    double *eta_redistributed);

typedef struct {
    int code;
    size_t radial_index;
    size_t frequency_index;
    size_t ray_index;
    size_t segment_index;
    size_t substep_index;
    size_t endpoint_index;
    double value;
    double term_previous;
    double term_previous2;
    double decay_ratio;
    double theoretical_limit;
    double interval_lower;
    double interval_upper;
    char message[160];
} LCMFError;

typedef struct {
    double value;
    double lower;
    double upper;
} LCMFInterval;

typedef struct {
    size_t nr;
    size_t nnu;
    const double *r_edge;
    const double *nu;
    const double *chi_total;
    const double *eta_fixed;
    const double *chi_coherent;
    double t_exp_s;
    LCMFInnerBC inner_bc;
    LCMFScatterMode scatter_mode;
    const double *B_inner;
    const double *dB_dtau_inner;
    LCMFBoundaryFn inner_irradiation;
    LCMFBoundaryFn outer_irradiation;
    LCMFRedistributionFn redistribution;
    void *boundary_ctx;
    void *redistribution_ctx;
} LCMFInput;

typedef struct {
    size_t n_mu;
    size_t n_r_eval;
    const double *r_eval;
    size_t max_source_iter;
    double source_rtol;
    int compute_hk;
    int store_intensity;
    int frequency_advection;
    int radial_characteristic;
} LCMFOptions;

typedef struct {
    double *J;
    double *H;
    double *K;
    double *I_minus;
    double *I_plus;
    double transport_resid_linf;
    double source_resid_linf;
    uint64_t clamp_count;
    uint64_t bdf_eta_negative_count;
    uint64_t bdf_eta_negative_plane_count;
    uint64_t solution_negative_excess_count;
    uint64_t solution_subtruncation_count;
    uint64_t solution_sign_indeterminate_subtruncation_count;
    uint64_t solution_roundoff_enclosure_restart_count;
    uint64_t sign_uncertain_count;
    uint64_t nonfinite_count;
    double bdf_eta_min;
    double solution_min;
    double solution_subtruncation_min;
    double solution_subtruncation_first_bound;
    double solution_subtruncation_first_h;
    double solution_subtruncation_first_scale;
    size_t source_iterations;
    size_t nr;
    size_t nnu;
    size_t n_mu;
    LCMFError bdf_eta_first;
    LCMFError solution_subtruncation_first;
    LCMFError solution_subtruncation_min_location;
    LCMFError error;
} LCMFResult;

/* Empirical O(h^2) coefficient: max(profile L1 * ns^2) over the six-grid
 * KA3 convergence battery, attained at ns=64 (round 7C replay). */
#define LCMF_TRUNCATION_ERROR_COEFFICIENT 27.80641753160013

typedef struct {
    size_t n_mu;
    double *mu;
    double *weight;
    double *p;
} LCMFRayCache;

typedef struct {
    uint64_t nr;
    uint64_t nnu;
    uint64_t iteration;
    uint64_t field_generation;
    uint32_t flags;
    double t_exp_s;
    double *r_edge;
    double *nu;
    double *dnu;
    double *chi_total;
    double *chi_coherent;
    double *eta_fixed;
    double *eta_coherent;
    double *eta_total;
    double *J_producer;
} LCMFFrozenField;

enum {
    LCMF_FROZEN_POST_DAMP = 1u << 0,
    LCMF_FROZEN_COHERENT = 1u << 1,
    LCMF_FROZEN_FREQUENCY_DESCENDING = 1u << 2
};

int lumina_cmf_validate_input(const LCMFInput *input,
                              const LCMFOptions *options,
                              LCMFError *error);
int lumina_cmf_ray_cache_build(const double *r_edge, size_t nr,
                               size_t radial_index, size_t n_mu,
                               LCMFRayCache *cache, LCMFError *error);
int lumina_cmf_ray_cache_build_at_radius(double target_r, size_t n_mu,
                                         LCMFRayCache *cache,
                                         LCMFError *error);
void lumina_cmf_ray_cache_free(LCMFRayCache *cache);

int lumina_cmf_sc_linear(double intensity_upstream, double source_upstream,
                         double source_downstream, double delta_tau,
                         double *intensity_downstream);
int lumina_cmf_sc_linear_signed(const LCMFInterval *intensity_upstream,
                                const LCMFInterval *source_upstream,
                                const LCMFInterval *source_downstream,
                                double delta_tau,
                                LCMFInterval *intensity_downstream);
int lumina_cmf_sc_quadratic_signed(const LCMFInterval *intensity_upstream,
                                   const LCMFInterval *coefficient0,
                                   const LCMFInterval *coefficient1,
                                   const LCMFInterval *coefficient2,
                                   double delta_s, double delta_tau,
                                   LCMFInterval *intensity_downstream);

#ifdef LCMF_TEST_HOOKS
int lumina_cmf_solution_guard_probe(const LCMFInterval *solution,
                                    double h, double local_scale,
                                    double global_scale,
                                    LCMFResult *result);
#endif

int lumina_cmf_field_solve(const LCMFInput *input,
                           const LCMFOptions *options,
                           LCMFResult *result);
int lumina_cmf_field_residual(const LCMFInput *input,
                              const LCMFResult *result, double *linf);
void lumina_cmf_result_free(LCMFResult *result);

int lumina_cmf_frozen_load(const char *binary_path, const char *manifest_path,
                           LCMFFrozenField *field, LCMFError *error);
void lumina_cmf_frozen_free(LCMFFrozenField *field);

const char *lumina_cmf_status_string(int status);

#ifdef __cplusplus
}
#endif
#endif
