#ifndef LUMINA_EMISSIVITY_PUBLICATION_H
#define LUMINA_EMISSIVITY_PUBLICATION_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum {
    EMISS_OK = 1, EMISS_EXACT_ZERO,
    EMISS_STALE_RF, EMISS_STALE_LINE, EMISS_STALE_POP, EMISS_STALE_OPACITY,
    EMISS_UNSAMPLED, EMISS_OOG, EMISS_MISS,
    EMISS_PROFILE_MISMATCH, EMISS_QUERY_HASH_MISMATCH,
    EMISS_INVALID_TE, EMISS_INVALID_NE, EMISS_ATOMIC_MISSING,
    EMISS_SOURCE_UNDEFINED, EMISS_COMPONENT_INCOMPLETE,
    EMISS_NEGATIVE_EVENT_UNSUPPORTED, EMISS_TRANSITION_EMPTY,
    EMISS_TRANSITION_NONFINITE, EMISS_TRANSITION_NOT_NORMALIZED,
    EMISS_ENERGY_NOT_CLOSED, EMISS_CDF_EMPTY, EMISS_CDF_STALE,
    EMISS_CDF_INVALID, EMISS_FORBIDDEN_PLANCK, EMISS_PARTIAL_PUBLISH,
    EMISS_NONFINITE
} EmissivityStatus;

typedef struct {
    uint64_t required_emissivity_generation, committed_emissivity_generation;
    uint64_t radfield_generation, line_view_generation, population_generation;
    uint64_t opacity_generation, te_generation;
    char atomic_model_sha256[65], grid_manifest_sha256[65];
    char source_manifest_sha256[65], cdf_manifest_sha256[65];
    size_t n_shells, n_bins;
    double *nu_edge;
    double *eta_bb, *eta_bf, *eta_ff, *eta_true_total;
    double *eta_scattering_source, *eta_total_for_declared_semantics;
    double *eta_reemit, *reemit_cdf;
    EmissivityStatus *cell_status, *component_status;
    uint64_t cdf_generation;
    unsigned channel_mask;
    EmissivityStatus redistribution_status;
} CpuEmissivityPublication;

typedef struct {
    uint64_t generation_required, generation_committed;
    uint64_t shells_attempted, shells_published, cells_attempted, cells_published;
    uint64_t bb_terms, bf_terms, ff_terms, exact_zero_terms;
    uint64_t transition_blocks_attempted, transition_blocks_published, transition_channels;
    uint64_t transition_empty, transition_nonfinite, transition_norm_fail, energy_closure_fail;
    uint64_t cdf_attempted, cdf_committed, cdf_empty, cdf_stale, cdf_invalid;
    uint64_t sampler_calls, sampler_draws, sampler_generation_mismatch;
    uint64_t blocked_stale_rf, blocked_stale_line, blocked_stale_pop, blocked_stale_opacity;
    uint64_t blocked_unsampled, blocked_oog, blocked_miss, blocked_source, blocked_atomic;
    uint64_t fallback_attempts, planck_attempts, raw_view_attempts, clamp_attempts;
    uint64_t floor_attempts, last_channel_attempts, partial_publish_attempts;
    uint64_t nonfinite_failures;
    uint64_t identity_seal_failures;
} A209Counters;

/* Small immutable token view bracketing A2-09's read of the raw Sobolev slab.
 * Numeric tau stays in OpacityState to avoid a second ~125M-cell copy, but a
 * writer that advances any bound generation during consumption invalidates
 * the private emissivity candidate. */
typedef struct {
    uint64_t tau_required_generation;
    uint64_t tau_computed_generation;
    uint64_t opacity_tau_generation;
    uint64_t population_generation;
    uint64_t opacity_population_generation;
    uint64_t te_generation;
    uint64_t opacity_te_generation;
    uint64_t nlte_population_generation;
    double opacity_epoch;
    double requested_epoch;
} A209LineGenerationView;

int a209_publication_init(CpuEmissivityPublication *p, size_t n_shells,
                          size_t n_bins);
void a209_publication_free(CpuEmissivityPublication *p);
double a209_publication_max_closure(const CpuEmissivityPublication *p,
                                    size_t *worst_cell);
int a209_publication_commit(CpuEmissivityPublication *published,
                            CpuEmissivityPublication *candidate);
int a209_publication_commit_counted(CpuEmissivityPublication *published,
                                    CpuEmissivityPublication *candidate,
                                    A209Counters *counter_sink);
int a209_grid_manifest_sha256(const double *nu_edge, size_t n_bins,
                              char out[65]);
int a209_source_manifest_sha256(unsigned channel_mask, char out[65]);
int a209_build_reemit_cdf(CpuEmissivityPublication *candidate,
                          unsigned channel_mask);
int a209_build_reemit_cdf_counted(CpuEmissivityPublication *candidate,
                                  unsigned channel_mask,
                                  A209Counters *counter_sink);
int a209_sample_reemit_frequency(const CpuEmissivityPublication *p,
                                 size_t shell, uint64_t required_generation,
                                 double uniform_draw, double *frequency);
EmissivityStatus a209_transition_block(const double *weight,
                                       const double *emitted_energy,
                                       size_t n_channels, double input_energy,
                                       double *probability,
                                       double *normalized_energy_error);
/* Direct Sobolev line emissivity.  This is deliberately population-native:
 * it never constructs or divides by a line source function, so the
 * n_l-(g_l/g_u)n_u -> 0 limit remains finite when n_u > 0. */
EmissivityStatus a209_sobolev_line_eta(double n_upper, double A_ul,
                                       double nu, double tau,
                                       double delta_nu, double *beta_escape,
                                       double *eta_nu);
EmissivityStatus a209_line_generation_bracket(
    const A209LineGenerationView *begin,
    const A209LineGenerationView *end);
const char *a209_status_name(EmissivityStatus status);
A209Counters *a209_counters(void);
void a209_counters_reset(void);
void a209_counters_print(FILE *stream);

#endif
