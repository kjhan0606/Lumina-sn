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
} A209Counters;

int a209_publication_init(CpuEmissivityPublication *p, size_t n_shells,
                          size_t n_bins);
void a209_publication_free(CpuEmissivityPublication *p);
double a209_publication_max_closure(const CpuEmissivityPublication *p,
                                    size_t *worst_cell);
int a209_publication_commit(CpuEmissivityPublication *published,
                            CpuEmissivityPublication *candidate);
int a209_build_reemit_cdf(CpuEmissivityPublication *candidate,
                          unsigned channel_mask);
int a209_sample_reemit_frequency(const CpuEmissivityPublication *p,
                                 size_t shell, uint64_t required_generation,
                                 double uniform_draw, double *frequency);
EmissivityStatus a209_transition_block(const double *weight,
                                       const double *emitted_energy,
                                       size_t n_channels, double input_energy,
                                       double *probability,
                                       double *normalized_energy_error);
const char *a209_status_name(EmissivityStatus status);
A209Counters *a209_counters(void);
void a209_counters_reset(void);
void a209_counters_print(FILE *stream);

#endif
