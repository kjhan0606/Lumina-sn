#ifndef LUMINA_OPACITY_PUBLICATION_H
#define LUMINA_OPACITY_PUBLICATION_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    A208_VALID = 1,
    A208_EXACT_ZERO,
    A208_UNSAMPLED,
    A208_OUT_OF_GRID,
    A208_MISS,
    A208_STALE_GENERATION,
    A208_QHASH_MISMATCH,
    A208_PROFILE_MISMATCH,
    A208_INVALID_POPULATION,
    A208_INVALID_PARTITION,
    A208_INVALID_TE,
    A208_INVALID_NE,
    A208_NONFINITE,
    A208_SOURCE_CANCELLATION_SINGULAR,
    A208_EVENT_MEASURE_UNAVAILABLE,
    A208_BLOCKED_NEGATIVE_OPACITY_SEMANTICS,
    A208_FORBIDDEN_FALLBACK
} A208Validity;

typedef struct { double value; A208Validity validity; uint64_t generation; } A208ValueView;
typedef struct { double value; A208Validity validity; } A208SignedBfNet;
typedef struct { double value; A208Validity validity; } A208NonnegativeEventMeasure;
typedef struct { double value; A208Validity validity; } A208TauInteractionMeasure;

typedef enum {
    A208_SIGNED_EQUATION = 1,
    A208_SEPARATE_NONNEG_EVENT_MEASURE,
    A208_BLOCK_UNSUPPORTED
} A208ConsumerCapability;

typedef struct {
    uint64_t generation_required, generation_committed;
    double epoch;
    uint64_t shell_geometry_hash, frequency_edge_hash, atomic_model_hash;
    uint64_t radiation_generation, line_jbar_generation;
    uint64_t population_generation, partition_generation, within_sl_generation;
    uint64_t te_generation, ne_generation, tau_generation;
    size_t n_shells, n_bins, n_lines, n_routes;
    double *frequency_edges;
    double *chi_es, *chi_bb, *chi_bf, *chi_ff, *chi_total;
    A208Validity *chi_validity; /* [4*n_shells*n_bins], ES/BB/BF/FF */
    double *tau_sobolev, *line_source_S;
    A208Validity *tau_validity, *line_source_validity;
    double *bf_net_route, *bf_event_measure;
    A208Validity *bf_route_validity;
} CpuOpacityPublication;

typedef struct {
    uint64_t generation_required, generation_committed;
    uint64_t shells_attempted, shells_published, cells_attempted, cells_published;
    uint64_t es_terms, bb_terms, bf_terms, ff_terms;
    uint64_t exact_zero_es, exact_zero_bb, exact_zero_bf, exact_zero_ff;
    uint64_t negative_tau_line_shells, negative_bb_line_shells;
    uint64_t negative_bf_route_shell_bins, negative_bf_shell_bins;
    uint64_t negative_total_shell_bins;
    uint64_t blocked_negative_transport, blocked_negative_formal;
    uint64_t blocked_negative_heating, blocked_negative_transition;
    uint64_t blocked_stale, blocked_unsampled, blocked_oog, blocked_miss;
    uint64_t blocked_profile, blocked_qhash, blocked_population, blocked_te, blocked_ne;
    uint64_t source_valid, source_exact_zero, source_negative;
    uint64_t source_cancellation_singular, event_measure_unavailable;
    uint64_t closure_failures, nonfinite_failures;
    uint64_t fallback_attempts, abs_attempts, zero_clamp_attempts, floor_attempts;
    uint64_t raw_view_attempts, partial_publish_attempts;
    uint64_t replay_line_blocks_attempted, replay_line_blocks_committed;
} A208Counters;

A208ValueView a208_signed_sobolev(double coefficient, double f_lu,
                                  double lambda_cm, double time_explosion,
                                  double n_lower, double n_upper,
                                  double g_lower, double g_upper,
                                  uint64_t generation);
A208ValueView a208_line_source(double prefactor, double n_lower,
                               double n_upper, double g_lower, double g_upper,
                               uint64_t generation);
int a208_bf_split(double gross, double stimulated_ratio, double exponent,
                  A208SignedBfNet *net,
                  A208NonnegativeEventMeasure *event_measure);
A208TauInteractionMeasure a208_tau_interaction_measure(A208ValueView tau);

int a208_publication_init(CpuOpacityPublication *p, size_t n_shells,
                          size_t n_bins, size_t n_lines, size_t n_routes);
void a208_publication_free(CpuOpacityPublication *p);
int a208_publication_commit(CpuOpacityPublication *public_p,
                            CpuOpacityPublication *candidate);
double a208_publication_max_closure(const CpuOpacityPublication *p,
                                    size_t *worst_cell);
int a208_capability_check(A208ConsumerCapability capability,
                          const A208ValueView *values, size_t count,
                          const char *consumer, uint64_t *blocked_counter,
                          size_t *first_negative);
const char *a208_validity_name(A208Validity validity);
A208Counters *a208_counters(void);
void a208_counters_reset(void);
void a208_report_counters(void);

#endif
