#ifndef LUMINA_POPULATION_CONTRACT_H
#define LUMINA_POPULATION_CONTRACT_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum {
    POP_OK = 0,
    POP_EXACT_ZERO,
    POP_INVALID_TE,
    POP_INVALID_PARTITION,
    POP_STALE_DERIVED_TEMPERATURE,
    POP_BF_STALE,
    POP_BF_UNSAMPLED,
    POP_BF_OOG,
    POP_BF_MISS,
    POP_BB_STALE,
    POP_BB_UNSAMPLED,
    POP_BB_OOG,
    POP_BB_MISS,
    POP_PROFILE_MISMATCH,
    POP_QUERY_HASH_MISMATCH,
    POP_ATOMIC_MISSING,
    POP_RANK_INCOMPLETE,
    POP_NE_NOT_CONVERGED,
    POP_SOLVE_FAILED,
    POP_NONFINITE,
    POP_FORBIDDEN_FALLBACK
} PopulationStatus;

typedef struct {
    uint64_t required_population_generation;
    uint64_t computed_population_generation;
    uint64_t te_generation;
    char te_manifest_sha256[65];
    char atomic_model_sha256[65];
    size_t n_shells;
    size_t n_items;
    PopulationStatus status;
} PopulationDerivedStamp;

typedef struct {
    uint64_t pop_generation_required, pop_generation_committed;
    uint64_t pop_shells_attempted, pop_shells_published;
    uint64_t pop_bf_terms, pop_bb_terms, pop_exact_zero_terms;
    uint64_t pop_blocked_stale, pop_blocked_unsampled;
    uint64_t pop_blocked_oog, pop_blocked_miss;
    uint64_t pop_blocked_profile, pop_blocked_qhash;
    uint64_t pop_blocked_te, pop_blocked_partition;
    uint64_t pop_rank_incomplete, pop_ne_not_converged;
    uint64_t pop_solve_failed, pop_nonfinite;
    uint64_t pop_generation_mismatch, pop_fallback_attempts;
    uint64_t pop_partial_publish_attempts;
} PopulationCounters;

typedef struct {
    size_t n_ions;
    size_t n_levels;
    const int *level_offset;       /* n_ions + 1 */
    const double *energy_eV;       /* n_levels */
    const int *g;                  /* n_levels */
    const int *runtime_membership; /* NULL or n_levels; negative means absent */
    const int *level_Z;            /* optional hash identity */
    const int *level_ion;          /* optional hash identity */
} PopulationAtomicView;

typedef struct {
    double *public_ion;
    double *public_level;
    double *public_ne;
    double *public_partition;
    double *work_ion;
    double *work_level;
    double *work_ne;
    double *work_partition;
    size_t n_ion_values, n_level_values, n_ne_values, n_partition_values;
    uint64_t required_generation;
    uint64_t *committed_generation;
    PopulationStatus status;
} PopulationTransaction;

const char *population_status_name(PopulationStatus status);

PopulationStatus population_te_manifest_sha256(const double *te,
                                                size_t n_shells,
                                                char out[65]);
PopulationStatus population_atomic_model_sha256(
    const PopulationAtomicView *atomic, char out[65]);

PopulationStatus population_partition_ion(const PopulationAtomicView *atomic,
                                          size_t ion, double te,
                                          double *out);
PopulationStatus population_partition_build(
    const PopulationAtomicView *atomic, const double *te, size_t n_shells,
    uint64_t required_population_generation, uint64_t te_generation,
    double *public_partition, PopulationDerivedStamp *stamp);
PopulationStatus population_partition_view_check(
    const PopulationDerivedStamp *stamp, const PopulationAtomicView *atomic,
    const double *te, size_t n_shells,
    uint64_t required_population_generation, uint64_t te_generation);

PopulationStatus population_lte_level_fraction(
    const PopulationAtomicView *atomic, size_t ion, size_t level,
    double te, double partition, double *fraction);

PopulationStatus population_rate_views_check(
    PopulationStatus bf_status, uint64_t bf_generation,
    PopulationStatus bb_status, uint64_t bb_generation,
    uint64_t required_rate_generation);
PopulationStatus population_dense_rank_check(const double *matrix, size_t n,
                                              double relative_tolerance);
PopulationStatus population_superlevel_aggregate(
    const double *level_population, const int *membership, size_t n_levels,
    size_t n_superlevels, double *super_population);

int population_transaction_begin(PopulationTransaction *tx,
                                 double *ion, size_t n_ion,
                                 double *level, size_t n_level,
                                 double *ne, size_t n_ne,
                                 double *partition, size_t n_partition,
                                 uint64_t required_generation,
                                 uint64_t *committed_generation);
void population_transaction_abort(PopulationTransaction *tx,
                                  PopulationStatus status);
PopulationStatus population_transaction_commit(PopulationTransaction *tx);

void population_counter_note(PopulationCounters *c, PopulationStatus status);
void population_counters_print(FILE *stream, const PopulationCounters *c);

#endif
