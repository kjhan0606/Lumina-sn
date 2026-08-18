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

    /* ★R0(2026-08-07): **분배함수 전용** 최상단 이온 준위 catalog.
     * 로더가 전리에너지 n 개 -> population n+1 개를 만들어 원소마다 최상단 population 의
     * 속박준위가 0 개다(실측 15/74).  단일 g 대입은 최대 80배 틀린다
     * (V II: g_first=1 vs Z(10kK)=80.9).
     *
     * 이 catalog 를 **공용 준위 배열에 넣지 않는** 이유(Codex 설계 판정 (c)):
     * 넣으면 NLTE 미지수 등록 · ma_radrecomb_target 의 전역 준위 인덱스 ·
     * BF 단면 [n_levels x n_freq] 계약 · macro-atom block reference ·
     * 충돌강도 sidecar 준위 수 검사가 전부 **없는 자료**를 요구하게 된다.
     * 전용 catalog 는 population_partition_build 하나만 읽는다. 그 값은 partition을
     * 통해 모든 material consumer에 영향을 주므로 atomic_model_sha256에도 포함된다.
     *
     * 이것은 runtime_membership 과 별개인 **thermodynamic membership** 이다. */
    size_t topion_n;               /* catalog 항목 수 (0 이면 없음) */
    const int *topion_ion_index;   /* topion_n — **ion-pop 인덱스**(level_offset 과 같은 키) */
    const double *topion_E_cm;     /* topion_n, cm^-1 */
    const double *topion_g;        /* topion_n */
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

/* Dense statistical-equilibrium solve diagnostics.  The solver operates on a
 * column-major matrix, preserves matrix/RHS bytes, and applies only algebraic
 * row/column equilibration plus mixed-precision iterative refinement.  It does
 * not clamp, floor, pin, or otherwise alter the physical solution. */
#define POP_DENSE_BACKWARD_ERROR_LIMIT 1.0e-12
typedef struct {
    size_t rank;
    int equilibration_iterations;
    int refinement_iterations;
    double pivot_growth;
    double initial_backward_error;
    double final_backward_error;
} PopulationLinearSolveDiagnostic;

/* Cancellation-free stationary solve for an irreducible continuous-time
 * generator.  The input is the pre-constraint, column-major rate matrix:
 * off-diagonal A[dest,source] entries are transition rates and each diagonal
 * is the negative source outflow.  GTH uses only the nonnegative
 * off-diagonals and rebuilds every outflow in long double; it does not clamp,
 * floor, exponentiate, or otherwise force a general linear solution positive.
 * generator_recognized=0 means the input is not eligible for this solver and
 * the caller may use its general linear path.  Once recognized, any non-OK
 * result is a genuine generator solve failure and must fail closed. */
#define POP_GENERATOR_COLUMN_ERROR_LIMIT 1.0e-12
#define POP_GENERATOR_RESIDUAL_LIMIT 1.0e-12
typedef struct {
    int generator_recognized;
    double input_column_relative_error;
    double exact_generator_componentwise_residual;
    double minimum_population;
    double maximum_population;
} PopulationGeneratorSolveDiagnostic;

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

/* Build the full-level fractions within each super-level from the exact same
 * generation-bound T_e/atomic view used for the partition functions.  Output
 * and stamp are published together only after every shell succeeds. */
PopulationStatus population_within_superlevel_build(
    const PopulationDerivedStamp *partition_stamp,
    const PopulationAtomicView *atomic,
    const double *te,
    size_t n_shells,
    uint64_t required_population_generation,
    uint64_t te_generation,
    size_t n_full_levels,
    int super_mode,
    size_t n_superlevels,
    const int *nlte_to_global_level,
    const int *full_to_superlevel,
    const int *super_anchor_global_level,
    double *fractions,
    PopulationDerivedStamp *fraction_stamp);

PopulationStatus population_lte_level_fraction(
    const PopulationAtomicView *atomic, size_t ion, size_t level,
    double te, double partition, double *fraction);

/* One line-level number-density contract shared by the bulk Sobolev writer
 * and A2-09.  LTE is reconstructed from the committed ion density and the
 * T_e partition; NLTE accepts only the already selected committed level
 * density.  Keeping both branches here makes their zero/nonfinite semantics
 * identical and directly testable. */
typedef enum {
    POP_LINE_VIEW_LTE_TE = 1,
    POP_LINE_VIEW_NLTE_COMMITTED = 2
} PopulationLineView;

PopulationStatus population_line_level_number_density(
    PopulationLineView view, const PopulationAtomicView *atomic,
    size_t ion, size_t level, double te, double partition,
    double ion_number_density, double nlte_level_number_density,
    double *number_density);

PopulationStatus population_rate_views_check(
    PopulationStatus bf_status, uint64_t bf_generation,
    PopulationStatus bb_status, uint64_t bb_generation,
    uint64_t required_rate_generation);
PopulationStatus population_dense_rank_check(const double *matrix, size_t n,
                                              double relative_tolerance);
PopulationStatus population_dense_solve_equilibrated(
    const double *matrix_column_major,
    const double *rhs,
    size_t n,
    double *solution,
    PopulationLinearSolveDiagnostic *diagnostic);
PopulationStatus population_generator_stationary_gth(
    const double *generator_column_major,
    size_t n,
    double total_population,
    double *solution,
    PopulationGeneratorSolveDiagnostic *diagnostic);
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
