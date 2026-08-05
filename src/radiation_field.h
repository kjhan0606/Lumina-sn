#ifndef LUMINA_RADIATION_FIELD_H
#define LUMINA_RADIATION_FIELD_H

#include <stddef.h>
#include <stdint.h>

/* A2-02 amended-union authority.  The edges are log-spaced over the exact
 * closed union recorded in validation/a2_02c/A2_02C_FREQUENCY_UNION.json. */
#define LUMINA_RADFIELD_N_BINS 4000
#define LUMINA_RADFIELD_NU_MIN_HZ 1.4402928950097124e12
#define LUMINA_RADFIELD_NU_MAX_HZ 4.032418413741097e16
#define LUMINA_RADFIELD_UNION_SHA256 \
    "1443c069eb710acb31c6470442637e43ad11eb57191fbc3a265363cd4d61321c"
#define LUMINA_RADFIELD_EDGE_SHA256 \
    "ec3f94d923b42e036afd3fde71fc63a2477d4c95ca252583b40ed730a0b48f76"

typedef struct {
    size_t count;
    double *values;
    const char *coordinate_units;
} RadiationFieldAxis;

typedef struct {
    size_t n_shells;
    size_t n_bins;
    double *values;
} RadiationFieldGrid;

typedef enum {
    RADIATION_FIELD_UNITS_ERG_S_NEG1_CM_NEG2_HZ_NEG1_SR_NEG1 = 1
} RadiationFieldUnits;

typedef enum {
    RADIATION_FIELD_FRAME_SHELL_COMOVING = 1,
    RADIATION_FIELD_FRAME_OBSERVER = 2
} RadiationFieldFrame;

/* K-FRESH-compatible discipline: a producer first advances required while the
 * public shadow stays stale, then sets computed=required only after validation. */
typedef struct {
    uint64_t required_generation;
    uint64_t computed_generation;
} RadiationFieldGeneration;

typedef enum {
    RADIATION_FIELD_PROVENANCE_NONE = 0,
    RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH = 1,
    RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY = 2,
    RADIATION_FIELD_PROVENANCE_DILUTE_PLANCK_LEGACY_APPROXIMATION = 3
} RadiationFieldProvenanceKind;

typedef struct {
    RadiationFieldProvenanceKind kind;
    const char *producer;
    const char *frequency_union_sha256;
    const char *frequency_edge_sha256;
    const char *raw_ledger_sha256;
    uint64_t contribution_count;
    uint64_t out_of_grid_contribution_count;
} RadiationFieldProvenance;

typedef enum {
    RADIATION_FIELD_VALID = 1,
    RADIATION_FIELD_EXACT_ZERO = 2,
    RADIATION_FIELD_UNSAMPLED = 3,
    RADIATION_FIELD_OUT_OF_GRID = 4,
    RADIATION_FIELD_STALE = 5
} RadiationFieldValidityState;

typedef struct {
    size_t n_shells;
    size_t n_bins;
    RadiationFieldValidityState *values;
} RadiationFieldValidity;

typedef enum {
    RADIATION_FIELD_ESTIMATOR_COUNT = 1,
    RADIATION_FIELD_ESTIMATOR_VARIANCE = 2,
    RADIATION_FIELD_DETERMINISTIC = 3
} RadiationFieldEstimatorStatisticKind;

typedef struct {
    RadiationFieldEstimatorStatisticKind kind;
    size_t n_shells;
    size_t n_bins;
    uint64_t *count;
    double *variance;
} RadiationFieldEstimatorStatistics;

/* Canonical A2 radiation-field schema.  Keep these ten member names in exact
 * one-to-one correspondence with ORDER_L0_JNU_OWNER_BY_CODEX.md section 2.1. */
typedef struct {
    RadiationFieldAxis shell_boundaries;
    RadiationFieldAxis frequency_bin_edges;
    RadiationFieldGrid J_nu;
    RadiationFieldUnits units;
    RadiationFieldFrame frame;
    double epoch;
    RadiationFieldGeneration generation;
    RadiationFieldProvenance provenance;
    RadiationFieldValidity validity;
    RadiationFieldEstimatorStatistics estimator_count_or_variance;
} RadiationField;

/* A2-02 amended selective-line derived-cache holder.  A2-03 defines ownership
 * and metadata only; A2-06 will allocate/produce it and add its checked view. */
typedef enum {
    LINE_JBAR_VALID = 1,
    LINE_JBAR_EXACT_ZERO = 2,
    LINE_JBAR_UNSAMPLED = 3,
    LINE_JBAR_OUT_OF_BB_DOMAIN = 4,
    LINE_JBAR_STALE = 5
} LineJbarValidityState;

typedef struct {
    RadiationFieldGeneration generation;
    size_t entry_count;
    uint64_t *shell_id;
    uint64_t *line_id;
    uint64_t *profile_id;
    const char **profile_hash;
    double *jbar_value;
    LineJbarValidityState *validity;
    uint64_t *sample_count;
    double *variance_or_standard_error;
    const char *q_set_hash;
    RadiationFieldUnits units;
    RadiationFieldFrame frame;
    RadiationFieldProvenance provenance;
} LineJbarCache;

typedef struct {
    size_t n_shells;
    size_t n_bins;
    double *raw_path_length;
    uint64_t *contribution_count;
    uint64_t out_of_grid_contribution_count;
} RadiationFieldAccumulator;

typedef struct {
    int enabled;
    RadiationField field;
    LineJbarCache line_jbar_cache;
    RadiationFieldAccumulator accumulator;
} RadiationFieldOwner;

/* Source-compatible name for A2-03 fixtures only; production owns an
 * always-enabled RadiationFieldOwner as of A2-04. */
typedef RadiationFieldOwner RadiationFieldShadow;

/* A2-04 producer transaction.  The public RadiationField is changed only by
 * radiation_field_commit().  MC supplies its raw path-length work buffer and
 * normalization factors; deterministic producers supply source-grid bin
 * averages.  Both forms enter the same validation/rebin/publish choke point. */
typedef struct {
    RadiationFieldProvenanceKind provenance_kind;
    const char *producer;
    const char *raw_ledger_sha256;
    uint64_t generation;
    double epoch;
    size_t n_shells;
    const double *v_inner;
    const double *v_outer;

    size_t source_n_bins;
    const double *source_frequency_bin_edges;
    const double *source_J_nu;
    const RadiationFieldValidityState *source_validity;
    RadiationFieldEstimatorStatisticKind statistic_kind;
    const uint64_t *source_count;
    const double *source_variance;

    const double *raw_path_length;
    const double *volume;
    double time_simulation;
    uint64_t contribution_count;
    uint64_t out_of_grid_contribution_count;
} RadiationFieldCommitRequest;

#ifdef __cplusplus
extern "C" {
#endif

int radiation_field_owner_init(RadiationFieldOwner *owner, size_t n_shells);
void radiation_field_owner_free(RadiationFieldOwner *owner);
int radiation_field_begin_mc(RadiationFieldOwner *owner,
                             const double *v_inner,
                             const double *v_outer,
                             size_t n_shells, double epoch,
                             uint64_t required_generation);
RadiationFieldAccumulator *radiation_field_accumulator_create(size_t n_shells);
void radiation_field_accumulator_free(RadiationFieldAccumulator *accumulator);
int radiation_field_accumulator_add(RadiationFieldAccumulator *accumulator,
                                    size_t shell, double comoving_nu,
                                    double path_length_measure);
int radiation_field_accumulator_reduce(RadiationFieldAccumulator *destination,
                                       const RadiationFieldAccumulator *source);
int radiation_field_commit(RadiationFieldOwner *owner,
                           const RadiationFieldCommitRequest *request);
int radiation_field_validate_owner(const RadiationFieldOwner *owner);
int radiation_field_dump_if_requested(const RadiationFieldOwner *owner);

#ifdef __cplusplus
}
#endif

#endif
