#ifndef LUMINA_RADIATION_FIELD_H
#define LUMINA_RADIATION_FIELD_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* Grid-containment contract, option B.  The NLTE/BF grid in lumina.h is the
 * sole authority; the canonical grid is its integer K-fold refinement.  The
 * integer offsets retain the previous canonical union with outward rounding,
 * without retaining an independent set of frequency literals. */
#define LUMINA_RADFIELD_REFINEMENT_K 2
#define LUMINA_RADFIELD_J_LO (-1754)
#define LUMINA_RADFIELD_J_HI 2112
#define LUMINA_RADFIELD_N_BINS \
    (LUMINA_RADFIELD_J_HI - LUMINA_RADFIELD_J_LO)
#define LUMINA_RADFIELD_DLOG \
    (log(NLTE_NU_MAX / NLTE_NU_MIN) / \
     ((double)LUMINA_RADFIELD_REFINEMENT_K * (double)NLTE_N_FREQ_BINS))
#define LUMINA_RADFIELD_NU_MIN_HZ \
    (NLTE_NU_MIN * exp((double)LUMINA_RADFIELD_J_LO * LUMINA_RADFIELD_DLOG))
#define LUMINA_RADFIELD_NU_MAX_HZ \
    (NLTE_NU_MIN * exp((double)LUMINA_RADFIELD_J_HI * LUMINA_RADFIELD_DLOG))
/* SHA-256 identities of the option-B union descriptor and edge-formula
 * descriptor above.  They intentionally invalidate old-grid seed assets. */
#define LUMINA_RADFIELD_UNION_SHA256 \
    "5be9f762c37871c275a50c22b4b033d5a0ad0edd67b03887ae6823ea1fa0378a"
#define LUMINA_RADFIELD_EDGE_SHA256 \
    "5e8d69a83353e6d4af7c0e84db48569d2a07e9fd07ed3a4e7da899be17ed2be8"

/* Changing K to a non-integral value (NB1), or away from the selected option,
 * is a compile-time contract violation rather than a runtime approximation. */
#ifdef __cplusplus
static_assert(LUMINA_RADFIELD_REFINEMENT_K == 2,
              "GRID_ALIGNMENT_VIOLATION: K must be the integer 2");
static_assert(LUMINA_RADFIELD_J_LO == -1754 &&
              LUMINA_RADFIELD_J_HI == 2112,
              "GRID_ALIGNMENT_VIOLATION: canonical origin moved");
static_assert(LUMINA_RADFIELD_N_BINS == 3866,
              "GRID_ALIGNMENT_VIOLATION: canonical offset span changed");
#else
_Static_assert(LUMINA_RADFIELD_REFINEMENT_K == 2,
               "GRID_ALIGNMENT_VIOLATION: K must be the integer 2");
_Static_assert(LUMINA_RADFIELD_J_LO == -1754 &&
               LUMINA_RADFIELD_J_HI == 2112,
               "GRID_ALIGNMENT_VIOLATION: canonical origin moved");
_Static_assert(LUMINA_RADFIELD_N_BINS == 3866,
               "GRID_ALIGNMENT_VIOLATION: canonical offset span changed");
#endif

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
    RADIATION_FIELD_PROVENANCE_DILUTE_PLANCK_LEGACY_APPROXIMATION = 3,
    /* R6: this names the deterministic line-profile quadrature itself.  It is
     * deliberately not CMFGEN_REPLAY: that provenance names the continuum
     * rebin, while this one names Jbar_l = Int(phi_l J_nu)dnu/Int(phi_l)dnu. */
    RADIATION_FIELD_PROVENANCE_CMFGEN_LINE_PROFILE_INTEGRAL = 4
} RadiationFieldProvenanceKind;

#define LUMINA_LINE_JBAR_DETERMINISTIC_PRODUCER \
    "A2-06:line-Jbar:deterministic-profile-integral:v1"

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
    /* COUNT means MC packet samples.  DETERMINISTIC means sample_count and
     * variance_or_standard_error are not applicable (stored as zero), not a
     * claim that an estimated variance is exactly zero. */
    RadiationFieldEstimatorStatisticKind statistic_kind;
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
    /* A2-06 owner-internal line-cache helpers (not part of the A2-03 public
     * schema): compact ascending line-id array for view/lookup, and owned
     * storage backing the per-entry profile_hash pointers. */
    /* A2-16: opaque SeedCapability* — the commit choke point revokes it the
     * moment the first commit publishes.  NULL when no seed is in play. */
    void     *seed_capability;
    uint64_t *line_ids_compact;
    size_t    line_n_compact;
    uint64_t  line_profile_id;
    char     *line_profile_hash_storage;
} RadiationFieldOwner;

/* Source-compatible name for A2-03 fixtures only; production owns an
 * always-enabled RadiationFieldOwner as of A2-04. */
typedef RadiationFieldOwner RadiationFieldShadow;

typedef enum {
    GRID_CONTAINMENT_OK = 0,
    GRID_CONTAINMENT_INVALID_GRID = 1,
    GRID_CONTAINMENT_LOW_SHORTFALL = 2,
    GRID_CONTAINMENT_HIGH_SHORTFALL = 3,
    GRID_CONTAINMENT_BOTH_SHORTFALL = 4
} GridContainmentStatus;

typedef struct {
    GridContainmentStatus status;
    double producer_min;
    double producer_max;
    double consumer_min;
    double consumer_max;
    double low_shortfall_hz;
    double high_shortfall_hz;
} GridContainmentResult;

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

    /* A2-06 (SPEC_A2_06_V5 1.1): selective line-Jbar block.  J_nu and the
     * line cache are two views of ONE atomic commit: both candidates validate
     * before either publishes; any failure leaves ALL public state unchanged.
     * MC form: sum/sumsq/count per (line,shell) from the packet-level
     * population (zero contributors included via line_n_packets).
     * Deterministic (replay) form: jbar/validity given directly. */
    size_t          line_n;              /* 0 = no line block this commit */
    const uint64_t *line_id;             /* [line_n] */
    const char     *line_q_set_hash;     /* SHA-256 hex */
    uint64_t        line_profile_id;
    const char     *line_profile_hash;   /* SHA-256 hex */
    RadiationFieldProvenanceKind line_provenance_kind;
    const char     *line_producer;
    const double   *line_sum;            /* [line_n*n_shells] MC only */
    const double   *line_sumsq;          /* [line_n*n_shells] MC only */
    const uint64_t *line_count;          /* [line_n*n_shells] MC only */
    uint64_t        line_n_packets;      /* N incl. zero contributors */
    int             line_error_latch;    /* accumulation failure => refuse */
    const double   *line_jbar;           /* [line_n*n_shells] deterministic */
    const int32_t  *line_validity;       /* [line_n*n_shells] deterministic */
} RadiationFieldCommitRequest;

#ifdef __cplusplus
extern "C" {
#endif

int radiation_field_owner_init(RadiationFieldOwner *owner, size_t n_shells);
void radiation_field_owner_free(RadiationFieldOwner *owner);
GridContainmentStatus grid_containment_check(
    const double *producer_edges, size_t producer_n_bins,
    const double *consumer_edges, size_t consumer_n_bins,
    size_t margin_bins, GridContainmentResult *out);
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

/* A2-05 (SPEC_A2_05_V2 R5): checked read-only view for rate consumers.
 * Success requires ALL of: enabled owner, canonical units and comoving frame,
 * expected epoch and shell count, generation required==computed==expected,
 * canonical bin count and finite ascending edges.  On failure no view is
 * produced and a distinct error code is returned; consumers must propagate
 * rate validity instead of substituting any value (never return rate 0).
 * The view aliases owner-internal storage and grants no write access. */
typedef enum {
    RADIATION_FIELD_VIEW_OK = 0,
    RADIATION_FIELD_VIEW_DISABLED = -1,
    RADIATION_FIELD_VIEW_UNITS_FRAME = -2,
    RADIATION_FIELD_VIEW_EPOCH_SHELLS = -3,
    RADIATION_FIELD_VIEW_STALE_GENERATION = -4,
    RADIATION_FIELD_VIEW_GRID = -5
} RadiationFieldViewStatus;

typedef struct {
    size_t n_shells;
    size_t n_bins;
    const double *frequency_bin_edges;   /* n_bins + 1, ascending, Hz */
    const double *J_nu;                  /* [n_shells][n_bins] bin averages */
    const RadiationFieldValidityState *validity;
    const uint64_t *count;
    uint64_t generation;
} RadiationFieldView;

int radiation_field_read_view(const RadiationFieldOwner *owner,
                              double expected_epoch,
                              size_t expected_n_shells,
                              uint64_t expected_generation,
                              RadiationFieldView *out);

/* A2-06 (SPEC_A2_06_V5 1.2): checked line-Jbar view.  Success requires ALL of
 * enabled, canonical units/comoving frame, epoch/shell count, generation
 * triple identity, q_set_hash and profile identity.  Failure fully
 * invalidates *out and returns a distinct code; consumers never see cache
 * arrays or any fallback (no zero, no coarse, no previous generation). */
typedef enum {
    LINE_JBAR_VIEW_OK = 0,
    LINE_JBAR_VIEW_DISABLED = -1,
    LINE_JBAR_VIEW_UNITS_FRAME = -2,
    LINE_JBAR_VIEW_EPOCH_SHELLS = -3,
    LINE_JBAR_VIEW_STALE_GENERATION = -4,
    LINE_JBAR_VIEW_QHASH = -5,
    LINE_JBAR_VIEW_PROFILE = -6
} LineJbarViewStatus;

typedef struct {
    size_t n_lines;                  /* Q_g size */
    size_t n_shells;
    const uint64_t *line_id;         /* [n_lines] ascending */
    const double   *jbar;            /* [n_lines*n_shells] */
    const LineJbarValidityState *validity;
    const uint64_t *count;
    const double   *se;              /* [n_lines*n_shells] standard error */
    RadiationFieldEstimatorStatisticKind statistic_kind;
    uint64_t generation;
} LineJbarView;

int radiation_field_line_jbar_view(const RadiationFieldOwner *owner,
                                   double expected_epoch,
                                   size_t expected_n_shells,
                                   uint64_t expected_generation,
                                   const char *expected_q_set_hash,
                                   uint64_t expected_profile_id,
                                   const char *expected_profile_hash,
                                   LineJbarView *out);

typedef struct {
    double jbar;
    LineJbarValidityState validity;
    uint64_t count;
    double se;
    RadiationFieldEstimatorStatisticKind statistic_kind;
} LineJbarValue;

/* MISS (line_id not in Q_g) is a distinct error, never a value. */
int line_jbar_lookup(const LineJbarView *view, size_t shell,
                     uint64_t line_id, LineJbarValue *out);

#ifdef __cplusplus
}
#endif

#endif
