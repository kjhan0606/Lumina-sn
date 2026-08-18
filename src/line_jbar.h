#ifndef LUMINA_LINE_JBAR_H
#define LUMINA_LINE_JBAR_H

/* A2-06 (SPEC_A2_06_V5>V4>V3>V2): selective line-Jbar estimator.
 *
 *   Jbar_lu,s = 1/(4pi V_s dt) * Sum_p Integral_seg eps(l) phi_lu(nu'(l)) dl
 *
 * phi = registered Gaussian (v_D = 10 km/s, +-4 Doppler, normalized), bound to
 * profile_id/hash.  Variance population is PACKET level including zero
 * contributors (A2-02C gate2 precedent):
 *   s^2 = (Q - S^2/N)/(N-1),  Var(Jbar) = norm^2 * N * s^2.
 * Accumulation: shared global sum/sumsq/count arrays; each thread keeps a
 * sparse per-packet partial and flushes with atomic adds at packet end.
 * Every add returns rc; any failure latches and must abort the commit. */

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#define LINE_JBAR_PROFILE_GAUSS_VD10 1
#define LINE_JBAR_VDOPPLER_CMS 1.0e6      /* 10 km/s */
#define LINE_JBAR_PROFILE_NDOPPLER 4.0    /* +-4 Doppler support */
#define LINE_JBAR_BB_LAMBDA_MIN_ANGSTROM 100.0
#define LINE_JBAR_BB_LAMBDA_MAX_ANGSTROM 20000.0
#define LINE_JBAR_BB_NU_MIN_HZ 1.49896229e14
#define LINE_JBAR_BB_NU_MAX_HZ 2.99792458e16
#define LINE_JBAR_BB_DOMAIN_CONTRACT_SHA256 \
    "3278062cf80281ffdcc4eb74ffc37e743cbdc51a128da5a319bfba7d3a6416c4"
#define LINE_JBAR_PROFILE_SHA256 \
    "f8572907be3ad2e9738a84dae1000338bb7100772cf1d3b52ec17561da409bbf"

static inline int line_jbar_frequency_in_bb_domain(double nu)
{
    return isfinite(nu) && nu > 0.0 &&
           nu >= LINE_JBAR_BB_NU_MIN_HZ && nu <= LINE_JBAR_BB_NU_MAX_HZ;
}

typedef enum {
    LINE_JBAR_SET_UNSPECIFIED = 0,
    LINE_JBAR_SET_RATE_GRAPH,
    LINE_JBAR_SET_ENERGY_DOMAIN
} LineJbarSetKind;

typedef struct {
    size_t n_q;              /* selected-set size; Q_g or Q_E */
    int    *line_id;         /* [n_q] global line index (deck order) */
    double *line_nu;         /* [n_q] rest-frame line frequency, Hz */
    size_t *by_nu;           /* [n_q] permutation: ascending line_nu */
    /* Compatibility field name: for Q_E this is the energy-set hash. */
    char    q_set_hash[65];  /* SHA-256 over role/domain + line_id list */
    char    domain_contract_hash[65]; /* empty only for legacy/unfiltered tests */
    uint64_t profile_id;
    char    profile_hash[65];
    LineJbarSetKind set_kind;
} LineJbarQSet;

/* Q_E deliberately uses the same numeric-set representation.  A single
 * Q_E cache can therefore serve energy consumers and the Q_g subset without
 * allocating a second overlapping Jbar slab. */
typedef LineJbarQSet LineJbarESet;

typedef enum {
    LINE_JBAR_SUBSET_OK = 0,
    LINE_JBAR_SUBSET_INVALID = -1,
    LINE_JBAR_SUBSET_MISSING_LINE = -2,
    LINE_JBAR_SUBSET_IDENTITY_MISMATCH = -3,
    LINE_JBAR_SUBSET_HASH_MISMATCH = -4,
    LINE_JBAR_SUBSET_FREQUENCY_MISMATCH = -5
} LineJbarSubsetStatus;

typedef struct {
    size_t   n_q, n_shells;
    double  *sum;            /* [n_q*n_shells] Sum_p y_p            (shared) */
    double  *sumsq;          /* [n_q*n_shells] Sum_p y_p^2          (shared) */
    uint64_t *count;         /* [n_q*n_shells] contributing packets (shared) */
    int      error_latch;    /* any accumulation failure => commit must fail */
} LineJbarAccumulator;

/* Per-thread sparse per-packet partial (open addressing, key=(q<<8)|shell). */
typedef struct {
    uint64_t *key;           /* capacity slots, key 0 = empty (keys are +1) */
    double   *value;
    size_t    capacity;      /* power of two */
    size_t    used;
    uint64_t *touched;       /* [<= capacity] insertion order for flush/clear */
} LineJbarPacketPartial;

typedef enum {
    LINE_JBAR_PROFILE_OK = 0,
    LINE_JBAR_PROFILE_INVALID_INPUT = -1,
    LINE_JBAR_PROFILE_UNCOVERED = -2,
    LINE_JBAR_PROFILE_NONFINITE = -3
} LineJbarProfileStatus;

typedef struct {
    size_t first_bin;
    size_t last_bin;
    size_t contributing_bins;
    double weight_sum;
    double weight_sum_lower;
    double error_upper_min;
    double error_upper_max;
} LineJbarProfileReport;

int  line_jbar_qset_build(LineJbarQSet *q, int n_lines,
                          const double *line_nu_all, const int *nlte_line_map,
                          const uint8_t *bb_in_domain /* NULL = all mapped */);
int  line_jbar_eset_build(LineJbarESet *e, int n_lines,
                          const double *line_nu_all);
LineJbarSubsetStatus line_jbar_qset_subset_of_eset(
                          const LineJbarQSet *q, const LineJbarESet *e,
                          size_t *first_missing_q);
void line_jbar_qset_free(LineJbarQSet *q);

/* Amended A2-02C BB rate graph.  Membership is the closed 100--20000 A
 * line-centre interval; the registered profile support enlarges the frequency
 * union, not the line-ID membership of Q_g. */
int line_jbar_bb_domain_mask_build(uint8_t *mask, int n_lines,
                                   const double *line_nu_all,
                                   const int *nlte_line_map,
                                   size_t *inside_enabled,
                                   size_t *outside_enabled);
int line_jbar_qset_profile_support_covered(const LineJbarQSet *q,
                                           double grid_nu_min,
                                           double grid_nu_max,
                                           size_t *first_bad_q);

int  line_jbar_accumulator_init(LineJbarAccumulator *a, size_t n_q,
                                size_t n_shells);
void line_jbar_accumulator_free(LineJbarAccumulator *a);

int  line_jbar_partial_init(LineJbarPacketPartial *p);
void line_jbar_partial_free(LineJbarPacketPartial *p);

/* One comoving path segment: nu' linear nu0->nu1, eps linear e0->e1 over
 * length L.  Adds eps*phi closed-form (erf) integrals for every Q_g line whose
 * +-4 Doppler support overlaps the swept band, into the packet partial. */
int line_jbar_segment_add(const LineJbarQSet *q, LineJbarPacketPartial *p,
                          int shell, double nu0, double nu1,
                          double e0, double e1, double length);

/* Packet end: y_p per touched (q,shell) -> atomic sum += y, sumsq += y^2,
 * count += 1; partial cleared.  rc != 0 latches a->error_latch. */
int line_jbar_packet_flush(LineJbarAccumulator *a, LineJbarPacketPartial *p);

/* Exact closed form of Integral_0^1 (e0+(e1-e0)t) phi(nu0+(nu1-nu0)t) L dt
 * for the registered Gaussian centred at nu_line (exposed for the selftest). */
double line_jbar_segment_phi_integral(double nu_line, double nu0, double nu1,
                                      double e0, double e1, double length);

/* Apply the registered nonnegative discrete Gaussian profile to every shell
 * of a shell-major fine field.  value_average is evaluated in the production
 * nearest-binary64 order.  error_average_upper encloses
 *     sum_i p_i * value_error_upper_i / sum_i p_i
 * by upward numerator and downward denominator arithmetic.  It is a proof
 * bound only; it never modifies value or value_error_upper.  The complete
 * +-ndoppler support must lie inside the sampled centre grid. */
LineJbarProfileStatus line_jbar_gaussian_discrete_shells(
    size_t n_shells, size_t n_bins,
    const double *nu, const double *dnu,
    const double *value, const double *value_error_upper,
    double line_nu, double vdoppler_cms, double ndoppler,
    double *value_average, double *error_average_upper,
    LineJbarProfileReport *report);

#endif
