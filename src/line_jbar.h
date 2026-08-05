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

#define LINE_JBAR_PROFILE_GAUSS_VD10 1
#define LINE_JBAR_VDOPPLER_CMS 1.0e6      /* 10 km/s */
#define LINE_JBAR_PROFILE_NDOPPLER 4.0    /* +-4 Doppler support */

typedef struct {
    size_t n_q;              /* Q_g size (lines) */
    int    *line_id;         /* [n_q] global line index (deck order) */
    double *line_nu;         /* [n_q] rest-frame line frequency, Hz */
    size_t *by_nu;           /* [n_q] permutation: ascending line_nu */
    char    q_set_hash[65];  /* SHA-256 hex over sorted line_id list */
    uint64_t profile_id;
    char    profile_hash[65];
} LineJbarQSet;

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

int  line_jbar_qset_build(LineJbarQSet *q, int n_lines,
                          const double *line_nu_all, const int *nlte_line_map,
                          const uint8_t *bb_in_domain /* NULL = all mapped */);
void line_jbar_qset_free(LineJbarQSet *q);

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

#endif
