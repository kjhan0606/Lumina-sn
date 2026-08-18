#ifndef LUMINA_BF_RATE_JNU_H
#define LUMINA_BF_RATE_JNU_H

/* A2-05 (SPEC_A2_05_V2): bound-free photoionization rate integrated directly
 * on the canonical RadiationField view.
 *
 *   Gamma = 4*pi * Integral_{nu_th}^{nu_max} J_nu * sigma(nu) / (h*nu) dnu
 *
 * J_nu is a bin average (constant within a bin); sigma is tabulated and
 * integrated piecewise-linearly against 1/(h*nu) inside each overlap interval,
 * so the threshold partial bin uses [max(nu_th, nu_lo), nu_hi] exactly (gate
 * contract 4).  Validity propagates per R6: within the integration range the
 * sigma-weighted contribution share of non-VALID bins (w_miss) decides the
 * result state; no value substitution ever happens.
 */

#include <stddef.h>
#include <stdint.h>

#include "radiation_field.h"

typedef enum {
    BF_RATE_VALID = 1,
    BF_RATE_EXACT_ZERO = 2,
    BF_RATE_UNSAMPLED = 3,
    BF_RATE_OUT_OF_GRID = 4,
    BF_RATE_STALE = 5
} BfRateValidityState;

typedef struct {
    double gamma;              /* s^-1; meaningful only when state==VALID or EXACT_ZERO */
    BfRateValidityState state;
    double w_miss;             /* sigma-weighted non-VALID contribution share */
    uint64_t sample_count;     /* summed estimator counts over judged bins */
    double gamma_poisson_var;  /* delta-method Var(gamma) from per-bin counts
                                * (ORDER 6.3 CI); exactly 0 for deterministic
                                * commits, whose VALID bins carry count==0 */
} BfRateResult;

/* sigma tabulation: photo cross-section points (nu ascending, cm^2), zero
 * below nu_threshold.  Matches the existing tabulated interpretation used by
 * the A2-02C builder bf kernel (4*pi*sigma/(h*nu), zero below threshold). */
typedef struct {
    size_t n_points;
    const double *nu;          /* Hz, ascending */
    const double *sigma;       /* cm^2 */
    double nu_threshold;       /* Hz */
} BfCrossSection;

/* w_miss tolerance below which missing bins are ignored (R6). */
#define BF_RATE_W_MISS_TOLERANCE 1.0e-3

int bf_rate_gamma_from_view(const RadiationFieldView *view, size_t shell,
                            const BfCrossSection *sigma, BfRateResult *out);

/* Legacy-grid adapter shared by production (nlte_bf_gamma_canonical) and the
 * L-1bf gate fixture, so both run byte-identical arithmetic: the per-bin
 * sigma_row (or the Kramers sigma_0*(nu_th/nu_c)^3 evaluation at legacy bin
 * centres) is re-encoded as a bin-constant step tabulation over the nfb-bin
 * log grid and integrated against the canonical view.  A stored full-bin
 * average in the physical threshold bin is rescaled onto its active
 * [nu_th,nu_hi] support while preserving integral sigma*dnu.  node_nu and
 * node_sigma are caller scratch of at least 2*nfb entries each. */
int bf_rate_gamma_legacy_grid(const RadiationFieldView *view, size_t shell,
                              int nfb, double nu_min, double d_log_nu,
                              const double *sigma_row, double sigma_0,
                              double nu_thresh,
                              double *node_nu, double *node_sigma,
                              BfRateResult *out);

#endif
