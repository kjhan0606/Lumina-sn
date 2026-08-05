#include "bf_rate_jnu.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BF_H_CGS 6.62607015e-27

/* Integral of a linear sigma segment against 1/nu on [a,b]:
 *   sigma(nu) = s0 + m*(nu - n0)
 *   I = Integral sigma(nu)/nu dnu = (s0 - m*n0)*ln(b/a) + m*(b - a)
 * Exact for the declared piecewise-linear tabulation (gate contract 4). */
static double segment_sigma_over_nu(double n0, double s0, double n1, double s1,
                                    double a, double b)
{
    if (b <= a) return 0.0;
    double m = (s1 - s0) / (n1 - n0);
    return (s0 - m * n0) * log(b / a) + m * (b - a);
}

int bf_rate_gamma_from_view(const RadiationFieldView *view, size_t shell,
                            const BfCrossSection *sigma, BfRateResult *out)
{
    if (!view || !view->J_nu || !sigma || !out || shell >= view->n_shells ||
        sigma->n_points < 2 || !(sigma->nu_threshold > 0.0))
        return -1;
    out->gamma = 0.0;
    out->state = BF_RATE_OUT_OF_GRID;
    out->w_miss = 0.0;
    out->sample_count = 0;
    out->gamma_poisson_var = 0.0;

    const double *edges = view->frequency_bin_edges;
    size_t nb = view->n_bins;
    double nu_lo_grid = edges[0], nu_hi_grid = edges[nb];
    double nu_start = sigma->nu_threshold > nu_lo_grid ? sigma->nu_threshold
                                                       : nu_lo_grid;
    if (nu_start >= nu_hi_grid) {
        /* Threshold above the canonical grid: structural absence. */
        out->state = BF_RATE_OUT_OF_GRID;
        return 0;
    }

    double valid_integral = 0.0;   /* sum J_b * K_b over VALID bins */
    double var_sum = 0.0;          /* sum (J_b*K_b)^2/N_b (Poisson delta method) */
    double valid_weight = 0.0;     /* sum K_b over VALID/EXACT_ZERO bins */
    double miss_weight = 0.0;      /* sum K_b over non-VALID bins */
    int saw_stale = 0, saw_unsampled = 0, saw_oog = 0, saw_positive = 0;

    size_t k = 0;                  /* sigma segment cursor */
    for (size_t b = 0; b < nb; ++b) {
        double lo = edges[b] > nu_start ? edges[b] : nu_start;
        double hi = edges[b + 1];
        if (hi <= nu_start) continue;
        if (lo >= nu_hi_grid) break;

        /* K_b = Integral_{lo}^{hi} sigma(nu)/(h*nu) dnu over sigma segments */
        double kernel = 0.0;
        while (k + 1 < sigma->n_points && sigma->nu[k + 1] <= lo) k++;
        for (size_t j = k; j + 1 < sigma->n_points; ++j) {
            /* Zero-width pairs are legal: duplicated edge nodes encode a
             * bin-constant (step) tabulation as piecewise linear. */
            if (!(sigma->nu[j + 1] > sigma->nu[j])) continue;
            double a = sigma->nu[j] > lo ? sigma->nu[j] : lo;
            double c = sigma->nu[j + 1] < hi ? sigma->nu[j + 1] : hi;
            if (c <= a) {
                if (sigma->nu[j] >= hi) break;
                continue;
            }
            kernel += segment_sigma_over_nu(sigma->nu[j], sigma->sigma[j],
                                            sigma->nu[j + 1], sigma->sigma[j + 1],
                                            a, c);
        }
        /* Outside the tabulated sigma support the contribution is zero
         * (builder bf_kernel convention), so nothing is added there. */
        kernel /= BF_H_CGS;
        if (kernel <= 0.0) continue;

        size_t index = shell * nb + b;
        RadiationFieldValidityState state = view->validity[index];
        double j_value = view->J_nu[index];
        if (state == RADIATION_FIELD_VALID) {
            valid_integral += j_value * kernel;
            valid_weight += kernel;
            saw_positive = 1;
            out->sample_count += view->count[index];
            if (view->count[index] > 0)
                var_sum += j_value * kernel * j_value * kernel /
                           (double)view->count[index];
        } else if (state == RADIATION_FIELD_EXACT_ZERO) {
            valid_weight += kernel;
            out->sample_count += view->count[index];
        } else if (state == RADIATION_FIELD_STALE) {
            saw_stale = 1;
            miss_weight += kernel;
        } else if (state == RADIATION_FIELD_UNSAMPLED) {
            saw_unsampled = 1;
            miss_weight += kernel;
        } else {
            saw_oog = 1;
            miss_weight += kernel;
        }
    }

    double total_weight = valid_weight + miss_weight;
    if (total_weight <= 0.0) {
        out->state = BF_RATE_OUT_OF_GRID;
        return 0;
    }
    out->w_miss = miss_weight / total_weight;
    if (out->w_miss > BF_RATE_W_MISS_TOLERANCE) {
        /* R6 precedence: STALE > UNSAMPLED > OUT_OF_GRID. */
        out->state = saw_stale ? BF_RATE_STALE
                   : saw_unsampled ? BF_RATE_UNSAMPLED
                   : BF_RATE_OUT_OF_GRID;
        (void)saw_oog;
        return 0;
    }
    out->gamma = 4.0 * M_PI * valid_integral;
    out->gamma_poisson_var = 16.0 * M_PI * M_PI * var_sum;
    /* EXACT_ZERO asserts the WHOLE range was observed and is zero; with any
     * unobserved remainder (w_miss in (0, tol]) the honest state is VALID
     * with the recorded w_miss (SPEC R6: w_miss <= 1e-3 -> VALID). */
    out->state = (saw_positive || miss_weight > 0.0) ? BF_RATE_VALID
                                                     : BF_RATE_EXACT_ZERO;
    return 0;
}

int bf_rate_gamma_legacy_grid(const RadiationFieldView *view, size_t shell,
                              int nfb, double nu_min, double d_log_nu,
                              const double *sigma_row, double sigma_0,
                              double nu_thresh,
                              double *node_nu, double *node_sigma,
                              BfRateResult *out)
{
    if (nfb < 1 || !(nu_min > 0.0) || !(d_log_nu > 0.0) ||
        !node_nu || !node_sigma || !out)
        return -1;
    double log_numin = log(nu_min);
    size_t np = 0;
    for (int bb = 0; bb < nfb; bb++) {
        double lo = exp(log_numin + bb * d_log_nu);
        double hi = exp(log_numin + (bb + 1) * d_log_nu);
        double nu_c = exp(log_numin + (bb + 0.5) * d_log_nu);
        double sg;
        if (sigma_row) {
            sg = sigma_row[bb];
            if (!(sg > 0.0) || !isfinite(sg)) sg = 0.0;
        } else if (lo >= nu_thresh) {
            sg = sigma_0 * pow(nu_thresh / nu_c, 3.0);
        } else if (hi > nu_thresh) {
            /* Threshold partial bin (lo < nu_th < hi, either side of the
             * centre): the step constant s* makes the [nu_th, hi] piece equal
             * the exact Kramers integral (the integrator never integrates
             * below nu_th):
             *   s* * ln(hi/nu_th) = sigma_0*nu_th^3/3 * (nu_th^-3 - hi^-3). */
            double x = nu_thresh / hi;
            sg = sigma_0 * (1.0 - x * x * x) / (3.0 * log(hi / nu_thresh));
        } else {
            sg = 0.0;
        }
        node_nu[np] = lo;    node_sigma[np] = sg;    np++;
        node_nu[np] = hi;    node_sigma[np] = sg;    np++;
    }
    BfCrossSection cs = {np, node_nu, node_sigma, nu_thresh};
    return bf_rate_gamma_from_view(view, shell, &cs, out);
}
