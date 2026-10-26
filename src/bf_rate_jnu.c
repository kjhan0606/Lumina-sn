#include "bf_rate_jnu.h"

#include <math.h>
#include <limits.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BF_H_CGS 6.62607015e-27
#define BF_C_CGS 2.99792458e10

int bf_rate_point_contract(size_t n,const double *nu,const double *width,
    const double *j,const double *absorption,const double *stimulated,
    const double *eta,double edge,BfPointRates *out)
{
    if (!out || !n || !nu || !width || !j || !absorption || !stimulated ||
        !eta || !isfinite(edge) || edge<=0.) return -1;
    long double sum[6]={0};
    for(size_t k=0;k<n;k++) {
        if(!isfinite(nu[k]) || nu[k]<=0. || (k && nu[k]<=nu[k-1]) ||
           !isfinite(width[k]) || width[k]<0. || !isfinite(j[k]) || j[k]<0. ||
           !isfinite(absorption[k]) || absorption[k]<0. ||
           !isfinite(stimulated[k]) || stimulated[k]<0. ||
           !isfinite(eta[k]) || eta[k]<0.) return -1;
        long double weight=4.L*M_PI*width[k]/((long double)BF_H_CGS*nu[k]);
        long double terms[3]={weight*absorption[k]*j[k],weight*eta[k],
                              weight*stimulated[k]*j[k]};
        long double energy=(long double)BF_H_CGS*(nu[k]-edge);
        for(int q=0;q<3;q++) {sum[q]+=terms[q];sum[q+3]+=terms[q]*energy;}
    }
    BfPointRates result={(double)sum[0],(double)sum[1],(double)sum[2],
                         (double)sum[3],(double)sum[4],(double)sum[5]};
    if(!isfinite(result.photo) || !isfinite(result.spontaneous) ||
       !isfinite(result.stimulated) || !isfinite(result.heat_photo) ||
       !isfinite(result.heat_spontaneous) || !isfinite(result.heat_stimulated)) return -1;
    *out=result;
    return 0;
}

int bf_rate_point_contract_milne(size_t n,const double *nu,const double *width,
    const double *j,const double *sigma,double edge,double milne_prefactor,
    double inverse_kT_Hz,BfPointRates *out)
{
    if(!out || !n || !nu || !width || !j || !sigma ||
       !isfinite(edge) || edge<=0. ||
       !isfinite(milne_prefactor) || milne_prefactor<0. ||
       !isfinite(inverse_kT_Hz) || inverse_kT_Hz<=0.) return -1;
    long double sum[6]={0};
    for(size_t k=0;k<n;k++) {
        if(!isfinite(nu[k]) || nu[k]<=0. || (k && nu[k]<=nu[k-1]) ||
           !isfinite(width[k]) || width[k]<0. || !isfinite(j[k]) || j[k]<0. ||
           !isfinite(sigma[k]) || sigma[k]<0. ||
           (sigma[k]>0. && nu[k]<edge)) return -1;
        double stimulated=0.,eta=0.;
        if(sigma[k]>0.) {
            stimulated=sigma[k]*milne_prefactor*
                exp(-(nu[k]-edge)*inverse_kT_Hz);
            eta=stimulated*(2.*BF_H_CGS*nu[k]*nu[k]*nu[k]/(BF_C_CGS*BF_C_CGS));
            if(!isfinite(stimulated) || !isfinite(eta) ||
               stimulated<0. || eta<0.) return -1;
        }
        long double weight=4.L*M_PI*width[k]/((long double)BF_H_CGS*nu[k]);
        long double terms[3]={weight*sigma[k]*j[k],weight*eta,
                              weight*stimulated*j[k]};
        long double energy=(long double)BF_H_CGS*(nu[k]-edge);
        for(int q=0;q<3;q++) {sum[q]+=terms[q];sum[q+3]+=terms[q]*energy;}
    }
    BfPointRates result={(double)sum[0],(double)sum[1],(double)sum[2],
                         (double)sum[3],(double)sum[4],(double)sum[5]};
    if(!isfinite(result.photo) || !isfinite(result.spontaneous) ||
       !isfinite(result.stimulated) || !isfinite(result.heat_photo) ||
       !isfinite(result.heat_spontaneous) ||
       !isfinite(result.heat_stimulated)) return -1;
    *out=result;
    return 0;
}

int bf_rate_point_get(const BfPointRateView *v,size_t route,size_t shell,
    uint64_t generation,double t,double ne,BfPointRates *out)
{
    if(!out || !v || !v->n_routes || !v->n_shells ||
       v->n_routes>SIZE_MAX/v->n_shells || !v->rates || !v->temperature ||
       !v->electron_density || !generation || generation!=v->radiation_generation ||
       route>=v->n_routes || shell>=v->n_shells ||
       (v->shell_ready && !v->shell_ready[shell]) || !isfinite(t) || t<=0. ||
       !isfinite(ne) || ne<=0. || v->temperature[shell]!=t ||
       v->electron_density[shell]!=ne) return -1;
    BfPointRates r=v->rates[route*v->n_shells+shell];
    if(!isfinite(r.photo) || r.photo<0. || !isfinite(r.spontaneous) || r.spontaneous<0. ||
       !isfinite(r.stimulated) || r.stimulated<0. || !isfinite(r.heat_photo) ||
       !isfinite(r.heat_spontaneous) || !isfinite(r.heat_stimulated)) return -1;
    *out=r;
    return 0;
}

int bf_rate_point_row(const BfPointRateView *v,size_t nl,size_t ns,
    size_t lower,uint64_t material_generation,size_t *begin,size_t *end)
{
    if(!v || !begin || !end || !nl || !ns || v->n_levels!=nl ||
       v->n_shells!=ns || lower>=nl || v->n_routes>INT_MAX ||
       !material_generation || material_generation!=v->material_generation ||
       !v->offset || v->offset[0]!=0 || v->offset[nl]!=(int)v->n_routes ||
       (v->n_routes && (!v->target || !v->probability || !v->threshold_Hz ||
        !v->sigma_edge || !v->phot_grid || !v->saha_lower_per_upper))) return -1;
    int b=v->offset[lower],e=v->offset[lower+1];
    if(b<0 || e<b || (size_t)e>v->n_routes) return -1;
    *begin=(size_t)b; *end=(size_t)e;
    return 0;
}

int bf_rate_result_is_consumable(const BfRateResult *result)
{
    if (!result ||
        (result->state != BF_RATE_VALID &&
         result->state != BF_RATE_EXACT_ZERO) ||
        !isfinite(result->gamma) || result->gamma < 0.0 ||
        !isfinite(result->w_miss) || result->w_miss < 0.0 ||
        result->w_miss > BF_RATE_W_MISS_TOLERANCE ||
        !isfinite(result->gamma_poisson_var) ||
        result->gamma_poisson_var < 0.0)
        return 0;
    if (result->state == BF_RATE_EXACT_ZERO &&
        (result->gamma != 0.0 || result->w_miss != 0.0 ||
         result->gamma_poisson_var != 0.0))
        return 0;
    return 1;
}

int bf_rate_view_jnu(const RadiationFieldView *view, size_t shell, size_t bin,
                     double *out)
{
    if (!out) return -1;
    *out = 0.0;
    if (!view || !view->J_nu || !view->validity ||
        shell >= view->n_shells || bin >= view->n_bins)
        return -1;

    size_t index = shell * view->n_bins + bin;
    RadiationFieldValidityState state = view->validity[index];
    double value = view->J_nu[index];
    if (state == RADIATION_FIELD_VALID) {
        if (!(value > 0.0) || !isfinite(value)) return -1;
        *out = value;
        return 0;
    }
    if (state == RADIATION_FIELD_EXACT_ZERO) {
        if (value != 0.0) return -1;
        return 0;
    }
    if (state == RADIATION_FIELD_UNSAMPLED ||
        state == RADIATION_FIELD_OUT_OF_GRID ||
        state == RADIATION_FIELD_STALE)
        return 1;
    return -1;
}

int bf_rate_view_log_band_jnu(const RadiationFieldView *view, size_t shell,
                              double nu_lo, double nu_hi, double *out)
{
    if (!out) return -1;
    *out = 0.0;
    if (!view || !view->frequency_bin_edges || !view->n_bins ||
        view->n_bins == SIZE_MAX || shell >= view->n_shells ||
        !isfinite(nu_lo) || !isfinite(nu_hi) ||
        !(nu_lo > 0.0) || !(nu_hi > nu_lo)) return -1;
    const double *edges = view->frequency_bin_edges;
    size_t nb = view->n_bins;
    if (!isfinite(edges[0]) || !isfinite(edges[nb]) ||
        !(edges[0] > 0.0) || !(edges[nb] > edges[0])) return -1;
    if (nu_lo < edges[0] || nu_hi > edges[nb]) return 1;
    /* 게시된 view의 단조 edges를 이진 탐색한다. 사용 구간도 직접 검사한다. */
    size_t low = 0, high = nb;
    while (low < high) {
        size_t mid = low + (high-low)/2;
        if (!isfinite(edges[mid]) || !isfinite(edges[mid+1]) ||
            !(edges[mid+1] > edges[mid])) return -1;
        if (edges[mid+1] <= nu_lo) low = mid+1;
        else high = mid;
    }
    double cursor = nu_lo, sum = 0.0;
    for (size_t b = low; cursor < nu_hi && b < nb; ++b) {
        if (!isfinite(edges[b]) || !isfinite(edges[b+1]) ||
            !(edges[b+1] > edges[b])) return -1;
        if (edges[b] > cursor) return 1;
        double end = edges[b+1] < nu_hi ? edges[b+1] : nu_hi;
        if (!(end > cursor)) return -1;
        double j;
        int status = bf_rate_view_jnu(view, shell, b, &j);
        if (status != 0) return status;
        sum += j * log1p((end-cursor)/cursor);
        cursor = end;
    }
    if (cursor != nu_hi) return 1;
    double result = sum/log1p((nu_hi-nu_lo)/nu_lo);
    if (!(result >= 0.0) || !isfinite(result)) return -1;
    *out = result;
    return 0;
}

double bf_rate_sigma_log_weight(double sigma_bin_average, double nu_lo,
                                double nu_hi, double nu_thresh)
{
    if (!(sigma_bin_average > 0.0) || !isfinite(sigma_bin_average) ||
        !(nu_lo > 0.0) || !isfinite(nu_lo) ||
        !(nu_hi > nu_lo) || !isfinite(nu_hi) ||
        !(nu_thresh > 0.0) || !isfinite(nu_thresh) ||
        !(nu_hi > nu_thresh))
        return 0.0;

    double active_lo = nu_lo > nu_thresh ? nu_lo : nu_thresh;
    double active_width = nu_hi - active_lo;
    if (!(active_width > 0.0)) return 0.0;
    double sigma_active = sigma_bin_average;
    if (nu_lo < nu_thresh)
        sigma_active *= (nu_hi - nu_lo) / active_width;
    return sigma_active * log(nu_hi / active_lo);
}

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
        /* A declared, identically-zero cross section with tabulated support
         * above the edge defines a zero rate, not an unobserved radiation
         * field. Keep structural truncation (e.g. the old grid ending below
         * the Si V edge) OUT_OF_GRID, even when its stored row is all zero. */
        int zero_curve = sigma->nu[sigma->n_points-1] > nu_start &&
                         sigma->nu[0] < nu_hi_grid;
        for (size_t j=0; zero_curve && j<sigma->n_points; j++)
            if (sigma->sigma[j] != 0.0) zero_curve = 0;
        if (zero_curve) {
            out->state = BF_RATE_EXACT_ZERO;
            return 0;
        }
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
            /* A stored CMFGEN value is the linear-frequency average over the
             * complete bin, including zero support below a physical edge.
             * Locate that conserved sigma*dnu mass on its actual active
             * sub-interval before coupling it to a non-constant J_nu. */
            if (sg > 0.0 && lo < nu_thresh && hi > nu_thresh)
                sg *= (hi - lo) / (hi - nu_thresh);
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
    /* Both lanes now carry an explicit physical edge.  For stored CMFGEN rows
     * the first partial-bin step was rescaled above, so this clamp locates the
     * full-bin-average mass without cutting it twice. */
    BfCrossSection cs = {np, node_nu, node_sigma, nu_thresh};
    return bf_rate_gamma_from_view(view, shell, &cs, out);
}
