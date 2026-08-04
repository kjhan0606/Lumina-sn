#include "lumina_cmf_field.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LCMF_PI 3.141592653589793238462643383279502884
#define LCMF_LOG_GRID_RTOL 1.0e-12
#define LCMF_FROZEN_VERSION 1u
#define LCMF_FROZEN_ENDIAN 0x01020304u
#define LCMF_C_LIGHT 2.99792458e10

typedef struct {
    uint32_t h[8];
    uint64_t bits;
    unsigned char block[64];
    size_t used;
} LCMFSha256;

static void set_error(LCMFError *error, int code, const char *message,
                      size_t ir, size_t inu, double value)
{
    if (error == NULL) return;
    memset(error, 0, sizeof(*error));
    error->code = code;
    error->radial_index = ir;
    error->frequency_index = inu;
    error->ray_index = SIZE_MAX;
    error->segment_index = SIZE_MAX;
    error->substep_index = SIZE_MAX;
    error->endpoint_index = SIZE_MAX;
    error->value = value;
    if (message != NULL) {
        (void)snprintf(error->message, sizeof(error->message), "%s", message);
    }
}

static int checked_product(size_t a, size_t b, size_t *out)
{
    if (out == NULL) return LCMF_EINVAL;
    if (a != 0 && b > SIZE_MAX / a) return LCMF_ENOMEM;
    *out = a * b;
    return LCMF_OK;
}

static void *checked_calloc(size_t count, size_t width)
{
    size_t bytes;
    if (checked_product(count, width, &bytes) != LCMF_OK) return NULL;
    if (bytes == 0) return NULL;
    return calloc(count, width);
}

static int finite_nonnegative(double value)
{
    return isfinite(value) && value >= 0.0;
}

static int gauss_legendre_unit(size_t n, double *nodes, double *weights)
{
    size_t i;
    const size_t half = (n + 1u) / 2u;
    if (n == 0 || nodes == NULL || weights == NULL) return LCMF_EINVAL;
    for (i = 0; i < half; ++i) {
        size_t iteration;
        double z = cos(LCMF_PI * ((double)i + 0.75) / ((double)n + 0.5));
        double derivative = 0.0;
        for (iteration = 0; iteration < 100u; ++iteration) {
            size_t j;
            double p0 = 1.0;
            double p1 = z;
            double pn;
            if (n == 1u) {
                pn = z;
                derivative = 1.0;
            } else {
                for (j = 2; j <= n; ++j) {
                    const double p2 = (((2.0 * (double)j - 1.0) * z * p1)
                                      - ((double)j - 1.0) * p0) / (double)j;
                    p0 = p1;
                    p1 = p2;
                }
                pn = p1;
                derivative = (double)n * (z * pn - p0) / (z * z - 1.0);
            }
            {
                const double next = z - pn / derivative;
                if (fabs(next - z) <= 4.0 * DBL_EPSILON * (1.0 + fabs(next))) {
                    z = next;
                    break;
                }
                z = next;
            }
        }
        if (iteration == 100u || !isfinite(z) || !isfinite(derivative)) {
            return LCMF_EGRID;
        }
        {
            const double w = 2.0 / ((1.0 - z * z) * derivative * derivative);
            nodes[i] = 0.5 * (1.0 - z);
            nodes[n - 1u - i] = 0.5 * (1.0 + z);
            weights[i] = 0.5 * w;
            weights[n - 1u - i] = 0.5 * w;
        }
    }
    return LCMF_OK;
}

int lumina_cmf_validate_input(const LCMFInput *input,
                              const LCMFOptions *options,
                              LCMFError *error)
{
    size_t i, k, cells;
    double dx = 0.0;
    if (error != NULL) memset(error, 0, sizeof(*error));
    if (input == NULL || options == NULL || input->r_edge == NULL ||
        input->nu == NULL || input->chi_total == NULL ||
        input->eta_fixed == NULL || input->nr < 2u || input->nnu == 0 ||
        options->n_mu == 0) {
        set_error(error, LCMF_EINVAL, "missing input array or zero dimension", 0, 0, 0.0);
        return LCMF_EINVAL;
    }
    if (input->scatter_mode == LCMF_SCAT_REDISTRIBUTION) {
        set_error(error, LCMF_EUNSUPPORTED, "redistribution is reserved for Stage 3.2", 0, 0, 0.0);
        return LCMF_EUNSUPPORTED;
    }
    if (input->scatter_mode != LCMF_SCAT_NONE &&
        input->scatter_mode != LCMF_SCAT_COHERENT) {
        set_error(error, LCMF_EINVAL, "unknown scattering mode", 0, 0,
                  (double)input->scatter_mode);
        return LCMF_EINVAL;
    }
    if (input->inner_bc != LCMF_BC_DIFFUSION &&
        input->inner_bc != LCMF_BC_IRRADIATION) {
        set_error(error, LCMF_EINVAL, "unknown inner boundary mode", 0, 0,
                  (double)input->inner_bc);
        return LCMF_EINVAL;
    }
    if (input->inner_bc == LCMF_BC_DIFFUSION &&
        (input->B_inner == NULL || input->dB_dtau_inner == NULL)) {
        set_error(error, LCMF_EINVAL, "diffusion boundary arrays are required", 0, 0, 0.0);
        return LCMF_EINVAL;
    }
    if (input->inner_bc == LCMF_BC_IRRADIATION &&
        input->inner_irradiation == NULL && input->r_edge[0] > 0.0) {
        set_error(error, LCMF_EINVAL, "irradiation callback is required for a core", 0, 0, 0.0);
        return LCMF_EINVAL;
    }
    if (options->frequency_advection &&
        (!isfinite(input->t_exp_s) || input->t_exp_s <= 0.0)) {
        set_error(error, LCMF_EINVAL, "positive finite expansion time is required", 0, 0,
                  input->t_exp_s);
        return LCMF_EINVAL;
    }
    if (options->max_source_iter == 0 || !isfinite(options->source_rtol) ||
        options->source_rtol <= 0.0) {
        set_error(error, LCMF_EINVAL, "invalid source iteration controls", 0, 0,
                  options->source_rtol);
        return LCMF_EINVAL;
    }
    if (options->radial_characteristic &&
        (!options->frequency_advection || options->n_mu != 1u)) {
        set_error(error, LCMF_EINVAL,
                  "radial characteristic diagnostic requires advection and n_mu=1", 0, 0, 0.0);
        return LCMF_EINVAL;
    }
    if ((options->r_eval == NULL && options->n_r_eval != 0u) ||
        (options->r_eval != NULL && options->n_r_eval == 0u)) {
        set_error(error, LCMF_EINVAL, "evaluation radii require both array and count", 0, 0, 0.0);
        return LCMF_EINVAL;
    }
    if (!isfinite(input->r_edge[0]) || input->r_edge[0] < 0.0) {
        set_error(error, LCMF_EGRID, "invalid first radial edge", 0, 0,
                  input->r_edge[0]);
        return LCMF_EGRID;
    }
    for (i = 1; i <= input->nr; ++i) {
        if (!isfinite(input->r_edge[i]) || input->r_edge[i] <= input->r_edge[i - 1u]) {
            set_error(error, LCMF_EGRID, "radial edges must strictly increase", i, 0,
                      input->r_edge[i]);
            return LCMF_EGRID;
        }
    }
    for (i = 0; i < options->n_r_eval; ++i) {
        const double r = options->r_eval[i];
        if (!isfinite(r) || r < input->r_edge[0] || r > input->r_edge[input->nr] ||
            (i > 0u && r <= options->r_eval[i - 1u])) {
            set_error(error, LCMF_EGRID, "evaluation radii must strictly increase inside the radial domain",
                      i, 0, r);
            return LCMF_EGRID;
        }
    }
    for (k = 0; k < input->nnu; ++k) {
        if (!isfinite(input->nu[k]) || input->nu[k] <= 0.0) {
            set_error(error, LCMF_EGRID, "frequency must be positive and finite", 0, k,
                      input->nu[k]);
            return LCMF_EGRID;
        }
        if (k > 0) {
            const double this_dx = log(input->nu[k - 1u] / input->nu[k]);
            if (!(this_dx > 0.0) || !isfinite(this_dx)) {
                set_error(error, LCMF_EGRID, "frequency must strictly descend", 0, k,
                          input->nu[k]);
                return LCMF_EGRID;
            }
            if (k == 1u) dx = this_dx;
            else if (fabs(this_dx - dx) > LCMF_LOG_GRID_RTOL * fabs(dx)) {
                set_error(error, LCMF_EGRID, "ln-frequency spacing is not uniform", 0, k,
                          this_dx);
                return LCMF_EGRID;
            }
        }
    }
    if (checked_product(input->nr, input->nnu, &cells) != LCMF_OK) {
        set_error(error, LCMF_ENOMEM, "grid size overflows size_t", 0, 0, 0.0);
        return LCMF_ENOMEM;
    }
    for (i = 0; i < input->nr; ++i) {
        for (k = 0; k < input->nnu; ++k) {
            const size_t q = i * input->nnu + k;
            const double chi = input->chi_total[q];
            const double eta = input->eta_fixed[q];
            const double coherent = input->chi_coherent == NULL ? 0.0 : input->chi_coherent[q];
            if (!finite_nonnegative(chi)) {
                set_error(error, isfinite(chi) ? LCMF_ENEGATIVE : LCMF_ENONFINITE,
                          "invalid total extinction", i, k, chi);
                return error == NULL ? LCMF_EINVAL : error->code;
            }
            if (!finite_nonnegative(eta)) {
                set_error(error, isfinite(eta) ? LCMF_ENEGATIVE : LCMF_ENONFINITE,
                          "invalid fixed emissivity", i, k, eta);
                return error == NULL ? LCMF_EINVAL : error->code;
            }
            if (!finite_nonnegative(coherent) || coherent > chi) {
                set_error(error, isfinite(coherent) ? LCMF_ENEGATIVE : LCMF_ENONFINITE,
                          "coherent extinction is outside [0,chi_total]", i, k, coherent);
                return error == NULL ? LCMF_EINVAL : error->code;
            }
        }
    }
    (void)cells;
    return LCMF_OK;
}

int lumina_cmf_ray_cache_build(const double *r_edge, size_t nr,
                               size_t radial_index, size_t n_mu,
                               LCMFRayCache *cache, LCMFError *error)
{
    double r;
    if (cache == NULL || r_edge == NULL || nr == 0 || radial_index >= nr || n_mu == 0) {
        set_error(error, LCMF_EINVAL, "invalid ray-cache request", radial_index, 0, 0.0);
        return LCMF_EINVAL;
    }
    r = 0.5 * (r_edge[radial_index] + r_edge[radial_index + 1u]);
    return lumina_cmf_ray_cache_build_at_radius(r, n_mu, cache, error);
}

int lumina_cmf_ray_cache_build_at_radius(double target_r, size_t n_mu,
                                         LCMFRayCache *cache, LCMFError *error)
{
    size_t m;
    int status;
    if (cache == NULL || !isfinite(target_r) || target_r < 0.0 || n_mu == 0u) {
        set_error(error, LCMF_EINVAL, "invalid radius ray-cache request", 0, 0, target_r);
        return LCMF_EINVAL;
    }
    memset(cache, 0, sizeof(*cache));
    cache->mu = (double *)checked_calloc(n_mu, sizeof(double));
    cache->weight = (double *)checked_calloc(n_mu, sizeof(double));
    cache->p = (double *)checked_calloc(n_mu, sizeof(double));
    if (cache->mu == NULL || cache->weight == NULL || cache->p == NULL) {
        lumina_cmf_ray_cache_free(cache);
        set_error(error, LCMF_ENOMEM, "ray-cache allocation failed", 0, 0, target_r);
        return LCMF_ENOMEM;
    }
    status = gauss_legendre_unit(n_mu, cache->mu, cache->weight);
    if (status != LCMF_OK) {
        lumina_cmf_ray_cache_free(cache);
        set_error(error, status, "Gauss-Legendre construction failed", 0, 0, target_r);
        return status;
    }
    for (m = 0; m < n_mu; ++m) {
        cache->p[m] = target_r * sqrt(1.0 - cache->mu[m] * cache->mu[m]);
    }
    cache->n_mu = n_mu;
    return LCMF_OK;
}

void lumina_cmf_ray_cache_free(LCMFRayCache *cache)
{
    if (cache == NULL) return;
    free(cache->mu);
    free(cache->weight);
    free(cache->p);
    memset(cache, 0, sizeof(*cache));
}

int lumina_cmf_sc_linear(double intensity_upstream, double source_upstream,
                         double source_downstream, double delta_tau,
                         double *intensity_downstream)
{
    double one_minus_e, ratio, psi_down, psi_up;
    if (intensity_downstream == NULL || !finite_nonnegative(intensity_upstream) ||
        !finite_nonnegative(source_upstream) || !finite_nonnegative(source_downstream) ||
        !isfinite(delta_tau) || delta_tau < 0.0) return LCMF_EINVAL;
    if (delta_tau == 0.0) {
        *intensity_downstream = intensity_upstream;
        return LCMF_OK;
    }
    one_minus_e = -expm1(-delta_tau);
    if (fabs(delta_tau) < 1.0e-4) {
        const double t = delta_tau;
        ratio = 1.0 - t * 0.5 + t * t / 6.0 - t * t * t / 24.0
                + t * t * t * t / 120.0;
    } else {
        ratio = one_minus_e / delta_tau;
    }
    psi_down = 1.0 - ratio;
    psi_up = one_minus_e - psi_down;
    *intensity_downstream = intensity_upstream * exp(-delta_tau)
                            + psi_up * source_upstream
                            + psi_down * source_downstream;
    if (!isfinite(*intensity_downstream)) return LCMF_ENONFINITE;
    if (*intensity_downstream < 0.0) return LCMF_ENEGATIVE;
    return LCMF_OK;
}

static int valid_interval(const LCMFInterval *interval)
{
    return interval != NULL && isfinite(interval->value) &&
           !isnan(interval->lower) && !isnan(interval->upper) &&
           interval->lower <= interval->value && interval->value <= interval->upper;
}

static double interval_radius(const LCMFInterval *interval)
{
    return fmax(interval->value - interval->lower,
                interval->upper - interval->value);
}

static double weighted_radius(double weight, const LCMFInterval *interval)
{
    return weight == 0.0 ? 0.0 : fabs(weight) * interval_radius(interval);
}

int lumina_cmf_sc_linear_signed(const LCMFInterval *intensity_upstream,
                                const LCMFInterval *source_upstream,
                                const LCMFInterval *source_downstream,
                                double delta_tau,
                                LCMFInterval *intensity_downstream)
{
    const double gamma_64 = (64.0 * DBL_EPSILON) /
                            (1.0 - 64.0 * DBL_EPSILON);
    double one_minus_e, ratio, psi_down, psi_up, attenuation;
    double term_i, term_u, term_d, propagated, kernel, radius;
    if (intensity_downstream == NULL || !valid_interval(intensity_upstream) ||
        !valid_interval(source_upstream) || !valid_interval(source_downstream) ||
        !isfinite(delta_tau) || delta_tau < 0.0) return LCMF_EINVAL;
    if (delta_tau == 0.0) {
        *intensity_downstream = *intensity_upstream;
        return LCMF_OK;
    }
    one_minus_e = -expm1(-delta_tau);
    if (fabs(delta_tau) < 1.0e-4) {
        const double t = delta_tau;
        ratio = 1.0 - t * 0.5 + t * t / 6.0 - t * t * t / 24.0
                + t * t * t * t / 120.0;
    } else {
        ratio = one_minus_e / delta_tau;
    }
    psi_down = 1.0 - ratio;
    psi_up = one_minus_e - psi_down;
    attenuation = exp(-delta_tau);
    term_i = attenuation * intensity_upstream->value;
    term_u = psi_up * source_upstream->value;
    term_d = psi_down * source_downstream->value;
    intensity_downstream->value = term_i + term_u + term_d;
    if (!isfinite(intensity_downstream->value) || !isfinite(attenuation) ||
        !isfinite(psi_up) || !isfinite(psi_down)) return LCMF_ENONFINITE;

    if (intensity_upstream->lower == 0.0 && intensity_upstream->upper == 0.0 &&
        source_upstream->lower == 0.0 && source_upstream->upper == 0.0 &&
        source_downstream->lower == 0.0 && source_downstream->upper == 0.0) {
        intensity_downstream->lower = 0.0;
        intensity_downstream->upper = 0.0;
        return LCMF_OK;
    }
    propagated = weighted_radius(attenuation, intensity_upstream)
               + weighted_radius(psi_up, source_upstream)
               + weighted_radius(psi_down, source_downstream);
    kernel = gamma_64 * (fabs(term_i) + fabs(term_u) + fabs(term_d) +
                         fabs(intensity_downstream->value));
    radius = propagated + kernel;
    if (isnan(radius)) return LCMF_ENONFINITE;
    intensity_downstream->lower = nextafter(intensity_downstream->value - radius,
                                            -INFINITY);
    intensity_downstream->upper = nextafter(intensity_downstream->value + radius,
                                            INFINITY);
    if (!isfinite(intensity_downstream->lower) ||
        !isfinite(intensity_downstream->upper)) return LCMF_ENONFINITE;
    return LCMF_OK;
}

static void quadratic_moments(double tau, double *j0, double *j1, double *j2)
{
    if (tau < 0.25) {
        double term0 = 1.0;
        double term1 = 0.5;
        double term2 = 1.0 / 3.0;
        size_t m;
        *j0 = term0;
        *j1 = term1;
        *j2 = term2;
        for (m = 1u; m <= 24u; ++m) {
            term0 *= -tau / (double)(m + 1u);
            term1 *= -tau / (double)(m + 2u);
            term2 *= -tau / (double)(m + 3u);
            *j0 += term0;
            *j1 += term1;
            *j2 += term2;
        }
    } else {
        const double em1 = expm1(-tau);
        *j0 = -em1 / tau;
        *j1 = (tau + em1) / (tau * tau);
        *j2 = (tau * tau - 2.0 * tau - 2.0 * em1) /
              (tau * tau * tau);
    }
}

int lumina_cmf_sc_quadratic_signed(const LCMFInterval *intensity_upstream,
                                   const LCMFInterval *coefficient0,
                                   const LCMFInterval *coefficient1,
                                   const LCMFInterval *coefficient2,
                                   double delta_s, double delta_tau,
                                   LCMFInterval *intensity_downstream)
{
    const double gamma_96 = (96.0 * DBL_EPSILON) /
                            (1.0 - 96.0 * DBL_EPSILON);
    double attenuation, j0, j1, j2, term_i, term0, term1, term2;
    double propagated, kernel, radius;
    if (intensity_downstream == NULL || !valid_interval(intensity_upstream) ||
        !valid_interval(coefficient0) || !valid_interval(coefficient1) ||
        !valid_interval(coefficient2) || !isfinite(delta_s) || delta_s < 0.0 ||
        !isfinite(delta_tau) || delta_tau < 0.0) return LCMF_EINVAL;
    if (delta_s == 0.0) {
        *intensity_downstream = *intensity_upstream;
        return LCMF_OK;
    }
    attenuation = exp(-delta_tau);
    quadratic_moments(delta_tau, &j0, &j1, &j2);
    term_i = attenuation * intensity_upstream->value;
    term0 = delta_s * j0 * coefficient0->value;
    term1 = delta_s * j1 * coefficient1->value;
    term2 = delta_s * j2 * coefficient2->value;
    intensity_downstream->value = term_i + term0 + term1 + term2;
    if (!isfinite(intensity_downstream->value) || !isfinite(attenuation) ||
        !isfinite(j0) || !isfinite(j1) || !isfinite(j2)) return LCMF_ENONFINITE;
    if (intensity_upstream->lower == 0.0 && intensity_upstream->upper == 0.0 &&
        coefficient0->lower == 0.0 && coefficient0->upper == 0.0 &&
        coefficient1->lower == 0.0 && coefficient1->upper == 0.0 &&
        coefficient2->lower == 0.0 && coefficient2->upper == 0.0) {
        intensity_downstream->lower = 0.0;
        intensity_downstream->upper = 0.0;
        return LCMF_OK;
    }
    propagated = weighted_radius(attenuation, intensity_upstream)
               + weighted_radius(delta_s * j0, coefficient0)
               + weighted_radius(delta_s * j1, coefficient1)
               + weighted_radius(delta_s * j2, coefficient2);
    kernel = gamma_96 * (fabs(term_i) + fabs(term0) + fabs(term1) +
                         fabs(term2) + fabs(intensity_downstream->value));
    radius = propagated + kernel;
    if (isnan(radius)) return LCMF_ENONFINITE;
    if (isinf(radius)) {
        intensity_downstream->lower = -INFINITY;
        intensity_downstream->upper = INFINITY;
    } else {
        intensity_downstream->lower = nextafter(intensity_downstream->value - radius,
                                                -INFINITY);
        intensity_downstream->upper = nextafter(intensity_downstream->value + radius,
                                                INFINITY);
    }
    return LCMF_OK;
}

static int sc_quadratic_nodal_signed(const LCMFInterval *intensity_upstream,
                                     const double z[3],
                                     const LCMFInterval emissivity[3],
                                     double upstream_z, double downstream_z,
                                     double delta_tau,
                                     LCMFInterval *intensity_downstream)
{
    const double gamma_96 = (96.0 * DBL_EPSILON) /
                            (1.0 - 96.0 * DBL_EPSILON);
    const double ds = downstream_z - upstream_z;
    double attenuation, j0, j1, j2, weight[3], term[3];
    double propagated, kernel, radius;
    size_t i;
    if (intensity_downstream == NULL || !valid_interval(intensity_upstream) ||
        !isfinite(upstream_z) || !isfinite(downstream_z) || !(ds > 0.0) ||
        !isfinite(delta_tau) || delta_tau < 0.0) return LCMF_EINVAL;
    attenuation = exp(-delta_tau);
    quadratic_moments(delta_tau, &j0, &j1, &j2);
    for (i = 0u; i < 3u; ++i) {
        const size_t j = (i + 1u) % 3u;
        const size_t k = (i + 2u) % 3u;
        const double denominator = (z[i] - z[j]) * (z[i] - z[k]);
        if (!valid_interval(&emissivity[i]) || denominator == 0.0) return LCMF_EINVAL;
        {
            const double q0 = ((upstream_z - z[j]) * (upstream_z - z[k])) /
                              denominator;
            const double q1 = ds * (2.0 * upstream_z - z[j] - z[k]) /
                              denominator;
            const double q2 = ds * ds / denominator;
            weight[i] = ds * (q0 * j0 + q1 * j1 + q2 * j2);
        }
        term[i] = weight[i] * emissivity[i].value;
    }
    intensity_downstream->value = attenuation * intensity_upstream->value +
                                  term[0] + term[1] + term[2];
    if (!isfinite(intensity_downstream->value) || !isfinite(attenuation) ||
        !isfinite(weight[0]) || !isfinite(weight[1]) || !isfinite(weight[2]))
        return LCMF_ENONFINITE;
    propagated = weighted_radius(attenuation, intensity_upstream) +
                 weighted_radius(weight[0], &emissivity[0]) +
                 weighted_radius(weight[1], &emissivity[1]) +
                 weighted_radius(weight[2], &emissivity[2]);
    kernel = gamma_96 * (fabs(attenuation * intensity_upstream->value) +
                         fabs(term[0]) + fabs(term[1]) + fabs(term[2]) +
                         fabs(intensity_downstream->value));
    radius = propagated + kernel;
    if (isnan(radius)) return LCMF_ENONFINITE;
    if (intensity_upstream->lower == 0.0 && intensity_upstream->upper == 0.0 &&
        emissivity[0].lower == 0.0 && emissivity[0].upper == 0.0 &&
        emissivity[1].lower == 0.0 && emissivity[1].upper == 0.0 &&
        emissivity[2].lower == 0.0 && emissivity[2].upper == 0.0) {
        intensity_downstream->lower = 0.0;
        intensity_downstream->upper = 0.0;
        return LCMF_OK;
    }
    if (isinf(radius)) {
        intensity_downstream->lower = -INFINITY;
        intensity_downstream->upper = INFINITY;
    } else {
        intensity_downstream->lower = nextafter(intensity_downstream->value - radius,
                                                -INFINITY);
        intensity_downstream->upper = nextafter(intensity_downstream->value + radius,
                                                INFINITY);
    }
    return LCMF_OK;
}

typedef struct {
    size_t count;
    double *z;
    size_t minus_target;
    size_t plus_target;
    size_t outward_start;
    int has_core;
} LCMFPath;

static int compare_double(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static void path_free(LCMFPath *path)
{
    if (path == NULL) return;
    free(path->z);
    memset(path, 0, sizeof(*path));
}

static int path_build(const LCMFInput *input, double target_r, double p,
                      LCMFPath *path)
{
    size_t j, n = 0, unique = 0, capacity;
    const double outer = input->r_edge[input->nr];
    const double inner = input->r_edge[0];
    const double target_z = sqrt(target_r * target_r - p * p);
    if (checked_product(2u, input->nr + 4u, &capacity) != LCMF_OK) return LCMF_ENOMEM;
    path->z = (double *)checked_calloc(capacity, sizeof(double));
    if (path->z == NULL) return LCMF_ENOMEM;
    path->minus_target = SIZE_MAX;
    path->plus_target = SIZE_MAX;
    path->has_core = p < inner;
    path->z[n++] = -sqrt(outer * outer - p * p);
    path->z[n++] = sqrt(outer * outer - p * p);
    for (j = 0; j < input->nr; ++j) {
        const double r = 0.5 * (input->r_edge[j] + input->r_edge[j + 1u]);
        if (r > p) {
            path->z[n++] = -sqrt(r * r - p * p);
            path->z[n++] = sqrt(r * r - p * p);
        }
    }
    if (path->has_core) {
        path->z[n++] = -sqrt(inner * inner - p * p);
        path->z[n++] = sqrt(inner * inner - p * p);
    } else {
        path->z[n++] = 0.0;
    }
    path->z[n++] = -target_z;
    path->z[n++] = target_z;
    qsort(path->z, n, sizeof(*path->z), compare_double);
    for (j = 0; j < n; ++j) {
        if (unique == 0u || path->z[j] != path->z[unique - 1u]) {
            path->z[unique++] = path->z[j];
        }
    }
    path->count = unique;
    for (j = 0; j < unique; ++j) {
        if (path->z[j] == -target_z) path->minus_target = j;
        if (path->z[j] == target_z) path->plus_target = j;
        if (path->has_core && path->z[j] == sqrt(inner * inner - p * p)) {
            path->outward_start = j;
        }
    }
    if (!path->has_core) {
        for (j = 0; j < unique; ++j) if (path->z[j] == 0.0) path->outward_start = j;
    }
    if (path->minus_target == SIZE_MAX || path->plus_target == SIZE_MAX || target_r < p) {
        path_free(path);
        return LCMF_EGRID;
    }
    return LCMF_OK;
}

static int radial_value(const LCMFInput *input, const double *array,
                        double r, size_t k, const char *quantity,
                        double *value, LCMFError *error)
{
    size_t low = 0, high = input->nr - 1u;
    const double first = 0.5 * (input->r_edge[0] + input->r_edge[1]);
    const double last = 0.5 * (input->r_edge[high] + input->r_edge[high + 1u]);
    if (r <= first) {
        const double second = 0.5 * (input->r_edge[1] + input->r_edge[2]);
        const double v0 = array[k];
        const double v1 = array[input->nnu + k];
        const double fraction = (r - first) / (second - first);
        if (v0 == 0.0 && v1 == 0.0) {
            /* The KA3 vacuum is the exact, identically-zero field. */
            *value = 0.0;
        } else if (!(v0 > 0.0) || !(v1 > 0.0)) {
            char message[160];
            (void)snprintf(message, sizeof(message),
                           "%s log radial extrapolation requires positive stencil values at inner face",
                           quantity);
            set_error(error, LCMF_ENEGATIVE, message, 0u, k,
                      !(v0 > 0.0) ? v0 : v1);
            return LCMF_ENEGATIVE;
        } else {
            *value = exp(log(v0) + fraction * (log(v1) - log(v0)));
        }
        low = 0u;
    } else if (r >= last) {
        const double previous = 0.5 * (input->r_edge[high - 1u] + input->r_edge[high]);
        const double v0 = array[(high - 1u) * input->nnu + k];
        const double v1 = array[high * input->nnu + k];
        const double fraction = (r - last) / (last - previous);
        if (v0 == 0.0 && v1 == 0.0) {
            /* The KA3 vacuum is the exact, identically-zero field. */
            *value = 0.0;
        } else if (!(v0 > 0.0) || !(v1 > 0.0)) {
            char message[160];
            (void)snprintf(message, sizeof(message),
                           "%s log radial extrapolation requires positive stencil values at outer face",
                           quantity);
            set_error(error, LCMF_ENEGATIVE, message, input->nr, k,
                      !(v0 > 0.0) ? v0 : v1);
            return LCMF_ENEGATIVE;
        } else {
            *value = exp(log(v1) + fraction * (log(v1) - log(v0)));
        }
        low = input->nr;
    } else {
    while (high - low > 1u) {
        const size_t middle = low + (high - low) / 2u;
        const double rm = 0.5 * (input->r_edge[middle] + input->r_edge[middle + 1u]);
        if (r < rm) high = middle; else low = middle;
    }
    {
        const double r0 = 0.5 * (input->r_edge[low] + input->r_edge[low + 1u]);
        const double r1 = 0.5 * (input->r_edge[high] + input->r_edge[high + 1u]);
        const double fraction = (r - r0) / (r1 - r0);
        const double v0 = array[low * input->nnu + k];
        const double v1 = array[high * input->nnu + k];
        *value = v0 + fraction * (v1 - v0);
    }
    }
    if (!isfinite(*value) || *value < 0.0 ||
        ((low == 0u || low == input->nr) && *value == 0.0 &&
         !(array[(low == 0u ? 0u : high - 1u) * input->nnu + k] == 0.0 &&
           array[(low == 0u ? 1u : high) * input->nnu + k] == 0.0))) {
        char message[160];
        const int status = isfinite(*value) ? LCMF_ENEGATIVE : LCMF_ENONFINITE;
        (void)snprintf(message, sizeof(message), "%s radial reconstruction failed at %s face",
                       quantity, low == 0u ? "inner" : (low == input->nr ? "outer" : "interior"));
        set_error(error, status, message, low, k, *value);
        return status;
    }
    return LCMF_OK;
}

static int sc_step(double intensity, double chi_up, double chi_down,
                   double eta_up, double eta_down, double ds, double *downstream)
{
    if (chi_up == 0.0 && chi_down == 0.0) {
        *downstream = intensity + 0.5 * (eta_up + eta_down) * ds;
        if (!isfinite(*downstream)) return LCMF_ENONFINITE;
        if (*downstream < 0.0) return LCMF_ENEGATIVE;
        return LCMF_OK;
    }
    if (!(chi_up > 0.0) || !(chi_down > 0.0)) return LCMF_ENEGATIVE;
    return lumina_cmf_sc_linear(intensity, eta_up / chi_up, eta_down / chi_down,
                                0.5 * (chi_up + chi_down) * ds, downstream);
}

static LCMFInterval point_interval(double value)
{
    LCMFInterval interval;
    interval.value = value;
    interval.lower = value;
    interval.upper = value;
    return interval;
}

static int sc_step_signed_eta(const LCMFInterval *intensity,
                              double chi_up, double chi_down,
                              const LCMFInterval *eta_up,
                              const LCMFInterval *eta_down,
                              double ds, LCMFInterval *downstream)
{
    const double gamma_16 = (16.0 * DBL_EPSILON) /
                            (1.0 - 16.0 * DBL_EPSILON);
    if (downstream == NULL || !valid_interval(intensity) ||
        !valid_interval(eta_up) || !valid_interval(eta_down) ||
        !isfinite(chi_up) || !isfinite(chi_down) || !isfinite(ds) || ds < 0.0)
        return LCMF_EINVAL;
    if (chi_up == 0.0 && chi_down == 0.0) {
        const double emission = 0.5 * (eta_up->value + eta_down->value) * ds;
        const double propagated = interval_radius(intensity)
                                + 0.5 * ds * (interval_radius(eta_up) +
                                              interval_radius(eta_down));
        const double kernel = gamma_16 * (fabs(intensity->value) + fabs(emission));
        const double radius = propagated + kernel;
        downstream->value = intensity->value + emission;
        if (!isfinite(downstream->value) || !isfinite(radius)) return LCMF_ENONFINITE;
        if (intensity->lower == 0.0 && intensity->upper == 0.0 &&
            eta_up->lower == 0.0 && eta_up->upper == 0.0 &&
            eta_down->lower == 0.0 && eta_down->upper == 0.0) {
            downstream->lower = 0.0;
            downstream->upper = 0.0;
        } else {
            downstream->lower = nextafter(downstream->value - radius, -INFINITY);
            downstream->upper = nextafter(downstream->value + radius, INFINITY);
        }
        return LCMF_OK;
    }
    if (!(chi_up > 0.0) || !(chi_down > 0.0)) return LCMF_ENEGATIVE;
    {
        LCMFInterval source_up, source_down;
        source_up.value = eta_up->value / chi_up;
        if (eta_up->lower == 0.0 && eta_up->upper == 0.0) {
            source_up.lower = 0.0;
            source_up.upper = 0.0;
        } else {
            source_up.lower = nextafter(eta_up->lower / chi_up, -INFINITY);
            source_up.upper = nextafter(eta_up->upper / chi_up, INFINITY);
        }
        source_down.value = eta_down->value / chi_down;
        if (eta_down->lower == 0.0 && eta_down->upper == 0.0) {
            source_down.lower = 0.0;
            source_down.upper = 0.0;
        } else {
            source_down.lower = nextafter(eta_down->lower / chi_down, -INFINITY);
            source_down.upper = nextafter(eta_down->upper / chi_down, INFINITY);
        }
        return lumina_cmf_sc_linear_signed(intensity, &source_up, &source_down,
                                           0.5 * (chi_up + chi_down) * ds,
                                           downstream);
    }
}

static int segment_residual(double intensity_up, double chi_up, double chi_down,
                            double eta_up, double eta_down, double ds,
                            double *residual)
{
    const double left_fraction = 0.5 - 1.0e-5;
    const double right_fraction = 0.5 + 1.0e-5;
    const double dchi = chi_down - chi_up;
    const double deta = eta_down - eta_up;
    const double chi_left = chi_up + left_fraction * dchi;
    const double chi_right = chi_up + right_fraction * dchi;
    const double eta_left = eta_up + left_fraction * deta;
    const double eta_right = eta_up + right_fraction * deta;
    const double ds_left = ds * left_fraction;
    const double ds_probe = ds * (right_fraction - left_fraction);
    double intensity_left, intensity_right, derivative, intensity_mid;
    double chi_mid, eta_mid, equation, scale;
    int status = sc_step(intensity_up, chi_up, chi_left, eta_up, eta_left,
                         ds_left, &intensity_left);
    if (status != LCMF_OK) return status;
    status = sc_step(intensity_left, chi_left, chi_right, eta_left, eta_right,
                     ds_probe, &intensity_right);
    if (status != LCMF_OK) return status;
    derivative = (intensity_right - intensity_left) / ds_probe;
    intensity_mid = 0.5 * (intensity_left + intensity_right);
    chi_mid = 0.5 * (chi_up + chi_down);
    eta_mid = 0.5 * (eta_up + eta_down);
    equation = derivative + chi_mid * intensity_mid - eta_mid;
    scale = fabs(derivative) + fabs(chi_mid * intensity_mid) + fabs(eta_mid);
    if (scale == 0.0) *residual = 0.0;
    else *residual = fabs(equation) / (scale + DBL_EPSILON * scale);
    return LCMF_OK;
}

static int segment_residual_signed(double intensity_up,
                                   double chi_up, double chi_down,
                                   double eta_up, double eta_down, double ds,
                                   double *residual)
{
    const double left_fraction = 0.5 - 1.0e-5;
    const double right_fraction = 0.5 + 1.0e-5;
    const double dchi = chi_down - chi_up;
    const double deta = eta_down - eta_up;
    const double chi_left = chi_up + left_fraction * dchi;
    const double chi_right = chi_up + right_fraction * dchi;
    const double eta_left = eta_up + left_fraction * deta;
    const double eta_right = eta_up + right_fraction * deta;
    const double ds_left = ds * left_fraction;
    const double ds_probe = ds * (right_fraction - left_fraction);
    LCMFInterval start = point_interval(intensity_up);
    LCMFInterval eta_start = point_interval(eta_up);
    LCMFInterval eta_l = point_interval(eta_left);
    LCMFInterval eta_r = point_interval(eta_right);
    LCMFInterval intensity_left, intensity_right;
    double derivative, intensity_mid, chi_mid, eta_mid, equation, scale;
    int status = sc_step_signed_eta(&start, chi_up, chi_left, &eta_start, &eta_l,
                                    ds_left, &intensity_left);
    if (status != LCMF_OK) return status;
    status = sc_step_signed_eta(&intensity_left, chi_left, chi_right,
                                &eta_l, &eta_r, ds_probe, &intensity_right);
    if (status != LCMF_OK) return status;
    derivative = (intensity_right.value - intensity_left.value) / ds_probe;
    intensity_mid = 0.5 * (intensity_left.value + intensity_right.value);
    chi_mid = 0.5 * (chi_up + chi_down);
    eta_mid = 0.5 * (eta_up + eta_down);
    equation = derivative + chi_mid * intensity_mid - eta_mid;
    scale = fabs(derivative) + fabs(chi_mid * intensity_mid) + fabs(eta_mid);
    *residual = scale == 0.0 ? 0.0
                            : fabs(equation) / (scale + DBL_EPSILON * scale);
    return isfinite(*residual) ? LCMF_OK : LCMF_ENONFINITE;
}

static int boundary_value(const LCMFInput *input, size_t k, double p,
                          int inner, double *value)
{
    const double radius = inner ? input->r_edge[0] : input->r_edge[input->nr];
    double mu = sqrt(1.0 - (p * p) / (radius * radius));
    if (!inner) mu = -mu;
    if (inner) {
        if (input->inner_bc == LCMF_BC_DIFFUSION) {
            *value = input->B_inner[k] + mu * input->dB_dtau_inner[k];
        } else {
            *value = input->inner_irradiation(input->boundary_ctx,p,mu,input->nu[k]);
        }
    } else if (input->outer_irradiation != NULL) {
        *value = input->outer_irradiation(input->boundary_ctx,p,mu,input->nu[k]);
    } else {
        *value = 0.0;
    }
    if (!isfinite(*value)) return LCMF_ENONFINITE;
    if (*value < 0.0) return LCMF_ENEGATIVE;
    return LCMF_OK;
}

static double local_radial_scale(const LCMFInput *input, double r)
{
    size_t low = 0u;
    size_t high = input->nr;
    if (r <= input->r_edge[0]) return input->r_edge[1] - input->r_edge[0];
    if (r >= input->r_edge[input->nr]) {
        return input->r_edge[input->nr] - input->r_edge[input->nr - 1u];
    }
    while (high - low > 1u) {
        const size_t middle = low + (high - low) / 2u;
        if (r < input->r_edge[middle]) high = middle;
        else low = middle;
    }
    return input->r_edge[low + 1u] - input->r_edge[low];
}

static int solve_static_ray(const LCMFInput *input, const double *eta_total,
                            size_t evaluation_index, double target_r, double p, size_t k,
                            double *minus, double *plus, double *max_residual,
                            LCMFError *error, size_t ray_index)
{
    LCMFPath path;
    double *intensity = NULL;
    size_t node;
    int status;
    memset(&path,0,sizeof(path));
    status = path_build(input,target_r,p,&path);
    if (status != LCMF_OK) return status;
    intensity = (double *)checked_calloc(path.count,sizeof(double));
    if (intensity == NULL) { path_free(&path); return LCMF_ENOMEM; }
    status = boundary_value(input,k,p,0,&intensity[0]);
    if (status != LCMF_OK) goto done;
    for (node=1u;node<path.count;++node) {
        double zu=path.z[node-1u], zd=path.z[node];
        double ds, midpoint_z, midpoint_r, h_loc;
        size_t sub, n_sub;
        if (path.has_core && node==path.outward_start) {
            status=boundary_value(input,k,p,1,&intensity[node]);
            if (status!=LCMF_OK) goto done;
            continue;
        }
        ds = zd - zu;
        midpoint_z = 0.5 * (zu + zd);
        midpoint_r = sqrt(p * p + midpoint_z * midpoint_z);
        h_loc = local_radial_scale(input, midpoint_r);
        if (!isfinite(ds) || !(ds > 0.0) || !isfinite(h_loc) || !(h_loc > 0.0) ||
            ds / h_loc > (double)SIZE_MAX) {
            set_error(error, LCMF_EGRID, "invalid SC subcycling scale", evaluation_index, k, ds);
            error->ray_index = ray_index;
            error->segment_index = node - 1u;
            status = LCMF_EGRID;
            goto done;
        }
        n_sub = (size_t)ceil(ds / h_loc);
        if (n_sub < 1u) n_sub = 1u;
        intensity[node] = intensity[node - 1u];
        for (sub = 0u; sub < n_sub; ++sub) {
            const double sub_zu = zu + ds * (double)sub / (double)n_sub;
            const double sub_zd = zu + ds * (double)(sub + 1u) / (double)n_sub;
            const double sub_ds = sub_zd - sub_zu;
            const double ru = sqrt(p * p + sub_zu * sub_zu);
            const double rd = sqrt(p * p + sub_zd * sub_zd);
            double chi_u, chi_d, eta_u, eta_d, residual, next;
            status = radial_value(input, input->chi_total, ru, k, "chi", &chi_u, error);
            if (status == LCMF_OK) {
                status = radial_value(input, input->chi_total, rd, k, "chi", &chi_d, error);
            }
            if (status == LCMF_OK) {
                status = radial_value(input, eta_total, ru, k, "eta", &eta_u, error);
            }
            if (status == LCMF_OK) {
                status = radial_value(input, eta_total, rd, k, "eta", &eta_d, error);
            }
            if (status != LCMF_OK) {
                error->ray_index = ray_index;
                error->segment_index = node - 1u;
                goto done;
            }
            status = sc_step(intensity[node], chi_u, chi_d, eta_u, eta_d, sub_ds, &next);
            if (status != LCMF_OK) {
                set_error(error,status,"static SC substep failed",evaluation_index,k,next);
                error->ray_index=ray_index; error->segment_index=node-1u;
                goto done;
            }
            status=segment_residual(intensity[node],chi_u,chi_d,eta_u,eta_d,sub_ds,&residual);
            if (status!=LCMF_OK) goto done;
            if (residual>*max_residual) *max_residual=residual;
            intensity[node] = next;
        }
    }
    *minus=intensity[path.minus_target]; *plus=intensity[path.plus_target];
done:
    free(intensity); path_free(&path); return status;
}

static LCMFInterval interpolate_history(const double *value,
                                        const double *lower,
                                        const double *upper,
                                        size_t left, size_t right,
                                        double fraction)
{
    const double gamma_16 = (16.0 * DBL_EPSILON) /
                            (1.0 - 16.0 * DBL_EPSILON);
    const double center = value[left] + fraction * (value[right] - value[left]);
    const double left_radius = fmax(value[left] - lower[left],
                                    upper[left] - value[left]);
    const double right_radius = fmax(value[right] - lower[right],
                                     upper[right] - value[right]);
    const double propagated = (fraction == 1.0 ? 0.0 : (1.0 - fraction) * left_radius) +
                              (fraction == 0.0 ? 0.0 : fraction * right_radius);
    const double kernel = gamma_16 * (fabs(value[left]) + fabs(value[right]) +
                                      fabs(center));
    LCMFInterval result;
    result.value = center;
    if (value[left] == 0.0 && lower[left] == 0.0 && upper[left] == 0.0 &&
        value[right] == 0.0 && lower[right] == 0.0 && upper[right] == 0.0) {
        result.lower = 0.0;
        result.upper = 0.0;
    } else {
        result.lower = nextafter(center - propagated - kernel, -INFINITY);
        result.upper = nextafter(center + propagated + kernel, INFINITY);
    }
    return result;
}

static LCMFInterval interval_linear3(double weight0, const LCMFInterval *value0,
                                     double weight1, const LCMFInterval *value1,
                                     double weight2, const LCMFInterval *value2)
{
    const double gamma_32 = (32.0 * DBL_EPSILON) /
                            (1.0 - 32.0 * DBL_EPSILON);
    const double term0 = weight0 * value0->value;
    const double term1 = weight1 * value1->value;
    const double term2 = weight2 * value2->value;
    const double center = term0 + term1 + term2;
    const double propagated = weighted_radius(weight0, value0) +
                              weighted_radius(weight1, value1) +
                              weighted_radius(weight2, value2);
    const double kernel = gamma_32 * (fabs(term0) + fabs(term1) +
                                      fabs(term2) + fabs(center));
    LCMFInterval result;
    result.value = center;
    if (value0->lower == 0.0 && value0->upper == 0.0 &&
        value1->lower == 0.0 && value1->upper == 0.0 &&
        value2->lower == 0.0 && value2->upper == 0.0) {
        result.lower = 0.0;
        result.upper = 0.0;
    } else {
        result.lower = nextafter(center - propagated - kernel, -INFINITY);
        result.upper = nextafter(center + propagated + kernel, INFINITY);
    }
    return result;
}

static LCMFInterval interpolate_quadratic_interval(const double z[3],
                                                   const LCMFInterval value[3],
                                                   double coordinate)
{
    const double w0 = ((coordinate - z[1]) * (coordinate - z[2])) /
                      ((z[0] - z[1]) * (z[0] - z[2]));
    const double w1 = ((coordinate - z[0]) * (coordinate - z[2])) /
                      ((z[1] - z[0]) * (z[1] - z[2]));
    const double w2 = ((coordinate - z[0]) * (coordinate - z[1])) /
                      ((z[2] - z[0]) * (z[2] - z[1]));
    return interval_linear3(w0, &value[0], w1, &value[1], w2, &value[2]);
}

static int branch_quadratic_stencil(const LCMFPath *path, size_t node,
                                    size_t index[3])
{
    size_t low = 0u;
    size_t high = path->count - 1u;
    if (node == 0u || node >= path->count ||
        (path->has_core && node == path->outward_start)) return LCMF_EGRID;
    if (path->has_core) {
        if (node < path->outward_start) high = path->outward_start - 1u;
        else low = path->outward_start;
    }
    if (high - low + 1u < 3u) return LCMF_EGRID;
    if (node + 1u <= high) {
        index[0] = node - 1u;
        index[1] = node;
        index[2] = node + 1u;
    } else {
        if (node < low + 2u) return LCMF_EGRID;
        index[0] = node - 2u;
        index[1] = node - 1u;
        index[2] = node;
    }
    return LCMF_OK;
}

static int segment_residual_quadratic(double intensity_up, double chi,
                                      const LCMFInterval *coefficient0,
                                      const LCMFInterval *coefficient1,
                                      const LCMFInterval *coefficient2,
                                      double ds, double *residual)
{
    const double left_fraction = 0.5 - 1.0e-5;
    const double right_fraction = 0.5 + 1.0e-5;
    const LCMFInterval start = point_interval(intensity_up);
    LCMFInterval left1, left2, right1, right2, intensity_left, intensity_right;
    double derivative, intensity_mid, eta_mid, equation, scale;
    int status;
    left1 = interval_linear3(left_fraction, coefficient1, 0.0, coefficient0,
                             0.0, coefficient2);
    left2 = interval_linear3(left_fraction * left_fraction, coefficient2,
                             0.0, coefficient0, 0.0, coefficient1);
    right1 = interval_linear3(right_fraction, coefficient1, 0.0, coefficient0,
                              0.0, coefficient2);
    right2 = interval_linear3(right_fraction * right_fraction, coefficient2,
                              0.0, coefficient0, 0.0, coefficient1);
    status = lumina_cmf_sc_quadratic_signed(&start, coefficient0, &left1, &left2,
                                             ds * left_fraction,
                                             chi * ds * left_fraction,
                                             &intensity_left);
    if (status != LCMF_OK) return status;
    status = lumina_cmf_sc_quadratic_signed(&start, coefficient0, &right1, &right2,
                                             ds * right_fraction,
                                             chi * ds * right_fraction,
                                             &intensity_right);
    if (status != LCMF_OK) return status;
    derivative = (intensity_right.value - intensity_left.value) /
                 (ds * (right_fraction - left_fraction));
    intensity_mid = 0.5 * (intensity_left.value + intensity_right.value);
    eta_mid = coefficient0->value + 0.5 * coefficient1->value +
              0.25 * coefficient2->value;
    equation = derivative + chi * intensity_mid - eta_mid;
    scale = fabs(derivative) + fabs(chi * intensity_mid) + fabs(eta_mid);
    *residual = scale == 0.0 ? 0.0
                            : fabs(equation) / (scale + DBL_EPSILON * scale);
    return isfinite(*residual) ? LCMF_OK : LCMF_ENONFINITE;
}

static LCMFInterval bdf_eta_interval(double physical_eta, double coefficient,
                                     const LCMFInterval *previous,
                                     const LCMFInterval *previous2,
                                     int second_order)
{
    const double gamma_32 = (32.0 * DBL_EPSILON) /
                            (1.0 - 32.0 * DBL_EPSILON);
    const double term_previous = (second_order ? 2.0 : 1.0) * coefficient *
                                 previous->value;
    const double term_previous2 = second_order ? 0.5 * coefficient *
                                                 previous2->value : 0.0;
    const double propagated = (second_order ? 2.0 : 1.0) * fabs(coefficient) *
                              interval_radius(previous) +
                              (second_order ? 0.5 * fabs(coefficient) *
                                              interval_radius(previous2) : 0.0);
    const double center = physical_eta + term_previous - term_previous2;
    const double kernel = gamma_32 * (fabs(physical_eta) + fabs(term_previous) +
                                      fabs(term_previous2) + fabs(center));
    LCMFInterval result;
    result.value = center;
    if (physical_eta == 0.0 && previous->lower == 0.0 && previous->upper == 0.0 &&
        (!second_order || (previous2->lower == 0.0 && previous2->upper == 0.0))) {
        result.lower = 0.0;
        result.upper = 0.0;
    } else {
        result.lower = nextafter(center - propagated - kernel, -INFINITY);
        result.upper = nextafter(center + propagated + kernel, INFINITY);
    }
    return result;
}

static void record_bdf_eta_negative(LCMFResult *result, size_t evaluation_index,
                                    size_t k, size_t ray_index, size_t segment_index,
                                    size_t substep_index, size_t endpoint_index,
                                    const LCMFInterval *eta,
                                    const LCMFInterval *previous,
                                    const LCMFInterval *previous2)
{
    ++result->bdf_eta_negative_count;
    if (eta->value < result->bdf_eta_min) result->bdf_eta_min = eta->value;
    if (result->bdf_eta_negative_count == 1u) {
        const double ratio = previous->value == 0.0
                           ? (previous2->value > 0.0 ? INFINITY : NAN)
                           : previous2->value / previous->value;
        set_error(&result->bdf_eta_first, LCMF_OK,
                  "BDF effective emissivity negative diagnostic",
                  evaluation_index, k, eta->value);
        result->bdf_eta_first.ray_index = ray_index;
        result->bdf_eta_first.segment_index = segment_index;
        result->bdf_eta_first.substep_index = substep_index;
        result->bdf_eta_first.endpoint_index = endpoint_index;
        result->bdf_eta_first.term_previous = previous->value;
        result->bdf_eta_first.term_previous2 = previous2->value;
        result->bdf_eta_first.decay_ratio = ratio;
        result->bdf_eta_first.theoretical_limit = 4.0;
        result->bdf_eta_first.interval_lower = eta->lower;
        result->bdf_eta_first.interval_upper = eta->upper;
    }
}

static double interval_abs_max(const LCMFInterval *interval)
{
    double scale;
    if (interval == NULL || !isfinite(interval->value) ||
        !isfinite(interval->lower) || !isfinite(interval->upper)) return 0.0;
    scale = fabs(interval->value);
    if (fabs(interval->lower) > scale) scale = fabs(interval->lower);
    if (fabs(interval->upper) > scale) scale = fabs(interval->upper);
    return scale;
}

static int record_solution_guard(LCMFResult *result, size_t evaluation_index,
                                 size_t k, size_t ray_index, size_t segment_index,
                                 size_t substep_index, const LCMFInterval *solution,
                                 double h, double local_scale, double global_scale)
{
    double scale, truncation_bound;
    if (result == NULL || solution == NULL || !isfinite(solution->value)) {
        if (result != NULL) {
            ++result->nonfinite_count;
            set_error(&result->error, LCMF_ENONFINITE,
                      "non-finite solution value", evaluation_index, k,
                      solution == NULL ? NAN : solution->value);
            result->error.ray_index = ray_index;
            result->error.segment_index = segment_index;
            result->error.substep_index = substep_index;
        }
        return LCMF_ENONFINITE;
    }
    scale = local_scale;
    if (global_scale > scale) scale = global_scale;
    truncation_bound = LCMF_TRUNCATION_ERROR_COEFFICIENT * h * h * scale;
    if (solution->upper < 0.0) {
        if (!isfinite(h) || h < 0.0 || !isfinite(scale) || scale < 0.0 ||
            !isfinite(truncation_bound)) {
            ++result->nonfinite_count;
            set_error(&result->error, LCMF_ENONFINITE,
                      "non-finite truncation guard scale", evaluation_index, k,
                      truncation_bound);
            result->error.ray_index = ray_index;
            result->error.segment_index = segment_index;
            result->error.substep_index = substep_index;
            return LCMF_ENONFINITE;
        }
        if (fabs(solution->value) > truncation_bound) {
            ++result->solution_negative_excess_count;
            if (result->error.code != LCMF_ENEGATIVE) {
                set_error(&result->error, LCMF_ENEGATIVE,
                          "certified negative solution exceeds truncation bound",
                          evaluation_index, k, solution->value);
                result->error.ray_index = ray_index;
                result->error.segment_index = segment_index;
                result->error.substep_index = substep_index;
                result->error.interval_lower = solution->lower;
                result->error.interval_upper = solution->upper;
                result->error.term_previous = scale;
                result->error.term_previous2 = h;
                result->error.theoretical_limit = truncation_bound;
            }
            return LCMF_ENEGATIVE;
        }
        ++result->solution_subtruncation_count;
        if (solution->value < result->solution_subtruncation_min) {
            result->solution_subtruncation_min = solution->value;
            set_error(&result->solution_subtruncation_min_location, LCMF_OK,
                      "minimum certified negative sub-truncation solution",
                      evaluation_index, k, solution->value);
            result->solution_subtruncation_min_location.ray_index = ray_index;
            result->solution_subtruncation_min_location.segment_index = segment_index;
            result->solution_subtruncation_min_location.substep_index = substep_index;
            result->solution_subtruncation_min_location.interval_lower = solution->lower;
            result->solution_subtruncation_min_location.interval_upper = solution->upper;
        }
        if (result->solution_subtruncation_count == 1u) {
            set_error(&result->solution_subtruncation_first, LCMF_OK,
                      "certified negative solution is sub-truncation",
                      evaluation_index, k, solution->value);
            result->solution_subtruncation_first.ray_index = ray_index;
            result->solution_subtruncation_first.segment_index = segment_index;
            result->solution_subtruncation_first.substep_index = substep_index;
            result->solution_subtruncation_first.interval_lower = solution->lower;
            result->solution_subtruncation_first.interval_upper = solution->upper;
            result->solution_subtruncation_first_bound = truncation_bound;
            result->solution_subtruncation_first_h = h;
            result->solution_subtruncation_first_scale = scale;
        }
    } else if (solution->lower <= 0.0 &&
               !(solution->value == 0.0 && solution->lower == 0.0 &&
                 solution->upper == 0.0)) {
        if (fabs(solution->value) <= truncation_bound) {
            ++result->solution_sign_indeterminate_subtruncation_count;
            return LCMF_OK;
        }
        ++result->sign_uncertain_count;
        if (result->error.code == LCMF_OK) {
            set_error(&result->error, LCMF_ESIGNUNCERTAIN,
                      "solution interval contains zero",
                      evaluation_index, k, solution->value);
            result->error.ray_index = ray_index;
            result->error.segment_index = segment_index;
            result->error.substep_index = substep_index;
            result->error.interval_lower = solution->lower;
            result->error.interval_upper = solution->upper;
        }
    }
    return LCMF_OK;
}

#ifdef LCMF_TEST_HOOKS
int lumina_cmf_solution_guard_probe(const LCMFInterval *solution,
                                    double h, double local_scale,
                                    double global_scale,
                                    LCMFResult *result)
{
    if (result == NULL) return LCMF_EINVAL;
    if (result->solution_subtruncation_min == 0.0 &&
        result->solution_subtruncation_count == 0u) {
        result->solution_subtruncation_min = INFINITY;
    }
    return record_solution_guard(result, 7u, 11u, 13u, 17u, 19u,
                                 solution, h, local_scale, global_scale);
}
#endif

static int solve_advection_ray(const LCMFInput *input, const double *eta_total,
                               size_t evaluation_index, double target_r, double p,
                               double *minus, double *plus, double *max_residual,
                               LCMFResult *result, size_t ray_index)
{
    LCMFPath path;
    double *planes = NULL, *plane_lower = NULL, *plane_upper = NULL;
    size_t plane_cells, k, node;
    const double dx = log(input->nu[0] / input->nu[1]);
    const double radial_span = input->r_edge[input->nr] - input->r_edge[0];
    const double frequency_span = log(input->nu[0] / input->nu[input->nnu - 1u]);
    const double a = 1.0 / (LCMF_C_LIGHT * input->t_exp_s);
    double ray_global_scale = 0.0;
    LCMFError *error = &result->error;
    int status = LCMF_OK;
    memset(&path, 0, sizeof(path));
    status = path_build(input, target_r, p, &path);
    if (status != LCMF_OK) return status;
    for (k = 0u; k < input->nnu; ++k) {
        double boundary_scale;
        size_t radial_index;
        status = boundary_value(input, k, p, 0, &boundary_scale);
        if (status != LCMF_OK) {
            path_free(&path);
            return status;
        }
        if (fabs(boundary_scale) > ray_global_scale)
            ray_global_scale = fabs(boundary_scale);
        if (path.has_core) {
            status = boundary_value(input, k, p, 1, &boundary_scale);
            if (status != LCMF_OK) {
                path_free(&path);
                return status;
            }
            if (fabs(boundary_scale) > ray_global_scale)
                ray_global_scale = fabs(boundary_scale);
        }
        if (input->r_edge[0] > 0.0) {
            status = boundary_value(input, k, 0.0, 1, &boundary_scale);
            if (status != LCMF_OK) {
                path_free(&path);
                return status;
            }
            if (fabs(boundary_scale) > ray_global_scale)
                ray_global_scale = fabs(boundary_scale);
        }
        for (radial_index = 0u; radial_index < input->nr; ++radial_index) {
            const size_t q = radial_index * input->nnu + k;
            const double source_scale = input->chi_total[q] > 0.0
                                      ? eta_total[q] / input->chi_total[q] : 0.0;
            if (isfinite(source_scale) && source_scale > ray_global_scale)
                ray_global_scale = source_scale;
        }
    }
    if (input->nnu < 3u || checked_product(path.count, input->nnu, &plane_cells) != LCMF_OK) {
        path_free(&path);
        return input->nnu < 3u ? LCMF_EGRID : LCMF_ENOMEM;
    }
    planes = (double *)checked_calloc(plane_cells, sizeof(double));
    plane_lower = (double *)checked_calloc(plane_cells, sizeof(double));
    plane_upper = (double *)checked_calloc(plane_cells, sizeof(double));
    if (planes == NULL || plane_lower == NULL || plane_upper == NULL) {
        free(planes); free(plane_lower); free(plane_upper);
        path_free(&path); return LCMF_ENOMEM;
    }
    for (k = 0u; k < input->nnu; ++k) {
        double *current = planes + k * path.count;
        double *current_lower = plane_lower + k * path.count;
        double *current_upper = plane_upper + k * path.count;
        double frequency_neighbor_scale = 0.0;
        int plane_eta_negative = 0;
        if (k > 0u) {
            const size_t first_neighbor = k > 1u ? k - 2u : k - 1u;
            size_t neighbor_frequency, neighbor_node;
            for (neighbor_frequency = first_neighbor; neighbor_frequency < k;
                 ++neighbor_frequency) {
                const double *neighbor_value = planes + neighbor_frequency * path.count;
                const double *neighbor_lower = plane_lower + neighbor_frequency * path.count;
                const double *neighbor_upper = plane_upper + neighbor_frequency * path.count;
                for (neighbor_node = 0u; neighbor_node < path.count; ++neighbor_node) {
                    LCMFInterval neighbor;
                    double neighbor_scale;
                    neighbor.value = neighbor_value[neighbor_node];
                    neighbor.lower = neighbor_lower[neighbor_node];
                    neighbor.upper = neighbor_upper[neighbor_node];
                    neighbor_scale = interval_abs_max(&neighbor);
                    if (neighbor_scale > frequency_neighbor_scale)
                        frequency_neighbor_scale = neighbor_scale;
                }
            }
        }
        status = boundary_value(input, k, p, 0, &current[0]);
        if (status != LCMF_OK) goto done;
        current_lower[0] = current[0];
        current_upper[0] = current[0];
        if (fabs(current[0]) > ray_global_scale) ray_global_scale = fabs(current[0]);
        for (node = 1u; node < path.count; ++node) {
            const double zu = path.z[node - 1u];
            const double zd = path.z[node];
            const double ds = zd - zu;
            const double midpoint_z = 0.5 * (zu + zd);
            const double midpoint_r = sqrt(p * p + midpoint_z * midpoint_z);
            const double h_loc = local_radial_scale(input, midpoint_r);
            size_t sub, n_sub;
            if (path.has_core && node == path.outward_start) {
                status = boundary_value(input, k, p, 1, &current[node]);
                if (status != LCMF_OK) goto done;
                current_lower[node] = current[node];
                current_upper[node] = current[node];
                continue;
            }
            if (!isfinite(ds) || !(ds > 0.0) || !isfinite(h_loc) || !(h_loc > 0.0) ||
                ds / h_loc > (double)SIZE_MAX) {
                set_error(error, LCMF_EGRID, "invalid advection SC subcycling scale",
                          evaluation_index, k, ds);
                error->ray_index = ray_index;
                error->segment_index = node - 1u;
                status = LCMF_EGRID;
                goto done;
            }
            n_sub = (size_t)ceil(ds / h_loc);
            if (n_sub < 1u) n_sub = 1u;
            current[node] = current[node - 1u];
            current_lower[node] = current_lower[node - 1u];
            current_upper[node] = current_upper[node - 1u];
            for (sub = 0u; sub < n_sub; ++sub) {
                const double fu = (double)sub / (double)n_sub;
                const double fd = (double)(sub + 1u) / (double)n_sub;
                const double sub_zu = zu + ds * fu;
                const double sub_zd = zu + ds * fd;
                const double sub_ds = sub_zd - sub_zu;
                const double ru = sqrt(p * p + sub_zu * sub_zu);
                const double rd = sqrt(p * p + sub_zd * sub_zd);
                double chi_u, chi_d, physical_eta_u, physical_eta_d, residual;
                LCMFInterval eta_u, eta_d, eta_mid, coefficient0, coefficient1;
                LCMFInterval coefficient2, previous_u, previous_d;
                LCMFInterval previous2_u = point_interval(0.0);
                LCMFInterval previous2_d = point_interval(0.0);
                LCMFInterval incoming, next;
                double local_solution_scale = 0.0, h_guard = 0.0;
                int quadratic_source = 0;
                status = radial_value(input, input->chi_total, ru, k, "chi", &chi_u, error);
                if (status == LCMF_OK) {
                    status = radial_value(input, input->chi_total, rd, k, "chi", &chi_d, error);
                }
                if (status == LCMF_OK) {
                    status = radial_value(input, eta_total, ru, k, "eta", &physical_eta_u, error);
                }
                if (status == LCMF_OK) {
                    status = radial_value(input, eta_total, rd, k, "eta", &physical_eta_d, error);
                }
                if (status != LCMF_OK) {
                    error->ray_index = ray_index;
                    error->segment_index = node - 1u;
                    error->substep_index = sub;
                    goto done;
                }
                eta_u = point_interval(physical_eta_u);
                eta_d = point_interval(physical_eta_d);
                eta_mid = point_interval(0.0);
                coefficient0 = point_interval(0.0);
                coefficient1 = point_interval(0.0);
                coefficient2 = point_interval(0.0);
                previous_u = point_interval(0.0);
                previous_d = point_interval(0.0);
                if (k == 1u) {
                    const double *previous = planes + (k - 1u) * path.count;
                    const double *previous_lower = plane_lower + (k - 1u) * path.count;
                    const double *previous_upper = plane_upper + (k - 1u) * path.count;
                    previous_u = interpolate_history(previous, previous_lower, previous_upper,
                                                     node - 1u, node, fu);
                    previous_d = interpolate_history(previous, previous_lower, previous_upper,
                                                     node - 1u, node, fd);
                    chi_u += 3.0 * a + 2.0 * a / dx;
                    chi_d += 3.0 * a + 2.0 * a / dx;
                    eta_u = bdf_eta_interval(physical_eta_u, 2.0 * a / dx - 3.0 * a,
                                             &previous_u, &previous2_u, 0);
                    eta_d = bdf_eta_interval(physical_eta_d, 2.0 * a / dx - 3.0 * a,
                                             &previous_d, &previous2_d, 0);
                } else if (k >= 2u) {
                    const double *previous = planes + (k - 1u) * path.count;
                    const double *previous2 = planes + (k - 2u) * path.count;
                    const double *previous_lower = plane_lower + (k - 1u) * path.count;
                    const double *previous_upper = plane_upper + (k - 1u) * path.count;
                    const double *previous2_lower = plane_lower + (k - 2u) * path.count;
                    const double *previous2_upper = plane_upper + (k - 2u) * path.count;
                    size_t stencil[3], q;
                    double stencil_z[3];
                    LCMFInterval history1[3], history2[3], effective_eta[3];
                    status = branch_quadratic_stencil(&path, node, stencil);
                    if (status != LCMF_OK) {
                        set_error(error, status, "quadratic history stencil crosses branch reset",
                                  evaluation_index, k, sub_zu);
                        error->ray_index = ray_index;
                        error->segment_index = node - 1u;
                        error->substep_index = sub;
                        goto done;
                    }
                    for (q = 0u; q < 3u; ++q) {
                        double physical_eta;
                        const size_t history_index = stencil[q];
                        const double radius = sqrt(p * p +
                                                   path.z[history_index] * path.z[history_index]);
                        stencil_z[q] = path.z[history_index];
                        status = radial_value(input, eta_total, radius, k, "eta",
                                              &physical_eta, error);
                        if (status != LCMF_OK) {
                            error->ray_index = ray_index;
                            error->segment_index = node - 1u;
                            error->substep_index = sub;
                            goto done;
                        }
                        history1[q].value = previous[history_index];
                        history1[q].lower = previous_lower[history_index];
                        history1[q].upper = previous_upper[history_index];
                        history2[q].value = previous2[history_index];
                        history2[q].lower = previous2_lower[history_index];
                        history2[q].upper = previous2_upper[history_index];
                        effective_eta[q] = bdf_eta_interval(physical_eta, a / dx,
                                                            &history1[q], &history2[q], 1);
                    }
                    previous_u = interpolate_quadratic_interval(stencil_z, history1, sub_zu);
                    previous_d = interpolate_quadratic_interval(stencil_z, history1, sub_zd);
                    previous2_u = interpolate_quadratic_interval(stencil_z, history2, sub_zu);
                    previous2_d = interpolate_quadratic_interval(stencil_z, history2, sub_zd);
                    eta_u = interpolate_quadratic_interval(stencil_z, effective_eta, sub_zu);
                    eta_mid = interpolate_quadratic_interval(
                        stencil_z, effective_eta, 0.5 * (sub_zu + sub_zd));
                    eta_d = interpolate_quadratic_interval(stencil_z, effective_eta, sub_zd);
                    coefficient0 = point_interval(eta_u.value);
                    coefficient1 = point_interval(-3.0 * eta_u.value +
                                                  4.0 * eta_mid.value - eta_d.value);
                    coefficient2 = point_interval(2.0 * eta_u.value -
                                                  4.0 * eta_mid.value +
                                                  2.0 * eta_d.value);
                    quadratic_source = 1;
                    chi_u += 3.0 * a + 1.5 * a / dx;
                    chi_d += 3.0 * a + 1.5 * a / dx;
                }
                if (!valid_interval(&eta_u) || !valid_interval(&eta_d) ||
                    (quadratic_source &&
                     (!valid_interval(&eta_mid) || !valid_interval(&coefficient0) ||
                      !valid_interval(&coefficient1) || !valid_interval(&coefficient2))) ||
                    !isfinite(chi_u) || !isfinite(chi_d)) {
                    ++result->nonfinite_count;
                    set_error(error, LCMF_ENONFINITE,
                              "non-finite BDF effective coefficient",
                              evaluation_index, k,
                              !isfinite(eta_u.value) ? eta_u.value : eta_d.value);
                    error->ray_index = ray_index;
                    error->segment_index = node - 1u;
                    error->substep_index = sub;
                    status = LCMF_ENONFINITE;
                    goto done;
                }
                if (k >= 2u && eta_u.value < 0.0) {
                    record_bdf_eta_negative(result, evaluation_index, k, ray_index,
                                            node - 1u, sub, 0u, &eta_u,
                                            &previous_u, &previous2_u);
                    plane_eta_negative = 1;
                }
                if (k >= 2u && eta_d.value < 0.0) {
                    record_bdf_eta_negative(result, evaluation_index, k, ray_index,
                                            node - 1u, sub, 1u, &eta_d,
                                            &previous_d, &previous2_d);
                    plane_eta_negative = 1;
                }
                incoming.value = current[node];
                incoming.lower = current_lower[node];
                incoming.upper = current_upper[node];
                if (quadratic_source) {
                    size_t stencil[3], q;
                    double stencil_z[3];
                    LCMFInterval effective_eta[3];
                    const double *previous = planes + (k - 1u) * path.count;
                    const double *previous2 = planes + (k - 2u) * path.count;
                    const double *previous_lower = plane_lower + (k - 1u) * path.count;
                    const double *previous_upper = plane_upper + (k - 1u) * path.count;
                    const double *previous2_lower = plane_lower + (k - 2u) * path.count;
                    const double *previous2_upper = plane_upper + (k - 2u) * path.count;
                    status = branch_quadratic_stencil(&path, node, stencil);
                    if (status != LCMF_OK) goto done;
                    for (q = 0u; q < 3u; ++q) {
                        double physical_eta;
                        LCMFInterval history1, history2;
                        const size_t history_index = stencil[q];
                        const double radius = sqrt(p * p +
                                                   path.z[history_index] * path.z[history_index]);
                        stencil_z[q] = path.z[history_index];
                        status = radial_value(input, eta_total, radius, k, "eta",
                                              &physical_eta, error);
                        if (status != LCMF_OK) goto done;
                        history1.value = previous[history_index];
                        history1.lower = previous_lower[history_index];
                        history1.upper = previous_upper[history_index];
                        history2.value = previous2[history_index];
                        history2.lower = previous2_lower[history_index];
                        history2.upper = previous2_upper[history_index];
                        effective_eta[q] = bdf_eta_interval(physical_eta, a / dx,
                                                            &history1, &history2, 1);
                    }
                    status = sc_quadratic_nodal_signed(
                        &incoming, stencil_z, effective_eta, sub_zu, sub_zd,
                        0.5 * (chi_u + chi_d) * sub_ds, &next);
                } else {
                    status = sc_step_signed_eta(&incoming, chi_u, chi_d,
                                                &eta_u, &eta_d, sub_ds, &next);
                }
                if (status != LCMF_OK) {
                    if (status == LCMF_ENONFINITE) ++result->nonfinite_count;
                    set_error(error, status, "signed advection SC substep failed",
                              evaluation_index, k, current[node]);
                    error->ray_index = ray_index;
                    error->segment_index = node - 1u;
                    error->substep_index = sub;
                    goto done;
                }
                if (k > 0u) {
                    local_solution_scale = frequency_neighbor_scale;
                    if (interval_abs_max(&incoming) > local_solution_scale)
                        local_solution_scale = interval_abs_max(&incoming);
                    if (interval_abs_max(&previous_u) > local_solution_scale)
                        local_solution_scale = interval_abs_max(&previous_u);
                    if (interval_abs_max(&previous_d) > local_solution_scale)
                        local_solution_scale = interval_abs_max(&previous_d);
                    if (interval_abs_max(&previous2_u) > local_solution_scale)
                        local_solution_scale = interval_abs_max(&previous2_u);
                    if (interval_abs_max(&previous2_d) > local_solution_scale)
                        local_solution_scale = interval_abs_max(&previous2_d);
                    h_guard = h_loc / radial_span;
                    if (dx / frequency_span > h_guard) h_guard = dx / frequency_span;
                }
                if (!isfinite(next.lower) || !isfinite(next.upper)) {
                    if (!isfinite(next.value) || k == 0u) {
                        ++result->nonfinite_count;
                        set_error(error, LCMF_ENONFINITE,
                                  "non-finite solution or seed-frequency enclosure",
                                  evaluation_index, k, next.value);
                        error->ray_index = ray_index;
                        error->segment_index = node - 1u;
                        error->substep_index = sub;
                        status = LCMF_ENONFINITE;
                        goto done;
                    }
                    ++result->solution_roundoff_enclosure_restart_count;
                    next.lower = nextafter(next.value, -INFINITY);
                    next.upper = nextafter(next.value, INFINITY);
                    if (!isfinite(next.lower) || !isfinite(next.upper)) {
                        ++result->nonfinite_count;
                        set_error(error, LCMF_ENONFINITE,
                                  "finite solution has no finite one-ulp enclosure",
                                  evaluation_index, k, next.value);
                        error->ray_index = ray_index;
                        error->segment_index = node - 1u;
                        error->substep_index = sub;
                        status = LCMF_ENONFINITE;
                        goto done;
                    }
                }
                if (k > 0u) {
                    status = record_solution_guard(result, evaluation_index, k, ray_index,
                                                   node - 1u, sub, &next, h_guard,
                                                   local_solution_scale, ray_global_scale);
                    if (status != LCMF_OK) goto done;
                    if (quadratic_source) {
                        status = segment_residual_quadratic(
                            current[node], 0.5 * (chi_u + chi_d), &coefficient0,
                            &coefficient1, &coefficient2, sub_ds, &residual);
                    } else {
                        status = segment_residual_signed(current[node], chi_u, chi_d,
                                                         eta_u.value, eta_d.value,
                                                         sub_ds, &residual);
                    }
                } else {
                    status = segment_residual(current[node], chi_u, chi_d,
                                              eta_u.value, eta_d.value,
                                              sub_ds, &residual);
                }
                if (status != LCMF_OK) goto done;
                if (residual > *max_residual) *max_residual = residual;
                current[node] = next.value;
                current_lower[node] = next.lower;
                current_upper[node] = next.upper;
                if (fabs(next.value) > ray_global_scale) ray_global_scale = fabs(next.value);
            }
        }
        if (plane_eta_negative) ++result->bdf_eta_negative_plane_count;
        minus[k] = current[path.minus_target];
        plus[k] = current[path.plus_target];
        if (plus[k] < result->solution_min) result->solution_min = plus[k];
    }
done:
    free(planes);
    free(plane_lower);
    free(plane_upper);
    path_free(&path);
    return status;
}

static int allocate_result(const LCMFInput *input, const LCMFOptions *options,
                           LCMFResult *result)
{
    size_t cells, intensities;
    const size_t n_r_eval = options->r_eval == NULL ? input->nr : options->n_r_eval;
    if (checked_product(n_r_eval,input->nnu,&cells)!=LCMF_OK) return LCMF_ENOMEM;
    result->J=(double*)checked_calloc(cells,sizeof(double));
    if (result->J==NULL) return LCMF_ENOMEM;
    if (options->compute_hk) {
        result->H=(double*)checked_calloc(cells,sizeof(double));
        result->K=(double*)checked_calloc(cells,sizeof(double));
        if (result->H==NULL || result->K==NULL) return LCMF_ENOMEM;
    }
    if (options->store_intensity) {
        if (checked_product(cells,options->n_mu,&intensities)!=LCMF_OK) return LCMF_ENOMEM;
        result->I_minus=(double*)checked_calloc(intensities,sizeof(double));
        result->I_plus=(double*)checked_calloc(intensities,sizeof(double));
        if (result->I_minus==NULL || result->I_plus==NULL) return LCMF_ENOMEM;
    }
    result->nr=n_r_eval; result->nnu=input->nnu; result->n_mu=options->n_mu;
    return LCMF_OK;
}

static void release_result_arrays(LCMFResult *result)
{
    free(result->J); result->J = NULL;
    free(result->H); result->H = NULL;
    free(result->K); result->K = NULL;
    free(result->I_minus); result->I_minus = NULL;
    free(result->I_plus); result->I_plus = NULL;
}

static void clear_result_values(LCMFResult *result, const LCMFOptions *options)
{
    size_t cells = result->nr * result->nnu;
    memset(result->J, 0, cells * sizeof(*result->J));
    if (result->H != NULL) memset(result->H, 0, cells * sizeof(*result->H));
    if (result->K != NULL) memset(result->K, 0, cells * sizeof(*result->K));
    if (result->I_minus != NULL) {
        memset(result->I_minus, 0, cells * options->n_mu * sizeof(*result->I_minus));
    }
    if (result->I_plus != NULL) {
        memset(result->I_plus, 0, cells * options->n_mu * sizeof(*result->I_plus));
    }
    result->transport_resid_linf = 0.0;
}

static int transport_sweep(const LCMFInput *input, const LCMFOptions *options,
                           const double *eta_total, LCMFResult *result)
{
    size_t i, m, k;
    const size_t n_r_eval = result->nr;
    int status = LCMF_OK;
    clear_result_values(result, options);
    for (i = 0u; i < n_r_eval; ++i) {
        LCMFRayCache cache;
        const double target_r = options->r_eval == NULL
                              ? 0.5 * (input->r_edge[i] + input->r_edge[i + 1u])
                              : options->r_eval[i];
        status = lumina_cmf_ray_cache_build_at_radius(target_r, options->n_mu,
                                                       &cache, &result->error);
        if (status != LCMF_OK) return status;
        if (options->radial_characteristic) {
            cache.mu[0] = 1.0;
            cache.weight[0] = 1.0;
            cache.p[0] = 0.0;
        }
        for (m = 0u; m < options->n_mu; ++m) {
            double *minus_plane = NULL;
            double *plus_plane = NULL;
            if (options->frequency_advection) {
                minus_plane = (double *)checked_calloc(input->nnu, sizeof(double));
                plus_plane = (double *)checked_calloc(input->nnu, sizeof(double));
                if (minus_plane == NULL || plus_plane == NULL) {
                    free(minus_plane); free(plus_plane);
                    lumina_cmf_ray_cache_free(&cache);
                    return LCMF_ENOMEM;
                }
                status = solve_advection_ray(input, eta_total, i, target_r, cache.p[m],
                                              minus_plane, plus_plane,
                                              &result->transport_resid_linf, result,
                                              i * options->n_mu + m);
                if (status != LCMF_OK) {
                    free(minus_plane); free(plus_plane);
                    lumina_cmf_ray_cache_free(&cache);
                    return status;
                }
            }
            for (k = 0u; k < input->nnu; ++k) {
                double minus, plus;
                const size_t q = i * input->nnu + k;
                const size_t iq = (i * options->n_mu + m) * input->nnu + k;
                if (options->frequency_advection) {
                    minus = minus_plane[k]; plus = plus_plane[k];
                } else {
                    status = solve_static_ray(input, eta_total, i, target_r, cache.p[m], k,
                                              &minus, &plus, &result->transport_resid_linf,
                                              &result->error, i * options->n_mu + m);
                    if (status != LCMF_OK) {
                        if (status == LCMF_ENONFINITE) ++result->nonfinite_count;
                        free(minus_plane); free(plus_plane);
                        lumina_cmf_ray_cache_free(&cache);
                        return status;
                    }
                }
                result->J[q] += 0.5 * cache.weight[m] * (plus + minus);
                if (options->compute_hk) {
                    result->H[q] += 0.5 * cache.weight[m] * cache.mu[m] * (plus - minus);
                    result->K[q] += 0.5 * cache.weight[m] * cache.mu[m] * cache.mu[m] *
                                    (plus + minus);
                }
                if (options->store_intensity) {
                    result->I_minus[iq] = minus;
                    result->I_plus[iq] = plus;
                }
            }
            free(minus_plane); free(plus_plane);
        }
        lumina_cmf_ray_cache_free(&cache);
    }
    return LCMF_OK;
}

static int transport_sweep_static_parallel(const LCMFInput *input,
                                           const LCMFOptions *options,
                                           const double *eta_total,
                                           LCMFResult *result)
{
    const size_t n_r_eval = result->nr;
    int shared_status = LCMF_OK;
    double shared_residual = 0.0;
    ptrdiff_t evaluation;
    clear_result_values(result, options);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(max:shared_residual)
#endif
    for (evaluation = 0; evaluation < (ptrdiff_t)n_r_eval; ++evaluation) {
        const size_t i = (size_t)evaluation;
        const double target_r = options->r_eval == NULL
                              ? 0.5 * (input->r_edge[i] + input->r_edge[i + 1u])
                              : options->r_eval[i];
        LCMFRayCache cache;
        LCMFError local_error;
        double local_residual = 0.0;
        size_t m, k;
        int local_status;
        memset(&cache, 0, sizeof(cache));
        memset(&local_error, 0, sizeof(local_error));
        local_status = lumina_cmf_ray_cache_build_at_radius(target_r, options->n_mu,
                                                             &cache, &local_error);
        for (m = 0u; local_status == LCMF_OK && m < options->n_mu; ++m) {
            for (k = 0u; local_status == LCMF_OK && k < input->nnu; ++k) {
                double minus, plus;
                const size_t q = i * input->nnu + k;
                const size_t iq = (i * options->n_mu + m) * input->nnu + k;
                local_status = solve_static_ray(input, eta_total, i, target_r, cache.p[m], k,
                                                &minus, &plus, &local_residual, &local_error,
                                                i * options->n_mu + m);
                if (local_status == LCMF_OK) {
                    result->J[q] += 0.5 * cache.weight[m] * (plus + minus);
                    if (options->compute_hk) {
                        result->H[q] += 0.5 * cache.weight[m] * cache.mu[m] * (plus - minus);
                        result->K[q] += 0.5 * cache.weight[m] * cache.mu[m] * cache.mu[m] *
                                        (plus + minus);
                    }
                    if (options->store_intensity) {
                        result->I_minus[iq] = minus;
                        result->I_plus[iq] = plus;
                    }
                }
            }
        }
        lumina_cmf_ray_cache_free(&cache);
        if (local_residual > shared_residual) shared_residual = local_residual;
        if (local_status != LCMF_OK) {
#ifdef _OPENMP
#pragma omp critical(stage31_cmf_error)
#endif
            {
                if (shared_status == LCMF_OK) {
                    shared_status = local_status;
                    result->error = local_error;
                }
            }
        }
    }
    result->transport_resid_linf = shared_residual;
    if (shared_status == LCMF_ENONFINITE) ++result->nonfinite_count;
    return shared_status;
}

int lumina_cmf_field_solve(const LCMFInput *input,
                           const LCMFOptions *options,
                           LCMFResult *result)
{
    int status;
    size_t i, cells;
    double *eta_total = NULL;
    double *previous_j = NULL;
    int coherent_active = 0;
    if (result == NULL) return LCMF_EINVAL;
    memset(result, 0, sizeof(*result));
    result->bdf_eta_min = INFINITY;
    result->solution_min = INFINITY;
    result->solution_subtruncation_min = INFINITY;
    status = lumina_cmf_validate_input(input, options, &result->error);
    if (status != LCMF_OK) return status;
    cells = input->nr * input->nnu;
    if (input->scatter_mode == LCMF_SCAT_COHERENT && input->chi_coherent != NULL) {
        for (i = 0u; i < cells; ++i) if (input->chi_coherent[i] > 0.0) coherent_active = 1;
    }
    if (coherent_active && (options->r_eval != NULL || options->frequency_advection)) {
        set_error(&result->error, LCMF_EINVAL,
                  "plain coherent iteration requires shell-center static evaluation", 0, 0, 0.0);
        return LCMF_EINVAL;
    }
    status=allocate_result(input,options,result);
    if (status!=LCMF_OK) { lumina_cmf_result_free(result); return status; }
    if (!coherent_active) {
        status = options->frequency_advection
               ? transport_sweep(input, options, input->eta_fixed, result)
               : transport_sweep_static_parallel(input, options, input->eta_fixed, result);
        result->source_iterations = 1u;
        result->source_resid_linf = 0.0;
    } else {
        size_t iteration;
        eta_total = (double *)checked_calloc(cells, sizeof(*eta_total));
        previous_j = (double *)checked_calloc(cells, sizeof(*previous_j));
        if (eta_total == NULL || previous_j == NULL) {
            free(eta_total); free(previous_j); release_result_arrays(result);
            return LCMF_ENOMEM;
        }
        status = LCMF_ENOCONV;
        for (iteration = 1u; iteration <= options->max_source_iter; ++iteration) {
            double max_delta = 0.0;
            double max_j = 0.0;
            for (i = 0u; i < cells; ++i) {
                eta_total[i] = input->eta_fixed[i] + input->chi_coherent[i] * previous_j[i];
                if (!finite_nonnegative(eta_total[i])) { status = LCMF_ENONFINITE; break; }
            }
            if (status == LCMF_ENONFINITE) break;
            status = transport_sweep_static_parallel(input, options, eta_total, result);
            if (status != LCMF_OK) break;
            for (i = 0u; i < cells; ++i) {
                const double delta = fabs(result->J[i] - previous_j[i]);
                if (delta > max_delta) max_delta = delta;
                if (fabs(result->J[i]) > max_j) max_j = fabs(result->J[i]);
            }
            result->source_resid_linf = max_delta / (max_j + DBL_MIN);
            result->source_iterations = iteration;
            if (result->source_resid_linf <= options->source_rtol) {
                status = LCMF_OK;
                break;
            }
            memcpy(previous_j, result->J, cells * sizeof(*previous_j));
            status = LCMF_ENOCONV;
        }
        free(previous_j); free(eta_total);
        if (status == LCMF_ENOCONV) {
            set_error(&result->error, status, "plain coherent source iteration did not converge",
                      0, 0, result->source_resid_linf);
        }
    }
    if (status != LCMF_OK) return status;
    if (result->nonfinite_count > 0u) return LCMF_ENONFINITE;
    if (result->solution_negative_excess_count > 0u) return LCMF_ENEGATIVE;
    if (result->sign_uncertain_count > 0u) return LCMF_ESIGNUNCERTAIN;
    return LCMF_OK;
}

int lumina_cmf_field_residual(const LCMFInput *input,
                              const LCMFResult *result, double *linf)
{
    if (input == NULL || result == NULL || linf == NULL) return LCMF_EINVAL;
    if (!isfinite(result->transport_resid_linf) || result->transport_resid_linf < 0.0)
        return LCMF_EINVAL;
    *linf = result->transport_resid_linf;
    return LCMF_OK;
}

void lumina_cmf_result_free(LCMFResult *result)
{
    if (result == NULL) return;
    release_result_arrays(result);
    memset(result, 0, sizeof(*result));
}

static uint32_t rotr32(uint32_t x, unsigned n)
{
    return (x >> n) | (x << (32u - n));
}

static void sha256_transform(LCMFSha256 *ctx, const unsigned char block[64])
{
    static const uint32_t constant[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t words[64], a,b,c,d,e,f,g,h;
    size_t i;
    for (i = 0; i < 16u; ++i) {
        words[i] = ((uint32_t)block[4u*i] << 24) | ((uint32_t)block[4u*i+1u] << 16)
                 | ((uint32_t)block[4u*i+2u] << 8) | (uint32_t)block[4u*i+3u];
    }
    for (i = 16u; i < 64u; ++i) {
        const uint32_t s0 = rotr32(words[i-15u],7)^rotr32(words[i-15u],18)^(words[i-15u]>>3);
        const uint32_t s1 = rotr32(words[i-2u],17)^rotr32(words[i-2u],19)^(words[i-2u]>>10);
        words[i] = words[i-16u] + s0 + words[i-7u] + s1;
    }
    a=ctx->h[0]; b=ctx->h[1]; c=ctx->h[2]; d=ctx->h[3];
    e=ctx->h[4]; f=ctx->h[5]; g=ctx->h[6]; h=ctx->h[7];
    for (i = 0; i < 64u; ++i) {
        const uint32_t s1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
        const uint32_t ch=(e&f)^((~e)&g);
        const uint32_t t1=h+s1+ch+constant[i]+words[i];
        const uint32_t s0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
        const uint32_t maj=(a&b)^(a&c)^(b&c);
        const uint32_t t2=s0+maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    ctx->h[0]+=a; ctx->h[1]+=b; ctx->h[2]+=c; ctx->h[3]+=d;
    ctx->h[4]+=e; ctx->h[5]+=f; ctx->h[6]+=g; ctx->h[7]+=h;
}

static void sha256_init(LCMFSha256 *ctx)
{
    static const uint32_t initial[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                                        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    memcpy(ctx->h, initial, sizeof(initial));
    ctx->bits = 0; ctx->used = 0;
}

static void sha256_update(LCMFSha256 *ctx, const unsigned char *data, size_t length)
{
    while (length > 0) {
        size_t take = 64u - ctx->used;
        if (take > length) take = length;
        memcpy(ctx->block + ctx->used, data, take);
        ctx->used += take; data += take; length -= take;
        ctx->bits += (uint64_t)take * 8u;
        if (ctx->used == 64u) { sha256_transform(ctx, ctx->block); ctx->used = 0; }
    }
}

static void sha256_final(LCMFSha256 *ctx, unsigned char digest[32])
{
    size_t i;
    const uint64_t bits = ctx->bits;
    ctx->block[ctx->used++] = 0x80u;
    if (ctx->used > 56u) {
        while (ctx->used < 64u) ctx->block[ctx->used++] = 0;
        sha256_transform(ctx, ctx->block); ctx->used = 0;
    }
    while (ctx->used < 56u) ctx->block[ctx->used++] = 0;
    for (i = 0; i < 8u; ++i) ctx->block[63u-i] = (unsigned char)(bits >> (8u*i));
    sha256_transform(ctx, ctx->block);
    for (i = 0; i < 8u; ++i) {
        digest[4u*i]=(unsigned char)(ctx->h[i]>>24); digest[4u*i+1u]=(unsigned char)(ctx->h[i]>>16);
        digest[4u*i+2u]=(unsigned char)(ctx->h[i]>>8); digest[4u*i+3u]=(unsigned char)ctx->h[i];
    }
}

static uint32_t load_u32_le(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}

static uint64_t load_u64_le(const unsigned char *p)
{
    uint64_t value = 0; size_t i;
    for (i = 0; i < 8u; ++i) value |= (uint64_t)p[i] << (8u*i);
    return value;
}

static double load_f64_le(const unsigned char *p)
{
    const uint64_t bits = load_u64_le(p); double value;
    memcpy(&value, &bits, sizeof(value)); return value;
}

static int read_exact(FILE *stream, unsigned char *data, size_t length, LCMFSha256 *sha)
{
    if (length > 0 && fread(data, 1, length, stream) != length) return LCMF_EIO;
    if (sha != NULL) sha256_update(sha, data, length);
    return LCMF_OK;
}

static int read_f64_array(FILE *stream, double *values, size_t count, LCMFSha256 *sha)
{
    unsigned char bytes[8]; size_t i;
    for (i = 0; i < count; ++i) {
        if (read_exact(stream, bytes, 8u, sha) != LCMF_OK) return LCMF_EIO;
        values[i] = load_f64_le(bytes);
    }
    return LCMF_OK;
}

static int hex_value(int c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int read_manifest_digest(const char *path, unsigned char digest[32])
{
    FILE *stream; char line[256]; char *hex; size_t i;
    stream = fopen(path, "rb"); if (stream == NULL) return LCMF_EIO;
    if (fgets(line, sizeof(line), stream) == NULL) { fclose(stream); return LCMF_ESCHEMA; }
    fclose(stream);
    hex = strstr(line, "sha256=");
    if (hex == NULL) return LCMF_ESCHEMA;
    hex += 7;
    for (i = 0; i < 32u; ++i) {
        const int hi = hex_value((unsigned char)hex[2u*i]);
        const int lo = hex_value((unsigned char)hex[2u*i+1u]);
        if (hi < 0 || lo < 0) return LCMF_ESCHEMA;
        digest[i] = (unsigned char)((hi << 4) | lo);
    }
    return LCMF_OK;
}

void lumina_cmf_frozen_free(LCMFFrozenField *field)
{
    if (field == NULL) return;
    free(field->r_edge); free(field->nu); free(field->dnu);
    free(field->chi_total); free(field->chi_coherent); free(field->eta_fixed);
    free(field->eta_coherent); free(field->eta_total); free(field->J_producer);
    memset(field, 0, sizeof(*field));
}

int lumina_cmf_frozen_load(const char *binary_path, const char *manifest_path,
                           LCMFFrozenField *field, LCMFError *error)
{
    FILE *stream = NULL; unsigned char header[64], expected[32], actual[32], extra;
    LCMFSha256 sha; size_t cells, nr, nnu; int status = LCMF_OK;
    if (binary_path == NULL || manifest_path == NULL || field == NULL) return LCMF_EINVAL;
    memset(field, 0, sizeof(*field)); sha256_init(&sha);
    status = read_manifest_digest(manifest_path, expected);
    if (status != LCMF_OK) { set_error(error,status,"invalid checksum manifest",0,0,0.0); return status; }
    stream = fopen(binary_path, "rb");
    if (stream == NULL) { set_error(error,LCMF_EIO,strerror(errno),0,0,0.0); return LCMF_EIO; }
    status = read_exact(stream, header, sizeof(header), &sha);
    if (status != LCMF_OK || memcmp(header,"LCMFCE01",8u)!=0 ||
        load_u32_le(header+8)!=LCMF_FROZEN_ENDIAN || load_u32_le(header+12)!=LCMF_FROZEN_VERSION ||
        load_u32_le(header+52)!=0u) { status=LCMF_ESCHEMA; goto fail; }
    field->nr=load_u64_le(header+16); field->nnu=load_u64_le(header+24);
    field->iteration=load_u64_le(header+32); field->field_generation=load_u64_le(header+40);
    field->flags=load_u32_le(header+48); field->t_exp_s=load_f64_le(header+56);
    if (field->nr==0 || field->nnu==0 || field->nr>SIZE_MAX || field->nnu>SIZE_MAX ||
        !isfinite(field->t_exp_s) || field->t_exp_s<=0.0 ||
        !(field->flags & LCMF_FROZEN_FREQUENCY_DESCENDING)) { status=LCMF_ESCHEMA; goto fail; }
    nr=(size_t)field->nr; nnu=(size_t)field->nnu;
    if (checked_product(nr,nnu,&cells)!=LCMF_OK) { status=LCMF_ENOMEM; goto fail; }
#define ALLOC_FIELD(name,count) do { field->name=(double*)checked_calloc((count),sizeof(double)); if(field->name==NULL){status=LCMF_ENOMEM;goto fail;} } while(0)
    ALLOC_FIELD(r_edge,nr+1u); ALLOC_FIELD(nu,nnu); ALLOC_FIELD(dnu,nnu);
    ALLOC_FIELD(chi_total,cells); ALLOC_FIELD(chi_coherent,cells); ALLOC_FIELD(eta_fixed,cells);
    ALLOC_FIELD(eta_coherent,cells); ALLOC_FIELD(eta_total,cells); ALLOC_FIELD(J_producer,cells);
#undef ALLOC_FIELD
#define READ_FIELD(name,count) do { if(read_f64_array(stream,field->name,(count),&sha)!=LCMF_OK){status=LCMF_EIO;goto fail;} } while(0)
    READ_FIELD(r_edge,nr+1u); READ_FIELD(nu,nnu); READ_FIELD(dnu,nnu);
    READ_FIELD(chi_total,cells); READ_FIELD(chi_coherent,cells); READ_FIELD(eta_fixed,cells);
    READ_FIELD(eta_coherent,cells); READ_FIELD(eta_total,cells); READ_FIELD(J_producer,cells);
#undef READ_FIELD
    if (fread(&extra,1,1,stream)!=0 || ferror(stream)) { status=LCMF_ESCHEMA; goto fail; }
    fclose(stream); stream=NULL; sha256_final(&sha,actual);
    if (memcmp(expected,actual,32u)!=0) { status=LCMF_ECHECKSUM; goto fail; }
    for (size_t i=0;i<nr;++i) {
        if (!isfinite(field->r_edge[i]) || field->r_edge[i+1u]<=field->r_edge[i]) { status=LCMF_ESCHEMA; goto fail; }
    }
    for (size_t k=0;k<nnu;++k) {
        if (!isfinite(field->nu[k]) || field->nu[k]<=0.0 || !isfinite(field->dnu[k]) || field->dnu[k]<=0.0 ||
            (k>0 && field->nu[k]>=field->nu[k-1u])) { status=LCMF_ESCHEMA; goto fail; }
    }
    for (size_t q=0;q<cells;++q) {
        if (!finite_nonnegative(field->chi_total[q]) || !finite_nonnegative(field->chi_coherent[q]) ||
            field->chi_coherent[q]>field->chi_total[q] || !finite_nonnegative(field->eta_fixed[q]) ||
            !finite_nonnegative(field->eta_coherent[q]) || !finite_nonnegative(field->eta_total[q]) ||
            !finite_nonnegative(field->J_producer[q]) ||
            field->eta_total[q] != field->eta_fixed[q] + field->eta_coherent[q]) { status=LCMF_ESCHEMA; goto fail; }
    }
    return LCMF_OK;
fail:
    if (stream != NULL) fclose(stream);
    lumina_cmf_frozen_free(field);
    set_error(error,status,"frozen field failed closed validation",0,0,0.0);
    return status;
}

const char *lumina_cmf_status_string(int status)
{
    switch (status) {
    case LCMF_OK:return "LCMF_OK"; case LCMF_EINVAL:return "LCMF_EINVAL";
    case LCMF_ENOMEM:return "LCMF_ENOMEM"; case LCMF_EGRID:return "LCMF_EGRID";
    case LCMF_ENEGATIVE:return "LCMF_ENEGATIVE"; case LCMF_ENONFINITE:return "LCMF_ENONFINITE";
    case LCMF_EUNSUPPORTED:return "LCMF_EUNSUPPORTED"; case LCMF_ENOCONV:return "LCMF_ENOCONV";
    case LCMF_EIO:return "LCMF_EIO"; case LCMF_ESCHEMA:return "LCMF_ESCHEMA";
    case LCMF_ECHECKSUM:return "LCMF_ECHECKSUM"; case LCMF_EHOMOLOGY:return "LCMF_EHOMOLOGY";
    case LCMF_ESIGNUNCERTAIN:return "LCMF_ESIGNUNCERTAIN";
    default:return "LCMF_UNKNOWN";
    }
}
