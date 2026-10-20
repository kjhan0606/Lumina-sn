#include "cmf_exact_sliding.h"
#include "cmf_error_envelope.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CMF_EXACT_C  2.99792458e10
#define CMF_EXACT_H  6.62607015e-27
#define CMF_EXACT_KB 1.380649e-16

typedef struct {
    int ns, nb, nr;
    size_t ray_stride;
    size_t segment_slots;
    double dlognu;
    double a_lam;
    double a_drift;
    const double *nu;
    const double *chi_tot;
    const double *chi_es;
    const double *S_fixed;
    double *rmid;
    double *p;
    int *rn;
    int *rsh;
    double *rz;
    int *rcore;
    double *rzin;
    int *shell_off;
    int *shell_k;
    int *shell_seg;
    double *shell_mu;
    int nsamp;
    int physical_cells;
    size_t inner_stride;
    const CMFExactInnerBoundary *boundary;
    double *half_outer;
    double *half_inner;
    double *core_mu;
    int raw_emissivity;
} ExactProblem;

static int exact_cell_streams(const ExactProblem *p)
{
#ifdef CMF_EXACT_TEST_FULL_CELL_BUFFER
    (void)p;
    return 0; /* 작은 직접시험의 기존 전체 버퍼 대조군 전용. */
#else
    return p->physical_cells;
#endif
}

static size_t exact_buffer_rays(const ExactProblem *p)
{
    if(!exact_cell_streams(p))return (size_t)p->nr;
    /* 두 방향 광선 버퍼 합8GiB 이내, 동시에 최대16광선. 한 광선조차
     * 한도를 넘으면 최소 한 광선이 필요하며 실제 할당 실패를 숨기지 않는다. */
    size_t limit=(size_t)8*1024*1024*1024/sizeof(double)/2/
                 (size_t)p->nb/p->ray_stride;
    if(limit<1)limit=1;
    if(limit>16)limit=16;
    return limit<(size_t)p->nr ? limit : (size_t)p->nr;
}

static double exact_planck(double nu, double temperature)
{
    if (!(nu > 0.0) || !(temperature > 0.0) ||
        !isfinite(nu) || !isfinite(temperature)) return NAN;
    double x = CMF_EXACT_H * nu / (CMF_EXACT_KB * temperature);
    double prefactor = 2.0 * CMF_EXACT_H * nu * nu * nu /
                       (CMF_EXACT_C * CMF_EXACT_C);
    double value;
    if (!(x > 0.0) || !isfinite(x) || !isfinite(prefactor)) return NAN;
    if (x > 50.0) {
        /* Algebraically identical Wien representation.  A returned zero can
         * only be IEEE underflow of the represented physical value; there is
         * no imposed x-threshold floor. */
        double e = exp(-x);
        value = prefactor * e / (-expm1(-x));
    } else {
        value = prefactor / expm1(x);
    }
    return value >= 0.0 && isfinite(value) ? value : NAN;
}

CMFExactStatus cmf_exact_diffusion_gradient(
    int n_bins, const double *nu, const double *dnu,
    double temperature, double luminosity, double radius,
    const double *chi, double *gradient_over_chi)
{
    if (n_bins < 2 || !nu || !dnu || !chi || !gradient_over_chi ||
        !(temperature > 0.0) || !isfinite(temperature) ||
        !(luminosity >= 0.0) || !isfinite(luminosity) ||
        !(radius > 0.0) || !isfinite(radius) ||
        (size_t)n_bins > SIZE_MAX / sizeof(double))
        return CMF_EXACT_INVALID_INPUT;
    double *weight = malloc((size_t)n_bins * sizeof(double));
    if (!weight) return CMF_EXACT_ALLOCATION_FAILED;
    long double integral = 0.0L;
    CMFExactStatus status = CMF_EXACT_INVALID_INPUT;
    for (int b = 0; b < n_bins; ++b) {
        if (!(nu[b] > 0.0) || !isfinite(nu[b]) ||
            (b > 0 && !(nu[b] > nu[b-1])) ||
            !(dnu[b] > 0.0) || !isfinite(dnu[b]) ||
            !(chi[b] > 0.0) || !isfinite(chi[b])) goto done;
        double x = CMF_EXACT_H * nu[b] / (CMF_EXACT_KB * temperature);
        double derivative = exact_planck(nu[b], temperature) *
                            (x / temperature) / (-expm1(-x));
        weight[b] = derivative / chi[b];
        if (!(weight[b] >= 0.0) || !isfinite(weight[b])) goto done;
        integral += (long double)weight[b] * dnu[b];
    }
    if (!(integral > 0.0L) || !isfinite(integral)) goto done;
    const long double pi = acosl(-1.0L);
    long double gradient = 3.0L * luminosity /
        (16.0L * pi * pi * (long double)radius * radius * integral);
    if (!(gradient >= 0.0L) || !isfinite(gradient)) goto done;
    for (int b = 0; b < n_bins; ++b) {
        weight[b] = (double)(gradient * weight[b]);
        if (!(weight[b] >= 0.0) || !isfinite(weight[b])) goto done;
    }
    memcpy(gradient_over_chi, weight, (size_t)n_bins * sizeof(double));
    status = CMF_EXACT_OK;
done:
    free(weight);
    return status;
}

const char *cmf_exact_status_name(CMFExactStatus status)
{
    switch (status) {
    case CMF_EXACT_OK: return "OK";
    case CMF_EXACT_INVALID_INPUT: return "INVALID_INPUT";
    case CMF_EXACT_ALLOCATION_FAILED: return "ALLOCATION_FAILED";
    case CMF_EXACT_NONFINITE: return "NONFINITE";
    case CMF_EXACT_NEGATIVE_RECURRENCE: return "NEGATIVE_RECURRENCE";
    case CMF_EXACT_NOT_CONVERGED: return "NOT_CONVERGED";
    case CMF_EXACT_ERROR_ENVELOPE_FAILED:
        return "ERROR_ENVELOPE_FAILED";
    default: return "UNKNOWN";
    }
}

static int checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (!out || (a != 0 && b > SIZE_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static void exact_problem_free(ExactProblem *p)
{
    if (!p) return;
    free(p->rmid); free(p->p); free(p->rn); free(p->rsh); free(p->rz);
    free(p->rcore); free(p->rzin); free(p->shell_off); free(p->shell_k);
    free(p->shell_seg); free(p->shell_mu);
    free(p->half_outer); free(p->half_inner); free(p->core_mu);
    memset(p, 0, sizeof(*p));
}

static CMFExactStatus exact_problem_build(
    ExactProblem *p, int ns, int nb, double dlognu,
    const double *nu, const double *r_inner, const double *r_outer,
    double time_explosion, const double *chi_tot, const double *chi_es,
    const double *S_fixed, int physical_cells,
    const CMFExactInnerBoundary *boundary)
{
    const int ncore = 16;
    if (!p || ns <= 0 || ns > INT_MAX-ncore-1 || nb < 2 || !(dlognu > 0.0) ||
        !isfinite(dlognu) || !nu || !r_inner || !r_outer ||
        !(time_explosion > 0.0) || !isfinite(time_explosion) ||
        !chi_tot || !chi_es || !S_fixed) return CMF_EXACT_INVALID_INPUT;
    memset(p, 0, sizeof(*p));
    p->ns = ns; p->nb = nb; p->nr = ns + ncore + (physical_cells ? 1 : 0);
    p->ray_stride = (size_t)ns + 1U;
    p->dlognu = dlognu;
    p->a_lam = 1.0 / (time_explosion * CMF_EXACT_C);
    p->a_drift = p->a_lam / dlognu;
    p->nu = nu; p->chi_tot = chi_tot; p->chi_es = chi_es;
    p->S_fixed = S_fixed;
    p->physical_cells = physical_cells;
    p->inner_stride = physical_cells ? (size_t)nb : 0U;
    p->boundary = boundary;
    if (checked_mul_size((size_t)p->nr, p->ray_stride,
                         &p->segment_slots) != 0)
        return CMF_EXACT_INVALID_INPUT;

    for (int b = 0; b < nb; ++b) {
        if (!(nu[b] > 0.0) || !isfinite(nu[b]) ||
            (b > 0 && !(nu[b] > nu[b - 1])) ||
            (physical_cells && b > 0 &&
             fabs(log(nu[b]/nu[b-1])-dlognu) > 64.0*DBL_EPSILON))
            return CMF_EXACT_INVALID_INPUT;
    }
    for (int s = 0; s < ns; ++s) {
        if (!(r_inner[s] > 0.0) || !(r_outer[s] > r_inner[s]) ||
            !isfinite(r_inner[s]) || !isfinite(r_outer[s]) ||
            (s > 0 && (r_inner[s] < r_inner[s - 1] ||
                       r_outer[s] <= r_outer[s - 1])) ||
            (physical_cells && s > 0 && r_inner[s] != r_outer[s-1]))
            return CMF_EXACT_INVALID_INPUT;
    }

    p->rmid = (double *)malloc((size_t)ns * sizeof(double));
    p->p = (double *)malloc((size_t)p->nr * sizeof(double));
    p->rn = (int *)calloc((size_t)p->nr, sizeof(int));
    p->rsh = (int *)malloc(p->segment_slots * sizeof(int));
    p->rz = (double *)malloc(p->segment_slots * sizeof(double));
    p->rcore = (int *)calloc((size_t)p->nr, sizeof(int));
    p->rzin = (double *)calloc((size_t)p->nr, sizeof(double));
    if (physical_cells) {
        p->half_outer = calloc(p->segment_slots, sizeof(double));
        p->half_inner = calloc(p->segment_slots, sizeof(double));
        p->core_mu = calloc((size_t)p->nr, sizeof(double));
    }
    if (!p->rmid || !p->p || !p->rn || !p->rsh || !p->rz ||
        !p->rcore || !p->rzin ||
        (physical_cells && (!p->half_outer || !p->half_inner || !p->core_mu))) {
        exact_problem_free(p);
        return CMF_EXACT_ALLOCATION_FAILED;
    }
    for (int s = 0; s < ns; ++s)
        p->rmid[s] = 0.5 * (r_inner[s] + r_outer[s]);
    for (int k = 0; k < ncore; ++k)
        p->p[k] = (physical_cells ? r_inner[0] : p->rmid[0]) *
                  (double)k / (double)ncore;
    if (physical_cells) p->p[ncore] = r_inner[0];
    for (int s = 0; s < ns; ++s)
        p->p[ncore + (physical_cells ? 1 : 0) + s] = p->rmid[s];

    for (int k = 0; k < p->nr; ++k) {
        double impact = p->p[k];
        int n = 0;
        size_t base = (size_t)k * p->ray_stride;
        for (int s = ns - 1; s >= 0; --s) {
            if ((physical_cells ? r_outer[s] : p->rmid[s]) <= impact) break;
            p->rsh[base + (size_t)n] = s;
            if (physical_cells) {
                double zhi = sqrt((r_outer[s]-impact)*(r_outer[s]+impact));
                double zlo = r_inner[s] > impact ?
                    sqrt((r_inner[s]-impact)*(r_inner[s]+impact)) : 0.0;
                double zm = p->rmid[s] > impact ?
                    sqrt((p->rmid[s]-impact)*(p->rmid[s]+impact)) : 0.0;
                p->rz[base + (size_t)n] = zm;
                p->half_outer[base + (size_t)n] = zhi-zm;
                p->half_inner[base + (size_t)n] = zm-zlo;
            } else {
                p->rz[base + (size_t)n] =
                    sqrt(p->rmid[s] * p->rmid[s] - impact * impact);
            }
            ++n;
        }
        p->rn[k] = n;
        double core_radius = physical_cells ? r_inner[0] : p->rmid[0];
        p->rcore[k] = impact < core_radius;
        p->rzin[k] = p->rcore[k]
            ? sqrt(core_radius * core_radius - impact * impact) : 0.0;
        if (physical_cells) p->core_mu[k] = p->rzin[k]/core_radius;
    }

    int *shell_count = (int *)calloc((size_t)ns, sizeof(int));
    int *fill = (int *)calloc((size_t)ns, sizeof(int));
    p->shell_off = (int *)malloc(((size_t)ns + 1U) * sizeof(int));
    if (!shell_count || !fill || !p->shell_off) {
        free(shell_count); free(fill); exact_problem_free(p);
        return CMF_EXACT_ALLOCATION_FAILED;
    }
    for (int k = 0; k < p->nr; ++k) {
        size_t base = (size_t)k * p->ray_stride;
        for (int i = 0; i < p->rn[k]; ++i) {
            int s = p->rsh[base+i];
            if (!physical_cells || p->p[k] <= p->rmid[s]) ++shell_count[s];
        }
    }
    p->shell_off[0] = 0;
    for (int s = 0; s < ns; ++s)
        p->shell_off[s + 1] = p->shell_off[s] + shell_count[s];
    p->nsamp = p->shell_off[ns];
    p->shell_k = (int *)malloc((size_t)p->nsamp * sizeof(int));
    p->shell_seg = (int *)malloc((size_t)p->nsamp * sizeof(int));
    p->shell_mu = (double *)malloc((size_t)p->nsamp * sizeof(double));
    if (!p->shell_k || !p->shell_seg || !p->shell_mu) {
        free(shell_count); free(fill); exact_problem_free(p);
        return CMF_EXACT_ALLOCATION_FAILED;
    }
    for (int k = 0; k < p->nr; ++k) {
        size_t base = (size_t)k * p->ray_stride;
        for (int i = 0; i < p->rn[k]; ++i) {
            int s = p->rsh[base + i];
            if (physical_cells && p->p[k] > p->rmid[s]) continue;
            int at = p->shell_off[s] + fill[s]++;
            p->shell_k[at] = k;
            p->shell_seg[at] = i;
            p->shell_mu[at] = p->rz[base + i] / p->rmid[s];
        }
    }
    for (int s = 0; s < ns; ++s) {
        int lo = p->shell_off[s], hi = p->shell_off[s + 1];
        for (int a = lo + 1; a < hi; ++a) {
            double mu = p->shell_mu[a];
            int ray = p->shell_k[a], seg = p->shell_seg[a], q = a - 1;
            while (q >= lo && p->shell_mu[q] > mu) {
                p->shell_mu[q + 1] = p->shell_mu[q];
                p->shell_k[q + 1] = p->shell_k[q];
                p->shell_seg[q + 1] = p->shell_seg[q];
                --q;
            }
            p->shell_mu[q + 1] = mu;
            p->shell_k[q + 1] = ray;
            p->shell_seg[q + 1] = seg;
        }
    }
    free(shell_count); free(fill);
    return CMF_EXACT_OK;
}

static int clip_bin(int bin, int nb)
{
    return bin < 0 ? 0 : (bin >= nb ? nb - 1 : bin);
}

typedef struct {
    double transmission;
    double emission;
} PositiveTransform;

typedef enum {
    POSITIVE_ROUND_NEAREST = 0,
    POSITIVE_ROUND_UPPER = 1,
    POSITIVE_ROUND_LOWER = -1
} PositiveRounding;

typedef struct {
    PositiveTransform value;
    PositiveTransform aggregate;
} PositiveWindowNode;

typedef struct {
    PositiveWindowNode *front;
    PositiveWindowNode *back;
    size_t front_size;
    size_t back_size;
    size_t capacity;
    PositiveRounding rounding;
} PositiveWindow;

static int positive_add_bound(double a, double b, PositiveRounding rounding,
                              double *result)
{
    if (!result || !(a >= 0.0) || !(b >= 0.0) ||
        !isfinite(a) || !isfinite(b)) return -1;
    double sum = a + b;
    if (!isfinite(sum)) return -1;
    if (rounding != POSITIVE_ROUND_NEAREST && a != 0.0 && b != 0.0) {
        double b_virtual = sum - a;
        double error = (a - (sum - b_virtual)) + (b - b_virtual);
        if (rounding == POSITIVE_ROUND_UPPER && error > 0.0)
            sum = nextafter(sum, INFINITY);
        else if (rounding == POSITIVE_ROUND_LOWER && error < 0.0)
            sum = nextafter(sum, 0.0);
    }
    if (!isfinite(sum) || sum < 0.0) return -1;
    *result = sum;
    return 0;
}

static int positive_multiply_bound(double a, double b,
                                   PositiveRounding rounding,
                                   double *result)
{
    if (!result || !(a >= 0.0) || !(b >= 0.0) ||
        !isfinite(a) || !isfinite(b)) return -1;
    if (a == 0.0 || b == 0.0) {
        *result = 0.0;
        return 0;
    }
    double product = a * b;
    if (!isfinite(product)) return -1;
    if (rounding != POSITIVE_ROUND_NEAREST) {
        /* Do not infer exactness from fma(a,b,-product): its residual can
         * itself underflow.  One unconditional adjacent step encloses the
         * exact product even in the subnormal regime.  These are proof
         * bounds only and never replace a physical field value. */
        if (product == 0.0) {
            if (rounding == POSITIVE_ROUND_UPPER)
                product = nextafter(0.0, INFINITY);
        } else if (rounding == POSITIVE_ROUND_UPPER)
            product = nextafter(product, INFINITY);
        else
            product = nextafter(product, 0.0);
    }
    if (!isfinite(product) || product < 0.0) return -1;
    *result = product;
    return 0;
}

/* The queue is stored from high to low frequency, while radiation traverses
 * the window from low to high.  Therefore concatenating stored A then B means
 * applying physical B then A.  Every coefficient and operation is
 * nonnegative; there is no old-large-term subtraction. */
static PositiveTransform positive_transform_reverse_compose(
    PositiveTransform a, PositiveTransform b, PositiveRounding rounding)
{
    PositiveTransform result = {NAN, NAN};
    double attenuated;
    if (positive_multiply_bound(b.transmission, a.transmission, rounding,
                                &result.transmission) != 0 ||
        positive_multiply_bound(b.transmission, a.emission, rounding,
                                &attenuated) != 0 ||
        positive_add_bound(b.emission, attenuated, rounding,
                           &result.emission) != 0) {
        result.transmission = NAN;
        result.emission = NAN;
    }
    return result;
}

static int positive_window_init(PositiveWindow *window, size_t capacity,
                                PositiveRounding rounding)
{
    if (!window) return -1;
    memset(window, 0, sizeof(*window));
    window->capacity = capacity;
    window->rounding = rounding;
    if (capacity == 0) return 0;
    if (capacity > SIZE_MAX / (2U * sizeof(PositiveWindowNode))) return -1;
    window->front = (PositiveWindowNode *)malloc(
        2U * capacity * sizeof(*window->front));
    if (!window->front) return -1;
    window->back = window->front + capacity;
    return 0;
}

static void positive_window_free(PositiveWindow *window)
{
    if (!window) return;
    free(window->front);
    memset(window, 0, sizeof(*window));
}

static int positive_window_push_back(PositiveWindow *window,
                                     PositiveTransform value)
{
    if (!window || window->back_size >= window->capacity ||
        !(value.transmission >= 0.0) || !(value.emission >= 0.0) ||
        !isfinite(value.transmission) || !isfinite(value.emission)) return -1;
    PositiveWindowNode *node = &window->back[window->back_size];
    node->value = value;
    node->aggregate = window->back_size == 0 ? value
        : positive_transform_reverse_compose(
              window->back[window->back_size - 1U].aggregate, value,
              window->rounding);
    ++window->back_size;
    return isfinite(node->aggregate.transmission) &&
           isfinite(node->aggregate.emission) ? 0 : -1;
}

static int positive_window_transfer(PositiveWindow *window)
{
    if (!window || window->front_size != 0) return -1;
    while (window->back_size != 0) {
        PositiveTransform value = window->back[--window->back_size].value;
        if (window->front_size >= window->capacity) return -1;
        PositiveWindowNode *node = &window->front[window->front_size];
        node->value = value;
        node->aggregate = window->front_size == 0 ? value
            : positive_transform_reverse_compose(
                  value, window->front[window->front_size - 1U].aggregate,
                  window->rounding);
        ++window->front_size;
        if (!isfinite(node->aggregate.transmission) ||
            !isfinite(node->aggregate.emission)) return -1;
    }
    return 0;
}

static int positive_window_pop_front(PositiveWindow *window)
{
    if (!window) return -1;
    if (window->front_size == 0 && positive_window_transfer(window) != 0)
        return -1;
    if (window->front_size == 0) return -1;
    --window->front_size;
    return 0;
}

static int positive_window_aggregate(const PositiveWindow *window,
                                     PositiveTransform *aggregate)
{
    if (!window || !aggregate) return -1;
    if (window->front_size == 0 && window->back_size == 0) {
        aggregate->transmission = 1.0;
        aggregate->emission = 0.0;
    } else if (window->front_size == 0) {
        *aggregate = window->back[window->back_size - 1U].aggregate;
    } else if (window->back_size == 0) {
        *aggregate = window->front[window->front_size - 1U].aggregate;
    } else {
        *aggregate = positive_transform_reverse_compose(
            window->front[window->front_size - 1U].aggregate,
            window->back[window->back_size - 1U].aggregate,
            window->rounding);
    }
    return isfinite(aggregate->transmission) &&
           isfinite(aggregate->emission) &&
           aggregate->transmission >= 0.0 && aggregate->emission >= 0.0
         ? 0 : -1;
}

/* raw η 모드의 양의 방출 연산자. κ=0에서 연속이며 κ<0도 그대로
 * 증폭한다. 계수는 같은 이산 연산에서 고정된 binary64 값으로 사용한다. */
static double exact_emission_factor(double depth,double transmission,
                                    double length,int raw)
{
    if(!raw)return 1.0-transmission;
    double tau=depth*length;
    if(tau==0.0)return length;
    return length*(-expm1(-tau)/tau);
}

static int exact_segment_positive_sliding(
    int nb, const double *dt1, const double *t1, const double *source,
    const double *source_cell, double beta, const double *upstream,
    int upstream_zero, double *output,int raw)
{
    int q = (int)floor(beta);
    double phi = beta - (double)q;
    int qtop;
    double psi;
    if (phi < 0.5) { qtop = q; psi = phi + 0.5; }
    else { qtop = q + 1; psi = phi - 0.5; }
    size_t window_count = qtop >= 2 ? (size_t)qtop - 1U : 0U;
    PositiveWindow window;
    if (positive_window_init(&window, window_count,
                             POSITIVE_ROUND_NEAREST) != 0) return -2;
    int b = nb - 1;
    for (int m = b + qtop - 1; m >= b + 1; --m) {
        int mm = clip_bin(m, nb);
        PositiveTransform value = {t1[mm], source_cell[mm]};
        if (positive_window_push_back(&window, value) != 0) {
            positive_window_free(&window);
            return -2;
        }
    }
    for (; b >= 0; --b) {
        double intensity = 0.0;
        if (!upstream_zero) {
            int i0 = clip_bin(b + q, nb), i1 = clip_bin(b + q + 1, nb);
            intensity = (1.0 - phi) * upstream[i0] + phi * upstream[i1];
        }
        int top = clip_bin(b + qtop, nb);
        double transmission = exp(-psi * dt1[top]);
        intensity = intensity * transmission +
                    exact_emission_factor(dt1[top],transmission,psi,raw) * source[top];
        PositiveTransform aggregate;
        if (positive_window_aggregate(&window, &aggregate) != 0) {
            positive_window_free(&window);
            return -2;
        }
        intensity = intensity * aggregate.transmission + aggregate.emission;
        double half = raw ? exp(-.5*dt1[b]) : sqrt(t1[b]);
        output[b] = intensity * half +
            exact_emission_factor(dt1[b],half,.5,raw) * source[b];
        if (!(output[b] >= 0.0) || !isfinite(output[b])) {
            positive_window_free(&window);
            return -2;
        }
        if (b == 0 || window_count == 0) continue;
        if (positive_window_pop_front(&window) != 0) {
            positive_window_free(&window);
            return -2;
        }
        PositiveTransform incoming = {t1[b], source_cell[b]};
        if (positive_window_push_back(&window, incoming) != 0) {
            positive_window_free(&window);
            return -2;
        }
    }
    positive_window_free(&window);
    return 0;
}

static int positive_two_product_sum_bound(
    double a, double x, double b, double y, PositiveRounding rounding,
    double *result)
{
    double ax, by;
    if (positive_multiply_bound(a, x, rounding, &ax) != 0 ||
        positive_multiply_bound(b, y, rounding, &by) != 0 ||
        positive_add_bound(ax, by, rounding, result) != 0) return -1;
    return 0;
}

/* Outward evaluation of the subtraction-free segment.  Coefficients such as
 * exp(-dt), 1-phi, and 1-transmission are the same fixed binary64 objects
 * used by the production sweep.  The proof therefore qualifies that
 * discrete operator; source-dependent multiply/add operations are directed
 * outward without claiming a continuum/discretization error bound. */
static int exact_segment_positive_bound(
    int nb, const double *dt1, const double *t1, const double *source,
    const double *source_cell, double beta, const double *upstream,
    int upstream_zero, double *output, PositiveRounding rounding,int raw)
{
    if (rounding == POSITIVE_ROUND_NEAREST) return -1;
    int q = (int)floor(beta);
    double phi = beta - (double)q;
    if (beta <= 0.5) {
        for (int b = 0; b < nb; ++b) {
            double intensity = 0.0;
            if (!upstream_zero) {
                int i0 = clip_bin(b + q, nb), i1 = clip_bin(b + q + 1, nb);
                if (positive_two_product_sum_bound(
                        1.0 - phi, upstream[i0], phi, upstream[i1],
                        rounding, &intensity) != 0) return -1;
            }
            double dt = dt1[b] * beta;
            double transmission = exp(-dt);
            if (positive_two_product_sum_bound(
                    transmission, intensity,
                    exact_emission_factor(dt1[b],transmission,beta,raw), source[b],
                    rounding, &output[b]) != 0) return -1;
        }
        return 0;
    }

    int qtop;
    double psi;
    if (phi < 0.5) { qtop = q; psi = phi + 0.5; }
    else { qtop = q + 1; psi = phi - 0.5; }
    size_t window_count = qtop >= 2 ? (size_t)qtop - 1U : 0U;
    PositiveWindow window;
    if (positive_window_init(&window, window_count, rounding) != 0) return -1;
    int b = nb - 1;
    for (int m = b + qtop - 1; m >= b + 1; --m) {
        int mm = clip_bin(m, nb);
        PositiveTransform value = {t1[mm], source_cell[mm]};
        if (positive_window_push_back(&window, value) != 0) {
            positive_window_free(&window);
            return -1;
        }
    }
    for (; b >= 0; --b) {
        double intensity = 0.0;
        if (!upstream_zero) {
            int i0 = clip_bin(b + q, nb), i1 = clip_bin(b + q + 1, nb);
            if (positive_two_product_sum_bound(
                    1.0 - phi, upstream[i0], phi, upstream[i1], rounding,
                    &intensity) != 0) {
                positive_window_free(&window);
                return -1;
            }
        }
        int top = clip_bin(b + qtop, nb);
        double transmission = exp(-psi * dt1[top]);
        if (positive_two_product_sum_bound(
                transmission, intensity,
                exact_emission_factor(dt1[top],transmission,psi,raw), source[top],
                rounding, &intensity) != 0) {
            positive_window_free(&window);
            return -1;
        }
        PositiveTransform aggregate;
        if (positive_window_aggregate(&window, &aggregate) != 0 ||
            positive_two_product_sum_bound(
                aggregate.transmission, intensity, 1.0,
                aggregate.emission, rounding, &intensity) != 0) {
            positive_window_free(&window);
            return -1;
        }
        double half = raw ? exp(-.5*dt1[b]) : sqrt(t1[b]);
        if (positive_two_product_sum_bound(
                half, intensity, exact_emission_factor(dt1[b],half,.5,raw), source[b], rounding,
                &output[b]) != 0) {
            positive_window_free(&window);
            return -1;
        }
        if (b == 0 || window_count == 0) continue;
        if (positive_window_pop_front(&window) != 0) {
            positive_window_free(&window);
            return -1;
        }
        PositiveTransform incoming = {t1[b], source_cell[b]};
        if (positive_window_push_back(&window, incoming) != 0) {
            positive_window_free(&window);
            return -1;
        }
    }
    positive_window_free(&window);
    return 0;
}

/* Algebraically identical to the independently validated July harness. */
static int exact_segment(int nb, const double *dt1, const double *t1,
                         const double *source, const double *source_cell,
                         double beta, const double *upstream,
                         int upstream_zero, double *output, CMFExactMode mode,
                         double *negative_value,int raw)
{
    int q = (int)floor(beta);
    double phi = beta - (double)q;
    if (beta <= 0.5) {
        for (int b = 0; b < nb; ++b) {
            double intensity = 0.0;
            if (!upstream_zero) {
                int i0 = clip_bin(b + q, nb), i1 = clip_bin(b + q + 1, nb);
                intensity = (1.0 - phi) * upstream[i0] + phi * upstream[i1];
            }
            double dt = dt1[b] * beta, transmission = exp(-dt);
            output[b] = intensity * transmission +
                        exact_emission_factor(dt1[b],transmission,beta,raw) * source[b];
            if(raw&&(!(output[b]>=0)||!isfinite(output[b])))return -2;
        }
        return 0;
    }

    int qtop;
    double psi;
    if (phi < 0.5) { qtop = q; psi = phi + 0.5; }
    else { qtop = q + 1; psi = phi - 0.5; }

    if (mode == CMF_EXACT_MODE_DIRECT_REFERENCE) {
        for (int b = 0; b < nb; ++b) {
            double x = (double)b + beta, intensity = 0.0;
            if (!upstream_zero) {
                int i0 = clip_bin(b + q, nb), i1 = clip_bin(b + q + 1, nb);
                intensity = (1.0 - phi) * upstream[i0] + phi * upstream[i1];
            }
            int m = (int)floor(x + 0.5);
            double cursor = x;
            for (;;) {
                double lower = m - 0.5 > (double)b ? m - 0.5 : (double)b;
                double length = cursor - lower;
                if (length > 0.0) {
                    int mm = clip_bin(m, nb);
                    double transmission = exp(-dt1[mm] * length);
                    intensity = intensity * transmission +
                                exact_emission_factor(dt1[mm],transmission,length,raw) * source[mm];
                }
                if (lower <= (double)b + 1.0e-12) break;
                cursor = lower;
                --m;
            }
            output[b] = intensity;
            if(raw&&(!(output[b]>=0)||!isfinite(output[b])))return -2;
        }
        return 0;
    }

    if (mode == CMF_EXACT_MODE_POSITIVE_SLIDING)
        return exact_segment_positive_sliding(
            nb, dt1, t1, source, source_cell, beta, upstream,
            upstream_zero, output,raw);

    int b = nb - 1;
    double window_tau = 0.0, window_source = 0.0, prefix = 1.0;
    for (int m = b + 1; m <= b + qtop - 1; ++m) {
        int mm = clip_bin(m, nb);
        window_source += source_cell[mm] * prefix;
        prefix *= t1[mm];
        window_tau += dt1[mm];
    }
    for (; b >= 0; --b) {
        double intensity = 0.0;
        if (!upstream_zero) {
            int i0 = clip_bin(b + q, nb), i1 = clip_bin(b + q + 1, nb);
            intensity = (1.0 - phi) * upstream[i0] + phi * upstream[i1];
        }
        int top = clip_bin(b + qtop, nb);
        double transmission = exp(-psi * dt1[top]);
        intensity = intensity * transmission +
                    (1.0 - transmission) * source[top];
        intensity = intensity * exp(-window_tau) + window_source;
        double half = sqrt(t1[b]);
        output[b] = intensity * half + (1.0 - half) * source[b];
        if (b == 0) break;
        if (qtop < 2) continue;
        int outgoing = clip_bin(b + qtop - 1, nb);
        double reduced_tau = window_tau - dt1[outgoing];
        window_source = source_cell[b] + t1[b] *
            (window_source - source_cell[outgoing] * exp(-reduced_tau));
        if (window_source < 0.0) {
            if (negative_value) *negative_value = window_source;
            return -1;
        }
        window_tau = reduced_tau + dt1[b];
    }
    return 0;
}

typedef enum {
    EXACT_SWEEP_OK = 0,
    EXACT_SWEEP_NEGATIVE,
    EXACT_SWEEP_ALLOCATION_FAILED,
    EXACT_SWEEP_NONFINITE
} ExactSweepStatus;

static int exact_fill_inner(
    const ExactProblem *p, double temperature, double scale,
    const double *J, int homogeneous, PositiveRounding rounding,
    double *inner)
{
    int copies = p->physical_cells ? p->nr : 1;
    for (int k = 0; k < copies; ++k) {
        for (int b = 0; b < p->nb; ++b) {
            size_t at = (size_t)k*p->inner_stride+(size_t)b;
            if (!p->boundary || p->boundary->kind == CMF_EXACT_INNER_BLACKBODY) {
                inner[at] = homogeneous ? 0.0 : scale*exact_planck(p->nu[b],temperature);
            } else {
                const CMFExactInnerBoundary *bc = p->boundary;
                double scatter, source, gradient;
                if (positive_multiply_bound(bc->scattering_ratio[b],J[b],rounding,
                                            &scatter) != 0 ||
                    positive_add_bound(homogeneous ? 0.0 : bc->source_fixed[b],
                                       scatter,rounding,&source) != 0 ||
                    positive_multiply_bound(p->core_mu[k],
                        homogeneous ? 0.0 : bc->gradient_over_chi[b],
                        rounding,&gradient) != 0 ||
                    positive_add_bound(source,gradient,rounding,&inner[at]) != 0)
                    return -1;
            }
            if (!(inner[at] >= 0.0) || !isfinite(inner[at])) return -1;
        }
    }
    return 0;
}

/* 한 실제 셀의 두 절반을 별도로 통과한다. in/out은 중심에서 저장하고
 * edge는 실제 경계의 강도다. turning cell이 중심 안쪽까지 도달하지 않으면
 * 그 셀은 수송에는 참여하지만 해당 중심의 각도 적분에서는 제외한다. */
static int exact_cell_ray_sweep(
    const ExactProblem *p, int ray,
    const double *dt1, const double *t1, const double *source,
    const double *source_cell, const double *inner,
    double *in, double *out, CMFExactMode mode, PositiveRounding rounding,
    double *negative,size_t buffer_base)
{
    int nb = p->nb, n = p->rn[ray];
    size_t base = (size_t)ray*p->ray_stride;
    double *edge = malloc((size_t)nb*sizeof(double));
    if (!edge) return -2;
    int rc = 0;
    for (int direction = 0; direction < 2 && rc == 0; ++direction) {
        if (direction && p->rcore[ray])
            memcpy(edge,inner+(size_t)ray*p->inner_stride,(size_t)nb*sizeof(double));
        for (int step = 0; step < n && rc == 0; ++step) {
            int i = direction ? n-1-step : step;
            size_t slot = base+(size_t)i;
            size_t cell = (size_t)p->rsh[slot]*(size_t)nb;
            double *sample = (direction ? out : in)+(slot-buffer_base)*(size_t)nb;
            double first = direction ? p->half_inner[slot] : p->half_outer[slot];
            double second = direction ? p->half_outer[slot] : p->half_inner[slot];
            int zero = !direction && !step;
            if (rounding == POSITIVE_ROUND_NEAREST) {
                rc = exact_segment(nb,dt1+cell,t1+cell,source+cell,source_cell+cell,
                    p->a_drift*first,zero ? NULL : edge,zero,sample,mode,negative,p->raw_emissivity);
                if (!rc) rc = exact_segment(nb,dt1+cell,t1+cell,source+cell,source_cell+cell,
                    p->a_drift*second,sample,0,edge,mode,negative,p->raw_emissivity);
            } else {
                rc = exact_segment_positive_bound(nb,dt1+cell,t1+cell,source+cell,
                    source_cell+cell,p->a_drift*first,zero ? NULL : edge,zero,sample,rounding,p->raw_emissivity);
                if (!rc) rc = exact_segment_positive_bound(nb,dt1+cell,t1+cell,source+cell,
                    source_cell+cell,p->a_drift*second,sample,0,edge,rounding,p->raw_emissivity);
                if (rc) rc = -2;
            }
        }
    }
    free(edge);
    return rc;
}

/* p는 impact 오름차순이며, 역순으로 소비하면 각 쉘의 기존 mu 오름차순
 * 사다리꼴 적분과 정확히 같은 합산 순서다. 묶음 안의 광선만 병렬 전파하고
 * 합산은 원래 순서로 수행한다. nearest와 양방향 오차 연산이 이 경로를 공유한다. */
static ExactSweepStatus exact_cell_stream_sweep(
    const ExactProblem *p,const double *dt1,const double *t1,
    const double *source,const double *source_cell,const double *inner,
    double *in,double *out,double *j_result,CMFExactMode mode,
    PositiveRounding rounding,uint64_t *negative_count_out,double *first_negative_out)
{
    size_t cells=(size_t)p->ns*p->nb,batch=exact_buffer_rays(p);
    double *previous=malloc(cells*sizeof(double));
    double *previous_mu=calloc((size_t)p->ns,sizeof(double));
    int *seen=calloc((size_t)p->ns,sizeof(int));
    if(!previous||!previous_mu||!seen){
        free(previous);free(previous_mu);free(seen);return EXACT_SWEEP_ALLOCATION_FAILED;
    }
    memset(j_result,0,cells*sizeof(double));
    ExactSweepStatus status=EXACT_SWEEP_OK;
    uint64_t negatives=0;double first_negative=NAN;
    for(int high=p->nr-1;high>=0;){
        int low=high+1>(int)batch ? high+1-(int)batch : 0;
        size_t buffer_base=(size_t)low*p->ray_stride;
        int failed=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1) reduction(|:failed)
#endif
        for(int k=low;k<=high;k++){
            if(p->rn[k]<=0)continue;
            double negative=NAN;
            int rc=exact_cell_ray_sweep(p,k,dt1,t1,source,source_cell,inner,
                in,out,mode,rounding,&negative,buffer_base);
            if(rc==-1){
#ifdef _OPENMP
#pragma omp critical(cmf_exact_negative)
#endif
                {if(!negatives)first_negative=negative;negatives++;}
            }else if(rc)failed=1;
        }
        if(negatives){status=EXACT_SWEEP_NEGATIVE;break;}
        if(failed){status=EXACT_SWEEP_ALLOCATION_FAILED;break;}
        for(int k=high;k>=low;k--){
            size_t base=(size_t)k*p->ray_stride;
            int invalid=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:invalid)
#endif
            for(int i=0;i<p->rn[k];i++){
                int s=p->rsh[base+(size_t)i];
                if(p->p[k]>p->rmid[s])continue;
                double mu=p->rz[base+(size_t)i]/p->rmid[s];
                if(mu<previous_mu[s]){invalid=1;continue;}
                size_t at=(base+(size_t)i-buffer_base)*(size_t)p->nb;
                size_t cell=(size_t)s*p->nb;
                for(int b=0;b<p->nb;b++){
                    double j;
                    if(rounding==POSITIVE_ROUND_NEAREST){
                        j=.5*(out[at+b]+in[at+b]);
                        double prev=seen[s] ? previous[cell+b] : j;
                        j_result[cell+b]+=.5*(prev+j)*(mu-previous_mu[s]);
                    }else{
                        double prev,average,weighted,sum;
                        if(positive_two_product_sum_bound(.5,out[at+b],.5,in[at+b],rounding,&j)){
                            invalid=1;break;
                        }
                        prev=seen[s] ? previous[cell+b] : j;
                        if(positive_add_bound(prev,j,rounding,&average)||
                           positive_multiply_bound(.5*(mu-previous_mu[s]),average,rounding,&weighted)||
                           positive_add_bound(j_result[cell+b],weighted,rounding,&sum)){
                            invalid=1;break;
                        }
                        j_result[cell+b]=sum;
                    }
                    previous[cell+b]=j;
                    if(!(j_result[cell+b]>=0)||!isfinite(j_result[cell+b]))invalid=1;
                }
                previous_mu[s]=mu;seen[s]=1;
            }
            if(invalid){status=EXACT_SWEEP_NONFINITE;break;}
        }
        if(status!=EXACT_SWEEP_OK)break;
        high=low-1;
    }
    if(status==EXACT_SWEEP_OK)for(int s=0;s<p->ns;s++)if(!seen[s])
        memcpy(j_result+(size_t)s*p->nb,source+(size_t)s*p->nb,(size_t)p->nb*sizeof(double));
    if(negative_count_out)*negative_count_out=negatives;
    if(first_negative_out)*first_negative_out=first_negative;
    free(previous);free(previous_mu);free(seen);
    return status;
}

/* One application of the production formal-solution object.  Keeping ray
 * propagation and angular reconstruction in this single routine is required
 * for the forthcoming K error-envelope path: a separately reconstructed
 * operator would not qualify the fixed point actually solved above. */
static ExactSweepStatus exact_formal_sweep(
    const ExactProblem *p, const double *dt1, const double *t1,
    const double *source, const double *source_cell, const double *inner,
    double *in, double *out, double *j_result, CMFExactMode mode,
    uint64_t *negative_count_out, double *first_negative_out)
{
    if(exact_cell_streams(p))return exact_cell_stream_sweep(p,dt1,t1,source,
        source_cell,inner,in,out,j_result,mode,POSITIVE_ROUND_NEAREST,
        negative_count_out,first_negative_out);
    uint64_t negative_count = 0;
    double first_negative = NAN;
    int segment_failure = 0;
    int nb = p->nb;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int k = 0; k < p->nr; ++k) {
        int n = p->rn[k];
        if (n <= 0) continue;
        if (p->physical_cells) {
            double negative = NAN;
            int rc = exact_cell_ray_sweep(p,k,dt1,t1,source,source_cell,inner,
                in,out,mode,POSITIVE_ROUND_NEAREST,&negative,0);
            if (rc) {
#ifdef _OPENMP
#pragma omp critical(cmf_exact_negative)
#endif
                {
                    if (rc == -1) {
                        if (!negative_count) first_negative = negative;
                        ++negative_count;
                    } else segment_failure = 1;
                }
            }
            continue;
        }
        int ray_failed = 0;
        size_t ray_base = (size_t)k * p->ray_stride;
        for (int i = 0; i < n; ++i) {
            int shell = p->rsh[ray_base + i];
            double ds = i + 1 < n
                ? p->rz[ray_base + i] - p->rz[ray_base + i + 1]
                : p->rz[ray_base + i] - p->rzin[k];
            const double *upstream = i > 0
                ? in + (ray_base + (size_t)i - 1U) * (size_t)nb : NULL;
            double neg = NAN;
            int rc = exact_segment(
                nb, dt1 + (size_t)shell * (size_t)nb,
                t1 + (size_t)shell * (size_t)nb,
                source + (size_t)shell * (size_t)nb,
                source_cell + (size_t)shell * (size_t)nb,
                p->a_drift * ds, upstream, i == 0,
                in + (ray_base + (size_t)i) * (size_t)nb, mode, &neg,0);
            if (rc != 0) {
#ifdef _OPENMP
#pragma omp critical(cmf_exact_negative)
#endif
                {
                    if (rc == -1) {
                        if (negative_count == 0) first_negative = neg;
                        ++negative_count;
                    } else {
                        segment_failure = 1;
                    }
                }
                ray_failed = 1;
                break;
            }
        }
        if (ray_failed) continue;
        for (int i = n - 1; i >= 0; --i) {
            int shell = p->rsh[ray_base + i];
            double ds = i + 1 < n
                ? p->rz[ray_base + i] - p->rz[ray_base + i + 1]
                : p->rz[ray_base + i] - p->rzin[k];
            const double *upstream;
            if (i == n - 1)
                upstream = p->rcore[k] ? inner
                    : in + (ray_base + (size_t)n - 1U) * (size_t)nb;
            else
                upstream = out + (ray_base + (size_t)i + 1U) * (size_t)nb;
            double neg = NAN;
            int rc = exact_segment(
                nb, dt1 + (size_t)shell * (size_t)nb,
                t1 + (size_t)shell * (size_t)nb,
                source + (size_t)shell * (size_t)nb,
                source_cell + (size_t)shell * (size_t)nb,
                p->a_drift * ds, upstream, 0,
                out + (ray_base + (size_t)i) * (size_t)nb, mode, &neg,0);
            if (rc != 0) {
#ifdef _OPENMP
#pragma omp critical(cmf_exact_negative)
#endif
                {
                    if (rc == -1) {
                        if (negative_count == 0) first_negative = neg;
                        ++negative_count;
                    } else {
                        segment_failure = 1;
                    }
                }
                break;
            }
        }
    }
    if (negative_count_out) *negative_count_out = negative_count;
    if (first_negative_out) *first_negative_out = first_negative;
    if (negative_count != 0) return EXACT_SWEEP_NEGATIVE;
    if (segment_failure != 0) return EXACT_SWEEP_ALLOCATION_FAILED;

    int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:invalid)
#endif
    for (int s = 0; s < p->ns; ++s) {
        int lo = p->shell_off[s], hi = p->shell_off[s + 1];
        for (int b = 0; b < nb; ++b) {
            size_t idx = (size_t)s * (size_t)nb + (size_t)b;
            if (hi <= lo) {
                j_result[idx] = source[idx];
                continue;
            }
            int ray = p->shell_k[lo], seg = p->shell_seg[lo];
            size_t at = ((size_t)ray * p->ray_stride + (size_t)seg) *
                        (size_t)nb + (size_t)b;
            double previous_mu = 0.0;
            double previous_j = 0.5 * (out[at] + in[at]);
            double sum = 0.0;
            for (int a = lo; a < hi; ++a) {
                ray = p->shell_k[a]; seg = p->shell_seg[a];
                double mu = p->shell_mu[a];
                at = ((size_t)ray * p->ray_stride + (size_t)seg) *
                     (size_t)nb + (size_t)b;
                double j = 0.5 * (out[at] + in[at]);
                sum += 0.5 * (previous_j + j) * (mu - previous_mu);
                previous_mu = mu;
                previous_j = j;
            }
            j_result[idx] = sum;
            if (!(sum >= 0.0) || !isfinite(sum)) invalid = 1;
        }
    }
    return invalid ? EXACT_SWEEP_NONFINITE : EXACT_SWEEP_OK;
}

static int exact_formal_sweep_bound(
    const ExactProblem *p, const double *dt1, const double *t1,
    const double *source, const double *source_cell, const double *inner,
    double *in, double *out, double *j_result, PositiveRounding rounding)
{
    if (rounding == POSITIVE_ROUND_NEAREST) return -1;
    if(exact_cell_streams(p))return exact_cell_stream_sweep(p,dt1,t1,source,
        source_cell,inner,in,out,j_result,CMF_EXACT_MODE_POSITIVE_SLIDING,
        rounding,NULL,NULL)==EXACT_SWEEP_OK ? 0 : -1;
    int failed = 0;
    int nb = p->nb;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(|:failed)
#endif
    for (int k = 0; k < p->nr; ++k) {
        int n = p->rn[k];
        if (n <= 0) continue;
        if (p->physical_cells) {
            double negative = NAN;
            if (exact_cell_ray_sweep(p,k,dt1,t1,source,source_cell,inner,in,out,
                    CMF_EXACT_MODE_POSITIVE_SLIDING,rounding,&negative,0) != 0)
                failed = 1;
            continue;
        }
        int ray_failed = 0;
        size_t ray_base = (size_t)k * p->ray_stride;
        for (int i = 0; i < n; ++i) {
            int shell = p->rsh[ray_base + i];
            double ds = i + 1 < n
                ? p->rz[ray_base + i] - p->rz[ray_base + i + 1]
                : p->rz[ray_base + i] - p->rzin[k];
            const double *upstream = i > 0
                ? in + (ray_base + (size_t)i - 1U) * (size_t)nb : NULL;
            if (exact_segment_positive_bound(
                    nb, dt1 + (size_t)shell * (size_t)nb,
                    t1 + (size_t)shell * (size_t)nb,
                    source + (size_t)shell * (size_t)nb,
                    source_cell + (size_t)shell * (size_t)nb,
                    p->a_drift * ds, upstream, i == 0,
                    in + (ray_base + (size_t)i) * (size_t)nb,
                    rounding,0) != 0) {
                failed = 1;
                ray_failed = 1;
                break;
            }
        }
        if (ray_failed) continue;
        for (int i = n - 1; i >= 0; --i) {
            int shell = p->rsh[ray_base + i];
            double ds = i + 1 < n
                ? p->rz[ray_base + i] - p->rz[ray_base + i + 1]
                : p->rz[ray_base + i] - p->rzin[k];
            const double *upstream;
            if (i == n - 1)
                upstream = p->rcore[k] ? inner
                    : in + (ray_base + (size_t)n - 1U) * (size_t)nb;
            else
                upstream = out + (ray_base + (size_t)i + 1U) * (size_t)nb;
            if (exact_segment_positive_bound(
                    nb, dt1 + (size_t)shell * (size_t)nb,
                    t1 + (size_t)shell * (size_t)nb,
                    source + (size_t)shell * (size_t)nb,
                    source_cell + (size_t)shell * (size_t)nb,
                    p->a_drift * ds, upstream, 0,
                    out + (ray_base + (size_t)i) * (size_t)nb,
                    rounding,0) != 0) {
                failed = 1;
                break;
            }
        }
    }
    if (failed) return -1;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:failed)
#endif
    for (int s = 0; s < p->ns; ++s) {
        int lo = p->shell_off[s], hi = p->shell_off[s + 1];
        for (int b = 0; b < nb; ++b) {
            size_t idx = (size_t)s * (size_t)nb + (size_t)b;
            if (hi <= lo) {
                j_result[idx] = source[idx];
                continue;
            }
            int ray = p->shell_k[lo], seg = p->shell_seg[lo];
            size_t at = ((size_t)ray * p->ray_stride + (size_t)seg) *
                        (size_t)nb + (size_t)b;
            double previous_mu = 0.0;
            double previous_j;
            if (positive_two_product_sum_bound(
                    0.5, out[at], 0.5, in[at], rounding,
                    &previous_j) != 0) {
                failed = 1;
                continue;
            }
            double sum = 0.0;
            for (int a = lo; a < hi; ++a) {
                ray = p->shell_k[a]; seg = p->shell_seg[a];
                double mu = p->shell_mu[a];
                at = ((size_t)ray * p->ray_stride + (size_t)seg) *
                     (size_t)nb + (size_t)b;
                double j, average, weighted, next_sum;
                if (positive_two_product_sum_bound(
                        0.5, out[at], 0.5, in[at], rounding, &j) != 0 ||
                    positive_add_bound(previous_j, j, rounding,
                                       &average) != 0 ||
                    positive_multiply_bound(
                        0.5 * (mu - previous_mu), average, rounding,
                        &weighted) != 0 ||
                    positive_add_bound(sum, weighted, rounding,
                                       &next_sum) != 0) {
                    failed = 1;
                    break;
                }
                sum = next_sum;
                previous_mu = mu;
                previous_j = j;
            }
            j_result[idx] = sum;
        }
    }
    return failed ? -1 : 0;
}

CMFExactStatus cmf_exact_characteristic_apply_positive_bounds(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, const double *input_J,
    double *lower, double *nearest, double *upper)
{
    if (!input_J || !lower || !nearest || !upper ||
        !(T_inner > 0.0) || !isfinite(T_inner) ||
        !(inner_boundary_scale >= 0.0) || !isfinite(inner_boundary_scale))
        return CMF_EXACT_INVALID_INPUT;
    ExactProblem p;
    CMFExactStatus status = exact_problem_build(
        &p, n_shells, n_bins, dlognu, nu, r_inner, r_outer,
        time_explosion, chi_tot, chi_es, S_fixed, 0, NULL);
    if (status != CMF_EXACT_OK) return status;
    size_t cells, segment_cells;
    if (checked_mul_size((size_t)n_shells, (size_t)n_bins, &cells) != 0 ||
        checked_mul_size(p.segment_slots, (size_t)n_bins,
                         &segment_cells) != 0 ||
        cells > SIZE_MAX / (4U * sizeof(double)) ||
        segment_cells > SIZE_MAX / (2U * sizeof(double))) {
        exact_problem_free(&p);
        return CMF_EXACT_INVALID_INPUT;
    }
    double *cell_storage = (double *)malloc(4U * cells * sizeof(double));
    double *segment_storage =
        (double *)malloc(2U * segment_cells * sizeof(double));
    double *inner = (double *)malloc((size_t)n_bins * sizeof(double));
    if (!cell_storage || !segment_storage || !inner) {
        free(cell_storage); free(segment_storage); free(inner);
        exact_problem_free(&p);
        return CMF_EXACT_ALLOCATION_FAILED;
    }
    double *dt1 = cell_storage;
    double *t1 = dt1 + cells;
    double *source = t1 + cells;
    double *source_cell = source + cells;
    double *in = segment_storage;
    double *out = in + segment_cells;
    int invalid = 0;
    for (size_t idx = 0; idx < cells; ++idx) {
        double ct = chi_tot[idx], ce = chi_es[idx];
        double sf = S_fixed[idx], j = input_J[idx];
        if (!(ct >= 0.0) || !(ce >= 0.0) ||
            (ct == 0.0 && ce != 0.0) || !(sf >= 0.0) || !(j >= 0.0) ||
            !isfinite(ct) || !isfinite(ce) ||
            !isfinite(sf) || !isfinite(j)) {
            invalid = 1;
            break;
        }
        double depth = (ct + p.a_lam * 4.0) / p.a_drift;
        if (!(depth >= 0.0) || !isfinite(depth)) {
            invalid = 1;
            break;
        }
        dt1[idx] = depth;
        t1[idx] = exp(-depth);
    }
    for (int b = 0; b < n_bins; ++b)
        inner[b] = inner_boundary_scale * exact_planck(nu[b], T_inner);
    for (int pass = -1; pass <= 1 && !invalid; ++pass) {
        PositiveRounding rounding = pass < 0 ? POSITIVE_ROUND_LOWER
            : (pass > 0 ? POSITIVE_ROUND_UPPER : POSITIVE_ROUND_NEAREST);
        for (size_t idx = 0; idx < cells; ++idx) {
            double ratio = chi_tot[idx] > 0.0
                         ? chi_es[idx] / chi_tot[idx] : 0.0;
            if (rounding == POSITIVE_ROUND_NEAREST) {
                source[idx] = S_fixed[idx] + ratio * input_J[idx];
                source_cell[idx] = (1.0 - t1[idx]) * source[idx];
            } else {
                double scattering;
                if (positive_multiply_bound(ratio, input_J[idx], rounding,
                                            &scattering) != 0 ||
                    positive_add_bound(S_fixed[idx], scattering, rounding,
                                       &source[idx]) != 0 ||
                    positive_multiply_bound(1.0 - t1[idx], source[idx],
                                            rounding,
                                            &source_cell[idx]) != 0) {
                    invalid = 1;
                    break;
                }
            }
        }
        if (invalid) break;
        double *destination = pass < 0 ? lower : (pass > 0 ? upper : nearest);
        if (rounding == POSITIVE_ROUND_NEAREST) {
            uint64_t negative_count = 0;
            double first_negative = NAN;
            ExactSweepStatus sweep_status = exact_formal_sweep(
                &p, dt1, t1, source, source_cell, inner, in, out,
                destination, CMF_EXACT_MODE_POSITIVE_SLIDING,
                &negative_count, &first_negative);
            if (sweep_status != EXACT_SWEEP_OK) invalid = 1;
        } else if (exact_formal_sweep_bound(
                       &p, dt1, t1, source, source_cell, inner, in, out,
                       destination, rounding) != 0) {
            invalid = 1;
        }
    }
    if (!invalid) {
        for (size_t idx = 0; idx < cells; ++idx) {
            if (!(lower[idx] >= 0.0) || !(nearest[idx] >= lower[idx]) ||
                !(upper[idx] >= nearest[idx]) || !isfinite(upper[idx])) {
                invalid = 1;
                break;
            }
        }
    }
    free(cell_storage); free(segment_storage); free(inner);
    exact_problem_free(&p);
    return invalid ? CMF_EXACT_NONFINITE : CMF_EXACT_OK;
}

typedef struct {
    const ExactProblem *problem;
    const double *dt1;
    const double *t1;
    const double *chi_tot;
    const double *chi_es;
    double *source;
    double *source_cell;
    double *inner;
    double *in;
    double *out;
    size_t cells;
} ExactEnvelopeOperator;

static double exact_source_weight(const ExactProblem *p, size_t idx)
{
    /* J와 경계의 단위는 I_nu다. homologous characteristic에서는
     * dI_nu/ds=eta_nu-(chi_nu+3/(ct))*I_nu이므로 source도 같은 총
     * 소광으로 나눈다. 기존 API의 4/(ct) 이산화는 소급 변경하지 않는다. */
    if(p->raw_emissivity)return 1.0/p->a_drift;
    if (!p->physical_cells) return 1.0;
    return p->chi_tot[idx]/(p->chi_tot[idx]+3.0*p->a_lam);
}

static double exact_scattering_coefficient(const ExactProblem *p,size_t idx)
{
    if(p->raw_emissivity)return p->chi_es[idx];
    return p->chi_tot[idx]>0 ? p->chi_es[idx]/p->chi_tot[idx] : 0;
}

static int exact_fill_scattering_source_upper(
    const double *input, size_t n, ExactEnvelopeOperator *operator_context)
{
    if (!input || !operator_context || n != operator_context->cells)
        return -1;
    for (size_t idx = 0; idx < n; ++idx) {
        double ratio = exact_scattering_coefficient(operator_context->problem,idx);
        if (positive_multiply_bound(
                ratio, input[idx], POSITIVE_ROUND_UPPER,
                &operator_context->source[idx]) != 0 ||
            positive_multiply_bound(exact_source_weight(operator_context->problem,idx),
                operator_context->source[idx],POSITIVE_ROUND_UPPER,
                &operator_context->source[idx]) != 0 ||
            positive_multiply_bound(
                exact_emission_factor(operator_context->dt1[idx],operator_context->t1[idx],
                    1.0,operator_context->problem->raw_emissivity),
                operator_context->source[idx], POSITIVE_ROUND_UPPER,
                &operator_context->source_cell[idx]) != 0) return -1;
    }
    return 0;
}

static int exact_apply_scattering_upper(const double *input,
                                        double *upper_output,
                                        size_t n, void *context)
{
    ExactEnvelopeOperator *operator_context =
        (ExactEnvelopeOperator *)context;
    if (!operator_context || !upper_output ||
        exact_fill_scattering_source_upper(
            input, n, operator_context) != 0) return -1;
    if (exact_fill_inner(operator_context->problem,0.0,0.0,input,1,
                         POSITIVE_ROUND_UPPER,operator_context->inner) != 0)
        return -1;
    return exact_formal_sweep_bound(
        operator_context->problem, operator_context->dt1,
        operator_context->t1, operator_context->source,
        operator_context->source_cell, operator_context->inner,
        operator_context->in, operator_context->out, upper_output,
        POSITIVE_ROUND_UPPER);
}

static int exact_absolute_difference_upper(double a, double b, double *upper)
{
    if (!upper || !isfinite(a) || !isfinite(b)) return -1;
    double negative_b = -b;
    double difference = a + negative_b;
    if (!isfinite(difference)) return -1;
    double b_virtual = difference - a;
    double error = (a - (difference - b_virtual)) +
                   (negative_b - b_virtual);
    return positive_add_bound(fabs(difference), fabs(error),
                              POSITIVE_ROUND_UPPER, upper);
}

static int exact_componentwise_error_envelope(
    const ExactProblem *p, size_t cells,
    const double *dt1, const double *t1,
    const double *chi_tot, const double *chi_es, const double *S_fixed,
    double *inner_physical, const double *J,
    double *source, double *source_cell, double *in, double *out,
    double *work, double *componentwise_error_upper,
    double max_scattering_ratio, size_t refinement_iterations,
    CMFExactReport *report)
{
    /* First enclose F(J) with the same positive formal object.  The lower
     * result is temporarily stored in componentwise_error_upper; work stores
     * the upper result and later the supersolution candidate. */
    for (int pass = -1; pass <= 1; pass += 2) {
        PositiveRounding rounding = pass < 0
                                  ? POSITIVE_ROUND_LOWER
                                  : POSITIVE_ROUND_UPPER;
        if (p->boundary && p->boundary->kind == CMF_EXACT_INNER_DIFFUSION &&
            exact_fill_inner(p,0.0,0.0,J,0,rounding,inner_physical) != 0)
            return -1;
        for (size_t idx = 0; idx < cells; ++idx) {
            double ratio = exact_scattering_coefficient(p,idx);
            double scattering;
            if (positive_multiply_bound(ratio, J[idx], rounding,
                                        &scattering) != 0 ||
                positive_add_bound(S_fixed[idx], scattering, rounding,
                                   &source[idx]) != 0 ||
                positive_multiply_bound(exact_source_weight(p,idx),source[idx],rounding,
                                        &source[idx]) != 0 ||
                positive_multiply_bound(exact_emission_factor(dt1[idx],t1[idx],1.0,p->raw_emissivity), source[idx], rounding,
                                        &source_cell[idx]) != 0)
                return -1;
        }
        double *destination = pass < 0 ? componentwise_error_upper : work;
        if (exact_formal_sweep_bound(
                p, dt1, t1, source, source_cell, inner_physical,
                in, out, destination, rounding) != 0) return -1;
    }

    double max_residual = 0.0;
    for (size_t idx = 0; idx < cells; ++idx) {
        double lower = componentwise_error_upper[idx];
        double upper = work[idx];
        double lower_distance, upper_distance;
        if (!(lower >= 0.0) || !(upper >= lower) ||
            exact_absolute_difference_upper(lower, J[idx],
                                            &lower_distance) != 0 ||
            exact_absolute_difference_upper(upper, J[idx],
                                            &upper_distance) != 0)
            return -1;
        double residual = lower_distance > upper_distance
                        ? lower_distance : upper_distance;
        componentwise_error_upper[idx] = residual;
        if (residual > max_residual) max_residual = residual;
    }
    if (!(max_scattering_ratio >= 0.0) ||
        !(max_scattering_ratio < 1.0) || !isfinite(max_residual)) return -1;

    double seed;
    if (max_residual == 0.0) {
        seed = 0.0;
    } else {
        double denominator = 1.0 - max_scattering_ratio;
        if (!(denominator > 0.0)) return -1;
        denominator = nextafter(denominator, 0.0);
        if (!(denominator > 0.0)) return -1;
        seed = max_residual / denominator;
        if (!isfinite(seed)) return -1;
        seed = nextafter(seed, INFINITY);
    }
    ExactEnvelopeOperator operator_context = {
        p, dt1, t1, chi_tot, chi_es, source, source_cell,
        inner_physical, in, out, cells
    };
    CMFEnvelopeReport envelope_report;
    CMFEnvelopeStatus envelope_status = CMF_ENVELOPE_INVALID_INPUT;
    size_t seed_attempts = 0;
    for (; seed_attempts < 64U; ++seed_attempts) {
        for (size_t idx = 0; idx < cells; ++idx) work[idx] = seed;
        envelope_status = cmf_error_envelope_verify(
            cells, componentwise_error_upper, work,
            exact_apply_scattering_upper, &operator_context,
            &envelope_report);
        if (envelope_status == CMF_ENVELOPE_OK) break;
        if (envelope_status != CMF_ENVELOPE_NOT_SUPERSOLUTION ||
            !(seed > 0.0) || seed > 0.5 * DBL_MAX) return -1;
        seed *= 2.0;
    }
    if (envelope_status != CMF_ENVELOPE_OK) return -1;
    envelope_status = cmf_error_envelope_refine(
        cells, componentwise_error_upper, work, refinement_iterations,
        exact_apply_scattering_upper, &operator_context, &envelope_report);
    if (envelope_status != CMF_ENVELOPE_OK) return -1;

    double bound_min = INFINITY, bound_max = 0.0;
    for (size_t idx = 0; idx < cells; ++idx) {
        componentwise_error_upper[idx] = work[idx];
        if (work[idx] < bound_min) bound_min = work[idx];
        if (work[idx] > bound_max) bound_max = work[idx];
    }
    if (report) {
        report->componentwise_error_envelope_verified = 1;
        report->componentwise_error_seed_attempts = seed_attempts + 1U;
        report->componentwise_error_refinement_iterations =
            envelope_report.iterations_completed;
        report->componentwise_residual_upper_max = max_residual;
        report->componentwise_error_upper_min = bound_min;
        report->componentwise_error_upper_max = bound_max;
    }
    return 0;
}

static CMFExactStatus cmf_exact_characteristic_solve_impl(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J, double *componentwise_error_upper,
    size_t envelope_refinement_iterations,
    int iteration_cap, double tolerance, CMFExactMode mode,
    CMFExactReport *report, int physical_cells,
    const CMFExactInnerBoundary *boundary,int raw_emissivity)
{
    CMFExactReport local;
    memset(&local, 0, sizeof(local));
    local.status = CMF_EXACT_INVALID_INPUT;
    local.mode = (int)mode;
    local.iteration_cap = iteration_cap;
    local.tolerance = tolerance;
    local.final_max_relative_change = INFINITY;
    local.final_max_absolute_change = INFINITY;
    local.max_scattering_ratio = INFINITY;
    local.fixed_point_absolute_error_bound = INFINITY;
    local.componentwise_residual_upper_max = INFINITY;
    local.componentwise_error_upper_min = INFINITY;
    local.componentwise_error_upper_max = INFINITY;
    local.first_negative_recurrence = NAN;
    if (report) *report = local;
    if (!J || iteration_cap < 2 || !(tolerance > 0.0) ||
        !isfinite(tolerance) ||
        ((!boundary || boundary->kind == CMF_EXACT_INNER_BLACKBODY) &&
         (!(T_inner > 0.0) || !isfinite(T_inner) ||
          !(inner_boundary_scale >= 0.0) || !isfinite(inner_boundary_scale))) ||
        (physical_cells && (!boundary ||
         (boundary->kind != CMF_EXACT_INNER_BLACKBODY &&
          boundary->kind != CMF_EXACT_INNER_DIFFUSION))) ||
        (boundary && boundary->kind == CMF_EXACT_INNER_DIFFUSION &&
         (!boundary->source_fixed || !boundary->scattering_ratio ||
          !boundary->gradient_over_chi)) ||
        (mode != CMF_EXACT_MODE_SLIDING &&
         mode != CMF_EXACT_MODE_DIRECT_REFERENCE &&
         mode != CMF_EXACT_MODE_POSITIVE_SLIDING) ||
        (componentwise_error_upper &&
         mode != CMF_EXACT_MODE_POSITIVE_SLIDING) ||
        (raw_emissivity && (!physical_cells || mode==CMF_EXACT_MODE_SLIDING)))
        return CMF_EXACT_INVALID_INPUT;

    ExactProblem p;
    CMFExactStatus status = exact_problem_build(
        &p, n_shells, n_bins, dlognu, nu, r_inner, r_outer,
        time_explosion, chi_tot, chi_es, S_fixed, physical_cells, boundary);
    if (status != CMF_EXACT_OK) {
        local.status = status;
        if (report) *report = local;
        return status;
    }
    p.raw_emissivity=raw_emissivity;
    local.n_rays = (size_t)p.nr;
    local.segment_slots = p.segment_slots;

    size_t cells, segment_cells, inner_cells;
    size_t buffer_slots=exact_buffer_rays(&p)*p.ray_stride;
    if (checked_mul_size((size_t)n_shells, (size_t)n_bins, &cells) != 0 ||
        checked_mul_size(buffer_slots, (size_t)n_bins,
                         &segment_cells) != 0 ||
        checked_mul_size(physical_cells ? (size_t)p.nr : 1U,
                         (size_t)n_bins,&inner_cells) != 0 ||
        cells > SIZE_MAX / sizeof(double) ||
        segment_cells > SIZE_MAX / sizeof(double) ||
        inner_cells > SIZE_MAX / sizeof(double)) {
        exact_problem_free(&p);
        local.status = CMF_EXACT_INVALID_INPUT;
        if (report) *report = local;
        return local.status;
    }
    local.ray_buffer_slots=buffer_slots;

    double *dt1 = (double *)malloc(cells * sizeof(double));
    double *t1 = (double *)malloc(cells * sizeof(double));
    double *in = (double *)malloc(segment_cells * sizeof(double));
    double *out = (double *)malloc(segment_cells * sizeof(double));
    double *source = (double *)malloc(cells * sizeof(double));
    double *source_cell = (double *)malloc(cells * sizeof(double));
    double *Jnew = (double *)malloc(cells * sizeof(double));
    double *inner = (double *)malloc(inner_cells * sizeof(double));
    if (!dt1 || !t1 || !in || !out || !source || !source_cell ||
        !Jnew || !inner) {
        free(dt1); free(t1); free(in); free(out); free(source);
        free(source_cell); free(Jnew); free(inner); exact_problem_free(&p);
        local.status = CMF_EXACT_ALLOCATION_FAILED;
        if (report) *report = local;
        return local.status;
    }

    int invalid = 0;
    double max_scattering_ratio = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:invalid) \
    reduction(max:max_scattering_ratio)
#endif
    for (size_t idx = 0; idx < cells; ++idx) {
        double ct = chi_tot[idx], ce = chi_es[idx], sf = S_fixed[idx], j = J[idx];
        /* ce may exceed ct when a mild, explicitly signed line opacity
         * (-0.5 <= tau < 0) subtracts from otherwise positive extinction.
         * That is not invalid by itself; the actual extinction must stay
         * nonnegative and the iteration must still converge. */
        if ((!raw_emissivity && (!(ct >= 0.0) || (ct == 0.0 && ce != 0.0) ||
            (physical_cells && ct == 0.0 && sf != 0.0))) || !(ce >= 0.0) ||
            !(sf >= 0.0) || !(j >= 0.0) || !isfinite(ct) || !isfinite(ce) ||
            !isfinite(sf) || !isfinite(j)) {
            invalid = 1;
            dt1[idx] = t1[idx] = 0.0;
        } else {
            double ratio = raw_emissivity ? 0 : (ct > 0.0 ? ce / ct : 0.0);
            if (ratio > max_scattering_ratio)
                max_scattering_ratio = ratio;
            double depth = (ct + p.a_lam * (physical_cells ? 3.0 : 4.0)) / p.a_drift;
            if ((!raw_emissivity && !(depth >= 0.0)) || !isfinite(depth)) invalid = 1;
            dt1[idx] = depth;
            t1[idx] = exp(-depth);
            if(!isfinite(t1[idx]) ||
                !isfinite(exact_emission_factor(depth,t1[idx],1.0,raw_emissivity)))invalid=1;
        }
    }
    if (invalid) {
        status = CMF_EXACT_NONFINITE;
        goto cleanup;
    }
    if (boundary && boundary->kind == CMF_EXACT_INNER_DIFFUSION) {
        for (int b = 0; b < n_bins; ++b) {
            double ratio = boundary->scattering_ratio[b];
            if (!(ratio >= 0.0) || !isfinite(ratio) ||
                !(boundary->source_fixed[b] >= 0.0) ||
                !isfinite(boundary->source_fixed[b]) ||
                !(boundary->gradient_over_chi[b] >= 0.0) ||
                !isfinite(boundary->gradient_over_chi[b])) {
                status = CMF_EXACT_INVALID_INPUT;
                goto cleanup;
            }
            if (ratio > max_scattering_ratio) max_scattering_ratio = ratio;
        }
    }
    if(raw_emissivity){
        /* 증폭이 있으면 max(χ_es/χ)는 K의 상한이 아니다. 같은 실제
         * 경계·수송의 양의 K(1)을 상향 평가해 반복 연산의 norm을 구한다. */
        for(size_t idx=0;idx<cells;idx++)Jnew[idx]=1.0;
        ExactEnvelopeOperator op={&p,dt1,t1,chi_tot,chi_es,source,source_cell,
            inner,in,out,cells};
        if(exact_apply_scattering_upper(Jnew,Jnew,cells,&op)!=0){
            status=CMF_EXACT_ERROR_ENVELOPE_FAILED;goto cleanup;
        }
        max_scattering_ratio=0;
        for(size_t idx=0;idx<cells;idx++)
            if(Jnew[idx]>max_scattering_ratio)max_scattering_ratio=Jnew[idx];
    }
    local.max_scattering_ratio = max_scattering_ratio;

    for (int k = 0; k < p.nr; ++k) {
        double drift = 0.0;
        size_t base = (size_t)k * p.ray_stride;
        for (int i = 0; i < p.rn[k]; ++i) {
            double ds = physical_cells
                ? p.half_outer[base+i]+p.half_inner[base+i]
                : (i + 1 < p.rn[k]
                    ? p.rz[base + i] - p.rz[base + i + 1]
                    : p.rz[base + i] - p.rzin[k]);
            drift += p.a_drift * ds;
        }
        if (drift > local.max_characteristic_drift_bins)
            local.max_characteristic_drift_bins = drift;
    }

    status = CMF_EXACT_NOT_CONVERGED;
    for (int iteration = 0; iteration < iteration_cap; ++iteration) {
        if (exact_fill_inner(&p,T_inner,inner_boundary_scale,J,0,
                             POSITIVE_ROUND_NEAREST,inner) != 0) {
            status = CMF_EXACT_NONFINITE;
            break;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t idx = 0; idx < cells; ++idx) {
            double ratio = exact_scattering_coefficient(&p,idx);
            source[idx] = exact_source_weight(&p,idx)*(S_fixed[idx] + ratio * J[idx]);
            source_cell[idx] = exact_emission_factor(dt1[idx],t1[idx],1.0,raw_emissivity)*source[idx];
        }

        uint64_t negative_count = 0;
        double first_negative = NAN;
        ExactSweepStatus sweep_status = exact_formal_sweep(
            &p, dt1, t1, source, source_cell, inner, in, out, Jnew, mode,
            &negative_count, &first_negative);
        if (sweep_status == EXACT_SWEEP_NEGATIVE) {
            local.negative_recurrence_count += negative_count;
            local.first_negative_recurrence = first_negative;
            status = CMF_EXACT_NEGATIVE_RECURRENCE;
            break;
        }
        if (sweep_status == EXACT_SWEEP_ALLOCATION_FAILED) {
            status = CMF_EXACT_ALLOCATION_FAILED;
            break;
        }
        if (sweep_status != EXACT_SWEEP_OK) {
            status = CMF_EXACT_NONFINITE;
            break;
        }

        double max_relative = 0.0;
        double max_absolute = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(max:max_relative,max_absolute)
#endif
        for (size_t idx = 0; idx < cells; ++idx) {
            double absolute = fabs(Jnew[idx] - J[idx]);
            double scale = fabs(J[idx]);
            double relative = scale > 0.0 ? absolute / scale
                : (absolute == 0.0 ? 0.0 : INFINITY);
            if (relative > max_relative) max_relative = relative;
            if (absolute > max_absolute) max_absolute = absolute;
            J[idx] = Jnew[idx];
        }
        local.iterations_used = iteration + 1;
        local.final_max_relative_change = max_relative;
        local.final_max_absolute_change = max_absolute;
        if (max_scattering_ratio < 1.0) {
            double factor = max_scattering_ratio == 0.0 ? 0.0
                : max_scattering_ratio / (1.0 - max_scattering_ratio);
            local.fixed_point_absolute_error_bound = factor * max_absolute;
        } else {
            local.fixed_point_absolute_error_bound = INFINITY;
        }
        if (iteration > 0 && max_relative < tolerance) {
            status = CMF_EXACT_OK;
            break;
        }
    }

    if (status == CMF_EXACT_OK && componentwise_error_upper &&
        exact_componentwise_error_envelope(
            &p, cells, dt1, t1, chi_tot, chi_es, S_fixed, inner, J,
            source, source_cell, in, out, Jnew, componentwise_error_upper,
            max_scattering_ratio, envelope_refinement_iterations,
            &local) != 0) {
        status = CMF_EXACT_ERROR_ENVELOPE_FAILED;
    }

cleanup:
    free(dt1); free(t1); free(in); free(out); free(source);
    free(source_cell); free(Jnew); free(inner); exact_problem_free(&p);
    local.status = status;
    if (report) *report = local;
    return status;
}


CMFExactStatus cmf_exact_characteristic_solve(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J,
    int iteration_cap, double tolerance, CMFExactMode mode,
    CMFExactReport *report)
{
    return cmf_exact_characteristic_solve_impl(
        n_shells, n_bins, dlognu, nu, r_inner, r_outer, time_explosion,
        T_inner, inner_boundary_scale, chi_tot, chi_es, S_fixed, J,
        NULL, 0, iteration_cap, tolerance, mode, report, 0, NULL,0);
}

CMFExactStatus cmf_exact_characteristic_solve_with_envelope(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J, double *componentwise_error_upper,
    size_t envelope_refinement_iterations,
    int iteration_cap, double tolerance, CMFExactMode mode,
    CMFExactReport *report)
{
    if (!componentwise_error_upper) return CMF_EXACT_INVALID_INPUT;
    return cmf_exact_characteristic_solve_impl(
        n_shells, n_bins, dlognu, nu, r_inner, r_outer, time_explosion,
        T_inner, inner_boundary_scale, chi_tot, chi_es, S_fixed, J,
        componentwise_error_upper, envelope_refinement_iterations,
        iteration_cap, tolerance, mode, report, 0, NULL,0);
}

CMFExactStatus cmf_exact_cell_solve(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    const CMFExactInnerBoundary *boundary,
    const double *chi_tot, const double *chi_es, const double *S_fixed,
    double *J, double *componentwise_error_upper,
    size_t envelope_refinement_iterations,
    int iteration_cap, double tolerance, CMFExactMode mode,
    CMFExactReport *report)
{
    if (!boundary) return CMF_EXACT_INVALID_INPUT;
    return cmf_exact_characteristic_solve_impl(
        n_shells,n_bins,dlognu,nu,r_inner,r_outer,time_explosion,
        boundary->temperature,boundary->blackbody_scale,chi_tot,chi_es,S_fixed,J,
        componentwise_error_upper,envelope_refinement_iterations,
        iteration_cap,tolerance,mode,report,1,boundary,0);
}

CMFExactStatus cmf_exact_cell_solve_raw_sources(
    int n_shells,int n_bins,double dlognu,const double *nu,
    const double *r_inner,const double *r_outer,double time_explosion,
    const CMFExactInnerBoundary *boundary,
    const double *chi_tot,const double *chi_es,const double *eta_true,
    double *J,double *componentwise_error_upper,
    size_t envelope_refinement_iterations,
    int iteration_cap,double tolerance,CMFExactMode mode,CMFExactReport *report)
{
    if(!boundary)return CMF_EXACT_INVALID_INPUT;
    return cmf_exact_characteristic_solve_impl(n_shells,n_bins,dlognu,nu,
        r_inner,r_outer,time_explosion,boundary->temperature,boundary->blackbody_scale,
        chi_tot,chi_es,eta_true,J,componentwise_error_upper,envelope_refinement_iterations,
        iteration_cap,tolerance,mode,report,1,boundary,1);
}
