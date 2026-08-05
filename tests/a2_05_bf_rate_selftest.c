#include "bf_rate_jnu.h"
#include "radiation_field.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define H_CGS 6.62607015e-27

static int failures;

#define CHECK(condition, label) do { \
    if (!(condition)) { \
        fprintf(stderr, "A2_05_BF_RATE_FAIL %s line=%d\n", label, __LINE__); \
        failures++; \
    } \
} while (0)

static double rel_err(double got, double want)
{
    return fabs(got - want) / fabs(want);
}

/* Hand-built view on a private log grid: the integrator contract is on the
 * view alone (the 4000-bin guarantee is radiation_field_read_view's job). */
typedef struct {
    size_t n_bins;
    double *edges;
    double *J;
    RadiationFieldValidityState *validity;
    uint64_t *count;
    RadiationFieldView view;
} TestField;

static void test_field_init(TestField *tf, size_t n_bins,
                            double nu_lo, double nu_hi)
{
    tf->n_bins = n_bins;
    tf->edges = (double *)malloc((n_bins + 1) * sizeof(double));
    tf->J = (double *)calloc(n_bins, sizeof(double));
    tf->validity = (RadiationFieldValidityState *)
        malloc(n_bins * sizeof(*tf->validity));
    tf->count = (uint64_t *)calloc(n_bins, sizeof(uint64_t));
    double step = log(nu_hi / nu_lo) / (double)n_bins;
    for (size_t b = 0; b <= n_bins; ++b)
        tf->edges[b] = nu_lo * exp(step * (double)b);
    for (size_t b = 0; b < n_bins; ++b) {
        tf->validity[b] = RADIATION_FIELD_VALID;
        tf->count[b] = 1;
    }
    tf->view.n_shells = 1;
    tf->view.n_bins = n_bins;
    tf->view.frequency_bin_edges = tf->edges;
    tf->view.J_nu = tf->J;
    tf->view.validity = tf->validity;
    tf->view.count = tf->count;
    tf->view.generation = 7;
}

static void test_field_free(TestField *tf)
{
    free(tf->edges); free(tf->J); free(tf->validity); free(tf->count);
}

/* Dense log-spaced tabulation of a smooth cross-section so the integrator's
 * piecewise-linear reading stays within 1e-6 of the smooth closed form. */
static void tabulate_sigma(double **nu_out, double **sigma_out, size_t n,
                           double nu_lo, double nu_hi,
                           double (*fn)(double, double, double),
                           double sigma0, double nu_th)
{
    double *nu = (double *)malloc(n * sizeof(double));
    double *sg = (double *)malloc(n * sizeof(double));
    double step = log(nu_hi / nu_lo) / (double)(n - 1);
    for (size_t i = 0; i < n; ++i) {
        nu[i] = nu_lo * exp(step * (double)i);
        sg[i] = fn(nu[i], sigma0, nu_th);
    }
    nu[0] = nu_lo;
    nu[n - 1] = nu_hi;
    *nu_out = nu;
    *sigma_out = sg;
}

static double sigma_const(double nu, double sigma0, double nu_th)
{
    (void)nu; (void)nu_th;
    return sigma0;
}

static double sigma_hydrogenic(double nu, double sigma0, double nu_th)
{
    double x = nu_th / nu;
    return sigma0 * x * x * x;
}

static void run_integrator_tests(void)
{
    const double NU_TH = 1.0e15;
    const double SIGMA0 = 6.3e-18;
    const double J0 = 1.0e-4;

    TestField tf;
    test_field_init(&tf, 512, 0.5e15, 5.0e16);

    /* --- Case 1: constant sigma, constant J, threshold mid-grid.
     * Gamma = 4*pi*J0*sigma0/h * ln(nu_top/nu_th) exactly (the linear-segment
     * log formula is exact for constant sigma, partial threshold bin included). */
    double sig2_nu[2] = {NU_TH, 6.0e16};
    double sig2_sg[2] = {SIGMA0, SIGMA0};
    BfCrossSection cs = {2, sig2_nu, sig2_sg, NU_TH};
    for (size_t b = 0; b < tf.n_bins; ++b) tf.J[b] = J0;
    BfRateResult r;
    CHECK(bf_rate_gamma_from_view(&tf.view, 0, &cs, &r) == 0, "c1-rc");
    CHECK(r.state == BF_RATE_VALID, "c1-state");
    double want1 = 4.0 * M_PI * J0 * SIGMA0 / H_CGS *
                   log(tf.edges[tf.n_bins] / NU_TH);
    CHECK(rel_err(r.gamma, want1) < 1.0e-12, "c1-closed-form");
    CHECK(r.w_miss == 0.0, "c1-wmiss");

    /* Threshold sensitivity (E_sym witness mechanics): moving nu_th up by one
     * full bin must remove exactly that bin's log span. */
    size_t tb = 0;
    while (tf.edges[tb + 1] <= NU_TH) tb++;
    double nu_th_b = tf.edges[tb + 1];
    BfCrossSection cs_shift = {2, sig2_nu, sig2_sg, nu_th_b};
    BfRateResult rs;
    CHECK(bf_rate_gamma_from_view(&tf.view, 0, &cs_shift, &rs) == 0, "c1s-rc");
    double want_delta = 4.0 * M_PI * J0 * SIGMA0 / H_CGS * log(nu_th_b / NU_TH);
    CHECK(rel_err(r.gamma - rs.gamma, want_delta) < 1.0e-9, "c1s-threshold-shift");

    /* --- Case 2: hydrogenic sigma ~ nu^-3 against power-law J bin averages.
     * Expected value assembled from independent closed forms per bin:
     *   K_b = sigma0*nu_th^3/(3h) * (a^-3 - b^-3),  J_b = avg of J0*(nu_th/nu)^2. */
    double *hnu, *hsg;
    size_t HPTS = 20000;
    tabulate_sigma(&hnu, &hsg, HPTS, NU_TH, 6.0e16, sigma_hydrogenic,
                   SIGMA0, NU_TH);
    BfCrossSection csh = {HPTS, hnu, hsg, NU_TH};
    double want2 = 0.0;
    for (size_t b = 0; b < tf.n_bins; ++b) {
        double lo = tf.edges[b], hi = tf.edges[b + 1];
        double jb = J0 * NU_TH * NU_TH * (1.0 / lo - 1.0 / hi) / (hi - lo);
        tf.J[b] = jb;
        double a = lo > NU_TH ? lo : NU_TH;
        if (a >= hi) continue;
        double kb = SIGMA0 * NU_TH * NU_TH * NU_TH / (3.0 * H_CGS) *
                    (1.0 / (a * a * a) - 1.0 / (hi * hi * hi));
        want2 += 4.0 * M_PI * jb * kb;
    }
    BfRateResult r2;
    CHECK(bf_rate_gamma_from_view(&tf.view, 0, &csh, &r2) == 0, "c2-rc");
    CHECK(r2.state == BF_RATE_VALID, "c2-state");
    CHECK(rel_err(r2.gamma, want2) < 1.0e-6, "c2-hydrogenic-closed-form");

    /* --- Case 3: validity contract (R6). */
    for (size_t b = 0; b < tf.n_bins; ++b) tf.J[b] = J0;

    /* 3a: heavy UNSAMPLED weight -> UNSAMPLED, no value published. */
    tf.validity[tb + 3] = RADIATION_FIELD_UNSAMPLED;
    BfRateResult r3;
    CHECK(bf_rate_gamma_from_view(&tf.view, 0, &cs, &r3) == 0, "c3a-rc");
    CHECK(r3.state == BF_RATE_UNSAMPLED, "c3a-state");
    CHECK(r3.gamma == 0.0, "c3a-no-value");
    CHECK(r3.w_miss > BF_RATE_W_MISS_TOLERANCE, "c3a-wmiss");

    /* 3b: STALE outranks UNSAMPLED. */
    tf.validity[tb + 5] = RADIATION_FIELD_STALE;
    CHECK(bf_rate_gamma_from_view(&tf.view, 0, &cs, &r3) == 0, "c3b-rc");
    CHECK(r3.state == BF_RATE_STALE, "c3b-precedence");
    tf.validity[tb + 3] = RADIATION_FIELD_VALID;
    tf.validity[tb + 5] = RADIATION_FIELD_VALID;

    /* 3c: negligible missing weight (hydrogenic far tail) -> VALID with
     * recorded w_miss, poisoned bin's J excluded from the integral. */
    for (size_t b = 0; b < tf.n_bins; ++b) {
        double lo = tf.edges[b], hi = tf.edges[b + 1];
        tf.J[b] = J0 * NU_TH * NU_TH * (1.0 / lo - 1.0 / hi) / (hi - lo);
    }
    size_t tail = tf.n_bins - 2;
    tf.validity[tail] = RADIATION_FIELD_UNSAMPLED;
    BfRateResult r3c;
    CHECK(bf_rate_gamma_from_view(&tf.view, 0, &csh, &r3c) == 0, "c3c-rc");
    CHECK(r3c.state == BF_RATE_VALID, "c3c-state");
    CHECK(r3c.w_miss > 0.0 && r3c.w_miss <= BF_RATE_W_MISS_TOLERANCE,
          "c3c-wmiss-band");
    CHECK(r3c.gamma < r2.gamma, "c3c-excludes-poisoned-bin");
    tf.validity[tail] = RADIATION_FIELD_VALID;

    /* 3d: EXACT_ZERO across the whole integration range. */
    for (size_t b = 0; b < tf.n_bins; ++b) {
        tf.validity[b] = RADIATION_FIELD_EXACT_ZERO;
        tf.count[b] = 3;
    }
    BfRateResult r3d;
    CHECK(bf_rate_gamma_from_view(&tf.view, 0, &cs, &r3d) == 0, "c3d-rc");
    CHECK(r3d.state == BF_RATE_EXACT_ZERO, "c3d-state");
    CHECK(r3d.gamma == 0.0, "c3d-zero");
    CHECK(r3d.sample_count > 0, "c3d-count");
    for (size_t b = 0; b < tf.n_bins; ++b) {
        tf.validity[b] = RADIATION_FIELD_VALID;
        tf.count[b] = 1;
    }

    /* 3e: OUT_OF_GRID -- threshold above the grid top. */
    BfCrossSection cs_oog = {2, sig2_nu, sig2_sg, tf.edges[tf.n_bins] * 2.0};
    BfRateResult r3e;
    CHECK(bf_rate_gamma_from_view(&tf.view, 0, &cs_oog, &r3e) == 0, "c3e-rc");
    CHECK(r3e.state == BF_RATE_OUT_OF_GRID, "c3e-state");

    /* --- Case 5: bin-constant (step) sigma encoded as duplicated edge nodes,
     * the representation the legacy-grid migration adapter emits.  Constant J
     * gives the exact per-step closed form sum(sigma_k * ln(hi_k/lo_k)). */
    for (size_t b = 0; b < tf.n_bins; ++b) tf.J[b] = J0;
    {
        double step_nu[6] = {2.0e15, 4.0e15, 4.0e15, 8.0e15, 8.0e15, 1.6e16};
        double step_sg[6] = {3.0e-18, 3.0e-18, 1.0e-18, 1.0e-18, 5.0e-19, 5.0e-19};
        BfCrossSection css = {6, step_nu, step_sg, 2.0e15};
        BfRateResult r5;
        CHECK(bf_rate_gamma_from_view(&tf.view, 0, &css, &r5) == 0, "c5-rc");
        CHECK(r5.state == BF_RATE_VALID, "c5-state");
        double want5 = 4.0 * M_PI * J0 / H_CGS *
                       (3.0e-18 * log(4.0e15 / 2.0e15) +
                        1.0e-18 * log(8.0e15 / 4.0e15) +
                        5.0e-19 * log(1.6e16 / 8.0e15));
        CHECK(rel_err(r5.gamma, want5) < 1.0e-12, "c5-step-closed-form");
    }

    /* --- Case 4: argument contract. */
    CHECK(bf_rate_gamma_from_view(NULL, 0, &cs, &r) == -1, "c4-null-view");
    CHECK(bf_rate_gamma_from_view(&tf.view, 1, &cs, &r) == -1, "c4-bad-shell");
    BfCrossSection cs_bad = {1, sig2_nu, sig2_sg, NU_TH};
    CHECK(bf_rate_gamma_from_view(&tf.view, 0, &cs_bad, &r) == -1, "c4-1pt-sigma");

    free(hnu); free(hsg);
    test_field_free(&tf);
}

static void run_read_view_tests(void)
{
    const double EPOCH = 19.48 * 86400.0;
    const double J0 = 2.5e-5;
    size_t n_shells = 2;
    size_t cells = n_shells * (size_t)LUMINA_RADFIELD_N_BINS;

    RadiationFieldOwner owner;
    CHECK(radiation_field_owner_init(&owner, n_shells) == 0, "rv-init");

    /* Pre-commit the field is unpublished: the view must refuse.  The first
     * mismatch reached is the epoch (init leaves 0), before generation. */
    RadiationFieldView view;
    CHECK(radiation_field_read_view(&owner, EPOCH, n_shells, 1, &view) <
          RADIATION_FIELD_VIEW_OK, "rv-precommit");

    double v_inner[2] = {1.0e8, 1.5e8};
    double v_outer[2] = {1.5e8, 2.0e8};
    double *values = (double *)malloc(cells * sizeof(double));
    RadiationFieldValidityState *validity =
        (RadiationFieldValidityState *)malloc(cells * sizeof(*validity));
    for (size_t i = 0; i < cells; ++i) {
        values[i] = J0;
        validity[i] = RADIATION_FIELD_VALID;
    }
    RadiationFieldCommitRequest request;
    memset(&request, 0, sizeof(request));
    request.provenance_kind = RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
    request.producer = "A2_05_SELFTEST_DETERMINISTIC";
    request.generation = 1;
    request.epoch = EPOCH;
    request.n_shells = n_shells;
    request.v_inner = v_inner;
    request.v_outer = v_outer;
    request.source_n_bins = LUMINA_RADFIELD_N_BINS;
    request.source_frequency_bin_edges = owner.field.frequency_bin_edges.values;
    request.source_J_nu = values;
    request.source_validity = validity;
    request.statistic_kind = RADIATION_FIELD_DETERMINISTIC;
    CHECK(radiation_field_commit(&owner, &request) == 0, "rv-commit");

    CHECK(radiation_field_read_view(&owner, EPOCH, n_shells, 1, &view) ==
          RADIATION_FIELD_VIEW_OK, "rv-ok");
    CHECK(view.n_bins == LUMINA_RADFIELD_N_BINS, "rv-bins");
    CHECK(view.J_nu == owner.field.J_nu.values, "rv-alias");

    /* Every failure mode maps to its own distinct code. */
    CHECK(radiation_field_read_view(&owner, EPOCH + 1.0, n_shells, 1, &view) ==
          RADIATION_FIELD_VIEW_EPOCH_SHELLS, "rv-epoch");
    CHECK(radiation_field_read_view(&owner, EPOCH, n_shells + 1, 1, &view) ==
          RADIATION_FIELD_VIEW_EPOCH_SHELLS, "rv-shells");
    CHECK(radiation_field_read_view(&owner, EPOCH, n_shells, 2, &view) ==
          RADIATION_FIELD_VIEW_STALE_GENERATION, "rv-generation");
    CHECK(radiation_field_read_view(&owner, EPOCH, n_shells, 0, &view) ==
          RADIATION_FIELD_VIEW_STALE_GENERATION, "rv-zero-generation");
    CHECK(radiation_field_read_view(NULL, EPOCH, n_shells, 1, &view) ==
          RADIATION_FIELD_VIEW_DISABLED, "rv-null");
    int saved_enabled = owner.enabled;
    owner.enabled = 0;
    CHECK(radiation_field_read_view(&owner, EPOCH, n_shells, 1, &view) ==
          RADIATION_FIELD_VIEW_DISABLED, "rv-disabled");
    owner.enabled = saved_enabled;

    /* End-to-end: constant-J committed field integrated with constant sigma
     * over the full canonical grid reproduces the log closed form per shell. */
    CHECK(radiation_field_read_view(&owner, EPOCH, n_shells, 1, &view) ==
          RADIATION_FIELD_VIEW_OK, "rv-reok");
    double nu_th = 1.0e15;
    double sig_nu[2] = {nu_th, LUMINA_RADFIELD_NU_MAX_HZ * 1.5};
    double sig_sg[2] = {6.3e-18, 6.3e-18};
    BfCrossSection cs = {2, sig_nu, sig_sg, nu_th};
    double want = 4.0 * M_PI * J0 * 6.3e-18 / H_CGS *
                  log(LUMINA_RADFIELD_NU_MAX_HZ / nu_th);
    for (size_t s = 0; s < n_shells; ++s) {
        BfRateResult r;
        CHECK(bf_rate_gamma_from_view(&view, s, &cs, &r) == 0, "rv-int-rc");
        CHECK(r.state == BF_RATE_VALID, "rv-int-state");
        CHECK(rel_err(r.gamma, want) < 1.0e-9, "rv-int-closed-form");
    }

    free(values); free(validity);
    radiation_field_owner_free(&owner);
}

/* Kramers adapter (bf_rate_gamma_legacy_grid, sigma_row == NULL): with a
 * constant-J view the whole Gamma must match an INDEPENDENT physical closed
 * form: full bins (lo >= nu_th) contribute sigma(nu_c)*ln(hi/lo)/h and the
 * threshold partial bin contributes the exact smooth Kramers integral
 * sigma0*nu_th^3/(3h)*(nu_th^-3 - hi^-3) -- by construction of s*.  Both
 * placements of nu_th (below and above the bin centre) are exercised. */
static void kramers_case(double th_factor, const char *tag)
{
    const double SIGMA0 = 6.3e-18;
    const double J0 = 1.0e-4;
    TestField tf;
    test_field_init(&tf, 512, 0.5e15, 5.0e16);
    for (size_t b = 0; b < tf.n_bins; ++b) tf.J[b] = J0;

    int nfb = 400;
    double nu_min = 0.6e15;
    double dln = log(4.0e16 / nu_min) / nfb;
    int tb = (int)(log(1.0e15 / nu_min) / dln);
    double lo = nu_min * exp(tb * dln), hi = nu_min * exp((tb + 1) * dln);
    double nu_th = sqrt(lo * hi) * th_factor;   /* inside (lo, hi) */
    if (!(nu_th > lo && nu_th < hi)) {
        fprintf(stderr, "A2_05_BF_RATE_FAIL kr-setup-%s\n", tag);
        failures++;
        return;
    }
    double *node_nu = malloc(2 * (size_t)nfb * sizeof(double));
    double *node_sg = malloc(2 * (size_t)nfb * sizeof(double));
    BfRateResult r;
    CHECK(bf_rate_gamma_legacy_grid(&tf.view, 0, nfb, nu_min, dln, NULL,
                                    SIGMA0, nu_th, node_nu, node_sg, &r) == 0,
          "kr-rc");
    CHECK(r.state == BF_RATE_VALID, "kr-state");
    /* Independent expectation: physical closed forms per piece. */
    double top = tf.edges[tf.n_bins];
    double xth = nu_th / hi;
    double want = 4.0 * M_PI * J0 * SIGMA0 / (3.0 * H_CGS) *
                  (1.0 - xth * xth * xth);              /* partial [nu_th,hi] */
    for (int bb = tb + 1; bb < nfb; bb++) {
        double blo = nu_min * exp(bb * dln);
        double bhi = nu_min * exp((bb + 1) * dln);
        double bc = nu_min * exp((bb + 0.5) * dln);
        double c = bhi < top ? bhi : top;
        if (c <= blo) break;
        want += 4.0 * M_PI * J0 / H_CGS *
                SIGMA0 * pow(nu_th / bc, 3.0) * log(c / blo);
    }
    CHECK(rel_err(r.gamma, want) < 1.0e-11, "kr-exact-sum");
    free(node_nu); free(node_sg);
    test_field_free(&tf);
}

static void run_kramers_adapter_test(void)
{
    /* bin half-width is exp(dln/2)-1 ~ 0.53%, so factors must stay inside
     * (0.9948, 1.0053) to remain within the threshold bin. */
    kramers_case(1.003, "above-centre");   /* centre < nu_th < hi */
    kramers_case(0.997, "below-centre");   /* lo < nu_th < centre */
}

/* EXACT_ZERO + small missing weight: with every sigma-weighted bin either
 * EXACT_ZERO or a tiny (w_miss <= tol) UNSAMPLED remainder, the honest state
 * is VALID with gamma 0 and w_miss recorded -- NOT EXACT_ZERO (R6). */
static void run_zero_plus_missing_test(void)
{
    const double SIGMA0 = 6.3e-18;
    TestField tf;
    test_field_init(&tf, 512, 0.5e15, 5.0e16);
    double *hnu, *hsg;
    size_t HPTS = 20000;
    tabulate_sigma(&hnu, &hsg, HPTS, 1.0e15, 6.0e16, sigma_hydrogenic,
                   SIGMA0, 1.0e15);
    BfCrossSection csh = {HPTS, hnu, hsg, 1.0e15};
    for (size_t b = 0; b < tf.n_bins; ++b) {
        tf.J[b] = 0.0;
        tf.validity[b] = RADIATION_FIELD_EXACT_ZERO;
        tf.count[b] = 2;
    }
    tf.validity[tf.n_bins - 2] = RADIATION_FIELD_UNSAMPLED;  /* tiny tail bin */
    tf.count[tf.n_bins - 2] = 0;
    BfRateResult r;
    CHECK(bf_rate_gamma_from_view(&tf.view, 0, &csh, &r) == 0, "zm-rc");
    CHECK(r.state == BF_RATE_VALID, "zm-state-valid-not-exact-zero");
    CHECK(r.gamma == 0.0, "zm-zero-value");
    CHECK(r.w_miss > 0.0 && r.w_miss <= BF_RATE_W_MISS_TOLERANCE, "zm-wmiss");
    free(hnu); free(hsg);
    test_field_free(&tf);
}

int main(void)
{
    run_integrator_tests();
    run_kramers_adapter_test();
    run_zero_plus_missing_test();
    run_read_view_tests();
    if (failures) {
        fprintf(stderr, "A2_05_BF_RATE_SELFTEST FAIL failures=%d\n", failures);
        return 1;
    }
    printf("A2_05_BF_RATE_SELFTEST PASS\n");
    return 0;
}
