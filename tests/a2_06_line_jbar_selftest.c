#include "line_jbar.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define C_CGS 2.99792458e10
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int failures;
#define CHECK(c, l) do { if (!(c)) { \
    fprintf(stderr, "A2_06_LINE_JBAR_FAIL %s line=%d\n", l, __LINE__); \
    failures++; } } while (0)

static double rel_err(double a, double b) { return fabs(a - b) / fabs(b); }

/* independent numeric quadrature of the segment integral */
static double quad(double nul, double nu0, double nu1, double e0, double e1,
                   double L)
{
    double dD = nul * (LINE_JBAR_VDOPPLER_CMS / C_CGS);
    double norm = sqrt(M_PI) * erf(LINE_JBAR_PROFILE_NDOPPLER) * dD;
    int n = 400000;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double t = (i + 0.5) / n;
        double nu = nu0 + (nu1 - nu0) * t;
        double x = (nu - nul) / dD;
        double phi = fabs(x) <= LINE_JBAR_PROFILE_NDOPPLER
                     ? exp(-x * x) / norm : 0.0;
        s += (e0 + (e1 - e0) * t) * phi;
    }
    return L * s / n;
}

int main(void)
{
    double nul = 1.0e15;
    double dD = nul * (LINE_JBAR_VDOPPLER_CMS / C_CGS);

    /* 1: closed form vs quadrature — segment fully covering the support */
    double got = line_jbar_segment_phi_integral(nul, nul - 10 * dD,
                                                nul + 10 * dD, 2.0, 5.0, 3.0);
    double want = quad(nul, nul - 10 * dD, nul + 10 * dD, 2.0, 5.0, 3.0);
    CHECK(rel_err(got, want) < 1e-6, "cover");
    /* full sweep with constant eps integrates to L*e/(dnu_seg)*[phi mass]:
     * with support fully inside, integral = L*e*1/(nu1-nu0) * ... check via
     * closed identity: Int phi dnu = 1  =>  value = L*e/(nu1-nu0). */
    double got_c = line_jbar_segment_phi_integral(nul, nul - 10 * dD,
                                                  nul + 10 * dD, 4.0, 4.0, 7.0);
    double want_c = 7.0 * 4.0 / (20.0 * dD);
    CHECK(rel_err(got_c, want_c) < 1e-12, "unit-mass");

    /* 2: partial overlaps, both directions, off-centre */
    struct { double a, b; } cases[] = {
        {nul - 2 * dD, nul + 1 * dD}, {nul + 1 * dD, nul - 2 * dD},
        {nul - 6 * dD, nul - 3 * dD}, {nul + 3.5 * dD, nul + 9 * dD},
        {nul - 0.3 * dD, nul + 0.2 * dD},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        got = line_jbar_segment_phi_integral(nul, cases[i].a, cases[i].b,
                                             1.5, 0.5, 2.0);
        want = quad(nul, cases[i].a, cases[i].b, 1.5, 0.5, 2.0);
        if (fabs(want) < 1e-300) CHECK(got == 0.0, "empty");
        else CHECK(rel_err(got, want) < 1e-5, "partial");
    }
    /* static segment (nu0 == nu1) */
    got = line_jbar_segment_phi_integral(nul, nul + dD, nul + dD, 3.0, 1.0, 2.0);
    double norm = sqrt(M_PI) * erf(4.0) * dD;
    CHECK(rel_err(got, 2.0 * 2.0 * exp(-1.0) / norm) < 1e-10, "static");
    /* outside support */
    CHECK(line_jbar_segment_phi_integral(nul, nul + 5 * dD, nul + 6 * dD,
                                         1, 1, 1) == 0.0, "outside");

    /* 3: Q-set build + hash determinism + accumulate/flush variance identity */
    double line_nu_all[5] = {2e15, 1e15, 3e15, 1.5e15, 8e14};
    int map[5] = {0, 1, -1, 2, 3};
    LineJbarQSet q1, q2;
    CHECK(line_jbar_qset_build(&q1, 5, line_nu_all, map, NULL) == 0, "qb");
    CHECK(line_jbar_qset_build(&q2, 5, line_nu_all, map, NULL) == 0, "qb2");
    CHECK(q1.n_q == 4, "qn");
    CHECK(strcmp(q1.q_set_hash, q2.q_set_hash) == 0, "qhash-det");
    CHECK(strlen(q1.q_set_hash) == 64, "qhash-len");
    /* ascending permutation */
    for (size_t i = 1; i < q1.n_q; i++)
        CHECK(q1.line_nu[q1.by_nu[i - 1]] <= q1.line_nu[q1.by_nu[i]], "qsort");

    LineJbarAccumulator acc;
    LineJbarPacketPartial pp;
    CHECK(line_jbar_accumulator_init(&acc, q1.n_q, 3) == 0, "acc");
    CHECK(line_jbar_partial_init(&pp) == 0, "pp");
    /* packet 1: two segments hitting line at 1e15 (q index of line_id==1) */
    size_t qi = 0;
    for (size_t i = 0; i < q1.n_q; i++) if (q1.line_id[i] == 1) qi = i;
    CHECK(line_jbar_segment_add(&q1, &pp, 2, nul - 2 * dD, nul + 2 * dD,
                                1.0, 1.0, 1.0) == 0, "seg1");
    CHECK(line_jbar_segment_add(&q1, &pp, 2, nul - 1 * dD, nul + 1 * dD,
                                2.0, 2.0, 1.0) == 0, "seg2");
    double y1 = quad(nul, nul - 2 * dD, nul + 2 * dD, 1.0, 1.0, 1.0) +
                quad(nul, nul - 1 * dD, nul + 1 * dD, 2.0, 2.0, 1.0);
    CHECK(line_jbar_packet_flush(&acc, &pp) == 0, "flush1");
    /* packet 2: one segment */
    CHECK(line_jbar_segment_add(&q1, &pp, 2, nul - 3 * dD, nul + 3 * dD,
                                0.5, 0.5, 4.0) == 0, "seg3");
    double y2 = quad(nul, nul - 3 * dD, nul + 3 * dD, 0.5, 0.5, 4.0);
    CHECK(line_jbar_packet_flush(&acc, &pp) == 0, "flush2");

    size_t cell = qi * 3 + 2;
    CHECK(rel_err(acc.sum[cell], y1 + y2) < 1e-5, "sum");
    CHECK(rel_err(acc.sumsq[cell], y1 * y1 + y2 * y2) < 1e-5, "sumsq");
    CHECK(acc.count[cell] == 2, "count");
    CHECK(acc.error_latch == 0, "latch");
    /* variance identity vs direct two-packet + zero-packet population N=4:
     * s^2 = (Q - S^2/N)/(N-1) must match numpy-style ddof=1 with two zeros. */
    {
        double S = acc.sum[cell], Q = acc.sumsq[cell];
        double N = 4.0;
        double s2 = (Q - S * S / N) / (N - 1.0);
        double mean = S / N;
        double direct = (pow(y1 - mean, 2) + pow(y2 - mean, 2) +
                         2.0 * pow(0.0 - mean, 2)) / (N - 1.0);
        CHECK(rel_err(s2, direct) < 1e-9, "variance-identity");
    }
    /* untouched cells stay zero */
    CHECK(acc.sum[qi * 3 + 0] == 0.0 && acc.count[qi * 3 + 0] == 0, "other-shell");

    /* 4: error latch on bad segment */
    CHECK(line_jbar_segment_add(&q1, &pp, 2, -1.0, 1e15, 1, 1, 1) != 0, "bad-seg");

    line_jbar_partial_free(&pp);
    line_jbar_accumulator_free(&acc);
    line_jbar_qset_free(&q1);
    line_jbar_qset_free(&q2);
    if (failures) {
        fprintf(stderr, "A2_06_LINE_JBAR_SELFTEST FAIL failures=%d\n", failures);
        return 1;
    }
    printf("A2_06_LINE_JBAR_SELFTEST PASS\n");
    return 0;
}
