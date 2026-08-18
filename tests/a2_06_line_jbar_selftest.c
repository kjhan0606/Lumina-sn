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

    /* Discrete deterministic profile: physical average stays nearest, while
     * the nonnegative componentwise error is accumulated outward. */
    {
        enum { PNS = 2, PNB = 129 };
        double pnu[PNB], pdnu[PNB], pvalue[PNS * PNB];
        double perror[PNS * PNB], paverage[PNS], pbound[PNS];
        double pbound_zero[PNS], zero[PNS * PNB];
        double pdlog = (LINE_JBAR_VDOPPLER_CMS / C_CGS) / 12.0;
        for (int i = 0; i < PNB; ++i) {
            pnu[i] = nul * exp(((double)i - 64.0) * pdlog);
            pdnu[i] = pnu[i] * pdlog;
            for (int s = 0; s < PNS; ++s) {
                size_t at = (size_t)s * PNB + (size_t)i;
                pvalue[at] = 3.0 + 0.01 * i + 0.25 * s;
                perror[at] = (1.0 + 0.5 * s) *
                             (1.0e-12 + 2.0e-15 * i);
                zero[at] = 0.0;
            }
        }
        LineJbarProfileReport profile_report;
        CHECK(line_jbar_gaussian_discrete_shells(
                  PNS, PNB, pnu, pdnu, pvalue, perror, nul,
                  LINE_JBAR_VDOPPLER_CMS, LINE_JBAR_PROFILE_NDOPPLER,
                  paverage, pbound, &profile_report) ==
                  LINE_JBAR_PROFILE_OK,
              "discrete-profile-status");
        CHECK(profile_report.contributing_bins > 80 &&
              profile_report.first_bin < profile_report.last_bin &&
              profile_report.weight_sum_lower > 0.0 &&
              profile_report.weight_sum_lower <= profile_report.weight_sum,
              "discrete-profile-support");
        for (int s = 0; s < PNS; ++s) {
            long double denominator = 0.0L, numerator = 0.0L;
            long double error_numerator = 0.0L;
            double nearest_denominator = 0.0, nearest_numerator = 0.0;
            for (int i = 0; i < PNB; ++i) {
                double x = (pnu[i] - nul) / dD;
                if (fabs(x) > LINE_JBAR_PROFILE_NDOPPLER) continue;
                double weight = exp(-x * x) * pdnu[i];
                denominator += (long double)weight;
                numerator += (long double)weight *
                             (long double)pvalue[(size_t)s * PNB + i];
                nearest_denominator += weight;
                nearest_numerator +=
                    weight * pvalue[(size_t)s * PNB + i];
                error_numerator += (long double)weight *
                    (long double)perror[(size_t)s * PNB + i];
            }
            long double average_oracle = numerator / denominator;
            long double bound_oracle = error_numerator / denominator;
            CHECK(fabsl((long double)paverage[s] - average_oracle) /
                  average_oracle < 2.0e-15L,
                  "discrete-profile-nearest-oracle");
            CHECK(paverage[s] == nearest_numerator / nearest_denominator,
                  "discrete-profile-physical-bit-identity");
            CHECK((long double)pbound[s] >= bound_oracle,
                  "discrete-profile-outward-bound");
            CHECK(pbound[s] >= 0.0 && isfinite(pbound[s]),
                  "discrete-profile-bound-finite");
        }
        CHECK(line_jbar_gaussian_discrete_shells(
                  PNS, PNB, pnu, pdnu, pvalue, zero, nul,
                  LINE_JBAR_VDOPPLER_CMS, LINE_JBAR_PROFILE_NDOPPLER,
                  paverage, pbound_zero, NULL) == LINE_JBAR_PROFILE_OK &&
              pbound_zero[0] == 0.0 && pbound_zero[1] == 0.0,
              "discrete-profile-exact-zero-error");
        double saved = perror[64];
        perror[64] = -1.0;
        CHECK(line_jbar_gaussian_discrete_shells(
                  PNS, PNB, pnu, pdnu, pvalue, perror, nul,
                  LINE_JBAR_VDOPPLER_CMS, LINE_JBAR_PROFILE_NDOPPLER,
                  paverage, pbound, NULL) == LINE_JBAR_PROFILE_NONFINITE,
              "discrete-profile-negative-error-rejected");
        perror[64] = saved;
        CHECK(line_jbar_gaussian_discrete_shells(
                  PNS, PNB - 50, pnu + 50, pdnu + 50,
                  pvalue + 50, perror + 50, nul,
                  LINE_JBAR_VDOPPLER_CMS, LINE_JBAR_PROFILE_NDOPPLER,
                  paverage, pbound, NULL) == LINE_JBAR_PROFILE_UNCOVERED,
              "discrete-profile-truncated-support-rejected");
    }

    /* 3: Q-set build + hash determinism + accumulate/flush variance identity */
    double line_nu_all[5] = {2e15, 1e15, 3e15, 1.5e15, 8e14};
    int map[5] = {0, 1, -1, 2, 3};
    LineJbarQSet q1, q2;
    CHECK(line_jbar_qset_build(&q1, 5, line_nu_all, map, NULL) == 0, "qb");
    CHECK(line_jbar_qset_build(&q2, 5, line_nu_all, map, NULL) == 0, "qb2");
    CHECK(q1.n_q == 4, "qn");
    CHECK(strcmp(q1.q_set_hash, q2.q_set_hash) == 0, "qhash-det");
    CHECK(strlen(q1.q_set_hash) == 64, "qhash-len");
    CHECK(q1.domain_contract_hash[0] == '\0', "unfiltered-domain-empty");
    /* ascending permutation */
    for (size_t i = 1; i < q1.n_q; i++)
        CHECK(q1.line_nu[q1.by_nu[i - 1]] <= q1.line_nu[q1.by_nu[i]], "qsort");

    /* Canonical BB_IN_DOMAIN graph: closed endpoints, mapped lines only, and
     * Q identity bound to the edge hash even when the selected IDs coincide. */
    {
        double dnu[5] = {
            LINE_JBAR_BB_NU_MIN_HZ,
            LINE_JBAR_BB_NU_MAX_HZ,
            nextafter(LINE_JBAR_BB_NU_MIN_HZ, 0.0),
            nextafter(LINE_JBAR_BB_NU_MAX_HZ, INFINITY),
            sqrt(LINE_JBAR_BB_NU_MIN_HZ * LINE_JBAR_BB_NU_MAX_HZ)
        };
        int dmap[5] = {0, 1, 2, 3, -1};
        uint8_t mask[5];
        size_t inside = 0, outside = 0;
        CHECK(line_jbar_bb_domain_mask_build(mask, 5, dnu, dmap,
                                             &inside, &outside) == 0,
              "domain-mask");
        CHECK(inside == 2 && outside == 2, "domain-count");
        CHECK(mask[0] == 1 && mask[1] == 1 && mask[2] == 0 &&
              mask[3] == 0 && mask[4] == 0, "domain-membership");
        CHECK(line_jbar_frequency_in_bb_domain(dnu[0]) &&
              line_jbar_frequency_in_bb_domain(dnu[1]), "domain-closed");

        LineJbarQSet filtered, same_ids_unbound;
        int same_map[5] = {0, 1, -1, -1, -1};
        CHECK(line_jbar_qset_build(&filtered, 5, dnu, dmap, mask) == 0,
              "domain-qset");
        CHECK(line_jbar_qset_build(&same_ids_unbound, 5, dnu, same_map,
                                   NULL) == 0, "unbound-qset");
        CHECK(filtered.n_q == 2 && same_ids_unbound.n_q == 2,
              "domain-qcount");
        CHECK(strcmp(filtered.domain_contract_hash,
                     LINE_JBAR_BB_DOMAIN_CONTRACT_SHA256) == 0,
              "domain-contract-hash");
        CHECK(strcmp(filtered.q_set_hash, same_ids_unbound.q_set_hash) != 0,
              "domain-hash-binding");
        CHECK(filtered.set_kind == LINE_JBAR_SET_RATE_GRAPH,
              "qg-role");
        CHECK(strcmp(filtered.q_set_hash,
              "ae6163fee5e036e2d751ba19559704401f6734338c413dbedc3b7517e97e1a30")
              == 0, "qg-hash-backward-identity");

        /* Q_E includes every registered in-domain line, even the unmapped
         * centre line.  Q_g must be a checked subset, never a second cache. */
        LineJbarESet energy, energy2, missing;
        CHECK(line_jbar_eset_build(&energy, 5, dnu) == 0,
              "energy-set");
        CHECK(line_jbar_eset_build(&energy2, 5, dnu) == 0,
              "energy-set-repeat");
        CHECK(energy.set_kind == LINE_JBAR_SET_ENERGY_DOMAIN &&
              energy.n_q == 3, "energy-role-count");
        CHECK(energy.line_id[0] == 0 && energy.line_id[1] == 1 &&
              energy.line_id[2] == 4, "energy-membership");
        CHECK(strcmp(energy.q_set_hash,
              "f781482b70a921a3e780e8ae8e111cabe41117d55c72e3aa1d1c5e3668ae1720")
              == 0, "energy-hash-known-answer");
        CHECK(strcmp(energy.q_set_hash, energy2.q_set_hash) == 0,
              "energy-hash-deterministic");
        size_t missing_q = SIZE_MAX;
        CHECK(line_jbar_qset_subset_of_eset(&filtered, &energy,
                                             &missing_q) ==
              LINE_JBAR_SUBSET_OK && missing_q == SIZE_MAX,
              "qg-subset-qe");

        char saved_hash[65];
        memcpy(saved_hash, filtered.q_set_hash, sizeof(saved_hash));
        filtered.q_set_hash[0] = filtered.q_set_hash[0] == 'a' ? 'b' : 'a';
        CHECK(line_jbar_qset_subset_of_eset(&filtered, &energy,
                                             &missing_q) ==
              LINE_JBAR_SUBSET_HASH_MISMATCH,
              "qg-corrupt-hash-rejected");
        memcpy(filtered.q_set_hash, saved_hash, sizeof(saved_hash));

        double saved_nu = energy.line_nu[1];
        energy.line_nu[1] = nextafter(saved_nu, INFINITY);
        CHECK(line_jbar_qset_subset_of_eset(&filtered, &energy,
                                             &missing_q) ==
              LINE_JBAR_SUBSET_FREQUENCY_MISMATCH && missing_q == 1,
              "qg-qe-frequency-mismatch");
        energy.line_nu[1] = saved_nu;

        double missing_nu[5];
        memcpy(missing_nu, dnu, sizeof(missing_nu));
        missing_nu[1] = nextafter(LINE_JBAR_BB_NU_MAX_HZ, INFINITY);
        CHECK(line_jbar_eset_build(&missing, 5, missing_nu) == 0,
              "missing-energy-build");
        CHECK(line_jbar_qset_subset_of_eset(&filtered, &missing,
                                             &missing_q) ==
              LINE_JBAR_SUBSET_MISSING_LINE && missing_q == 1,
              "qg-missing-from-qe");
        double width = LINE_JBAR_PROFILE_NDOPPLER *
                       LINE_JBAR_VDOPPLER_CMS / C_CGS;
        size_t bad_q = SIZE_MAX;
        CHECK(line_jbar_qset_profile_support_covered(
                  &filtered, LINE_JBAR_BB_NU_MIN_HZ * (1.0 - width),
                  LINE_JBAR_BB_NU_MAX_HZ * (1.0 + width), &bad_q) == 0,
              "profile-support-covered");
        CHECK(bad_q == SIZE_MAX, "profile-support-no-bad-q");
        CHECK(line_jbar_qset_profile_support_covered(
                  &filtered,
                  nextafter(LINE_JBAR_BB_NU_MIN_HZ * (1.0 - width), INFINITY),
                  LINE_JBAR_BB_NU_MAX_HZ * (1.0 + width), &bad_q) != 0,
              "profile-support-red-truncation");
        CHECK(bad_q == 0, "profile-support-red-witness");
        line_jbar_qset_free(&filtered);
        line_jbar_qset_free(&same_ids_unbound);
        line_jbar_qset_free(&energy);
        line_jbar_qset_free(&energy2);
        line_jbar_qset_free(&missing);
    }

    LineJbarAccumulator acc;
    LineJbarPacketPartial pp;
    LineJbarAccumulator overflow_acc;
    memset(&overflow_acc, 0, sizeof(overflow_acc));
    CHECK(line_jbar_accumulator_init(&overflow_acc, SIZE_MAX, 2) != 0 &&
          overflow_acc.sum == NULL && overflow_acc.sumsq == NULL &&
          overflow_acc.count == NULL,
          "accumulator-size-overflow-fail-closed");
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
