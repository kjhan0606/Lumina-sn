#include "line_net_rate.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int fail(const char *reason)
{
    fprintf(stderr, "[LINE-NET][SELFTEST][FAIL] %s\n", reason);
    return 4;
}

static int close_rel(double got, double want, double rel)
{
    return isfinite(got) && isfinite(want) &&
           fabs(got - want) <= rel * fmax(fabs(want), DBL_MIN);
}

static int direct_cmfgen_exponx(double x, double *beta, double *companion)
{
    if (!beta || !companion || !isfinite(x)) return -1;
    if (fabs(x) < 1.0e-3) {
        *beta = 1.0 - x * (0.5 - x / 6.0 * (1.0 - x / 4.0));
    } else if (x < 40.0) {
        *beta = (1.0 - exp(-x)) / x;
    } else {
        *beta = 1.0 / x;
    }
    *companion = fabs(x) < 1.0e-3
        ? 0.5 - x / 6.0 * (1.0 - x / 4.0)
        : (1.0 - *beta) / x;
    return isfinite(*beta) && isfinite(*companion) ? 0 : -1;
}

int main(void)
{
    LineNetResult result, before;

    {
        const double h = 6.62607015e-27;
        const double expected_exact_coefficient = 0.026540088545744744;
        if (line_net_exact_sobolev_coefficient() !=
            expected_exact_coefficient)
            return fail("exact Sobolev coefficient known answer");
        const double sobolev_coefficient = 2.6540281e-2;
        const double f_lu = 0.25;
        const double nu = 2.0e15;
        const double wavelength_cm = 2.99792458e10 / nu;
        const double exact_ratio = 0.75;
        const double B_lu = exact_ratio * sobolev_coefficient * f_lu *
                            wavelength_cm * nu / 2.99792458e10 *
                            LINE_NET_FOUR_PI / (h * nu);
        double ratio = NAN;
        if (line_net_einstein_opacity_ratio(
                sobolev_coefficient, f_lu, wavelength_cm,
                B_lu, nu, &ratio) != 0 ||
            !close_rel(ratio, exact_ratio, 4.0 * DBL_EPSILON))
            return fail("Einstein/Sobolev opacity-ratio identity");
        if (line_net_einstein_opacity_ratio(
                sobolev_coefficient, 0.0, wavelength_cm,
                B_lu, nu, &ratio) == 0 ||
            !isnan(ratio) ||
            line_net_einstein_opacity_ratio(
                sobolev_coefficient, f_lu, 0.0,
                B_lu, nu, &ratio) == 0 ||
            !isnan(ratio))
            return fail("Einstein/Sobolev invalid-input rejection");
    }

    /* Canonical finite CMFGEN fixture v2, SHA256
     * 5a967bbbf6f374c69c6ae5fd63d420d1fadc002c04ddf2fbbef24192a81951a0. */
    double fixture_q = NAN;
    if (line_net_cmfgen_internal_to_cgs(365727380514.8772, 0.00145662,
                                        0.997943, &fixture_q) != 0 ||
        !close_rel(fixture_q, 0.6680659609711768, 8.0 * DBL_EPSILON))
        return fail("CMFGEN finite known answer");

    LineNetSobolevMaterial material;
    if (line_net_sobolev_material(
            2.5e7, 3.0e6, 4.0e14, 0.25, 2.0e11,
            LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK, 1,
            &material) != 0 || material.srce_chk_applied ||
        !(material.raw_integrated_opacity > 0.0) ||
        material.effective_integrated_opacity !=
            material.raw_integrated_opacity ||
        !(material.emission_per_sr > 0.0))
        return fail("positive Sobolev material");
    if (line_net_sobolev_material(
            2.5e7, 3.0e6, 4.0e14, -0.5000000000000001, 2.0e11,
            LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK, 1,
            &material) != 0 || !material.srce_chk_applied ||
        !(material.raw_integrated_opacity < 0.0) ||
        material.effective_integrated_opacity != 1.0e-10)
        return fail("SRCE_CHK threshold/material");
    if (line_net_sobolev_material(
            0.0, 3.0e6, 4.0e14, 0.0, 2.0e11,
            LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK, 1,
            &material) != 0 || !material.exact_zero_provenance ||
        material.emission_per_sr != 0.0 ||
        material.effective_integrated_opacity != 0.0)
        return fail("Sobolev exact-zero provenance");

    /* Pre-registered R2 physical-inversion witness: Fe III line 2164811,
     * shell 0.  Its raw tau remains -0.9581055; only the separately typed
     * CMFGEN benchmark view supplies tau_eff to the non-overlap operator. */
    const double witness_tau = -0.95810554931907788;
    const double witness_nupper = 0.13653029676205097;
    const double witness_Aul = 22516280.0;
    const double witness_nu = 156861500000000.0;
    const double witness_time = 1683072.0;
    if (line_net_sobolev_material(
            witness_nupper, witness_Aul, witness_nu, witness_tau,
            witness_time, LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK, 1,
            &material) != 0 || !material.srce_chk_applied ||
        material.raw_integrated_opacity != -0.002978559783123111 ||
        material.effective_integrated_opacity != 1.0e-10 ||
        material.effective_tau != 3.216673893026498e-08 ||
        material.emission_per_sr != 2.542659490566559e-07)
        return fail("pre-registered Fe III inversion material witness");
    LineNetSobolevRadiation radiation;
    if (line_net_sobolev_radiation(
            &material, 2.5e-8, 3.0e-13, witness_nu, witness_time,
            &radiation) != 0 ||
        radiation.beta != 0.9999999839166307 ||
        radiation.one_minus_beta_over_tau != 0.4999999946388769 ||
        radiation.continuum_term != 2.4999999597915765e-08 ||
        radiation.local_emission_term != 4.089453157232628e-05 ||
        radiation.jbar != 4.09195315719242e-05 ||
        radiation.jbar_absolute_uncertainty != 2.9999999517498923e-13)
        return fail("pre-registered Fe III Sobolev Jbar witness");

    /* Micro-parity against a literal transcription of CMFGEN EXPONX.  The
     * signed fixtures cross the small-x branch, both sides of zero, the
     * ordinary branch, and the x>=40 asymptote. */
    static const double tau_fixture[] = {
        -0.5, -0.125, -0.001, -0.0001, 0.0,
         0.0001, 0.001, 1.0, 40.0
    };
    for (size_t k = 0; k < sizeof(tau_fixture) / sizeof(tau_fixture[0]); ++k) {
        double got_beta, got_companion, direct_beta, direct_companion;
        if (line_net_cmfgen_exponx(
                tau_fixture[k], &got_beta, &got_companion) != 0 ||
            direct_cmfgen_exponx(
                tau_fixture[k], &direct_beta, &direct_companion) != 0 ||
            got_beta != direct_beta || got_companion != direct_companion)
            return fail("CMFGEN EXPONX micro-parity");
    }

    /* Applying the produced Jbar to the existing direct energy bracket must
     * reduce algebraically to CMFGEN's beta*(eta-chi*J_cont), including the
     * physical mild-negative-opacity interval and tau=0 finite limit. */
    static const double net_tau_fixture[] = {-0.5, -0.125, 0.0, 1.0, 40.0};
    for (size_t k = 0;
         k < sizeof(net_tau_fixture) / sizeof(net_tau_fixture[0]); ++k) {
        if (line_net_sobolev_material(
                2.5e7, 3.0e6, 4.0e14, net_tau_fixture[k], 2.0e11,
                LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK, 1,
                &material) != 0 ||
            line_net_sobolev_radiation(
                &material, 2.0e-7, 0.0, 4.0e14, 2.0e11,
                &radiation) != 0)
            return fail("Sobolev direct-bracket fixture build");
        LineNetComponentInput direct_input = {
            .emission_per_sr = material.emission_per_sr,
            .integrated_opacity = material.effective_integrated_opacity,
            .jbar = radiation.jbar,
            .jbar_absolute_uncertainty = 0.0,
            .other_net_absolute_uncertainty_per_sr = 0.0,
            .deck_scale = 1.0,
            .exact_zero_provenance = material.exact_zero_provenance,
        };
        LineNetResult direct_result;
        LineNetStatus direct_status =
            line_net_rate_evaluate(&direct_input, &direct_result);
        double expected_net = radiation.beta *
            fma(-material.effective_integrated_opacity, 2.0e-7,
                material.emission_per_sr);
        if ((direct_status != LINE_NET_OK_COOLING &&
             direct_status != LINE_NET_OK_HEATING) ||
            !close_rel(direct_result.net_per_sr, expected_net,
                       32.0 * DBL_EPSILON))
            return fail("Sobolev Jbar/direct-bracket CMFGEN identity");
    }

    /* Injected provenance/input defects must fail closed; the operator must
     * never turn them into a finite publication. */
    LineNetSobolevMaterial corrupt = material;
    corrupt.effective_tau = NAN;
    memset(&radiation, 0xa5, sizeof(radiation));
    if (line_net_sobolev_radiation(
            &corrupt, 2.5e-8, 3.0e-13, witness_nu, witness_time,
            &radiation) == 0 || radiation.jbar != 0.0 ||
        line_net_sobolev_radiation(
            &material, -2.5e-8, 3.0e-13, witness_nu, witness_time,
            &radiation) == 0 ||
        line_net_cmfgen_exponx(-1000.0, &radiation.beta,
                               &radiation.one_minus_beta_over_tau) == 0)
        return fail("Sobolev injected-defect fail-closed control");

    double scl = 0.0;
    double photon_eV = 6.62607015e-27 * 4.0e14 / 1.602176634e-12;
    if (line_net_cmfgen_scl_ln(
            1.0, 1.0 + 0.997943 * photon_eV, 4.0e14,
            1.0e9, 0.5, 1.0e30, &scl) != 0 ||
        !close_rel(scl, 0.997943, 8.0 * DBL_EPSILON))
        return fail("SCL_LN finite scale");
    if (line_net_cmfgen_scl_ln(
            1.0, 1.0 + 0.25 * photon_eV, 4.0e14,
            1.0e9, 0.5, 1.0e30, &scl) != 0 || scl != 1.0)
        return fail("SCL_LN mismatch fallback");
    if (line_net_cmfgen_scl_ln(
            1.0, 1.0 + 0.997943 * photon_eV, 4.0e14,
            1.0e30, 0.5, 1.0e30, &scl) != 0 || scl != 1.0)
        return fail("SCL_LN density cutoff");

    LineNetComponentInput cooling = {
        .emission_per_sr = 10.0,
        .integrated_opacity = 2.0,
        .jbar = 3.0,
        .jbar_absolute_uncertainty = 0.0,
        .other_net_absolute_uncertainty_per_sr = 0.0,
        .deck_scale = 1.0,
        .exact_zero_provenance = 0
    };
    if (line_net_rate_evaluate(&cooling, &result) != LINE_NET_OK_COOLING ||
        result.net_per_sr != 4.0 || result.cooling != 4.0 * LINE_NET_FOUR_PI ||
        result.heating != 0.0)
        return fail("finite cooling sign/split");

    LineNetComponentInput heating = cooling;
    heating.emission_per_sr = 2.0;
    heating.integrated_opacity = 3.0;
    heating.jbar = 1.0;
    if (line_net_rate_evaluate(&heating, &result) != LINE_NET_OK_HEATING ||
        result.net_per_sr != -1.0 || result.cooling != 0.0 ||
        result.heating != LINE_NET_FOUR_PI)
        return fail("finite heating sign/split");

    LineNetComponentInput exact_zero = cooling;
    exact_zero.emission_per_sr = 0.0;
    exact_zero.integrated_opacity = 0.0;
    exact_zero.jbar = 0.0;
    exact_zero.exact_zero_provenance = 1;
    if (line_net_rate_evaluate(&exact_zero, &result) != LINE_NET_EXACT_ZERO ||
        result.signed_rate != 0.0 || result.cooling != 0.0 ||
        result.heating != 0.0)
        return fail("typed exact zero");

    LineNetComponentInput cancelled = cooling;
    cancelled.emission_per_sr = 6.0;
    cancelled.integrated_opacity = 2.0;
    cancelled.jbar = 3.0;
    if (line_net_rate_evaluate(&cancelled, &result) !=
            LINE_NET_UNRESOLVED_CANCELLATION ||
        !isinf(result.cancellation_condition) || result.cooling != 0.0 ||
        result.heating != 0.0)
        return fail("unproven nonzero-component zero");

    LineNetComponentInput uncertain = cooling;
    uncertain.emission_per_sr = 6.25;
    uncertain.integrated_opacity = 2.0;
    uncertain.jbar = 3.0;
    uncertain.jbar_absolute_uncertainty = 0.2;
    if (line_net_rate_evaluate(&uncertain, &result) !=
            LINE_NET_UNRESOLVED_CANCELLATION ||
        !(result.signed_rate > 0.0) || !(result.absolute_uncertainty >
                                        result.signed_rate) ||
        result.cooling != 0.0 || result.heating != 0.0)
        return fail("uncertainty-covered sign was published");

    /* Classic product/subtraction witness.  Separate rounded multiplication
     * yields 1.0, but the exact double-input product is 1-2^-54.  FMA must
     * preserve the finite positive net rather than returning zero. */
    double d = ldexp(1.0, -27);
    LineNetComponentInput fma_witness = cooling;
    fma_witness.emission_per_sr = 1.0;
    fma_witness.integrated_opacity = 1.0 + d;
    fma_witness.jbar = 1.0 - d;
    if (line_net_rate_evaluate(&fma_witness, &result) !=
            LINE_NET_OK_COOLING ||
        result.net_per_sr != ldexp(1.0, -54) ||
        !(result.cancellation_condition > 1.0e16))
        return fail("FMA large-minus-large witness");

    LineNetComponentInput contradictory = exact_zero;
    contradictory.emission_per_sr = 1.0;
    before = result;
    if (line_net_rate_evaluate(&contradictory, &result) !=
            LINE_NET_INVALID_INPUT || result.status != LINE_NET_INVALID_INPUT)
        return fail("contradictory exact-zero provenance");
    (void)before;

    LineNetComponentInput invalid = cooling;
    invalid.jbar = NAN;
    memset(&result, 0xa5, sizeof(result));
    if (line_net_rate_evaluate(&invalid, &result) != LINE_NET_INVALID_INPUT ||
        result.status != LINE_NET_INVALID_INPUT)
        return fail("nonfinite input");

    printf("[LINE-NET][SELFTEST] status=PASS fixture_q_cgs=%.17g "
           "fma_net=%.17g finite_cooling=1 finite_heating=1 exact_zero=1 "
           "unresolved_cancellation=2 cmfgen_exponx_parity=1 "
           "sobolev_direct_bracket_identity=1 "
           "feiii_line2164811_shell0=1 negative_control=1 "
           "clamp=0 floor=0 jitter=0 repair=0 rc=0\n",
           fixture_q, ldexp(1.0, -54));
    return 0;
}
