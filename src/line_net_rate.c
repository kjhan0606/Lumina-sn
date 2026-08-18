#include "line_net_rate.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define LINE_NET_H_PLANCK 6.62607015e-27
#define LINE_NET_C_LIGHT 2.99792458e10
#define LINE_NET_M_ELECTRON 9.1093837015e-28
#define LINE_NET_E_SI_COULOMB 1.602176634e-19
#define LINE_NET_EV_TO_ERG 1.602176634e-12
#define LINE_NET_CMFGEN_INTERNAL_OPACITY_TO_CGS 1.0e-10

static LineNetStatus line_net_fail(LineNetResult *result,
                                   LineNetStatus status)
{
    if (result) result->status = status;
    return status;
}

LineNetStatus line_net_rate_evaluate(const LineNetComponentInput *input,
                                     LineNetResult *result)
{
    if (!result) return LINE_NET_INVALID_INPUT;
    memset(result, 0, sizeof(*result));
    result->status = LINE_NET_INVALID_INPUT;
    if (!input || !isfinite(input->emission_per_sr) ||
        input->emission_per_sr < 0.0 ||
        !isfinite(input->integrated_opacity) ||
        !isfinite(input->jbar) || input->jbar < 0.0 ||
        !isfinite(input->jbar_absolute_uncertainty) ||
        input->jbar_absolute_uncertainty < 0.0 ||
        !isfinite(input->other_net_absolute_uncertainty_per_sr) ||
        input->other_net_absolute_uncertainty_per_sr < 0.0 ||
        !isfinite(input->deck_scale) || input->deck_scale <= 0.0 ||
        (input->exact_zero_provenance != 0 &&
         input->exact_zero_provenance != 1))
        return LINE_NET_INVALID_INPUT;

    result->emission_per_sr = input->emission_per_sr;
    result->absorption_per_sr = input->integrated_opacity * input->jbar;
    result->net_per_sr = fma(-input->integrated_opacity, input->jbar,
                             input->emission_per_sr);

    double uncertainty_per_sr = fma(fabs(input->integrated_opacity),
                                    input->jbar_absolute_uncertainty,
                                    input->other_net_absolute_uncertainty_per_sr);
    double factor = LINE_NET_FOUR_PI * input->deck_scale;
    result->signed_rate = result->net_per_sr * factor;
    result->absolute_uncertainty = uncertainty_per_sr * factor;

    double component_magnitude = fabs(result->emission_per_sr) +
                                 fabs(result->absorption_per_sr);
    if (result->net_per_sr != 0.0)
        result->cancellation_condition = component_magnitude /
                                         fabs(result->net_per_sr);
    else
        result->cancellation_condition = component_magnitude == 0.0
                                       ? 0.0 : INFINITY;

    if (!isfinite(result->absorption_per_sr) ||
        !isfinite(result->net_per_sr) || !isfinite(uncertainty_per_sr) ||
        !isfinite(factor) || !isfinite(result->signed_rate) ||
        !isfinite(result->absolute_uncertainty) ||
        isnan(result->cancellation_condition))
        return line_net_fail(result, LINE_NET_INVALID_INPUT);

    if (input->exact_zero_provenance) {
        if (result->net_per_sr != 0.0)
            return line_net_fail(result, LINE_NET_INVALID_INPUT);
        return line_net_fail(result, LINE_NET_EXACT_ZERO);
    }

    /* A zero made from nonzero components is a cancellation, not an exact-zero
     * publication.  Likewise a nonzero sign covered by the supplied physical
     * input bound remains unresolved. */
    if (result->signed_rate == 0.0 ||
        fabs(result->signed_rate) <= result->absolute_uncertainty)
        return line_net_fail(result, LINE_NET_UNRESOLVED_CANCELLATION);

    if (result->signed_rate > 0.0) {
        result->cooling = result->signed_rate;
        return line_net_fail(result, LINE_NET_OK_COOLING);
    }
    result->heating = -result->signed_rate;
    return line_net_fail(result, LINE_NET_OK_HEATING);
}

int line_net_sobolev_material(double n_upper, double A_ul, double nu,
                              double tau, double time_explosion,
                              LineNetNegativeOpacityPolicy policy,
                              unsigned num_simultaneous_lines,
                              LineNetSobolevMaterial *material)
{
    if (!material) return -1;
    memset(material, 0, sizeof(*material));
    if (!isfinite(n_upper) || n_upper < 0.0 ||
        !isfinite(A_ul) || A_ul < 0.0 ||
        !isfinite(nu) || nu <= 0.0 || !isfinite(tau) ||
        !isfinite(time_explosion) || time_explosion <= 0.0 ||
        (policy != LINE_NET_NEGATIVE_OPACITY_NONE &&
         policy != LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK) ||
        num_simultaneous_lines == 0)
        return -1;

    material->emission_per_sr =
        n_upper * A_ul * LINE_NET_H_PLANCK * nu / LINE_NET_FOUR_PI;
    material->raw_integrated_opacity =
        tau * nu / (LINE_NET_C_LIGHT * time_explosion);
    material->effective_integrated_opacity =
        material->raw_integrated_opacity;
    material->effective_tau = tau;
    if (!isfinite(material->emission_per_sr) ||
        material->emission_per_sr < 0.0 ||
        !isfinite(material->raw_integrated_opacity)) {
        memset(material, 0, sizeof(*material));
        return -1;
    }

    if (policy == LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK &&
        tau < -0.5) {
        material->effective_integrated_opacity =
            LINE_NET_CMFGEN_INTERNAL_OPACITY_TO_CGS /
            (double)num_simultaneous_lines;
        material->effective_tau =
            material->effective_integrated_opacity * LINE_NET_C_LIGHT *
            time_explosion / nu;
        material->srce_chk_applied = 1;
    }
    material->exact_zero_provenance =
        n_upper == 0.0 && tau == 0.0;
    return isfinite(material->effective_integrated_opacity) &&
           isfinite(material->effective_tau) ? 0 : -1;
}

int line_net_cmfgen_exponx(double tau, double *beta,
                           double *one_minus_beta_over_tau)
{
    if (!beta || !one_minus_beta_over_tau) return -1;
    *beta = NAN;
    *one_minus_beta_over_tau = NAN;
    if (!isfinite(tau)) return -1;

    double value;
    double companion;
    if (fabs(tau) < 1.0e-3) {
        /* Exact transcription of CMFGEN subs/exponx.f.  Evaluate the
         * companion from the shared polynomial before the subtraction so
         * tau=0 and small signed tau do not lose their finite limit. */
        companion = 0.5 - tau / 6.0 * (1.0 - tau / 4.0);
        value = 1.0 - tau * companion;
    } else if (tau < 40.0) {
        value = (1.0 - exp(-tau)) / tau;
        companion = (1.0 - value) / tau;
    } else {
        value = 1.0 / tau;
        companion = (1.0 - value) / tau;
    }
    if (!isfinite(value) || value <= 0.0 || !isfinite(companion) ||
        companion <= 0.0)
        return -1;
    *beta = value;
    *one_minus_beta_over_tau = companion;
    return 0;
}

int line_net_sobolev_radiation(const LineNetSobolevMaterial *material,
                               double continuum_j,
                               double continuum_j_absolute_uncertainty,
                               double nu, double time_explosion,
                               LineNetSobolevRadiation *radiation)
{
    if (!radiation) return -1;
    memset(radiation, 0, sizeof(*radiation));
    if (!material || !isfinite(material->raw_integrated_opacity) ||
        !isfinite(material->effective_integrated_opacity) ||
        !isfinite(material->effective_tau) ||
        !isfinite(material->emission_per_sr) ||
        material->emission_per_sr < 0.0 ||
        (material->srce_chk_applied != 0 &&
         material->srce_chk_applied != 1) ||
        (material->exact_zero_provenance != 0 &&
         material->exact_zero_provenance != 1) ||
        !isfinite(continuum_j) || continuum_j < 0.0 ||
        !isfinite(continuum_j_absolute_uncertainty) ||
        continuum_j_absolute_uncertainty < 0.0 ||
        !isfinite(nu) || nu <= 0.0 ||
        !isfinite(time_explosion) || time_explosion <= 0.0)
        return -1;

    double beta, companion;
    if (line_net_cmfgen_exponx(material->effective_tau, &beta,
                               &companion) != 0)
        return -1;
    double continuum_term = beta * continuum_j;
    double source_scale = LINE_NET_C_LIGHT * time_explosion / nu;
    double local_emission_term =
        material->emission_per_sr * source_scale * companion;
    double jbar = continuum_term + local_emission_term;
    double uncertainty = beta * continuum_j_absolute_uncertainty;
    if (uncertainty > 0.0 && isfinite(uncertainty))
        uncertainty = nextafter(uncertainty, INFINITY);
    if (!isfinite(continuum_term) || continuum_term < 0.0 ||
        !isfinite(source_scale) || source_scale <= 0.0 ||
        !isfinite(local_emission_term) || local_emission_term < 0.0 ||
        !isfinite(jbar) || jbar < 0.0 ||
        !isfinite(uncertainty) || uncertainty < 0.0)
        return -1;

    radiation->beta = beta;
    radiation->one_minus_beta_over_tau = companion;
    radiation->continuum_term = continuum_term;
    radiation->local_emission_term = local_emission_term;
    radiation->jbar = jbar;
    radiation->jbar_absolute_uncertainty = uncertainty;
    return 0;
}

int line_net_einstein_opacity_ratio(double sobolev_coefficient,
                                    double f_lu, double wavelength_cm,
                                    double B_lu, double nu,
                                    double *ratio)
{
    if (!ratio) return -1;
    *ratio = NAN;
    if (!isfinite(sobolev_coefficient) || sobolev_coefficient <= 0.0 ||
        !isfinite(f_lu) || f_lu <= 0.0 ||
        !isfinite(wavelength_cm) || wavelength_cm <= 0.0 ||
        !isfinite(B_lu) || B_lu <= 0.0 ||
        !isfinite(nu) || nu <= 0.0)
        return -1;
    /* raw_integrated_opacity is tau*nu/(c*t_exp), while tau contains
     * wavelength_cm*t_exp.  Keep the serialized wavelength-frequency
     * product explicit rather than assuming wavelength_cm*nu/c == 1. */
    double sobolev_per_population = sobolev_coefficient * f_lu *
        wavelength_cm * nu / LINE_NET_C_LIGHT;
    double einstein_per_population =
        LINE_NET_H_PLANCK * nu * B_lu / LINE_NET_FOUR_PI;
    double value = einstein_per_population / sobolev_per_population;
    if (!isfinite(sobolev_per_population) ||
        !isfinite(einstein_per_population) ||
        !isfinite(value) || value <= 0.0)
        return -1;
    *ratio = value;
    return 0;
}

double line_net_exact_sobolev_coefficient(void)
{
    const double e_esu = LINE_NET_E_SI_COULOMB * LINE_NET_C_LIGHT / 10.0;
    return (LINE_NET_FOUR_PI / 4.0) * e_esu * e_esu /
           (LINE_NET_M_ELECTRON * LINE_NET_C_LIGHT);
}

int line_net_cmfgen_scl_ln(double lower_super_energy_eV,
                           double upper_super_energy_eV, double line_nu,
                           double atom_number_density,
                           double scl_ln_fac, double density_limit,
                           double *deck_scale)
{
    if (!deck_scale) return -1;
    *deck_scale = 0.0;
    if (!isfinite(lower_super_energy_eV) ||
        !isfinite(upper_super_energy_eV) ||
        !isfinite(line_nu) || line_nu <= 0.0 ||
        !isfinite(atom_number_density) || atom_number_density < 0.0 ||
        !isfinite(scl_ln_fac) || scl_ln_fac < 0.0 ||
        !isfinite(density_limit) || density_limit <= 0.0)
        return -1;

    double ratio = (upper_super_energy_eV - lower_super_energy_eV) *
                   LINE_NET_EV_TO_ERG /
                   (LINE_NET_H_PLANCK * line_nu);
    if (!isfinite(ratio)) return -1;
    double scale = ratio;
    if (atom_number_density >= density_limit ||
        fabs(ratio - 1.0) > scl_ln_fac)
        scale = 1.0;
    if (!isfinite(scale) || scale <= 0.0) return -1;
    *deck_scale = scale;
    return 0;
}

int line_net_cmfgen_internal_to_cgs(double etal_mat, double znet,
                                    double deck_scale, double *q_cgs)
{
    if (!q_cgs || !isfinite(etal_mat) || etal_mat < 0.0 ||
        !isfinite(znet) || !isfinite(deck_scale) || deck_scale <= 0.0)
        return -1;
    double q = (etal_mat * znet) * deck_scale *
               LINE_NET_CMFGEN_RE_INTERNAL_TO_CGS;
    if (!isfinite(q)) return -1;
    *q_cgs = q;
    return 0;
}

const char *line_net_status_name(LineNetStatus status)
{
    switch (status) {
    case LINE_NET_OK_COOLING: return "OK_COOLING";
    case LINE_NET_OK_HEATING: return "OK_HEATING";
    case LINE_NET_EXACT_ZERO: return "EXACT_ZERO";
    case LINE_NET_UNRESOLVED_CANCELLATION:
        return "UNRESOLVED_CANCELLATION";
    case LINE_NET_INVALID_INPUT: return "INVALID_INPUT";
    default: return "UNKNOWN";
    }
}
