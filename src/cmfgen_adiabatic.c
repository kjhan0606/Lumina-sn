#include "cmfgen_adiabatic.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define CMFGEN_AD_K_BOLTZMANN 1.380649e-16
#define CMFGEN_AD_HOMOLOGY_RTOL 1.0e-10

static int finite_positive(double value)
{
    return isfinite(value) && value > 0.0;
}

static double four_term_sum(double a, double b, double c, double d)
{
    const double terms[4] = {a, b, c, d};
    double sum = 0.0;
    double correction = 0.0;
    for (size_t i = 0; i < 4; ++i) {
        double adjusted = terms[i] - correction;
        double next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    return sum;
}

CmfgenAdiabaticStatus cmfgen_adiabatic_v3_homologous_evaluate(
    const CmfgenAdiabaticInput *input, CmfgenAdiabaticCell *out)
{
    if (!input || !out || !input->radius_cm || !input->velocity_cm_s ||
        !input->temperature_K || !input->n_atom_cm3 ||
        !input->n_electron_cm3 || !input->internal_energy_atom_erg)
        return CMFGEN_ADIABATIC_INVALID_ARGUMENT;
    if (input->n_shells < 2)
        return CMFGEN_ADIABATIC_NEED_TWO_SHELLS;
    if (!finite_positive(input->epoch_s))
        return isfinite(input->epoch_s)
            ? CMFGEN_ADIABATIC_INVALID_ARGUMENT
            : CMFGEN_ADIABATIC_NONFINITE_INPUT;

    for (size_t s = 0; s < input->n_shells; ++s) {
        double r = input->radius_cm[s];
        double v = input->velocity_cm_s[s];
        if (!isfinite(r) || !isfinite(v) ||
            !isfinite(input->temperature_K[s]) ||
            !isfinite(input->n_atom_cm3[s]) ||
            !isfinite(input->n_electron_cm3[s]) ||
            !isfinite(input->internal_energy_atom_erg[s]))
            return CMFGEN_ADIABATIC_NONFINITE_INPUT;
        if (r <= 0.0 || (s > 0 && r <= input->radius_cm[s - 1]))
            return CMFGEN_ADIABATIC_INVALID_RADIUS_GRID;
        if (v <= 0.0)
            return CMFGEN_ADIABATIC_INVALID_VELOCITY;
        if (input->temperature_K[s] <= 0.0)
            return CMFGEN_ADIABATIC_INVALID_TEMPERATURE;
        if (input->n_atom_cm3[s] <= 0.0 ||
            input->n_electron_cm3[s] < 0.0)
            return CMFGEN_ADIABATIC_INVALID_DENSITY;
        if (input->internal_energy_atom_erg[s] < 0.0)
            return CMFGEN_ADIABATIC_INVALID_INTERNAL_ENERGY;

        double homologous_radius = v * input->epoch_s;
        if (!isfinite(homologous_radius))
            return CMFGEN_ADIABATIC_NONFINITE_INPUT;
        double scale = fmax(fabs(r), DBL_MIN);
        if (fabs(homologous_radius - r) / scale >
            CMFGEN_AD_HOMOLOGY_RTOL)
            return CMFGEN_ADIABATIC_NON_HOMOLOGOUS;
    }

    CmfgenAdiabaticCell *candidate = calloc(
        input->n_shells, sizeof(*candidate));
    if (!candidate)
        return CMFGEN_ADIABATIC_ALLOCATION_FAILED;

    for (size_t s = 0; s < input->n_shells; ++s) {
        size_t neighbor = s == 0 ? 1 : s - 1;
        double dr = input->radius_cm[s] - input->radius_cm[neighbor];
        double v = input->velocity_cm_s[s];
        double temperature = input->temperature_K[s];
        double n_atom = input->n_atom_cm3[s];
        double n_electron = input->n_electron_cm3[s];
        double gamma = n_electron / n_atom;
        double gamma_neighbor = input->n_electron_cm3[neighbor] /
                                input->n_atom_cm3[neighbor];

        double dT_dr = (temperature - input->temperature_K[neighbor]) / dr;
        double dgamma_dr = (gamma - gamma_neighbor) / dr;
        double du_dr = (input->internal_energy_atom_erg[s] -
                        input->internal_energy_atom_erg[neighbor]) / dr;

        CmfgenAdiabaticCell *cell = &candidate[s];
        cell->temperature_gradient =
            1.5 * (n_atom + n_electron) * CMFGEN_AD_K_BOLTZMANN *
            v * dT_dr;
        cell->velocity_divergence =
            3.0 * (n_atom + n_electron) * CMFGEN_AD_K_BOLTZMANN *
            temperature * v / input->radius_cm[s];
        cell->electron_fraction_gradient =
            1.5 * n_atom * CMFGEN_AD_K_BOLTZMANN * temperature *
            v * dgamma_dr;
        cell->internal_energy_gradient = n_atom * v * du_dr;
        cell->signed_total = four_term_sum(
            cell->temperature_gradient, cell->velocity_divergence,
            cell->electron_fraction_gradient,
            cell->internal_energy_gradient);

        if (!isfinite(gamma) || !isfinite(gamma_neighbor) ||
            !isfinite(dT_dr) || !isfinite(dgamma_dr) || !isfinite(du_dr) ||
            !isfinite(cell->temperature_gradient) ||
            !isfinite(cell->velocity_divergence) ||
            !isfinite(cell->electron_fraction_gradient) ||
            !isfinite(cell->internal_energy_gradient) ||
            !isfinite(cell->signed_total)) {
            free(candidate);
            return CMFGEN_ADIABATIC_NONFINITE_RESULT;
        }
        if (cell->signed_total == 0.0)
            cell->signed_total = 0.0;
        cell->cooling = cell->signed_total > 0.0
            ? cell->signed_total : 0.0;
        cell->heating = cell->signed_total < 0.0
            ? -cell->signed_total : 0.0;
    }

    memcpy(out, candidate, input->n_shells * sizeof(*out));
    free(candidate);
    return CMFGEN_ADIABATIC_OK;
}

const char *cmfgen_adiabatic_status_name(CmfgenAdiabaticStatus status)
{
    static const char *const names[] = {
        "CMFGEN_ADIABATIC_OK",
        "CMFGEN_ADIABATIC_INVALID_ARGUMENT",
        "CMFGEN_ADIABATIC_NEED_TWO_SHELLS",
        "CMFGEN_ADIABATIC_NONFINITE_INPUT",
        "CMFGEN_ADIABATIC_INVALID_RADIUS_GRID",
        "CMFGEN_ADIABATIC_INVALID_VELOCITY",
        "CMFGEN_ADIABATIC_INVALID_TEMPERATURE",
        "CMFGEN_ADIABATIC_INVALID_DENSITY",
        "CMFGEN_ADIABATIC_INVALID_INTERNAL_ENERGY",
        "CMFGEN_ADIABATIC_NON_HOMOLOGOUS",
        "CMFGEN_ADIABATIC_ALLOCATION_FAILED",
        "CMFGEN_ADIABATIC_NONFINITE_RESULT"
    };
    if (status < CMFGEN_ADIABATIC_OK ||
        status > CMFGEN_ADIABATIC_NONFINITE_RESULT)
        return "CMFGEN_ADIABATIC_UNKNOWN";
    return names[status];
}
