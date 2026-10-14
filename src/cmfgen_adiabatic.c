#include "cmfgen_adiabatic.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

CmfgenAdiabaticStatus cmfgen_material_time_work(
    const CmfgenMaterialEnergyState *current,
    const CmfgenMaterialEnergyState *previous, CmfgenMaterialTimeWork *out)
{
    if(!current || !previous || !out || current->n_shells==0 ||
       current->n_shells!=previous->n_shells ||
       current->n_shells>SIZE_MAX/sizeof(*out) ||
       !finite_positive(previous->epoch_s) || !isfinite(current->epoch_s) ||
       !(current->epoch_s>previous->epoch_s))return CMFGEN_ADIABATIC_INVALID_ARGUMENT;
    const CmfgenMaterialEnergyState *states[2]={current,previous};
    for(int k=0;k<2;++k){
        const CmfgenMaterialEnergyState *state=states[k];
        if(!state->density_g_cm3 || !state->temperature_K || !state->n_atom_cm3 ||
           !state->n_electron_cm3 || !state->internal_energy_atom_erg)
            return CMFGEN_ADIABATIC_INVALID_ARGUMENT;
        for(size_t s=0;s<state->n_shells;++s){
            if(!finite_positive(state->density_g_cm3[s]) ||
               !finite_positive(state->n_atom_cm3[s]) ||
               !(state->n_electron_cm3[s]>=0) || !isfinite(state->n_electron_cm3[s]))
                return CMFGEN_ADIABATIC_INVALID_DENSITY;
            if(!finite_positive(state->temperature_K[s]))
                return CMFGEN_ADIABATIC_INVALID_TEMPERATURE;
            if(!(state->internal_energy_atom_erg[s]>=0) ||
               !isfinite(state->internal_energy_atom_erg[s]))
                return CMFGEN_ADIABATIC_INVALID_INTERNAL_ENERGY;
        }
    }
    long double scale=(long double)previous->epoch_s/current->epoch_s;
    scale=scale*scale*scale;
    const long double dt=(long double)current->epoch_s-previous->epoch_s;
    CmfgenMaterialTimeWork *trial=calloc(current->n_shells,sizeof(*trial));
    if(!trial)return CMFGEN_ADIABATIC_ALLOCATION_FAILED;
    CmfgenAdiabaticStatus status=CMFGEN_ADIABATIC_OK;
    for(size_t s=0;s<current->n_shells;++s){
        long double q=(long double)current->density_g_cm3[s]/previous->density_g_cm3[s];
        if(fabsl(q/scale-1)>CMFGEN_AD_HOMOLOGY_RTOL){
            status=CMFGEN_ADIABATIC_NON_HOMOLOGOUS;break;
        }
        long double nnew=(long double)current->n_atom_cm3[s]+current->n_electron_cm3[s];
        long double nold=(long double)previous->n_atom_cm3[s]+previous->n_electron_cm3[s];
        /* EVAL_TEMP_DDT_V2: 과거는 붕괴 처방 적용 후 원자당 상태다.
         * 밀도비로 원자수를 추측하지 않고 현재 원자수로 차분을 환산한다. */
        long double atom_scale=(long double)current->n_atom_cm3[s]/previous->n_atom_cm3[s];
        CmfgenMaterialTimeWork *cell=&trial[s];
        cell->thermal_time=1.5L*CMFGEN_AD_K_BOLTZMANN*
            (nnew*current->temperature_K[s]-atom_scale*nold*previous->temperature_K[s])/dt;
        cell->atomic_time=(long double)current->n_atom_cm3[s]*
            (current->internal_energy_atom_erg[s]-previous->internal_energy_atom_erg[s])/dt;
        cell->expansion=3.0L*nnew*CMFGEN_AD_K_BOLTZMANN*
            current->temperature_K[s]*log1pl(dt/previous->epoch_s)/dt;
        cell->signed_total=four_term_sum(cell->thermal_time,cell->atomic_time,cell->expansion,0);
        if(!isfinite(cell->thermal_time) || !isfinite(cell->atomic_time) ||
           !isfinite(cell->expansion) || !isfinite(cell->signed_total)){
            status=CMFGEN_ADIABATIC_NONFINITE_RESULT;break;
        }
        cell->cooling=cell->signed_total>0?cell->signed_total:0;
        cell->heating=cell->signed_total<0?-cell->signed_total:0;
    }
    if(status==CMFGEN_ADIABATIC_OK)memcpy(out,trial,current->n_shells*sizeof(*out));
    free(trial);return status;
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
