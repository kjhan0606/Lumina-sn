#include "atomic_internal_energy.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ATOMIC_ENERGY_EV_TO_ERG 1.602176634e-12
#define ATOMIC_ENERGY_CM1_TO_ERG 1.9864458571489286e-16
#define ATOMIC_ENERGY_K_BOLTZMANN 1.380649e-16
#define ATOMIC_ENERGY_CLOSURE_RTOL 1.0e-8

static PopulationAtomicView atomic_energy_view(const AtomicData *atom)
{
    PopulationAtomicView view;
    memset(&view, 0, sizeof(view));
    if (!atom) return view;
    view.n_ions = (size_t)atom->n_ion_pops;
    view.n_levels = (size_t)atom->n_levels;
    view.level_offset = atom->level_offset;
    view.energy_eV = atom->level_energy_eV;
    view.g = atom->level_g;
    view.level_Z = atom->level_Z;
    view.level_ion = atom->level_ion;
    view.topion_n = atom->topion_n;
    view.topion_ion_index = atom->topion_ion_index;
    view.topion_E_cm = atom->topion_E_cm;
    view.topion_g = atom->topion_g;
    return view;
}

static int kahan_add(double value, double *sum, double *correction)
{
    double adjusted = value - *correction;
    double next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
    return isfinite(next) ? 0 : -1;
}

static AtomicInternalEnergyStatus ionization_base_eV(
        const AtomicData *atom, int Z, int stage, double *base)
{
    if (!atom || !base || Z <= 0 || stage < 0 || stage > Z)
        return ATOMIC_INTERNAL_ENERGY_MISSING_IONIZATION;
    if (atom->n_ionization < 0 || atom->n_ionization_reference < 0 ||
        (atom->n_ionization > 0 &&
         (!atom->ioniz_Z || !atom->ioniz_ion ||
          !atom->ioniz_energy_eV)) ||
        (atom->n_ionization_reference > 0 &&
         (!atom->ioniz_ref_Z || !atom->ioniz_ref_ion ||
          !atom->ioniz_ref_energy_eV)) ||
        (stage > 0 && atom->n_ionization == 0 &&
         atom->n_ionization_reference == 0))
        return ATOMIC_INTERNAL_ENERGY_MISSING_IONIZATION;
    double sum = 0.0, correction = 0.0;
    for (int q = 0; q < stage; ++q) {
        int matches = 0;
        double chi = NAN;
        for (int k = 0; k < atom->n_ionization; ++k) {
            if (atom->ioniz_Z[k] == Z && atom->ioniz_ion[k] == q) {
                chi = atom->ioniz_energy_eV[k];
                matches++;
            }
        }
        /* The model deck owns rate/population physics.  Consult the separate
         * energy-zero reference only when that deck has no row for this link;
         * a reference row can never override a regular row. */
        if (matches == 0) {
            for (int k = 0; k < atom->n_ionization_reference; ++k) {
                if (atom->ioniz_ref_Z[k] == Z &&
                    atom->ioniz_ref_ion[k] == q) {
                    chi = atom->ioniz_ref_energy_eV[k];
                    matches++;
                }
            }
        }
        if (matches != 1 || !isfinite(chi) || chi <= 0.0)
            return ATOMIC_INTERNAL_ENERGY_MISSING_IONIZATION;
        if (kahan_add(chi, &sum, &correction) != 0)
            return ATOMIC_INTERNAL_ENERGY_NONFINITE;
    }
    sum += correction;
    if (!isfinite(sum) || sum < 0.0)
        return ATOMIC_INTERNAL_ENERGY_NONFINITE;
    *base = sum;
    return ATOMIC_INTERNAL_ENERGY_OK;
}

static int population_closes(double sum, double expected)
{
    double scale = fmax(fabs(expected), DBL_MIN);
    return isfinite(sum) && fabs(sum - expected) <=
           ATOMIC_ENERGY_CLOSURE_RTOL * scale;
}

static AtomicInternalEnergyStatus catalog_ion_energy(
        const AtomicData *atom, const NLTEConfig *nlte,
        const PlasmaState *plasma, const PopulationAtomicView *view,
        size_t ion, size_t shell, size_t n_shells,
        double ion_population, double partition, double ionization_eV,
        double *energy_density, double *represented_population)
{
    if (!energy_density || !represented_population)
        return ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;
    int lo = atom->level_offset[ion];
    int hi = atom->level_offset[ion + 1];
    if (lo < 0 || hi <= lo || (size_t)hi > (size_t)atom->n_levels)
        return ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;
    double ground_eV = INFINITY;
    for (int level = lo; level < hi; ++level) {
        if (atom->level_g[level] > 0 &&
            isfinite(atom->level_energy_eV[level]) &&
            atom->level_energy_eV[level] < ground_eV)
            ground_eV = atom->level_energy_eV[level];
    }
    if (!isfinite(ground_eV))
        return ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;

    double population_sum = 0.0, population_correction = 0.0;
    double energy_sum = 0.0, energy_correction = 0.0;
    int mapped_count = 0;
    for (int level = lo; level < hi; ++level) {
        if (atom->level_g[level] <= 0 ||
            !isfinite(atom->level_energy_eV[level]))
            return ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;
        double population = NAN;
        int nlte_index = nlte->global_to_nlte_level
                       ? nlte->global_to_nlte_level[level] : -1;
        if (nlte_index >= 0) {
            if (nlte_index >= nlte->n_nlte_levels_total ||
                !nlte->nlte_level_populations)
                return ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;
            population = nlte->nlte_level_populations[
                (size_t)nlte_index * n_shells + shell];
            mapped_count++;
        } else {
            double fraction = NAN;
            PopulationStatus status = population_lte_level_fraction(
                view, ion, (size_t)level, plasma->T_e[shell], partition,
                &fraction);
            if (status != POP_OK && status != POP_EXACT_ZERO)
                return status == POP_INVALID_PARTITION
                    ? ATOMIC_INTERNAL_ENERGY_INVALID_PARTITION
                    : ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;
            population = ion_population * fraction;
        }
        if (!isfinite(population) || population < 0.0)
            return ATOMIC_INTERNAL_ENERGY_INVALID_POPULATION;
        double excitation_eV = atom->level_energy_eV[level] - ground_eV;
        if (!isfinite(excitation_eV) || excitation_eV < 0.0)
            return ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;
        if (kahan_add(population, &population_sum,
                      &population_correction) != 0 ||
            kahan_add(population * (ionization_eV + excitation_eV),
                      &energy_sum, &energy_correction) != 0)
            return ATOMIC_INTERNAL_ENERGY_NONFINITE;
    }
    population_sum += population_correction;
    energy_sum += energy_correction;
    /* Projection construction maps either every full level of a target ion or
     * none of them.  A mixed view would combine two incompatible owners. */
    if (mapped_count != 0 && mapped_count != hi - lo)
        return ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;
    /* For an untracked ion, the LTE reconstruction must close to the ion
     * population owner.  For a fully tracked ion, the level SE solution is
     * the stage-population owner; single-total SE intentionally lets that
     * stage total differ from the upstream ionization estimate.  The caller
     * checks the stronger per-element nuclei closure after all stages. */
    if (mapped_count == 0 &&
        !population_closes(population_sum, ion_population))
        return ATOMIC_INTERNAL_ENERGY_POPULATION_CLOSURE;
    *represented_population = population_sum;
    *energy_density = energy_sum * ATOMIC_ENERGY_EV_TO_ERG;
    return isfinite(*energy_density) && *energy_density >= 0.0
        ? ATOMIC_INTERNAL_ENERGY_OK : ATOMIC_INTERNAL_ENERGY_NONFINITE;
}

static AtomicInternalEnergyStatus topion_energy(
        const AtomicData *atom, const PlasmaState *plasma,
        size_t ion, size_t shell, double ion_population, double partition,
        double ionization_eV, double *energy_density)
{
    if (!atom->topion_ion_index || !atom->topion_E_cm ||
        !atom->topion_g)
        return ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;
    double ground_cm = INFINITY;
    size_t count = 0;
    for (size_t k = 0; k < atom->topion_n; ++k) {
        if ((size_t)atom->topion_ion_index[k] != ion) continue;
        if (!(atom->topion_g[k] > 0.0) ||
            !isfinite(atom->topion_E_cm[k]))
            return ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;
        if (atom->topion_E_cm[k] < ground_cm)
            ground_cm = atom->topion_E_cm[k];
        count++;
    }
    if (count == 0 || !isfinite(ground_cm))
        return ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;
    double weight_sum = 0.0, weight_correction = 0.0;
    double energy_sum = 0.0, energy_correction = 0.0;
    for (size_t k = 0; k < atom->topion_n; ++k) {
        if ((size_t)atom->topion_ion_index[k] != ion) continue;
        double excitation_cm = atom->topion_E_cm[k] - ground_cm;
        double excitation_erg = excitation_cm * ATOMIC_ENERGY_CM1_TO_ERG;
        double x = excitation_erg /
                   (ATOMIC_ENERGY_K_BOLTZMANN * plasma->T_e[shell]);
        double weight = x < 745.0 ? atom->topion_g[k] * exp(-x) : 0.0;
        double population = ion_population * weight / partition;
        double total_erg = ionization_eV * ATOMIC_ENERGY_EV_TO_ERG +
                           excitation_erg;
        if (!isfinite(weight) || weight < 0.0 ||
            !isfinite(population) || population < 0.0 ||
            kahan_add(population, &weight_sum, &weight_correction) != 0 ||
            kahan_add(population * total_erg,
                      &energy_sum, &energy_correction) != 0)
            return ATOMIC_INTERNAL_ENERGY_NONFINITE;
    }
    weight_sum += weight_correction;
    energy_sum += energy_correction;
    if (!population_closes(weight_sum, ion_population))
        return ATOMIC_INTERNAL_ENERGY_POPULATION_CLOSURE;
    *energy_density = energy_sum;
    return isfinite(*energy_density) && *energy_density >= 0.0
        ? ATOMIC_INTERNAL_ENERGY_OK : ATOMIC_INTERNAL_ENERGY_NONFINITE;
}

AtomicInternalEnergyStatus atomic_internal_energy_build(
        const AtomicData *atom, const NLTEConfig *nlte,
        const PlasmaState *plasma, size_t n_shells,
        uint64_t required_population_generation,
        uint64_t required_te_generation,
        AtomicInternalEnergyCell *output)
{
    if (!atom || !nlte || !plasma || !output || n_shells == 0 ||
        atom->n_ion_pops <= 0 || atom->n_levels < 0 ||
        !atom->ion_pop_Z || !atom->ion_pop_stage || !atom->level_offset ||
        !atom->ion_number_density || !atom->partition_functions ||
        !plasma->T_e ||
        (atom->n_levels > 0 &&
         (!atom->level_energy_eV || !atom->level_g)))
        return ATOMIC_INTERNAL_ENERGY_INVALID_ARGUMENT;
    if (required_population_generation == 0 || required_te_generation == 0 ||
        atom->population_committed_generation !=
            required_population_generation ||
        nlte->population_committed_generation !=
            required_population_generation ||
        plasma->T_e_generation != required_te_generation)
        return ATOMIC_INTERNAL_ENERGY_STALE_GENERATION;
    PopulationAtomicView view = atomic_energy_view(atom);
    PopulationStatus stamp_status = population_partition_view_check(
        &atom->partition_stamp, &view, plasma->T_e, n_shells,
        required_population_generation, required_te_generation);
    if (stamp_status != POP_OK)
        return ATOMIC_INTERNAL_ENERGY_STALE_GENERATION;
    for (size_t s = 0; s < n_shells; ++s)
        if (!isfinite(plasma->T_e[s]) || plasma->T_e[s] <= 0.0)
            return ATOMIC_INTERNAL_ENERGY_INVALID_TEMPERATURE;

    AtomicInternalEnergyCell *candidate = calloc(
        n_shells, sizeof(*candidate));
    double *represented_population = calloc(
        (size_t)atom->n_ion_pops, sizeof(*represented_population));
    if (!candidate || !represented_population) {
        free(candidate);
        free(represented_population);
        return ATOMIC_INTERNAL_ENERGY_ALLOCATION_FAILED;
    }
    AtomicInternalEnergyStatus result = ATOMIC_INTERNAL_ENERGY_OK;
    for (size_t s = 0; s < n_shells && result == ATOMIC_INTERNAL_ENERGY_OK;
         ++s) {
        memset(represented_population, 0,
               (size_t)atom->n_ion_pops * sizeof(*represented_population));
        double atom_sum = 0.0, atom_correction = 0.0;
        double energy_sum = 0.0, energy_correction = 0.0;
        for (size_t ion = 0; ion < (size_t)atom->n_ion_pops; ++ion) {
            double population = atom->ion_number_density[ion*n_shells+s];
            double partition = atom->partition_functions[ion*n_shells+s];
            if (!isfinite(population) || population < 0.0) {
                result = ATOMIC_INTERNAL_ENERGY_INVALID_POPULATION;
                break;
            }
            if (!isfinite(partition) || partition <= 0.0) {
                result = ATOMIC_INTERNAL_ENERGY_INVALID_PARTITION;
                break;
            }
            int lo = atom->level_offset[ion];
            int hi = atom->level_offset[ion + 1];
            if (lo < 0 || hi < lo || (size_t)hi > (size_t)atom->n_levels) {
                result = ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;
                break;
            }
            /* An upstream stage estimate may be exactly zero while the
             * single-total level SE assigns that fully mapped stage a finite
             * population.  Only an untracked zero stage may be skipped. */
            int mapped_count = 0;
            for (int level = lo; level < hi; ++level) {
                int nlte_index = nlte->global_to_nlte_level
                               ? nlte->global_to_nlte_level[level] : -1;
                if (nlte_index >= 0) mapped_count++;
            }
            if (mapped_count != 0 && mapped_count != hi - lo) {
                result = ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL;
                break;
            }
            if (population == 0.0 && mapped_count == 0) continue;
            double base_eV = NAN;
            result = ionization_base_eV(
                atom, atom->ion_pop_Z[ion], atom->ion_pop_stage[ion],
                &base_eV);
            if (result != ATOMIC_INTERNAL_ENERGY_OK) break;
            double ion_energy = NAN;
            if (lo < hi)
                result = catalog_ion_energy(
                    atom, nlte, plasma, &view, ion, s, n_shells,
                    population, partition, base_eV, &ion_energy,
                    &represented_population[ion]);
            else {
                result = topion_energy(
                    atom, plasma, ion, s, population, partition,
                    base_eV, &ion_energy);
                represented_population[ion] = population;
            }
            if (result != ATOMIC_INTERNAL_ENERGY_OK) break;
            if (kahan_add(ion_energy, &energy_sum, &energy_correction) != 0) {
                result = ATOMIC_INTERNAL_ENERGY_NONFINITE;
                break;
            }
        }
        /* A single-total level solve may redistribute nuclei between adjacent
         * ion stages, so per-ion equality is deliberately not required.  It
         * must still conserve every element total.  Group by Z directly rather
         * than relying on a particular ion-ladder layout. */
        for (size_t ion = 0;
             ion < (size_t)atom->n_ion_pops &&
             result == ATOMIC_INTERNAL_ENERGY_OK; ++ion) {
            int Z = atom->ion_pop_Z[ion];
            int seen = 0;
            for (size_t prior = 0; prior < ion; ++prior)
                if (atom->ion_pop_Z[prior] == Z) { seen = 1; break; }
            if (seen) continue;
            double expected = 0.0, expected_correction = 0.0;
            double represented = 0.0, represented_correction = 0.0;
            for (size_t other = ion;
                 other < (size_t)atom->n_ion_pops; ++other) {
                if (atom->ion_pop_Z[other] != Z) continue;
                if (kahan_add(
                        atom->ion_number_density[other*n_shells+s],
                        &expected, &expected_correction) != 0 ||
                    kahan_add(represented_population[other],
                              &represented,
                              &represented_correction) != 0) {
                    result = ATOMIC_INTERNAL_ENERGY_NONFINITE;
                    break;
                }
            }
            expected += expected_correction;
            represented += represented_correction;
            if (result != ATOMIC_INTERNAL_ENERGY_OK) break;
            if (!population_closes(represented, expected)) {
                fprintf(stderr,
                        "[ATOMIC-INTERNAL-ENERGY][POPULATION-CLOSURE] "
                        "shell=%zu Z=%d represented=%.17g expected=%.17g "
                        "difference=%.17g\n", s, Z, represented, expected,
                        represented - expected);
                result = ATOMIC_INTERNAL_ENERGY_POPULATION_CLOSURE;
                break;
            }
        }
        for (size_t ion = 0;
             ion < (size_t)atom->n_ion_pops &&
             result == ATOMIC_INTERNAL_ENERGY_OK; ++ion) {
            if (kahan_add(represented_population[ion],
                          &atom_sum, &atom_correction) != 0) {
                result = ATOMIC_INTERNAL_ENERGY_NONFINITE;
                break;
            }
        }
        atom_sum += atom_correction;
        energy_sum += energy_correction;
        if (result != ATOMIC_INTERNAL_ENERGY_OK) break;
        if (!isfinite(atom_sum) || atom_sum <= 0.0 ||
            !isfinite(energy_sum) || energy_sum < 0.0) {
            result = ATOMIC_INTERNAL_ENERGY_NONFINITE;
            break;
        }
        candidate[s].n_atom_cm3 = atom_sum;
        candidate[s].energy_density_erg_cm3 = energy_sum;
        candidate[s].internal_energy_atom_erg = energy_sum / atom_sum;
        if (!isfinite(candidate[s].internal_energy_atom_erg) ||
            candidate[s].internal_energy_atom_erg < 0.0) {
            result = ATOMIC_INTERNAL_ENERGY_NONFINITE;
            break;
        }
    }
    if (result == ATOMIC_INTERNAL_ENERGY_OK)
        memcpy(output, candidate, n_shells * sizeof(*output));
    free(represented_population);
    free(candidate);
    return result;
}

const char *atomic_internal_energy_status_name(
        AtomicInternalEnergyStatus status)
{
    static const char *const names[] = {
        "ATOMIC_INTERNAL_ENERGY_OK",
        "ATOMIC_INTERNAL_ENERGY_INVALID_ARGUMENT",
        "ATOMIC_INTERNAL_ENERGY_STALE_GENERATION",
        "ATOMIC_INTERNAL_ENERGY_INVALID_TEMPERATURE",
        "ATOMIC_INTERNAL_ENERGY_INVALID_POPULATION",
        "ATOMIC_INTERNAL_ENERGY_INVALID_PARTITION",
        "ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL",
        "ATOMIC_INTERNAL_ENERGY_MISSING_IONIZATION",
        "ATOMIC_INTERNAL_ENERGY_POPULATION_CLOSURE",
        "ATOMIC_INTERNAL_ENERGY_ALLOCATION_FAILED",
        "ATOMIC_INTERNAL_ENERGY_NONFINITE"
    };
    if (status < ATOMIC_INTERNAL_ENERGY_OK ||
        status > ATOMIC_INTERNAL_ENERGY_NONFINITE)
        return "ATOMIC_INTERNAL_ENERGY_UNKNOWN";
    return names[status];
}
