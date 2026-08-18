#ifndef LUMINA_ATOMIC_INTERNAL_ENERGY_H
#define LUMINA_ATOMIC_INTERNAL_ENERGY_H

#include "lumina.h"

#include <stddef.h>
#include <stdint.h>

typedef enum {
    ATOMIC_INTERNAL_ENERGY_OK = 0,
    ATOMIC_INTERNAL_ENERGY_INVALID_ARGUMENT,
    ATOMIC_INTERNAL_ENERGY_STALE_GENERATION,
    ATOMIC_INTERNAL_ENERGY_INVALID_TEMPERATURE,
    ATOMIC_INTERNAL_ENERGY_INVALID_POPULATION,
    ATOMIC_INTERNAL_ENERGY_INVALID_PARTITION,
    ATOMIC_INTERNAL_ENERGY_MISSING_LEVEL,
    ATOMIC_INTERNAL_ENERGY_MISSING_IONIZATION,
    ATOMIC_INTERNAL_ENERGY_POPULATION_CLOSURE,
    ATOMIC_INTERNAL_ENERGY_ALLOCATION_FAILED,
    ATOMIC_INTERNAL_ENERGY_NONFINITE
} AtomicInternalEnergyStatus;

typedef struct {
    double n_atom_cm3;
    double energy_density_erg_cm3;
    double internal_energy_atom_erg;
} AtomicInternalEnergyCell;

/* CMFGEN neutral-ground-reference excitation+ionization internal energy.
 * Mapped levels consume the generation-bound NLTE population; untracked
 * catalog levels use LTE@T_e from the same partition stamp.  Each represented
 * untracked ion must close back to its ion-population owner; tracked stages
 * may exchange population under single-total SE but must close per element.
 * On failure output is byte-preserved. */
AtomicInternalEnergyStatus atomic_internal_energy_build(
    const AtomicData *atom,
    const NLTEConfig *nlte,
    const PlasmaState *plasma,
    size_t n_shells,
    uint64_t required_population_generation,
    uint64_t required_te_generation,
    AtomicInternalEnergyCell *output);

const char *atomic_internal_energy_status_name(
    AtomicInternalEnergyStatus status);

#endif
