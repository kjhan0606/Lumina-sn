#ifndef LUMINA_PHYSICS_COMPARISON_H
#define LUMINA_PHYSICS_COMPARISON_H

#include "lumina.h"

#include <stddef.h>

typedef enum {
    PHYSICS_COMPARISON_OK = 0,
    PHYSICS_COMPARISON_NOT_REQUESTED,
    PHYSICS_COMPARISON_INVALID_ARGUMENT,
    PHYSICS_COMPARISON_STALE_GENERATION,
    PHYSICS_COMPARISON_INVALID_GRID,
    PHYSICS_COMPARISON_INVALID_VALUE,
    PHYSICS_COMPARISON_IO_ERROR
} PhysicsComparisonStatus;

typedef struct {
    const char *lane;
    int iteration;
    double epoch_s;
    size_t n_shells;
    const double *r_inner_cm;
    const double *r_outer_cm;
    const double *v_inner_cm_s;
    const double *v_outer_cm_s;
    const double *temperature_K;
    const double *electron_density_cm3;
    const double *atom_density_cm3;
    const double *internal_energy_atom_erg;
    const RadiationFieldView *radiation;
    const CpuOpacityPublication *opacity;
    const CpuEmissivityPublication *emissivity;
    const ElectronTemperaturePublication *temperature_publication;
} PhysicsComparisonSnapshotInput;

/* Writes shell, spectral and manifest files.  The manifest is renamed last and
 * is the only commit marker.  No final file is published after validation
 * failure. */
PhysicsComparisonStatus physics_comparison_snapshot_write(
    const char *output_directory,
    const PhysicsComparisonSnapshotInput *input);

/* Production adapter.  An unset LUMINA_PHYSICS_COMPARISON_DIR is a no-op.
 * When requested, failure is returned to the caller and must be fail-closed. */
PhysicsComparisonStatus physics_comparison_dump_if_requested(
    const char *lane,
    int iteration,
    const Geometry *geometry,
    const AtomicData *atom,
    const PlasmaState *plasma,
    const OpacityState *opacity,
    const NLTEConfig *nlte);

const char *physics_comparison_status_name(PhysicsComparisonStatus status);

#endif
