#ifndef LUMINA_CMFGEN_ADIABATIC_H
#define LUMINA_CMFGEN_ADIABATIC_H

#include <stddef.h>

/* Steady, homologous counterpart of CMFGEN EVAL_ADIABATIC_V3.  This is a
 * signed vector producer only: it does not authorize A2-10 publication. */
typedef enum {
    CMFGEN_ADIABATIC_OK = 0,
    CMFGEN_ADIABATIC_INVALID_ARGUMENT,
    CMFGEN_ADIABATIC_NEED_TWO_SHELLS,
    CMFGEN_ADIABATIC_NONFINITE_INPUT,
    CMFGEN_ADIABATIC_INVALID_RADIUS_GRID,
    CMFGEN_ADIABATIC_INVALID_VELOCITY,
    CMFGEN_ADIABATIC_INVALID_TEMPERATURE,
    CMFGEN_ADIABATIC_INVALID_DENSITY,
    CMFGEN_ADIABATIC_INVALID_INTERNAL_ENERGY,
    CMFGEN_ADIABATIC_NON_HOMOLOGOUS,
    CMFGEN_ADIABATIC_ALLOCATION_FAILED,
    CMFGEN_ADIABATIC_NONFINITE_RESULT
} CmfgenAdiabaticStatus;

typedef struct {
    size_t n_shells;
    double epoch_s;
    const double *radius_cm;                 /* increasing inner -> outer */
    const double *velocity_cm_s;             /* shell-center homologous v */
    const double *temperature_K;
    const double *n_atom_cm3;                /* total nuclei density */
    const double *n_electron_cm3;
    const double *internal_energy_atom_erg;  /* neutral-ground reference */
} CmfgenAdiabaticInput;

typedef struct {
    double temperature_gradient;       /* signed erg s^-1 cm^-3 */
    double velocity_divergence;        /* signed erg s^-1 cm^-3 */
    double electron_fraction_gradient; /* signed erg s^-1 cm^-3 */
    double internal_energy_gradient;   /* signed erg s^-1 cm^-3 */
    double signed_total;               /* positive=cooling, negative=heating */
    double cooling;                    /* max(signed_total, 0) */
    double heating;                    /* max(-signed_total, 0) */
} CmfgenAdiabaticCell;

/* On every non-OK return, out[0..n_shells) is byte-preserved.  The geometry
 * must satisfy r=v*epoch to 1e-10 relative error; a time-dependent/non-
 * homologous model requires another producer rather than a silent fallback. */
CmfgenAdiabaticStatus cmfgen_adiabatic_v3_homologous_evaluate(
    const CmfgenAdiabaticInput *input, CmfgenAdiabaticCell *out);

const char *cmfgen_adiabatic_status_name(CmfgenAdiabaticStatus status);

#endif
