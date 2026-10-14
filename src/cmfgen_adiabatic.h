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

/* 동일한 Lagrange 셸들의 실제 과거/현재 물질 상태. 같은시점 수렴반복의
 * 직전 후보를 과거 상태로 넣으면 안 된다. internal_energy는 중성 바닥
 * 기준의 들뜸+이온화 에너지이고 핵붕괴 에너지는 침적 항에서 별도 처리한다. */
typedef struct {
    size_t n_shells;
    double epoch_s;
    const double *density_g_cm3;
    const double *temperature_K;
    const double *n_atom_cm3;
    const double *n_electron_cm3;
    const double *internal_energy_atom_erg;
} CmfgenMaterialEnergyState;

typedef struct {
    double thermal_time;   /* 1.5 k n_atom_new delta[(1+ne/natom)T]/dt */
    double atomic_time;    /* n_atom_new delta(E_excitation+ionization per atom)/dt */
    double expansion;      /* P_new log(V_new/V_old)/dt; 동형팽창 V비=(t_new/t_old)^3 */
    double signed_total, cooling, heating; /* erg s^-1 cm^-3 */
} CmfgenMaterialTimeWork;

/* CMFGEN EVAL_TEMP_DDT_V2의 이산식(Hillier/Dessart2012 Eq27-30의 물질 쪽).
 * previous에는 필요한 원소 붕괴 처방이 이미 적용되어 있어야 한다.
 * Requires homologous
 * density evolution and identical mass-shell ordering. Output is transactional.
 * This replaces (does not add to) steady advective work in a time evolution. */
CmfgenAdiabaticStatus cmfgen_material_time_work(
    const CmfgenMaterialEnergyState *current,
    const CmfgenMaterialEnergyState *previous, CmfgenMaterialTimeWork *out);

#endif
