/**
 * LUMINA-SN Simulation State Implementation
 * simulation_state.c - Integrated plasma-transport state management
 *
 * Implements Sobolev line opacity calculation and efficient line lookup
 * for Monte Carlo radiative transfer with realistic atomic physics.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "simulation_state.h"

/* ============================================================================
 * PHYSICAL CONSTANTS
 * ============================================================================ */

/* Sobolev constant: π e² / (m_e c) in CGS */
/* = π × (4.803e-10)² / (9.109e-28 × 2.998e10) = 2.654e-2 cm² s⁻¹ */
#define SOBOLEV_CONST  2.6540281e-2

/* Thomson cross-section [cm²] */
#define SIGMA_THOMSON  6.6524587158e-25

/* Task Order #30: Global Fe-group blue opacity scale (legacy) */
double g_fe_blue_scale = 0.5;  /* Default: 0.5× for blue Fe-group lines */

/* Task Order #30 v2: Physics Overrides - Global Instance
 * NOW WITH CONTINUUM OPACITY - Use physical T_boundary!
 */
PhysicsOverrides g_physics_overrides = {
    .t_boundary = 13000.0,              /* PHYSICAL value with continuum opacity */
    .ir_thermalization_frac = 0.50,     /* Physical thermalization */
    .base_thermalization_frac = 0.35,   /* Physical thermalization */
    .blue_opacity_scalar = 0.50,        /* Moderate Fe-group reduction */
    .ir_wavelength_min = 7000.0,        /* IR starts at 7000 Å */
    .blue_wavelength_min = 3500.0,      /* Blue range: 3500-4500 Å */
    .blue_wavelength_max = 4500.0,
    /* NEW: Continuum opacity controls */
    .enable_continuum_opacity = true,   /* Enable by default */
    .bf_opacity_scale = 1.0,            /* Standard bound-free */
    .ff_opacity_scale = 1.0,            /* Standard free-free */
    .R_photosphere = 0.0,               /* Set at runtime */
    .enable_dilution_factor = true      /* Enable NLTE dilution */
};

PhysicsOverrides physics_overrides_default(void) {
    PhysicsOverrides defaults = {
        .t_boundary = 13000.0,          /* Physical temperature with continuum */
        .ir_thermalization_frac = 0.50,
        .base_thermalization_frac = 0.35,
        .blue_opacity_scalar = 0.50,
        .ir_wavelength_min = 7000.0,
        .blue_wavelength_min = 3500.0,
        .blue_wavelength_max = 4500.0,
        .enable_continuum_opacity = true,
        .bf_opacity_scale = 1.0,
        .ff_opacity_scale = 1.0,
        .R_photosphere = 0.0,
        .enable_dilution_factor = true,
        /* NEW: Wavelength-dependent fluorescence (Task Order #30 v2.1) */
        .enable_wavelength_fluorescence = true,   /* Enable by default */
        .uv_cutoff_angstrom = 3000.0,             /* UV is λ < 3000 Å */
        .uv_to_blue_probability = 0.85,           /* 85% of UV → blue fluorescence */
        .blue_fluor_min_angstrom = 3500.0,        /* Fluorescence range 3500-5500 Å */
        .blue_fluor_max_angstrom = 5500.0,
        .blue_scatter_probability = 0.70          /* 70% of blue photons scatter */
    };
    return defaults;
}

/* Legacy "hack" mode for backwards compatibility */
PhysicsOverrides physics_overrides_legacy_hack(void) {
    PhysicsOverrides legacy = {
        .t_boundary = 60000.0,          /* UV-Leak hack */
        .ir_thermalization_frac = 0.95, /* IR-Kill hack */
        .base_thermalization_frac = 0.50,
        .blue_opacity_scalar = 0.40,
        .ir_wavelength_min = 7000.0,
        .blue_wavelength_min = 3500.0,
        .blue_wavelength_max = 4500.0,
        .enable_continuum_opacity = false, /* Disabled in hack mode */
        .bf_opacity_scale = 0.0,
        .ff_opacity_scale = 0.0,
        .R_photosphere = 0.0,
        .enable_dilution_factor = false,
        /* Legacy mode: simple thermal re-emission at T_boundary */
        .enable_wavelength_fluorescence = false,
        .uv_cutoff_angstrom = 3000.0,
        .uv_to_blue_probability = 0.0,
        .blue_fluor_min_angstrom = 3500.0,
        .blue_fluor_max_angstrom = 5500.0,
        .blue_scatter_probability = 0.0
    };
    return legacy;
}

void physics_overrides_set(const PhysicsOverrides *overrides) {
    if (overrides) {
        g_physics_overrides = *overrides;
        /* Sync legacy variable */
        g_fe_blue_scale = overrides->blue_opacity_scalar;
    }
}

/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

int simulation_state_init(SimulationState *state,
                          const AtomicData *atomic,
                          int n_shells,
                          double t_exp)
{
    memset(state, 0, sizeof(SimulationState));

    if (n_shells > MAX_SHELLS) {
        fprintf(stderr, "Error: n_shells (%d) exceeds MAX_SHELLS (%d)\n",
                n_shells, MAX_SHELLS);
        return -1;
    }

    state->atomic_data = atomic;
    state->n_shells = n_shells;
    state->t_explosion = t_exp;

    /* Allocate shell states */
    state->shells = (ShellState *)calloc(n_shells, sizeof(ShellState));
    if (!state->shells) {
        return -1;
    }

    /* Initialize each shell */
    for (int i = 0; i < n_shells; i++) {
        state->shells[i].shell_id = i;
        state->shells[i].t_exp = t_exp;
        plasma_state_init(&state->shells[i].plasma);
    }

    /* Allocate global line index */
    state->n_lines_total = atomic->n_lines;
    state->line_freq_order = (int64_t *)malloc(atomic->n_lines * sizeof(int64_t));
    state->line_freq_sorted = (double *)malloc(atomic->n_lines * sizeof(double));

    if (!state->line_freq_order || !state->line_freq_sorted) {
        simulation_state_free(state);
        return -1;
    }

    /* Default solar abundances */
    abundances_set_solar(&state->abundances);

    state->initialized = true;
    return 0;
}

void simulation_set_shell_geometry(SimulationState *state,
                                    int shell_id,
                                    double r_inner,
                                    double r_outer)
{
    if (shell_id < 0 || shell_id >= state->n_shells) return;

    ShellState *shell = &state->shells[shell_id];
    shell->r_inner = r_inner;
    shell->r_outer = r_outer;

    /* Homologous expansion: v = r / t */
    shell->v_inner = r_inner / state->t_explosion;
    shell->v_outer = r_outer / state->t_explosion;
}

void simulation_set_shell_density(SimulationState *state,
                                   int shell_id,
                                   double rho)
{
    if (shell_id < 0 || shell_id >= state->n_shells) return;
    state->shells[shell_id].plasma.rho = rho;
}

void simulation_set_shell_temperature(SimulationState *state,
                                       int shell_id,
                                       double T)
{
    if (shell_id < 0 || shell_id >= state->n_shells) return;
    state->shells[shell_id].plasma.T = T;
}

void simulation_set_abundances(SimulationState *state,
                                const Abundances *ab)
{
    memcpy(&state->abundances, ab, sizeof(Abundances));

    /* Also set for each shell (uniform composition) */
    for (int i = 0; i < state->n_shells; i++) {
        memcpy(&state->shells[i].abundances, ab, sizeof(Abundances));
    }
}

void simulation_set_stratified_abundances(SimulationState *state)
{
    /*
     * TASK ORDER #32: Si-TAPER for Si II Velocity Correction
     *
     * BASELINE CONFIGURATION (restored):
     *   - v < 10,500 km/s: Si = 25% (photospheric zone)
     *   - v > 10,500 km/s: Linear taper from 25% to 2% over full velocity range
     *
     * Combined with OPACITY_SCALE = 0.001 for extreme opacity softening.
     * This configuration previously achieved Δv = +12 km/s from target.
     */

    /* Baseline parameters */
    const double v_cutoff = 10500.0 * 1e5;       /* Start taper at 10,500 km/s */
    const double v_taper_end = 25000.0 * 1e5;    /* Full taper to outer boundary */
    const double Si_inner = 0.25;                 /* Si fraction in photosphere */
    const double Si_outer = 0.02;                 /* Minimal outer Si */

    printf("[ABUNDANCES] Task Order #32: BASELINE CONFIG\n");
    printf("  v < 10,500 km/s: Si=25%% (photospheric zone)\n");
    printf("  v > 10,500 km/s: Si TAPERS from 25%% → 2%%\n");
    printf("  OPACITY_SCALE = 0.001 (extreme softening)\n\n");

    for (int i = 0; i < state->n_shells; i++) {
        ShellState *shell = &state->shells[i];
        Abundances *ab = &shell->abundances;

        /* Shell center velocity */
        double v_center = 0.5 * (shell->v_inner + shell->v_outer);

        memset(ab, 0, sizeof(Abundances));

        if (v_center < v_cutoff) {
            /*
             * PHOTOSPHERIC ZONE (v < 10,500 km/s):
             * Si+Fe mixed - THIS is where Si II 6355 forms!
             */
            ab->mass_fraction[14] = Si_inner;    /* Si - dominant for Si II 6355 */
            ab->mass_fraction[16] = 0.10;        /* S  - for S II */
            ab->mass_fraction[20] = 0.08;        /* Ca */
            ab->mass_fraction[26] = 0.35;        /* Fe - for Fe II */
            ab->mass_fraction[27] = 0.05;        /* Co */
            ab->mass_fraction[28] = 0.07;        /* Ni */
        } else {
            /*
             * OUTER IME LAYER (v > 10,500 km/s):
             * LINEAR TAPER of Si abundance to reduce high-velocity opacity
             */
            double taper_frac = (v_center - v_cutoff) / (v_taper_end - v_cutoff);
            if (taper_frac > 1.0) taper_frac = 1.0;
            if (taper_frac < 0.0) taper_frac = 0.0;

            double X_Si_tapered = Si_inner - (Si_inner - Si_outer) * taper_frac;

            ab->mass_fraction[6]  = 0.05 + 0.10 * taper_frac;  /* C increases outward */
            ab->mass_fraction[8]  = 0.08 + 0.15 * taper_frac;  /* O increases outward */
            ab->mass_fraction[12] = 0.05;                       /* Mg */
            ab->mass_fraction[14] = X_Si_tapered;               /* Si - TAPERED */
            ab->mass_fraction[16] = 0.12;                       /* S */
            ab->mass_fraction[20] = 0.10;                       /* Ca */
            ab->mass_fraction[26] = 0.08 - 0.05 * taper_frac;  /* Fe decreases outward */
            ab->mass_fraction[28] = 0.03;                       /* Ni */
        }

        /* Build element list */
        ab->n_elements = 0;
        for (int Z = 1; Z <= MAX_ATOMIC_NUMBER; Z++) {
            if (ab->mass_fraction[Z] > 0.0) {
                ab->elements[ab->n_elements++] = Z;
            }
        }

        /* Print summary for selected shells */
        double v_km = v_center / 1e5;
        if (i == 0 || i == 3 || i == 5 || i == 10 || i == state->n_shells - 1) {
            const char *zone = (v_center < v_cutoff) ? "PHOTO" : "TAPER";
            printf("  Shell %2d (v=%5.0f km/s) [%s]: Si=%4.1f%%, Fe=%4.1f%%\n",
                   i, v_km, zone,
                   ab->mass_fraction[14] * 100,
                   ab->mass_fraction[26] * 100);
        }
    }

    /* Update global abundances with first shell values */
    memcpy(&state->abundances, &state->shells[0].abundances, sizeof(Abundances));

    printf("\n");
}

void simulation_set_scaled_abundances(SimulationState *state,
                                       double Si_scale, double Fe_scale,
                                       double Ca_scale, double S_scale)
{
    /*
     * Set stratified abundances with user-specified scaling factors.
     * This enables optimizer-driven parameter sweeps over abundances.
     *
     * Scaling is applied AFTER the base stratification, so:
     *   X_Si_final = X_Si_base * Si_scale
     *
     * Mass fractions are renormalized to sum to 1.0 after scaling.
     */
    printf("[ABUNDANCES] Setting SCALED stratified composition:\n");
    printf("  Scaling: Si×%.2f, Fe×%.2f, Ca×%.2f, S×%.2f\n",
           Si_scale, Fe_scale, Ca_scale, S_scale);

    /*
     * Stratification for SN Ia at maximum light (W7-like):
     *   - Photosphere at ~10,000 km/s: Si+Fe mixed zone
     *   - Outer shells (>12,000 km/s): Si-dominated IME layer
     *   - Si II 6355 forms at/near the photosphere
     *
     * With v_inner=10000 km/s, v_outer=25000 km/s, n_shells=30:
     *   Shell 0: 10000-10500 km/s (photosphere)
     *   Shell 4: 12000-12500 km/s (IME transition)
     */
    int inner_outer_boundary = 3;  /* Lowered from 10 to put Si at photosphere */

    for (int i = 0; i < state->n_shells; i++) {
        ShellState *shell = &state->shells[i];
        Abundances *ab = &shell->abundances;

        memset(ab, 0, sizeof(Abundances));

        if (i <= inner_outer_boundary) {
            /*
             * PHOTOSPHERIC ZONE (shells 0-3, v=10000-12000 km/s):
             * Si+Fe mixed zone - Si II 6355 formation region!
             */
            ab->mass_fraction[14] = 0.30 * Si_scale;   /* Si - dominant for Si II 6355 */
            ab->mass_fraction[16] = 0.10 * S_scale;    /* S  - for S II W-feature */
            ab->mass_fraction[20] = 0.08 * Ca_scale;   /* Ca */
            ab->mass_fraction[26] = 0.35 * Fe_scale;   /* Fe - for Fe II features */
            ab->mass_fraction[27] = 0.07;              /* Co */
            ab->mass_fraction[28] = 0.10;              /* Ni */
        } else {
            /*
             * OUTER IME LAYER (shells 4+, v>12000 km/s):
             * Si-dominated intermediate mass elements
             */
            ab->mass_fraction[6]  = 0.05;                  /* C */
            ab->mass_fraction[8]  = 0.08;                  /* O */
            ab->mass_fraction[12] = 0.05;                  /* Mg */
            ab->mass_fraction[14] = 0.40 * Si_scale;       /* Si */
            ab->mass_fraction[16] = 0.15 * S_scale;        /* S */
            ab->mass_fraction[20] = 0.12 * Ca_scale;       /* Ca */
            ab->mass_fraction[26] = 0.10 * Fe_scale;       /* Fe */
            ab->mass_fraction[28] = 0.05;                  /* Ni */
        }

        /* Renormalize mass fractions to sum to 1.0 */
        double total = 0.0;
        for (int Z = 1; Z <= MAX_ATOMIC_NUMBER; Z++) {
            total += ab->mass_fraction[Z];
        }
        if (total > 0.0) {
            for (int Z = 1; Z <= MAX_ATOMIC_NUMBER; Z++) {
                ab->mass_fraction[Z] /= total;
            }
        }

        /* Build element list */
        ab->n_elements = 0;
        for (int Z = 1; Z <= MAX_ATOMIC_NUMBER; Z++) {
            if (ab->mass_fraction[Z] > 0.0) {
                ab->elements[ab->n_elements++] = Z;
            }
        }

        /* Print summary for selected shells */
        if (i == 0 || i == inner_outer_boundary + 1 || i == state->n_shells - 1) {
            double v_center = 0.5 * (shell->v_inner + shell->v_outer) / 1e5;
            const char *zone = (i <= inner_outer_boundary) ? "INNER" : "OUTER";
            printf("  Shell %2d (v=%.0f km/s) [%s]: Si=%.1f%%, S=%.1f%%, Fe=%.1f%%\n",
                   i, v_center, zone,
                   ab->mass_fraction[14] * 100,
                   ab->mass_fraction[16] * 100,
                   ab->mass_fraction[26] * 100);
        }
    }

    /* Update global abundances with outer shell values */
    if (state->n_shells > inner_outer_boundary + 1) {
        memcpy(&state->abundances, &state->shells[inner_outer_boundary + 1].abundances,
               sizeof(Abundances));
    }

    printf("\n");
}

void simulation_state_free(SimulationState *state)
{
    if (state->shells) {
        for (int i = 0; i < state->n_shells; i++) {
            free(state->shells[i].active_lines);
        }
        free(state->shells);
    }

    free(state->line_freq_order);
    free(state->line_freq_sorted);

    memset(state, 0, sizeof(SimulationState));
}

/* ============================================================================
 * PLASMA CALCULATION
 * ============================================================================ */

int simulation_compute_plasma(SimulationState *state)
{
    int failures = 0;

    printf("[SIMULATION] Computing plasma state for %d shells...\n", state->n_shells);

    /* Determine photospheric radius (innermost shell) for dilution factor */
    double R_ph = 0.0;
    if (g_physics_overrides.enable_dilution_factor) {
        if (g_physics_overrides.R_photosphere > 0.0) {
            R_ph = g_physics_overrides.R_photosphere;
        } else if (state->n_shells > 0) {
            /* Default: use inner radius of innermost shell */
            R_ph = state->shells[0].r_inner;
        }
        printf("[SIMULATION] Dilution factor enabled, R_ph = %.3e cm\n", R_ph);
    }

    for (int i = 0; i < state->n_shells; i++) {
        ShellState *shell = &state->shells[i];

        /* Use per-shell abundances if set, otherwise fall back to global */
        const Abundances *ab = (shell->abundances.n_elements > 0) ?
                               &shell->abundances : &state->abundances;

        int status;

        if (g_physics_overrides.enable_dilution_factor && R_ph > 0.0) {
            /* Calculate dilution factor for this shell */
            double r_mid = 0.5 * (shell->r_inner + shell->r_outer);
            double W = calculate_dilution_factor(r_mid, R_ph);

            /* Use diluted ionization solver for NLTE correction */
            status = solve_ionization_balance_diluted(
                state->atomic_data,
                ab,
                shell->plasma.T,
                shell->plasma.rho,
                W,
                &shell->plasma
            );

            /* Diagnostic for outer shells */
            if (i >= state->n_shells - 3) {
                double v_mid = r_mid / state->t_explosion / 1e5;
                printf("  [NLTE] Shell %d (v=%.0f km/s): W=%.4f, n_e=%.2e\n",
                       i, v_mid, W, shell->plasma.n_e);
            }
        } else {
            /* Standard LTE solver */
            status = solve_ionization_balance(
                state->atomic_data,
                ab,
                shell->plasma.T,
                shell->plasma.rho,
                &shell->plasma
            );
        }

        if (status != 0) {
            fprintf(stderr, "  Warning: Shell %d failed to converge\n", i);
            failures++;
        }

        /* Compute Thomson scattering coefficient */
        shell->sigma_thomson_ne = shell->plasma.n_e * SIGMA_THOMSON;
        shell->tau_electron = shell->sigma_thomson_ne * (shell->r_outer - shell->r_inner);

        /* Diagnostic: Print continuum opacity for selected shells */
        if (g_physics_overrides.enable_continuum_opacity && (i == 0 || i == state->n_shells/2)) {
            /* Test at optical wavelength (5000 Å) and IR (8000 Å) */
            double nu_5000 = 2.998e10 / (5000e-8);  /* 5000 Å in Hz */
            double nu_8000 = 2.998e10 / (8000e-8);  /* 8000 Å in Hz */

            double kappa_bf_5000 = calculate_bf_opacity(nu_5000, shell->plasma.T,
                                                         shell->plasma.n_e, &shell->plasma);
            double kappa_ff_5000 = calculate_ff_opacity(nu_5000, shell->plasma.T,
                                                         shell->plasma.n_e, &shell->plasma);
            double kappa_bf_8000 = calculate_bf_opacity(nu_8000, shell->plasma.T,
                                                         shell->plasma.n_e, &shell->plasma);
            double kappa_ff_8000 = calculate_ff_opacity(nu_8000, shell->plasma.T,
                                                         shell->plasma.n_e, &shell->plasma);

            double dr = shell->r_outer - shell->r_inner;
            double tau_cont_5000 = (kappa_bf_5000 + kappa_ff_5000) * dr;
            double tau_cont_8000 = (kappa_bf_8000 + kappa_ff_8000) * dr;

            printf("  [CONTINUUM κ] Shell %d: T=%.0fK, n_e=%.2e\n", i, shell->plasma.T, shell->plasma.n_e);
            printf("    5000Å: κ_bf=%.2e, κ_ff=%.2e, τ_cont=%.2e\n",
                   kappa_bf_5000, kappa_ff_5000, tau_cont_5000);
            printf("    8000Å: κ_bf=%.2e, κ_ff=%.2e, τ_cont=%.2e\n",
                   kappa_bf_8000, kappa_ff_8000, tau_cont_8000);
        }
    }

    printf("[SIMULATION] Plasma computation complete (%d failures)\n", failures);
    return failures;
}

/* ============================================================================
 * CONTINUUM OPACITY IMPLEMENTATION (Task Order #30 v2)
 * ============================================================================
 * Physical bound-free and free-free opacity to replace the "60,000 K hack".
 *
 * The key insight: our original model required T_boundary = 60,000 K because
 * we were missing continuum opacity. Photons in the IR had no mechanism to
 * be absorbed other than lines, leading to IR excess.
 *
 * By implementing bf/ff opacity:
 *   - IR photons can be absorbed by free-free transitions
 *   - UV photons can ionize atoms (bound-free)
 *   - Energy is naturally redistributed without artificial hacks
 */

/* Physical constants for continuum opacity */
#define CONST_SIGMA_BF_0  7.906e-18    /* H ground state bf cross-section [cm²] */
#define CONST_FF_COEFF    3.692e8      /* ff opacity coefficient [cgs] */
#define CONST_H_EV        4.135667696e-15  /* Planck constant [eV·s] */
#define CONST_EV          1.602176634e-12  /* Electron volt [erg] */
#define CONST_K_B_CGS     1.380649e-16     /* Boltzmann constant [erg/K] */
#ifndef CONST_H_CGS
#define CONST_H_CGS       6.62607015e-27   /* Planck constant [erg·s] */
#endif

double calculate_bf_opacity(double nu, double T, double n_e,
                            const PlasmaState *plasma)
{
    /*
     * Bound-free (photoionization) opacity using Kramers' formula:
     *
     *   κ_bf = Σ_ions σ_bf(ν) × n_ion × (1 - e^(-hν/kT))
     *
     * where σ_bf = σ_0 × (Z_eff^4 / n^5) × (ν_n / ν)^3 × g_bf
     *
     * Simplified approach: Sum over dominant species (Si, Fe, Ca, S)
     * that contribute to continuum opacity in SN Ia ejecta.
     */

    if (!g_physics_overrides.enable_continuum_opacity) {
        return 0.0;
    }

    double kT = CONST_K_B_CGS * T;
    double hnu = CONST_H_CGS * nu;

    /* Stimulated emission correction */
    double stim_corr = 1.0;
    if (hnu / kT < 30.0) {
        stim_corr = 1.0 - exp(-hnu / kT);
    }

    double kappa_bf = 0.0;

    /* Sum over important ions for SN Ia:
     * Si II, Si III, Fe II, Fe III, Ca II, S II
     *
     * Use simplified Kramers cross-section with Z_eff from Slater rules
     */

    /* Ionization thresholds (eV) for important species */
    const double chi_Si_II = 16.35;   /* Si II → Si III */
    const double chi_Fe_II = 16.19;   /* Fe II → Fe III */
    const double chi_Ca_II = 11.87;   /* Ca II → Ca III */
    const double chi_S_II  = 23.34;   /* S II → S III */

    /* Effective nuclear charges (Slater rules approximation) */
    const double Z_eff_Si = 4.15;
    const double Z_eff_Fe = 5.5;
    const double Z_eff_Ca = 4.0;
    const double Z_eff_S  = 5.0;

    /* Photon energy in eV */
    double E_photon_eV = hnu / CONST_EV;

    /* Si II contribution */
    if (E_photon_eV > chi_Si_II) {
        double n_Si_II = plasma->n_ion[14][1];
        double nu_thresh = chi_Si_II * CONST_EV / CONST_H_CGS;
        double ratio = nu_thresh / nu;
        double sigma = CONST_SIGMA_BF_0 * pow(Z_eff_Si, 4) / 32.0 * pow(ratio, 3);
        kappa_bf += sigma * n_Si_II * stim_corr;
    }

    /* Fe II contribution */
    if (E_photon_eV > chi_Fe_II) {
        double n_Fe_II = plasma->n_ion[26][1];
        double nu_thresh = chi_Fe_II * CONST_EV / CONST_H_CGS;
        double ratio = nu_thresh / nu;
        double sigma = CONST_SIGMA_BF_0 * pow(Z_eff_Fe, 4) / 32.0 * pow(ratio, 3);
        kappa_bf += sigma * n_Fe_II * stim_corr;
    }

    /* Ca II contribution */
    if (E_photon_eV > chi_Ca_II) {
        double n_Ca_II = plasma->n_ion[20][1];
        double nu_thresh = chi_Ca_II * CONST_EV / CONST_H_CGS;
        double ratio = nu_thresh / nu;
        double sigma = CONST_SIGMA_BF_0 * pow(Z_eff_Ca, 4) / 32.0 * pow(ratio, 3);
        kappa_bf += sigma * n_Ca_II * stim_corr;
    }

    /* S II contribution */
    if (E_photon_eV > chi_S_II) {
        double n_S_II = plasma->n_ion[16][1];
        double nu_thresh = chi_S_II * CONST_EV / CONST_H_CGS;
        double ratio = nu_thresh / nu;
        double sigma = CONST_SIGMA_BF_0 * pow(Z_eff_S, 4) / 32.0 * pow(ratio, 3);
        kappa_bf += sigma * n_S_II * stim_corr;
    }

    return kappa_bf * g_physics_overrides.bf_opacity_scale;
}

double calculate_ff_opacity(double nu, double T, double n_e,
                            const PlasmaState *plasma)
{
    /*
     * Free-free (Bremsstrahlung) opacity using Kramers' law:
     *
     * The classic Kramers' formula for ff opacity per unit mass is:
     *
     *   κ_ff = 3.68 × 10^22 × g_ff × (ρ / T^(7/2)) × (1/ν³) × (1 - e^(-hν/kT))
     *
     * In terms of opacity per unit length (cm⁻¹):
     *
     *   α_ff = κ_ff × ρ
     *        = 3.68 × 10^22 × g_ff × (ρ² / T^(7/2)) × (1/ν³) × (1 - e^(-hν/kT))
     *
     * Or in more explicit form for electron-ion bremsstrahlung:
     *
     *   α_ff = 3.692e8 × g_ff × Z² × n_e × n_ion / (T^(1/2) × ν³) × (1 - e^(-hν/kT))
     *
     * This is the KEY physics for IR absorption in SN ejecta!
     *
     * References:
     *   - Rybicki & Lightman (1979) Eq. 5.18a
     *   - Gray (2005) "Observation and Analysis of Stellar Photospheres"
     */

    if (!g_physics_overrides.enable_continuum_opacity) {
        return 0.0;
    }

    double kT = CONST_K_B_CGS * T;
    double hnu = CONST_H_CGS * nu;

    /* Stimulated emission correction */
    double stim_corr = 1.0;
    if (hnu / kT < 30.0) {
        stim_corr = 1.0 - exp(-hnu / kT);
    }

    /* Gaunt factor (wavelength-averaged) */
    double g_ff = 1.1;

    /* Temperature factor: 1/√T */
    double T_factor = 1.0 / sqrt(T);

    /* Frequency factor: 1/ν³ */
    double nu_factor = 1.0 / (nu * nu * nu);

    /* Sum over ions: Σ Z² × n_ion */
    double Z2_n_sum = 0.0;

    /* Important ions for SN Ia (singly and doubly ionized) */
    /* Z² contributions weighted by ion density */

    /* Silicon: Z=14 */
    Z2_n_sum += 1.0 * 1.0 * plasma->n_ion[14][1];   /* Si II: charge 1 */
    Z2_n_sum += 2.0 * 2.0 * plasma->n_ion[14][2];   /* Si III: charge 2 */

    /* Sulfur: Z=16 */
    Z2_n_sum += 1.0 * 1.0 * plasma->n_ion[16][1];   /* S II */
    Z2_n_sum += 2.0 * 2.0 * plasma->n_ion[16][2];   /* S III */

    /* Calcium: Z=20 */
    Z2_n_sum += 1.0 * 1.0 * plasma->n_ion[20][1];   /* Ca II */
    Z2_n_sum += 2.0 * 2.0 * plasma->n_ion[20][2];   /* Ca III */

    /* Iron: Z=26 */
    Z2_n_sum += 1.0 * 1.0 * plasma->n_ion[26][1];   /* Fe II */
    Z2_n_sum += 2.0 * 2.0 * plasma->n_ion[26][2];   /* Fe III */
    Z2_n_sum += 3.0 * 3.0 * plasma->n_ion[26][3];   /* Fe IV */

    /* Cobalt: Z=27 */
    Z2_n_sum += 1.0 * 1.0 * plasma->n_ion[27][1];   /* Co II */
    Z2_n_sum += 2.0 * 2.0 * plasma->n_ion[27][2];   /* Co III */

    /* Nickel: Z=28 */
    Z2_n_sum += 1.0 * 1.0 * plasma->n_ion[28][1];   /* Ni II */
    Z2_n_sum += 2.0 * 2.0 * plasma->n_ion[28][2];   /* Ni III */

    /* Final ff opacity */
    double kappa_ff = CONST_FF_COEFF * g_ff * T_factor * n_e * Z2_n_sum * nu_factor * stim_corr;

    return kappa_ff * g_physics_overrides.ff_opacity_scale;
}

double calculate_continuum_opacity(double nu, const ShellState *shell)
{
    /*
     * Total continuum opacity: κ_cont = κ_bf + κ_ff
     *
     * In SN Ia ejecta:
     *   - UV: Dominated by bound-free (photoionization)
     *   - IR: Dominated by free-free (bremsstrahlung)
     *
     * This is the KEY physics that allows us to use physical T_boundary!
     */

    if (!g_physics_overrides.enable_continuum_opacity) {
        return 0.0;
    }

    double T = shell->plasma.T;
    double n_e = shell->plasma.n_e;

    double kappa_bf = calculate_bf_opacity(nu, T, n_e, &shell->plasma);
    double kappa_ff = calculate_ff_opacity(nu, T, n_e, &shell->plasma);

    return kappa_bf + kappa_ff;
}

double calculate_tau_continuum(double nu, const ShellState *shell)
{
    /*
     * Continuum optical depth through shell:
     *   τ_cont = κ_cont × Δr
     *
     * where Δr = r_outer - r_inner
     */

    double kappa = calculate_continuum_opacity(nu, shell);
    double dr = shell->r_outer - shell->r_inner;

    return kappa * dr;
}

/* ============================================================================
 * DILUTION FACTOR IMPLEMENTATION (NLTE Correction)
 * ============================================================================ */

double calculate_dilution_factor(double r, double R_ph)
{
    /*
     * Geometric dilution factor:
     *   W = 0.5 × [1 - sqrt(1 - (R_ph/r)²)]
     *
     * Physical meaning:
     *   - At photosphere (r = R_ph): W = 0.5
     *   - Far from photosphere: W → (R_ph/r)²/4 → 0
     *
     * The dilution factor accounts for the fact that the radiation
     * field in the outer ejecta is not isotropic - photons mostly
     * come from the direction of the photosphere.
     */

    if (!g_physics_overrides.enable_dilution_factor || R_ph <= 0.0) {
        return 0.5;  /* No dilution - full radiation field */
    }

    if (r <= R_ph) {
        return 0.5;  /* At or below photosphere */
    }

    double ratio = R_ph / r;
    double ratio_sq = ratio * ratio;

    /* Protect against numerical issues */
    if (ratio_sq >= 1.0) {
        return 0.5;
    }

    double W = 0.5 * (1.0 - sqrt(1.0 - ratio_sq));

    return W;
}

double apply_dilution_to_temperature(double T, double W)
{
    /*
     * In a diluted radiation field, the effective radiation temperature
     * is reduced:
     *   T_rad_eff = W^0.25 × T
     *
     * This mimics the NLTE effect where the radiation field driving
     * ionization is weaker than a full blackbody at temperature T.
     *
     * Note: This is a simplified treatment. Full NLTE requires solving
     * rate equations, but this captures the leading-order effect.
     */

    if (W >= 0.5 || !g_physics_overrides.enable_dilution_factor) {
        return T;  /* No dilution correction */
    }

    /* T_eff = W^0.25 × T */
    double T_eff = pow(W, 0.25) * T;

    /* Don't reduce below 50% of original temperature */
    if (T_eff < 0.5 * T) {
        T_eff = 0.5 * T;
    }

    return T_eff;
}

/* ============================================================================
 * SOBOLEV OPTICAL DEPTH CALCULATION
 * ============================================================================ */

double calculate_tau_sobolev(const Line *line, double n_lower, double t_exp)
{
    /*
     * Sobolev optical depth (oscillator strength form):
     *   τ = (π e² / m_e c) × f_lu × λ × n_lower × t_exp
     *
     * In CGS:
     *   τ = SOBOLEV_CONST × f_lu × λ × n_lower × t_exp
     *
     * where SOBOLEV_CONST = π e² / (m_e c) = 2.654×10⁻² cm² s⁻¹
     *
     * MODIFICATION (Task Order Investigation):
     * -----------------------------------------
     * The raw τ values are ~10^9 due to density normalization.
     * We apply two corrections:
     *   1. OPACITY_SCALE: Rescales τ to bring it into physical range
     *   2. TAU_MAX_CAP: Hard cap for numerical stability
     *
     * Final formula: τ_final = min(τ_raw × OPACITY_SCALE, TAU_MAX_CAP)
     */

    if (n_lower <= 0.0 || line->f_lu <= 0.0) {
        return 0.0;
    }

    double tau = SOBOLEV_CONST * line->f_lu * line->wavelength * n_lower * t_exp;

    /* Apply opacity scaling (if OPACITY_SCALE < 1.0) */
    tau *= OPACITY_SCALE;

    /*
     * Task Order #30: PhysicsOverrides-based Blue Opacity Reduction
     * Reduce Fe-group (Z=21-28) opacity using configurable parameters.
     */
    int Z = line->atomic_number;
    double wl_A = line->wavelength * 1e8;  /* Convert cm to Angstrom */
    if (Z >= 21 && Z <= 28 &&
        wl_A >= g_physics_overrides.blue_wavelength_min &&
        wl_A <= g_physics_overrides.blue_wavelength_max) {
        tau *= g_physics_overrides.blue_opacity_scalar;
    }

    /* Apply hard cap to prevent numerical divergence */
    if (tau > TAU_MAX_CAP) {
        tau = TAU_MAX_CAP;
    }

    return tau;
}

/* Physical constants for Einstein A formulation */
#define CONST_H_CGS     6.62607015e-27    /* Planck constant [erg·s] */
#define CONST_K_B_CGS   1.380649e-16      /* Boltzmann constant [erg/K] */
#define CONST_C_CGS     2.99792458e10     /* Speed of light [cm/s] */
#define CONST_PI        3.14159265358979323846

double calculate_tau_sobolev_A(const Line *line, double n_lower, double t_exp, double T)
{
    /*
     * Sobolev optical depth (Einstein A coefficient form):
     *   τ = (λ³ / 8π) × A_ul × (g_u / g_l) × n_lower × t_exp × (1 - exp(-hν/kT))^-1
     *
     * This is mathematically equivalent to the f_lu form but uses different
     * atomic data inputs. The stimulated emission correction factor
     * (1 - exp(-hν/kT))^-1 accounts for the reduction of effective absorption
     * due to stimulated emission at high temperatures.
     *
     * Reference: TARDIS implementation, Mihalas (1978) Stellar Atmospheres
     */

    if (n_lower <= 0.0 || line->A_ul <= 0.0) {
        return 0.0;
    }

    /* Calculate λ³ / 8π factor */
    double lambda = line->wavelength;  /* cm */
    double lambda_cubed = lambda * lambda * lambda;
    double geom_factor = lambda_cubed / (8.0 * CONST_PI);

    /* Statistical weight ratio (g_u / g_l) */
    /* Note: For Si II 6347: g_u=4, g_l=2, ratio=2 */
    /* We need to look this up from level data, but approximate for now */
    double g_ratio = 2.0;  /* Default for typical dipole transitions */

    /* Stimulated emission correction: (1 - exp(-hν/kT))^-1 */
    double hnu = CONST_H_CGS * line->nu;  /* erg */
    double kT = CONST_K_B_CGS * T;        /* erg */
    double x = hnu / kT;

    double stim_correction;
    if (x > 30.0) {
        /* High frequency / low T: stimulated emission negligible */
        stim_correction = 1.0;
    } else if (x < 0.01) {
        /* Low frequency / high T: classical limit */
        stim_correction = 1.0 / x;  /* ≈ kT / hν */
    } else {
        stim_correction = 1.0 / (1.0 - exp(-x));
    }

    /* Calculate tau */
    double tau = geom_factor * line->A_ul * g_ratio * n_lower * t_exp * stim_correction;

    /* Apply hard cap to prevent numerical divergence */
    if (tau > TAU_MAX_CAP) {
        tau = TAU_MAX_CAP;
    }

    return tau;
}

int64_t calculate_shell_opacities(SimulationState *state, int shell_id)
{
    ShellState *shell = &state->shells[shell_id];
    const AtomicData *atomic = state->atomic_data;

    /* First pass: count active lines */
    int64_t n_active = 0;

    for (int64_t i = 0; i < atomic->n_lines; i++) {
        const Line *line = &atomic->lines[i];

        /* Get lower level population */
        int Z = line->atomic_number;
        int ion = line->ion_number;
        int level_lower = line->level_number_lower;

        /* Get ion density */
        double n_ion = shell->plasma.n_ion[Z][ion];
        if (n_ion <= 0.0) continue;

        /* Calculate level population fraction */
        double U = shell->plasma.partition_function[Z][ion];
        double pop_frac = calculate_level_population_fraction(
            atomic, Z, ion, level_lower, shell->plasma.T, U
        );

        double n_lower = n_ion * pop_frac;

        /* Calculate Sobolev tau */
        double tau = calculate_tau_sobolev(line, n_lower, shell->t_exp);

        if (tau > TAU_MIN_ACTIVE) {
            n_active++;
        }
    }

    /* Allocate active lines array */
    if (shell->active_lines) {
        free(shell->active_lines);
    }

    shell->active_lines = (ActiveLine *)malloc(n_active * sizeof(ActiveLine));
    if (!shell->active_lines && n_active > 0) {
        shell->n_active_lines = 0;
        return 0;
    }

    /* Second pass: populate active lines */
    int64_t idx = 0;

    for (int64_t i = 0; i < atomic->n_lines && idx < n_active; i++) {
        const Line *line = &atomic->lines[i];

        int Z = line->atomic_number;
        int ion = line->ion_number;
        int level_lower = line->level_number_lower;

        double n_ion = shell->plasma.n_ion[Z][ion];
        if (n_ion <= 0.0) continue;

        double U = shell->plasma.partition_function[Z][ion];
        double pop_frac = calculate_level_population_fraction(
            atomic, Z, ion, level_lower, shell->plasma.T, U
        );

        double n_lower = n_ion * pop_frac;
        double tau = calculate_tau_sobolev(line, n_lower, shell->t_exp);

        if (tau > TAU_MIN_ACTIVE) {
            shell->active_lines[idx].line_idx = i;
            shell->active_lines[idx].nu = line->nu;
            shell->active_lines[idx].tau_sobolev = tau;
            idx++;
        }
    }

    shell->n_active_lines = idx;
    return idx;
}

/* Comparison function for qsort: sort by frequency ascending */
static int compare_active_lines(const void *a, const void *b)
{
    const ActiveLine *la = (const ActiveLine *)a;
    const ActiveLine *lb = (const ActiveLine *)b;

    if (la->nu < lb->nu) return -1;
    if (la->nu > lb->nu) return 1;
    return 0;
}

int64_t simulation_compute_opacities(SimulationState *state)
{
    printf("[SIMULATION] Computing Sobolev opacities for %ld lines...\n",
           (long)state->atomic_data->n_lines);

    /*
     * DIAGNOSTIC: Verify Si II 6347/6371 lines exist in atomic data
     */
    const AtomicData *atomic = state->atomic_data;
    int si_ii_found = 0;
    double si_ii_6347_nu = 0.0, si_ii_6371_nu = 0.0;
    int64_t si_ii_6347_idx = -1, si_ii_6371_idx = -1;

    for (int64_t i = 0; i < atomic->n_lines; i++) {
        const Line *line = &atomic->lines[i];
        if (line->atomic_number == 14 && line->ion_number == 1) {
            double wl_A = line->wavelength / 1e-8;  /* cm -> Å */
            if (wl_A > 6345.0 && wl_A < 6350.0) {
                si_ii_6347_nu = line->nu;
                si_ii_6347_idx = i;
                si_ii_found++;
            } else if (wl_A > 6369.0 && wl_A < 6374.0) {
                si_ii_6371_nu = line->nu;
                si_ii_6371_idx = i;
                si_ii_found++;
            }
        }
    }

    printf("[SI II CHECK] Found %d Si II 6355 doublet lines in atomic data:\n", si_ii_found);
    if (si_ii_6347_idx >= 0) {
        printf("  Si II 6347: idx=%ld, nu=%.4e Hz, f_lu=%.3f\n",
               (long)si_ii_6347_idx, si_ii_6347_nu,
               atomic->lines[si_ii_6347_idx].f_lu);
    }
    if (si_ii_6371_idx >= 0) {
        printf("  Si II 6371: idx=%ld, nu=%.4e Hz, f_lu=%.3f\n",
               (long)si_ii_6371_idx, si_ii_6371_nu,
               atomic->lines[si_ii_6371_idx].f_lu);
    }

    int64_t total_active = 0;

    for (int i = 0; i < state->n_shells; i++) {
        int64_t n_active = calculate_shell_opacities(state, i);

        /* Sort active lines by frequency */
        if (n_active > 0) {
            qsort(state->shells[i].active_lines, n_active,
                  sizeof(ActiveLine), compare_active_lines);

            /* Build frequency-binned index for this shell */
            build_frequency_binned_index(&state->shells[i].line_index,
                                          state->shells[i].active_lines,
                                          n_active);
        }

        total_active += n_active;

        /*
         * DIAGNOSTIC: Check Si II tau in photospheric shells (0-3)
         */
        if (i <= 3 && si_ii_found > 0) {
            ShellState *shell = &state->shells[i];

            /* Find Si II in active lines */
            double tau_6347 = 0.0, tau_6371 = 0.0;
            for (int64_t j = 0; j < shell->n_active_lines; j++) {
                if (shell->active_lines[j].line_idx == si_ii_6347_idx) {
                    tau_6347 = shell->active_lines[j].tau_sobolev;
                }
                if (shell->active_lines[j].line_idx == si_ii_6371_idx) {
                    tau_6371 = shell->active_lines[j].tau_sobolev;
                }
            }

            /* Also compute Si II ion population for diagnostic */
            double n_Si_II = shell->plasma.n_ion[14][1];
            double v_mid = 0.5 * (shell->v_inner + shell->v_outer) / 1e5;

            printf("  [SI II τ] Shell %d (v=%.0f km/s): n_Si_II=%.2e, "
                   "τ(6347)=%.2e, τ(6371)=%.2e\n",
                   i, v_mid, n_Si_II, tau_6347, tau_6371);
        }

        if ((i + 1) % 5 == 0 || i == state->n_shells - 1) {
            printf("  Shell %d: %ld active lines\n", i, (long)n_active);
        }
    }

    state->total_active_lines = total_active;
    state->opacities_computed = true;

    /* Build global frequency index */
    simulation_build_line_index(state);

    /* Calculate memory usage */
    size_t mem = state->n_shells * sizeof(ShellState);
    mem += total_active * sizeof(ActiveLine);
    mem += state->n_lines_total * (sizeof(int64_t) + sizeof(double));
    state->memory_usage_mb = mem / 1e6;

    printf("[SIMULATION] Total active lines: %ld (%.2f MB)\n",
           (long)total_active, state->memory_usage_mb);

    return total_active;
}

void simulation_build_line_index(SimulationState *state)
{
    const AtomicData *atomic = state->atomic_data;

    /* Copy from atomic data's sorted index if available */
    if (atomic->sorted_line_indices && atomic->sorted_line_nu) {
        memcpy(state->line_freq_order, atomic->sorted_line_indices,
               atomic->n_lines * sizeof(int64_t));
        memcpy(state->line_freq_sorted, atomic->sorted_line_nu,
               atomic->n_lines * sizeof(double));
    } else {
        /* Build our own sorted index */
        for (int64_t i = 0; i < atomic->n_lines; i++) {
            state->line_freq_order[i] = i;
            state->line_freq_sorted[i] = atomic->lines[i].nu;
        }

        /* Sort by frequency (simple bubble sort for now - can optimize) */
        for (int64_t i = 0; i < atomic->n_lines - 1; i++) {
            for (int64_t j = i + 1; j < atomic->n_lines; j++) {
                if (state->line_freq_sorted[j] < state->line_freq_sorted[i]) {
                    double tmp_nu = state->line_freq_sorted[i];
                    state->line_freq_sorted[i] = state->line_freq_sorted[j];
                    state->line_freq_sorted[j] = tmp_nu;

                    int64_t tmp_idx = state->line_freq_order[i];
                    state->line_freq_order[i] = state->line_freq_order[j];
                    state->line_freq_order[j] = tmp_idx;
                }
            }
        }
    }
}

/* ============================================================================
 * FREQUENCY-BINNED INDEX IMPLEMENTATION
 * ============================================================================
 * O(1) lookup for finding starting search position in sorted line arrays.
 * Divides frequency range into logarithmic bins for fast index lookup.
 */

void build_frequency_binned_index(FrequencyBinnedIndex *index,
                                   const ActiveLine *lines,
                                   int64_t n_lines)
{
    if (n_lines == 0) {
        index->initialized = false;
        index->n_lines = 0;
        return;
    }

    /* Set frequency range (use global constants or derive from data) */
    index->nu_min = LINE_INDEX_NU_MIN;
    index->nu_max = LINE_INDEX_NU_MAX;
    index->n_lines = n_lines;

    /* Calculate logarithmic bin width */
    double log_nu_min = log(index->nu_min);
    double log_nu_max = log(index->nu_max);
    index->d_log_nu = (log_nu_max - log_nu_min) / LINE_INDEX_N_BINS;

    /* Initialize all bin starts to end of array */
    for (int b = 0; b <= LINE_INDEX_N_BINS; b++) {
        index->bin_start[b] = n_lines;
    }

    /* Single pass: assign bin starts */
    for (int64_t i = 0; i < n_lines; i++) {
        double nu = lines[i].nu;

        /* Compute bin index */
        int bin;
        if (nu <= index->nu_min) {
            bin = 0;
        } else if (nu >= index->nu_max) {
            bin = LINE_INDEX_N_BINS;
        } else {
            bin = (int)((log(nu) - log_nu_min) / index->d_log_nu);
            if (bin < 0) bin = 0;
            if (bin >= LINE_INDEX_N_BINS) bin = LINE_INDEX_N_BINS - 1;
        }

        /* Update bin start if this is the first line in this bin */
        if (index->bin_start[bin] > i) {
            index->bin_start[bin] = i;
        }
    }

    /* Fill in empty bins: each empty bin should point to next non-empty bin */
    int64_t next_start = n_lines;
    for (int b = LINE_INDEX_N_BINS; b >= 0; b--) {
        if (index->bin_start[b] == n_lines) {
            index->bin_start[b] = next_start;
        } else {
            next_start = index->bin_start[b];
        }
    }

    index->initialized = true;
}

int64_t frequency_index_find_start(const FrequencyBinnedIndex *index, double nu)
{
    if (!index->initialized || index->n_lines == 0) {
        return 0;
    }

    /* Clamp to valid range */
    if (nu <= index->nu_min) {
        return 0;
    }
    if (nu >= index->nu_max) {
        return index->n_lines;
    }

    /* Compute bin index */
    double log_nu_min = log(index->nu_min);
    int bin = (int)((log(nu) - log_nu_min) / index->d_log_nu);

    /* Clamp bin */
    if (bin < 0) bin = 0;
    if (bin >= LINE_INDEX_N_BINS) bin = LINE_INDEX_N_BINS - 1;

    return index->bin_start[bin];
}

/* ============================================================================
 * LINE LOOKUP (HOT PATH)
 * ============================================================================ */

int64_t find_lines_in_window(const ShellState *shell,
                              double nu_min, double nu_max,
                              int64_t *first_idx, int64_t *last_idx)
{
    if (shell->n_active_lines == 0) {
        *first_idx = 0;
        *last_idx = 0;
        return 0;
    }

    const ActiveLine *lines = shell->active_lines;
    int64_t n = shell->n_active_lines;

    /* Binary search for first line with nu >= nu_min */
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (lines[mid].nu < nu_min) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    *first_idx = lo;

    /* Binary search for first line with nu > nu_max */
    hi = n;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (lines[mid].nu <= nu_max) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    *last_idx = lo;

    return *last_idx - *first_idx;
}

double get_next_line_interaction(const ShellState *shell,
                                  double nu_cmf, double r, double mu,
                                  double t_exp,
                                  int64_t *line_idx, double *tau_line)
{
    *line_idx = -1;
    *tau_line = 0.0;

    if (shell->n_active_lines == 0) {
        return 1e99;  /* MISS_DISTANCE */
    }

    /*
     * In Sobolev approximation, a photon interacts with a line when its
     * comoving frequency equals the line frequency.
     *
     * The comoving frequency changes as the photon moves through the
     * velocity gradient:
     *   ν_cmf(s) = ν_cmf(0) × (1 - β × Δμ)
     *
     * For homologous expansion:
     *   ν_cmf(r) ∝ 1 / r  (for radial motion)
     *
     * Distance to resonance with line ν_line:
     *   d = (c / (ν_cmf / ν_line - 1)) × something...
     *
     * Simplified: find closest line in frequency space
     */

    const ActiveLine *lines = shell->active_lines;
    int64_t n = shell->n_active_lines;

    /*
     * Find lines near current comoving frequency.
     *
     * In a SN ejecta with v ~ 10000 km/s (β ~ 0.03), the Doppler shift
     * across a shell can change the comoving frequency by several percent.
     * We search within ±5% to catch all potential resonances.
     */
    double nu_min = nu_cmf * 0.95;
    double nu_max = nu_cmf * 1.05;

    int64_t first, last;
    int64_t n_in_range = find_lines_in_window(shell, nu_min, nu_max, &first, &last);

    if (n_in_range == 0) {
        /* No lines in immediate vicinity - find closest */
        /* Binary search for closest line */
        int64_t lo = 0, hi = n;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if (lines[mid].nu < nu_cmf) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        /* Check neighbors */
        double min_dist = 1e99;
        int64_t best_idx = -1;

        if (lo > 0 && fabs(lines[lo-1].nu - nu_cmf) < min_dist) {
            min_dist = fabs(lines[lo-1].nu - nu_cmf);
            best_idx = lo - 1;
        }
        if (lo < n && fabs(lines[lo].nu - nu_cmf) < min_dist) {
            min_dist = fabs(lines[lo].nu - nu_cmf);
            best_idx = lo;
        }

        if (best_idx >= 0) {
            *line_idx = lines[best_idx].line_idx;
            *tau_line = lines[best_idx].tau_sobolev;

            /* Approximate distance based on frequency shift needed */
            double delta_nu = fabs(lines[best_idx].nu - nu_cmf);
            double v_th = r / t_exp;  /* Thermal velocity ~ expansion velocity */
            double d = CONST_C * delta_nu / (nu_cmf * v_th / r) * t_exp;

            return d;
        }

        return 1e99;
    }

    /* Find the line with highest optical depth in range */
    double max_tau = 0.0;
    int64_t best_idx = -1;

    for (int64_t i = first; i < last; i++) {
        if (lines[i].tau_sobolev > max_tau) {
            max_tau = lines[i].tau_sobolev;
            best_idx = i;
        }
    }

    if (best_idx >= 0) {
        *line_idx = lines[best_idx].line_idx;
        *tau_line = lines[best_idx].tau_sobolev;

        /* Distance to line (simplified) */
        double delta_nu = fabs(lines[best_idx].nu - nu_cmf);
        if (delta_nu < 1e-10 * nu_cmf) {
            return 0.0;  /* Already at resonance */
        }

        double v = r / t_exp;
        double d = CONST_C * delta_nu / (nu_cmf * v / r);

        return d;
    }

    return 1e99;
}

/* ============================================================================
 * TRANSPORT INTEGRATION
 * ============================================================================ */

int simulation_from_numba_model(SimulationState *state,
                                 const NumbaModel *model,
                                 const AtomicData *atomic)
{
    int n_shells = (int)model->n_shells;

    int status = simulation_state_init(state, atomic, n_shells, model->time_explosion);
    if (status != 0) return status;

    /* Set shell geometry */
    for (int i = 0; i < n_shells; i++) {
        double r_in = model->r_inner[i];
        double r_out = model->r_outer[i];

        simulation_set_shell_geometry(state, i, r_in, r_out);
    }

    return 0;
}

void simulation_to_numba_plasma(const SimulationState *state,
                                 NumbaPlasma *plasma)
{
    /* Copy electron densities */
    for (int i = 0; i < state->n_shells && i < (int)plasma->n_shells; i++) {
        if (plasma->electron_density) {
            plasma->electron_density[i] = state->shells[i].plasma.n_e;
        }
    }
}

/* ============================================================================
 * DIAGNOSTICS
 * ============================================================================ */

void simulation_print_summary(const SimulationState *state)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║              SIMULATION STATE SUMMARY                         ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Shells:           %10d                                 ║\n", state->n_shells);
    printf("║  t_explosion:      %10.2f s (%.2f days)                  ║\n",
           state->t_explosion, state->t_explosion / 86400.0);
    printf("║  Total lines:      %10ld                                 ║\n", (long)state->n_lines_total);
    printf("║  Active lines:     %10ld                                 ║\n", (long)state->total_active_lines);
    printf("║  Memory usage:     %10.2f MB                              ║\n", state->memory_usage_mb);
    printf("║  Initialized:      %10s                                 ║\n",
           state->initialized ? "Yes" : "No");
    printf("║  Opacities ready:  %10s                                 ║\n",
           state->opacities_computed ? "Yes" : "No");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
}

void simulation_print_shell(const SimulationState *state, int shell_id)
{
    if (shell_id < 0 || shell_id >= state->n_shells) return;

    const ShellState *shell = &state->shells[shell_id];

    printf("\n=== Shell %d ===\n", shell_id);
    printf("  Geometry:\n");
    printf("    r_inner: %.4e cm (v = %.4e cm/s)\n", shell->r_inner, shell->v_inner);
    printf("    r_outer: %.4e cm (v = %.4e cm/s)\n", shell->r_outer, shell->v_outer);
    printf("  Plasma:\n");
    printf("    T:       %.1f K\n", shell->plasma.T);
    printf("    ρ:       %.4e g/cm³\n", shell->plasma.rho);
    printf("    n_e:     %.4e cm⁻³\n", shell->plasma.n_e);
    printf("  Opacities:\n");
    printf("    Active lines: %ld\n", (long)shell->n_active_lines);
    printf("    τ_electron:   %.4e\n", shell->tau_electron);

    /* Show top 5 strongest lines */
    if (shell->n_active_lines > 0) {
        printf("  Strongest lines:\n");
        printf("    %-12s %-12s %-12s\n", "λ [Å]", "τ_Sobolev", "Z/ion");

        /* Find top 5 by tau */
        double max_tau[5] = {0};
        int64_t max_idx[5] = {-1, -1, -1, -1, -1};

        for (int64_t i = 0; i < shell->n_active_lines; i++) {
            double tau = shell->active_lines[i].tau_sobolev;
            for (int j = 0; j < 5; j++) {
                if (tau > max_tau[j]) {
                    /* Shift down */
                    for (int k = 4; k > j; k--) {
                        max_tau[k] = max_tau[k-1];
                        max_idx[k] = max_idx[k-1];
                    }
                    max_tau[j] = tau;
                    max_idx[j] = i;
                    break;
                }
            }
        }

        for (int j = 0; j < 5 && max_idx[j] >= 0; j++) {
            int64_t i = max_idx[j];
            int64_t line_idx = shell->active_lines[i].line_idx;
            const Line *line = &state->atomic_data->lines[line_idx];
            double wl_A = line->wavelength / CONST_ANGSTROM;

            printf("    %-12.2f %-12.4e %d/%d\n",
                   wl_A, shell->active_lines[i].tau_sobolev,
                   line->atomic_number, line->ion_number);
        }
    }
}

int simulation_validate(const SimulationState *state)
{
    int errors = 0;

    if (!state->initialized) {
        printf("Error: State not initialized\n");
        errors++;
    }

    if (state->n_shells <= 0) {
        printf("Error: No shells defined\n");
        errors++;
    }

    for (int i = 0; i < state->n_shells; i++) {
        const ShellState *shell = &state->shells[i];

        if (shell->plasma.T <= 0) {
            printf("Error: Shell %d has invalid temperature\n", i);
            errors++;
        }

        if (shell->plasma.rho <= 0) {
            printf("Error: Shell %d has invalid density\n", i);
            errors++;
        }

        if (shell->r_outer <= shell->r_inner) {
            printf("Error: Shell %d has invalid geometry\n", i);
            errors++;
        }
    }

    return errors;
}
