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
    .enable_dilution_factor = true,     /* Enable NLTE dilution */
    /* Line interaction type: MACROATOM for proper P-Cygni profiles */
    .line_interaction_type = LINE_MACROATOM,
    .use_macro_atom = 1
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
        .blue_scatter_probability = 0.70,         /* 70% of blue photons scatter */
        /* Line interaction type: MACROATOM for proper P-Cygni profiles */
        .line_interaction_type = LINE_MACROATOM,
        /* Enable macro-atom by default for physical fluorescence */
        .use_macro_atom = 1,
        /* Virtual packet spawning (TARDIS-style) - disabled by default */
        .enable_virtual_packets = 0,
        .n_virtual_packets = 10
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
        .blue_scatter_probability = 0.0,
        /* Legacy mode: use SCATTER */
        .line_interaction_type = LINE_SCATTER,
        .use_macro_atom = 0,
        /* Virtual packet spawning - disabled in legacy mode */
        .enable_virtual_packets = 0,
        .n_virtual_packets = 10
    };
    return legacy;
}

/* TARDIS-compatible mode: NO extra physics layers
 * Line interaction type determines photon fate (SCATTER, DOWNBRANCH, or MACROATOM).
 * All LUMINA-specific thermalization/fluorescence DISABLED.
 *
 * Default to MACROATOM mode for proper P-Cygni profiles.
 * Set LUMINA_LINE_INTERACTION_TYPE=0 for SCATTER mode if needed.
 */
PhysicsOverrides physics_overrides_tardis_mode(void) {
    PhysicsOverrides tardis = {
        /* Use shell temperature (set dynamically during transport) */
        .t_boundary = 10000.0,
        /* NO extra thermalization - line interaction type decides everything */
        .ir_thermalization_frac = 0.0,
        .base_thermalization_frac = 0.0,
        /* No blue opacity scaling */
        .blue_opacity_scalar = 1.0,
        .ir_wavelength_min = 7000.0,
        .blue_wavelength_min = 3500.0,
        .blue_wavelength_max = 4500.0,
        /* Continuum opacity - disabled like TARDIS by default */
        .enable_continuum_opacity = false,
        .bf_opacity_scale = 0.0,
        .ff_opacity_scale = 0.0,
        .R_photosphere = 0.0,
        .enable_dilution_factor = true,     /* TARDIS uses W factor */
        /* NO wavelength-dependent fluorescence tricks */
        .enable_wavelength_fluorescence = false,
        .uv_cutoff_angstrom = 3000.0,
        .uv_to_blue_probability = 0.0,
        .blue_fluor_min_angstrom = 3500.0,
        .blue_fluor_max_angstrom = 5500.0,
        .blue_scatter_probability = 0.0,
        /* Line interaction: MACROATOM for proper P-Cygni profiles
         * Set LUMINA_LINE_INTERACTION_TYPE=0 for SCATTER mode if needed */
        .line_interaction_type = LINE_MACROATOM,
        .use_macro_atom = 1,  /* Deprecated - use line_interaction_type */
        /* Virtual packet spawning - ENABLED in TARDIS mode for proper comparison */
        .enable_virtual_packets = 1,
        .n_virtual_packets = 10
    };
    return tardis;
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
     * TASK ORDER #017 - TARDIS-MATCHED STRATIFICATION
     * ================================================
     *
     * Updated to match TARDIS benchmark model abundances:
     *   - v < 12,000 km/s: Fe-rich core (X_Si = 10%, X_Fe = 60%)
     *   - 12,000 < v < 15,000 km/s: IME zone (X_Si = 50%)
     *   - 15,000 < v < 18,000 km/s: IME-O zone (X_Si = 40%)
     *   - v > 18,000 km/s: C/O envelope (X_Si = 2%)
     *
     * Key insight: TARDIS has a SHARP cutoff at v = 18,000 km/s!
     * This prevents Si II from forming in the outermost shells,
     * fixing the ~2,400 km/s velocity offset.
     *
     * Reference: sn2011fe_synthetic/abundances.csv (TARDIS export)
     */

    /* TARDIS-matched velocity boundaries */
    const double v_fe_core_end   = 12000.0 * 1e5;   /* Fe-core → IME transition */
    const double v_ime_zone_end  = 15000.0 * 1e5;   /* IME → IME-O transition */
    const double v_si_cutoff     = 18000.0 * 1e5;   /* SHARP Si cutoff! */

    printf("[ABUNDANCES] Task Order #017 - TARDIS-MATCHED STRATIFICATION\n");
    printf("  v < 12,000 km/s: Fe-core (X_Si=10%%, X_Fe=60%%)\n");
    printf("  12,000 < v < 15,000 km/s: IME zone (X_Si=50%%)\n");
    printf("  15,000 < v < 18,000 km/s: IME-O zone (X_Si=40%%)\n");
    printf("  v > 18,000 km/s: C/O envelope (X_Si=2%%) **SHARP CUTOFF**\n\n");

    for (int i = 0; i < state->n_shells; i++) {
        ShellState *shell = &state->shells[i];
        Abundances *ab = &shell->abundances;

        /* Shell center velocity */
        double v_center = 0.5 * (shell->v_inner + shell->v_outer);

        memset(ab, 0, sizeof(Abundances));

        if (v_center < v_fe_core_end) {
            /*
             * FE-RICH CORE (v < 12,000 km/s):
             * TARDIS shells 0-3: dominated by Fe-group elements
             */
            ab->mass_fraction[14] = 0.10;    /* Si = 10% */
            ab->mass_fraction[16] = 0.03;    /* S */
            ab->mass_fraction[20] = 0.02;    /* Ca */
            ab->mass_fraction[26] = 0.60;    /* Fe = 60% - dominant */
            ab->mass_fraction[27] = 0.10;    /* Co */
            ab->mass_fraction[28] = 0.15;    /* Ni */
        } else if (v_center < v_ime_zone_end) {
            /*
             * IME ZONE (12,000 < v < 15,000 km/s):
             * TARDIS shells 4-9: Si-dominated intermediate mass elements
             */
            ab->mass_fraction[14] = 0.50;    /* Si = 50% - MAXIMUM */
            ab->mass_fraction[16] = 0.20;    /* S */
            ab->mass_fraction[18] = 0.03;    /* Ar */
            ab->mass_fraction[20] = 0.10;    /* Ca */
            ab->mass_fraction[26] = 0.15;    /* Fe */
            ab->mass_fraction[28] = 0.02;    /* Ni */
        } else if (v_center < v_si_cutoff) {
            /*
             * IME-O ZONE (15,000 < v < 18,000 km/s):
             * TARDIS shells 10-15: Si with O mixing
             */
            ab->mass_fraction[8]  = 0.30;    /* O = 30% */
            ab->mass_fraction[12] = 0.05;    /* Mg */
            ab->mass_fraction[14] = 0.40;    /* Si = 40% */
            ab->mass_fraction[16] = 0.15;    /* S */
            ab->mass_fraction[20] = 0.05;    /* Ca */
            ab->mass_fraction[26] = 0.05;    /* Fe */
        } else {
            /*
             * C/O ENVELOPE (v > 18,000 km/s):
             * TARDIS shells 16-29: Carbon-Oxygen rich, almost NO Si!
             * THIS IS THE KEY FIX: Sharp cutoff prevents Si II at high v.
             */
            ab->mass_fraction[6]  = 0.50;    /* C = 50% */
            ab->mass_fraction[8]  = 0.48;    /* O = 48% */
            ab->mass_fraction[14] = 0.02;    /* Si = 2% - MINIMAL */
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
        if (i == 0 || i == 3 || i == 5 || i == 10 || i == 15 || i == 20 || i == state->n_shells - 1) {
            const char *zone;
            if (v_center < v_fe_core_end) zone = "FE-CORE";
            else if (v_center < v_ime_zone_end) zone = "IME";
            else if (v_center < v_si_cutoff) zone = "IME-O";
            else zone = "C/O-ENV";

            printf("  Shell %2d (v=%5.0f km/s) [%7s]: X_Si=%5.1f%%, X_Fe=%5.1f%%\n",
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

    /* Sort DESCENDING by frequency (high nu first).
     * This is required for TARDIS-style transport: as photons redshift
     * traveling outward, they encounter lines from high to low frequency. */
    if (la->nu > lb->nu) return -1;
    if (la->nu < lb->nu) return 1;
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
 * TARDIS-STYLE LINE INTERACTION (CORRECT SOBOLEV ALGORITHM)
 * ============================================================================
 *
 * This implements the CORRECT Sobolev approximation as used in TARDIS:
 *
 * 1. Lines are sorted by frequency (decreasing, so higher nu first)
 * 2. Packet tracks next_line_id to know where it left off
 * 3. As packet travels, its comoving frequency redshifts (decreases)
 * 4. It passes through lines in order, accumulating optical depth
 * 5. When cumulative τ exceeds random τ_event, it interacts with that line
 *
 * The key insight is that packets interact with lines IN ORDER OF FREQUENCY,
 * not by "jumping" to the strongest line in a window.
 */

double get_line_interaction_tardis_style(
    const ShellState *shell,
    double nu_cmf, double r, double mu,
    double t_exp,
    int64_t start_line_id,
    double tau_event,
    double d_max,  /* Maximum distance to search for lines */
    int64_t *line_idx, double *tau_line,
    int64_t *next_line_id)
{
    *line_idx = -1;
    *tau_line = 0.0;
    *next_line_id = start_line_id;

    if (shell->n_active_lines == 0) {
        return 1e99;
    }

    const ActiveLine *lines = shell->active_lines;
    int64_t n_lines = shell->n_active_lines;

    /* Lines are sorted by frequency (descending: high nu first).
     * As the packet travels outward, its comoving frequency DEcreases
     * (redshifts due to velocity gradient).
     * So we iterate through lines in order of decreasing frequency.
     */

    double tau_cumulative = 0.0;
    double v = r / t_exp;  /* Local expansion velocity */

    /* Find starting position: first line with nu <= nu_cmf */
    int64_t cur_line = start_line_id;

    /* If starting fresh (start_line_id == 0), find where nu_cmf sits */
    if (start_line_id == 0) {
        /* Binary search for first line with nu <= nu_cmf */
        int64_t lo = 0, hi = n_lines;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if (lines[mid].nu > nu_cmf) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        cur_line = lo;
    }

    (void)d_max;  /* Kept for API compatibility */

    /* Cumulative τ approach (TARDIS-style):
     * Accumulate τ across lines until τ_cumulative > τ_event, then interact
     * with the line that pushed us over the threshold.
     *
     * Key insight: tau_event is sampled ONCE per packet trajectory. We need
     * to track how much τ has been accumulated so far. When we reach a line
     * and τ_cumulative + τ_line > τ_event, we interact with that line.
     *
     * Note: The caller passes tau_event but we don't have tau_accumulated from
     * the packet. This function will accumulate τ starting from 0, which is
     * reset for this shell/direction. For proper multi-step tracking, we'd
     * need to track tau_accumulated in the packet struct.
     */
    for (int64_t i = cur_line; i < n_lines; i++) {
        double nu_line = lines[i].nu;

        /* Skip lines with frequency higher than current comoving frequency */
        if (nu_line > nu_cmf * 1.001) {
            continue;
        }

        /* Calculate distance to this line */
        double nu_ratio = nu_cmf / nu_line;
        double d_line_i;

        if (nu_ratio < 1.0001) {
            /* Already at or past this line */
            d_line_i = 1e8;  /* Very small: 1 km */
        } else {
            d_line_i = CONST_C * (nu_ratio - 1.0) * t_exp;
        }

        /* Get τ for this line */
        double tau_this_line = lines[i].tau_sobolev;

        /* Accumulate τ */
        tau_cumulative += tau_this_line;

        /* Check if we exceed the random threshold */
        if (tau_cumulative > tau_event) {
            /* Interact with THIS line */
            *line_idx = lines[i].line_idx;
            *tau_line = tau_this_line;
            *next_line_id = i;
            return d_line_i;
        }

        /* Stop searching if too far below current frequency */
        if (nu_line < nu_cmf * 0.85) {
            break;
        }
    }

    /* No interaction - return large distance */
    *next_line_id = n_lines;
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

/* ============================================================================
 * TEMPERATURE ITERATION (Radiative Equilibrium)
 * ============================================================================
 * TARDIS-style temperature convergence using MC J-estimators.
 *
 * Physical Basis:
 * ---------------
 * In radiative equilibrium, the radiation field J_ν and temperature T are
 * related through the mean intensity integral:
 *
 *   J = ∫ J_ν dν  (frequency-integrated mean intensity)
 *
 * For a grey atmosphere in radiative equilibrium:
 *   σT⁴/π = J
 *
 * Therefore:
 *   T_rad = (π J / σ)^{1/4}
 *
 * where σ = 5.670374e-5 erg/cm²/s/K⁴ (Stefan-Boltzmann constant)
 *
 * MC Estimator:
 * -------------
 * During transport, we accumulate:
 *   j_estimator[shell] += energy × distance
 *
 * After normalization by volume and total luminosity:
 *   J_est = j_estimator / (4π × volume × c × Δt)
 *
 * The factor 4π accounts for solid angle integration, c converts path length
 * to time, and Δt is the simulation time step.
 *
 * Reference: Lucy 2005, A&A 429, 19; TARDIS documentation
 */

/* Stefan-Boltzmann constant [erg/cm²/s/K⁴] */
#define CONST_STEFAN_BOLTZMANN 5.670374419e-5

/* Speed of light [cm/s] - ensure consistent with other modules */
#ifndef CONST_C
#define CONST_C 2.99792458e10
#endif

/* Pi */
#ifndef CONST_PI
#define CONST_PI 3.14159265358979323846
#endif

/**
 * Initialize MC estimators
 *
 * Allocates arrays for j_estimator, nu_bar_estimator, and volume.
 * Optionally allocates j_blue_estimator for line-specific tracking.
 */
int mc_estimators_init(MCEstimators *est, int n_shells, int64_t n_lines)
{
    memset(est, 0, sizeof(MCEstimators));

    est->n_shells = n_shells;
    est->n_lines = n_lines;
    est->total_packets = 0;
    est->total_energy = 0.0;

    /* Allocate primary estimator arrays */
    est->j_estimator = (double *)calloc(n_shells, sizeof(double));
    est->nu_bar_estimator = (double *)calloc(n_shells, sizeof(double));
    est->volume = (double *)calloc(n_shells, sizeof(double));

    if (!est->j_estimator || !est->nu_bar_estimator || !est->volume) {
        mc_estimators_free(est);
        return -1;
    }

    /* Optionally allocate j_blue for line-specific estimators */
    if (n_lines > 0) {
        est->j_blue_estimator = (double *)calloc(n_lines * n_shells, sizeof(double));
        if (!est->j_blue_estimator) {
            mc_estimators_free(est);
            return -1;
        }
    }

    return 0;
}

/**
 * Reset all estimators to zero
 *
 * Called at the start of each temperature iteration to clear
 * accumulated values from previous MC run.
 */
void mc_estimators_reset(MCEstimators *est)
{
    if (est->j_estimator) {
        memset(est->j_estimator, 0, est->n_shells * sizeof(double));
    }
    if (est->nu_bar_estimator) {
        memset(est->nu_bar_estimator, 0, est->n_shells * sizeof(double));
    }
    if (est->j_blue_estimator && est->n_lines > 0) {
        memset(est->j_blue_estimator, 0, est->n_lines * est->n_shells * sizeof(double));
    }

    est->total_packets = 0;
    est->total_energy = 0.0;
}

/**
 * Free estimator memory
 */
void mc_estimators_free(MCEstimators *est)
{
    free(est->j_estimator);
    free(est->nu_bar_estimator);
    free(est->j_blue_estimator);
    free(est->volume);

    memset(est, 0, sizeof(MCEstimators));
}

/**
 * Update estimators during MC transport
 *
 * Called for each path segment during packet propagation.
 * Accumulates energy × distance for mean intensity estimation.
 *
 * @param est        Estimator structure
 * @param shell_id   Current shell index
 * @param energy_cmf Comoving frame energy [erg]
 * @param distance   Path length in shell [cm]
 * @param nu_cmf     Comoving frame frequency [Hz]
 */
void mc_estimators_update(MCEstimators *est, int shell_id,
                          double energy_cmf, double distance, double nu_cmf)
{
    if (shell_id < 0 || shell_id >= est->n_shells) {
        return;
    }

    /*
     * The MC estimator for mean intensity is:
     *   j_est += ε × l
     *
     * where ε is the packet energy and l is the path length.
     * This sums the energy-weighted path length through each shell.
     *
     * After normalization: J = j_est / (4π V c Δt)
     */
    est->j_estimator[shell_id] += energy_cmf * distance;

    /*
     * Frequency-weighted estimator for radiation temperature:
     *   nu_bar_est += ε × l × ν
     *
     * Used to compute the mean frequency of the radiation field.
     */
    est->nu_bar_estimator[shell_id] += energy_cmf * distance * nu_cmf;
}

/**
 * Compute shell volumes for normalization
 *
 * V = (4π/3) × (r_out³ - r_in³)
 *
 * These are needed to convert accumulated estimators to physical
 * mean intensity values.
 */
void mc_estimators_compute_volumes(MCEstimators *est, const SimulationState *state)
{
    for (int i = 0; i < est->n_shells && i < state->n_shells; i++) {
        double r_in = state->shells[i].r_inner;
        double r_out = state->shells[i].r_outer;

        /* Shell volume: V = (4π/3)(r_out³ - r_in³) */
        double V = (4.0 / 3.0) * CONST_PI *
                   (r_out * r_out * r_out - r_in * r_in * r_in);

        est->volume[i] = V;
    }
}

/**
 * Normalize estimators after MC run
 *
 * Converts accumulated sums to physical mean intensity J [erg/cm²/s/sr].
 *
 * The normalization factor accounts for:
 *   - Volume of each shell (spatial averaging)
 *   - Speed of light (path length to time conversion)
 *   - Solid angle (4π for full sphere)
 *
 * Final formula:
 *   J = j_est / (4π × V × c)
 *
 * NOTE: When packets carry PHYSICAL energy (erg/s), we do NOT multiply
 * by total_energy again. The total_energy parameter is stored for
 * reference but not used in normalization.
 *
 * For backwards compatibility with normalized packets (sum=1), pass
 * total_energy = L_requested to scale up to physical units.
 */
void mc_estimators_normalize(MCEstimators *est, double total_energy)
{
    if (total_energy <= 0.0) {
        fprintf(stderr, "[MC_ESTIMATORS] Warning: total_energy = 0, skipping normalization\n");
        return;
    }

    est->total_energy = total_energy;

    for (int64_t i = 0; i < est->n_shells; i++) {
        double V = est->volume[i];
        if (V <= 0.0) continue;

        /*
         * Normalization factor: 1 / (4π × V × c)
         *
         * The factor of 4π comes from integrating over solid angle.
         * The factor of c converts path length (cm) to time (s).
         *
         * After normalization, J has units [erg/cm²/s/sr].
         *
         * NOTE: When packets carry physical luminosity (erg/s), the
         * j_estimator accumulates (erg/s × cm). Dividing by (V × c)
         * gives (erg/s / cm² / sr) = mean intensity.
         *
         * We do NOT multiply by total_energy since packets already
         * carry physical energy. This is DIFFERENT from TARDIS where
         * packets are normalized to sum=1 and then scaled.
         */
        double norm = 1.0 / (4.0 * CONST_PI * V * CONST_C);

        est->j_estimator[i] *= norm;
        est->nu_bar_estimator[i] *= norm;
    }
}

/**
 * Get radiation temperature from J-estimator
 *
 * Uses Stefan-Boltzmann law inverted:
 *   T_rad = (π J / σ)^{1/4}
 *
 * @param J_est  Mean intensity [erg/cm²/s/sr]
 * @return Radiation temperature [K]
 */
double J_to_T_rad(double J_est)
{
    if (J_est <= 0.0) {
        return 0.0;
    }

    /*
     * From Stefan-Boltzmann:
     *   F = σ T⁴  (flux)
     *   J = F / π (mean intensity for isotropic radiation)
     *
     * Therefore:
     *   J = σ T⁴ / π
     *   T = (π J / σ)^{1/4}
     */
    double T_rad = pow(CONST_PI * J_est / CONST_STEFAN_BOLTZMANN, 0.25);

    return T_rad;
}

/**
 * Get mean intensity from temperature (inverse)
 *
 * J = σ T⁴ / π
 *
 * @param T  Temperature [K]
 * @return Mean intensity [erg/cm²/s/sr]
 */
double T_to_J(double T)
{
    if (T <= 0.0) {
        return 0.0;
    }

    double T4 = T * T * T * T;
    double J = CONST_STEFAN_BOLTZMANN * T4 / CONST_PI;

    return J;
}

/**
 * Update shell temperatures using dilution factor approach
 *
 * TARDIS-style temperature iteration using the geometric dilution factor:
 *   W = 0.5 × [1 - sqrt(1 - (R_in/r)²)]
 *
 * For a shell at radius r with inner boundary R_in:
 *   T_rad = T_inner × W^0.25
 *
 * This gives the expected temperature drop due to geometric dilution
 * of radiation from the inner boundary.
 *
 * The MC estimator j_est is used to REFINE this estimate:
 *   - High j_est in a shell → more absorption → increase T
 *   - Low j_est → less absorption → decrease T
 *
 * Combined formula:
 *   T_new = T_old + damping × (T_target - T_old)
 *
 * where T_target = T_inner × W^0.25 × correction_factor
 *
 * @param state   Simulation state to update (temperatures modified in place)
 * @param est     Normalized MC estimators (used for correction)
 * @param damping Damping factor (0.5-0.8 recommended)
 * @return Maximum relative temperature change (for convergence check)
 */
double simulation_update_temperatures(SimulationState *state,
                                       const MCEstimators *est,
                                       double damping)
{
    double max_delta_T = 0.0;

    printf("[T-ITERATION] Updating shell temperatures (damping=%.2f):\n", damping);

    /* Get inner boundary temperature */
    double T_inner = state->shells[0].plasma.T;
    double R_inner = state->shells[0].r_inner;

    /* Compute mean J across all shells (for relative scaling) */
    double J_mean = 0.0;
    int n_valid = 0;
    for (int i = 0; i < state->n_shells && i < est->n_shells; i++) {
        if (est->j_estimator[i] > 0.0) {
            J_mean += est->j_estimator[i];
            n_valid++;
        }
    }
    if (n_valid > 0) J_mean /= n_valid;

    for (int i = 0; i < state->n_shells && i < est->n_shells; i++) {
        double T_old = state->shells[i].plasma.T;
        double r_mid = 0.5 * (state->shells[i].r_inner + state->shells[i].r_outer);

        /*
         * Compute dilution factor W (TARDIS formula)
         * W = 0.5 × [1 - sqrt(1 - (R_in/r)²)]
         *
         * At r = R_in: W = 0.5
         * At r >> R_in: W → R_in²/(4r²) ≈ 0
         */
        double x = R_inner / r_mid;
        double W;
        if (x >= 1.0) {
            W = 0.5;  /* At or inside inner boundary */
        } else if (x < 0.01) {
            W = x * x / 4.0;  /* Far field approximation */
        } else {
            W = 0.5 * (1.0 - sqrt(1.0 - x * x));
        }

        /*
         * Base temperature from geometric dilution:
         *   T_rad = T_inner × W^0.25
         *
         * For W = 0.5: T_rad = 0.84 × T_inner
         * For W = 0.1: T_rad = 0.56 × T_inner
         */
        double T_geometric = T_inner * pow(W, 0.25);

        /*
         * MC correction: adjust based on relative J in this shell
         * If j_est > J_mean: shell is hotter than average → increase T
         * If j_est < J_mean: shell is cooler → decrease T
         *
         * Use mild correction to avoid instability
         */
        double correction = 1.0;
        if (J_mean > 0.0 && est->j_estimator[i] > 0.0) {
            double j_ratio = est->j_estimator[i] / J_mean;
            /* Limit correction to ±20% */
            correction = 1.0 + 0.1 * (j_ratio - 1.0);
            if (correction < 0.8) correction = 0.8;
            if (correction > 1.2) correction = 1.2;
        }

        double T_target = T_geometric * correction;

        /* Apply damping to prevent oscillations */
        double T_new = T_old + damping * (T_target - T_old);

        /* Enforce physical bounds (Task Order #018 fix: raised minimum to 5000 K)
         * Minimum 5000 K prevents ionization collapse at low T
         * Maximum is 1.1 × T_inner to prevent unphysical heating */
        if (T_new < 5000.0) T_new = 5000.0;
        if (T_new > T_inner * 1.1) T_new = T_inner * 1.1;

        /* Compute relative change */
        double delta_T = fabs(T_new - T_old) / T_old;
        if (delta_T > max_delta_T) {
            max_delta_T = delta_T;
        }

        /* Update temperature */
        state->shells[i].plasma.T = T_new;

        /* Print diagnostics for selected shells */
        if (i == 0 || i == state->n_shells / 2 || i == state->n_shells - 1) {
            double v_mid = 0.5 * (state->shells[i].v_inner + state->shells[i].v_outer) / 1e5;
            printf("  Shell %2d (v=%5.0f km/s): T_old=%6.0fK, W=%.3f, T_geo=%6.0fK, T_new=%6.0fK (ΔT=%.1f%%)\n",
                   i, v_mid, T_old, W, T_geometric, T_new, delta_T * 100.0);
        }
    }

    printf("  Max relative T change: %.2f%%\n", max_delta_T * 100.0);

    return max_delta_T;
}

/* ============================================================================
 * LUMINOSITY ESTIMATORS (TARDIS-style Convergence Strategy)
 * ============================================================================
 * Implementation of luminosity tracking for T_inner update.
 *
 * TARDIS uses luminosity convergence to adjust the inner boundary temperature:
 *   - If L_emitted > L_requested: T_inner is too high, reduce it
 *   - If L_emitted < L_requested: T_inner is too low, increase it
 *
 * The update formula uses L ∝ T^4 (Stefan-Boltzmann):
 *   T_new = T_old × (L_emitted / L_requested)^0.25
 */

void luminosity_estimators_init(LuminosityEstimators *lum,
                                 double L_requested,
                                 double T_inner,
                                 double fraction)
{
    lum->L_requested = L_requested;
    lum->L_emitted = 0.0;
    lum->L_absorbed = 0.0;
    lum->L_inner = 0.0;
    lum->fraction = (fraction > 0.0 && fraction <= 1.0) ? fraction : 0.8;
    lum->T_inner = T_inner;
    lum->T_inner_new = T_inner;
}

void luminosity_estimators_reset(LuminosityEstimators *lum)
{
    lum->L_emitted = 0.0;
    lum->L_absorbed = 0.0;
    lum->L_inner = 0.0;
    /* Keep L_requested, T_inner, and fraction */
}

void luminosity_estimators_add_emitted(LuminosityEstimators *lum, double energy)
{
    lum->L_emitted += energy;
}

void luminosity_estimators_add_absorbed(LuminosityEstimators *lum, double energy)
{
    lum->L_absorbed += energy;
    lum->L_inner += energy;  /* Absorbed packets contribute to inner luminosity */
}

double luminosity_update_T_inner(LuminosityEstimators *lum, double damping)
{
    /*
     * FIXED TARDIS-STYLE T_inner update (Task Order #018 fix)
     *
     * Problem: Original implementation had a positive feedback loop:
     *   - If escape fraction < target fraction, L_ratio < 1
     *   - This causes T to decrease continuously
     *   - Lower T → even lower escape fraction → runaway cooling
     *
     * Fix: Use ADAPTIVE target based on actual escape fraction, and limit
     *      the maximum correction per iteration to prevent instability.
     *
     * TARDIS actually computes T_rad from W × T_inner using the dilution
     * factor W. For shell temperatures, they use:
     *   T_rad = W^0.25 × T_inner
     *
     * For inner boundary, the key insight is that we want radiative
     * equilibrium, not luminosity matching. We keep T_inner stable and
     * only update shell temperatures.
     */

    if (lum->L_requested <= 0.0 || lum->L_emitted <= 0.0) {
        /* Can't compute correction - keep current T_inner */
        lum->T_inner_new = lum->T_inner;
        return lum->T_inner;
    }

    /* Compute actual escape fraction */
    double total_L = lum->L_emitted + lum->L_absorbed;
    double actual_escape_frac = (total_L > 0) ? lum->L_emitted / total_L : 0.5;

    /* Use ADAPTIVE target: blend fixed fraction with actual escape fraction
     * This prevents runaway when the model has different opacity than assumed.
     * Weight: 0.3 × fixed_fraction + 0.7 × actual_escape_fraction
     */
    double adaptive_frac = 0.3 * lum->fraction + 0.7 * actual_escape_frac;
    if (adaptive_frac < 0.3) adaptive_frac = 0.3;  /* Minimum bound */
    if (adaptive_frac > 0.9) adaptive_frac = 0.9;  /* Maximum bound */

    double L_target = lum->L_requested * adaptive_frac;
    double L_ratio = lum->L_emitted / L_target;

    /* LIMIT correction factor to prevent instability
     * Max correction: ±10% per iteration (0.9 to 1.1)
     * This is crucial for convergence stability!
     */
    double correction = pow(L_ratio, 0.25);
    if (correction < 0.90) correction = 0.90;  /* Max 10% decrease */
    if (correction > 1.10) correction = 1.10;  /* Max 10% increase */

    /* Target temperature */
    double T_target = lum->T_inner * correction;

    /* Apply damping (reduced from 0.7 to 0.5 for more stability) */
    double effective_damping = (damping > 0.5) ? 0.5 : damping;
    double T_new = lum->T_inner + effective_damping * (T_target - lum->T_inner);

    /* Enforce physical bounds (raised minimum from 2000K to 5000K) */
    if (T_new < 5000.0) T_new = 5000.0;
    if (T_new > 100000.0) T_new = 100000.0;

    lum->T_inner_new = T_new;

    printf("[LUMINOSITY] L_emitted=%.3e, L_target=%.3e (frac=%.2f→%.2f), ratio=%.3f\n",
           lum->L_emitted, L_target, lum->fraction, adaptive_frac, L_ratio);
    printf("[LUMINOSITY] T_inner: %.0fK → %.0fK (correction=%.3f, damping=%.2f)\n",
           lum->T_inner, T_new, correction, effective_damping);

    /* Update T_inner for next iteration */
    lum->T_inner = T_new;

    return T_new;
}

bool luminosity_converged(const LuminosityEstimators *lum, double threshold)
{
    if (lum->L_requested <= 0.0) {
        return false;
    }

    /* Target is L_emitted = L_requested × fraction */
    double L_target = lum->L_requested * lum->fraction;
    double relative_error = fabs(lum->L_emitted - L_target) / L_target;
    return relative_error < threshold;
}

/* ============================================================================
 * RADIATION FIELD (for stimulated emission)
 * ============================================================================
 * Implementation of mean intensity storage for macro-atom stimulated emission.
 */

int radiation_field_init(RadiationField *rf, int64_t n_lines)
{
    rf->n_lines = n_lines;
    rf->initialized = false;

    if (n_lines <= 0) {
        rf->J_nu = NULL;
        return 0;
    }

    rf->J_nu = (double *)calloc(n_lines, sizeof(double));
    if (!rf->J_nu) {
        return -1;
    }

    return 0;
}

void radiation_field_free(RadiationField *rf)
{
    free(rf->J_nu);
    rf->J_nu = NULL;
    rf->n_lines = 0;
    rf->initialized = false;
}

void radiation_field_from_estimators(RadiationField *rf,
                                      const MCEstimators *est,
                                      const SimulationState *state,
                                      const AtomicData *atomic)
{
    /*
     * Populate J_nu at each line frequency by interpolating from shell J-estimators.
     *
     * For each line, we estimate J_ν at the line frequency by using the
     * shell-averaged J estimator from the shell where the line is most active.
     *
     * This is a simplified approach - a full implementation would track
     * frequency-resolved J in each shell, but that requires significant memory.
     *
     * Physical approximation:
     * We use the average J across all shells as a first approximation.
     * This is reasonable for lines that form across many shells.
     */

    if (!rf->J_nu || rf->n_lines == 0 || !est || !state || !atomic) {
        return;
    }

    /* Compute average J across shells */
    double J_avg = 0.0;
    int n_valid = 0;

    for (int64_t i = 0; i < est->n_shells; i++) {
        if (est->j_estimator[i] > 0.0) {
            J_avg += est->j_estimator[i];
            n_valid++;
        }
    }

    if (n_valid > 0) {
        J_avg /= n_valid;
    }

    /* Assign average J to all line frequencies
     * A more sophisticated approach would weight by shell temperature
     * or interpolate based on line formation depth, but this provides
     * a reasonable first approximation for stimulated emission.
     */
    for (int64_t i = 0; i < rf->n_lines && i < atomic->n_lines; i++) {
        rf->J_nu[i] = J_avg;
    }

    rf->initialized = true;

    printf("[RADIATION FIELD] Populated J_nu for %ld lines (J_avg = %.2e erg/cm²/s/Hz/sr)\n",
           (long)rf->n_lines, J_avg);
}

/* ============================================================================
 * TASK ORDER #034: TARDIS PLASMA STATE INJECTION
 * ============================================================================
 * Load pre-computed plasma state from TARDIS directly, bypassing all C-side
 * ionization and level population calculations.
 *
 * This ensures "Code-level 1:1 Correspondence" in transport by using
 * TARDIS's own calculated optical environment.
 * ============================================================================ */

#ifdef HAVE_HDF5
#include <hdf5.h>
#include <hdf5_hl.h>

/**
 * Load TARDIS plasma state from HDF5 file
 *
 * File structure expected:
 *   /geometry/r_inner, r_outer, v_inner, v_outer, t_explosion
 *   /plasma/electron_densities, t_rad, w
 *   /ion_number_density/data[n_ions, n_shells], atomic_number[], ion_number[]
 *   /tau_sobolev/data[n_lines, n_shells] (optional)
 *
 * @param state      SimulationState to populate
 * @param atomic     AtomicData (for line information)
 * @param filename   Path to HDF5 file
 * @return 0 on success, -1 on error
 */
int simulation_load_plasma_state(SimulationState *state,
                                  const AtomicData *atomic,
                                  const char *filename)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║   TASK ORDER #035: Loading TARDIS Plasma State (Golden Data)  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    printf("[Inject] Opening plasma state file: %s\n", filename);

    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "[Inject] Error: Cannot open file %s\n", filename);
        return -1;
    }

    int n_shells = state->n_shells;
    int64_t n_lines = atomic->n_lines;
    hsize_t dims[2];
    H5T_class_t type_class;
    size_t type_size;

    /* ================================================================
     * 1. Load Electron Density (n_e)
     * ================================================================ */
    double *n_e_buffer = (double *)malloc(n_shells * sizeof(double));
    if (H5LTread_dataset_double(file_id, "/electron_density", n_e_buffer) < 0) {
        fprintf(stderr, "[Inject] Error: Failed to read electron_density\n");
        free(n_e_buffer);
        H5Fclose(file_id);
        return -1;
    }

    /* ================================================================
     * 2. Load Electron Temperature (t_electrons)
     * ================================================================ */
    double *t_e_buffer = (double *)malloc(n_shells * sizeof(double));
    if (H5LTread_dataset_double(file_id, "/t_electrons", t_e_buffer) < 0) {
        fprintf(stderr, "[Inject] Error: Failed to read t_electrons\n");
        free(n_e_buffer);
        free(t_e_buffer);
        H5Fclose(file_id);
        return -1;
    }

    /* Apply Plasma Properties to Shells */
    for (int i = 0; i < n_shells; i++) {
        state->shells[i].plasma.n_e = n_e_buffer[i];
        state->shells[i].plasma.T = t_e_buffer[i];

        /* Update Thomson opacity based on new n_e */
        state->shells[i].sigma_thomson_ne = n_e_buffer[i] * SIGMA_THOMSON;
        state->shells[i].tau_electron = state->shells[i].sigma_thomson_ne *
            (state->shells[i].r_outer - state->shells[i].r_inner);
    }

    printf("[Inject] Updated T and n_e for %d shells.\n", n_shells);
    printf("[Inject]   n_e:   %.3e - %.3e cm^-3\n", n_e_buffer[0], n_e_buffer[n_shells-1]);
    printf("[Inject]   T_e:   %.0f - %.0f K\n", t_e_buffer[0], t_e_buffer[n_shells-1]);

    free(n_e_buffer);
    free(t_e_buffer);

    /* ================================================================
     * 3. Load Sobolev Optical Depths (The "Answer Key")
     * ================================================================ */
    /* TARDIS Export Format: (N_lines, N_shells) flattened row-major */
    if (H5LTget_dataset_info(file_id, "/tau_sobolev", dims, &type_class, &type_size) < 0) {
        fprintf(stderr, "[Inject] Error: Failed to get tau_sobolev info\n");
        H5Fclose(file_id);
        return -1;
    }

    int64_t file_n_lines = dims[0];
    int file_n_shells = (int)dims[1];

    printf("[Inject] Tau Sobolev dimensions: %ld lines × %d shells\n",
           (long)file_n_lines, file_n_shells);

    if (file_n_shells != n_shells) {
        fprintf(stderr, "[Inject] Error: Shell count mismatch! File=%d, State=%d\n",
                file_n_shells, n_shells);
        H5Fclose(file_id);
        return -1;
    }

    /* Allocate buffer for all taus */
    size_t total_taus = file_n_lines * file_n_shells;
    double *tau_buffer = (double *)malloc(total_taus * sizeof(double));
    if (!tau_buffer) {
        fprintf(stderr, "[Inject] Memory allocation failed for tau buffer (%.2f MB)\n",
                total_taus * 8.0 / 1e6);
        H5Fclose(file_id);
        return -1;
    }

    if (H5LTread_dataset_double(file_id, "/tau_sobolev", tau_buffer) < 0) {
        fprintf(stderr, "[Inject] Failed to read tau_sobolev\n");
        free(tau_buffer);
        H5Fclose(file_id);
        return -1;
    }

    /* Compute statistics */
    double tau_max = 0.0, tau_sum = 0.0;
    int64_t n_active_total = 0;

    for (size_t i = 0; i < total_taus; i++) {
        if (tau_buffer[i] > TAU_MIN_ACTIVE) {
            n_active_total++;
            tau_sum += tau_buffer[i];
            if (tau_buffer[i] > tau_max) tau_max = tau_buffer[i];
        }
    }

    printf("[Inject] Tau statistics:\n");
    printf("[Inject]   Max tau:     %.2e\n", tau_max);
    printf("[Inject]   Mean tau:    %.4f (of active)\n",
           n_active_total > 0 ? tau_sum / n_active_total : 0.0);
    printf("[Inject]   Active:      %ld / %ld\n", (long)n_active_total, (long)total_taus);

    /* ================================================================
     * 4. Distribute Taus to Shells (Populate Active Lines)
     * ================================================================ */
    printf("[Inject] Populating active lines in shells...\n");

    /* Clear existing active lines */
    for (int s = 0; s < n_shells; s++) {
        state->shells[s].n_active_lines = 0;
    }

    /* Use the minimum of file lines and atomic lines */
    int64_t lines_to_process = file_n_lines < n_lines ? file_n_lines : n_lines;

    int64_t total_injected = 0;
    for (int s = 0; s < n_shells; s++) {
        for (int64_t l = 0; l < lines_to_process; l++) {
            /* Access: line index l, shell index s */
            /* TARDIS python .values is (lines, shells), C row-major: l * n_shells + s */
            double tau = tau_buffer[l * n_shells + s];

            if (tau > TAU_MIN_ACTIVE) {
                int64_t idx = state->shells[s].n_active_lines;

                if (idx < MAX_ACTIVE_LINES) {
                    state->shells[s].active_lines[idx].line_idx = l;
                    state->shells[s].active_lines[idx].nu = atomic->lines[l].nu;
                    state->shells[s].active_lines[idx].tau_sobolev = tau;
                    state->shells[s].n_active_lines++;
                    total_injected++;
                }
            }
        }
    }

    printf("[Inject] Injection complete. Total active line-shell pairs: %ld\n",
           (long)total_injected);

    /* ================================================================
     * 5. Load Ion Densities (for verification)
     * ================================================================ */
    if (H5Lexists(file_id, "/ion_density", H5P_DEFAULT) > 0) {
        H5LTget_dataset_info(file_id, "/ion_density", dims, NULL, NULL);
        int n_ions = (int)dims[0];
        printf("[Inject] Ion densities available: %d ions\n", n_ions);

        /* Could load and populate plasma.n_ion if needed */
    }

    free(tau_buffer);
    H5Fclose(file_id);

    printf("\n[Inject] *** GOLDEN DATA INJECTION COMPLETE ***\n");
    printf("[Inject] LUMINA is now running with TARDIS-computed tau_sobolev.\n");
    printf("[Inject] If spectrum differs from TARDIS, transport logic has bugs.\n\n");

    return 0;
}

#else
/* Stub when HDF5 is not available */
int simulation_load_plasma_state(SimulationState *state,
                                  const AtomicData *atomic,
                                  const char *filename)
{
    (void)state;
    (void)atomic;
    fprintf(stderr, "[INJECT] ERROR: HDF5 support not compiled in\n");
    fprintf(stderr, "[INJECT] Rebuild with: make HAVE_HDF5=1\n");
    return -1;
}
#endif /* HAVE_HDF5 */
