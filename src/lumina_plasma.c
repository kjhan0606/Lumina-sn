/* lumina_plasma.c — Phase 4: Plasma Solver and Convergence
 * Implements TARDIS mc_rad_field_solver.py for T_rad, W updates.
 * Implements T_inner convergence from escape fraction. */

#include "lumina.h" /* Phase 4 - Step 1 */

#ifdef __cplusplus   /* Phase 6 - Step 9: extern C guard for NVCC */
extern "C" {         /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */

/* ============================================================ */
/* Phase 4 - Step 2: Radiation field solver                     */
/* (mc_rad_field_solver.py: estimate_dilute_planck_radiation_field) */
/* ============================================================ */

void solve_radiation_field(Estimators *est, double time_explosion,
                            double time_simulation, double *volume,
                            OpacityState *opacity, PlasmaState *plasma) {
    int n_shells = est->n_shells; /* Phase 4 - Step 2 */

    for (int i = 0; i < n_shells; i++) { /* Phase 4 - Step 2 */
        /* Phase 4 - Step 2: T_rad from nubar/j ratio */
        /* TARDIS: T_RADIATIVE_ESTIMATOR_CONSTANT * nu_bar / j */
        if (est->j_estimator[i] > 0.0) { /* Phase 4 - Step 2 */
            double T_rad_new = T_RADIATIVE_CONSTANT * /* Phase 4 - Step 2 */
                est->nu_bar_estimator[i] / est->j_estimator[i]; /* Phase 4 - Step 2 */

            /* Phase 4 - Step 2: W from j vs Planck(T_rad) */
            /* TARDIS: W = j / (4 * sigma_sb * T^4 * t_sim * V) */
            double W_new = est->j_estimator[i] / /* Phase 4 - Step 2 */
                (4.0 * SIGMA_SB * pow(T_rad_new, 4) * /* Phase 4 - Step 2 */
                 time_simulation * volume[i]); /* Phase 4 - Step 2 */

            plasma->T_rad[i] = T_rad_new; /* Phase 4 - Step 2 */
            plasma->W[i] = W_new; /* Phase 4 - Step 2 */
        }
    }
}

/* ============================================================ */
/* Phase 4 - Step 3: T_inner update from escape fraction        */
/* ============================================================ */

void update_t_inner(MCConfig *config, double escape_fraction) {
    /* Phase 4 - Step 3: TARDIS convergence formula */
    /* T_inner_new = T_inner * (L_requested / L_emitted)^0.25 */
    /* L_emitted = L_inner * escape_fraction */
    /* L_inner = 4 * pi * r_inner^2 * sigma_sb * T_inner^4 */
    /* So: T_inner_new = T_inner * (1 / escape_fraction)^0.25 */
    if (escape_fraction > 0.0) { /* Phase 4 - Step 3 */
        double correction = pow(1.0 / escape_fraction, 0.25); /* Phase 4 - Step 3 */
        /* Phase 4 - Step 3: Apply damping */
        double damped = 1.0 + config->damping_constant * (correction - 1.0); /* Phase 4 - Step 3 */
        config->T_inner *= damped; /* Phase 4 - Step 3 */
    }
}

/* ============================================================ */
/* Phase 4 - Step 4: Recompute tau_sobolev from new plasma      */
/* This is needed for convergence iterations.                   */
/* For the initial implementation, we keep tau_sobolev frozen    */
/* from TARDIS (since we're validating transport, not plasma).   */
/* ============================================================ */

/* Phase 4 - Step 4: Recompute transition probabilities from j_blue */
/* In the TARDIS-faithful version, we'd recompute macro-atom */
/* probabilities here using the MC j_blue estimators. */
/* For now: probabilities are frozen from TARDIS export. */

/* Phase 4 - Step 5: Spectrum binning */
void bin_escaped_packet(Spectrum *spec, double nu, double energy,
                         double time_explosion) {
    /* Phase 4 - Step 5: Convert frequency to wavelength in Angstrom */
    double lambda_A = C_SPEED_OF_LIGHT / nu * 1.0e8; /* Phase 4 - Step 5 */

    /* Phase 4 - Step 5: Find bin index */
    if (lambda_A < spec->lambda_min || lambda_A >= spec->lambda_max) { /* Phase 4 - Step 5 */
        return; /* Phase 4 - Step 5: outside spectrum range */
    }

    double dlambda = (spec->lambda_max - spec->lambda_min) / spec->n_bins; /* Phase 4 - Step 5 */
    int bin = (int)((lambda_A - spec->lambda_min) / dlambda); /* Phase 4 - Step 5 */
    if (bin >= 0 && bin < spec->n_bins) { /* Phase 4 - Step 5 */
        /* Phase 4 - Step 5: Add luminosity density */
        /* L_lambda = E_packet / (t_exp * dlambda) */
        spec->flux[bin] += energy / (time_explosion * dlambda); /* Phase 4 - Step 5 */
    }
}

#ifdef __cplusplus   /* Phase 6 - Step 9: close extern C guard */
}                    /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */
