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
                            OpacityState *opacity, PlasmaState *plasma,
                            double damping_constant) {
    int n_shells = est->n_shells; /* Phase 4 - Step 2 */

    for (int i = 0; i < n_shells; i++) { /* Phase 4 - Step 2 */
        /* Phase 4 - Step 2: T_rad from nubar/j ratio */
        /* TARDIS: T_RADIATIVE_ESTIMATOR_CONSTANT * nu_bar / j */
        if (est->j_estimator[i] > 0.0) { /* Phase 4 - Step 2 */
            double T_rad_est = T_RADIATIVE_CONSTANT * /* Phase 4 - Step 2 */
                est->nu_bar_estimator[i] / est->j_estimator[i]; /* Phase 4 - Step 2 */

            /* Phase 4 - Step 2: W from j vs Planck(T_rad) */
            /* TARDIS: W = j / (4 * sigma_sb * T^4 * t_sim * V) */
            double W_est = est->j_estimator[i] / /* Phase 4 - Step 2 */
                (4.0 * SIGMA_SB * pow(T_rad_est, 4) * /* Phase 4 - Step 2 */
                 time_simulation * volume[i]); /* Phase 4 - Step 2 */

            /* Task #072: TARDIS damping (base.py: converge() for W and T_rad)
             * new_value = old_value + damping_constant * (estimated - old_value)
             * damping_constant = 0.5 by default in TARDIS */
            plasma->T_rad[i] = plasma->T_rad[i] +
                damping_constant * (T_rad_est - plasma->T_rad[i]);
            plasma->W[i] = plasma->W[i] +
                damping_constant * (W_est - plasma->W[i]);
        }
    }
}

/* ============================================================ */
/* Phase 4 - Step 3: T_inner update from escape fraction        */
/* ============================================================ */

void update_t_inner(MCConfig *config, double L_emitted) {
    /* Phase 4 - Step 3: TARDIS convergence formula (base.py: estimate_t_inner)
     * T_inner_est = T_inner * (L_emitted / L_requested)^(t_inner_update_exponent)
     * t_inner_update_exponent = -0.5 (TARDIS default)
     * Then damping: T_inner_new = T_inner + d * (T_inner_est - T_inner) */
    if (L_emitted > 0.0) {
        double luminosity_ratio = L_emitted / config->luminosity_requested;
        double T_inner_estimated = config->T_inner * pow(luminosity_ratio, -0.5);
        /* TARDIS damping: T_inner_new = T_inner + d * (T_inner_est - T_inner) */
        config->T_inner += config->damping_constant *
            (T_inner_estimated - config->T_inner);
    }
}

/* ============================================================ */
/* Task #072: Plasma solver — tau_sobolev recomputation          */
/* ============================================================ */

/* Task #072: Helper — find ion population index for (Z, ion_stage) */
static int find_ion_pop_idx(AtomicData *atom, int Z, int ion_stage) {
    for (int e = 0; e < atom->n_elements; e++) {
        if (atom->element_Z[e] != Z) continue;
        int offset = atom->elem_ion_offset[e];
        int n_pops = atom->elem_ion_offset[e + 1] - offset;
        if (ion_stage < n_pops)
            return offset + ion_stage;
        return -1;
    }
    return -1;
}

/* Task #072: Helper — find ionization energy for (Z, ion_stage) -> (Z, ion_stage+1) */
static double find_ioniz_energy(AtomicData *atom, int Z, int ion_stage) {
    for (int i = 0; i < atom->n_ionization; i++) {
        if (atom->ioniz_Z[i] == Z && atom->ioniz_ion[i] == ion_stage)
            return atom->ioniz_energy_eV[i];
    }
    return 1e10; /* impossibly high — prevents ionization */
}

/* Task #072: Helper — interpolate zeta factor for (Z, ion_stage) at temperature T */
static double interpolate_zeta(AtomicData *atom, int Z, int ion_stage, double T) {
    /* Find zeta entry for this (Z, ion) */
    int zidx = -1;
    for (int i = 0; i < atom->n_zeta_ions; i++) {
        if (atom->zeta_Z[i] == Z && atom->zeta_ion[i] == ion_stage) {
            zidx = i;
            break;
        }
    }
    if (zidx < 0) return 1.0; /* no zeta data -> LTE (zeta=1) */

    int nt = atom->n_zeta_temps;
    double *temps = atom->zeta_temps;
    double *vals = atom->zeta_data + zidx * nt;

    /* Clamp to grid bounds */
    if (T <= temps[0]) return vals[0];
    if (T >= temps[nt - 1]) return vals[nt - 1];

    /* Linear interpolation */
    for (int i = 0; i < nt - 1; i++) {
        if (T >= temps[i] && T <= temps[i + 1]) {
            double frac = (T - temps[i]) / (temps[i + 1] - temps[i]);
            return vals[i] + frac * (vals[i + 1] - vals[i]);
        }
    }
    return vals[nt - 1];
}

/* Task #072 Step 4a: Compute partition functions
 * TARDIS formula (LevelBoltzmannFactorDiluteLTE):
 *   bf = g * exp(-E / kT_rad)  for ALL levels (both metastable & non-metastable)
 *   bf[non-metastable] *= W
 *   Z = sum(bf) = Z_meta(T_rad) + W * Z_non(T_rad)
 * Note: T_rad is used for BOTH metastable and non-metastable levels.
 * T_e only enters the Saha ionization equation, NOT the partition function. */
static void compute_partition_functions(AtomicData *atom, PlasmaState *plasma,
                                         int n_shells) {
    for (int ip = 0; ip < atom->n_ion_pops; ip++) {
        int lev_start = atom->level_offset[ip];
        int lev_end   = atom->level_offset[ip + 1];

        for (int s = 0; s < n_shells; s++) {
            double T_rad = plasma->T_rad[s];
            double W     = plasma->W[s];

            double Z_meta = 0.0;
            double Z_non_meta = 0.0;

            for (int l = lev_start; l < lev_end; l++) {
                double E_eV = atom->level_energy_eV[l];
                int g = atom->level_g[l];
                int is_meta = atom->level_metastable[l];

                /* ALL levels use T_rad for Boltzmann factor (TARDIS convention) */
                double boltz = (E_eV * EV_TO_ERG) / (K_BOLTZMANN * T_rad);
                if (boltz < 500.0) { /* avoid underflow */
                    double bf = g * exp(-boltz);
                    if (is_meta)
                        Z_meta += bf;
                    else
                        Z_non_meta += bf;
                }
            }

            double Z_total = Z_meta + W * Z_non_meta;
            if (Z_total < 1e-300) Z_total = 1e-300; /* prevent division by zero */
            atom->partition_functions[ip * n_shells + s] = Z_total;
        }
    }
}

/* Task #072: Helper — compute LTE partition function at given T */
static double compute_lte_partition(AtomicData *atom, int ip, double T, int n_shells, int shell) {
    (void)n_shells; (void)shell;
    int lev_start = atom->level_offset[ip];
    int lev_end   = atom->level_offset[ip + 1];
    double Z = 0.0;
    for (int l = lev_start; l < lev_end; l++) {
        double boltz = (atom->level_energy_eV[l] * EV_TO_ERG) / (K_BOLTZMANN * T);
        if (boltz < 500.0)
            Z += atom->level_g[l] * exp(-boltz);
    }
    return (Z > 1e-300) ? Z : 1e-300;
}

/* Task #072 Step 4b: Compute ion number densities (Saha + nebular)
 * Uses TARDIS formula (Mazzali & Lucy 1993, eq. 14):
 *   phi_nebular = phi_lte * W * (zeta*delta + W*(1-zeta)) * sqrt(T_e/T_rad)
 *   ratio = phi_nebular / n_e
 *
 * phi_lte = (Z_{i+1}/Z_i) * 2 * g_electron * exp(-chi * beta_rad)
 * g_electron = (2*pi*m_e*kB*T_rad/h^2)^1.5
 * delta = (T_e/T_rad) * exp(chi * (beta_rad - beta_electron))  for chi >= chi_0
 * beta_rad = 1/(kB*T_rad), beta_electron = 1/(kB*T_e)
 */
static void compute_ion_populations(AtomicData *atom, PlasmaState *plasma,
                                     int n_shells) {
    for (int e = 0; e < atom->n_elements; e++) {
        int Z_elem = atom->element_Z[e];
        double mass_amu = atom->element_mass_amu[e];
        int ip_start = atom->elem_ion_offset[e];
        int ip_end   = atom->elem_ion_offset[e + 1];
        int n_pops   = ip_end - ip_start;

        for (int s = 0; s < n_shells; s++) {
            double T_rad = plasma->T_rad[s];
            double T_e   = plasma->T_e_T_rad_ratio * T_rad;
            double W     = plasma->W[s];
            double n_e   = plasma->n_electron[s];
            double rho   = plasma->rho[s];
            double abund = atom->abundances[e * n_shells + s];

            double n_element = (abund * rho) / (mass_amu * AMU);

            /* g_electron = (2*pi*m_e*kB*T_rad/h^2)^1.5 */
            double g_electron = pow(2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T_rad
                                     / (H_PLANCK * H_PLANCK), 1.5);

            double beta_rad = 1.0 / (K_BOLTZMANN * T_rad);
            double beta_electron = 1.0 / (K_BOLTZMANN * T_e);

            double *ratios = (double *)calloc(n_pops, sizeof(double));

            for (int k = 0; k < n_pops - 1; k++) {
                int ip_cur  = ip_start + k;
                int ip_next = ip_start + k + 1;

                /* LTE partition functions at T_rad for Saha equation */
                double Z_cur  = compute_lte_partition(atom, ip_cur,  T_rad, n_shells, s);
                double Z_next = compute_lte_partition(atom, ip_next, T_rad, n_shells, s);

                double chi_eV  = find_ioniz_energy(atom, Z_elem, k);
                double chi_erg = chi_eV * EV_TO_ERG;

                /* phi_lte = (Z_{i+1}/Z_i) * 2 * g_electron * exp(-chi * beta_rad) */
                double phi_lte = (Z_next / Z_cur) * 2.0 * g_electron *
                                 exp(-chi_erg * beta_rad);

                /* delta: radiation field correction (Mazzali & Lucy 1993) */
                /* delta = (T_e / T_rad) * exp(chi * (beta_rad - beta_electron)) */
                /* This is for chi >= chi_0 (typical case) */
                double delta = (T_e / T_rad) * exp(chi_erg * (beta_rad - beta_electron));

                /* Zeta factor */
                double zeta = interpolate_zeta(atom, Z_elem, k, T_rad);

                /* TARDIS nebular phi:
                 * phi_neb = phi_lte * W * (zeta*delta + W*(1-zeta)) * sqrt(T_e/T_rad) */
                double phi_neb = phi_lte * W *
                    (zeta * delta + W * (1.0 - zeta)) * sqrt(T_e / T_rad);

                /* ratio n_{i+1}/n_i = phi_nebular / n_e */
                double ratio;
                if (n_e > 0.0) {
                    ratio = phi_neb / n_e;
                } else {
                    ratio = 1e10;
                }
                if (ratio > 1e30) ratio = 1e30;
                ratios[k] = ratio;
            }

            /* Normalize: n_0 * (1 + r_0 + r_0*r_1 + ...) = n_element */
            double sum = 1.0;
            double product = 1.0;
            for (int k = 0; k < n_pops - 1; k++) {
                product *= ratios[k];
                if (product > 1e30) { product = 1e30; break; }
                sum += product;
            }

            double n_0 = n_element / sum;
            atom->ion_number_density[ip_start * n_shells + s] = n_0;
            product = 1.0;
            for (int k = 0; k < n_pops - 1; k++) {
                product *= ratios[k];
                double n_ion = n_0 * product;
                if (n_ion < 1e-300) n_ion = 1e-300;
                atom->ion_number_density[(ip_start + k + 1) * n_shells + s] = n_ion;
            }

            free(ratios);
        }
    }
}

/* Task #072 Step 4c: Compute electron density (iterative)
 * Uses the correct TARDIS nebular Saha formula with TARDIS-style damped iteration:
 *   n_e_new_damped = 0.5 * (n_e_computed + n_e_old)
 *   convergence threshold: 5% (TARDIS default)
 *   max iterations: 100 (TARDIS default) */
static void compute_electron_density(AtomicData *atom, PlasmaState *plasma,
                                      int n_shells) {
    for (int s = 0; s < n_shells; s++) {
        double n_e = plasma->n_electron[s];
        if (n_e <= 0.0) n_e = 1e6;

        double T_rad = plasma->T_rad[s];
        double T_e   = plasma->T_e_T_rad_ratio * T_rad;
        double W     = plasma->W[s];
        double rho   = plasma->rho[s];

        double g_electron = pow(2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T_rad
                                 / (H_PLANCK * H_PLANCK), 1.5);
        double beta_rad = 1.0 / (K_BOLTZMANN * T_rad);
        double beta_electron = 1.0 / (K_BOLTZMANN * T_e);

        for (int iteration = 0; iteration < 100; iteration++) {
            double n_e_old = n_e;

            /* Recompute ion populations for all elements in this shell */
            for (int e = 0; e < atom->n_elements; e++) {
                int Z_elem = atom->element_Z[e];
                double mass_amu = atom->element_mass_amu[e];
                int ip_start = atom->elem_ion_offset[e];
                int ip_end   = atom->elem_ion_offset[e + 1];
                int n_pops   = ip_end - ip_start;
                double abund = atom->abundances[e * n_shells + s];
                double n_element = (abund * rho) / (mass_amu * AMU);

                /* Compute ionization ratios using TARDIS nebular formula */
                double product = 1.0;
                double sum_norm = 1.0;
                double ratios_local[64]; /* max ion stages per element */
                int max_k = (n_pops - 1 < 63) ? n_pops - 1 : 63;

                for (int k = 0; k < max_k; k++) {
                    int ip_cur  = ip_start + k;
                    int ip_next = ip_start + k + 1;
                    double Z_cur  = compute_lte_partition(atom, ip_cur,  T_rad, n_shells, s);
                    double Z_next = compute_lte_partition(atom, ip_next, T_rad, n_shells, s);
                    double chi_eV = find_ioniz_energy(atom, Z_elem, k);
                    double chi_erg = chi_eV * EV_TO_ERG;

                    double phi_lte = (Z_next / Z_cur) * 2.0 * g_electron *
                                     exp(-chi_erg * beta_rad);
                    double delta = (T_e / T_rad) * exp(chi_erg * (beta_rad - beta_electron));
                    double zeta = interpolate_zeta(atom, Z_elem, k, T_rad);
                    double phi_neb = phi_lte * W *
                        (zeta * delta + W * (1.0 - zeta)) * sqrt(T_e / T_rad);

                    double ratio = (n_e > 0.0) ? phi_neb / n_e : 1e10;
                    if (ratio > 1e30) ratio = 1e30;
                    ratios_local[k] = ratio;

                    product *= ratio;
                    if (product > 1e30) { product = 1e30; sum_norm += product; break; }
                    sum_norm += product;
                }

                double n_0 = n_element / sum_norm;
                atom->ion_number_density[ip_start * n_shells + s] = n_0;
                product = 1.0;
                for (int k = 0; k < max_k; k++) {
                    product *= ratios_local[k];
                    double n_ion = n_0 * product;
                    if (n_ion < 1e-300) n_ion = 1e-300;
                    atom->ion_number_density[(ip_start + k + 1) * n_shells + s] = n_ion;
                }
            }

            /* Sum electron density: n_e_new = sum(ion_stage * n_ion) */
            double n_e_new = 0.0;
            for (int ip = 0; ip < atom->n_ion_pops; ip++) {
                int charge = atom->ion_pop_stage[ip];
                n_e_new += charge * atom->ion_number_density[ip * n_shells + s];
            }
            if (n_e_new < 1.0) n_e_new = 1.0;

            /* TARDIS-style damped update: n_e = 0.5 * (n_e_new + n_e_old) */
            n_e = 0.5 * (n_e_new + n_e_old);
            plasma->n_electron[s] = n_e;

            /* TARDIS convergence: 5% relative threshold */
            if (n_e_old > 0.0 && fabs(n_e_new - n_e_old) / n_e_old < 0.05) break;
        }
    }
}

/* Task #072 Step 4d: Compute tau_sobolev from ion populations */
static void compute_tau_sobolev(AtomicData *atom, PlasmaState *plasma,
                                 OpacityState *opacity, double time_explosion) {
    int n_lines = opacity->n_lines;
    int n_shells = opacity->n_shells;

    for (int line = 0; line < n_lines; line++) {
        int Z         = atom->line_atomic_number[line];
        int ion_stage = atom->line_ion_number[line];
        int lev_lower = atom->line_level_lower[line];
        int lev_upper = atom->line_level_upper[line];
        double f_lu   = atom->line_f_lu[line];
        double lam_cm = atom->line_wavelength_cm[line];

        /* Find ion population index */
        int ip = find_ion_pop_idx(atom, Z, ion_stage);
        if (ip < 0) {
            for (int s = 0; s < n_shells; s++)
                opacity->tau_sobolev[line * n_shells + s] = 1e-100;
            continue;
        }

        /* Find level data for lower and upper */
        int lev_start = atom->level_offset[ip];
        int lev_end   = atom->level_offset[ip + 1];

        /* Search for lower and upper levels */
        int lower_idx = -1, upper_idx = -1;
        for (int l = lev_start; l < lev_end; l++) {
            if (atom->level_num[l] == lev_lower) lower_idx = l;
            if (atom->level_num[l] == lev_upper) upper_idx = l;
            if (lower_idx >= 0 && upper_idx >= 0) break;
        }

        if (lower_idx < 0 || upper_idx < 0) {
            for (int s = 0; s < n_shells; s++)
                opacity->tau_sobolev[line * n_shells + s] = 1e-100;
            continue;
        }

        double E_lower = atom->level_energy_eV[lower_idx];
        double E_upper = atom->level_energy_eV[upper_idx];
        int g_lower    = atom->level_g[lower_idx];
        int g_upper    = atom->level_g[upper_idx];
        int meta_lower = atom->level_metastable[lower_idx];
        int meta_upper = atom->level_metastable[upper_idx];

        for (int s = 0; s < n_shells; s++) {
            double T_rad = plasma->T_rad[s];
            double W     = plasma->W[s];
            double n_ion = atom->ion_number_density[ip * n_shells + s];
            double Z_part = atom->partition_functions[ip * n_shells + s];

            /* TARDIS level population formula (nebular):
             * Non-metastable: n_k = W * (g_k / Z) * n_ion * exp(-E_k / kT_rad)
             * Metastable:     n_k =     (g_k / Z) * n_ion * exp(-E_k / kT_rad)
             * Note: BOTH use T_rad for Boltzmann factor (not T_e for metastable)
             * T_e only enters the partition function for metastable levels */
            double beta_rad = 1.0 / (K_BOLTZMANN * T_rad);

            /* Lower level population */
            double n_lower;
            {
                double boltz = E_lower * EV_TO_ERG * beta_rad;
                double weight = meta_lower ? 1.0 : W;
                if (boltz < 500.0)
                    n_lower = n_ion * weight * g_lower * exp(-boltz) / Z_part;
                else
                    n_lower = 0.0;
            }

            /* Upper level population */
            double n_upper;
            {
                double boltz = E_upper * EV_TO_ERG * beta_rad;
                double weight = meta_upper ? 1.0 : W;
                if (boltz < 500.0)
                    n_upper = n_ion * weight * g_upper * exp(-boltz) / Z_part;
                else
                    n_upper = 0.0;
            }

            /* Stimulated emission correction */
            double stim_corr = 1.0;
            if (n_lower > 0.0 && n_upper > 0.0) {
                stim_corr = 1.0 - (g_lower * n_upper) / (g_upper * n_lower);
                if (stim_corr < 0.0) stim_corr = 0.0; /* population inversion -> no absorption */
            }

            /* tau_sobolev = SOBOLEV_COEFF * f_lu * lambda_cm * t_exp * n_lower * stim_corr */
            double tau = SOBOLEV_COEFF * f_lu * lam_cm * time_explosion * n_lower * stim_corr;
            if (tau < 1e-100) tau = 1e-100;
            opacity->tau_sobolev[line * n_shells + s] = tau;
        }
    }
}

/* Task #072 Step 4e: Master plasma state update */
void compute_plasma_state(AtomicData *atom, PlasmaState *plasma,
                          OpacityState *opacity, double time_explosion) {
    int n_shells = plasma->n_shells;

    printf("  [Plasma] Computing partition functions...\n");
    compute_partition_functions(atom, plasma, n_shells);

    printf("  [Plasma] Computing electron density (iterative)...\n");
    compute_electron_density(atom, plasma, n_shells);
    printf("    n_e[0]=%.4e, n_e[%d]=%.4e\n",
           plasma->n_electron[0], n_shells - 1, plasma->n_electron[n_shells - 1]);

    printf("  [Plasma] Computing ion populations...\n");
    compute_ion_populations(atom, plasma, n_shells);

    /* Copy self-consistent n_e back to opacity for transport */
    for (int s = 0; s < n_shells; s++)
        opacity->electron_density[s] = plasma->n_electron[s];

    printf("  [Plasma] Computing tau_sobolev...\n");
    compute_tau_sobolev(atom, plasma, opacity, time_explosion);

    /* Print tau stats for key lines */
    int n_lines = opacity->n_lines;
    double tau_min = 1e99, tau_max = 0.0;
    int n_significant = 0;
    for (int l = 0; l < n_lines; l++) {
        double t = opacity->tau_sobolev[l * n_shells + 0]; /* shell 0 */
        if (t > tau_max) tau_max = t;
        if (t < tau_min && t > 1e-100) tau_min = t;
        if (t > 1.0) n_significant++;
    }
    printf("    Shell 0: tau_min=%.2e, tau_max=%.2e, lines with tau>1: %d/%d\n",
           tau_min, tau_max, n_significant, n_lines);
}

/* Spectrum binning: energy is luminosity in erg/s, output L_lambda in erg/s/cm */
void bin_escaped_packet(Spectrum *spec, double nu, double energy) {
    double lambda_A = C_SPEED_OF_LIGHT / nu * 1.0e8; /* frequency → wavelength (Å) */

    if (lambda_A < spec->lambda_min || lambda_A >= spec->lambda_max) {
        return;
    }

    double dlambda_A = (spec->lambda_max - spec->lambda_min) / spec->n_bins;
    int bin = (int)((lambda_A - spec->lambda_min) / dlambda_A);
    if (bin >= 0 && bin < spec->n_bins) {
        /* L_lambda [erg/s/cm] = luminosity [erg/s] / dlambda [cm] */
        double dlambda_cm = dlambda_A * 1.0e-8;
        spec->flux[bin] += energy / dlambda_cm;
    }
}

#ifdef __cplusplus   /* Phase 6 - Step 9: close extern C guard */
}                    /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */
