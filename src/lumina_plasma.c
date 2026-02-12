/* lumina_plasma.c — Phase 4: Plasma Solver and Convergence
 * Implements TARDIS mc_rad_field_solver.py for T_rad, W updates.
 * Implements T_inner convergence from escape fraction. */

#include "lumina.h" /* Phase 4 - Step 1 */
#ifdef _OPENMP
#include <omp.h>
#endif

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

/* ============================================================ */
/* Dynamic macro-atom transition probability recomputation      */
/* ============================================================ */

static inline double beta_sobolev(double tau) {
    if (tau < 1e-6) return 1.0 - 0.5 * tau;   /* Taylor expansion */
    if (tau > 500.0) return 1.0 / tau;          /* asymptotic */
    return (1.0 - exp(-tau)) / tau;
}

static inline double planck_bnu(double T, double nu) {
    double x = H_PLANCK * nu / (K_BOLTZMANN * T);
    if (x > 500.0) return 0.0;
    return (2.0 * H_PLANCK * nu * nu * nu / (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT))
           / (exp(x) - 1.0);
}

void compute_transition_probabilities(AtomicData *atom, PlasmaState *plasma,
                                       OpacityState *opacity,
                                       double damping_constant, int apply_damping) {
    int n_shells = opacity->n_shells;
    int n_levels = opacity->n_macro_levels;
    int n_trans  = opacity->n_macro_transitions;

    /* Find max block size for temp buffer */
    int max_block = 0;
    for (int lev = 0; lev < n_levels; lev++) {
        int bs = opacity->macro_block_references[lev + 1] -
                 opacity->macro_block_references[lev];
        if (bs > max_block) max_block = bs;
    }
    double *rates_buf = (double *)malloc(max_block * sizeof(double));

    for (int s = 0; s < n_shells; s++) {
        double W     = plasma->W[s];
        double T_rad = plasma->T_rad[s];

        for (int lev = 0; lev < n_levels; lev++) {
            int block_start = opacity->macro_block_references[lev];
            int block_end   = opacity->macro_block_references[lev + 1];
            if (block_start >= block_end) continue;

            /* Phase 1: Compute raw rates into temp buffer */
            double sum_rates = 0.0;

            for (int tid = block_start; tid < block_end; tid++) {
                int ttype   = opacity->transition_type[tid];
                int line_id = opacity->transition_line_id[tid];
                double rate = 0.0;

                if (line_id >= 0 && line_id < atom->n_lines) {
                    double tau = opacity->tau_sobolev[line_id * n_shells + s];
                    double beta = beta_sobolev(tau);

                    if (ttype == -1) {
                        /* BB emission: A_ul * beta_sobolev */
                        rate = atom->line_A_ul[line_id] * beta;
                    } else if (ttype == 0) {
                        /* Internal down: A_ul * (1 - beta_sobolev) */
                        rate = atom->line_A_ul[line_id] * (1.0 - beta);
                    } else if (ttype == 1) {
                        /* Internal up: B_lu * W * B_nu(T_rad, nu_line) */
                        double nu_line = atom->line_nu[line_id];
                        rate = atom->line_B_lu[line_id] * W * planck_bnu(T_rad, nu_line);
                    }
                }
                if (rate < 0.0) rate = 0.0;
                rates_buf[tid - block_start] = rate;
                sum_rates += rate;
            }

            /* Phase 2: Normalize and apply (with optional damping) */
            if (sum_rates > 0.0) {
                for (int tid = block_start; tid < block_end; tid++) {
                    double p_new = rates_buf[tid - block_start] / sum_rates;
                    if (apply_damping) {
                        double p_old = opacity->transition_probabilities[tid * n_shells + s];
                        p_new = p_old + damping_constant * (p_new - p_old);
                    }
                    opacity->transition_probabilities[tid * n_shells + s] = p_new;
                }
            }
            /* If sum_rates == 0: keep existing probabilities (degenerate level) */
        }
    }

    free(rates_buf);
    printf("  [TransProb] Recomputed %d transitions x %d shells (damping=%s)\n",
           n_trans, n_shells, apply_damping ? "on" : "off");
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

/* ============================================================ */
/* NLTE: Full NLTE Rate Equation Solver                         */
/* Targets: Si,Ca,Fe,S,Co,Ni II/III (6 pairs, ~3500 levels)    */
/* ============================================================ */

/* NLTE target ion definitions: 6 element pairs (12 ions) */
static const int NLTE_TARGET_Z[]   = { 14, 14, 20, 20, 26, 26, 16, 16, 27, 27, 28, 28 };
static const int NLTE_TARGET_ION[] = {  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,  1,  2 };

/* Step 1.5: Charge Exchange reaction table
 * Forward: A^(ion_A) + B^(ion_B) → A^(ion_A+1) + B^(ion_B-1)
 * Reverse via detailed balance: k_rev = k_fwd * exp(|ΔE|/kT)
 * Rate coefficients from Kingdon & Ferland (1996), generic 1e-9 cm³/s */
static const ChargeExchangeReaction CE_REACTIONS[CE_N_REACTIONS] = {
  /* Z_A ion_A  Z_B ion_B  rate       alpha  ΔE_eV  */
  {  26,   1,    27,   2,   1.0e-9,    0.0,  -0.89 },  /* Fe+ + Co2+ → Fe2+ + Co+ */
  {  26,   1,    28,   2,   1.0e-9,    0.0,  -1.98 },  /* Fe+ + Ni2+ → Fe2+ + Ni+ */
  {  27,   1,    28,   2,   1.0e-9,    0.0,  -1.09 },  /* Co+ + Ni2+ → Co2+ + Ni+ */
  {  20,   1,    14,   2,   1.0e-9,    0.0,  -4.48 },  /* Ca+ + Si2+ → Ca2+ + Si+ */
};

/* Step 1.5: Get total ion number density for (Z, ion_stage, shell).
 * Uses NLTE level populations if available, otherwise nebular density. */
static double nlte_get_ion_density(NLTEConfig *nlte, AtomicData *atom,
                                    int Z, int ion_stage, int shell,
                                    int n_shells) {
    /* Check if this (Z, ion_stage) is an NLTE ion → sum level populations */
    if (nlte != NULL) {
        for (int i = 0; i < nlte->n_nlte_ions; i++) {
            if (nlte->nlte_Z[i] == Z && nlte->nlte_ion[i] == ion_stage) {
                int lev_s = nlte->nlte_ion_level_offset[i];
                int lev_e = nlte->nlte_ion_level_offset[i + 1];
                double sum = 0.0;
                for (int l = lev_s; l < lev_e; l++)
                    sum += nlte->nlte_level_populations[l * n_shells + shell];
                if (sum > 0.0) return sum;
                break;  /* found ion but no populations yet, fall through */
            }
        }
    }
    /* Fall back to nebular ion_number_density */
    int ip = find_ion_pop_idx(atom, Z, ion_stage);
    if (ip >= 0)
        return atom->ion_number_density[ip * n_shells + shell];
    return 0.0;
}

/* van Regemorter collision rate constant:
 * C_ij = 14.5 * a_0^2 * sqrt(2*pi*k_B/(m_e)) * n_e * f_ij / sqrt(T_e) * exp(-dE/kT)
 * Numerically: coeff = 14.5 * (5.29e-9)^2 * sqrt(2*pi*1.38e-16/9.11e-28)
 *            = 14.5 * 2.8e-17 * sqrt(9.53e11) = 14.5 * 2.8e-17 * 9.76e5 = 3.96e-10
 * We use the standard form: C_12 = 2.16e-6 * n_e * f_lu * exp(-dE/kT) / (g_1*sqrt(T_e)) * g_bar
 * where g_bar ~ 0.2 for allowed (van Regemorter), 1.0 for forbidden (Axelrod)
 */
#define VAN_REGEMORTER_COEFF  2.16e-6  /* effective Gaunt factor included */
#define AXELROD_OMEGA         1.0      /* collision strength for forbidden trans */

int nlte_init(NLTEConfig *nlte, AtomicData *atom, OpacityState *opacity,
              int n_shells) {
    memset(nlte, 0, sizeof(NLTEConfig));
    nlte->enabled = 1;
    nlte->n_freq_bins = NLTE_N_FREQ_BINS;
    nlte->nu_min = NLTE_NU_MIN;
    nlte->nu_max = NLTE_NU_MAX;
    nlte->d_log_nu = log(NLTE_NU_MAX / NLTE_NU_MIN) / NLTE_N_FREQ_BINS;

    /* Set up target ions */
    nlte->n_nlte_ions = NLTE_MAX_IONS;
    for (int i = 0; i < NLTE_MAX_IONS; i++) {
        nlte->nlte_Z[i]   = NLTE_TARGET_Z[i];
        nlte->nlte_ion[i]  = NLTE_TARGET_ION[i];
    }

    /* Build level index maps */
    /* First pass: count levels per NLTE ion */
    nlte->nlte_ion_level_offset[0] = 0;
    for (int i = 0; i < NLTE_MAX_IONS; i++) {
        int Z = nlte->nlte_Z[i];
        int ion = nlte->nlte_ion[i];
        int count = 0;
        for (int l = 0; l < atom->n_levels; l++) {
            if (atom->level_Z[l] == Z && atom->level_ion[l] == ion)
                count++;
        }
        nlte->nlte_ion_level_offset[i + 1] = nlte->nlte_ion_level_offset[i] + count;
    }
    nlte->n_nlte_levels_total = nlte->nlte_ion_level_offset[NLTE_MAX_IONS];
    printf("  [NLTE] Total NLTE levels: %d\n", nlte->n_nlte_levels_total);
    for (int i = 0; i < NLTE_MAX_IONS; i++) {
        int n = nlte->nlte_ion_level_offset[i + 1] - nlte->nlte_ion_level_offset[i];
        printf("    Z=%d ion=%d: %d levels\n", nlte->nlte_Z[i], nlte->nlte_ion[i], n);
    }

    /* Second pass: build bidirectional level maps */
    nlte->nlte_to_global_level = (int *)malloc(nlte->n_nlte_levels_total * sizeof(int));
    nlte->global_to_nlte_level = (int *)malloc(atom->n_levels * sizeof(int));
    for (int l = 0; l < atom->n_levels; l++)
        nlte->global_to_nlte_level[l] = -1;

    int *cursor = (int *)calloc(NLTE_MAX_IONS, sizeof(int)); /* per-ion insertion cursor */
    for (int l = 0; l < atom->n_levels; l++) {
        for (int i = 0; i < NLTE_MAX_IONS; i++) {
            if (atom->level_Z[l] == nlte->nlte_Z[i] &&
                atom->level_ion[l] == nlte->nlte_ion[i]) {
                int nlte_idx = nlte->nlte_ion_level_offset[i] + cursor[i];
                nlte->nlte_to_global_level[nlte_idx] = l;
                nlte->global_to_nlte_level[l] = nlte_idx;
                cursor[i]++;
                break;
            }
        }
    }
    free(cursor);

    /* Build line -> NLTE ion map */
    int n_lines = opacity->n_lines;
    nlte->nlte_line_map = (int *)malloc(n_lines * sizeof(int));
    int n_nlte_lines = 0;
    for (int line = 0; line < n_lines; line++) {
        nlte->nlte_line_map[line] = -1;
        int Z   = atom->line_atomic_number[line];
        int ion = atom->line_ion_number[line];
        for (int i = 0; i < NLTE_MAX_IONS; i++) {
            if (Z == nlte->nlte_Z[i] && ion == nlte->nlte_ion[i]) {
                nlte->nlte_line_map[line] = i;
                n_nlte_lines++;
                break;
            }
        }
    }
    printf("  [NLTE] Lines mapped to NLTE ions: %d / %d\n", n_nlte_lines, n_lines);

    /* Allocate results arrays */
    nlte->nlte_level_populations = (double *)calloc(
        (size_t)nlte->n_nlte_levels_total * n_shells, sizeof(double));
    nlte->j_nu_estimator = (double *)calloc(
        (size_t)n_shells * NLTE_N_FREQ_BINS, sizeof(double));
    nlte->J_nu = (double *)calloc(
        (size_t)n_shells * NLTE_N_FREQ_BINS, sizeof(double));

    printf("  [NLTE] Initialization complete. Memory: %.1f MB\n",
           ((double)nlte->n_nlte_levels_total * n_shells * 8 +
            (double)n_shells * NLTE_N_FREQ_BINS * 16) / 1048576.0);
    return 0;
}

void nlte_free(NLTEConfig *nlte) {
    free(nlte->nlte_to_global_level);
    free(nlte->global_to_nlte_level);
    free(nlte->nlte_line_map);
    free(nlte->nlte_level_populations);
    free(nlte->j_nu_estimator);
    free(nlte->J_nu);
}

/* Normalize raw j_nu estimator to physical J_nu [erg/s/cm^2/Hz/sr] */
void nlte_normalize_j_nu(NLTEConfig *nlte, double time_simulation,
                          double *volume, int n_shells) {
    for (int s = 0; s < n_shells; s++) {
        for (int b = 0; b < nlte->n_freq_bins; b++) {
            int idx = s * nlte->n_freq_bins + b;
            double raw = nlte->j_nu_estimator[idx];

            /* Compute bin width in Hz */
            double log_nu_lo = log(nlte->nu_min) + b * nlte->d_log_nu;
            double log_nu_hi = log_nu_lo + nlte->d_log_nu;
            double delta_nu = exp(log_nu_hi) - exp(log_nu_lo);

            /* J_nu = j_raw / (4*pi * V * t_sim * delta_nu) */
            if (raw > 0.0 && volume[s] > 0.0 && delta_nu > 0.0) {
                nlte->J_nu[idx] = raw /
                    (4.0 * M_PI_VAL * volume[s] * time_simulation * delta_nu);
            } else {
                nlte->J_nu[idx] = 1e-30; /* floor */
            }
        }
    }
}

/* Interpolate J_nu at a given frequency from the histogram */
static double nlte_get_J_at_nu(NLTEConfig *nlte, int shell, double nu) {
    if (nu <= nlte->nu_min || nu >= nlte->nu_max)
        return 1e-30;
    double log_ratio = log(nu / nlte->nu_min);
    int bin = (int)(log_ratio / nlte->d_log_nu);
    if (bin < 0) bin = 0;
    if (bin >= nlte->n_freq_bins) bin = nlte->n_freq_bins - 1;
    return nlte->J_nu[shell * nlte->n_freq_bins + bin];
}

/* Column-oriented Gaussian elimination with partial pivoting for Ax=b.
 * A is N x N column-major matrix, b is N x 1 RHS vector.
 * Inner loop iterates rows within a column = stride-1 = cache-friendly.
 * Solution returned in b. Returns 0 on success, -1 on singular matrix. */
static int gauss_solve(double *A, double *b, int N) {
    /* Column-major: A(i,j) = A[j*N + i] */
    for (int k = 0; k < N; k++) {
        /* Partial pivoting: find max in column k, rows k..N-1 (contiguous) */
        int max_row = k;
        double max_val = fabs(A[k * N + k]);
        for (int i = k + 1; i < N; i++) {
            double v = fabs(A[k * N + i]);
            if (v > max_val) { max_val = v; max_row = i; }
        }
        if (max_val < 1e-300) return -1;

        /* Swap rows k and max_row across all columns + b */
        if (max_row != k) {
            for (int j = 0; j < N; j++) {
                double tmp = A[j * N + k];
                A[j * N + k] = A[j * N + max_row];
                A[j * N + max_row] = tmp;
            }
            double tmp = b[k]; b[k] = b[max_row]; b[max_row] = tmp;
        }

        /* Compute multipliers in column k (contiguous write) */
        double pivot_inv = 1.0 / A[k * N + k];
        for (int i = k + 1; i < N; i++)
            A[k * N + i] *= pivot_inv;

        /* Update trailing submatrix column-by-column (inner loop contiguous!) */
        for (int j = k + 1; j < N; j++) {
            double A_kj = A[j * N + k]; /* pivot row element in column j */
            for (int i = k + 1; i < N; i++)
                A[j * N + i] -= A[k * N + i] * A_kj;
        }

        /* Update RHS using multipliers */
        double b_k = b[k];
        for (int i = k + 1; i < N; i++)
            b[i] -= A[k * N + i] * b_k;

        /* Zero multipliers (restore matrix for back-substitution) */
        for (int i = k + 1; i < N; i++)
            A[k * N + i] = 0.0;
    }

    /* Back substitution */
    for (int k = N - 1; k >= 0; k--) {
        double sum = b[k];
        for (int j = k + 1; j < N; j++)
            sum -= A[j * N + k] * b[j];
        b[k] = sum / A[k * N + k];
    }
    return 0;
}

/* Assemble NLTE rate matrix for one ion pair in one shell.
 * Outputs column-major A_cm[N*N] and RHS b[N] (both must be pre-zeroed).
 * Called by both CPU (gauss_solve) and GPU (cuBLAS batched) paths. */
void nlte_assemble_rate_matrix(NLTEConfig *nlte, AtomicData *atom,
                                PlasmaState *plasma, OpacityState *opacity,
                                int ion_idx_lo, int ion_idx_hi,
                                int shell, double time_explosion,
                                double *A_cm, double *b, int N) {
    (void)time_explosion;

    int lev_start = nlte->nlte_ion_level_offset[ion_idx_lo];
    int n_shells = plasma->n_shells;
    double T_rad = plasma->T_rad[shell];
    double T_e   = plasma->T_e_T_rad_ratio * T_rad;
    double n_e   = plasma->n_electron[shell];

    /* Column-major access: ACM(i,j) = A_cm[j*N + i] */
    #define ACM(i,j) A_cm[(j) * N + (i)]

    /* ---- Radiative bound-bound rates from line data ---- */
    int n_lines = opacity->n_lines;
    for (int line = 0; line < n_lines; line++) {
        int map = nlte->nlte_line_map[line];
        if (map < ion_idx_lo || map > ion_idx_hi) continue;

        int ion_s = atom->line_ion_number[line];
        int ip = find_ion_pop_idx(atom, atom->line_atomic_number[line], ion_s);
        if (ip < 0) continue;
        int lev_base = atom->level_offset[ip];
        int lev_top  = atom->level_offset[ip + 1];

        int lower_global = -1, upper_global = -1;
        for (int l = lev_base; l < lev_top; l++) {
            if (atom->level_num[l] == atom->line_level_lower[line]) lower_global = l;
            if (atom->level_num[l] == atom->line_level_upper[line]) upper_global = l;
            if (lower_global >= 0 && upper_global >= 0) break;
        }
        if (lower_global < 0 || upper_global < 0) continue;

        int i_lo = nlte->global_to_nlte_level[lower_global] - lev_start;
        int i_up = nlte->global_to_nlte_level[upper_global] - lev_start;
        if (i_lo < 0 || i_lo >= N || i_up < 0 || i_up >= N) continue;

        double nu_line = atom->line_nu[line];
        double J_line = nlte_get_J_at_nu(nlte, shell, nu_line);

        double R_absorb = atom->line_B_lu[line] * J_line;
        double R_stim   = atom->line_B_ul[line] * J_line;
        double R_spont  = atom->line_A_ul[line];

        double dE = fabs(atom->level_energy_eV[upper_global] -
                         atom->level_energy_eV[lower_global]) * EV_TO_ERG;
        int g_lo = atom->level_g[lower_global];
        int g_up = atom->level_g[upper_global];
        double f_lu = atom->line_f_lu[line];

        double C_up = 0.0;
        if (T_e > 0.0 && dE > 0.0) {
            double exp_factor = exp(-dE / (K_BOLTZMANN * T_e));
            if (f_lu > 1e-10) {
                C_up = VAN_REGEMORTER_COEFF * n_e * f_lu *
                       exp_factor / (g_lo * sqrt(T_e)) * 0.2;
            } else {
                C_up = 8.63e-6 * n_e * AXELROD_OMEGA *
                       exp_factor / (g_lo * sqrt(T_e));
            }
        }
        double C_down = (g_lo > 0 && g_up > 0 && T_e > 0.0) ?
            C_up * ((double)g_lo / (double)g_up) *
            exp(dE / (K_BOLTZMANN * T_e)) : 0.0;

        double total_up   = R_absorb + C_up;
        double total_down = R_stim + R_spont + C_down;

        ACM(i_up, i_lo) += total_up;
        ACM(i_lo, i_up) += total_down;
        ACM(i_lo, i_lo) -= total_up;
        ACM(i_up, i_up) -= total_down;
    }

    /* ---- Photoionization / Recombination ---- */
    int Z_elem = nlte->nlte_Z[ion_idx_lo];
    double chi_eV = find_ioniz_energy(atom, Z_elem, nlte->nlte_ion[ion_idx_lo]);
    double chi_erg = chi_eV * EV_TO_ERG;
    double nu_edge = chi_erg / H_PLANCK;

    int n_lo_levels = nlte->nlte_ion_level_offset[ion_idx_lo + 1] -
                      nlte->nlte_ion_level_offset[ion_idx_lo];
    int ground_hi = n_lo_levels;

    if (ground_hi < N && nu_edge > 0.0 && nu_edge < nlte->nu_max) {
        double Z_eff = (double)(Z_elem - nlte->nlte_ion[ion_idx_lo]);
        if (Z_eff < 1.0) Z_eff = 1.0;
        double sigma_0 = 7.91e-18 / (Z_eff * Z_eff);

        for (int lev = 0; lev < n_lo_levels; lev++) {
            int global_lev = nlte->nlte_to_global_level[lev_start + lev];
            double E_lev = atom->level_energy_eV[global_lev] * EV_TO_ERG;
            double nu_thresh = (chi_erg - E_lev) / H_PLANCK;
            if (nu_thresh <= 0.0) continue;
            double sigma_lev = sigma_0;

            double R_bf = 0.0;
            for (int bb = 0; bb < nlte->n_freq_bins; bb++) {
                double log_nu_lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
                double nu_bin = exp(log_nu_lo + 0.5 * nlte->d_log_nu);
                if (nu_bin < nu_thresh) continue;
                double delta_nu = exp(log_nu_lo + nlte->d_log_nu) - exp(log_nu_lo);
                double J_bin = nlte->J_nu[shell * nlte->n_freq_bins + bb];
                double sigma = sigma_lev * pow(nu_thresh / nu_bin, 3.0);
                R_bf += 4.0 * M_PI_VAL * J_bin * sigma / (H_PLANCK * nu_bin) * delta_nu;
            }

            double n_star_ratio = 1.0;
            if (T_e > 0.0 && n_e > 0.0) {
                int g_lev = atom->level_g[global_lev];
                double thermal_deBroglie = pow(H_PLANCK * H_PLANCK /
                    (2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T_e), 1.5);
                double chi_lev_erg = chi_erg - E_lev;
                if (chi_lev_erg > 0.0) {
                    double exp_factor = exp(chi_lev_erg / (K_BOLTZMANN * T_e));
                    int g_ion = 1;
                    if (ground_hi < N) {
                        int global_ghi = nlte->nlte_to_global_level[lev_start + ground_hi];
                        g_ion = atom->level_g[global_ghi];
                        if (g_ion < 1) g_ion = 1;
                    }
                    n_star_ratio = n_e * thermal_deBroglie *
                        (double)g_lev / (2.0 * (double)g_ion) * exp_factor;
                    if (n_star_ratio > 1e30) n_star_ratio = 1e30;
                }
            }
            double R_rec = R_bf * n_star_ratio;

            if (R_bf > 0.0 && lev < N && ground_hi < N) {
                ACM(ground_hi, lev) += R_bf;
                ACM(lev, lev)       -= R_bf;
                ACM(lev, ground_hi) += R_rec;
                ACM(ground_hi, ground_hi) -= R_rec;
            }
        }
    }

    /* ---- Step 1.5: Charge Exchange rates ---- */
    int Z_pair = nlte->nlte_Z[ion_idx_lo]; /* element Z for this ion pair */
    int ion_lo_stage = nlte->nlte_ion[ion_idx_lo];   /* e.g. 1 for II */
    int ion_hi_stage = nlte->nlte_ion[ion_idx_hi];   /* e.g. 2 for III */

    for (int r = 0; r < CE_N_REACTIONS; r++) {
        const ChargeExchangeReaction *ce = &CE_REACTIONS[r];
        double k_fwd = ce->rate_coeff * pow(T_e / 1e4, ce->alpha);
        double k_rev = k_fwd * exp(fabs(ce->delta_E_eV) * EV_TO_ERG /
                                    (K_BOLTZMANN * T_e));

        /* Case 1: This pair is element A (forward: A^ion_A → A^(ion_A+1))
         * Requires: Z_pair == Z_A, ion_A == lower ion, ion_A+1 == upper ion */
        if (ce->Z_A == Z_pair && ce->ion_A == ion_lo_stage &&
            ce->ion_A + 1 == ion_hi_stage) {
            double n_partner = nlte_get_ion_density(nlte, atom,
                ce->Z_B, ce->ion_B, shell, n_shells);
            double n_partner_lower = nlte_get_ion_density(nlte, atom,
                ce->Z_B, ce->ion_B - 1, shell, n_shells);

            double R_fwd = k_fwd * n_partner;    /* [s⁻¹] per A^ion_A ion */
            double R_rev = k_rev * n_partner_lower; /* [s⁻¹] per A^(ion_A+1) ion */

            /* Forward: all A^ion_A levels → A^(ion_A+1) ground */
            for (int lev = 0; lev < n_lo_levels; lev++) {
                ACM(ground_hi, lev) += R_fwd;
                ACM(lev, lev)       -= R_fwd;
            }
            /* Reverse: A^(ion_A+1) ground → A^ion_A ground */
            ACM(0, ground_hi)          += R_rev;
            ACM(ground_hi, ground_hi)  -= R_rev;
        }

        /* Case 2: This pair is element B (forward: B^ion_B → B^(ion_B-1))
         * Requires: Z_pair == Z_B, ion_B == upper ion, ion_B-1 == lower ion */
        if (ce->Z_B == Z_pair && ce->ion_B == ion_hi_stage &&
            ce->ion_B - 1 == ion_lo_stage) {
            double n_partner = nlte_get_ion_density(nlte, atom,
                ce->Z_A, ce->ion_A, shell, n_shells);
            double n_partner_upper = nlte_get_ion_density(nlte, atom,
                ce->Z_A, ce->ion_A + 1, shell, n_shells);

            double R_fwd = k_fwd * n_partner;        /* B^ion_B → B^(ion_B-1) */
            double R_rev = k_rev * n_partner_upper;   /* B^(ion_B-1) → B^ion_B */

            /* Forward: B^(ion_B) ground → B^(ion_B-1) ground
             * (ion_B is the upper ion in this pair, ground_hi is its first level) */
            ACM(0, ground_hi)          += R_fwd;
            ACM(ground_hi, ground_hi)  -= R_fwd;
            /* Reverse: all B^(ion_B-1) levels → B^ion_B ground */
            for (int lev = 0; lev < n_lo_levels; lev++) {
                ACM(ground_hi, lev) += R_rev;
                ACM(lev, lev)       -= R_rev;
            }
        }
    }

    /* ---- Conservation equation: replace last row ---- */
    int Z_nl = nlte->nlte_Z[ion_idx_lo];
    double n_total = 0.0;
    for (int i = ion_idx_lo; i <= ion_idx_hi; i++) {
        int ip = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[i]);
        if (ip >= 0)
            n_total += atom->ion_number_density[ip * n_shells + shell];
    }
    for (int j = 0; j < N; j++)
        ACM(N - 1, j) = 1.0;
    b[N - 1] = n_total;

    #undef ACM
}

/* CPU NLTE solver: assemble + Gauss elimination for one ion pair in one shell */
static void nlte_solve_ion_shell(NLTEConfig *nlte, AtomicData *atom,
                                  PlasmaState *plasma, OpacityState *opacity,
                                  int ion_idx_lo, int ion_idx_hi,
                                  int shell, double time_explosion) {
    int lev_start = nlte->nlte_ion_level_offset[ion_idx_lo];
    int N = nlte->nlte_ion_level_offset[ion_idx_hi + 1] - lev_start;
    if (N <= 0) return;
    int n_shells = plasma->n_shells;

    double *A_cm = (double *)calloc((size_t)N * N, sizeof(double));
    double *b = (double *)calloc((size_t)N, sizeof(double));

    nlte_assemble_rate_matrix(nlte, atom, plasma, opacity,
                               ion_idx_lo, ion_idx_hi, shell, time_explosion,
                               A_cm, b, N);

    int ret = gauss_solve(A_cm, b, N);

    if (ret == 0) {
        for (int i = 0; i < N; i++) {
            double pop = b[i];
            if (pop < 0.0) pop = 1e-30;
            nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] = pop;
        }
    } else {
        /* Singular matrix: fall back to Boltzmann at T_rad */
        double T_rad = plasma->T_rad[shell];
        double n_total = b[N - 1]; /* conservation RHS was stored here before solve failed */
        /* Recompute n_total since b may be corrupted */
        int Z_nl = nlte->nlte_Z[ion_idx_lo];
        n_total = 0.0;
        for (int i = ion_idx_lo; i <= ion_idx_hi; i++) {
            int ip = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[i]);
            if (ip >= 0)
                n_total += atom->ion_number_density[ip * n_shells + shell];
        }
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            int global = nlte->nlte_to_global_level[lev_start + i];
            double E = atom->level_energy_eV[global] * EV_TO_ERG;
            int g = atom->level_g[global];
            double pop = (double)g * exp(-E / (K_BOLTZMANN * T_rad));
            nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] = pop;
            sum += pop;
        }
        if (sum > 0.0) {
            double scale = n_total / sum;
            for (int i = 0; i < N; i++)
                nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] *= scale;
        }
    }

    free(A_cm);
    free(b);
}

/* Update tau_sobolev for NLTE lines using NLTE level populations */
static void nlte_update_tau_sobolev(NLTEConfig *nlte, AtomicData *atom,
                                     OpacityState *opacity,
                                     double time_explosion, int n_shells) {
    int n_lines = opacity->n_lines;

    for (int line = 0; line < n_lines; line++) {
        int ion_idx = nlte->nlte_line_map[line];
        if (ion_idx < 0) continue; /* not an NLTE line */

        int Z     = atom->line_atomic_number[line];
        int ion_s = atom->line_ion_number[line];
        double f_lu   = atom->line_f_lu[line];
        double lam_cm = atom->line_wavelength_cm[line];

        /* Find the NLTE level indices for lower and upper */
        int ip = find_ion_pop_idx(atom, Z, ion_s);
        if (ip < 0) continue;
        int lev_base = atom->level_offset[ip];
        int lev_top  = atom->level_offset[ip + 1];

        int lower_global = -1, upper_global = -1;
        for (int l = lev_base; l < lev_top; l++) {
            if (atom->level_num[l] == atom->line_level_lower[line]) lower_global = l;
            if (atom->level_num[l] == atom->line_level_upper[line]) upper_global = l;
            if (lower_global >= 0 && upper_global >= 0) break;
        }
        if (lower_global < 0 || upper_global < 0) continue;

        int nlte_lo = nlte->global_to_nlte_level[lower_global];
        int nlte_up = nlte->global_to_nlte_level[upper_global];
        if (nlte_lo < 0 || nlte_up < 0) continue;

        int g_lo = atom->level_g[lower_global];
        int g_up = atom->level_g[upper_global];

        for (int s = 0; s < n_shells; s++) {
            double n_lower = nlte->nlte_level_populations[nlte_lo * n_shells + s];
            double n_upper = nlte->nlte_level_populations[nlte_up * n_shells + s];

            /* Stimulated emission correction */
            double stim_corr = 1.0;
            if (n_lower > 0.0 && n_upper > 0.0 && g_lo > 0 && g_up > 0) {
                stim_corr = 1.0 - ((double)g_lo * n_upper) / ((double)g_up * n_lower);
                if (stim_corr < 0.0) stim_corr = 0.0;
            }

            double tau = SOBOLEV_COEFF * f_lu * lam_cm * time_explosion *
                         n_lower * stim_corr;
            if (tau < 1e-100) tau = 1e-100;
            opacity->tau_sobolev[line * n_shells + s] = tau;
        }
    }
}

/* Master NLTE solver: solve all ions in all shells, update tau.
 * Step 1.5: Iterative CE convergence wrapper — because CE couples
 * different elements, we iterate until ion densities converge. */
void nlte_solve_all(NLTEConfig *nlte, AtomicData *atom, PlasmaState *plasma,
                     OpacityState *opacity, double time_explosion,
                     int n_shells) {
    printf("  [NLTE] Solving rate equations (with CE coupling)...\n");

    int n_pairs = nlte->n_nlte_ions / 2;
    int pairs[][2] = { {0,1}, {2,3}, {4,5}, {6,7}, {8,9}, {10,11} };
    const char *names[] = { "Si", "Ca", "Fe", "S", "Co", "Ni" };

    int ce_max_iter = 5;
    double ce_threshold = 1e-2;  /* 1% relative convergence on ion totals */
    double ce_damping = 0.5;     /* 50% damping */

    /* Save old ion totals for convergence check (n_nlte_ions * n_shells) */
    int n_ion_totals = nlte->n_nlte_ions * n_shells;
    double *old_ion_totals = (double *)calloc(n_ion_totals, sizeof(double));
    size_t pop_size = (size_t)nlte->n_nlte_levels_total * n_shells;
    double *old_pops = (double *)malloc(pop_size * sizeof(double));

    for (int ce_iter = 0; ce_iter < ce_max_iter; ce_iter++) {
        /* Save current populations + compute old ion totals */
        memcpy(old_pops, nlte->nlte_level_populations, pop_size * sizeof(double));
        for (int ii = 0; ii < nlte->n_nlte_ions; ii++) {
            int lev_s = nlte->nlte_ion_level_offset[ii];
            int lev_e = nlte->nlte_ion_level_offset[ii + 1];
            for (int s = 0; s < n_shells; s++) {
                double sum = 0.0;
                for (int l = lev_s; l < lev_e; l++)
                    sum += nlte->nlte_level_populations[l * n_shells + s];
                old_ion_totals[ii * n_shells + s] = sum;
            }
        }

        /* Solve all 6 ion pairs */
        for (int p = 0; p < n_pairs; p++) {
            int lo = pairs[p][0], hi = pairs[p][1];
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1)
            #endif
            for (int s = 0; s < n_shells; s++) {
                nlte_solve_ion_shell(nlte, atom, plasma, opacity,
                                     lo, hi, s, time_explosion);
            }
        }

        /* Apply damping for iter >= 1 */
        if (ce_iter > 0) {
            for (size_t i = 0; i < pop_size; i++) {
                double n_new = nlte->nlte_level_populations[i];
                double n_old = old_pops[i];
                nlte->nlte_level_populations[i] = n_old +
                    ce_damping * (n_new - n_old);
            }
        }

        /* Convergence: max relative change of ion totals */
        double max_rel_change = 0.0;
        if (ce_iter == 0) {
            /* Check if any old ion totals were nonzero */
            int has_prior = 0;
            for (int k = 0; k < n_ion_totals; k++) {
                if (old_ion_totals[k] > 1.0) { has_prior = 1; break; }
            }
            if (!has_prior) {
                printf("    CE iter %d: first solve (no prior populations)\n",
                       ce_iter + 1);
                continue;
            }
        }

        for (int ii = 0; ii < nlte->n_nlte_ions; ii++) {
            int lev_s = nlte->nlte_ion_level_offset[ii];
            int lev_e = nlte->nlte_ion_level_offset[ii + 1];
            for (int s = 0; s < n_shells; s++) {
                double new_total = 0.0;
                for (int l = lev_s; l < lev_e; l++)
                    new_total += nlte->nlte_level_populations[l * n_shells + s];
                double old_total = old_ion_totals[ii * n_shells + s];
                if (old_total > 1.0) {
                    double rel = fabs(new_total - old_total) / old_total;
                    if (rel > max_rel_change) max_rel_change = rel;
                }
            }
        }

        printf("    CE iter %d: max_ion_rel_change = %.2e\n",
               ce_iter + 1, max_rel_change);

        if (max_rel_change < ce_threshold) {
            printf("    CE converged in %d iterations\n", ce_iter + 1);
            break;
        }
    }
    free(old_pops);
    free(old_ion_totals);

    /* Print ion pair level counts */
    for (int p = 0; p < n_pairs; p++) {
        int lo = pairs[p][0], hi = pairs[p][1];
        int n_levels = nlte->nlte_ion_level_offset[hi + 1] -
                       nlte->nlte_ion_level_offset[lo];
        printf("    %s (%d levels): done\n", names[p], n_levels);
    }

    /* Update tau_sobolev for NLTE lines */
    printf("  [NLTE] Updating tau_sobolev from NLTE populations...\n");
    nlte_update_tau_sobolev(nlte, atom, opacity, time_explosion, n_shells);

    /* Print diagnostics: compare total NLTE vs nebular ion densities */
    for (int p = 0; p < n_pairs; p++) {
        int lo = pairs[p][0];
        int lev_s = nlte->nlte_ion_level_offset[lo];
        int lev_e = nlte->nlte_ion_level_offset[lo + 1];
        double sum_nlte = 0.0;
        for (int l = lev_s; l < lev_e; l++)
            sum_nlte += nlte->nlte_level_populations[l * n_shells + 0];
        int ip = find_ion_pop_idx(atom, nlte->nlte_Z[lo], nlte->nlte_ion[lo]);
        double n_neb = (ip >= 0) ? atom->ion_number_density[ip * n_shells + 0] : 0.0;
        printf("    %s II shell 0: NLTE n_total=%.3e, nebular n_ion=%.3e\n",
               names[p], sum_nlte, n_neb);
    }
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
