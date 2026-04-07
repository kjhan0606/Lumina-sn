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
            double T_e   = plasma->T_e[s];
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

                /* Dilute partition functions (W-weighted, consistent with level pops) */
                double Z_cur  = atom->partition_functions[ip_cur  * n_shells + s];
                double Z_next = atom->partition_functions[ip_next * n_shells + s];

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
        double T_e   = plasma->T_e[s];
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
                    /* Dilute partition functions (W-weighted, consistent with level pops) */
                    double Z_cur  = atom->partition_functions[ip_cur  * n_shells + s];
                    double Z_next = atom->partition_functions[ip_next * n_shells + s];
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

/* ============================================================ */
/* P6: Self-consistent per-shell electron temperature           */
/*                                                              */
/* Default mode (self_consistent=0): T_e = ratio × T_rad       */
/* Self-consistent (self_consistent=1): Compton-adiabatic       */
/*   balance with collisional coupling correction + gamma heat  */
/* ============================================================ */
void compute_electron_temperature(PlasmaState *plasma, GammaDeposition *gamma_dep,
                                   double time_explosion, int n_shells,
                                   int self_consistent) {
    if (!self_consistent) {
        /* Default: uniform ratio */
        for (int s = 0; s < n_shells; s++)
            plasma->T_e[s] = plasma->T_e_T_rad_ratio * plasma->T_rad[s];
        return;
    }

    /* Self-consistent T_e from energy balance:
     *
     * Heating:
     *   Compton: q_C = (T_rad - T_e) / t_Compton
     *   Collisional (line/PI thermalization): q_coll ≈ f_coll × (T_rad - T_e) / t_coll
     *   Gamma-ray: q_gamma = Q_gamma / (1.5 × n_e × k_B)
     *
     * Cooling:
     *   Adiabatic: q_ad = 2 × T_e / t_exp  (homologous, γ=5/3)
     *
     * Steady state: q_C + q_coll + q_gamma = q_ad
     *   (Γ_C + Γ_coll)(T_rad - T_e) + G = Γ_ad × T_e
     *   T_e = (Γ_eff × T_rad + G) / (Γ_eff + Γ_ad)
     *
     * Γ_C = 8 σ_T u_rad / (3 m_e c)  [s⁻¹]
     *   u_rad = 4 W σ_SB T_rad⁴ / c
     * Γ_coll ≈ 10 × Γ_C  (collisional coupling >> Compton in photosphere)
     * Γ_ad = 2 / t_exp  [s⁻¹]
     * G = Q_gamma / (1.5 × n_e × k_B)  [K/s]
     */
    double t_exp = time_explosion;
    double Gamma_ad = 2.0 / t_exp;
    /* Collisional boost: line/PI interactions couple T_e to T_rad
     * much more strongly than Compton alone. Factor ~10-20 calibrated
     * to reproduce T_e/T_rad ≈ 0.9 for typical inner shells. */
    double f_coll_boost = 12.0;

    for (int s = 0; s < n_shells; s++) {
        double T_rad = plasma->T_rad[s];
        double W     = plasma->W[s];
        double n_e   = plasma->n_electron[s];
        if (T_rad <= 0.0 || n_e <= 0.0) {
            plasma->T_e[s] = plasma->T_e_T_rad_ratio * T_rad;
            continue;
        }

        /* Compton coupling rate */
        double u_rad = 4.0 * W * SIGMA_SB * T_rad * T_rad * T_rad * T_rad / C_SPEED_OF_LIGHT;
        double Gamma_C = 8.0 * SIGMA_THOMSON * u_rad / (3.0 * M_ELECTRON * C_SPEED_OF_LIGHT);

        /* Effective coupling = Compton + collisional (boosted) */
        double Gamma_eff = Gamma_C * (1.0 + f_coll_boost);

        /* Gamma-ray heating temperature rate */
        double G = 0.0;
        if (gamma_dep != NULL && gamma_dep->heating_rate != NULL && gamma_dep->heating_rate[s] > 0.0)
            G = gamma_dep->heating_rate[s] / (1.5 * n_e * K_BOLTZMANN);

        /* Steady state: T_e = (Γ_eff × T_rad + G) / (Γ_eff + Γ_ad) */
        double T_e = (Gamma_eff * T_rad + G) / (Gamma_eff + Gamma_ad);

        /* Clamp to physical range */
        if (T_e < 0.3 * T_rad) T_e = 0.3 * T_rad;
        if (T_e > 1.5 * T_rad) T_e = 1.5 * T_rad;

        plasma->T_e[s] = T_e;
    }
}

void compute_transition_probabilities(AtomicData *atom, PlasmaState *plasma,
                                       OpacityState *opacity,
                                       NLTEConfig *nlte,
                                       double damping_constant, int apply_damping) {
    int n_shells = opacity->n_shells;
    int n_levels = opacity->n_macro_levels;
    int n_trans  = opacity->n_macro_transitions;

    /* Use J_nu histogram for internal_up if NLTE is active and J_nu populated */
    int use_j_nu = (nlte != NULL && nlte->enabled && nlte->J_nu != NULL);

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
                        /* Internal up: B_lu * J_nu (MC histogram or W*B_nu fallback) */
                        double nu_line = atom->line_nu[line_id];
                        if (use_j_nu) {
                            double J_line = nlte_get_J_at_nu(nlte, s, nu_line);
                            rate = atom->line_B_lu[line_id] * J_line;
                        } else {
                            rate = atom->line_B_lu[line_id] * W * planck_bnu(T_rad, nu_line);
                        }
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
    printf("  [TransProb] Recomputed %d transitions x %d shells (damping=%s, J_src=%s)\n",
           n_trans, n_shells, apply_damping ? "on" : "off",
           use_j_nu ? "MC_histogram" : "W*Bnu");
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

    /* Line overlap correction (enabled by LUMINA_OVERLAP_CORR=1) */
    if (getenv("LUMINA_OVERLAP_CORR") && atoi(getenv("LUMINA_OVERLAP_CORR")) > 0) {
        printf("  [Plasma] Applying line overlap corrections...\n");
        apply_overlap_corrections(atom, opacity, plasma);
    }

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
/* Bound-free (photoionization) opacity                        */
/* Kramers cross-section grid: chi_bf[shell][freq_bin]         */
/* ============================================================ */

void bf_opacity_init(BFOpacity *bf, int n_shells) {
    bf->enabled = 1;
    bf->n_freq_bins = NLTE_N_FREQ_BINS;
    bf->n_shells = n_shells;
    bf->nu_min = NLTE_NU_MIN;
    bf->nu_max = NLTE_NU_MAX;
    bf->d_log_nu = log(NLTE_NU_MAX / NLTE_NU_MIN) / (double)NLTE_N_FREQ_BINS;
    bf->chi_bf = (double *)calloc((size_t)n_shells * NLTE_N_FREQ_BINS, sizeof(double));
    bf->activation_level = (int *)malloc((size_t)n_shells * NLTE_N_FREQ_BINS * sizeof(int));
    memset(bf->activation_level, -1, (size_t)n_shells * NLTE_N_FREQ_BINS * sizeof(int));
}

void bf_opacity_free(BFOpacity *bf) {
    free(bf->chi_bf);
    free(bf->activation_level);
    memset(bf, 0, sizeof(*bf));
}

/* Look up macro-atom activation level for BF absorption at given frequency.
 * Returns macro-atom level index (global level idx) or -1 for thermal fallback. */
int bf_get_activation_level(BFOpacity *bf, int shell, double nu) {
    if (!bf->enabled || !bf->activation_level || nu < bf->nu_min || nu >= bf->nu_max)
        return -1;
    int bin = (int)(log(nu / bf->nu_min) / bf->d_log_nu);
    if (bin < 0 || bin >= bf->n_freq_bins) return -1;
    return bf->activation_level[shell * bf->n_freq_bins + bin];
}

/* Interpolate chi_bf at arbitrary frequency (linear in log-nu grid) */
double bf_get_chi(BFOpacity *bf, int shell, double nu) {
    if (!bf->enabled || nu < bf->nu_min || nu >= bf->nu_max) return 0.0;
    double log_ratio = log(nu / bf->nu_min);
    int bin = (int)(log_ratio / bf->d_log_nu);
    if (bin < 0) return 0.0;
    if (bin >= bf->n_freq_bins - 1) return bf->chi_bf[shell * bf->n_freq_bins + bf->n_freq_bins - 1];
    /* Linear interpolation between bins */
    double frac = log_ratio / bf->d_log_nu - (double)bin;
    double chi0 = bf->chi_bf[shell * bf->n_freq_bins + bin];
    double chi1 = bf->chi_bf[shell * bf->n_freq_bins + bin + 1];
    return chi0 + frac * (chi1 - chi0);
}

/* Compute chi_bf grid for all shells and frequency bins.
 * Uses Kramers hydrogenic cross-section: sigma(nu) = sigma_0 * (nu_edge/nu)^3
 * where sigma_0 = 7.91e-18 / Z_eff^2 cm^2.
 * Sums over all ions and their levels weighted by level population. */
/* P7: Tabulated ground-state photoionization cross-sections from CMFGEN data.
 * Returns σ₀ in cm² (1 Mb = 1e-18 cm²). Ions not in table return 0 → Kramers fallback.
 * Sources: CMFGEN phot_data files (C,Mg,Ca,Cr,Fe,Co,Ni); estimated for Si,S,Ti. */
static double get_bf_sigma0(int Z, int stage) {
    switch (Z) {
    case 6:  return (stage == 1) ? 3.75e-18 : (stage == 2) ? 1.27e-18 : 0;  /* C  */
    case 12: return (stage == 1) ? 0.23e-18 : (stage == 2) ? 5.42e-18 : 0;  /* Mg */
    case 14: return (stage == 1) ? 1.00e-18 : (stage == 2) ? 3.00e-18 : 0;  /* Si (est) */
    case 16: return (stage == 1) ? 2.00e-18 : (stage == 2) ? 3.00e-18 : 0;  /* S  (est) */
    case 20: return (stage == 1) ? 0.31e-18 : (stage == 2) ? 1.92e-18 : 0;  /* Ca */
    case 22: return (stage == 1) ? 3.00e-18 : (stage == 2) ? 2.00e-18 : 0;  /* Ti (est) */
    case 24: return (stage == 1) ? 2.00e-18 : (stage == 2) ? 2.00e-18 : 0;  /* Cr (est) */
    case 26: return (stage == 1) ? 5.26e-18 : (stage == 2) ? 8.82e-18 : 0;  /* Fe */
    case 27: return (stage == 1) ?10.10e-18 : (stage == 2) ? 2.00e-18 : 0;  /* Co */
    case 28: return (stage == 1) ? 7.27e-18 : (stage == 2) ? 3.00e-18 : 0;  /* Ni */
    default: return 0;
    }
}

void compute_bf_opacity(BFOpacity *bf, AtomicData *atom, PlasmaState *plasma,
                         int n_shells) {
    if (!bf->enabled) return;

    /* Zero the grid and activation table */
    size_t grid_size = (size_t)n_shells * bf->n_freq_bins;
    memset(bf->chi_bf, 0, grid_size * sizeof(double));
    memset(bf->activation_level, -1, grid_size * sizeof(int));

    /* Per-bin dominant absorber tracking: chi contribution from best ion */
    double *best_chi = (double *)calloc(grid_size, sizeof(double));
    int    *best_ip  = (int *)malloc(grid_size * sizeof(int));
    memset(best_ip, -1, grid_size * sizeof(int));

    /* Precompute ground-state level index of the NEXT-HIGHER ion for each ion pop.
     * When ion ip (Z, stage) absorbs BF, the atom becomes (Z, stage+1).
     * We activate macro-atom at ground state of (Z, stage+1). */
    int *ionized_ground = (int *)malloc(atom->n_ion_pops * sizeof(int));
    for (int ip = 0; ip < atom->n_ion_pops; ip++) {
        ionized_ground[ip] = -1;
        int Z_ion = atom->ion_pop_Z[ip];
        int next_stage = atom->ion_pop_stage[ip] + 1;
        /* Find ion pop for (Z, next_stage) */
        for (int jp = 0; jp < atom->n_ion_pops; jp++) {
            if (atom->ion_pop_Z[jp] == Z_ion && atom->ion_pop_stage[jp] == next_stage) {
                /* Find ground level (level_num=0) of that ion */
                int ls = atom->level_offset[jp];
                int le = atom->level_offset[jp + 1];
                for (int l = ls; l < le; l++) {
                    if (atom->level_num[l] == 0) {
                        ionized_ground[ip] = l;
                        break;
                    }
                }
                break;
            }
        }
    }

    /* Precompute bin center frequencies */
    double *nu_bin = (double *)malloc(bf->n_freq_bins * sizeof(double));
    for (int b = 0; b < bf->n_freq_bins; b++) {
        nu_bin[b] = bf->nu_min * exp((b + 0.5) * bf->d_log_nu);
    }

    for (int ip = 0; ip < atom->n_ion_pops; ip++) {
        int Z_ion = atom->ion_pop_Z[ip];
        int stage = atom->ion_pop_stage[ip];
        /* Skip neutrals (no ionization from neutral ground to ion) for this simple model,
         * and skip highest ion stages (nothing to ionize to) */
        if (stage < 1) continue;

        /* Find ionization energy for this ion */
        double chi_eV = -1.0;
        for (int k = 0; k < atom->n_ionization; k++) {
            if (atom->ioniz_Z[k] == Z_ion && atom->ioniz_ion[k] == stage) {
                chi_eV = atom->ioniz_energy_eV[k];
                break;
            }
        }
        if (chi_eV <= 0.0) continue;
        double chi_erg = chi_eV * EV_TO_ERG;

        /* P7: Tabulated cross-section (CMFGEN) or Kramers fallback */
        double sigma_0 = get_bf_sigma0(Z_ion, stage);
        if (sigma_0 <= 0.0) {
            int Z_eff = Z_ion - stage;
            if (Z_eff < 1) Z_eff = 1;
            sigma_0 = 7.91e-18 / ((double)Z_eff * (double)Z_eff);
        }

        int lev_start = atom->level_offset[ip];
        int lev_end   = atom->level_offset[ip + 1];

        for (int s = 0; s < n_shells; s++) {
            double T_rad = plasma->T_rad[s];
            double W     = plasma->W[s];
            double n_ion = atom->ion_number_density[ip * n_shells + s];
            double Z_part = atom->partition_functions[ip * n_shells + s];
            double beta_rad = 1.0 / (K_BOLTZMANN * T_rad);

            if (n_ion < 1e-30 || Z_part < 1e-300) continue;

            for (int l = lev_start; l < lev_end; l++) {
                double E_eV = atom->level_energy_eV[l];
                int g = atom->level_g[l];
                int is_meta = atom->level_metastable[l];

                double boltz = E_eV * EV_TO_ERG * beta_rad;
                if (boltz > 50.0) continue;  /* negligible population */

                /* Level population (dilute Boltzmann) */
                double weight = is_meta ? 1.0 : W;
                double n_level = n_ion * weight * g * exp(-boltz) / Z_part;
                if (n_level < 1e-30) continue;

                /* Ionization edge for this level: nu_edge = (chi_ion - E_level) / h */
                double E_level_erg = E_eV * EV_TO_ERG;
                double nu_edge = (chi_erg - E_level_erg) / H_PLANCK;
                if (nu_edge <= bf->nu_min) continue;  /* edge below our grid */

                /* Find starting bin for this edge */
                int bin_start = 0;
                if (nu_edge > bf->nu_min) {
                    bin_start = (int)(log(nu_edge / bf->nu_min) / bf->d_log_nu);
                    if (bin_start < 0) bin_start = 0;
                }

                /* Add contribution to all bins above the edge */
                for (int b = bin_start; b < bf->n_freq_bins; b++) {
                    double nu = nu_bin[b];
                    if (nu < nu_edge) continue;
                    double ratio = nu_edge / nu;
                    double chi_contrib = n_level * sigma_0 * ratio * ratio * ratio;
                    int idx = s * bf->n_freq_bins + b;
                    bf->chi_bf[idx] += chi_contrib;

                    /* Track dominant absorber for macro-atom activation */
                    if (chi_contrib > best_chi[idx]) {
                        best_chi[idx] = chi_contrib;
                        best_ip[idx] = ip;
                    }
                }
            }
        }
    }

    /* Build activation level table from dominant absorber */
    int n_activated = 0;
    for (size_t idx = 0; idx < grid_size; idx++) {
        if (best_ip[idx] >= 0 && ionized_ground[best_ip[idx]] >= 0) {
            bf->activation_level[idx] = ionized_ground[best_ip[idx]];
            n_activated++;
        }
    }

    free(best_chi);
    free(best_ip);
    free(ionized_ground);

    /* --- Free-free (bremsstrahlung) opacity --- */
    for (int s = 0; s < n_shells; s++) {
        double T_e = plasma->T_e[s];
        double n_e = plasma->n_electron[s];
        if (T_e <= 0.0 || n_e <= 0.0) continue;

        double sqrt_Te_inv = 1.0 / sqrt(T_e);
        double kT_e = K_BOLTZMANN * T_e;

        /* Sum Z_eff^2 * n_ion over all ions */
        double Z2_n_sum = 0.0;
        for (int ip = 0; ip < atom->n_ion_pops; ip++) {
            int ion_stage = atom->ion_pop_stage[ip];  /* 0=neutral, 1=II, 2=III */
            if (ion_stage < 1) continue;              /* neutrals don't contribute */
            double Z_eff = (double)ion_stage;
            double n_ion = atom->ion_number_density[ip * n_shells + s];
            Z2_n_sum += Z_eff * Z_eff * n_ion;
        }

        double coeff = C_FF_OPACITY * sqrt_Te_inv * n_e * Z2_n_sum;

        for (int b = 0; b < bf->n_freq_bins; b++) {
            double nu = nu_bin[b];
            double nu3 = nu * nu * nu;
            double stim = 1.0 - exp(-H_PLANCK * nu / kT_e);
            bf->chi_bf[s * bf->n_freq_bins + b] += coeff / nu3 * stim;
        }
    }

    free(nu_bin);

    /* Print diagnostics: BF and FF contributions separately for shell 0 */
    double chi_bf_max_opt = 0.0, chi_bf_max_uv = 0.0;
    double chi_ff_max_opt = 0.0, chi_ff_max_uv = 0.0;
    {
        /* Recompute FF-only for shell 0 for diagnostics */
        double T_e0 = plasma->T_e[0];
        double n_e0 = plasma->n_electron[0];
        double sqrt_Te_inv0 = (T_e0 > 0.0) ? 1.0 / sqrt(T_e0) : 0.0;
        double kT_e0 = K_BOLTZMANN * T_e0;
        double Z2_n_sum0 = 0.0;
        for (int ip = 0; ip < atom->n_ion_pops; ip++) {
            int ion_stage = atom->ion_pop_stage[ip];
            if (ion_stage < 1) continue;
            double Z_eff = (double)ion_stage;
            double n_ion = atom->ion_number_density[ip * n_shells + 0];
            Z2_n_sum0 += Z_eff * Z_eff * n_ion;
        }
        double coeff0 = C_FF_OPACITY * sqrt_Te_inv0 * n_e0 * Z2_n_sum0;

        for (int b = 0; b < bf->n_freq_bins; b++) {
            double nu = bf->nu_min * exp((b + 0.5) * bf->d_log_nu);
            double lam_A = C_SPEED_OF_LIGHT / nu * 1e8;
            double chi_total = bf->chi_bf[0 * bf->n_freq_bins + b];

            /* FF contribution at this freq */
            double nu3 = nu * nu * nu;
            double stim = (kT_e0 > 0.0) ? 1.0 - exp(-H_PLANCK * nu / kT_e0) : 0.0;
            double chi_ff = (coeff0 > 0.0) ? coeff0 / nu3 * stim : 0.0;
            double chi_bf = chi_total - chi_ff;

            if (lam_A >= 3500.0 && lam_A <= 9000.0) {
                if (chi_bf > chi_bf_max_opt) chi_bf_max_opt = chi_bf;
                if (chi_ff > chi_ff_max_opt) chi_ff_max_opt = chi_ff;
            }
            if (lam_A >= 1000.0 && lam_A < 3500.0) {
                if (chi_bf > chi_bf_max_uv) chi_bf_max_uv = chi_bf;
                if (chi_ff > chi_ff_max_uv) chi_ff_max_uv = chi_ff;
            }
        }
    }
    double chi_e0 = plasma->n_electron[0] * SIGMA_THOMSON;
    printf("  [BF+FF] Shell 0 (optical): chi_bf=%.2e  chi_ff=%.2e  chi_e=%.2e  (bf/e=%.2e  ff/e=%.2e)\n",
           chi_bf_max_opt, chi_ff_max_opt, chi_e0, chi_bf_max_opt/chi_e0, chi_ff_max_opt/chi_e0);
    printf("  [BF+FF] Shell 0 (UV):      chi_bf=%.2e  chi_ff=%.2e  chi_e=%.2e  (bf/e=%.2e  ff/e=%.2e)\n",
           chi_bf_max_uv, chi_ff_max_uv, chi_e0, chi_bf_max_uv/chi_e0, chi_ff_max_uv/chi_e0);
    printf("  [BF] Macro-atom activation: %d/%d bins have valid levels\n",
           n_activated, (int)grid_size);
}

/* Sample Planck frequency using Bjorkman-Wood method (4-random) */
double sample_planck_frequency(double T, RNG *rng) {
    double kT_h = K_BOLTZMANN * T / H_PLANCK;
    double xi0 = rng_uniform(rng);
    double l_coef = M_PI_VAL * M_PI_VAL * M_PI_VAL * M_PI_VAL / 90.0;
    double target = xi0 * l_coef;
    double cumsum = 0.0;
    double l_min = 1.0;
    for (int l = 1; l <= 1000; l++) {
        double ld = (double)l;
        double l_inv4 = 1.0 / (ld * ld * ld * ld);
        cumsum += l_inv4;
        if (cumsum >= target) {
            l_min = ld;
            break;
        }
    }
    double r1 = rng_uniform(rng);
    double r2 = rng_uniform(rng);
    double r3 = rng_uniform(rng);
    double r4 = rng_uniform(rng);
    if (r1 < 1e-300) r1 = 1e-300;
    if (r2 < 1e-300) r2 = 1e-300;
    if (r3 < 1e-300) r3 = 1e-300;
    if (r4 < 1e-300) r4 = 1e-300;
    double x = -log(r1 * r2 * r3 * r4) / l_min;
    return x * kT_h;
}

/* BF absorption event: thermalize packet — re-emit as Planck(T_rad) */
void bf_absorption_event(RPacket *pkt, double time_explosion,
                          PlasmaState *plasma, OpacityState *opacity,
                          RNG *rng) {
    /* 1. New isotropic direction */
    pkt->mu = rng_mu(rng);

    /* 2. Sample new comoving frequency from Planck(T_rad) */
    double T_rad = plasma->T_rad[pkt->current_shell_id];
    double comov_nu = sample_planck_frequency(T_rad, rng);

    /* 3. Transform to lab frame (inline Doppler to avoid lumina_transport.c dependency) */
    double beta = pkt->r / (C_SPEED_OF_LIGHT * time_explosion);
    double doppler = 1.0 - beta * pkt->mu;
    pkt->nu = comov_nu / doppler;  /* inv_doppler = 1/doppler */

    /* 4. Reinitialize next_line_id for new frequency (binary search) */
    double comov_check = pkt->nu * doppler;
    int lo = 0, hi = opacity->n_lines;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (opacity->line_list_nu[mid] > comov_check)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo == opacity->n_lines) lo = opacity->n_lines - 1;
    pkt->next_line_id = lo;
}

/* ============================================================ */
/* NLTE: Full NLTE Rate Equation Solver                         */
/* Targets: Si,Ca,Fe,S,Co,Ni,C,Mg,Ti,Cr II/III (10 pairs)     */
/* ============================================================ */

/* NLTE target ion definitions: 10 element pairs (20 ions) */
static const int NLTE_TARGET_Z[]   = { 14, 14, 20, 20, 26, 26, 16, 16, 27, 27, 28, 28,
                                         6,  6, 12, 12, 22, 22, 24, 24 };
static const int NLTE_TARGET_ION[] = {  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,
                                         1,  2,  1,  2,  1,  2,  1,  2 };

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
  {  26,   1,    24,   2,   1.0e-9,    0.0,  -1.47 },  /* Fe+ + Cr2+ → Fe2+ + Cr+ */
  {  26,   1,    22,   2,   1.0e-9,    0.0,  -0.77 },  /* Fe+ + Ti2+ → Fe2+ + Ti+ */
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
double nlte_get_J_at_nu(NLTEConfig *nlte, int shell, double nu) {
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
                                double *A_cm, double *b, int N,
                                GammaDeposition *gamma_dep) {
    (void)time_explosion;

    int lev_start = nlte->nlte_ion_level_offset[ion_idx_lo];
    int n_shells = plasma->n_shells;
    double T_rad = plasma->T_rad[shell];
    double T_e   = plasma->T_e[shell];
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

    /* ---- Non-thermal gamma-ray ionization ---- */
    if (gamma_dep != NULL && gamma_dep->nonthermal_ioniz_rate[shell] > 0.0) {
        double R_nt_total = gamma_dep->nonthermal_ioniz_rate[shell]; /* ionizations/s/cm³ */

        /* Compute total atom number density in shell from all elements */
        double n_total_atoms = 0.0;
        for (int e = 0; e < atom->n_elements; e++) {
            double X_e = atom->abundances[e * n_shells + shell];
            double A_e = atom->element_mass_amu[e];
            n_total_atoms += X_e * plasma->rho[shell] / (A_e * AMU);
        }

        /* Per-particle ionization rate (distributed equally per atom) */
        if (n_total_atoms > 0.0 && ground_hi < N) {
            double R_nt_per_particle = R_nt_total / n_total_atoms; /* [s⁻¹] */

            /* Apply: ground state of lower ion → ground state of upper ion */
            ACM(ground_hi, 0) += R_nt_per_particle;
            ACM(0, 0)         -= R_nt_per_particle;
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
                                  int shell, double time_explosion,
                                  GammaDeposition *gamma_dep) {
    int lev_start = nlte->nlte_ion_level_offset[ion_idx_lo];
    int N = nlte->nlte_ion_level_offset[ion_idx_hi + 1] - lev_start;
    if (N <= 0) return;
    int n_shells = plasma->n_shells;

    double *A_cm = (double *)calloc((size_t)N * N, sizeof(double));
    double *b = (double *)calloc((size_t)N, sizeof(double));

    nlte_assemble_rate_matrix(nlte, atom, plasma, opacity,
                               ion_idx_lo, ion_idx_hi, shell, time_explosion,
                               A_cm, b, N, gamma_dep);

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
                     int n_shells, GammaDeposition *gamma_dep) {
    printf("  [NLTE] Solving rate equations (with CE coupling)...\n");

    int n_pairs = nlte->n_nlte_ions / 2;
    int pairs[][2] = { {0,1}, {2,3}, {4,5}, {6,7}, {8,9}, {10,11},
                       {12,13}, {14,15}, {16,17}, {18,19} };
    const char *names[] = { "Si", "Ca", "Fe", "S", "Co", "Ni",
                            "C", "Mg", "Ti", "Cr" };

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

        /* Solve all ion pairs */
        for (int p = 0; p < n_pairs; p++) {
            int lo = pairs[p][0], hi = pairs[p][1];
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1)
            #endif
            for (int s = 0; s < n_shells; s++) {
                nlte_solve_ion_shell(nlte, atom, plasma, opacity,
                                     lo, hi, s, time_explosion, gamma_dep);
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

/* ============================================================ */
/* Gamma-ray energy deposition from 56Ni/56Co decay             */
/* ============================================================ */

/* Physical constants for 56Ni/56Co decay */
#define LAMBDA_NI56   1.318e-6    /* 56Ni decay constant [s⁻¹], t½=6.077d */
#define LAMBDA_CO56   1.038e-7    /* 56Co decay constant [s⁻¹], t½=77.27d */
#define Q_NI56        2.803e-6    /* 56Ni decay energy [erg/decay] (1.75 MeV) */
#define Q_CO56        5.976e-6    /* 56Co decay energy [erg/decay] (3.73 MeV) */
#define KAPPA_GAMMA   0.025       /* Gray gamma-ray opacity [cm²/g] (Swartz+ 1995) */
#define ETA_NONTHERMAL 0.05       /* Fraction of deposition → ionization (Kozma & Fransson 1992) */
#define W_ION_EV      35.0        /* Mean energy per ion pair [eV] */

void gamma_deposition_init(GammaDeposition *gd, int n_shells) {
    gd->n_shells = n_shells;
    gd->heating_rate = (double *)calloc(n_shells, sizeof(double));
    gd->nonthermal_ioniz_rate = (double *)calloc(n_shells, sizeof(double));
}

void gamma_deposition_free(GammaDeposition *gd) {
    free(gd->heating_rate);
    free(gd->nonthermal_ioniz_rate);
    gd->heating_rate = NULL;
    gd->nonthermal_ioniz_rate = NULL;
}

void compute_gamma_deposition(GammaDeposition *gd, AtomicData *atom,
                               PlasmaState *plasma, Geometry *geo) {
    int n_shells = gd->n_shells;
    double t_exp = geo->time_explosion;

    /* Find element indices for Ni(Z=28) and Co(Z=27) */
    int elem_ni = -1, elem_co = -1;
    for (int e = 0; e < atom->n_elements; e++) {
        if (atom->element_Z[e] == 28) elem_ni = e;
        if (atom->element_Z[e] == 27) elem_co = e;
    }

    /* Bateman equation for 56Ni → 56Co → 56Fe:
     * N_Ni(t) = N_Ni(0) × exp(-λ_Ni × t)
     * N_Co(t) = N_Ni(0) × [λ_Ni/(λ_Co-λ_Ni)] × [exp(-λ_Ni×t) - exp(-λ_Co×t)]
     *           + N_Co(0) × exp(-λ_Co × t)
     * Note: At t=0, all Ni is 56Ni; Co abundance is initial 56Co (if any). */
    double exp_ni = exp(-LAMBDA_NI56 * t_exp);
    double exp_co = exp(-LAMBDA_CO56 * t_exp);
    double bateman_factor = LAMBDA_NI56 / (LAMBDA_CO56 - LAMBDA_NI56);

    /* Compute per-shell energy generation and outward column density */
    double *epsilon_gamma = (double *)calloc(n_shells, sizeof(double)); /* erg/s/cm³ */
    double *column_density = (double *)calloc(n_shells, sizeof(double)); /* g/cm² */

    for (int s = 0; s < n_shells; s++) {
        double rho = plasma->rho[s];

        /* Number density of 56Ni and 56Co from mass fractions.
         * We use the current Ni/Co abundances as initial mass fractions,
         * then apply time evolution. Ni mass = 56 amu. */
        double X_ni = (elem_ni >= 0) ? atom->abundances[elem_ni * n_shells + s] : 0.0;
        double X_co = (elem_co >= 0) ? atom->abundances[elem_co * n_shells + s] : 0.0;

        /* Initial number densities at t=0 */
        double n_ni_0 = X_ni * rho / (56.0 * AMU); /* cm⁻³ */
        double n_co_0 = X_co * rho / (56.0 * AMU);

        /* Current number densities from decay */
        double n_ni = n_ni_0 * exp_ni;
        double n_co = n_ni_0 * bateman_factor * (exp_ni - exp_co) + n_co_0 * exp_co;
        if (n_co < 0.0) n_co = 0.0;

        /* Local gamma-ray energy generation rate [erg/s/cm³] */
        epsilon_gamma[s] = LAMBDA_NI56 * n_ni * Q_NI56 + LAMBDA_CO56 * n_co * Q_CO56;
    }

    /* Outward column density: Σ(s) = Σ_{s'=s}^{N-1} ρ(s') × Δr(s') */
    column_density[n_shells - 1] = plasma->rho[n_shells - 1] *
        (geo->r_outer[n_shells - 1] - geo->r_inner[n_shells - 1]);
    for (int s = n_shells - 2; s >= 0; s--) {
        double dr = geo->r_outer[s] - geo->r_inner[s];
        column_density[s] = column_density[s + 1] + plasma->rho[s] * dr;
    }

    /* Deposition fraction and rates */
    for (int s = 0; s < n_shells; s++) {
        double tau_gamma = KAPPA_GAMMA * column_density[s];
        double f_dep = 1.0 - exp(-tau_gamma);

        gd->heating_rate[s] = epsilon_gamma[s] * f_dep;
        gd->nonthermal_ioniz_rate[s] = ETA_NONTHERMAL * gd->heating_rate[s]
                                        / (W_ION_EV * EV_TO_ERG);
    }

    free(epsilon_gamma);
    free(column_density);
}

/* ============================================================ */
/* Sobolev line overlap correction                               */
/* ============================================================ */

void apply_overlap_corrections(AtomicData *atom, OpacityState *opacity,
                                PlasmaState *plasma) {
    int n_lines = opacity->n_lines;
    int n_shells = opacity->n_shells;

    /* Work on a copy of the original tau values */
    size_t tau_size = (size_t)n_lines * n_shells;
    double *tau_orig = (double *)malloc(tau_size * sizeof(double));
    memcpy(tau_orig, opacity->tau_sobolev, tau_size * sizeof(double));

    for (int s = 0; s < n_shells; s++) {
        double T_rad = plasma->T_rad[s];

        for (int i = 0; i < n_lines; i++) {
            double tau_i = tau_orig[i * n_shells + s];
            if (tau_i < 1e-10) continue; /* skip negligible lines */

            double nu_i = opacity->line_list_nu[i];
            int Z_i = atom->line_atomic_number[i];
            double mass_amu = 2.0 * Z_i; /* rough: A ≈ 2Z */
            double v_th = sqrt(2.0 * K_BOLTZMANN * T_rad / (mass_amu * AMU));
            double delta_nu_th = nu_i * v_th / C_SPEED_OF_LIGHT;

            if (delta_nu_th <= 0.0) continue;

            /* Scan forward neighbors (lower frequency, j > i in descending array) */
            double tau_overlap = 0.0;
            for (int j = i + 1; j < n_lines && j <= i + 10; j++) {
                double dnu = nu_i - opacity->line_list_nu[j];
                if (dnu > 3.0 * delta_nu_th) break;
                double tau_j = tau_orig[j * n_shells + s];
                tau_overlap += tau_j * exp(-(dnu / delta_nu_th) * (dnu / delta_nu_th));
            }

            /* Scan backward neighbors (higher frequency) */
            for (int j = i - 1; j >= 0 && j >= i - 10; j--) {
                double dnu = opacity->line_list_nu[j] - nu_i;
                if (dnu > 3.0 * delta_nu_th) break;
                double tau_j = tau_orig[j * n_shells + s];
                tau_overlap += tau_j * exp(-(dnu / delta_nu_th) * (dnu / delta_nu_th));
            }

            /* Apply correction: tau_eff = tau_i² / (tau_i + tau_overlap) */
            if (tau_overlap > 0.01 * tau_i) {
                double correction = tau_i / (tau_i + tau_overlap);
                opacity->tau_sobolev[i * n_shells + s] = tau_i * correction;
            }
        }
    }

    free(tau_orig);
}

/* Rescale geometry and density for a new epoch (homologous expansion).
   v = r/t is invariant; r(t_new) = v * t_new, rho ~ t^-3. */
void rescale_epoch(Geometry *geo, PlasmaState *plasma, double t_new) {
    double t_ref = geo->time_explosion;
    double ratio = t_new / t_ref;
    double rho_scale = 1.0 / (ratio * ratio * ratio);
    for (int i = 0; i < geo->n_shells; i++) {
        geo->r_inner[i] = geo->v_inner[i] * t_new;
        geo->r_outer[i] = geo->v_outer[i] * t_new;
        plasma->rho[i] *= rho_scale;
    }
    geo->time_explosion = t_new;
}

/* ============================================================ */
/* P5: Formal integral spectrum (noise-free, p-z formalism)     */
/*                                                              */
/* For each observed frequency nu_obs, integrate along rays     */
/* with impact parameters p from 0 to r_outer:                 */
/*   I_nu(p) = I_core * exp(-tau_tot) + sum_lines S_l*(1-e^-tau_l)*e^-tau_above */
/*   L_nu = 4*pi * integral( I_nu(p) * 2*pi*p dp )            */
/* ============================================================ */

/* Binary search in descending-sorted array: find first index with val <= target */
static int bsearch_descending_le(const double *arr, int n, double target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] > target) lo = mid + 1;
        else hi = mid;
    }
    return lo; /* first index where arr[lo] <= target */
}

/* Binary search in descending-sorted array: find last index with val >= target */
static int bsearch_descending_ge(const double *arr, int n, double target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] >= target) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1; /* last index where arr[lo-1] >= target */
}

void compute_formal_integral_spectrum(
    Geometry *geo, PlasmaState *plasma, OpacityState *opacity,
    AtomicData *atom, NLTEConfig *nlte, double T_inner,
    Spectrum *spec_formal, int n_impact)
{
    int n_shells = geo->n_shells;
    int n_lines  = opacity->n_lines;
    double t_exp = geo->time_explosion;
    double ct    = C_SPEED_OF_LIGHT * t_exp;
    double r_phot  = geo->r_inner[0];
    double r_outer = geo->r_outer[n_shells - 1];
    double beta_max = r_outer / (C_SPEED_OF_LIGHT * t_exp); /* v_outer / c */
    double dp = r_outer / n_impact;

    printf("\n=== Formal Integral Spectrum ===\n");
    printf("  Impact parameters: %d, beta_max=%.4f\n", n_impact, beta_max);

    /* Zero output spectrum */
    for (int b = 0; b < spec_formal->n_bins; b++)
        spec_formal->flux[b] = 0.0;

    /* For each wavelength bin (parallelized) */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 10)
    #endif
    for (int bin = 0; bin < spec_formal->n_bins; bin++) {
        double lambda_cm = spec_formal->wavelength[bin] * 1.0e-8;
        double nu_obs = C_SPEED_OF_LIGHT / lambda_cm;

        /* Frequency range of lines that could resonate within the ejecta */
        double nu_line_min = nu_obs * (1.0 - beta_max);
        double nu_line_max = nu_obs * (1.0 + beta_max);

        /* Binary search in line_list_nu (sorted DESCENDING by frequency) */
        int l_first = bsearch_descending_le(opacity->line_list_nu, n_lines, nu_line_max);
        int l_last  = bsearch_descending_ge(opacity->line_list_nu, n_lines, nu_line_min);

        double L_nu_integral = 0.0;

        for (int ip = 0; ip < n_impact; ip++) {
            double p = dp * (ip + 0.5);
            double p2 = p * p;
            if (p >= r_outer) continue;

            double z_max_ejecta = sqrt(r_outer * r_outer - p2);
            double z_phot = (p < r_phot) ? sqrt(r_phot * r_phot - p2) : 0.0;

            double I_nu = 0.0;
            double tau_acc = 0.0;

            /* Walk from observer side inward (z decreasing).
             * Lines sorted nu descending → as index increases, nu decreases, z increases.
             * So iterate from l_last (lowest nu, largest z) backward to l_first. */
            for (int l = l_last; l >= l_first; l--) {
                double nu_l = opacity->line_list_nu[l];
                double z = ct * (1.0 - nu_l / nu_obs);

                /* Check z within valid ejecta range for this impact parameter */
                if (z > z_max_ejecta || z < -z_max_ejecta) continue;
                if (p < r_phot && z < z_phot) continue; /* behind photosphere */

                double r = sqrt(p2 + z * z);
                if (r < r_phot || r > r_outer) continue;

                /* Find shell */
                int shell = -1;
                for (int s = 0; s < n_shells; s++) {
                    if (r >= geo->r_inner[s] && r < geo->r_outer[s]) {
                        shell = s;
                        break;
                    }
                }
                if (shell < 0) continue;

                double tau_sob = opacity->tau_sobolev[l * n_shells + shell];
                if (tau_sob < 1e-5) continue; /* skip negligible lines */

                /* Source function: J_nu if NLTE available, else W * B_nu(T_rad) */
                double S;
                if (nlte != NULL && nlte->enabled) {
                    S = nlte_get_J_at_nu(nlte, shell, nu_l);
                    if (S <= 0.0)
                        S = plasma->W[shell] * planck_bnu(plasma->T_rad[shell], nu_l);
                } else {
                    S = plasma->W[shell] * planck_bnu(plasma->T_rad[shell], nu_l);
                }

                /* Line contribution: S * (1 - exp(-tau)) * exp(-tau_accumulated) */
                double one_minus_exp = (tau_sob > 500.0) ? 1.0 : (1.0 - exp(-tau_sob));
                I_nu += S * one_minus_exp * exp(-tau_acc);
                tau_acc += tau_sob;
            }

            /* Inner boundary: Planck at T_inner */
            if (p < r_phot) {
                I_nu += planck_bnu(T_inner, nu_obs) * exp(-tau_acc);
            }

            /* Integrate: L_nu += I_nu * 2*pi*p * dp */
            L_nu_integral += I_nu * p * dp;
        }

        /* L_nu = 4*pi * integral(I_nu * 2*pi*p dp) = 8*pi^2 * sum */
        double L_nu = 8.0 * M_PI_VAL * M_PI_VAL * L_nu_integral;

        /* Convert L_nu [erg/s/Hz] to L_lambda [erg/s/cm]: L_lambda = L_nu * c / lambda^2 */
        spec_formal->flux[bin] = L_nu * C_SPEED_OF_LIGHT / (lambda_cm * lambda_cm);
    }

    printf("  Formal integral spectrum computed.\n");
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
