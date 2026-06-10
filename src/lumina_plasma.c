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

static inline double planck_bnu(double T, double nu);
/* Binned-J estimator: fit dilute Planck (T_rad,W) to the frequency-resolved
 * J_nu histogram instead of the nu_bar/j Wien moments. Returns 1 on success
 * (writes *T_out,*W_out), 0 if the histogram is unavailable/empty. */
static int fit_dilute_planck_binned_j(const Estimators *est, int shell,
                                      double volume, double time_simulation,
                                      double *T_out, double *W_out);

/* ============================================================ */
/* Phase 4 - Step 2: Radiation field solver                     */
/* (mc_rad_field_solver.py: estimate_dilute_planck_radiation_field) */
/* ============================================================ */

void solve_radiation_field(Estimators *est, double time_explosion,
                            double time_simulation, double *volume,
                            OpacityState *opacity, PlasmaState *plasma,
                            double damping_constant) {
    int n_shells = est->n_shells; /* Phase 4 - Step 2 */

    /* Physical dilution-factor ceiling. The Lucy (W,T_rad) estimator can rail to
     * unphysical W>1 when the field is strongly line-blanketed (mean-nu redshift
     * pins T_rad low, so W = piJ/(sigma T_rad^4) inflates). W is a dilution
     * factor and must be <=1. Gated by LUMINA_W_CAP (0/unset = no cap, baseline;
     * a positive value caps W_est there, typically 1.0). Clamping the estimate
     * keeps the damped value bounded too (convex combo of two <=cap values). */
    double w_cap = 0.0;
    { const char *e = getenv("LUMINA_W_CAP"); if (e) w_cap = atof(e); }

    /* Binned-J estimator: derive (T_rad,W) from a dilute-Planck FIT to the
     * frequency-resolved J_nu histogram rather than the nu_bar/j moments. The
     * first frequency moment collapses under a redshifted fluorescence cascade
     * (mean-nu pulled to the red tail -> T_rad rails to ~2000-3000K, W explodes);
     * a shape fit is robust because W absorbs amplitude while T is set by the
     * SED slope. Gated by LUMINA_BINNED_J_ESTIMATOR (0/unset = moment baseline). */
    static int binned_j_init = 0, binned_j_on = 0;
    if (!binned_j_init) {
        const char *e = getenv("LUMINA_BINNED_J_ESTIMATOR");
        binned_j_on = (e && atoi(e) != 0);
        if (binned_j_on)
            printf("  [binned-J] radiation-field estimator: dilute-Planck fit to J_nu histogram\n");
        binned_j_init = 1;
    }

    /* Fixed thermal-structure mode: when LUMINA_FIXED_TRAD_PROFILE points to a
     * "shell T_rad W" file, OVERRIDE plasma T_rad/W with that profile and skip the
     * estimator update entirely. This decouples the opacity temperature from the
     * radiation estimator, which under a redshifting fluorescence cascade forms a
     * positive-feedback loop (T_rad collapse -> Boltzmann/Saha over-populate low
     * levels -> red lines thicken -> field reddens -> T_rad collapses further).
     * Feeding the self-consistent SCATTER-converged structure isolates the line
     * redistribution from that runaway, testing whether fluorescence cures the
     * blue deficit / line morphology once the thermal structure is held correct. */
    static int    fixed_init = 0, fixed_on = 0, fixed_n = 0;
    static double *fixed_T = NULL, *fixed_W = NULL;
    if (!fixed_init) {
        fixed_init = 1;
        const char *fp = getenv("LUMINA_FIXED_TRAD_PROFILE");
        if (fp && *fp) {
            FILE *f = fopen(fp, "r");
            if (f) {
                fixed_T = (double *)calloc(n_shells, sizeof(double));
                fixed_W = (double *)calloc(n_shells, sizeof(double));
                char ln[256];
                while (fgets(ln, sizeof(ln), f)) {
                    if (ln[0] == '#') continue;
                    int s; double T, W;
                    if (sscanf(ln, "%d %lf %lf", &s, &T, &W) == 3 &&
                        s >= 0 && s < n_shells) {
                        fixed_T[s] = T; fixed_W[s] = W; fixed_n++;
                    }
                }
                fclose(f);
                fixed_on = (fixed_n == n_shells);
                printf("  [fixed-Trad] %s: loaded %d/%d shells -> %s\n", fp,
                       fixed_n, n_shells, fixed_on ? "ACTIVE (estimator frozen)"
                                                   : "INCOMPLETE, ignored");
            } else {
                printf("  [fixed-Trad] could not open %s, ignored\n", fp);
            }
        }
    }
    if (fixed_on) {
        for (int i = 0; i < n_shells; i++) {
            plasma->T_rad[i] = fixed_T[i];
            plasma->W[i]     = fixed_W[i];
        }
        return;
    }

    for (int i = 0; i < n_shells; i++) { /* Phase 4 - Step 2 */
        /* Phase 4 - Step 2: T_rad from nubar/j ratio */
        /* TARDIS: T_RADIATIVE_ESTIMATOR_CONSTANT * nu_bar / j */
        if (est->j_estimator[i] > 0.0) { /* Phase 4 - Step 2 */
            double T_rad_est, W_est;
            int got_fit = 0;
            if (binned_j_on)
                got_fit = fit_dilute_planck_binned_j(est, i, volume[i],
                                                     time_simulation,
                                                     &T_rad_est, &W_est);
            if (!got_fit) {
                T_rad_est = T_RADIATIVE_CONSTANT * /* Phase 4 - Step 2 */
                    est->nu_bar_estimator[i] / est->j_estimator[i]; /* Phase 4 - Step 2 */

                /* Phase 4 - Step 2: W from j vs Planck(T_rad) */
                /* TARDIS: W = j / (4 * sigma_sb * T^4 * t_sim * V) */
                W_est = est->j_estimator[i] / /* Phase 4 - Step 2 */
                    (4.0 * SIGMA_SB * pow(T_rad_est, 4) * /* Phase 4 - Step 2 */
                     time_simulation * volume[i]); /* Phase 4 - Step 2 */
            }

            if (w_cap > 0.0 && W_est > w_cap) W_est = w_cap;

            /* Task #072: TARDIS damping (base.py: converge() for W and T_rad)
             * new_value = old_value + damping_constant * (estimated - old_value)
             * damping_constant = 0.5 by default in TARDIS */
            plasma->T_rad[i] = plasma->T_rad[i] +
                damping_constant * (T_rad_est - plasma->T_rad[i]);
            plasma->W[i] = plasma->W[i] +
                damping_constant * (W_est - plasma->W[i]);
        }
    }

    /* Diagnostic: dump the per-shell binned J_nu SED + fitted (T_rad,W) so the
     * dilute-Planck fit can be inspected offline (is a hot T_rad a real blue
     * field or a Wien-suppressed-bin log-fit artifact?). Gated, last-iter only. */
    if (binned_j_on && est->j_nu_estimator && est->nlte_n_freq_bins > 0 &&
        getenv("LUMINA_JNU_SED_DUMP") && atoi(getenv("LUMINA_JNU_SED_DUMP"))) {
        FILE *sf = fopen("lumina_jnu_sed.csv", "w");
        if (sf) {
            int nb = est->nlte_n_freq_bins;
            double nu_lo0 = est->nlte_nu_min, dlog = est->nlte_d_log_nu;
            fprintf(sf, "shell,T_rad,W,bin,nu_lo,nu_hi,J_nu\n");
            for (int i = 0; i < n_shells; i++) {
                if (volume[i] <= 0.0 || time_simulation <= 0.0) continue;
                double norm = 1.0 / (4.0 * M_PI_VAL * volume[i] * time_simulation);
                const double *raw = &est->j_nu_estimator[(size_t)i * nb];
                for (int b = 0; b < nb; b++) {
                    double nu_a = nu_lo0 * exp((double)b * dlog);
                    double nu_b = nu_lo0 * exp((double)(b + 1) * dlog);
                    double dnu = nu_b - nu_a;
                    double j = (raw[b] > 0.0 && dnu > 0.0) ? raw[b] * norm / dnu : 0.0;
                    fprintf(sf, "%d,%.2f,%.6e,%d,%.6e,%.6e,%.6e\n",
                            i, plasma->T_rad[i], plasma->W[i], b, nu_a, nu_b, j);
                }
            }
            fclose(sf);
            printf("  [JNU-SED] dumped per-shell J_nu histogram to lumina_jnu_sed.csv\n");
        }
    }
}

/* Fit a dilute Planck W*B_nu(T) to the per-shell J_nu histogram.
 *
 * J_nu[b] = raw[b] / (4*pi * V * t_sim * dnu_b)   [erg/s/cm^2/Hz/sr]
 * Total mean intensity J = sum_b J_nu[b]*dnu_b.
 * For a dilute Planck, J = W * sigma_SB*T^4/pi, so at any trial T the amplitude
 * is fixed analytically by W(T) = pi*J/(sigma_SB*T^4); only the SHAPE (T) is
 * searched. We minimize an energy-weighted log-space residual so the fit tracks
 * the SED peak rather than the redshift-sensitive first moment.
 *
 * Returns 1 and writes (*T_out,*W_out) on success, 0 if histogram unavailable. */
static int fit_dilute_planck_binned_j(const Estimators *est, int shell,
                                      double volume, double time_simulation,
                                      double *T_out, double *W_out) {
    if (est->j_nu_estimator == NULL || est->nlte_n_freq_bins <= 0) return 0;
    if (volume <= 0.0 || time_simulation <= 0.0) return 0;

    int    nb    = est->nlte_n_freq_bins;
    double nu_lo0 = est->nlte_nu_min;
    double dlog  = est->nlte_d_log_nu;
    const double *raw = &est->j_nu_estimator[(size_t)shell * nb];
    double norm = 1.0 / (4.0 * M_PI_VAL * volume * time_simulation);

    /* Build J_nu, bin centers/widths, total J, and energy weights. */
    double *Jnu = (double *)malloc((size_t)nb * sizeof(double));
    double *nu_c = (double *)malloc((size_t)nb * sizeof(double));
    double *wgt = (double *)malloc((size_t)nb * sizeof(double));
    if (!Jnu || !nu_c || !wgt) { free(Jnu); free(nu_c); free(wgt); return 0; }

    double J_tot = 0.0;
    int n_pos = 0;
    for (int b = 0; b < nb; b++) {
        double nu_a = nu_lo0 * exp((double)b * dlog);
        double nu_b = nu_lo0 * exp((double)(b + 1) * dlog);
        double dnu  = nu_b - nu_a;
        nu_c[b] = 0.5 * (nu_a + nu_b);
        double j = (raw[b] > 0.0 && dnu > 0.0) ? raw[b] * norm / dnu : 0.0;
        Jnu[b]  = j;
        double ener = j * dnu;          /* energy content of this bin */
        wgt[b]  = ener;
        J_tot  += ener;
        if (j > 0.0) n_pos++;
    }
    if (J_tot <= 0.0 || n_pos < 4) { free(Jnu); free(nu_c); free(wgt); return 0; }

    /* Two-pass T search: coarse log grid, then local linear refine. */
    const double T_LO = 1500.0, T_HI = 50000.0;
    double best_T = 0.0, best_res = -1.0;
    for (int pass = 0; pass < 2; pass++) {
        double t_lo, t_hi; int nstep;
        if (pass == 0) { t_lo = T_LO; t_hi = T_HI; nstep = 80; }
        else {
            /* refine within +-1.5 coarse steps around best_T (log spacing) */
            double f = exp(1.5 * log(T_HI / T_LO) / 80.0);
            t_lo = best_T / f; t_hi = best_T * f; nstep = 60;
            if (t_lo < T_LO) t_lo = T_LO;
            if (t_hi > T_HI) t_hi = T_HI;
        }
        for (int k = 0; k <= nstep; k++) {
            double T = (pass == 0)
                ? t_lo * pow(t_hi / t_lo, (double)k / nstep)
                : t_lo + (t_hi - t_lo) * (double)k / nstep;
            double W_T = M_PI_VAL * J_tot / (SIGMA_SB * pow(T, 4));
            if (W_T <= 0.0) continue;
            double res = 0.0, wsum = 0.0;
            for (int b = 0; b < nb; b++) {
                if (Jnu[b] <= 0.0) continue;
                double model = W_T * planck_bnu(T, nu_c[b]);
                if (model <= 0.0) continue;
                double d = log(Jnu[b]) - log(model);
                res  += wgt[b] * d * d;
                wsum += wgt[b];
            }
            if (wsum <= 0.0) continue;
            res /= wsum;
            if (best_res < 0.0 || res < best_res) { best_res = res; best_T = T; }
        }
    }

    free(Jnu); free(nu_c); free(wgt);
    if (best_T <= 0.0) return 0;
    *T_out = best_T;
    *W_out = M_PI_VAL * J_tot / (SIGMA_SB * pow(best_T, 4));
    return 1;
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
    /* ion_stage is the ABSOLUTE carsus ion number (neutral=0). The ion-pop
       ladder may start above neutral (CMFGEN omits Ti I / Mn I), so slot index
       != absolute ion number in general. Match against the absolute stage
       stored in ion_pop_stage rather than assuming a relative offset. */
    for (int e = 0; e < atom->n_elements; e++) {
        if (atom->element_Z[e] != Z) continue;
        int start = atom->elem_ion_offset[e];
        int end   = atom->elem_ion_offset[e + 1];
        for (int ip = start; ip < end; ip++) {
            if (atom->ion_pop_stage[ip] == ion_stage) return ip;
        }
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

/* P4 (2026-05-14): ζ override env knobs for sensitivity probing.
 *   LUMINA_ZETA_OVERRIDE_ZMASK    comma-list of Z (e.g. "27,28")
 *   LUMINA_ZETA_OVERRIDE_IONMASK  comma-list of ion_number values matching csv (e.g. "2")
 *   LUMINA_ZETA_OVERRIDE_VAL      replacement ζ (default 0.5)
 * Probes M-L hybrid response to ζ for known Carsus placeholder rows
 * (Z=27 Co III ≡ Z=28 Ni III ≡ Z=22 Ti III; Z=27 Co II ≡ Z=21 Sc II). */
static unsigned int zeta_override_z_mask = 0;
static unsigned int zeta_override_ion_mask = 0;
static double zeta_override_val = 0.5;
static int zeta_override_initialized = 0;

static void init_zeta_override(void) {
    if (zeta_override_initialized) return;
    zeta_override_initialized = 1;

    const char *zm = getenv("LUMINA_ZETA_OVERRIDE_ZMASK");
    if (zm && *zm) {
        const char *p = zm;
        while (*p) {
            int z = atoi(p);
            if (z > 0 && z < 32) zeta_override_z_mask |= (1u << z);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }
    const char *im = getenv("LUMINA_ZETA_OVERRIDE_IONMASK");
    if (im && *im) {
        const char *p = im;
        while (*p) {
            int ii = atoi(p);
            if (ii >= 0 && ii < 8) zeta_override_ion_mask |= (1u << ii);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }
    const char *vv = getenv("LUMINA_ZETA_OVERRIDE_VAL");
    if (vv) zeta_override_val = atof(vv);

    if (zeta_override_z_mask != 0 && zeta_override_ion_mask != 0) {
        printf("  [ζ-override] val=%.3f zmask=0x%x ionmask=0x%x\n",
               zeta_override_val, zeta_override_z_mask, zeta_override_ion_mask);
    }
}

/* P3 (2026-05-14): Two-component ion-lock — W-threshold conditional LTE-at-T_e.
 *   LUMINA_LOCK_W_THRESH   shell W threshold (>= activates LTE-at-T_e lock; default 0 = off)
 *   LUMINA_LOCK_ZMASK      comma-list of Z (default iron-peak 21..28 when active)
 *   LUMINA_LOCK_IONMASK    comma-list of k array indices (default k=1,2 = II→III + III→IV)
 * Effect: inner shells (W>thresh) get phi = phi_LTE_at_Te (pure Saha at T_e, more
 * recombined → less Fe III → less UV opacity). Outer shells keep M-L hybrid. */
static double lock_w_thresh = 0.0;
static unsigned int lock_z_mask = 0;
static unsigned int lock_ion_mask = 0;
static int lock_initialized = 0;

static void init_twocomp_lock(void) {
    if (lock_initialized) return;
    lock_initialized = 1;

    const char *wt = getenv("LUMINA_LOCK_W_THRESH");
    if (wt) lock_w_thresh = atof(wt);

    const char *zm = getenv("LUMINA_LOCK_ZMASK");
    if (zm && *zm) {
        const char *p = zm;
        while (*p) {
            int z = atoi(p);
            if (z > 0 && z < 32) lock_z_mask |= (1u << z);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }
    const char *im = getenv("LUMINA_LOCK_IONMASK");
    if (im && *im) {
        const char *p = im;
        while (*p) {
            int ii = atoi(p);
            if (ii >= 0 && ii < 8) lock_ion_mask |= (1u << ii);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }

    if (lock_w_thresh > 0.0) {
        printf("  [2-comp-lock] W_thresh=%.3f zmask=0x%x ionmask=0x%x\n",
               lock_w_thresh, lock_z_mask, lock_ion_mask);
    }
}

static inline double apply_twocomp_lock(double phi_neb, double phi_LTE_at_Te,
                                         int Z, int k, double W) {
    if (lock_w_thresh <= 0.0 || W < lock_w_thresh) return phi_neb;
    if (lock_z_mask != 0 &&
        (Z < 0 || Z >= 32 || !(lock_z_mask & (1u << Z))))
        return phi_neb;
    if (lock_ion_mask != 0 &&
        (k < 0 || k >= 8 || !(lock_ion_mask & (1u << k))))
        return phi_neb;
    return phi_LTE_at_Te;
}

/* (2) NLTE no-ML-lock (2026-05-14): replace phi_neb-derived n_total in the
 * NLTE pair conservation row with the element's total mass-conserved density.
 * Removes the Mihalas-Lucy "soft lock" on the (II+III) pair total, letting
 * the rate matrix redistribute mass across all ion stages it tracks.
 * Probe to test whether the structural cooling-feedback divergence stems
 * from the phi_neb input itself rather than from level/ion redistribution. */
static int nlte_no_ml_lock_init = 0;
static int nlte_no_ml_lock_enabled = 0;
static void init_nlte_no_ml_lock(void) {
    if (nlte_no_ml_lock_init) return;
    nlte_no_ml_lock_init = 1;
    const char *e = getenv("LUMINA_NLTE_NO_ML_LOCK");
    if (e && atoi(e) != 0) {
        nlte_no_ml_lock_enabled = 1;
        printf("  [NLTE no-ML-lock] pair n_total -> n_element (mass conservation)\n");
    }
}

double nlte_pair_total_density(NLTEConfig *nlte, AtomicData *atom,
                               PlasmaState *plasma,
                               int Z_nl, int ion_idx_lo, int ion_idx_hi,
                               int shell) {
    init_nlte_no_ml_lock();
    int n_shells = plasma->n_shells;
    if (nlte_no_ml_lock_enabled) {
        for (int e = 0; e < atom->n_elements; e++) {
            if (atom->element_Z[e] != Z_nl) continue;
            double mass_amu = atom->element_mass_amu[e];
            double rho = plasma->rho[shell];
            double abund = atom->abundances[e * n_shells + shell];
            return (abund * rho) / (mass_amu * AMU);
        }
        return 0.0;
    }
    double n_total = 0.0;
    for (int i = ion_idx_lo; i <= ion_idx_hi; i++) {
        int ip = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[i]);
        if (ip >= 0)
            n_total += atom->ion_number_density[ip * n_shells + shell];
    }
    return n_total;
}

/* Task #072: Helper — interpolate zeta factor for (Z, ion_stage) at temperature T */
static double interpolate_zeta(AtomicData *atom, int Z, int ion_stage, double T) {
    /* P4 override: replace ζ with constant value for masked (Z, ion) pairs */
    if (zeta_override_z_mask != 0 && zeta_override_ion_mask != 0 &&
        Z > 0 && Z < 32 && ion_stage >= 0 && ion_stage < 8 &&
        (zeta_override_z_mask & (1u << Z)) &&
        (zeta_override_ion_mask & (1u << ion_stage))) {
        return zeta_override_val;
    }

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


/* Mihalas-Lucy phi_neb correction (Path d, 2026-05-13):
 *   phi_neb' = phi_neb * boost * (T_e/T_rad)^t_ratio_exp
 * Default boost=1.0, t_ratio_exp=0 (no change). Use Z/ion bitmasks
 * to apply selectively. Targets over-ionization at SN photosphere
 * where interpolate_zeta() falls back to 1.0 (no ζ data). */
static double ml_phi_neb_boost = 1.0;
static double ml_phi_neb_t_ratio_exp = 0.0;
static unsigned int ml_phi_neb_z_mask = 0;
static unsigned int ml_phi_neb_ion_mask = 0;
static int ml_phi_neb_initialized = 0;

static void init_ml_phi_neb_correction(void) {
    if (ml_phi_neb_initialized) return;
    ml_phi_neb_initialized = 1;

    const char *eb = getenv("LUMINA_ML_PHI_NEB_BOOST");
    if (eb) ml_phi_neb_boost = atof(eb);
    const char *ee = getenv("LUMINA_ML_PHI_NEB_T_RATIO_EXP");
    if (ee) ml_phi_neb_t_ratio_exp = atof(ee);

    const char *zm = getenv("LUMINA_ML_PHI_NEB_ZMASK");
    if (zm && *zm) {
        const char *p = zm;
        while (*p) {
            int z = atoi(p);
            if (z > 0 && z < 32) ml_phi_neb_z_mask |= (1u << z);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }
    const char *im = getenv("LUMINA_ML_PHI_NEB_IONMASK");
    if (im && *im) {
        const char *p = im;
        while (*p) {
            int ii = atoi(p);
            if (ii >= 0 && ii < 8) ml_phi_neb_ion_mask |= (1u << ii);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }

    if (ml_phi_neb_boost != 1.0 || ml_phi_neb_t_ratio_exp != 0.0) {
        printf("  [M-L phi_neb] boost=%g t_ratio_exp=%g zmask=0x%x ionmask=0x%x\n",
               ml_phi_neb_boost, ml_phi_neb_t_ratio_exp,
               ml_phi_neb_z_mask, ml_phi_neb_ion_mask);
    }
}

static inline double apply_ml_phi_neb_correction(double phi_neb, int Z, int k,
                                                  double T_e, double T_rad) {
    if (ml_phi_neb_z_mask != 0 &&
        (Z < 0 || Z >= 32 || !(ml_phi_neb_z_mask & (1u << Z))))
        return phi_neb;
    if (ml_phi_neb_ion_mask != 0 &&
        (k < 0 || k >= 8 || !(ml_phi_neb_ion_mask & (1u << k))))
        return phi_neb;

    double factor = ml_phi_neb_boost;
    if (ml_phi_neb_t_ratio_exp != 0.0 && T_rad > 0.0) {
        double t_ratio = T_e / T_rad;
        if (t_ratio > 0.0) factor *= pow(t_ratio, ml_phi_neb_t_ratio_exp);
    }
    return phi_neb * factor;
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
/* steady-state nebular-Saha ion partition for ONE shell (all elements). Extracted
 * so the coupled Newton can reconcile only the shells it does NOT own. */
static void compute_ion_populations_shell(AtomicData *atom, PlasmaState *plasma,
                                          int s, int n_shells) {
    for (int e = 0; e < atom->n_elements; e++) {
        int Z_elem = atom->element_Z[e];
        double mass_amu = atom->element_mass_amu[e];
        int ip_start = atom->elem_ion_offset[e];
        int ip_end   = atom->elem_ion_offset[e + 1];
        int n_pops   = ip_end - ip_start;

        {
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
                /* Absolute ion stage of ip_cur. Equals k only when the element's ion
                 * ladder starts at neutral. Ti/Mn dropped their neutral stage from the
                 * ref, so they start at ion 1; keying ionizE/zeta on k would query the
                 * absent ion-0 energy -> find_ioniz_energy returns the 1e10 sentinel ->
                 * the whole element collapses into a level-less stage. Use the real stage. */
                int stage = atom->ion_pop_stage[ip_cur];

                /* Dilute partition functions (W-weighted, consistent with level pops) */
                double Z_cur  = atom->partition_functions[ip_cur  * n_shells + s];
                double Z_next = atom->partition_functions[ip_next * n_shells + s];

                double chi_eV  = find_ioniz_energy(atom, Z_elem, stage);
                double chi_erg = chi_eV * EV_TO_ERG;

                /* Decomposed phi_neb to avoid 0*inf cancellation:
                 *   Original:  phi_neb = phi_lte * W * (zeta*delta + W*(1-zeta)) * sqrt(Te/Trad)
                 *              where phi_lte=exp(-chi*beta_rad) and delta has exp(+chi*(beta_rad-beta_e))
                 *              -> at low T_rad+high chi, phi_lte->0 AND delta->inf simultaneously => NaN.
                 *   Identity:  phi_lte * delta = (Te/Trad) * phi_LTE_at_Te
                 *              (the exp(-chi*beta_rad) and exp(+chi*beta_rad) cancel analytically)
                 *   Each phi_LTE_* is a single exp(negative): underflows safely to 0, never inf. */
                double prefactor = (Z_next / Z_cur) * 2.0 * g_electron;
                double phi_LTE_at_Trad = prefactor * exp(-chi_erg * beta_rad);
                double phi_LTE_at_Te   = prefactor * exp(-chi_erg * beta_electron);

                double zeta = interpolate_zeta(atom, Z_elem, stage, T_rad);
                double sqrt_te_tr = sqrt(T_e / T_rad);
                double phi_neb = W * sqrt_te_tr *
                    (zeta * (T_e / T_rad) * phi_LTE_at_Te +
                     W * (1.0 - zeta) * phi_LTE_at_Trad);
                phi_neb = apply_ml_phi_neb_correction(phi_neb, Z_elem, stage, T_e, T_rad);
                phi_neb = apply_twocomp_lock(phi_neb, phi_LTE_at_Te, Z_elem, stage, W);

                /* ratio n_{i+1}/n_i = phi_nebular / n_e */
                double ratio;
                if (n_e > 0.0) {
                    ratio = phi_neb / n_e;
                } else {
                    ratio = 1e10;
                }
                if (!isfinite(ratio) || ratio < 0.0) ratio = 0.0;
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

static void compute_ion_populations(AtomicData *atom, PlasmaState *plasma,
                                     int n_shells) {
    init_ml_phi_neb_correction();
    init_zeta_override();
    init_twocomp_lock();
    for (int s = 0; s < n_shells; s++)
        compute_ion_populations_shell(atom, plasma, s, n_shells);
}

/* Task #072 Step 4c: Compute electron density (iterative)
 * Uses the correct TARDIS nebular Saha formula with TARDIS-style damped iteration:
 *   n_e_new_damped = 0.5 * (n_e_computed + n_e_old)
 *   convergence threshold: 5% (TARDIS default)
 *   max iterations: 100 (TARDIS default) */
static void compute_electron_density(AtomicData *atom, PlasmaState *plasma,
                                      int n_shells) {
    init_ml_phi_neb_correction();
    init_zeta_override();
    init_twocomp_lock();
    for (int s = 0; s < n_shells; s++) {
        double n_e = plasma->n_electron[s];
        if (!isfinite(n_e) || n_e <= 0.0) n_e = 1e6;

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
                    int stage = atom->ion_pop_stage[ip_cur];  /* absolute stage; != k for Ti/Mn (no neutral) */
                    /* Dilute partition functions (W-weighted, consistent with level pops) */
                    double Z_cur  = atom->partition_functions[ip_cur  * n_shells + s];
                    double Z_next = atom->partition_functions[ip_next * n_shells + s];
                    double chi_eV = find_ioniz_energy(atom, Z_elem, stage);
                    double chi_erg = chi_eV * EV_TO_ERG;

                    /* Decomposed phi_neb (see compute_ion_populations for full comment) */
                    double prefactor = (Z_next / Z_cur) * 2.0 * g_electron;
                    double phi_LTE_at_Trad = prefactor * exp(-chi_erg * beta_rad);
                    double phi_LTE_at_Te   = prefactor * exp(-chi_erg * beta_electron);
                    double zeta = interpolate_zeta(atom, Z_elem, stage, T_rad);
                    double sqrt_te_tr = sqrt(T_e / T_rad);
                    double phi_neb = W * sqrt_te_tr *
                        (zeta * (T_e / T_rad) * phi_LTE_at_Te +
                         W * (1.0 - zeta) * phi_LTE_at_Trad);
                    phi_neb = apply_ml_phi_neb_correction(phi_neb, Z_elem, stage, T_e, T_rad);
                    phi_neb = apply_twocomp_lock(phi_neb, phi_LTE_at_Te, Z_elem, stage, W);

                    double ratio = (n_e > 0.0) ? phi_neb / n_e : 1e10;
                    if (!isfinite(ratio) || ratio < 0.0) ratio = 0.0;
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
                double n_ion_contrib = atom->ion_number_density[ip * n_shells + s];
                if (isfinite(n_ion_contrib) && n_ion_contrib > 0.0)
                    n_e_new += charge * n_ion_contrib;
            }
            if (!isfinite(n_e_new) || n_e_new < 1.0) n_e_new = 1.0;

            /* TARDIS-style damped update: n_e = 0.5 * (n_e_new + n_e_old) */
            n_e = 0.5 * (n_e_new + n_e_old);
            plasma->n_electron[s] = n_e;

            /* TARDIS convergence: 5% relative threshold */
            if (n_e_old > 0.0 && fabs(n_e_new - n_e_old) / n_e_old < 0.05) break;
        }
    }
}

/* Task #072 Step 4d: Compute tau_sobolev from ion populations */
/* Per-Z line-opacity zero-out via LUMINA_OPACITY_SKIP_Z (comma list, e.g. "8,6"). */
static int opacity_skip_z[100];
static int opacity_skip_z_init = 0;
static void opacity_skip_z_load(void) {
    if (opacity_skip_z_init) return;
    opacity_skip_z_init = 1;
    const char *e = getenv("LUMINA_OPACITY_SKIP_Z");
    if (!e || !*e) return;
    char buf[256]; strncpy(buf, e, sizeof(buf)-1); buf[sizeof(buf)-1]=0;
    char *tok = strtok(buf, ", \t");
    while (tok) {
        int z = atoi(tok);
        if (z > 0 && z < 100) opacity_skip_z[z] = 1;
        tok = strtok(NULL, ", \t");
    }
    printf("  [OPACITY] LUMINA_OPACITY_SKIP_Z active (line tau zeroed): ");
    for (int i = 1; i < 100; i++) if (opacity_skip_z[i]) printf("Z=%d ", i);
    printf("\n");
}
/* A2: was non-static but unreferenced (dead-code audit Tier 3.3b finding).
 * Kept as a static helper; redundant with the inline lookup at
 * compute_tau_sobolev:line 636 below. */
__attribute__((unused))
static int opacity_skip_z_is_masked(int Z) {
    opacity_skip_z_load();
    return (Z > 0 && Z < 100 && opacity_skip_z[Z]);
}

static void compute_tau_sobolev(AtomicData *atom, PlasmaState *plasma,
                                 OpacityState *opacity, double time_explosion) {
    int n_lines = opacity->n_lines;
    int n_shells = opacity->n_shells;
    opacity_skip_z_load();

    for (int line = 0; line < n_lines; line++) {
        int Z         = atom->line_atomic_number[line];
        int ion_stage = atom->line_ion_number[line];
        int lev_lower = atom->line_level_lower[line];
        int lev_upper = atom->line_level_upper[line];
        double f_lu   = atom->line_f_lu[line];
        double lam_cm = atom->line_wavelength_cm[line];

        /* Diagnostic: zero-out lines for masked Z */
        if (Z > 0 && Z < 100 && opacity_skip_z[Z]) {
            for (int s = 0; s < n_shells; s++)
                opacity->tau_sobolev[line * n_shells + s] = 1e-100;
            continue;
        }

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

/* Probe-B fix (task #29): write the NLTE-solved ion stage back into
 * atom->ion_number_density so the BULK (non-NLTE-tracked) line opacity built by
 * compute_tau_sobolev uses the rate-solved ionization instead of the
 * over-ionized nebular phi_neb. scripts/probe_b_three_way_ionization.py showed
 * the NLTE solve yields the correct II-dominant iron curtain at line-forming
 * shells (Fe fIII 0.06-0.31) while phi_neb stays III-dominant (0.98-0.998), and
 * opacity was discarding the NLTE split. Each pair is rescaled to its EXISTING
 * nebular pair-total (which the NLTE conservation already preserves), so only the
 * II/III split changes — abundances and untracked stages are untouched. The O
 * triplet overlap pairs (indices 14,15: slot 29 shared) are skipped. After the
 * write-back the bulk tau is rebuilt; callers then re-apply the per-line NLTE tau
 * override on top. Gated by LUMINA_NLTE_OPACITY_IONSTAGE=1 (default off = no-op).
 * Shared by the CPU and GPU NLTE solvers. */
void nlte_writeback_ion_stage(NLTEConfig *nlte, AtomicData *atom,
                              PlasmaState *plasma, OpacityState *opacity,
                              double time_explosion, int n_shells,
                              int pairs[][2], int n_pairs) {
    const char *e = getenv("LUMINA_NLTE_OPACITY_IONSTAGE");
    if (!e || e[0] != '1') return;

    int n_writeback = 0;
    for (int p = 0; p < n_pairs; p++) {
        if (p == 14 || p == 15) continue; /* O triplet overlap: skip */
        int lo = pairs[p][0], hi = pairs[p][1];
        int ip_lo = find_ion_pop_idx(atom, nlte->nlte_Z[lo], nlte->nlte_ion[lo]);
        int ip_hi = find_ion_pop_idx(atom, nlte->nlte_Z[hi], nlte->nlte_ion[hi]);
        if (ip_lo < 0 || ip_hi < 0) continue;
        int los = nlte->nlte_ion_level_offset[lo];
        int loe = nlte->nlte_ion_level_offset[lo + 1];
        int his = nlte->nlte_ion_level_offset[hi];
        int hie = nlte->nlte_ion_level_offset[hi + 1];
        for (int s = 0; s < n_shells; s++) {
            double sum_lo = 0.0, sum_hi = 0.0;
            for (int l = los; l < loe; l++)
                sum_lo += nlte->nlte_level_populations[(size_t)l * n_shells + s];
            for (int l = his; l < hie; l++)
                sum_hi += nlte->nlte_level_populations[(size_t)l * n_shells + s];
            double tot = sum_lo + sum_hi;
            double T_neb = atom->ion_number_density[(size_t)ip_lo * n_shells + s]
                         + atom->ion_number_density[(size_t)ip_hi * n_shells + s];
            if (tot > 0.0 && T_neb > 0.0 && isfinite(tot)) {
                double new_lo = T_neb * sum_lo / tot;
                double new_hi = T_neb * sum_hi / tot;
                if (new_lo < 1e-300) new_lo = 1e-300;
                if (new_hi < 1e-300) new_hi = 1e-300;
                atom->ion_number_density[(size_t)ip_lo * n_shells + s] = new_lo;
                atom->ion_number_density[(size_t)ip_hi * n_shells + s] = new_hi;
            }
        }
        n_writeback++;
    }
    printf("  [NLTE] LUMINA_NLTE_OPACITY_IONSTAGE: wrote NLTE ion split back to "
           "ion_number_density for %d pairs; rebuilding bulk tau\n", n_writeback);
    compute_tau_sobolev(atom, plasma, opacity, time_explosion);
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
    /* #297 (γ) outer-shell T_e damper: per-shell multiplier on T_e for
     * shells s >= LUMINA_OUTER_TE_DAMP_SMIN. Used to cool the outer
     * source-function region without touching the inner photosphere
     * (T_inner-pin convergence preserved). 1000 K hard floor. */
    static int outer_init = 0;
    static double outer_damp = 1.0;
    static int outer_smin = 999;
    if (!outer_init) {
        const char *e1 = getenv("LUMINA_OUTER_TE_DAMP_FACTOR");
        const char *e2 = getenv("LUMINA_OUTER_TE_DAMP_SMIN");
        if (e1) outer_damp = atof(e1);
        if (e2) outer_smin = atoi(e2);
        outer_init = 1;
        if (outer_damp != 1.0)
            printf("  [GAMMA] Outer-shell T_e damper: factor=%.3f for s>=%d\n",
                   outer_damp, outer_smin);
    }

    if (!self_consistent) {
        /* Default: uniform ratio */
        for (int s = 0; s < n_shells; s++)
            plasma->T_e[s] = plasma->T_e_T_rad_ratio * plasma->T_rad[s];
    } else {

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
     * much more strongly than Compton alone. Default 12 reproduces
     * TARDIS-like T_e/T_rad ≈ 0.97 inner. Set LUMINA_F_COLL_BOOST=0
     * for pure Compton-adiabatic balance (T_e ≈ 0.7 T_rad inner) —
     * Path 3 lever against R_rec/R_bf saturation. */
    static int boost_init = 0;
    static double f_coll_boost = 12.0;
    if (!boost_init) {
        const char *e = getenv("LUMINA_F_COLL_BOOST");
        if (e) f_coll_boost = atof(e);
        boost_init = 1;
    }

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
    }  /* end self-consistent branch */

    /* Apply outer-shell T_e damper after T_e is set (both branches). */
    if (outer_damp != 1.0 && outer_smin < n_shells) {
        for (int s = outer_smin; s < n_shells; s++) {
            plasma->T_e[s] *= outer_damp;
            if (plasma->T_e[s] < 1000.0) plasma->T_e[s] = 1000.0;
        }
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

    /* P5: optional cap on B_lu·J_ν internal-up rate. When the carsus 3.4M-line
     * macro-atom is iron-forest-trapped, J_ν(UV) inflates 10-100× above LTE,
     * driving runaway UV pumping that locks the cascade. Capping J_line at
     * (cap_factor × W·B_ν(T_rad)) re-imposes a Mazzali-Lucy expectation
     * without disabling NLTE entirely. Set with LUMINA_J_CAP_FACTOR (>0 to
     * enable; typical 2-10). 0/unset = no cap. */
    static double j_cap_factor = -1.0;
    if (j_cap_factor < 0.0) {
        const char *e = getenv("LUMINA_J_CAP_FACTOR");
        j_cap_factor = (e ? atof(e) : 0.0);
        if (j_cap_factor < 0.0) j_cap_factor = 0.0;
    }

    /* Detailed-rate shot-noise regularization (the documented `detailed` mode
     * requirement, SIROCCO/Python macro-atom docs). At low T_rad in outer
     * shells a single MC photon in a bin can spike J_ν → one internal-up rate
     * dominates the source-level block → shot noise → convergence runaway.
     * Floor the binned-J up-rate at j_floor_factor·W·B_ν(T_rad): under-sampled
     * (near-zero) bins are pulled toward the smooth dilute-Planck anchor while
     * well-sampled bins keep their MC value. Symmetric to j_cap_factor (which
     * caps from above). Set with LUMINA_J_FLOOR_FACTOR (>0 enable; typical
     * 0.05-0.5). 0/unset = no floor. */
    static double j_floor_factor = -1.0;
    if (j_floor_factor < 0.0) {
        const char *e = getenv("LUMINA_J_FLOOR_FACTOR");
        j_floor_factor = (e ? atof(e) : 0.0);
        if (j_floor_factor < 0.0) j_floor_factor = 0.0;
    }

    /* Emit-only boost on Fe II / Ni II UV→opt transitions, isolating the
     * spontaneous-emission lever from the radiative-pumping lever. Only
     * affects ttype == -1 (BB emit) for lines on Z={26,28}, ion=1 within
     * [LAM_MIN, LAM_MAX]. Three independent windows so UVtg ([3200,3800]),
     * fluo ([3800,4500]), and CaK ([2600,3200]) can be tuned separately.
     * Defaults: factor=1.0. */
    static double uvopt_emit_boost  = -1.0;
    static double uvopt_lam_min     = -1.0;
    static double uvopt_lam_max     = -1.0;
    static double uvopt_emit_boost2 = -1.0;
    static double uvopt_lam_min2    = -1.0;
    static double uvopt_lam_max2    = -1.0;
    static double uvopt_emit_boost3 = -1.0;
    static double uvopt_lam_min3    = -1.0;
    static double uvopt_lam_max3    = -1.0;
    /* W4 narrow window with its own Z/ion mask so we can target iron-peak III
     * for [3000,3100] bump suppress without altering W1/W2/W3 (Fe II+Ni II). */
    static double uvopt_emit_boost4 = -1.0;
    static double uvopt_lam_min4    = -1.0;
    static double uvopt_lam_max4    = -1.0;
    static unsigned int uvopt_z_mask4   = 0;  /* falls back to global if unset */
    static unsigned int uvopt_ion_mask4 = 0;  /* falls back to global if unset */
    /* P1 2026-05-13: UV internal-down suppress — break UV→UV cascade in
     * iron-peak macro-atom by scaling ttype=0 rate when destination line is UV.
     * Defaults: factor=1.0 (no-op), thresh=4000 Å, Z∈{21..28}, ion∈{1,2}. */
    static double macro_uv_idown_factor = -1.0;
    static double macro_uv_idown_thresh = -1.0;
    static unsigned int macro_uv_idown_z_mask = 0;
    static unsigned int macro_uv_idown_ion_mask = 0;
    /* Global Z mask for boost (bitmap over Z 1..30). Default = Fe(26) | Ni(28). */
    static unsigned int uvopt_z_mask = 0;
    /* Global ion-stage mask for boost (bitmap over ion 0..7). Default = ion=1 (II species only). */
    static unsigned int uvopt_ion_mask = 0;
    if (uvopt_emit_boost < 0.0) {
        const char *e   = getenv("LUMINA_UVOPT_EMIT_BOOST");
        const char *emn = getenv("LUMINA_UVOPT_EMIT_LAM_MIN");
        const char *emx = getenv("LUMINA_UVOPT_EMIT_LAM_MAX");
        const char *e2   = getenv("LUMINA_UVOPT_EMIT_BOOST2");
        const char *emn2 = getenv("LUMINA_UVOPT_EMIT_LAM_MIN2");
        const char *emx2 = getenv("LUMINA_UVOPT_EMIT_LAM_MAX2");
        const char *e3   = getenv("LUMINA_UVOPT_EMIT_BOOST3");
        const char *emn3 = getenv("LUMINA_UVOPT_EMIT_LAM_MIN3");
        const char *emx3 = getenv("LUMINA_UVOPT_EMIT_LAM_MAX3");
        const char *e4   = getenv("LUMINA_UVOPT_EMIT_BOOST4");
        const char *emn4 = getenv("LUMINA_UVOPT_EMIT_LAM_MIN4");
        const char *emx4 = getenv("LUMINA_UVOPT_EMIT_LAM_MAX4");
        uvopt_emit_boost  = (e    ? atof(e)    : 1.0);
        uvopt_lam_min     = (emn  ? atof(emn)  : 4500.0);
        uvopt_lam_max     = (emx  ? atof(emx)  : 7000.0);
        uvopt_emit_boost2 = (e2   ? atof(e2)   : 1.0);
        uvopt_lam_min2    = (emn2 ? atof(emn2) : 3800.0);
        uvopt_lam_max2    = (emx2 ? atof(emx2) : 4500.0);
        uvopt_emit_boost3 = (e3   ? atof(e3)   : 1.0);
        uvopt_lam_min3    = (emn3 ? atof(emn3) : 2600.0);
        uvopt_lam_max3    = (emx3 ? atof(emx3) : 3200.0);
        uvopt_emit_boost4 = (e4   ? atof(e4)   : 1.0);
        uvopt_lam_min4    = (emn4 ? atof(emn4) : 3000.0);
        uvopt_lam_max4    = (emx4 ? atof(emx4) : 3100.0);
        /* Allow factors in (0, +inf): >1 boosts emit, <1 suppresses. Branching
         * ratios re-normalize to 1.0 in the rates loop below, so a sub-unity
         * factor redistributes probability to the unaffected branches. */
        if (uvopt_emit_boost  <= 0.0) uvopt_emit_boost  = 1.0;
        if (uvopt_emit_boost2 <= 0.0) uvopt_emit_boost2 = 1.0;
        if (uvopt_emit_boost3 <= 0.0) uvopt_emit_boost3 = 1.0;
        if (uvopt_emit_boost4 <= 0.0) uvopt_emit_boost4 = 1.0;
        const char *zmask = getenv("LUMINA_UVOPT_EMIT_ZMASK");
        if (zmask && *zmask) {
            const char *p = zmask;
            while (*p) {
                int z = atoi(p);
                if (z > 0 && z < 32) uvopt_z_mask |= (1u << z);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else {
            uvopt_z_mask = (1u << 26) | (1u << 28);  /* Fe II + Ni II default */
        }
        const char *imask = getenv("LUMINA_UVOPT_EMIT_IONMASK");
        if (imask && *imask) {
            const char *p = imask;
            while (*p) {
                int ii = atoi(p);
                if (ii >= 0 && ii < 8) uvopt_ion_mask |= (1u << ii);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else {
            uvopt_ion_mask = (1u << 1);  /* default: ion=1 (II species) — backward compat */
        }
        /* Per-window masks for W4 (fall back to global if unset). */
        const char *zmask4 = getenv("LUMINA_UVOPT_EMIT_ZMASK4");
        if (zmask4 && *zmask4) {
            const char *p = zmask4;
            while (*p) {
                int z = atoi(p);
                if (z > 0 && z < 32) uvopt_z_mask4 |= (1u << z);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else {
            uvopt_z_mask4 = uvopt_z_mask;
        }
        const char *imask4 = getenv("LUMINA_UVOPT_EMIT_IONMASK4");
        if (imask4 && *imask4) {
            const char *p = imask4;
            while (*p) {
                int ii = atoi(p);
                if (ii >= 0 && ii < 8) uvopt_ion_mask4 |= (1u << ii);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else {
            uvopt_ion_mask4 = uvopt_ion_mask;
        }

        /* P1 UV internal-down suppress (env once) */
        const char *euid    = getenv("LUMINA_MACRO_UV_IDOWN_FACTOR");
        const char *euidth  = getenv("LUMINA_MACRO_UV_IDOWN_THRESH");
        const char *euidz   = getenv("LUMINA_MACRO_UV_IDOWN_ZMASK");
        const char *euidim  = getenv("LUMINA_MACRO_UV_IDOWN_IONMASK");
        macro_uv_idown_factor = (euid   ? atof(euid)   : 1.0);
        macro_uv_idown_thresh = (euidth ? atof(euidth) : 4000.0);
        if (macro_uv_idown_factor <= 0.0) macro_uv_idown_factor = 1.0;
        if (euidz && *euidz) {
            const char *p = euidz;
            while (*p) {
                int z = atoi(p);
                if (z > 0 && z < 32) macro_uv_idown_z_mask |= (1u << z);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else {
            /* default: Sc..Ni (21..28) */
            for (int z = 21; z <= 28; z++) macro_uv_idown_z_mask |= (1u << z);
        }
        if (euidim && *euidim) {
            const char *p = euidim;
            while (*p) {
                int ii = atoi(p);
                if (ii >= 0 && ii < 8) macro_uv_idown_ion_mask |= (1u << ii);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else {
            macro_uv_idown_ion_mask = (1u << 1) | (1u << 2);  /* default II + III */
        }
    }

    /* Find max block size for temp buffer */
    int max_block = 0;
    for (int lev = 0; lev < n_levels; lev++) {
        int bs = opacity->macro_block_references[lev + 1] -
                 opacity->macro_block_references[lev];
        if (bs > max_block) max_block = bs;
    }
    double *rates_buf = (double *)malloc(max_block * sizeof(double));

    /* ---- k-packet thermal pool (collisional macro-atom) ----
     * A purely-radiative macro-atom cascade has no thermalization channel and
     * systematically redshifts (the down-branch A_ul·β is radiation-field-
     * independent). Real CMFGEN/TARDIS couples line energy to the electron pool
     * via COLLISIONS: collisional de-excitation transfers hν to a free electron
     * (a "k-packet"), which re-excites a level sampled from the thermal
     * collisional emissivity — pinning re-emission near the local Planck peak.
     * Here we build two per-shell tables consumed by the GPU selector:
     *   p_kpacket[lev][s]   = P(macro-atom at lev deactivates collisionally)
     *                       = ΣC_down / (ΣC_down + Σradiative_rates)
     *   kpacket_cdf[s][lev] = cumulative dist over which level a k-packet
     *                         re-excites, weight = n_lower·C_up·dE.
     * Collisional rates reuse the NLTE van Regemorter/Axelrod forms. Gated by
     * LUMINA_KPACKET (default off). Collisional internal-up (staying a macro-
     * atom without thermalizing) is folded into the k-packet re-excitation
     * (second-order; the dominant down-coll→k-packet→thermal-up loop is kept). */
    static int kpacket_init = 0;
    static int kpacket_mode = 0;
    if (!kpacket_init) {
        const char *e = getenv("LUMINA_KPACKET");
        if (e && atoi(e) != 0) kpacket_mode = 1;
        kpacket_init = 1;
    }
    /* Per-line cached lookups (static; line→global-level map is iteration- and
     * shell-invariant). glo/gup = global level idx of lower/upper; ip = ion-pop
     * slot. -1 = unresolved (skip). */
    static int   kp_n_lines_cached = -1;
    static int  *kp_glo = NULL, *kp_gup = NULL, *kp_ip = NULL;
    const double VAN_REG_COEFF = 2.16e-6, AX_OMEGA = 1.0; /* match NLTE solver */
    if (kpacket_mode) {
        if (kp_n_lines_cached != atom->n_lines) {
            free(kp_glo); free(kp_gup); free(kp_ip);
            kp_glo = (int *)malloc(atom->n_lines * sizeof(int));
            kp_gup = (int *)malloc(atom->n_lines * sizeof(int));
            kp_ip  = (int *)malloc(atom->n_lines * sizeof(int));
            for (int line = 0; line < atom->n_lines; line++) {
                int ip = find_ion_pop_idx(atom, atom->line_atomic_number[line],
                                          atom->line_ion_number[line]);
                kp_ip[line] = ip; kp_glo[line] = -1; kp_gup[line] = -1;
                if (ip < 0) continue;
                int lev_base = atom->level_offset[ip];
                int lev_top  = atom->level_offset[ip + 1];
                for (int l = lev_base; l < lev_top; l++) {
                    if (atom->level_num[l] == atom->line_level_lower[line]) kp_glo[line] = l;
                    if (atom->level_num[l] == atom->line_level_upper[line]) kp_gup[line] = l;
                    if (kp_glo[line] >= 0 && kp_gup[line] >= 0) break;
                }
            }
            kp_n_lines_cached = atom->n_lines;
        }
        if (!opacity->p_kpacket)
            opacity->p_kpacket = (double *)calloc((size_t)n_levels * n_shells, sizeof(double));
        if (!opacity->kpacket_cdf)
            opacity->kpacket_cdf = (double *)calloc((size_t)n_shells * n_levels, sizeof(double));
    }
    /* Per-shell k-packet re-excitation weight accumulator (one level slot each). */
    double *kp_emiss = kpacket_mode ?
        (double *)malloc(n_levels * sizeof(double)) : NULL;

    for (int s = 0; s < n_shells; s++) {
        double W     = plasma->W[s];
        double T_rad = plasma->T_rad[s];
        double T_e   = plasma->T_e ? plasma->T_e[s] : T_rad;
        double n_e   = plasma->n_electron ? plasma->n_electron[s] :
                       (opacity->electron_density ? opacity->electron_density[s] : 0.0);
        double inv_sqrt_Te = (T_e > 0.0) ? 1.0 / sqrt(T_e) : 0.0;
        if (kpacket_mode) for (int j = 0; j < n_levels; j++) kp_emiss[j] = 0.0;

        for (int lev = 0; lev < n_levels; lev++) {
            int block_start = opacity->macro_block_references[lev];
            int block_end   = opacity->macro_block_references[lev + 1];
            if (block_start >= block_end) continue;

            /* Phase 1: Compute raw rates into temp buffer */
            double sum_rates = 0.0;
            double kp_deact  = 0.0;  /* collisional deactivation rate out of lev */

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
                        /* Emit-only λ-window scale (env-controlled, four windows).
                         * W1/W2/W3 use the global Z/ion mask (defaults Fe+Ni II).
                         * W4 has its own Z/ion mask so it can target iron-peak III
                         * for [3000,3100] bump suppress without altering W1/W2/W3. */
                        int Z   = atom->line_atomic_number[line_id];
                        int ion = atom->line_ion_number[line_id];
                        double lambda_A = 2.99792458e18 / atom->line_nu[line_id];
                        if ((uvopt_emit_boost != 1.0 || uvopt_emit_boost2 != 1.0 ||
                             uvopt_emit_boost3 != 1.0) &&
                            Z > 0 && Z < 32 && ion >= 0 && ion < 8 &&
                            (uvopt_ion_mask & (1u << ion)) && (uvopt_z_mask & (1u << Z))) {
                            if (uvopt_emit_boost != 1.0 &&
                                lambda_A >= uvopt_lam_min && lambda_A < uvopt_lam_max)
                                rate *= uvopt_emit_boost;
                            if (uvopt_emit_boost2 != 1.0 &&
                                lambda_A >= uvopt_lam_min2 && lambda_A < uvopt_lam_max2)
                                rate *= uvopt_emit_boost2;
                            if (uvopt_emit_boost3 != 1.0 &&
                                lambda_A >= uvopt_lam_min3 && lambda_A < uvopt_lam_max3)
                                rate *= uvopt_emit_boost3;
                        }
                        if (uvopt_emit_boost4 != 1.0 &&
                            Z > 0 && Z < 32 && ion >= 0 && ion < 8 &&
                            (uvopt_ion_mask4 & (1u << ion)) && (uvopt_z_mask4 & (1u << Z)) &&
                            lambda_A >= uvopt_lam_min4 && lambda_A < uvopt_lam_max4) {
                            rate *= uvopt_emit_boost4;
                        }
                    } else if (ttype == 0) {
                        /* Internal down: A_ul * (1 - beta_sobolev) */
                        rate = atom->line_A_ul[line_id] * (1.0 - beta);
                        /* P1: UV internal-down suppress (break UV→UV cascade) */
                        if (macro_uv_idown_factor != 1.0) {
                            int Z   = atom->line_atomic_number[line_id];
                            int ion = atom->line_ion_number[line_id];
                            double lambda_A = 2.99792458e18 / atom->line_nu[line_id];
                            if (lambda_A < macro_uv_idown_thresh &&
                                Z > 0 && Z < 32 && ion >= 0 && ion < 8 &&
                                (macro_uv_idown_z_mask & (1u << Z)) &&
                                (macro_uv_idown_ion_mask & (1u << ion))) {
                                rate *= macro_uv_idown_factor;
                            }
                        }
                    } else if (ttype == 1) {
                        /* Internal up: B_lu * J_nu (MC histogram or W*B_nu fallback) */
                        double nu_line = atom->line_nu[line_id];
                        if (use_j_nu) {
                            double J_line = nlte_get_J_at_nu(nlte, s, nu_line);
                            if (j_cap_factor > 0.0 || j_floor_factor > 0.0) {
                                double J_lte = W * planck_bnu(T_rad, nu_line);
                                if (j_cap_factor > 0.0) {
                                    double J_max = j_cap_factor * J_lte;
                                    if (J_line > J_max) J_line = J_max;
                                }
                                if (j_floor_factor > 0.0) {
                                    double J_min = j_floor_factor * J_lte;
                                    if (J_line < J_min) J_line = J_min;
                                }
                            }
                            rate = atom->line_B_lu[line_id] * J_line;
                        } else {
                            rate = atom->line_B_lu[line_id] * W * planck_bnu(T_rad, nu_line);
                        }
                    }

                    /* k-packet collisional channels (van Regemorter / Axelrod;
                     * exp(±dE/kTe) cancels in C_down so it needs only g_up). */
                    if (kpacket_mode && n_e > 0.0 && T_e > 0.0 &&
                        kp_glo[line_id] >= 0 && kp_gup[line_id] >= 0) {
                        int    glo = kp_glo[line_id], gup = kp_gup[line_id];
                        double g_lo = (double)atom->level_g[glo];
                        double g_up = (double)atom->level_g[gup];
                        double f_lu = atom->line_f_lu[line_id];
                        double dE   = H_PLANCK * atom->line_nu[line_id]; /* erg */
                        if (ttype == -1 && g_up > 0.0) {
                            /* collisional de-excitation rate (lev is upper level) */
                            double C_down = (f_lu > 1e-10)
                                ? VAN_REG_COEFF * n_e * f_lu * 0.2 * inv_sqrt_Te / g_up
                                : 8.63e-6 * n_e * AX_OMEGA * inv_sqrt_Te / g_up;
                            kp_deact += C_down;
                        } else if (ttype == 1 && g_lo > 0.0) {
                            /* k-packet re-excitation weight n_lower·C_up·dE,
                             * deposited at the upper (destination) level. The
                             * exp(-dE/kTe) inside C_up provides the thermal
                             * (red-peaked) weighting that pins re-emission to
                             * the local Planck peak. */
                            int    ip = kp_ip[line_id];
                            double n_lower = 0.0;
                            if (ip >= 0) {
                                double n_ion = atom->ion_number_density[(size_t)ip * n_shells + s];
                                double Zp    = atom->partition_functions[(size_t)ip * n_shells + s];
                                double boltz = atom->level_energy_eV[glo] * EV_TO_ERG /
                                               (K_BOLTZMANN * T_rad);
                                double wgt = atom->level_metastable[glo] ? 1.0 : W;
                                if (Zp > 0.0 && boltz < 500.0)
                                    n_lower = n_ion * wgt * g_lo * exp(-boltz) / Zp;
                            }
                            if (n_lower > 0.0) {
                                double exp_up = exp(-dE / (K_BOLTZMANN * T_e));
                                double C_up = (f_lu > 1e-10)
                                    ? VAN_REG_COEFF * n_e * f_lu * exp_up * 0.2 * inv_sqrt_Te / g_lo
                                    : 8.63e-6 * n_e * AX_OMEGA * exp_up * inv_sqrt_Te / g_lo;
                                double w = n_lower * C_up * dE;
                                int dst = opacity->destination_level_id[tid];
                                if (w > 0.0 && dst >= 0 && dst < n_levels)
                                    kp_emiss[dst] += w;
                            }
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

            /* k-packet deactivation probability for this level: collisional
             * deactivation competes with all radiative channels (sum_rates). */
            if (kpacket_mode) {
                double denom = sum_rates + kp_deact;
                opacity->p_kpacket[(size_t)lev * n_shells + s] =
                    (denom > 0.0) ? (kp_deact / denom) : 0.0;
            }
        }

        /* Build the per-shell k-packet re-excitation CDF (cumulative over
         * levels, contiguous per shell for GPU binary search). Normalized to
         * end at 1.0; a flat fallback if the shell has no collisional weight. */
        if (kpacket_mode) {
            double *cdf = opacity->kpacket_cdf + (size_t)s * n_levels;
            double tot = 0.0;
            for (int j = 0; j < n_levels; j++) tot += kp_emiss[j];
            if (tot > 0.0) {
                double acc = 0.0;
                for (int j = 0; j < n_levels; j++) {
                    acc += kp_emiss[j];
                    cdf[j] = acc / tot;
                }
                cdf[n_levels - 1] = 1.0;
            } else {
                /* no collisional weight: degenerate uniform (rarely hit; the
                 * GPU side only samples when a k-packet actually forms). */
                for (int j = 0; j < n_levels; j++)
                    cdf[j] = (double)(j + 1) / (double)n_levels;
            }
        }
    }

    free(rates_buf);
    free(kp_emiss);

    if (kpacket_mode) {
        /* Diagnostic: mean k-packet deactivation prob at inner/outer shell. */
        double pk0 = 0.0, pkN = 0.0; long n0 = 0, nN = 0;
        int sN = n_shells - 1;
        for (int lev = 0; lev < n_levels; lev++) {
            double a = opacity->p_kpacket[(size_t)lev * n_shells + 0];
            double b = opacity->p_kpacket[(size_t)lev * n_shells + sN];
            if (a > 0.0) { pk0 += a; n0++; }
            if (b > 0.0) { pkN += b; nN++; }
        }
        printf("  [KPACKET] mean p_kpacket: shell0=%.3e (%ld lev>0)  shell%d=%.3e (%ld lev>0)\n",
               n0 ? pk0 / n0 : 0.0, n0, sN, nN ? pkN / nN : 0.0, nN);
    }
    printf("  [TransProb] Recomputed %d transitions x %d shells (damping=%s, J_src=%s, j_cap=%.2g, j_floor=%.2g, W1=%.2g W2=%.2g W3=%.2g W4=%.2g[%g-%g] uv_idown=%.2g[<%.0fÅ])\n",
           n_trans, n_shells, apply_damping ? "on" : "off",
           use_j_nu ? "MC_histogram" : "W*Bnu", j_cap_factor, j_floor_factor,
           uvopt_emit_boost, uvopt_emit_boost2, uvopt_emit_boost3,
           uvopt_emit_boost4, uvopt_lam_min4, uvopt_lam_max4,
           macro_uv_idown_factor, macro_uv_idown_thresh);

    /* Diagnose at multiple shells to expose inner Fe II vs outer C/S/Si physics. */
    diag_macro_branch(atom, plasma, opacity, 0);
    if (n_shells > 3)  diag_macro_branch(atom, plasma, opacity, 3);
    if (n_shells > 10) diag_macro_branch(atom, plasma, opacity, n_shells / 3);
}

/* [MA-BRANCH] Effective branching diagnostic.
 * For each macro-atom level, find its strongest Sobolev line at `diag_shell`,
 * bin the level by that strong-line's band, and report aggregated branching
 * probabilities (p_emit / p_iup / p_idn) across ALL its transitions.
 *
 * UV-strong (band 0) levels get an extra breakdown:
 *   - destination band of BB-emit (where does UV resonance scatter land?)
 *   - per-(Z, ion) split (which species own UV-strong levels?)
 *   - first 2 Fe II UV-strong levels printed transition-by-transition.
 *
 * Reads `transition_probabilities[]` and `tau_sobolev[]` as stored — does NOT
 * recompute. Safe to call any time after compute_plasma_state(). */
void diag_macro_branch(AtomicData *atom, PlasmaState *plasma,
                       OpacityState *opacity, int diag_shell) {
    int n_shells = opacity->n_shells;
    int n_levels = opacity->n_macro_levels;
    (void)plasma;
    {
        const double tau_strong = 1.0;
        int n_per_band[MA_FATE_NBANDS] = {0};
        double sum_emit[MA_FATE_NBANDS] = {0};
        double sum_iup [MA_FATE_NBANDS] = {0};
        double sum_idn [MA_FATE_NBANDS] = {0};
        /* Band-resolved BB-emission destination, per strong-line band group.
         * sum_emit_dest[group][dest_band] = ∑ p_emit landing in dest_band */
        double sum_emit_dest[MA_FATE_NBANDS][MA_FATE_NBANDS] = {{0}};
        /* Per-(Z, ion) breakdown for UV-strong-line levels (band 0).
         * Tracks: n_levels, sum p_iup, sum p_idn, sum p_emit per dest band. */
        #define DIAG_ZMAX 31
        #define DIAG_IMAX 5
        int    ion_n   [DIAG_ZMAX][DIAG_IMAX] = {{0}};
        double ion_iup [DIAG_ZMAX][DIAG_IMAX] = {{0}};
        double ion_idn [DIAG_ZMAX][DIAG_IMAX] = {{0}};
        double ion_emit_dest[DIAG_ZMAX][DIAG_IMAX][MA_FATE_NBANDS] = {{{0}}};
        /* [3000,3100]Å bump sub-band BB-emit per-(Z, ion) tracker.
         * Independent of max_band — picks up ANY upper level that has at least
         * one BB-emit channel landing in the bump window, regardless of which
         * band its strongest line lives in. */
        double bump_emit_per_ion[DIAG_ZMAX][DIAG_IMAX] = {{0}};
        int    bump_lev_per_ion[DIAG_ZMAX][DIAG_IMAX] = {{0}};
        int n_idle = 0, n_weak = 0;
        int n_dumped_fe2_uv = 0;
        const int N_DUMP_FE2_UV = 2;
        const double C_LIGHT_AA = 2.99792458e18;  /* Hz·Å */

        for (int lev = 0; lev < n_levels; lev++) {
            int block_start = opacity->macro_block_references[lev];
            int block_end   = opacity->macro_block_references[lev + 1];
            if (block_start >= block_end) { n_idle++; continue; }

            /* Find the strongest line touching this level (any ttype). */
            double tau_max  = 0.0;
            int    max_band = -1;
            for (int tid = block_start; tid < block_end; tid++) {
                int line_id = opacity->transition_line_id[tid];
                if (line_id < 0 || line_id >= atom->n_lines) continue;
                double tau = opacity->tau_sobolev[line_id * n_shells + diag_shell];
                if (tau > tau_max) {
                    tau_max  = tau;
                    max_band = macro_atom_fate_band_from_nu(atom->line_nu[line_id]);
                }
            }
            if (tau_max < tau_strong || max_band < 0) { n_weak++; continue; }

            double p_emit_band[MA_FATE_NBANDS] = {0};
            double p_iup = 0.0, p_idn = 0.0;
            double level_bump_p_emit = 0.0;  /* BB-emit weight in [3000,3100]Å */
            /* Element of this level: take from strongest line. */
            int strong_line_id = -1;
            for (int tid = block_start; tid < block_end; tid++) {
                int line_id = opacity->transition_line_id[tid];
                if (line_id < 0 || line_id >= atom->n_lines) continue;
                double tau = opacity->tau_sobolev[line_id * n_shells + diag_shell];
                if (tau == tau_max) { strong_line_id = line_id; break; }
            }
            int level_Z   = strong_line_id >= 0 ? atom->line_atomic_number[strong_line_id] : -1;
            int level_ion = strong_line_id >= 0 ? atom->line_ion_number[strong_line_id]    : -1;

            for (int tid = block_start; tid < block_end; tid++) {
                double p    = opacity->transition_probabilities[tid * n_shells + diag_shell];
                int    ttype  = opacity->transition_type[tid];
                int    line_id = opacity->transition_line_id[tid];
                if (ttype == -1) {
                    if (line_id >= 0 && line_id < atom->n_lines) {
                        int b = macro_atom_fate_band_from_nu(atom->line_nu[line_id]);
                        p_emit_band[b] += p;
                        double lam_em = C_LIGHT_AA / atom->line_nu[line_id];
                        if (lam_em >= 3000.0 && lam_em <= 3100.0)
                            level_bump_p_emit += p;
                    }
                } else if (ttype == 0) {
                    p_idn += p;
                } else if (ttype == 1) {
                    p_iup += p;
                }
            }

            double p_emit_tot = 0.0;
            for (int b = 0; b < MA_FATE_NBANDS; b++) p_emit_tot += p_emit_band[b];
            n_per_band[max_band]++;
            sum_emit[max_band] += p_emit_tot;
            sum_iup [max_band] += p_iup;
            sum_idn [max_band] += p_idn;
            for (int b = 0; b < MA_FATE_NBANDS; b++)
                sum_emit_dest[max_band][b] += p_emit_band[b];

            /* UV-strong levels: track per-(Z, ion) breakdown */
            if (max_band == 0 && level_Z >= 0 && level_Z < DIAG_ZMAX
                && level_ion >= 0 && level_ion < DIAG_IMAX) {
                ion_n  [level_Z][level_ion]++;
                ion_iup[level_Z][level_ion] += p_iup;
                ion_idn[level_Z][level_ion] += p_idn;
                for (int b = 0; b < MA_FATE_NBANDS; b++)
                    ion_emit_dest[level_Z][level_ion][b] += p_emit_band[b];
            }
            /* [3000,3100]Å bump per-(Z, ion): any level with non-zero bump emit */
            if (level_bump_p_emit > 0.0
                && level_Z >= 0 && level_Z < DIAG_ZMAX
                && level_ion >= 0 && level_ion < DIAG_IMAX) {
                bump_emit_per_ion[level_Z][level_ion] += level_bump_p_emit;
                bump_lev_per_ion[level_Z][level_ion]++;
            }

            /* Dump first N_DUMP Fe II UV-strong levels that ALSO have BB-emission
             * exits (skip ground-multiplet members with only I-UP). These are
             * the cascade-relevant upper levels of UV resonance lines. */
            int has_emit = (p_emit_band[0] + p_emit_band[1] + p_emit_band[2]
                            + p_emit_band[3]) > 0.0;
            if (max_band == 0 && level_Z == 26 && level_ion == 1 && has_emit &&
                n_dumped_fe2_uv < N_DUMP_FE2_UV) {
                printf("    [MA-DUMP] Fe II UV-strong level lev=%d (Z=%d ion=%d) "
                       "tau_max=%.2e at shell %d, block size=%d\n",
                       lev, level_Z, level_ion, tau_max, diag_shell,
                       block_end - block_start);
                printf("      tid  ttype  line_id    lambda(A)     A_ul[s^-1]    "
                       "tau_sob       p\n");
                for (int tid = block_start; tid < block_end; tid++) {
                    double p     = opacity->transition_probabilities[tid * n_shells + diag_shell];
                    int    ttype = opacity->transition_type[tid];
                    int    lid   = opacity->transition_line_id[tid];
                    double lam = 0.0, A_ul = 0.0, tau = 0.0;
                    if (lid >= 0 && lid < atom->n_lines) {
                        lam   = C_LIGHT_AA / atom->line_nu[lid];
                        A_ul  = atom->line_A_ul[lid];
                        tau   = opacity->tau_sobolev[lid * n_shells + diag_shell];
                    }
                    const char *ts =
                        ttype == -1 ? "EMIT  " :
                        ttype ==  0 ? "I-DN  " :
                        ttype ==  1 ? "I-UP  " : "?     ";
                    printf("      %4d %s %7d  %10.2f   %10.3e   %10.3e  %10.3e\n",
                           tid, ts, lid, lam, A_ul, tau, p);
                }
                n_dumped_fe2_uv++;
            }
        }

        static const char *band_name[MA_FATE_NBANDS] = {
            "UVblnk", "CaIIKb", "UVtgt ", "fluor ",
            "green ", "red   ", "NIR1  ", "NIR2  "};
        printf("  [MA-BRANCH] Strong-line (tau > %.1f) branching at shell %d\n",
               tau_strong, diag_shell);
        printf("    bin by strongest-line band  |    n  | <p_emit> | <p_iup> | <p_idn>\n");
        for (int b = 0; b < MA_FATE_NBANDS; b++) {
            if (n_per_band[b] == 0) continue;
            double inv = 1.0 / (double)n_per_band[b];
            printf("    strong-line in %s        | %5d |  %6.4f  | %6.4f  | %6.4f\n",
                   band_name[b], n_per_band[b],
                   sum_emit[b] * inv, sum_iup[b] * inv, sum_idn[b] * inv);
        }
        printf("    weak-line (tau<=%.1f, skipped) | %5d |\n", tau_strong, n_weak);
        if (n_idle) printf("    orphan/empty block          | %5d |\n", n_idle);

        /* BB-emission destination band, normalized by total UV-strong p_emit. */
        if (n_per_band[0] > 0) {
            double s = sum_emit[0]; if (s <= 0) s = 1.0;
            printf("    UV-strong levels' BB-emit destination band (%% of p_emit):\n     ");
            for (int b = 0; b < MA_FATE_NBANDS; b++)
                printf(" %s=%5.1f%%", band_name[b], 100.0 * sum_emit_dest[0][b] / s);
            printf("\n");
            double up = sum_iup[0] / (double)n_per_band[0];
            double dn = sum_idn[0] / (double)n_per_band[0];
            printf("    UV-strong-group: <p_iup>/<p_idn>=%.2f"
                   "  (Mazzali-Lucy expects <p_idn> >> <p_iup>;"
                   " ratio>=1 => J_nu(UV) pumping the cascade up).\n",
                   dn > 0 ? up / dn : -1.0);
        }

        /* Per-(Z, ion) breakdown for UV-strong-line levels.
         * Rank ions by n_levels; print top 6 with their fluor/UV/iup/idn mass. */
        {
            int idx_Z[DIAG_ZMAX * DIAG_IMAX], idx_I[DIAG_ZMAX * DIAG_IMAX];
            int n_ions = 0;
            for (int Z = 1; Z < DIAG_ZMAX; Z++)
                for (int I = 0; I < DIAG_IMAX; I++)
                    if (ion_n[Z][I] > 0) {
                        idx_Z[n_ions] = Z; idx_I[n_ions] = I; n_ions++;
                    }
            for (int i = 0; i < n_ions; i++)
                for (int j = i + 1; j < n_ions; j++)
                    if (ion_n[idx_Z[j]][idx_I[j]] > ion_n[idx_Z[i]][idx_I[i]]) {
                        int tz = idx_Z[i]; idx_Z[i] = idx_Z[j]; idx_Z[j] = tz;
                        int ti = idx_I[i]; idx_I[i] = idx_I[j]; idx_I[j] = ti;
                    }
            int top = n_ions < 6 ? n_ions : 6;
            if (top > 0) {
                printf("    UV-strong levels by ion (top %d by count):\n", top);
                printf("       ion       n  | <p_iup> <p_idn> | UV%%  CaK%%  Utg%%  flu%%  grn%%  red%% NIR1%% NIR2%%\n");
                for (int k = 0; k < top; k++) {
                    int Z = idx_Z[k], I = idx_I[k];
                    int n = ion_n[Z][I]; if (n <= 0) continue;
                    double inv = 1.0 / (double)n;
                    double esum = 0.0;
                    for (int b = 0; b < MA_FATE_NBANDS; b++) esum += ion_emit_dest[Z][I][b];
                    if (esum <= 0.0) esum = 1.0;
                    /* Roman numeral for ion stage. */
                    static const char *rom[5] = {"I  ", "II ", "III", "IV ", "V  "};
                    printf("      Z=%2d %s %5d | %6.4f  %6.4f | "
                           "%4.1f %4.1f %4.1f %4.1f %4.1f %4.1f %4.1f %4.1f\n",
                           Z, rom[I], n,
                           ion_iup[Z][I] * inv, ion_idn[Z][I] * inv,
                           100.0 * ion_emit_dest[Z][I][0] / esum,
                           100.0 * ion_emit_dest[Z][I][1] / esum,
                           100.0 * ion_emit_dest[Z][I][2] / esum,
                           100.0 * ion_emit_dest[Z][I][3] / esum,
                           100.0 * ion_emit_dest[Z][I][4] / esum,
                           100.0 * ion_emit_dest[Z][I][5] / esum,
                           100.0 * ion_emit_dest[Z][I][6] / esum,
                           100.0 * ion_emit_dest[Z][I][7] / esum);
                }
            }
        }

        /* [3000,3100]Å bump BB-emit per-(Z, ion) — identifies which iron-peak
         * species generates the residual bump in champion 152761. */
        {
            double bump_total = 0.0;
            for (int Z = 1; Z < DIAG_ZMAX; Z++)
                for (int I = 0; I < DIAG_IMAX; I++)
                    bump_total += bump_emit_per_ion[Z][I];
            if (bump_total > 0.0) {
                int idx_Z[DIAG_ZMAX * DIAG_IMAX], idx_I[DIAG_ZMAX * DIAG_IMAX];
                int n_bump = 0;
                for (int Z = 1; Z < DIAG_ZMAX; Z++)
                    for (int I = 0; I < DIAG_IMAX; I++)
                        if (bump_emit_per_ion[Z][I] > 0.0) {
                            idx_Z[n_bump] = Z; idx_I[n_bump] = I; n_bump++;
                        }
                for (int i = 0; i < n_bump; i++)
                    for (int j = i + 1; j < n_bump; j++)
                        if (bump_emit_per_ion[idx_Z[j]][idx_I[j]] >
                            bump_emit_per_ion[idx_Z[i]][idx_I[i]]) {
                            int tz = idx_Z[i]; idx_Z[i] = idx_Z[j]; idx_Z[j] = tz;
                            int ti = idx_I[i]; idx_I[i] = idx_I[j]; idx_I[j] = ti;
                        }
                int top = n_bump < 10 ? n_bump : 10;
                static const char *rom[5] = {"I  ", "II ", "III", "IV ", "V  "};
                printf("    [3000,3100]Å BUMP BB-emit by ion (top %d, total p=%.4f):\n",
                       top, bump_total);
                printf("       ion    n_lev  | sum_p_emit |  %%total\n");
                for (int k = 0; k < top; k++) {
                    int Z = idx_Z[k], I = idx_I[k];
                    printf("      Z=%2d %s   %5d  | %10.4f | %5.1f%%\n",
                           Z, rom[I], bump_lev_per_ion[Z][I],
                           bump_emit_per_ion[Z][I],
                           100.0 * bump_emit_per_ion[Z][I] / bump_total);
                }
            } else {
                printf("    [3000,3100]Å BUMP BB-emit: no levels with bump emit channels\n");
            }
        }
        #undef DIAG_ZMAX
        #undef DIAG_IMAX
    }
}

/* ---- Task #7: Frozen-in recombination freeze-out (Chugai/Potashov single-epoch) ----
 * The outer ejecta of a young SN Ia carries a fossil ionization plateau (CMFGEN
 * DDC15 0.976d: outer <Z>~0.53 at ~2500 K) that NO steady-state ionization can
 * produce: recombination has frozen out because tau_rec/t_exp grows as t^2
 * (Dessart & Hillier 2008) and reaches >300x in the outermost shells. The cure
 * is the missing TIME-DEPENDENT physics, not a steady-state lever.
 *
 * Per shell we (i) build the faithful per-ion radiative recombination coefficient
 * alpha_rr(Z,i,T_e) as the Milne integral of the CMFGEN photoionization cross
 * sections (atom->cmfgen_sigma_bf) against the Planck function B(T_e) -- the same
 * integral the NLTE assembly does against J_nu, but against B (LTE recomb). NOT a
 * hydrogenic approximation. (ii) Find the parameter-free freeze-out epoch
 *   t_0 = sqrt(alpha_bar * n_e(t_exp) * t_exp^3)   [criterion tau_rec(t_0)=t_0].
 * If t_0 >= t_exp the shell never decouples -> leave the steady-state solution
 * (NLTE already matches the inner region). Otherwise (iii) integrate the
 * recombination cascade in ion-fraction space
 *   dy_{e,k}/dt = a_{e,k} n_e(t) y_{e,k+1} - a_{e,k-1} n_e(t) y_{e,k}
 * over the homologous density history n_e(t) ~ t^-3 from t_0 (seeded at the
 * equilibrium ~singly-ionized partition) to t_exp, with a self-consistent n_e.
 *
 * Validated in Python before this port (scripts/frozen_in_milne_prototype.py):
 * reproduces the outer <Z>~0.53 plateau (0.600) with ZERO fitting and resolves
 * the per-element split (O stays II; Si/Fe recombine toward neutral).
 *
 * Gated LUMINA_FROZENIN=1 (default off -> byte-identical). Needs cmfgen sigma_bf.
 * Pairs with LUMINA_NLTE_ION_LOCK so the downstream NLTE solve preserves the
 * frozen per-ion totals (otherwise NLTE re-solves ion balance and undoes it). */
#define FROZENIN_MAXSTAGE 4

static int frozenin_active(void) {
    const char *e = getenv("LUMINA_FROZENIN");
    return (e && e[0] == '1');
}

/* Milne radiative-recombination coefficient [cm^3/s] producing ion-pop `ip`
 * (charge stage k) by recombination from stage k+1. `ip_next` (charge k+1)
 * supplies the recombining-ion ground statistical weight. T = T_e. */
static double frozenin_alpha_rr(AtomicData *atom, int ip, int ip_next, double T) {
    if (!atom->cmfgen_loaded || T <= 0.0) return 0.0;
    int Z = atom->ion_pop_Z[ip];
    int stage = atom->ion_pop_stage[ip];
    double chi_ion_eV = find_ioniz_energy(atom, Z, stage);
    if (chi_ion_eV <= 0.0) return 0.0;
    double chi_ion_erg = chi_ion_eV * EV_TO_ERG;

    int g_ion = 1;
    if (ip_next >= 0) {
        int gnd = atom->level_offset[ip_next];
        g_ion = atom->level_g[gnd];
        if (g_ion < 1) g_ion = 1;
    }
    double lam3 = pow(H_PLANCK * H_PLANCK /
                      (2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T), 1.5);
    int nfreq = atom->cmfgen_n_freq_bins;
    double log_numin = log(atom->cmfgen_nu_min);
    double d_log_nu = (log(atom->cmfgen_nu_max) - log_numin) / nfreq;

    int g0 = atom->level_offset[ip];
    int g1 = atom->level_offset[ip + 1];
    double a_tot = 0.0;
    for (int gl = g0; gl < g1; gl++) {
        if (!atom->cmfgen_has_sigma[gl]) continue;
        double E_l_erg = atom->level_energy_eV[gl] * EV_TO_ERG;
        double chi_l = chi_ion_erg - E_l_erg;       /* binding energy of level l */
        if (chi_l <= 0.0) continue;
        double nu_th = chi_l / H_PLANCK;
        const double *sigma_row = &atom->cmfgen_sigma_bf[(size_t)gl * (size_t)nfreq];
        double Rbf = 0.0;
        for (int bb = 0; bb < nfreq; bb++) {
            double log_nu_lo = log_numin + bb * d_log_nu;
            double nu_c = exp(log_nu_lo + 0.5 * d_log_nu);
            if (nu_c < nu_th) continue;
            double sig = sigma_row[bb];
            if (sig <= 0.0) continue;
            double x = H_PLANCK * nu_c / (K_BOLTZMANN * T);
            if (x > 700.0) continue;
            double dnu = exp(log_nu_lo + d_log_nu) - exp(log_nu_lo);
            double B = (2.0 * H_PLANCK * nu_c * nu_c * nu_c /
                        (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT)) / expm1(x);
            Rbf += 4.0 * M_PI_VAL * B * sig / (H_PLANCK * nu_c) * dnu;
        }
        a_tot += Rbf * lam3 * (double)atom->level_g[gl] / (2.0 * (double)g_ion)
                 * exp(chi_l / (K_BOLTZMANN * T));
    }
    return a_tot;
}

/* dy/dt for the homologous recombination cascade (fraction space). State y is
 * [nelem * FROZENIN_MAXSTAGE], indexed [e*MS + k] with k = absolute charge.
 * n_e(t) = (t_exp/t)^3 * sum_e n_elem0[e] * <Z>_e(t) (self-consistent). */
static void frozenin_deriv(int nelem, const double *alpha, const double *n_elem0,
                           double t_exp, double t, const double *y, double *dydt) {
    const int MS = FROZENIN_MAXSTAGE;
    double scale = (t_exp / t) * (t_exp / t) * (t_exp / t);
    double ne = 0.0;
    for (int e = 0; e < nelem; e++) {
        double zbar = 0.0;
        for (int k = 0; k < MS; k++) zbar += k * y[e * MS + k];
        ne += n_elem0[e] * zbar;
    }
    ne *= scale;
    for (int e = 0; e < nelem; e++) {
        for (int k = 0; k < MS; k++) {
            double inflow  = (k + 1 < MS) ? alpha[e * MS + k] * ne * y[e * MS + k + 1] : 0.0;
            double outflow = (k - 1 >= 0) ? alpha[e * MS + k - 1] * ne * y[e * MS + k] : 0.0;
            dydt[e * MS + k] = inflow - outflow;
        }
    }
}

/* Integrate the cascade from t0 to t_exp with RK4 on a log-spaced time grid
 * (density ~ t^-3 -> log nodes are natural; coupling is order-unity at t0 by the
 * freeze-out criterion, so the system is non-stiff over [t0, t_exp]). */
static void frozenin_integrate(int nelem, const double *alpha, const double *n_elem0,
                               double t0, double t_exp, double *y) {
    const int MS = FROZENIN_MAXSTAGE;
    int n = nelem * MS;
    const int NT = 400;
    double k1[FROZENIN_MAXSTAGE * 64], k2[FROZENIN_MAXSTAGE * 64];
    double k3[FROZENIN_MAXSTAGE * 64], k4[FROZENIN_MAXSTAGE * 64];
    double ytmp[FROZENIN_MAXSTAGE * 64];
    double ratio = pow(t_exp / t0, 1.0 / NT);
    double t = t0;
    for (int step = 0; step < NT; step++) {
        double tn = (step == NT - 1) ? t_exp : t * ratio;
        double H = tn - t;
        /* The self-consistent n_e ~ na*(t_exp/t)^3 is enormous near t0 (t0<<t_exp),
         * so the recomb rate alpha*n_e can give alpha*n_e*H >> 1 in the first
         * macro-steps. Fixed explicit RK4 blows up there. Substep so the local
         * stiffness (max recomb rate * substep) stays small -> stable. */
        double ne_now = 0.0;
        for (int e = 0; e < nelem; e++) {
            double zbar = 0.0;
            for (int k = 0; k < MS; k++) zbar += k * y[e * MS + k];
            ne_now += n_elem0[e] * zbar;
        }
        ne_now *= (t_exp / t) * (t_exp / t) * (t_exp / t);
        double amax = 0.0;
        for (int i = 0; i < n; i++) if (alpha[i] > amax) amax = alpha[i];
        double rate = amax * ne_now;
        int nsub = (int)(rate * H / 0.05) + 1;
        if (nsub < 1) nsub = 1;
        if (nsub > 100000) nsub = 100000;
        double h = H / nsub;
        for (int sub = 0; sub < nsub; sub++) {
            double ts = t + sub * h;
            frozenin_deriv(nelem, alpha, n_elem0, t_exp, ts, y, k1);
            for (int i = 0; i < n; i++) ytmp[i] = y[i] + 0.5 * h * k1[i];
            frozenin_deriv(nelem, alpha, n_elem0, t_exp, ts + 0.5 * h, ytmp, k2);
            for (int i = 0; i < n; i++) ytmp[i] = y[i] + 0.5 * h * k2[i];
            frozenin_deriv(nelem, alpha, n_elem0, t_exp, ts + 0.5 * h, ytmp, k3);
            for (int i = 0; i < n; i++) ytmp[i] = y[i] + h * k3[i];
            frozenin_deriv(nelem, alpha, n_elem0, t_exp, ts + h, ytmp, k4);
            for (int i = 0; i < n; i++)
                y[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t = tn;
    }
    /* clip + renormalize per element (number conservation) */
    for (int e = 0; e < nelem; e++) {
        double sum = 0.0;
        for (int k = 0; k < MS; k++) {
            if (y[e * MS + k] < 0.0) y[e * MS + k] = 0.0;
            sum += y[e * MS + k];
        }
        if (sum > 0.0)
            for (int k = 0; k < MS; k++) y[e * MS + k] /= sum;
    }
}

/* Replace the steady-state ion partition with the frozen-in freeze-out cascade
 * for shells whose t_0 < t_exp. Overwrites atom->ion_number_density and
 * plasma->n_electron for those shells only; inner/unfrozen shells untouched. */
/* shells frozen by apply_frozenin_freezeout (set here, read by the coupled Newton
 * so the time-dependent block and the frozen-in cascade partition the grid
 * cleanly with no gap and no double-write). */
static unsigned char *frozenin_is_frozen = NULL;
static int frozenin_is_frozen_n = 0;

static void apply_frozenin_freezeout(AtomicData *atom, PlasmaState *plasma,
                                     int n_shells, double time_explosion) {
    if (!frozenin_active()) return;
    if (!atom->cmfgen_loaded) {
        printf("  [FROZENIN][WARN] cmfgen sigma_bf not loaded; skipping (Milne alpha needs it)\n");
        return;
    }
    {
        /* The frozen per-ion totals survive the downstream NLTE solve only if a
         * per-ion lock pins each ion's level sum to ion_number_density. Either
         * gate does that (plasma.c:4438). PER_ION_RESCALE is preferred: it locks
         * the totals WITHOUT the transport-only plasma freeze that ION_LOCK
         * triggers (cuda.cu:3188), which would skip this very function. */
        const char *lk = getenv("LUMINA_NLTE_ION_LOCK");
        const char *rs = getenv("LUMINA_NLTE_PER_ION_RESCALE");
        int locked = (lk && lk[0] == '1') || (rs && rs[0] == '1');
        if (!locked)
            printf("  [FROZENIN][WARN] neither LUMINA_NLTE_PER_ION_RESCALE nor "
                   "LUMINA_NLTE_ION_LOCK set; downstream NLTE may re-solve ion "
                   "balance and undo the frozen partition\n");
    }

    const int MS = FROZENIN_MAXSTAGE;
    double t_exp = time_explosion;
    int nelem = atom->n_elements;
    if (nelem > 64) {
        printf("  [FROZENIN][WARN] n_elements=%d exceeds scratch bound 64; skipping\n", nelem);
        return;
    }

    double *n_elem0  = (double *)malloc((size_t)nelem * sizeof(double));
    double *f_frac   = (double *)malloc((size_t)nelem * sizeof(double));
    double *a_rep    = (double *)malloc((size_t)nelem * sizeof(double));
    double *alpha    = (double *)malloc((size_t)nelem * MS * sizeof(double));
    double *y        = (double *)malloc((size_t)nelem * MS * sizeof(double));
    int    *topstage = (int *)malloc((size_t)nelem * sizeof(int));

    /* (re)allocate the per-shell frozen flag for the coupled-Newton partition */
    if (frozenin_is_frozen_n != n_shells) {
        free(frozenin_is_frozen);
        frozenin_is_frozen = (unsigned char *)calloc((size_t)n_shells, 1);
        frozenin_is_frozen_n = frozenin_is_frozen ? n_shells : 0;
    }
    if (frozenin_is_frozen) memset(frozenin_is_frozen, 0, (size_t)n_shells);

    int n_frozen = 0;
    for (int s = 0; s < n_shells; s++) {
        double T_e = plasma->T_e[s];
        double rho = plasma->rho[s];
        if (!(T_e > 0.0) || !(rho > 0.0)) continue;

        /* element number densities at t_exp + per-(element,stage) Milne alpha */
        double n_atom_tot = 0.0;
        for (int e = 0; e < nelem; e++) {
            double abund = atom->abundances[e * n_shells + s];
            double mass_amu = atom->element_mass_amu[e];
            n_elem0[e] = (abund * rho) / (mass_amu * AMU);
            n_atom_tot += n_elem0[e];

            int ip0 = atom->elem_ion_offset[e];
            int ip1 = atom->elem_ion_offset[e + 1];
            int tops = 0;
            for (int k = 0; k < MS; k++) alpha[e * MS + k] = 0.0;
            for (int ip = ip0; ip < ip1; ip++) {
                int stage = atom->ion_pop_stage[ip];
                if (stage >= 0 && stage < MS && stage > tops) tops = stage;
                if (stage < 0 || stage >= MS - 1) continue; /* need stage+1 to recomb in */
                int ip_next = (ip + 1 < ip1 &&
                               atom->ion_pop_stage[ip + 1] == stage + 1) ? ip + 1 : -1;
                if (ip_next < 0) continue;
                alpha[e * MS + stage] = frozenin_alpha_rr(atom, ip, ip_next, T_e);
            }
            topstage[e] = (tops < MS) ? tops : MS - 1;
            /* representative alpha for t_0 = recomb producing the lowest stage */
            a_rep[e] = (alpha[e * MS + 0] > 0.0) ? alpha[e * MS + 0]
                       : 2.6e-13 * pow(T_e / 1e4, -0.8);
        }
        if (n_atom_tot <= 0.0) continue;
        for (int e = 0; e < nelem; e++) f_frac[e] = n_elem0[e] / n_atom_tot;

        /* self-consistent freeze: t_0(n_e) and cascade n_e converge together */
        double ne_sc = plasma->n_electron[s];
        if (!(ne_sc > 0.0)) ne_sc = 1.0;
        int froze = 0;
        double new_ne = ne_sc;
        for (int it = 0; it < 16; it++) {
            double a_bar = 0.0;
            for (int e = 0; e < nelem; e++) a_bar += f_frac[e] * a_rep[e];
            double t0 = sqrt(a_bar * ne_sc * t_exp * t_exp * t_exp);
            if (!(t0 < t_exp)) { froze = 0; break; }   /* never decouples -> steady-state */
            froze = 1;

            for (int e = 0; e < nelem; e++) {
                for (int k = 0; k < MS; k++) y[e * MS + k] = 0.0;
                int ks = (topstage[e] < 1) ? topstage[e] : 1;  /* seed ~singly ionized */
                y[e * MS + ks] = 1.0;
            }
            frozenin_integrate(nelem, alpha, n_elem0, t0, t_exp, y);

            new_ne = 0.0;
            for (int e = 0; e < nelem; e++) {
                double zbar = 0.0;
                for (int k = 0; k < MS; k++) zbar += k * y[e * MS + k];
                new_ne += n_elem0[e] * zbar;
            }
            if (!(new_ne > 0.0)) new_ne = 1.0;
            double drel = fabs(new_ne - ne_sc) / (ne_sc + 1.0);
            ne_sc = 0.5 * ne_sc + 0.5 * new_ne;
            if (drel < 1e-3) break;
        }
        if (!froze) continue;

        /* write frozen ion partition + electron density for this shell */
        for (int e = 0; e < nelem; e++) {
            int ip0 = atom->elem_ion_offset[e];
            int ip1 = atom->elem_ion_offset[e + 1];
            for (int ip = ip0; ip < ip1; ip++) {
                int stage = atom->ion_pop_stage[ip];
                double n_ion = (stage >= 0 && stage < MS)
                               ? y[e * MS + stage] * n_elem0[e] : 0.0;
                if (n_ion < 1e-300) n_ion = 1e-300;
                atom->ion_number_density[ip * n_shells + s] = n_ion;
            }
        }
        plasma->n_electron[s] = new_ne;
        if (frozenin_is_frozen) frozenin_is_frozen[s] = 1;
        n_frozen++;
    }

    printf("  [FROZENIN] freeze-out applied: %d/%d shells frozen (t_0<t_exp), "
           "rest steady-state\n", n_frozen, n_shells);

    free(n_elem0); free(f_frac); free(a_rep);
    free(alpha); free(y); free(topstage);
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

    /* Task #7: frozen-in recombination freeze-out (gated LUMINA_FROZENIN).
     * Overrides outer-shell ion populations + n_e with the time-dependent
     * cascade; no-op (byte-identical) when off. */
    apply_frozenin_freezeout(atom, plasma, n_shells, time_explosion);

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
    /* [TAU-DIAG] band-resolved counts (shell 0). lambda_A = c / nu in Å. */
    int n_uv = 0, n_uv_sig = 0;          /* 1700 - 3000 Å */
    int n_blue = 0, n_blue_sig = 0;      /* 3000 - 4500 Å */
    int n_opt = 0, n_opt_sig = 0;        /* 4500 - 7000 Å */
    double tau_sum_uv = 0.0, tau_sum_blue = 0.0, tau_sum_opt = 0.0;
    for (int l = 0; l < n_lines; l++) {
        double t = opacity->tau_sobolev[l * n_shells + 0]; /* shell 0 */
        if (t > tau_max) tau_max = t;
        if (t < tau_min && t > 1e-100) tau_min = t;
        if (t > 1.0) n_significant++;
        double nu = atom->line_nu[l];
        double lam_A = (nu > 0.0) ? (C_SPEED_OF_LIGHT / nu * 1e8) : 0.0;
        if (lam_A >= 1700.0 && lam_A < 3000.0)      { n_uv++;   if (t > 1.0) n_uv_sig++;   tau_sum_uv += t; }
        else if (lam_A >= 3000.0 && lam_A < 4500.0) { n_blue++; if (t > 1.0) n_blue_sig++; tau_sum_blue += t; }
        else if (lam_A >= 4500.0 && lam_A < 7000.0) { n_opt++;  if (t > 1.0) n_opt_sig++;  tau_sum_opt += t; }
    }
    printf("    Shell 0: tau_min=%.2e, tau_max=%.2e, lines with tau>1: %d/%d\n",
           tau_min, tau_max, n_significant, n_lines);
    printf("    [TAU-DIAG] tau>1 by band: UV(1700-3000)=%d/%d (sum=%.1f) | "
           "blue(3000-4500)=%d/%d (sum=%.1f) | opt(4500-7000)=%d/%d (sum=%.1f)\n",
           n_uv_sig, n_uv, tau_sum_uv,
           n_blue_sig, n_blue, tau_sum_blue,
           n_opt_sig, n_opt, tau_sum_opt);

    /* [TAU-BY-ION] per-(Z,ion) shell-0 line-opacity decomposition.
       Identifies which iron-peak ions build the "iron curtain" that drives
       reabsorption / T_inner overshoot. Gated: LUMINA_TAU_BY_ION=1.
       LUMINA_TAU_BY_ION_SHELL=k selects shell k (default 0, the inner edge). */
    if (getenv("LUMINA_TAU_BY_ION") && atoi(getenv("LUMINA_TAU_BY_ION")) > 0) {
        const int ZMAX = 31, IMAX = 16;
        int sh = 0;
        if (getenv("LUMINA_TAU_BY_ION_SHELL"))
            sh = atoi(getenv("LUMINA_TAU_BY_ION_SHELL"));
        if (sh < 0) sh = 0;
        if (sh >= n_shells) sh = n_shells - 1;
        /* Wavelength window (rest-frame Å); default broad-band [1700,10000].
           Narrow it (e.g. 4050/4150) to attribute a specific feature. */
        double lam_lo = 1700.0, lam_hi = 10000.0;
        if (getenv("LUMINA_TAU_BY_ION_LAM_LO"))
            lam_lo = atof(getenv("LUMINA_TAU_BY_ION_LAM_LO"));
        if (getenv("LUMINA_TAU_BY_ION_LAM_HI"))
            lam_hi = atof(getenv("LUMINA_TAU_BY_ION_LAM_HI"));
        static double tau_zi[31 * 16];   /* sum of tau over window */
        static int    cnt_zi[31 * 16];   /* count of tau>1 lines */
        memset(tau_zi, 0, sizeof(tau_zi));
        memset(cnt_zi, 0, sizeof(cnt_zi));
        double tau_tot = 0.0;
        for (int l = 0; l < n_lines; l++) {
            double nu = atom->line_nu[l];
            double lam_A = (nu > 0.0) ? (C_SPEED_OF_LIGHT / nu * 1e8) : 0.0;
            if (lam_A < lam_lo || lam_A >= lam_hi) continue;
            double t = opacity->tau_sobolev[(size_t)l * n_shells + sh];
            int Z = atom->line_atomic_number[l];
            int ion = atom->line_ion_number[l];
            if (Z < 0 || Z >= ZMAX || ion < 0 || ion >= IMAX) continue;
            tau_zi[Z * IMAX + ion] += t;
            if (t > 1.0) cnt_zi[Z * IMAX + ion]++;
            tau_tot += t;
        }
        /* print top contributors by tau-sum */
        printf("    [TAU-BY-ION] shell %d (tau_tot[%.0f-%.0fA]=%.3e) top ions:\n",
               sh, lam_lo, lam_hi, tau_tot);
        for (int rank = 0; rank < 15; rank++) {
            int best = -1; double bestv = 0.0;
            for (int k = 0; k < ZMAX * IMAX; k++)
                if (tau_zi[k] > bestv) { bestv = tau_zi[k]; best = k; }
            if (best < 0 || bestv <= 0.0) break;
            int Z = best / IMAX, ion = best % IMAX;
            printf("      Z=%2d ion=%2d  tau_sum=%.3e (%.1f%%)  N(tau>1)=%d\n",
                   Z, ion, bestv, 100.0 * bestv / (tau_tot > 0 ? tau_tot : 1.0),
                   cnt_zi[best]);
            tau_zi[best] = 0.0;  /* consume */
        }
    }

    /* [ION-FRAC] per-shell ionization fractions for selected elements.
       Tests whether a carrier ion (e.g. Co II) is over-populated due to
       under-ionization. Gated: LUMINA_ION_FRAC_DUMP=1.
       LUMINA_ION_FRAC_Z="27,26,28,20" picks elements (default Co,Fe,Ni,Ca). */
    if (getenv("LUMINA_ION_FRAC_DUMP") && atoi(getenv("LUMINA_ION_FRAC_DUMP")) > 0) {
        int z_list[8] = {27, 26, 28, 20, 0, 0, 0, 0};
        int n_z = 4;
        const char *zenv = getenv("LUMINA_ION_FRAC_Z");
        if (zenv) {
            n_z = 0;
            char buf[128]; strncpy(buf, zenv, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
            char *tok = strtok(buf, ",");
            while (tok && n_z < 8) { z_list[n_z++] = atoi(tok); tok = strtok(NULL, ","); }
        }
        for (int zi = 0; zi < n_z; zi++) {
            int Z = z_list[zi];
            printf("    [ION-FRAC] Z=%d  (stage 0=I,1=II,2=III,...) by shell:\n", Z);
            for (int s = 0; s < n_shells; s++) {
                double tot = 0.0, frac[6] = {0,0,0,0,0,0};
                for (int ip = 0; ip < atom->n_ion_pops; ip++) {
                    if (atom->ion_pop_Z[ip] != Z) continue;
                    tot += atom->ion_number_density[(size_t)ip * n_shells + s];
                }
                if (tot <= 0.0) continue;
                for (int ip = 0; ip < atom->n_ion_pops; ip++) {
                    if (atom->ion_pop_Z[ip] != Z) continue;
                    int st = atom->ion_pop_stage[ip];
                    if (st >= 0 && st < 6)
                        frac[st] = atom->ion_number_density[(size_t)ip * n_shells + s] / tot;
                }
                printf("      shell %2d: n_tot=%.3e  I=%.3f II=%.3f III=%.3f IV=%.3f V=%.3f\n",
                       s, tot, frac[0], frac[1], frac[2], frac[3], frac[4]);
            }
        }
    }
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
 * Sources: CMFGEN phot_data files (C,Mg,Ca,Cr,Fe,Co,Ni); estimated for Si,S,Ti,Al,Sc,V,Mn.
 * Non-static so the CUDA NLTE GEMM (lumina_nlte_gemm.cu) can share the same table. */
double get_bf_sigma0(int Z, int stage) {
    switch (Z) {
    case 6:  return (stage == 1) ? 3.75e-18 : (stage == 2) ? 1.27e-18 : 0;  /* C  */
    case 12: return (stage == 1) ? 0.23e-18 : (stage == 2) ? 5.42e-18 : 0;  /* Mg */
    case 13: return (stage == 1) ? 0.30e-18 : (stage == 2) ? 1.00e-18 : (stage == 3) ? 0.80e-18 : 0;  /* Al (est, UV multiplets) */
    case 14: return (stage == 1) ? 1.00e-18 : (stage == 2) ? 3.00e-18 : 0;  /* Si (est) */
    case 16: return (stage == 1) ? 2.00e-18 : (stage == 2) ? 3.00e-18 : 0;  /* S  (est) */
    case 20: return (stage == 1) ? 0.31e-18 : (stage == 2) ? 1.92e-18 : 0;  /* Ca */
    case 21: return (stage == 1) ? 3.00e-18 : (stage == 2) ? 2.50e-18 : (stage == 3) ? 2.00e-18 : 0;  /* Sc (est, d-shell) */
    case 22: return (stage == 1) ? 3.00e-18 : (stage == 2) ? 2.00e-18 : 0;  /* Ti (est) */
    case 23: return (stage == 1) ? 3.50e-18 : (stage == 2) ? 3.00e-18 : (stage == 3) ? 2.50e-18 : 0;  /* V (est, d-shell) */
    case 24: return (stage == 1) ? 2.00e-18 : (stage == 2) ? 2.00e-18 : 0;  /* Cr (est) */
    case 25: return (stage == 1) ? 2.50e-18 : (stage == 2) ? 3.00e-18 : (stage == 3) ? 2.50e-18 : 0;  /* Mn (est, d-shell) */
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

    /* Precompute bin center frequencies (used by both CPU and free-free paths) */
    double *nu_bin = (double *)malloc(bf->n_freq_bins * sizeof(double));
    for (int b = 0; b < bf->n_freq_bins; b++) {
        nu_bin[b] = bf->nu_min * exp((b + 0.5) * bf->d_log_nu);
    }

#ifdef LUMINA_HAS_CUDA_BF_GEMM
    /* Task #39: GPU GEMM path (TF32 tensor cores) when CMFGEN sigma_bf is
     * loaded and LUMINA_BF_GEMM=1. Fills chi_bf[s,f] = sum_l n_level[s,l] *
     * sigma_bf[l,f] in a single batched GEMM, then jumps straight to free-free. */
    if (atom->cmfgen_loaded && getenv("LUMINA_BF_GEMM")) {
        if (bf_gemm_compute(bf, atom, plasma, n_shells) == 0) {
            goto compute_ff;
        }
        /* GEMM failed — fall through to CPU loop */
    }
#endif

    /* Per-bin dominant absorber tracking: chi contribution from best ion */
    double *best_chi = (double *)calloc(grid_size, sizeof(double));
    int    *best_ip  = (int *)malloc(grid_size * sizeof(int));
    memset(best_ip, -1, grid_size * sizeof(int));

    /* [BF-DIAG] one-shot tally of CMFGEN-vs-Kramers per-level σ_bf usage.
     * Only the first call with non-empty active levels prints; subsequent iters
     * reuse the same atomic data. Set LUMINA_BF_DIAG=0 to suppress. */
    static int bf_diag_emitted = 0;
    long bf_diag_cmfgen_levels = 0;
    long bf_diag_kramers_levels = 0;

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
        double sigma_0_kramers = get_bf_sigma0(Z_ion, stage);
        if (sigma_0_kramers <= 0.0) {
            int Z_eff = Z_ion - stage;
            if (Z_eff < 1) Z_eff = 1;
            sigma_0_kramers = 7.91e-18 / ((double)Z_eff * (double)Z_eff);
        }

        int lev_start = atom->level_offset[ip];
        int lev_end   = atom->level_offset[ip + 1];

        /* Task #38: Per-level CMFGEN ν-dependent σ_bf when available.
         * Baked grid layout matches bf->n_freq_bins exactly, so we can index
         * directly without interpolation. Falls back to Kramers per-level. */
        const int  use_cmfgen = atom->cmfgen_loaded &&
                                atom->cmfgen_n_freq_bins == bf->n_freq_bins;
        const double *sigma_grid = use_cmfgen ? atom->cmfgen_sigma_bf : NULL;
        const int    *has_sigma  = use_cmfgen ? atom->cmfgen_has_sigma : NULL;

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

                int level_has_cmfgen = use_cmfgen && has_sigma[l];
                const double *sigma_row = level_has_cmfgen
                    ? &sigma_grid[(size_t)l * (size_t)bf->n_freq_bins]
                    : NULL;
                if (!bf_diag_emitted && s == 0) {
                    if (level_has_cmfgen) bf_diag_cmfgen_levels++;
                    else                  bf_diag_kramers_levels++;
                }

                /* Add contribution to all bins above the edge */
                for (int b = bin_start; b < bf->n_freq_bins; b++) {
                    double nu = nu_bin[b];
                    if (nu < nu_edge) continue;
                    double sigma;
                    if (sigma_row) {
                        sigma = sigma_row[b];
                        if (sigma <= 0.0) continue;
                    } else {
                        double ratio = nu_edge / nu;
                        sigma = sigma_0_kramers * ratio * ratio * ratio;
                    }
                    double chi_contrib = n_level * sigma;
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

    {
        long total = bf_diag_cmfgen_levels + bf_diag_kramers_levels;
        if (!bf_diag_emitted && total > 0) {
            double pct = 100.0 * bf_diag_cmfgen_levels / (double)total;
            printf("[BF-DIAG] σ_bf source over %ld active levels (shell 0): "
                   "CMFGEN=%ld (%.1f%%), Kramers=%ld (%.1f%%)\n",
                   total, bf_diag_cmfgen_levels, pct,
                   bf_diag_kramers_levels, 100.0 - pct);
            bf_diag_emitted = 1;
        }
    }

#ifdef LUMINA_HAS_CUDA_BF_GEMM
compute_ff:
#endif
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

/* NLTE target ion definitions: 16 element pairs over 31 slots
 * Original 10: Si,Ca,Fe,S,Co,Ni,C,Mg,Ti,Cr II/III
 * Added 4 (#207-era): Al,Sc,V,Mn II/III for 3000-3500Å UV line blanketing
 * Added 1 (#273): O II/III for O I 7773 triplet line-formation completeness
 * #281 refactor (#279 item 2): full O triplet (I,II,III) via overlap.
 *   Slot 28 = O I, Slot 29 = O II, Slot 30 = O III.
 *   Pair 14 = (slot 28, slot 29) = (O I, O II)  — populates O I 3s⁵S° → 3p⁵P (7774)
 *   Pair 15 = (slot 29, slot 30) = (O II, O III) — keeps O III NLTE (matches CMFGEN)
 *   Slot 29 is shared (single global→nlte map entry); sequential solve in outer iter
 *   loop converges {O I, O II, O III} as one block. */
static const int NLTE_TARGET_Z[]   = { 14, 14, 20, 20, 26, 26, 16, 16, 27, 27, 28, 28,
                                         6,  6, 12, 12, 22, 22, 24, 24,
                                        13, 13, 21, 21, 23, 23, 25, 25,
                                         8,  8,  8 };
static const int NLTE_TARGET_ION[] = {  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,
                                         1,  2,  1,  2,  1,  2,  1,  2,
                                         1,  2,  1,  2,  1,  2,  1,  2,
                                         0,  1,  2 };

/* Heavy.2 / Task #139: Dielectronic recombination Burgess-form table.
 * Sources (provenance per entry):
 *   BADNELL: clist_K master fit (AUTOSTRUCTURE, Strathclyde, 2023-05-12).
 *   NORAD:   R-matrix unified RR+DR by Nahar et al. (Fe I-VI, Ni II).
 *   MAZZOTTA: Mazzotta+1998 LS-coupling fits (low-T-inaccurate floor).
 *   EST_ISOEL: isoelectronic interpolation placeholder (flagged).
 * Convention: ion_recomb = the recombining (upper) ion stage.
 * E.g. Fe III → Fe II is stored as {Z=26, ion_recomb=2}. */
static const DRCoefficient DR_TABLE[] = {
    /* --- Badnell (parsed from clist_K) ------------------------------ */
    /* Si III → Si II  (Z=14, N=12, Mg-like) */
    {14, 2, 5,
     {2.930e-06, 2.803e-06, 9.023e-05, 6.909e-03, 2.582e-05},
     {1.162e+02, 5.721e+03, 3.477e+04, 1.176e+05, 3.505e+06},
     DR_SOURCE_BADNELL},
    /* Si II → Si I    (Z=14, N=13, Al-like) */
    {14, 1, 6,
     {3.408e-08, 1.913e-07, 1.679e-07, 7.523e-07, 8.386e-05, 4.083e-03},
     {2.431e+01, 1.293e+02, 4.272e+02, 3.729e+03, 5.514e+04, 1.295e+05},
     DR_SOURCE_BADNELL},
    /* S III → S II    (Z=16, N=14, Si-like) */
    {16, 2, 7,
     {3.040e-07, 4.393e-07, 1.609e-06, 4.980e-06, 3.457e-05, 8.617e-03, 9.284e-04},
     {5.016e+01, 3.266e+02, 3.102e+03, 1.210e+04, 4.969e+04, 2.010e+05, 2.575e+05},
     DR_SOURCE_BADNELL},
    /* S II → S I      (Z=16, N=15, P-like) */
    {16, 1, 7,
     {7.300e-08, 2.577e-07, 4.961e-08, 9.520e-07, 9.586e-07, 6.849e-04, 6.539e-04},
     {5.077e+02, 6.007e+02, 2.342e+03, 7.269e+03, 2.190e+04, 1.483e+05, 1.906e+05},
     DR_SOURCE_BADNELL},
    /* C III → C II    (Z=6,  N=4,  Be-like) */
    {6, 2, 6,
     {3.489e-06, 2.222e-07, 1.954e-05, 4.212e-03, 2.037e-04, 2.936e-04},
     {2.660e+03, 3.756e+03, 2.566e+04, 1.400e+05, 1.801e+06, 4.307e+06},
     DR_SOURCE_BADNELL},
    /* C II → C I      (Z=6,  N=5,  B-like) */
    {6, 1, 5,
     {6.346e-09, 9.793e-09, 1.634e-06, 8.369e-04, 3.355e-04},
     {1.217e+01, 7.380e+01, 1.523e+04, 1.207e+05, 2.144e+05},
     DR_SOURCE_BADNELL},
    /* Mg III → Mg II  (Z=12, N=10, Ne-like) */
    {12, 2, 3,
     {6.269e-06, 9.181e-04, 3.082e-04},
     {4.104e+05, 5.766e+05, 7.310e+05},
     DR_SOURCE_BADNELL},
    /* Mg II → Mg I    (Z=12, N=11, Na-like) */
    {12, 1, 4,
     {3.871e-08, 4.732e-07, 1.599e-03, 2.628e-05},
     {8.415e+03, 1.682e+04, 5.000e+04, 2.759e+05},
     DR_SOURCE_BADNELL},
    /* Ca III → Ca II  (Z=20, N=18, Ar-like) */
    {20, 2, 3,
     {3.843e-04, 8.040e-03, 8.670e-03},
     {2.282e+05, 3.682e+05, 4.479e+05},
     DR_SOURCE_BADNELL},

    /* --- NORAD (Nahar OSU R-matrix unified RR+DR total) -------------
     * Source files: data/atomic/dr_norad/{Fe1..Fe5,Ni1}.csv (81-pt grid)
     * Burgess fits at data/atomic/dr_norad/burgess_fits.txt (max err <1.1%
     * over 100 ≤ T ≤ 1e5 K). NOTE: these are TOTAL recomb (RR+DR), not
     * DR-only — slight double-counting with Milne RR (~10-20% at SN T_e),
     * acceptable floor for Phase 1 stabilization. Can be refined to
     * DR-only later by subtracting low-n RR column. */
    /* Fe II → Fe I  (Nahar Bautista Pradhan 1997 ApJ 479 497) */
    {26, 1, 5,
     {3.389049e-08, 1.406860e-07, 4.492892e-07, 1.494212e-06, 4.035439e-04},
     {1.080998e+02, 7.995730e+02, 2.676773e+03, 1.335320e+04, 5.680572e+04},
     DR_SOURCE_NORAD},
    /* Fe III → Fe II (Nahar 1997 PRA 55 1980) */
    {26, 2, 6,
     {7.553554e-08, 2.179776e-07, 9.813920e-07, 4.370443e-04, 7.175326e-06, 2.206982e-05},
     {9.418447e+01, 7.125228e+02, 3.095721e+03, 1.415009e+05, 1.190955e+04, 3.291757e+04},
     DR_SOURCE_NORAD},
    /* Fe IV → Fe III (Nahar 1996 PRA 53 2417) */
    {26, 3, 6,
     {2.762952e-07, 6.854108e-07, 1.112431e-05, 3.071916e-05, 2.100067e-06, 9.690480e-04},
     {9.502922e+01, 7.045133e+02, 1.554180e+04, 4.609481e+04, 3.392230e+03, 2.483458e+05},
     DR_SOURCE_NORAD},
    /* Fe V → Fe IV (Nahar Bautista Pradhan 1998 PRA 58 4593) */
    {26, 4, 6,
     {3.918497e-07, 1.262495e-06, 4.004534e-06, 2.069640e-05, 5.013299e-05, 1.753367e-03},
     {1.016419e+02, 7.656969e+02, 3.443625e+03, 1.488907e+04, 6.201225e+04, 2.933028e+05},
     DR_SOURCE_NORAD},
    /* Fe VI → Fe V (Nahar Bautista 1999 ApJS 120 327) */
    {26, 5, 6,
     {1.054419e-06, 2.771906e-06, 7.897993e-06, 3.215862e-05, 1.542221e-03, 9.613848e-05},
     {8.824877e+01, 7.638482e+02, 3.536893e+03, 1.466235e+04, 2.688904e+05, 6.446002e+04},
     DR_SOURCE_NORAD},
    /* Ni II → Ni I (Nahar Bautista 2001 ApJS 137 201) */
    {28, 1, 6,
     {1.591064e-07, 6.798302e-07, 1.811202e-06, 4.724703e-04, 7.704725e-05, 5.987436e-06},
     {1.116045e+02, 7.047342e+02, 2.845402e+03, 1.435708e+05, 5.704062e+04, 1.181851e+04},
     DR_SOURCE_NORAD},
    /* Si II → Si I  (Z=14, Nahar 2000 ApJS 126 537 — Si-sequence R-matrix unified RR+DR)
     * Burgess 6-term fit, max rel err 0.96%, median 0.42% over T=100..1e5 K.
     * α(8000 K) ≈ 1.65e-12 cm³/s. */
    {14, 1, 6,
     {3.692246e-08, 1.246370e-07, 4.742018e-07, 2.341124e-06, 1.499678e-05, 7.501304e-04},
     {1.008423e+02, 7.150889e+02, 3.024003e+03, 1.108178e+04, 3.860763e+04, 1.118590e+05},
     DR_SOURCE_NORAD},
    /* Si III → Si II (Z=14, Nahar Pradhan Zhang 2000 ApJS 131 375 — R-matrix unified RR+DR)
     * Burgess 5-term fit, max rel err 1.64%, median 0.85% over T=100..1e5 K.
     * α(8000 K) ≈ 2.84e-12 cm³/s.  Closes Si II→III over-ionization at SN photosphere
     * (n_e·α ≈ 7e-3 s⁻¹ > R_bf_ground ≈ 5.9e-3 from logs/diag_ratebal_154431). */
    {14, 2, 5,
     {9.651958e-08, 3.283584e-07, 1.361371e-06, 6.897597e-03, 2.329208e-05},
     {1.094415e+02, 8.894031e+02, 4.719150e+03, 1.160976e+05, 2.613932e+04},
     DR_SOURCE_NORAD},

    /* --- Mazzotta+1998 LS-coupling floor for K-like+ iron-peak -------
     * Source files: data/atomic/dr_mazzotta/Z{Z}_ion{n}.txt
     * 17 from CHIANTI v11.0.2 drparams (Mazzotta1998); Ca from original CDS.
     * Burgess form: alpha_DR(T) = T^-1.5 * sum(c_i * exp(-E_i / T)).
     * Convention: file's "ion" is 1-indexed roman of the recombining ion;
     *   here ion_recomb = roman - 1 (0-indexed stage; II=1, III=2).
     * KNOWN LIMITATION: LS-coupling misses near-threshold resonances —
     *   underestimates low-T (T_e < 5e4 K) DR by 10×–10³× for Fe-peak.
     *   This is the "floor" — should be replaced by AUTOSTRUCT (Task #141)
     *   or Open-ADAS submissions when available.
     * Mazzotta entries duplicating Badnell (Ca III) or NORAD (Fe II–IV,
     *   Ni II) are skipped — keep the higher-quality data. */

    /* O II → O I  (Z=8, recombining = O II) — Mazzotta1998 single-term fit
     * (CDS table1.dat row "O II", converted eV→K: c_K = c_eV * 11604.519^1.5,
     *  E_K = E_eV * 11604.519). Added in #273 with O II/III NLTE pair for
     *  O I 7773 triplet line-formation completeness. Mazzotta LS-coupling
     *  floor; near-threshold resonances missing — refine later. */
    {8, 1, 1, {1.211336e-03}, {1.810305e+05}, DR_SOURCE_MAZZOTTA},
    /* O III → O II (Z=8, recombining = O III) — Mazzotta1998 single-term fit
     * (CDS table1.dat row "O III"). */
    {8, 2, 1, {4.775338e-03}, {2.115504e+05}, DR_SOURCE_MAZZOTTA},
    /* Ca II → Ca I  (Z=20, recombining = Ca II = stage 1) */
    {20, 1, 1, {1.987015e-04}, {3.585796e+04}, DR_SOURCE_MAZZOTTA},
    /* Sc II → Sc I  (Z=21, stage 1) */
    {21, 1, 1, {2.426200e-03}, {5.883500e+04}, DR_SOURCE_MAZZOTTA},
    /* Sc III → Sc II (Z=21, K-like recomb) — AUTOSTRUCTURE FULL-PHYSICS deck
     * MXCONF=5 (3d+4s+4p+4d+4f), MXCCF=15 captured Sc II configs, COREX='3-4'.
     * adf09 Burgess 6-term fit, gate max |rel err| 0.07%, transient 1.0% peak
     * (T~2000 K, where DR is small). α(8000 K)=1.14e-12 cm³/s ≈ 110× Mazzotta
     * peak-fit floor (and ~11× the MXCONF=2 pilot, confirming 4l channels dominate
     * at SN photospheric T). adf09: /home/kjhan/local/autostructure/runs/sc3_dr_full/ */
    {21, 2, 6,
     {2.3542e-07, 3.6318e-07, 4.7949e-11, 6.8761e-07, 1.9013e-05, 3.3273e-04},
     {8.4060e+02, 9.5042e+02, 1.2952e+03, 7.5144e+03, 5.8950e+04, 1.1377e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* Sc III → Sc II (Z=21, stage 2) — Mazzotta1998 fallback (kept below AS entry,
     * never matched by dr_lookup since AS line above hits first). */
    {21, 2, 1, {1.309500e-02}, {1.151200e+05}, DR_SOURCE_MAZZOTTA},
    /* Ti II → Ti I  (Z=22, stage 1) */
    {22, 1, 1, {4.242400e-03}, {1.122200e+05}, DR_SOURCE_MAZZOTTA},
    /* Ti III → Ti II (Z=22, Sc-like recomb) — AUTOSTRUCTURE plan-C deck
     * MXCONF=3 (3d²+3d4s+3d4p), MXCCF=6, NMAX=10, LMAX=5, COREX='3-4'.
     * adf09 5-term fit (rel err <0.5% at 4-100 kK). α(8000K)=1.81e-12 cm³/s.
     * adf09: /home/kjhan/local/autostructure/runs/ti3_dr_planC/ */
    {22, 2, 5,
     {4.8753e-08, 1.2221e-06, 1.8819e-06, 2.1847e-05, 1.1183e-03},
     {6.1884e+02, 3.2501e+03, 1.1777e+04, 6.8805e+04, 1.3899e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* Ti III → Ti II (Z=22, stage 2) — Mazzotta1998 fallback (never matched). */
    {22, 2, 1, {2.351900e-02}, {1.972800e+05}, DR_SOURCE_MAZZOTTA},
    /* V II → V I    (Z=23, stage 1) */
    {23, 1, 1, {3.524100e-03}, {1.357700e+05}, DR_SOURCE_MAZZOTTA},
    /* V III → V II (Z=23, Ti-like recomb) — AUTOSTRUCTURE plan-C deck
     * MXCONF=3 (3d³+3d²4s+3d²4p), MXCCF=6, NMAX=10, LMAX=5, COREX='3-4'.
     * adf09 5-term fit (rel err <0.6% at 4-100 kK). α(8000K)=3.96e-12 cm³/s.
     * adf09: /home/kjhan/local/autostructure/runs/v3_dr_planC/ */
    {23, 2, 5,
     {3.5549e-07, 1.7437e-06, 3.8422e-06, 1.6154e-05, 1.1303e-03},
     {1.3153e+02, 1.7990e+03, 1.0450e+04, 4.6119e+04, 1.8102e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* V III → V II (Z=23, stage 2) — Mazzotta1998 fallback (never matched). */
    {23, 2, 1, {1.436500e-02}, {1.659400e+05}, DR_SOURCE_MAZZOTTA},
    /* Cr II → Cr I  (Z=24, stage 1) */
    {24, 1, 1, {2.678200e-03}, {2.021500e+05}, DR_SOURCE_MAZZOTTA},
    /* Cr III → Cr II (Z=24, V-like recomb) — AUTOSTRUCTURE plan-C deck
     * MXCONF=3 (3d⁴+3d³4s+3d³4p), MXCCF=6, NMAX=10, LMAX=5, COREX='3-4'.
     * adf09 5-term fit, gate max |rel err| 0.22% (T∈[4000,100000] K).
     * α(8000 K)=5.47e-12 cm³/s ≈ 10⁴× Mazzotta1998 (V-like 3d⁴ huge resonance density).
     * adf09: /home/kjhan/local/autostructure/runs/cr3_dr_planC/ */
    {24, 2, 5,
     {3.1657e-07, 1.4880e-06, 1.0498e-05, 2.7076e-05, 1.0159e-03},
     {2.5698e+03, 4.4490e+03, 1.0659e+04, 4.9445e+04, 2.2249e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* Cr III → Cr II (Z=24, stage 2) — Mazzotta1998 fallback (never matched). */
    {24, 2, 1, {9.758900e-03}, {1.360000e+05}, DR_SOURCE_MAZZOTTA},
    /* Mn II → Mn I  (Z=25, stage 1) */
    {25, 1, 1, {1.121200e-03}, {1.329900e+05}, DR_SOURCE_MAZZOTTA},
    /* Mn III → Mn II (Z=25, Cr-like recomb) — AUTOSTRUCTURE plan-C deck
     * MXCONF=3 (3d⁵+3d⁴4s+3d⁴4p), MXCCF=6, NMAX=10, LMAX=5, COREX='3-4'.
     * adf09 5-term fit (rel err <0.4% at 4-100 kK). α(8000K)=1.41e-12 cm³/s.
     * adf09: /home/kjhan/local/autostructure/runs/mn3_dr_planC/ */
    {25, 2, 5,
     {2.8136e-09, 6.0328e-08, 5.4823e-06, 1.5525e-05, 1.1715e-03},
     {9.3288e+02, 2.2496e+03, 1.4007e+04, 6.2004e+04, 2.6043e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* Mn III → Mn II (Z=25, stage 2) — Mazzotta1998 fallback (never matched). */
    {25, 2, 1, {7.785800e-03}, {2.004100e+05}, DR_SOURCE_MAZZOTTA},
    /* Co II → Co I  (Z=27, stage 1) */
    {27, 1, 2,
     {2.855200e-04, 2.838900e-04},
     {5.779000e+04, 2.658600e+05},
     DR_SOURCE_MAZZOTTA},
    /* Co III → Co II (Z=27, Mn-like recomb) — AUTOSTRUCTURE plan-C deck
     * MXCONF=3 (3d⁷+3d⁶4s+3d⁶4p), MXCCF=6 (3d⁸+3d⁷4s+3d⁷4p+3d⁶4s²+3d⁶4s4p+3d⁶4p²),
     * NMAX=10, LMAX=5, COREX='3-4'. adf09 5-term fit, gate max |rel err| 0.25%
     * (T∈[4000,100000] K). α(8000 K)=6.42e-13 cm³/s ≈ 64× Mazzotta1998.
     * Co III pilot (MXCONF=2) gave only ~1× Mazzotta — Mn-like 7-valence 3d⁷
     * requires 4p-capture channel for SN photospheric T_e=8000 K.
     * adf09: /home/kjhan/local/autostructure/runs/co3_dr_planB/ */
    {27, 2, 5,
     {2.4215e-11, 2.9793e-07, 7.7205e-07, 1.2043e-05, 4.2601e-04},
     {1.5709e+02, 1.7237e+03, 1.0538e+04, 5.5318e+04, 2.6117e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* Co III → Co II (Z=27, stage 2) — Mazzotta1998 fallback (kept below AS line,
     * never matched by dr_lookup since AS entry above hits first). */
    {27, 2, 1, {2.858100e-03}, {1.444800e+05}, DR_SOURCE_MAZZOTTA},
    /* Ni III → Ni II (Z=28, Co-like recomb) — AUTOSTRUCTURE plan-C deck
     * MXCONF=3 (3d⁸+3d⁷4s+3d⁷4p), MXCCF=6, NMAX=10, LMAX=5, COREX='3-4'.
     * adf09 5-term fit, gate max |rel err| 0.01% (T∈[4000,100000] K).
     * α(8000 K)=6.7e-13 cm³/s ≈ 10¹⁰× Mazzotta1998 floor (Mazzotta has E=2.6e5 K barrier).
     * adf09: /home/kjhan/local/autostructure/runs/ni3_dr_planC/ */
    {28, 2, 5,
     {1.6796e-07, 4.3130e-07, 3.3611e-06, 1.9387e-05, 4.0494e-04},
     {1.4518e+03, 6.9076e+03, 2.4518e+04, 8.3081e+04, 2.5458e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* Ni III → Ni II (Z=28, stage 2) — Mazzotta1998 fallback (never matched). */
    {28, 2, 1, {9.215000e-03}, {2.604100e+05}, DR_SOURCE_MAZZOTTA},

    /* --- AUTOSTRUCT self-compute: pending Task #141 ----------------- */
};
#define DR_N_ENTRIES ((int)(sizeof(DR_TABLE) / sizeof(DR_TABLE[0])))

/* Diagnostic: empirical multiplier on α_DR via env var LUMINA_DR_BOOST.
 * Used to test the LS-coupling-underestimate hypothesis (Mazzotta+1998
 * misses near-threshold autoionizing resonances; Fe-peak underestimated
 * by 10–10³× at SN T_e ~ 5e3–1e4 K). Default 1.0 (no scaling). Env var
 * accepts decimal floats, e.g. "10", "100", "31.6". Per-source masks
 * ("LUMINA_DR_BOOST_MAZZOTTA", "LUMINA_DR_BOOST_BADNELL" etc.) stack
 * multiplicatively if present. */
static double dr_boost_factor(DRSource src) {
    static int initialized = 0;
    static double boost_all = 1.0;
    static double boost_per_src[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    if (!initialized) {
        const char *all = getenv("LUMINA_DR_BOOST");
        if (all) boost_all = atof(all);
        const char *names[5] = {"NONE","BADNELL","NORAD","MAZZOTTA","AUTOSTRUCT"};
        for (int i = 1; i <= 4; i++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "LUMINA_DR_BOOST_%s", names[i]);
            const char *v = getenv(buf);
            if (v) boost_per_src[i] = atof(v);
        }
        initialized = 1;
    }
    double f = boost_all;
    if ((int)src >= 0 && (int)src < 6) f *= boost_per_src[(int)src];
    return f;
}

double dr_alpha_eval(const DRCoefficient *coef, double T_e) {
    if (!coef || coef->n_terms <= 0) return 0.0;
    if (T_e < 1.0) T_e = 1.0;
    double sum = 0.0;
    for (int i = 0; i < coef->n_terms; i++) {
        double arg = -coef->E_i[i] / T_e;
        if (arg < -700.0) continue;
        sum += coef->c_i[i] * exp(arg);
    }
    return sum * pow(T_e, -1.5) * dr_boost_factor(coef->source);
}

const DRCoefficient* dr_lookup(int Z, int ion_recomb) {
    for (int i = 0; i < DR_N_ENTRIES; i++) {
        if (DR_TABLE[i].Z == Z && DR_TABLE[i].ion_recomb == ion_recomb)
            return &DR_TABLE[i];
    }
    return NULL;
}

/* Step 1.5: Charge Exchange reaction table
 * Forward: A^(ion_A) + B^(ion_B) → A^(ion_A+1) + B^(ion_B-1)
 * Convention: A,B chosen so forward is exothermic (IP(A:1→2) < IP(B:1→2)).
 * Reverse via detailed balance: k_rev = k_fwd * exp(|ΔE|/kT)
 * Rate coefficients from Kingdon & Ferland (1996), generic 1e-9 cm³/s.
 *
 * Task #140 (Heavy.3) expansion (rows 7-17, 11 new) targets Gate B-4
 * Ni II 36× over-recombination by adding Ni-X (X=Cr,Mn,Ti,V) cross-coupling
 * channels — reverse direction (Ni+ + X2+ → Ni2+ + X+) acts as a Ni II
 * depletion pathway when Ni+ is over-saturated. Also adds Co-X (X=Cr,Mn,
 * Ti,V) and Fe-X (X=Mn,V,Sc) to round out the iron-peak network.
 * ΔE_eV computed from NIST II→III ionization potentials. */
static const ChargeExchangeReaction CE_REACTIONS[CE_N_REACTIONS] = {
  /* Z_A ion_A  Z_B ion_B  rate       alpha  ΔE_eV  */
  {  26,   1,    27,   2,   1.0e-9,    0.0,  -0.89 },  /* Fe+ + Co2+ → Fe2+ + Co+ */
  {  26,   1,    28,   2,   1.0e-9,    0.0,  -1.98 },  /* Fe+ + Ni2+ → Fe2+ + Ni+ */
  {  27,   1,    28,   2,   1.0e-9,    0.0,  -1.09 },  /* Co+ + Ni2+ → Co2+ + Ni+ */
  {  20,   1,    14,   2,   1.0e-9,    0.0,  -4.48 },  /* Ca+ + Si2+ → Ca2+ + Si+ */
  {  26,   1,    24,   2,   1.0e-9,    0.0,  -1.47 },  /* Fe+ + Cr2+ → Fe2+ + Cr+ */
  {  26,   1,    22,   2,   1.0e-9,    0.0,  -0.77 },  /* Fe+ + Ti2+ → Fe2+ + Ti+ */
  /* --- Heavy.3 expansion (Ni-coupling for Gate B-4 fix) --- */
  {  25,   1,    26,   2,   1.0e-9,    0.0,  -0.56 },  /* Mn+ + Fe2+ → Mn2+ + Fe+ */
  {  23,   1,    26,   2,   1.0e-9,    0.0,  -1.58 },  /* V+  + Fe2+ → V2+  + Fe+ */
  {  21,   1,    26,   2,   1.0e-9,    0.0,  -3.40 },  /* Sc+ + Fe2+ → Sc2+ + Fe+ */
  {  24,   1,    27,   2,   1.0e-9,    0.0,  -0.59 },  /* Cr+ + Co2+ → Cr2+ + Co+ */
  {  25,   1,    27,   2,   1.0e-9,    0.0,  -1.44 },  /* Mn+ + Co2+ → Mn2+ + Co+ */
  {  22,   1,    27,   2,   1.0e-9,    0.0,  -3.50 },  /* Ti+ + Co2+ → Ti2+ + Co+ */
  {  23,   1,    27,   2,   1.0e-9,    0.0,  -2.46 },  /* V+  + Co2+ → V2+  + Co+ */
  {  24,   1,    28,   2,   1.0e-9,    0.0,  -1.68 },  /* Cr+ + Ni2+ → Cr2+ + Ni+ */
  {  25,   1,    28,   2,   1.0e-9,    0.0,  -2.53 },  /* Mn+ + Ni2+ → Mn2+ + Ni+ */
  {  22,   1,    28,   2,   1.0e-9,    0.0,  -4.59 },  /* Ti+ + Ni2+ → Ti2+ + Ni+ */
  {  23,   1,    28,   2,   1.0e-9,    0.0,  -3.55 },  /* V+  + Ni2+ → V2+  + Ni+ */
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

/* Mihalas-Lucy ion-lock gate (env-cached, iter-aware). */
int nlte_ion_lock_active(int current_iter) {
    static int init = 0;
    static int enabled = 0;
    static int start_iter = 0;
    if (!init) {
        const char *e1 = getenv("LUMINA_NLTE_ION_LOCK");
        const char *e2 = getenv("LUMINA_NLTE_LOCK_START_ITER");
        if (e1 && atoi(e1) != 0) enabled = 1;
        if (e2) start_iter = atoi(e2);
        init = 1;
    }
    return enabled && current_iter >= start_iter;
}

/* Per-ion rescale gate (env-cached). Decouples the post-solve per-ion rescale
 * path from LUMINA_NLTE_ION_LOCK, which also triggers freeze-plasma-transport-only
 * in the iter driver (cuda.cu / main.c). Set LUMINA_NLTE_PER_ION_RESCALE=1 to enable
 * only the rescale path at plasma.c:2941/2989 (and cuda.cu:458/520) — useful for
 * isolating the #223 combined-Σ collapse fix without freezing plasma. */
int nlte_per_ion_rescale_active(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        const char *e = getenv("LUMINA_NLTE_PER_ION_RESCALE");
        if (e && atoi(e) != 0) enabled = 1;
        init = 1;
    }
    return enabled;
}

/* Skip rate-matrix assembly for ion pairs whose total density has collapsed to
 * essentially zero (trace species in shells where the element abundance -> 0).
 * Assembly is 96.6% of NLTE wall time; a dead pair contributes ~0 populations
 * whether solved or Boltzmann-filled, so the costly per-level assembly is pure
 * waste. With the gate on, the assembly loop leaves the (already-zeroed) matrix
 * untouched -> the GPU getrf flags it singular -> the existing inline
 * Boltzmann@T_rad fallback fills the ~0 pops. Env-gated (default off) so the
 * gate-off path stays byte-identical for A/B verification. */
int nlte_skip_dead_pairs(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        const char *e = getenv("LUMINA_NLTE_SKIP_DEAD");
        if (e && atoi(e) != 0) enabled = 1;
        init = 1;
    }
    return enabled;
}

/* Time-dependent ionization gate (env-cached). When set, nlte_assemble_rate_matrix
 * adds a backward-Euler dn_i/dt term to the rate rows using Dt = time_explosion
 * and a fully-ionized t=0 initial condition. This is the single-epoch (option A)
 * frozen-in approximation: ion stages whose net recombination rate << 1/Dt cannot
 * relax in the SN's age and stay over-ionized (the tau_rec >> t_exp outer layers),
 * while fast-recombining inner shells relax to the steady-state solution
 * (Dt->infinity recovers the steady-state matrix exactly). Default off. */
int nlte_timedep_active(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        const char *e = getenv("LUMINA_TIMEDEP_ION");
        if (e && atoi(e) != 0) enabled = 1;
        init = 1;
    }
    return enabled;
}

/* Boltzmann-ceiling margin for the NLTE finite-garbage sanity gate.
 * A near-singular (but not exactly singular) rate matrix can yield a FINITE
 * solution whose excited-level pops sit 1e9-1e11x above the ion ground state —
 * far past the LTE ceiling x_i/x_0 <= g_i/g_0. Such solutions pass the
 * isfinite/info checks; the conservation rescale fixes only the sum, not the
 * inverted shape. The gate rejects a solve when any level exceeds
 * (g_i/g_ground) * margin, routing it to the Boltzmann@T_rad fallback.
 * Margin tunable via LUMINA_NLTE_INV_CEIL (default 1e4); <=0 disables the gate. */
double nlte_inv_ceiling(void) {
    static int init = 0;
    static double margin = 1e4;
    if (!init) {
        const char *e = getenv("LUMINA_NLTE_INV_CEIL");
        if (e) margin = atof(e);
        init = 1;
    }
    return margin;
}

/* ============================================================
 * Task #20: real radiative-equilibrium electron temperature.
 *
 * Replaces the parametric Compton + fudge-boost / adiabatic-only balance
 * (compute_electron_temperature self-consistent branch) with a genuine
 * per-shell heating = cooling solve, gated by LUMINA_RADEQ_TE=1.
 *
 *   Heating  H = photoionization + Compton + gamma-ray deposition
 *   Cooling  C = radiative recombination + free-free + collisional
 *                bound-bound (line) + adiabatic expansion
 *
 * Net(T_e) = H(T_e) - C(T_e) is monotone-decreasing in T_e, solved by
 * bisection on [0.1, 2.0]*T_rad. Operator-split: NLTE level populations
 * and J_nu are lagged (held at the previous iteration's values during the
 * T_e solve), matching CMFGEN's linearization. Method-faithfulness
 * (Phase-1), NOT a tuning knob — no free parameters.
 * ============================================================ */
typedef struct {
    int nlte_lo;      /* NLTE level index of lower level, or -1 if not NLTE-tracked */
    int nlte_up;      /* NLTE level index of upper level, or -1 if not NLTE-tracked */
    int lo_g, up_g;   /* GLOBAL level indices (for nebular pop / E / g / meta lookup) */
    int ip;           /* ion-population index (for n_ion / partition fn lookup) */
    int line;         /* line index (for tau_sobolev[line*n_shells+s] lookup) */
    double dE;        /* transition energy [erg] */
    double beta;      /* dE / k_B  [K]  (precomputed) */
    double coeff;     /* collisional rate coefficient: K*f_lu (permitted) or
                       * 8.63e-6*omega (forbidden); EXCLUDES n_e and 1/(g*sqrtTe) */
    double A_ul;      /* spontaneous emission rate [s^-1] (radiative-escape cooling) */
    int g_lo, g_up;   /* statistical weights */
} RadEqLine;

static RadEqLine *radeq_lines = NULL;
static long radeq_n_lines = -1;

/* Physical two-level destruction probability for one bb line at (n_e,T_e,tau):
 * eps = C_ul/(C_ul + A_ul*beta_esc). Exported for cmfgen_assemble's per-line
 * thermal-channel accumulation (LUMINA_CMFGEN_LINE_EPS_PHYS). Returns -1 when
 * the RADEQ line table is not built yet or the line is unknown (caller falls
 * back to the legacy fully-thermal treatment). */
double radeq_line_eps_phys(int line, double n_e, double T_e, double tau);

static double radeq_beta_esc(double tau);

double radeq_line_eps_phys(int line, double n_e, double T_e, double tau) {
    static int *line2k = NULL;
    static int  line2k_n = 0;
    if (!radeq_lines || radeq_n_lines <= 0 || line < 0 || T_e <= 0.0)
        return -1.0;
    if (!line2k) {
        int maxline = 0;
        for (long k = 0; k < radeq_n_lines; k++)
            if (radeq_lines[k].line > maxline) maxline = radeq_lines[k].line;
        int *m = (int *)malloc((size_t)(maxline + 1) * sizeof(int));
        if (!m) return -1.0;
        for (int i = 0; i <= maxline; i++) m[i] = -1;
        for (long k = 0; k < radeq_n_lines; k++) m[radeq_lines[k].line] = (int)k;
        line2k_n = maxline + 1;
        line2k = m;
    }
    if (line >= line2k_n) return -1.0;
    int k = line2k[line];
    if (k < 0) return -1.0;
    const RadEqLine *rl = &radeq_lines[k];
    double C_ul = n_e * rl->coeff / ((double)rl->g_up * sqrt(T_e));
    double denom = C_ul + rl->A_ul * radeq_beta_esc(tau);
    return (denom > 0.0) ? C_ul / denom : 1.0;
}

static void build_radeq_line_table(NLTEConfig *nlte, AtomicData *atom,
                                   OpacityState *opacity) {
    if (radeq_n_lines >= 0) return;  /* already built */
    int n_lines = opacity->n_lines;
    long count = 0;
    /* van-Regemorter Gaunt factor for PERMITTED lines. The NLTE rate matrix uses
     * 0.2 for permitted bb collisions; the cooling table historically used 1.0.
     * Default 1.0 = byte-identical; LUMINA_RADEQ_VR_GAUNT=0.2 reconciles the
     * cooling-rate magnitude (and ETLA SE branching) with the population solver. */
    double vr_gaunt = 1.0;
    { const char *vg = getenv("LUMINA_RADEQ_VR_GAUNT"); if (vg) vr_gaunt = atof(vg); }
    /* Build the cooling table from ALL valid bb lines (not just NLTE-tracked),
     * since collisionally-excited line cooling is the dominant outer-ejecta
     * coolant and is carried by the full line census, not the few NLTE levels.
     * Per-level populations are resolved at solve time: NLTE pop where the level
     * is tracked, dilute-Boltzmann (nebular) otherwise. */
    for (int pass = 0; pass < 2; pass++) {
        long k = 0;
        for (int line = 0; line < n_lines; line++) {
            int ion_s = atom->line_ion_number[line];
            int ip = find_ion_pop_idx(atom, atom->line_atomic_number[line], ion_s);
            if (ip < 0) continue;
            int lev_base = atom->level_offset[ip];
            int lev_top  = atom->level_offset[ip + 1];
            int lo_g = -1, up_g = -1;
            for (int l = lev_base; l < lev_top; l++) {
                if (atom->level_num[l] == atom->line_level_lower[line]) lo_g = l;
                if (atom->level_num[l] == atom->line_level_upper[line]) up_g = l;
                if (lo_g >= 0 && up_g >= 0) break;
            }
            if (lo_g < 0 || up_g < 0) continue;
            double dE = fabs(atom->level_energy_eV[up_g] -
                             atom->level_energy_eV[lo_g]) * EV_TO_ERG;
            if (dE <= 0.0) continue;
            if (pass == 1) {
                double f_lu = atom->line_f_lu[line];
                radeq_lines[k].nlte_lo = nlte->global_to_nlte_level[lo_g];
                radeq_lines[k].nlte_up = nlte->global_to_nlte_level[up_g];
                radeq_lines[k].lo_g = lo_g;
                radeq_lines[k].up_g = up_g;
                radeq_lines[k].ip   = ip;
                radeq_lines[k].line = line;
                radeq_lines[k].dE   = dE;
                radeq_lines[k].beta = dE / K_BOLTZMANN;
                radeq_lines[k].coeff = (f_lu > 1e-10) ?
                    VAN_REGEMORTER_COEFF * f_lu * vr_gaunt : 8.63e-6 * AXELROD_OMEGA;
                radeq_lines[k].A_ul = atom->line_A_ul ? atom->line_A_ul[line] : 0.0;
                radeq_lines[k].g_lo = atom->level_g[lo_g];
                radeq_lines[k].g_up = atom->level_g[up_g];
            }
            k++;
        }
        if (pass == 0) {
            count = k;
            radeq_lines = (RadEqLine *)malloc((size_t)(count > 0 ? count : 1) *
                                              sizeof(RadEqLine));
        }
    }
    radeq_n_lines = count;
    printf("  [RADEQ] collisional line-cooling table: %ld bb transitions (all ions)\n",
           radeq_n_lines);
}

/* Net energy rate H(T_e) - C(T_e) [erg/s/cm^3] for one shell. The lagged,
 * T_e-independent heating (H_photo + H_gamma) and the per-line cooling
 * coefficients (a/b/beta over the active line set) are precomputed by the
 * caller; everything T_e-dependent is evaluated here so the bisection
 * re-evaluates cheaply. */
/* Bound-free radiative cooling, the EMISSION half of the detailed-balance pair.
 * Built (per shell) from the SAME σ_bf, f_above weight, AND the SAME lagged level
 * population n_lev used by the photoheating integral, but with the radiation field
 * replaced by the local Wien Planck function B_ν(T_e)=(2hν³/c²)e^{−hν/kT_e}:
 *   C_emit(T_e) = Σ_ν [ Σ_l n_lev · 4π (2hν³/c²) σ_bf f_above dν ] · e^{−hν/kT_e}
 *               = Σ_ν emit_nu[ν] · e^{−hν/kT_e}
 * The bracketed sum is T_e-independent and accumulated into emit_nu[] alongside
 * H_photo. Because heating uses n_lev·J_ν and cooling uses n_lev·B_ν over the
 * identical ν-grid, the bound-free net = n_lev ∫4π(J_ν−B_ν)σ_bf f_above dν cancels
 * BIN-BY-BIN at LTE (J_ν=B_ν) and, crucially, the cooling carries n_lev too — so a
 * spiking departure coefficient b_l=n_lev/n*_l grows BOTH terms together, bounding
 * the net by (J_ν−B_ν) and restoring the outer-shell thermostat that the old
 * n*_l-weighted (Saha) cooling lacked. */
static double radeq_recomb_cool(double T_e, const double *emit_nu,
                                const double *nu_mid, int nbins) {
    double inv_kT = 1.0 / (K_BOLTZMANN * T_e);
    double sum = 0.0;
    for (int bb = 0; bb < nbins; bb++) {
        if (emit_nu[bb] == 0.0) continue;
        sum += emit_nu[bb] * exp(-H_PLANCK * nu_mid[bb] * inv_kT);
    }
    return sum;
}

/* Collisional bound-bound (line) cooling, evaluated from per-shell precomputed
 * coefficients. For each active line the net (excitation - deexcitation) cooling
 * reduces analytically (the deexcitation exp(dE/kT) cancels the excitation
 * exp(-dE/kT)) to  (n_e/sqrtTe) * [a*exp(-beta/Te) - b],  with
 *   a = dE*coeff*n_lo/g_lo  (excitation),  b = dE*coeff*n_up/g_up (deexcitation).
 * Both a,b are T_e-independent (lagged pops), so only one exp per line per eval.
 * When nonneg!=0, each per-line term is floored at 0 so spurious NLTE level
 * inversions cannot turn this coolant into a (numerical) heating term. */
static double radeq_line_cool(double T_e, double n_e,
                              const double *a, const double *b,
                              const double *beta, long n_active, int nonneg) {
    double sqrtTe = sqrt(T_e);
    double sum = 0.0;
    for (long m = 0; m < n_active; m++) {
        double br = a[m] * exp(-beta[m] / T_e) - b[m];
        if (nonneg && br < 0.0) br = 0.0;
        sum += br;
    }
    return n_e / sqrtTe * sum;
}

/* Sobolev escape probability beta_esc(tau) = (1-e^-tau)/tau (Castor 1970),
 * with the optically-thin limit beta_esc(0)=1 handled smoothly. */
static double radeq_beta_esc(double tau) {
    if (tau <= 1e-6) return 1.0;
    if (tau > 700.0) return 1.0 / tau;          /* avoid exp underflow; beta->1/tau */
    return (1.0 - exp(-tau)) / tau;
}

/* A3 incr-1 / Phase-1: diagonal-Λ* radiation response (heating side). The lagged
 * H_photo used the frozen binned J_ν; this lets the bf-absorbed field follow the
 * trial T_e via the ALI linearization ΔJ_b = Λ*_b·ε_b·(B_ν(T_e)−blag_b), where
 * Λ*_b=∂J/∂S is the formal-solve diagonal operator and ε_b=χ_abs/χ_tot the thermal
 * absorption fraction. The caller folds Λ*·ε into lstar[bb], so W_lstar=1 in the
 * faithful pure-CMFGEN path (the τ-proxy fallback passes its own scale). gbin[bb]=
 * Σ n_lev·(4πσ f_above dν) is the SAME per-bin photo-weight that built H_photo.
 * FAITHFUL ANCHOR: the caller sets blag_b = J*_b (the frozen per-bin mean intensity
 * that built H_photo) in the pure-CMFGEN path, so bf-net + H_resp collapses to
 * Σ_bb gbin·(1−Λ*ε)·(J*−B_ν(T_e)) — the streaming fraction keeps a non-zero
 * restoring slope dr/dT_e<0 → thick limit thermalizes T_e to the J* color temp.
 * (Anchoring at B_ν(Te_lag), as the τ-proxy fallback still does, makes the pair
 * cancel identically and pins T_e at the seed.) gbin==NULL → 0 (byte-identical). */
static double radeq_Hresp(double T_e, const double *gbin, const double *lstar,
                          const double *blag, double W_lstar,
                          const double *nu_mid, int nbins) {
    if (!gbin) return 0.0;
    double H_resp = 0.0;
    for (int bb = 0; bb < nbins; bb++) {
        if (gbin[bb] == 0.0 || lstar[bb] == 0.0) continue;
        double dB = planck_bnu(T_e, nu_mid[bb]) - blag[bb];
        H_resp += gbin[bb] * lstar[bb] * W_lstar * dB;
    }
    return H_resp;
}

/* A3 (A): ETLA T_e-responsive bound-bound cooling. The signed/floored/escape forms
 * all feed a LAGGED upper population n_up into the cooling sum, so inside the Newton
 * the cooling does NOT track the trial T_e — a lagged non-SE inversion flips it to
 * spurious heating (signed runaway) or the escape form omits the B_lu·J̄ absorption
 * source (catastrophic over-cool). Here n_up is recomputed in LOCAL statistical
 * equilibrium at the trial (T_e,n_e) of the equivalent two-level atom:
 *   n_up = n_lo·(C_lu + R_lu)/(C_ul + R_ul),
 * with C_lu=n_e·q_lu, C_ul=n_e·q_ul the (same-coeff) collisional rates and
 * R_lu=B_lu·J̄·β_esc, R_ul=(A_ul+B_ul·J̄)·β_esc the Sobolev-escape radiative rates.
 * Net collisional cooling C = n_e·Σ dE·(n_lo·q_lu − n_up·q_ul). NOTE this is
 * SIGNED: it cools iff J̄≤B_ν(T_e) and would flip to absorption-pumped line
 * HEATING for a super-thermal J̄ (q_lu·R_ul < q_ul·R_lu). The lagged binned-MC J̄
 * is known over-hard/noisy at the iron-curtain UV edge in exactly the sh11-24
 * transition shells, which could re-arm the T_e runaway through radiative pumping.
 * Faithful guard: cap n_up at its local Boltzmann ceiling n_lo·(g_up/g_lo)·e^{−β/Te}
 * — the exact no-line-heating constraint. It preserves all genuine sub-thermal
 * cooling and the LTE limit (capped net = 0 there), and only floors the spurious
 * pumping-heating branch. A[m]=coeff/g_lo, B[m]=coeff/g_up so A/B=g_up/g_lo
 * (q_lu=A·e^{−β/Te}/√Te, q_ul=B/√Te).
 *
 * MODE (hybrid, line_respond==2): the 3-arm batch (164668/9/70) showed pure SE
 * recompute REMOVES the photosphere coolant — where the lagged binned-MC J̄ is
 * super-thermal the SE n_up→Boltzmann zeros the net cooling, so sh0-8 T_e runs up
 * to ~8900K even though signed's lagged near-thermal n_up cooled it correctly to
 * 4402K. Pure ETLA only WINS in the transition (sh14/18 8-11kK→~5kK). The faithful,
 * non-spatial discriminator is the SIGN of the lagged term itself: use the lagged
 * n_up cooling where it is a genuine coolant (≥0, as at the photosphere), and fall
 * back to the capped-SE n_up only where the lagged inversion produces unphysical
 * self-heating (signed<0, as in the transition). nup_lag carries the lagged pop. */
static double radeq_line_cool_etla(double T_e, double n_e,
                                   const double *A, const double *B,
                                   const double *beta, const double *dE,
                                   const double *nlo, const double *nup_lag,
                                   const double *Rlu, const double *Rul,
                                   long n_active, int hybrid) {
    double invsqrt = 1.0 / sqrt(T_e);
    double sum = 0.0;
    for (long m = 0; m < n_active; m++) {
        if (nlo[m] <= 0.0 || B[m] <= 0.0) continue;
        double exb = exp(-beta[m] / T_e);
        double qlu = A[m] * invsqrt * exb;
        double qul = B[m] * invsqrt;
        if (hybrid) {
            double net_lag = dE[m] * (nlo[m] * qlu - nup_lag[m] * qul);
            if (net_lag > 0.0) { sum += net_lag; continue; }  /* genuine coolant */
        }
        double Clu = n_e * qlu, Cul = n_e * qul;
        double denom = Cul + Rul[m];
        if (denom <= 0.0) continue;
        double nup = nlo[m] * (Clu + Rlu[m]) / denom;
        double nup_lte = nlo[m] * (A[m] / B[m]) * exb;  /* Boltzmann ceiling */
        if (nup > nup_lte) nup = nup_lte;               /* no line-pumped heating */
        sum += dE[m] * (nlo[m] * qlu - nup * qul);
    }
    return n_e * sum;
}

/* ---- Option-2 integral radiative equilibrium: CMFGEN line opacity/source ----
 * Registered once per CMFGEN outer iteration. All arrays are lagged at the
 * assemble-time T_e (Te_lag); chi_line/chi_abs/chi_tot/S_fixed/J are
 * [n_shells*n_bins], nu/dnu are [n_bins]. NULL chi_line disables the term. */
static const double *g_lre_chi_line = NULL, *g_lre_chi_abs = NULL,
                    *g_lre_chi_tot = NULL, *g_lre_S_fixed = NULL,
                    *g_lre_J = NULL, *g_lre_nu = NULL, *g_lre_dnu = NULL,
                    *g_lre_lambda_star = NULL;
static int g_lre_nshells = 0, g_lre_nbins = 0;
/* assemble-time T_e SNAPSHOT (copied, not aliased): the pre-Newton bisection
 * rewrites plasma->T_e between registration and the coupled Newton, so reading
 * plasma->T_e there yields the WRONG lag for the eta_lag subtraction in
 * radeq_line_re wherever the bisection moved T_e. */
static double *g_lre_te_lag = NULL;
static int g_lre_te_lag_n = 0;

void radeq_set_line_re_source(const double *chi_line, const double *chi_abs,
                              const double *chi_tot, const double *S_fixed,
                              const double *J, const double *nu,
                              const double *dnu, const double *lambda_star,
                              const double *T_e_assemble,
                              int n_shells, int n_bins) {
    g_lre_chi_line = chi_line; g_lre_chi_abs = chi_abs;
    g_lre_chi_tot = chi_tot;   g_lre_S_fixed = S_fixed;
    g_lre_J = J; g_lre_nu = nu; g_lre_dnu = dnu;
    g_lre_lambda_star = lambda_star;
    g_lre_nshells = n_shells;  g_lre_nbins = n_bins;
    if (T_e_assemble && n_shells > 0) {
        if (g_lre_te_lag_n != n_shells) {
            free(g_lre_te_lag);
            g_lre_te_lag = (double *)malloc((size_t)n_shells * sizeof(double));
            g_lre_te_lag_n = g_lre_te_lag ? n_shells : 0;
        }
        if (g_lre_te_lag)
            memcpy(g_lre_te_lag, T_e_assemble,
                   (size_t)n_shells * sizeof(double));
    } else {
        free(g_lre_te_lag); g_lre_te_lag = NULL; g_lre_te_lag_n = 0;
    }
}

/* Option-2 radiative line term for one shell, H_line(T_e) = 4π∫χ_line(J−S_eff)dν
 * [erg/s/cm³], added to the HEATING side of radeq_net (J>S ⇒ net heating).
 * T_e-RESPONSIVE (codex form (b)): the lagged NLTE line emissivity is carried,
 * and only the thermal Planck piece is re-evaluated at the trial T_e:
 *   η_lag = S_fixed·χ_tot − χ_abs·B(Te_lag)   (= χ_line·S_line_lag)
 *   η_pre = η_lag + χ_line·(B(Te) − B(Te_lag))
 * so dH_line/dT_e = −4π∫χ_line·dB/dT_e dν ≤ 0 (restoring → root exists).
 * NO-PUMPING CLAMP (floor at B(Te)): η_eff = max(η_pre, χ_line·B(Te)) caps the
 * spurious heating from sub-Planck lagged sources in the thin-UV J≫S bins (the
 * A2 over-heat hazard); LTE (J=B, S_lag=B) gives exactly 0, genuine cooling
 * (S_pre>J) is preserved. */
static double radeq_line_re(double T_e, double Te_lag, int s) {
    if (!g_lre_chi_line || s < 0 || s >= g_lre_nshells) return 0.0;
    int nb = g_lre_nbins;
    const double *cl = &g_lre_chi_line[(size_t)s * nb];
    const double *ca = &g_lre_chi_abs[(size_t)s * nb];
    const double *ct = &g_lre_chi_tot[(size_t)s * nb];
    const double *sf = &g_lre_S_fixed[(size_t)s * nb];
    const double *Jb = &g_lre_J[(size_t)s * nb];
    double H = 0.0;
    for (int b = 0; b < nb; b++) {
        if (cl[b] <= 0.0) continue;
        double nu = g_lre_nu[b];
        double B_lag = planck_bnu(Te_lag, nu);
        double B_te  = planck_bnu(T_e,   nu);
        double eta_lag = sf[b] * ct[b] - ca[b] * B_lag;     /* = χ_line·S_line_lag */
        if (eta_lag < 0.0) eta_lag = 0.0;
        double eta_pre = eta_lag + cl[b] * (B_te - B_lag);
        double eta_flr = cl[b] * B_te;                       /* no-pumping floor */
        double eta_eff = (eta_pre > eta_flr) ? eta_pre : eta_flr;
        H += (cl[b] * Jb[b] - eta_eff) * g_lre_dnu[b];
    }
    return 4.0 * M_PI_VAL * H;
}

static double radeq_net(double T_e, double T_rad, double n_e,
                        double H_photo, double H_gamma,
                        double compton_heat_coef, double ff_coef,
                        double Gamma_ad,
                        const double *a, const double *b, const double *beta,
                        long n_active, int nonneg, double C_line_const,
                        const double *emit_nu, const double *nu_mid, int nbins) {
    double H = H_photo + H_gamma + compton_heat_coef * (T_rad - T_e);

    double C = ff_coef * sqrt(T_e);                     /* free-free emission */
    C += 1.5 * n_e * K_BOLTZMANN * T_e * Gamma_ad;      /* adiabatic expansion */

    /* Bound-free radiative cooling (detailed-balance emission half; cancels
     * H_photo bin-by-bin at LTE since both carry n_lev over the same ν-grid) */
    C += radeq_recomb_cool(T_e, emit_nu, nu_mid, nbins);

    /* Bound-bound (line) cooling: T_e-independent radiative-escape constant when
     * the escape form is active (C_line_const), else the T_e-dependent
     * collisional-difference form over the active line set. */
    C += C_line_const;
    C += radeq_line_cool(T_e, n_e, a, b, beta, n_active, nonneg);

    return H - C;
}

void compute_radiative_equilibrium_te(PlasmaState *plasma, GammaDeposition *gamma_dep,
                                      NLTEConfig *nlte, AtomicData *atom,
                                      OpacityState *opacity,
                                      double time_explosion, int n_shells) {
    if (nlte == NULL || nlte->nlte_level_populations == NULL ||
        nlte->J_nu == NULL) {
        /* No lagged NLTE state yet (pre-NLTE iters) → ratio fallback. */
        for (int s = 0; s < n_shells; s++)
            plasma->T_e[s] = plasma->T_e_T_rad_ratio * plasma->T_rad[s];
        return;
    }
    build_radeq_line_table(nlte, atom, opacity);

    const double GFF = 1.2;            /* thermally averaged free-free Gaunt */
    const double FF_COEF = 1.426e-27;  /* erg cm^3 s^-1 K^-1/2 */
    double Gamma_ad = 2.0 / time_explosion;
    int use_cmfgen = atom->cmfgen_loaded &&
                     atom->cmfgen_n_freq_bins == nlte->n_freq_bins;

    /* Under-relaxation: blend solved T_e with persisted plasma->T_e[s] to damp
     * the floor↔ceiling oscillation. LUMINA_RADEQ_DAMP=η (default 0.5). */
    double radeq_damp = 0.5;
    {
        const char *de = getenv("LUMINA_RADEQ_DAMP");
        if (de) radeq_damp = atof(de);
        if (radeq_damp < 0.0) radeq_damp = 0.0;
        if (radeq_damp > 1.0) radeq_damp = 1.0;
    }

    /* Floor per-line collisional cooling at 0 so spurious NLTE level inversions
     * cannot flip the dominant outer-ejecta coolant into a heating term. ON by
     * default; set LUMINA_RADEQ_COOL_NONNEG=0 to allow signed (inversion) terms. */
    int cool_nonneg = 1;
    {
        const char *cn = getenv("LUMINA_RADEQ_COOL_NONNEG");
        if (cn) cool_nonneg = atoi(cn);
    }

    /* Restrict collisional bound-bound cooling to NLTE-tracked levels, whose
     * populations satisfy statistical equilibrium so the excitation/deexcitation
     * net self-cancels at LTE. The ~2.5M untracked lines carry only lagged
     * dilute-Boltzmann(T_rad) pops that are NOT in SE with the solved T_e, so their
     * a*exp(-beta/Te)-b residual (clipped by the nonneg floor) becomes a spurious
     * one-way coolant that over-cools the photosphere (C_coll swamps H_photo ~20x
     * at shell 0). Their radiative line energy is already carried by the macro-atom
     * transport, so dropping them from the gas-energy balance removes a double count.
     * Default ON (faithful); set LUMINA_RADEQ_COOL_NLTE_ONLY=0 for the old all-lines
     * behavior. */
    int cool_nlte_only = 1;
    {
        const char *co = getenv("LUMINA_RADEQ_COOL_NLTE_ONLY");
        if (co) cool_nlte_only = atoi(co);
    }

    /* Radiative-escape form of the bound-bound gas cooling (physics-verified
     * faithful form). The collisional-difference cooling (a*exp(-beta/Te)-b) is
     * the gas energy loss ONLY in exact statistical equilibrium; with lagged
     * (non-SE) populations the per-line net is biased positive and the nonneg
     * floor rectifies it into a large phantom coolant that over-cools the
     * photosphere. The conserved-energy identity in SE,
     *   (n_lo C_lu - n_up C_ul) dE = n_up A_ul beta_esc dE,
     * lets us instead use the manifestly non-negative radiative-escape form
     *   C_bb = sum_lines beta_esc(tau_sobolev) * A_ul * n_up * dE,
     * with beta_esc(tau)=(1-e^-tau)/tau (Castor 1970). beta_esc multiplies the
     * RADIATIVE (A_ul n_up) term, never a collisional difference; thick lines
     * (tau>>1 -> beta_esc->1/tau) are correctly trapped and do not cool the gas.
     * No double-count: the macro-atom transport here is purely radiative
     * (LUMINA_KPACKET off) so it exchanges no collisional energy with the
     * electron pool; this term is the sole owner of bb gas cooling. Default ON;
     * set LUMINA_RADEQ_COOL_ESCAPE=0 for the old SE-only collisional form. */
    int cool_escape = 1;
    {
        const char *ce = getenv("LUMINA_RADEQ_COOL_ESCAPE");
        if (ce) cool_escape = atoi(ce);
    }

    /* Option-2 integral radiative equilibrium (LUMINA_RADEQ_LINE_RE=1): replace
     * the collisional/escape bound-bound cooling with the T_e-responsive
     * radiative line term 4π∫χ_line(J−S_l)dν over the registered CMFGEN line
     * opacity/source. Owns the outer thin shells the Newton skips (H_photo→0),
     * which is where the +200% blow-up lives. OFF → byte-identical. */
    int line_re = 0;
    { const char *lr = getenv("LUMINA_RADEQ_LINE_RE"); if (lr) line_re = atoi(lr); }
    line_re = line_re && g_lre_chi_line && g_lre_nshells == n_shells;

    /* B2 hybrid closure: the term-by-term heating=cooling bisection is a valid
     * T_e closure ONLY where a real heating term anchors it. In the optically-thin
     * frozen-in outer ejecta photoheating collapses (H_photo→0, no γ-deposition),
     * so the bisection is ill-posed and runs T_e up to a spurious 6000-8400 K.
     * There the ionization is frozen (τ_rec/t_exp ≳ 1, Dessart&Hillier 2008
     * τ_rec/t_exp ∝ t²) and the electrons couple to the trapped radiation field,
     * so T_e tracks the dilute-Planck color temperature: T_e = ratio·T_rad
     * (TARDIS prescription, Kerzendorf&Sim 2014). We switch on the SAME
     * τ_rec/t_exp criterion that parameterizes the frozen-in IONIZATION, so the
     * two are one physical statement. Inner/photosphere (τ_rec/t_exp < thr) keeps
     * the thermalization-anchored bisection. Default OFF; LUMINA_RADEQ_HYBRID=1.
     * Threshold via LUMINA_RADEQ_HYBRID_TAUREC (default 1.0). */
    int radeq_hybrid = 0;
    double hybrid_taurec_thr = 1.0;
    {
        const char *hy = getenv("LUMINA_RADEQ_HYBRID");
        if (hy) radeq_hybrid = atoi(hy);
        const char *ht = getenv("LUMINA_RADEQ_HYBRID_TAUREC");
        if (ht) hybrid_taurec_thr = atof(ht);
    }
    /* Frozen-zone T_e closure mode (LUMINA_RADEQ_HYBRID_MODE):
     *   0 "color"   : T_e = ratio·T_rad — the trapped dilute-Planck color temp
     *                 (TARDIS form). Valid at τ≳1, but in the thin outer zone the
     *                 matter DECOUPLES below the radiation color temp, so this
     *                 sits ~1000 K high vs CMFGEN's 2505 K.
     *   1 "nebular" : local heating=cooling balance against the DILUTE field
     *                 W·B_ν(T_rad) (not the shot-noise-collapsed MC J_ν). Photo-
     *                 heating ∝ W·B(T_rad) but cooling ∝ full local emissivity, so
     *                 T_e decouples freely BELOW T_rad → lands near the gas temp.
     *   2 "adiab"   : pure adiabatic frozen-T, T_e=T_e(t_0)·(t_0/t)^{3(γ-1)},
     *                 γ=5/3, anchored at the frozen-in t_0=√(ᾱ n_e t_exp³). Without
     *                 a residual radiative heating floor this collapses toward 0,
     *                 demonstrating the floor IS the nebular term (mode 1).
     *   3 "blanket" : nebular balance but heated by a BLANKETED expansion-opacity
     *                 source J_eff(ν) instead of the bare dilute Planck W·B_ν(T_rad).
     *                 Per ν-bin: J_eff = β_bin·W·B_ν(T_rad) + (1−β_bin)·S̄_bin, where
     *                 β_bin=(1−e^{−τ_bin})/τ_bin from the OVERLAP-summed Sobolev depth
     *                 of all lines in the bin (line overlap, NOT Σ independent
     *                 sources), and S̄_bin is the (1−e^{−τ})-weighted mean of the NLTE
     *                 line source fns. Retains the iron-curtain non-Planckian shape;
     *                 Stage-0 de-risk of whether blanketing alone cures nebular's +8%
     *                 over-heat (Karp+77, Eastman&Pinto93, Pinto&Eastman00). */
    int hybrid_mode = 0;
    {
        const char *hm = getenv("LUMINA_RADEQ_HYBRID_MODE");
        if (hm) {
            if (!strcmp(hm, "nebular") || !strcmp(hm, "1")) hybrid_mode = 1;
            else if (!strcmp(hm, "adiab") || !strcmp(hm, "2")) hybrid_mode = 2;
            else if (!strcmp(hm, "blanket") || !strcmp(hm, "3")) hybrid_mode = 3;
            else hybrid_mode = 0;
        }
    }
    long n_frozen = 0;   /* count of shells set by the frozen color-temp branch */
    /* Fallback-visibility counters (no physics change): the bisection has no
     * bracketed root when f_lo<=0 (cooling dominates even cold -> Tlo) or
     * f_hi>=0 (heating dominates even at 2*T_rad -> Thi, the +200% spike
     * source), and a 1000K-floored T_e is under-determined. Surface them. */
    long n_pin_lo = 0, n_pin_hi = 0, n_floor_bis = 0;

    /* Per-shell bound-free EMISSION cooling integrand, accumulated on the SAME
     * frequency grid used for the photoheating J_ν integral so the two cancel
     * bin-by-bin at LTE. emit_nu[bb] = Σ_l n_lev·4π(2hν³/c²)σ_bf f_above dν
     * (T_e-independent); the e^{−hν/kT_e} Wien factor is applied per bisection
     * eval in radeq_recomb_cool. nu_mid[bb] = bin-center frequency (grid-only). */
    int nfb = nlte->n_freq_bins;
    double *emit_nu = (double *)malloc((size_t)nfb * sizeof(double));
    double *nu_mid  = (double *)malloc((size_t)nfb * sizeof(double));
    for (int bb = 0; bb < nfb; bb++) {
        double lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
        nu_mid[bb] = exp(lo + 0.5 * nlte->d_log_nu);
    }

    /* Blanketed expansion-opacity heating field J_eff[ν-bin] (mode 3 only): the
     * line-overlap-summed Sobolev escape blended with the NLTE line sources. Built
     * once per shell by binning the line list onto the heating ν-grid. */
    double *Jeff_blanket = NULL, *bl_tau = NULL, *bl_w = NULL, *bl_wS = NULL;
    if (hybrid_mode == 3) {
        Jeff_blanket = (double *)malloc((size_t)nfb * sizeof(double));
        bl_tau = (double *)malloc((size_t)nfb * sizeof(double));
        bl_w   = (double *)malloc((size_t)nfb * sizeof(double));
        bl_wS  = (double *)malloc((size_t)nfb * sizeof(double));
    }

    /* Per-shell compacted (active) collisional-cooling coefficients:
     *   a[m] = dE*coeff*n_lo/g_lo,  b[m] = dE*coeff*n_up/g_up,  beta[m] = dE/k_B.
     * Only lines with a nonzero lower/upper population are kept (active), which
     * prunes the bulk of high-lying levels whose Boltzmann pop underflows. */
    size_t nl_alloc = (size_t)(radeq_n_lines > 0 ? radeq_n_lines : 1);
    double *ca   = (double *)malloc(nl_alloc * sizeof(double));
    double *cb   = (double *)malloc(nl_alloc * sizeof(double));
    double *cbet = (double *)malloc(nl_alloc * sizeof(double));

    for (int s = 0; s < n_shells; s++) {
        double T_rad = plasma->T_rad[s];
        double W     = plasma->W[s];
        double n_e   = plasma->n_electron[s];
        if (T_rad <= 0.0 || n_e <= 0.0) {
            plasma->T_e[s] = plasma->T_e_T_rad_ratio * T_rad;
            continue;
        }
        /* Te_lag = assemble-time T_e of the registered CMFGEN line opacity:
         * the registration SNAPSHOT when present (immune to earlier solvers
         * rewriting plasma->T_e), else plasma->T_e[s] at entry. */
        double Te_lag = (g_lre_te_lag && s < g_lre_te_lag_n &&
                         g_lre_te_lag[s] > 100.0) ? g_lre_te_lag[s]
                      : (plasma->T_e[s] > 100.0) ? plasma->T_e[s]
                        : plasma->T_e_T_rad_ratio * T_rad;

        /* ---- bound-free heating (n_lev·J_ν) and the matching emission-cooling
         *      integrand (n_lev·B_ν), both accumulated on the same ν-grid so the
         *      detailed-balance pair cancels bin-by-bin at LTE ---- */
        double H_photo = 0.0;
        double H_photo_dilute = 0.0;   /* same bf heating but vs W·B_ν(T_rad) field */
        double H_photo_blanket = 0.0;  /* bf heating vs blanketed J_eff(ν) (mode 3) */
        double dbg_nlev_bf = 0.0;      /* Σ n_lev·σ_bf(ν_th): bf-absorber capacity */
        double dbg_Jth = 0.0;          /* Σ n_lev·σ_bf(ν_th)·J(ν_th): edge-weighted J */
        double beta_rad_h = 1.0 / (K_BOLTZMANN * T_rad);
        for (int bb = 0; bb < nfb; bb++) emit_nu[bb] = 0.0;

        /* ---- Blanketed expansion-opacity heating field J_eff[ν-bin] (mode 3) ----
         * Bin every line onto the heating ν-grid; per bin form the OVERLAP-summed
         * Sobolev escape β_bin=(1−e^{−Στ})/Στ and the (1−e^{−τ})-weighted mean NLTE
         * line source S̄_bin, then J_eff = β_bin·W·B_ν(T_rad) + (1−β_bin)·S̄_bin.
         * Line-thick bins (iron curtain) thermalize to the local line sources; thin
         * bins see the penetrating diluted photospheric field. Built once per shell. */
        if (hybrid_mode == 3 && opacity != NULL && opacity->tau_sobolev != NULL) {
            int onl = opacity->n_lines;
            for (int bb = 0; bb < nfb; bb++) { bl_tau[bb] = 0.0; bl_w[bb] = 0.0; bl_wS[bb] = 0.0; }
            double inv_dln = 1.0 / nlte->d_log_nu;
            double lnu_min = log(nlte->nu_min);
            for (int l = 0; l < onl; l++) {
                double tau_l = opacity->tau_sobolev[(size_t)l * n_shells + s];
                if (tau_l <= 0.0) continue;
                double nu_l = opacity->line_list_nu[l];
                if (nu_l <= nlte->nu_min) continue;
                int bb = (int)((log(nu_l) - lnu_min) * inv_dln);
                if (bb < 0 || bb >= nfb) continue;
                double w = 1.0 - exp(-tau_l);   /* line absorption fraction */
                double S_l = (opacity->line_source_S != NULL)
                           ? opacity->line_source_S[(size_t)l * n_shells + s] : 0.0;
                if (S_l <= 0.0) S_l = W * planck_bnu(T_rad, nu_l);  /* dilute fallback */
                bl_tau[bb] += tau_l;
                bl_w[bb]   += w;
                bl_wS[bb]  += w * S_l;
            }
            for (int bb = 0; bb < nfb; bb++) {
                double nu_bin = nu_mid[bb];
                double WB = W * planck_bnu(T_rad, nu_bin);
                if (bl_w[bb] <= 0.0) { Jeff_blanket[bb] = WB; continue; }
                double tb = bl_tau[bb];
                double beta_bin = (tb > 1.0e-6) ? (1.0 - exp(-tb)) / tb : 1.0;
                double Sbar = bl_wS[bb] / bl_w[bb];
                Jeff_blanket[bb] = beta_bin * WB + (1.0 - beta_bin) * Sbar;
            }
        }
        for (int i = 0; i < nlte->n_nlte_ions; i++) {
            int Z = nlte->nlte_Z[i];
            int ion_stage = nlte->nlte_ion[i];
            double chi_erg = find_ioniz_energy(atom, Z, ion_stage) * EV_TO_ERG;
            if (chi_erg <= 0.0) continue;
            double sigma0 = get_bf_sigma0(Z, ion_stage);
            if (sigma0 <= 0.0) {
                int Zeff = Z - ion_stage; if (Zeff < 1) Zeff = 1;
                sigma0 = 7.91e-18 / ((double)Zeff * (double)Zeff);
            }
            int ls = nlte->nlte_ion_level_offset[i];
            int le = nlte->nlte_ion_level_offset[i + 1];
            for (int l = ls; l < le; l++) {
                int g = nlte->nlte_to_global_level[l];
                double E_lev = atom->level_energy_eV[g] * EV_TO_ERG;
                double nu_th = (chi_erg - E_lev) / H_PLANCK;
                if (nu_th <= 0.0) continue;
                double n_lev = nlte->nlte_level_populations[(size_t)l * n_shells + s];
                if (n_lev <= 0.0) continue;
                int has = use_cmfgen && atom->cmfgen_has_sigma[g];
                const double *srow = has ?
                    &atom->cmfgen_sigma_bf[(size_t)g * atom->cmfgen_n_freq_bins] : NULL;
                double integ = 0.0;
                double integ_dilute = 0.0;
                double integ_blanket = 0.0;
                int edge_done = 0;
                for (int bb = 0; bb < nfb; bb++) {
                    double nu_bin = nu_mid[bb];
                    if (nu_bin < nu_th) continue;
                    double lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
                    double dnu = exp(lo + nlte->d_log_nu) - exp(lo);
                    double sig = srow ? srow[bb] : sigma0 * pow(nu_th / nu_bin, 3.0);
                    if (sig <= 0.0) continue;
                    double f_above = 1.0 - nu_th / nu_bin;
                    double geom = 4.0 * M_PI_VAL * sig * f_above * dnu;
                    double bnu_pref = 2.0 * H_PLANCK * nu_bin * nu_bin * nu_bin /
                                      (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT);
                    double J = nlte->J_nu[(size_t)s * nlte->n_freq_bins + bb];
                    integ += geom * J;                          /* heating: J_ν */
                    if (!edge_done) {   /* edge (threshold-crossing) bf capacity */
                        dbg_nlev_bf += n_lev * sig;
                        dbg_Jth     += n_lev * sig * J;
                        edge_done = 1;
                    }
                    /* dilute-Planck heating: W·B_ν(T_rad), the nebular field the
                     * decoupled frozen gas actually sees (Mazzali&Lucy 1993). */
                    double x = H_PLANCK * nu_bin * beta_rad_h;
                    if (x < 500.0) {
                        double bnu = bnu_pref / (exp(x) - 1.0);
                        integ_dilute += geom * W * bnu;
                    }
                    if (hybrid_mode == 3)
                        integ_blanket += geom * Jeff_blanket[bb];
                    emit_nu[bb] += n_lev * geom * bnu_pref;     /* cooling: B_ν Wien */
                }
                H_photo += n_lev * integ;
                H_photo_dilute += n_lev * integ_dilute;
                H_photo_blanket += n_lev * integ_blanket;
            }
        }

        double H_gamma = (gamma_dep && gamma_dep->heating_rate &&
                          gamma_dep->heating_rate[s] > 0.0) ?
                          gamma_dep->heating_rate[s] : 0.0;

        double u_rad = 4.0 * W * SIGMA_SB * T_rad * T_rad * T_rad * T_rad /
                       C_SPEED_OF_LIGHT;
        double Gamma_C = 8.0 * SIGMA_THOMSON * u_rad /
                         (3.0 * M_ELECTRON * C_SPEED_OF_LIGHT);
        double compton_heat_coef = 1.5 * n_e * K_BOLTZMANN * Gamma_C; /* ×(T_rad-T_e) */
        double ff_coef = FF_COEF * GFF * n_e * n_e;                   /* ×sqrt(T_e) */

        /* Build the active collisional-cooling coefficient arrays for this shell.
         * Per-level pop = NLTE pop where the level is tracked, else dilute-Boltzmann
         * (nebular), identical to the Sobolev opacity formula:
         *   n_k = (meta?1:W) * (g_k/U) * n_ion * exp(-E_k/kT_rad). */
        double beta_rad = 1.0 / (K_BOLTZMANN * T_rad);
        double C_bb_esc = 0.0;   /* radiative-escape bound-bound cooling (T_e-indep) */
        long n_active = 0;
        /* Option-2: the radiative line term owns bb; skip the collisional/escape
         * assembly so C_bb_esc and n_active stay 0. */
        for (long k = 0; !line_re && k < radeq_n_lines; k++) {
            const RadEqLine *rl = &radeq_lines[k];
            if (cool_nlte_only && (rl->nlte_lo < 0 || rl->nlte_up < 0))
                continue;   /* SE-only: skip untracked lagged-Boltzmann lines */
            double n_ion = atom->ion_number_density[(size_t)rl->ip * n_shells + s];
            double U     = atom->partition_functions[(size_t)rl->ip * n_shells + s];
            if (U <= 0.0) U = 1.0;
            double nlo_k, nup_k;
            if (rl->nlte_lo >= 0) {
                nlo_k = nlte->nlte_level_populations[(size_t)rl->nlte_lo * n_shells + s];
            } else {
                double bz = atom->level_energy_eV[rl->lo_g] * EV_TO_ERG * beta_rad;
                double wt = atom->level_metastable[rl->lo_g] ? 1.0 : W;
                nlo_k = (bz < 500.0) ? n_ion * wt * rl->g_lo * exp(-bz) / U : 0.0;
            }
            if (rl->nlte_up >= 0) {
                nup_k = nlte->nlte_level_populations[(size_t)rl->nlte_up * n_shells + s];
            } else {
                double bz = atom->level_energy_eV[rl->up_g] * EV_TO_ERG * beta_rad;
                double wt = atom->level_metastable[rl->up_g] ? 1.0 : W;
                nup_k = (bz < 500.0) ? n_ion * wt * rl->g_up * exp(-bz) / U : 0.0;
            }
            if (nlo_k <= 0.0 && nup_k <= 0.0) continue;   /* inactive: no population */
            if (cool_escape) {
                /* Radiative-escape form: beta_esc(tau)*A_ul*n_up*dE (>=0 always). */
                if (nup_k > 0.0 && rl->A_ul > 0.0 && opacity->tau_sobolev) {
                    double tau = opacity->tau_sobolev[(size_t)rl->line * n_shells + s];
                    C_bb_esc += radeq_beta_esc(tau) * rl->A_ul * nup_k * rl->dE;
                }
                continue;   /* escape form replaces the collisional-difference arrays */
            }
            ca[n_active]   = rl->dE * rl->coeff * nlo_k / rl->g_lo;
            cb[n_active]   = rl->dE * rl->coeff * nup_k / rl->g_up;
            cbet[n_active] = rl->beta;
            n_active++;
        }

        /* Diagnostic: term-by-term breakdown at T_e=T_rad (Compton term = 0). */
        if (getenv("LUMINA_RADEQ_DIAG") &&
            (s == 0 || s == n_shells / 2 || s == (3 * n_shells) / 4 ||
             s == (7 * n_shells) / 8 || s == n_shells - 1)) {
            double Tt = T_rad, sqTt = sqrt(Tt);
            double C_ff = ff_coef * sqTt;
            double C_ad = 1.5 * n_e * K_BOLTZMANN * Tt * Gamma_ad;
            double C_rec = radeq_recomb_cool(Tt, emit_nu, nu_mid, nfb);
            double C_coll = cool_escape ? C_bb_esc :
                            radeq_line_cool(Tt, n_e, ca, cb, cbet, n_active, cool_nonneg);
            printf("  [RADEQ-DIAG s=%d] n_e=%.2e Trad=%.0f nact=%ld | H_photo=%.3e H_gamma=%.3e"
                   " | C_ff=%.3e C_ad=%.3e C_rec=%.3e C_bb=%.3e (%s)\n",
                   s, n_e, T_rad, n_active, H_photo, H_gamma, C_ff, C_ad, C_rec, C_coll,
                   cool_escape ? "escape" : "collis");
            /* A-latent discriminator: if dbg_nlev_bf≈0 the inflated thin-UV J has no
             * bf absorbers to act on (Defect A latent); if it is healthy yet H_photo
             * stays tiny, the J isn't reaching the edges. */
            printf("  [RADEQ-BFCAP s=%d] nlev_bf=%.3e Jth_wt=%.3e H_photo=%.3e\n",
                   s, dbg_nlev_bf, dbg_Jth, H_photo);
        }

        /* Bootstrap guard: with no photoionization/gamma heating estimate
         * (e.g. NLTE pops not yet populated, or J_nu still zero on the first
         * NLTE iter), the balance has only Compton heating vs. full radiative
         * cooling and collapses to the floor. Fall back to ratio T_e until a
         * real heating estimate is available. */
        if (H_photo <= 0.0 && H_gamma <= 0.0) {
            plasma->T_e[s] = plasma->T_e_T_rad_ratio * T_rad;
            continue;
        }

        /* ---- B2 hybrid frozen-zone branch ----
         * Where the recombination timescale exceeds the expansion time the gas is
         * frozen-in and the heating=cooling bisection has no anchor; set T_e from
         * the trapped-radiation color temperature instead. α_rec here is a
         * representative coefficient used ONLY to locate the freeze-out crossover
         * (a CRITERION, not a rate entering the energy balance), so an O(1)
         * ambiguity in its value is immaterial. */
        if (radeq_hybrid) {
            double alpha_rec = 2.6e-13 * pow(T_rad / 1.0e4, -0.8);  /* cm^3 s^-1 */
            double tau_rec = (n_e > 0.0 && alpha_rec > 0.0) ?
                             1.0 / (n_e * alpha_rec) : 0.0;
            if (tau_rec >= hybrid_taurec_thr * time_explosion) {
                double T_e;
                if (hybrid_mode == 1 || hybrid_mode == 3) {
                    /* nebular(1): heating=cooling vs the bare dilute W·B_ν(T_rad).
                     * blanket(3): same balance but vs the blanketed expansion-opacity
                     * field J_eff(ν) — keeps the iron-curtain shape. Both let T_e
                     * decouple freely below T_rad. Wide low range. */
                    double H_src = (hybrid_mode == 3) ? H_photo_blanket : H_photo_dilute;
                    double Tlo = 0.02 * T_rad, Thi = T_rad;
                    double f_lo = radeq_net(Tlo, T_rad, n_e, H_src, H_gamma,
                                            compton_heat_coef, ff_coef, Gamma_ad,
                                            ca, cb, cbet, n_active, cool_nonneg,
                                            C_bb_esc, emit_nu, nu_mid, nfb);
                    double f_hi = radeq_net(Thi, T_rad, n_e, H_src, H_gamma,
                                            compton_heat_coef, ff_coef, Gamma_ad,
                                            ca, cb, cbet, n_active, cool_nonneg,
                                            C_bb_esc, emit_nu, nu_mid, nfb);
                    if (f_lo <= 0.0)      T_e = Tlo;
                    else if (f_hi >= 0.0) T_e = Thi;
                    else {
                        for (int it = 0; it < 60 && (Thi - Tlo) > 1.0; it++) {
                            double Tm = 0.5 * (Tlo + Thi);
                            double fm = radeq_net(Tm, T_rad, n_e, H_src, H_gamma,
                                                  compton_heat_coef, ff_coef, Gamma_ad,
                                                  ca, cb, cbet, n_active, cool_nonneg,
                                                  C_bb_esc, emit_nu, nu_mid, nfb);
                            if (fm > 0.0) Tlo = Tm; else Thi = Tm;
                        }
                        T_e = 0.5 * (Tlo + Thi);
                    }
                    if (T_e < 500.0) T_e = 500.0;
                } else if (hybrid_mode == 2) {
                    /* adiabatic frozen-T: T_e(t)=T_e(t_0)·(t_0/t)^{3(γ-1)}, γ=5/3.
                     * t_0=√(ᾱ n_e t_exp³) = same freeze anchor as frozen-in ioniz.;
                     * seed T_e(t_0) from the current color temp (equilibrium proxy). */
                    double t0 = sqrt(alpha_rec * n_e * time_explosion *
                                     time_explosion * time_explosion);
                    double ratio = (t0 > 0.0 && t0 < time_explosion) ?
                                   t0 / time_explosion : 1.0;
                    T_e = T_rad * ratio * ratio;          /* (t_0/t)^2 */
                    if (T_e < 500.0) T_e = 500.0;
                } else {
                    /* color (default): T_e = ratio·T_rad (trapped dilute Planck). */
                    T_e = plasma->T_e_T_rad_ratio * T_rad;
                }
                double T_e_old = plasma->T_e[s];
                if (radeq_damp < 1.0 && T_e_old > 100.0)
                    T_e = radeq_damp * T_e + (1.0 - radeq_damp) * T_e_old;
                plasma->T_e[s] = T_e;
                n_frozen++;
                continue;
            }
        }

        /* ---- bisection on Net(T_e) (monotone decreasing) ---- */
        double Tlo = 0.1 * T_rad, Thi = 2.0 * T_rad;
        double f_lo = radeq_net(Tlo, T_rad, n_e, H_photo, H_gamma,
                                compton_heat_coef, ff_coef, Gamma_ad,
                                ca, cb, cbet, n_active, cool_nonneg, C_bb_esc,
                                emit_nu, nu_mid, nfb)
                    + (line_re ? radeq_line_re(Tlo, Te_lag, s) : 0.0);
        double f_hi = radeq_net(Thi, T_rad, n_e, H_photo, H_gamma,
                                compton_heat_coef, ff_coef, Gamma_ad,
                                ca, cb, cbet, n_active, cool_nonneg, C_bb_esc,
                                emit_nu, nu_mid, nfb)
                    + (line_re ? radeq_line_re(Thi, Te_lag, s) : 0.0);
        double T_e;
        if (f_lo <= 0.0) {
            T_e = Tlo;                 /* cooling dominates even when cold */
            n_pin_lo++;
        } else if (f_hi >= 0.0) {
            T_e = Thi;                 /* heating dominates even when hot */
            n_pin_hi++;
        } else {
            for (int it = 0; it < 60 && (Thi - Tlo) > 1.0; it++) {
                double Tm = 0.5 * (Tlo + Thi);
                double fm = radeq_net(Tm, T_rad, n_e, H_photo, H_gamma,
                                      compton_heat_coef, ff_coef, Gamma_ad,
                                      ca, cb, cbet, n_active, cool_nonneg, C_bb_esc,
                                      emit_nu, nu_mid, nfb)
                          + (line_re ? radeq_line_re(Tm, Te_lag, s) : 0.0);
                if (fm > 0.0) Tlo = Tm; else Thi = Tm;
            }
            T_e = 0.5 * (Tlo + Thi);
        }
        if (T_e < 1000.0) { T_e = 1000.0; n_floor_bis++; }
        /* Under-relaxation against the persisted T_e to damp oscillation. */
        double T_e_old = plasma->T_e[s];
        if (radeq_damp < 1.0 && T_e_old > 100.0)
            T_e = radeq_damp * T_e + (1.0 - radeq_damp) * T_e_old;
        plasma->T_e[s] = T_e;
    }
    free(emit_nu);
    free(nu_mid);
    free(ca);
    free(cb);
    free(cbet);
    free(Jeff_blanket);
    free(bl_tau);
    free(bl_w);
    free(bl_wS);

    printf("  [RADEQ] T_e/T_rad: shell0=%.3f shell%d=%.3f (T_e[0]=%.0f K)\n",
           plasma->T_rad[0] > 0 ? plasma->T_e[0] / plasma->T_rad[0] : 0.0,
           n_shells - 1,
           plasma->T_rad[n_shells - 1] > 0 ?
               plasma->T_e[n_shells - 1] / plasma->T_rad[n_shells - 1] : 0.0,
           plasma->T_e[0]);
    printf("  [RADEQ] NO-ROOT fallback: %ld pinned-hi (T_e=2*T_rad, heating>cool), "
           "%ld pinned-lo (T_e=0.1*T_rad), %ld floor-pinned (1000K) of %d shells\n",
           n_pin_hi, n_pin_lo, n_floor_bis, n_shells);
    if (radeq_hybrid) {
        const char *mname = hybrid_mode == 1 ? "nebular(W·B_ν)" :
                            hybrid_mode == 2 ? "adiab(t_0/t)^2" :
                            hybrid_mode == 3 ? "blanket(J_eff)" : "color(ratio·T_rad)";
        printf("  [RADEQ] hybrid[%s]: %ld/%d outer shells frozen "
               "(τ_rec/t_exp≥%.2f) T_e[last]=%.0f K\n",
               mname, n_frozen, n_shells, hybrid_taurec_thr,
               plasma->T_e[n_shells - 1]);
    }
}

/* ============================================================
 * PATH-A / A2: per-shell COUPLED-NEWTON solve of {n_e, T_e}
 * ------------------------------------------------------------
 * The structural fix for the operator-split runaways. Instead of solving the
 * electron temperature (compute_radiative_equilibrium_te) and the ionization /
 * electron density (compute_plasma_state) sequentially — each holding the other
 * fixed, the Gauss-Seidel fixed point that ping-pongs on the stiff T_e↔n_e
 * coupling — we linearize and solve the TWO residuals SIMULTANEOUSLY per shell:
 *
 *   r1 = Net(T_e, n_e)              integral radiative equilibrium (HD2012 eq.27),
 *                                   reusing the validated radeq_net residual
 *   r2 = n_e − Σ_ion Z·n_ion(T_e,n_e)  charge conservation, reusing the code's own
 *                                   nebular-Saha ionization (which already depends
 *                                   on BOTH T_e and T_rad)
 *
 * by Newton-Raphson with a 2×2 numerical Jacobian and a damped line search. A
 * perturbation in ionization is met by the linearized T_e response inside the
 * SAME iteration (and vice versa), so the ping-pong is structurally impossible
 * (Hubeny&Lanz 1995; Hillier&Dessart 2012 §3.1). This is increment 1 (the
 * tightest {T_e↔ionization} block); J stays in the outer loop (A3 adds its
 * diagonal local-ALI (Lambda-star) / VEF response), level populations stay
 * slaved to the lagged NLTE
 * solve (A2 increment 2 folds them in), and the frozen outer shells keep their
 * validated frozen-in cascade untouched (only non-frozen inner shells, where the
 * runaway lived, are governed by the coupled block). Env-gated
 * LUMINA_COUPLED_NEWTON=1; OFF = byte-identical sequential path.
 * ============================================================ */

/* Σ_ion (ion_stage)·n_ion summed over all elements for one shell, evaluated at a
 * trial (T_e, n_e). Mirrors compute_ion_populations' nebular-Saha exactly (same
 * phi_neb, same zeta/ML/two-comp corrections) so the coupled residual uses the
 * code's own ionization physics — only solved simultaneously with T_e. */
static double coupled_charge_density(AtomicData *atom, PlasmaState *plasma,
                                     int s, double T_e, double n_e, int n_shells) {
    double T_rad = plasma->T_rad[s];
    double W     = plasma->W[s];
    double rho   = plasma->rho[s];
    if (T_rad <= 0.0 || n_e <= 0.0) return 0.0;
    double g_electron = pow(2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T_rad
                             / (H_PLANCK * H_PLANCK), 1.5);
    double beta_rad = 1.0 / (K_BOLTZMANN * T_rad);
    double beta_electron = 1.0 / (K_BOLTZMANN * T_e);
    double sqrt_te_tr = sqrt(T_e / T_rad);

    double ne_sum = 0.0;
    for (int e = 0; e < atom->n_elements; e++) {
        int Z_elem = atom->element_Z[e];
        double mass_amu = atom->element_mass_amu[e];
        int ip_start = atom->elem_ion_offset[e];
        int ip_end   = atom->elem_ion_offset[e + 1];
        int n_pops   = ip_end - ip_start;
        double abund = atom->abundances[e * n_shells + s];
        double n_element = (abund * rho) / (mass_amu * AMU);
        if (n_element <= 0.0 || n_pops <= 0) continue;

        /* ratios[k] = n_{k+1}/n_k = phi_neb(T_e)/n_e; accumulate the charge moment
         * Σ stage·(n_ion/n_0) and the normalization Σ (n_ion/n_0) in one pass. */
        double sum = 1.0, product = 1.0;
        double r_run = 1.0;       /* running n_k/n_0 */
        double charge = 0.0;      /* Σ stage·(n_ion/n_0), scaled by n_0 below */
        for (int k = 0; k < n_pops; k++) {
            if (k > 0) {
                int ip_cur  = ip_start + k - 1;
                int ip_next = ip_start + k;
                int stage = atom->ion_pop_stage[ip_cur];
                double Z_cur  = atom->partition_functions[ip_cur  * n_shells + s];
                double Z_next = atom->partition_functions[ip_next * n_shells + s];
                if (Z_cur <= 0.0) Z_cur = 1.0;
                if (Z_next <= 0.0) Z_next = 1.0;
                double chi_erg = find_ioniz_energy(atom, Z_elem, stage) * EV_TO_ERG;
                double prefactor = (Z_next / Z_cur) * 2.0 * g_electron;
                double phi_LTE_at_Trad = prefactor * exp(-chi_erg * beta_rad);
                double phi_LTE_at_Te   = prefactor * exp(-chi_erg * beta_electron);
                double zeta = interpolate_zeta(atom, Z_elem, stage, T_rad);
                double phi_neb = W * sqrt_te_tr *
                    (zeta * (T_e / T_rad) * phi_LTE_at_Te +
                     W * (1.0 - zeta) * phi_LTE_at_Trad);
                phi_neb = apply_ml_phi_neb_correction(phi_neb, Z_elem, stage, T_e, T_rad);
                phi_neb = apply_twocomp_lock(phi_neb, phi_LTE_at_Te, Z_elem, stage, W);
                double ratio = phi_neb / n_e;
                if (!isfinite(ratio) || ratio < 0.0) ratio = 0.0;
                if (ratio > 1e30) ratio = 1e30;
                product *= ratio;
                if (product > 1e30) product = 1e30;
                sum += product;
                r_run = product;
            }
            /* stage charge of ip_start+k relative to neutral=0 baseline */
            int stage_k = atom->ion_pop_stage[ip_start + k];
            charge += (double)stage_k * r_run;
        }
        double n_0 = n_element / sum;
        ne_sum += n_0 * charge;
    }
    return ne_sum;
}

/* A2 PRIMARY (J_ν photoionization): effective stage j→j+1 photoionization rate
 * per ion in stage j (ion-pop `ip`),
 *   Γ_j = Σ_l (g_l e^{−E_l/kT_e}/U_j) ∫ 4π J_ν σ_l(ν)/(hν) dν,
 * the LTE-level-weighted, J_ν-driven direct photoionization rate. Replaces the
 * detailed-balance closure Γ=α(T_e)·φ_neb: the dilute non-Planckian field J_ν
 * (already binned by the transport/NLTE solve) drives ionization directly, so
 * there is NO exp(−χ_ion/kT_e) Saha sensitivity (kills the T_e-coupled runaway)
 * and NO wrong-W nebular baseline. In the thick limit J_ν→B(T_e) it recovers
 * α·φ_LTE = Saha, so inner shells are unchanged. Uses the SAME CMFGEN σ_bf levels
 * as frozenin_alpha_rr (grid bin-aligned with nlte->J_nu). Returns −1.0 if the ion
 * has no usable cmfgen σ_bf (caller then falls back to the φ_neb closure). */
static double coupled_photoion_rate_jnu(AtomicData *atom, NLTEConfig *nlte,
                                        int ip, int s, double T_e, int n_shells,
                                        const double *jblend_lstar,
                                        const double *jblend_b, double jblend_W,
                                        double wbfloor_T) {
    (void)n_shells;
    if (!atom->cmfgen_loaded || T_e <= 0.0) return -1.0;
    int Z = atom->ion_pop_Z[ip];
    int stage = atom->ion_pop_stage[ip];
    double chi_ion_eV = find_ioniz_energy(atom, Z, stage);
    if (chi_ion_eV <= 0.0) return -1.0;
    double chi_ion_erg = chi_ion_eV * EV_TO_ERG;
    int nfb = nlte->n_freq_bins;
    double log_numin = log(nlte->nu_min);
    double d_log_nu  = nlte->d_log_nu;
    double kT = K_BOLTZMANN * T_e;
    int g0 = atom->level_offset[ip];
    int g1 = atom->level_offset[ip + 1];
    double U = 0.0, num = 0.0;
    int any_sigma = 0;
    for (int gl = g0; gl < g1; gl++) {
        double E_l_erg = atom->level_energy_eV[gl] * EV_TO_ERG;
        double w_l = (double)atom->level_g[gl] * exp(-E_l_erg / kT);
        if (!isfinite(w_l) || w_l <= 0.0) continue;
        U += w_l;                                  /* partition normalization */
        if (!atom->cmfgen_has_sigma[gl]) continue;
        double chi_l = chi_ion_erg - E_l_erg;      /* binding energy of level l */
        if (chi_l <= 0.0) continue;
        double nu_th = chi_l / H_PLANCK;
        const double *srow = &atom->cmfgen_sigma_bf[(size_t)gl * (size_t)nfb];
        double Rj = 0.0;
        for (int bb = 0; bb < nfb; bb++) {
            double lo = log_numin + bb * d_log_nu;
            double nu_c = exp(lo + 0.5 * d_log_nu);
            if (nu_c < nu_th) continue;
            double sig = srow[bb];
            if (sig <= 0.0) continue;
            double J = nlte->J_nu[(size_t)s * nfb + bb];
            /* B3-1 diagonal-Λ* blend: where the gas is thick to its own bf
             * continuum (Λ*=1−e^{−τ_bf}→1) replace the spuriously-hot lagged
             * non-local FUV field with the local thermal pool W·B_ν(T_e^lag);
             * at the C I/O I edges B_ν is Wien-suppressed so the over-ionizing
             * ionizing flux collapses to the trial T_e. NULL → bare lagged J. */
            if (jblend_lstar) {
                double L = jblend_lstar[bb];
                J = (1.0 - L) * J + L * jblend_W * jblend_b[bb];
            }
            /* DIAGNOSTIC PROBE (LUMINA_COUPLED_JNU_WBFLOOR=<T_inner>): floor
             * the integrand field at the geometrically diluted photospheric
             * Planck W*B_nu(T_inner). Tests the "inner FUV J over-thermalized
             * to local cold T_e -> Gamma 5-1000x low" diagnosis at the rate
             * level. NOT a faithful fix (the faithful fix is line-forest
             * scattering in the formal solver). */
            if (wbfloor_T > 0.0) {
                double Jf = jblend_W * planck_bnu(wbfloor_T, nu_c);
                if (J < Jf) J = Jf;
            }
            if (J <= 0.0) continue;
            double dnu = exp(lo + d_log_nu) - exp(lo);
            Rj += 4.0 * M_PI_VAL * J * sig / (H_PLANCK * nu_c) * dnu;
        }
        num += w_l * Rj;
        any_sigma = 1;
    }
    if (!any_sigma || !(U > 0.0) || !isfinite(num)) return -1.0;
    double gamma = num / U;
    if (!isfinite(gamma) || gamma < 0.0) return -1.0;
    return gamma;
}

/* A2 increment-2: TIME-DEPENDENT charge density for one shell at trial (T_e,n_e).
 *
 * Replaces the steady-state nebular-Saha balance with one implicit backward-Euler
 * step of the per-element ionization rate equation (HD2012 eq.26) over the system
 * age Δt = t_exp, from a fully-ionized (top-stage) initial condition:
 *
 *   (y_k − y_k^init)/Δt = Γ_{k-1} y_{k-1} + α_k n_e y_{k+1}
 *                                          − (Γ_k + α_{k-1} n_e) y_k
 *
 * with y = stage fractions (Σ y = 1), y_init = top stage only. The photoionization
 * rate is closed by DETAILED BALANCE with the validated nebular-Saha:
 * Γ_k = α_k(T_e)·φ_neb_k (so n_e cancels in Γ). This guarantees the two trusted
 * limits by construction:
 *   - fast rates (Γ,α n_e ≫ 1/Δt): 1/Δt negligible → Γ_k y_k = α_k n_e y_{k+1}
 *     ⇒ y_{k+1}/y_k = φ_neb/n_e  = exactly the steady nebular-Saha (inner shells).
 *   - slow rates (≪ 1/Δt): y ≈ y_init fully ionized  = the frozen-in fossil (outer).
 *   - comparable (transition zone): the physically-correct partial freeze-out.
 * No new free parameters. α_k reuses frozenin_alpha_rr (Milne, CMFGEN σ_bf). */
static double coupled_charge_density_tdep(AtomicData *atom, PlasmaState *plasma,
                                          int s, double T_e, double n_e,
                                          double t_exp, int n_shells,
                                          int write_pops,
                                          const double *gamma_jnu) {
    double T_rad = plasma->T_rad[s];
    double W     = plasma->W[s];
    double rho   = plasma->rho[s];
    if (T_rad <= 0.0 || n_e <= 0.0 || t_exp <= 0.0) return 0.0;
    double g_electron = pow(2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T_rad
                             / (H_PLANCK * H_PLANCK), 1.5);
    double beta_rad = 1.0 / (K_BOLTZMANN * T_rad);
    double beta_electron = 1.0 / (K_BOLTZMANN * T_e);
    double sqrt_te_tr = sqrt(T_e / T_rad);
    double inv_dt = 1.0 / t_exp;

    /* DIAGNOSTIC (LUMINA_COUPLED_JNU_ALPHA_TRAD=1): for pairs driven by the J_ν
     * photoionization rate, evaluate the recombination α at T_rad (the field's
     * color temperature) instead of T_e, so BOTH halves of detailed balance sit
     * at the same temperature. Confirms whether the J_ν over-ionization is purely
     * the T_e<T_field mismatch (over-ion should collapse) vs a deeper J_ν flaw.
     * Off by default. Cached once (benign OMP write race). */
    static int jnu_alpha_trad = -1;
    if (jnu_alpha_trad < 0) {
        const char *e = getenv("LUMINA_COUPLED_JNU_ALPHA_TRAD");
        jnu_alpha_trad = (e && atoi(e)) ? 1 : 0;
    }

    enum { MAXST = 64 };
    double dl[MAXST], dg[MAXST], du[MAXST], rhs[MAXST];   /* tridiagonal + RHS */
    double yv[MAXST], cp[MAXST];                          /* solution + Thomas scratch */

    double ne_sum = 0.0;
    for (int e = 0; e < atom->n_elements; e++) {
        int Z_elem = atom->element_Z[e];
        double mass_amu = atom->element_mass_amu[e];
        int ip_start = atom->elem_ion_offset[e];
        int ip_end   = atom->elem_ion_offset[e + 1];
        int n_pops   = ip_end - ip_start;
        double abund = atom->abundances[e * n_shells + s];
        double n_element = (abund * rho) / (mass_amu * AMU);
        if (n_element <= 0.0 || n_pops <= 0) continue;
        if (n_pops > MAXST) n_pops = MAXST;     /* defensive cap */
        if (n_pops == 1) { /* only one stage: charge fixed by its stage */
            ne_sum += n_element * (double)atom->ion_pop_stage[ip_start];
            if (write_pops)
                atom->ion_number_density[ip_start * n_shells + s] =
                    (n_element > 1e-300) ? n_element : 1e-300;
            continue;
        }

        /* per-pair (j,j+1) photoionization Γ_j and recombination α_j */
        for (int j = 0; j < n_pops; j++) { dl[j] = dg[j] = du[j] = rhs[j] = 0.0; }
        for (int j = 0; j < n_pops - 1; j++) {
            int ip_cur  = ip_start + j;
            int ip_next = ip_start + j + 1;
            int stage = atom->ion_pop_stage[ip_cur];
            double Z_cur  = atom->partition_functions[ip_cur  * n_shells + s];
            double Z_next = atom->partition_functions[ip_next * n_shells + s];
            if (Z_cur <= 0.0) Z_cur = 1.0;
            if (Z_next <= 0.0) Z_next = 1.0;
            int using_jnu = (gamma_jnu && gamma_jnu[ip_cur] >= 0.0);
            /* recomb T: T_e physically; T_rad only under the same-T diagnostic for
             * J_ν-driven pairs (matches α's color temp to the photoion field) */
            double T_rec = (jnu_alpha_trad && using_jnu) ? T_rad : T_e;
            double alpha_j = frozenin_alpha_rr(atom, ip_cur, ip_next, T_rec); /* k+1→k */
            if (!isfinite(alpha_j) || alpha_j < 0.0) alpha_j = 0.0;

            double Gamma_j;
            if (using_jnu) {
                /* PRIMARY: direct J_ν photoionization (no exp(−χ/kT_e) Saha lever) */
                Gamma_j = gamma_jnu[ip_cur];
            } else {
                /* fallback: detailed-balance Γ = α(T_e)·φ_neb (nebular Saha) */
                double chi_erg = find_ioniz_energy(atom, Z_elem, stage) * EV_TO_ERG;
                double prefactor = (Z_next / Z_cur) * 2.0 * g_electron;
                double phi_LTE_at_Trad = prefactor * exp(-chi_erg * beta_rad);
                double phi_LTE_at_Te   = prefactor * exp(-chi_erg * beta_electron);
                double zeta = interpolate_zeta(atom, Z_elem, stage, T_rad);
                double phi_neb = W * sqrt_te_tr *
                    (zeta * (T_e / T_rad) * phi_LTE_at_Te +
                     W * (1.0 - zeta) * phi_LTE_at_Trad);
                phi_neb = apply_ml_phi_neb_correction(phi_neb, Z_elem, stage, T_e, T_rad);
                phi_neb = apply_twocomp_lock(phi_neb, phi_LTE_at_Te, Z_elem, stage, W);
                if (!isfinite(phi_neb) || phi_neb < 0.0) phi_neb = 0.0;
                Gamma_j = alpha_j * phi_neb;               /* detailed balance: k→k+1 */
            }
            if (!isfinite(Gamma_j) || Gamma_j < 0.0) Gamma_j = 0.0;
            double rec_j = alpha_j * n_e;                  /* recomb rate k+1→k */

            /* assemble: row j (loss Γ_j up + gain rec into j from j+1),
             *           row j+1 (loss rec_j down + gain Γ_j into j+1 from j) */
            dg[j]     += Gamma_j;          /* ionization out of stage j */
            du[j]     += -rec_j;           /* recomb in from j+1 */
            dg[j + 1] += rec_j;            /* recomb out of stage j+1 */
            dl[j + 1] += -Gamma_j;         /* ionization in from j */
        }
        /* backward-Euler diagonal + fully-ionized initial condition (top stage) */
        for (int j = 0; j < n_pops; j++) dg[j] += inv_dt;
        rhs[n_pops - 1] = inv_dt;          /* y_init = top stage = 1 */

        /* Thomas solve of the tridiagonal (dl=sub, dg=diag, du=super) */
        double bet = dg[0];
        if (!(fabs(bet) > 0.0)) continue;
        yv[0] = rhs[0] / bet;
        for (int j = 1; j < n_pops; j++) {
            cp[j] = du[j - 1] / bet;
            bet = dg[j] - dl[j] * cp[j];
            if (!(fabs(bet) > 1e-300)) { bet = (bet >= 0 ? 1e-300 : -1e-300); }
            yv[j] = (rhs[j] - dl[j] * yv[j - 1]) / bet;
        }
        for (int j = n_pops - 2; j >= 0; j--) yv[j] -= cp[j + 1] * yv[j + 1];

        /* The backward-Euler tridiagonal conserves Sum y_j = 1 analytically (the
         * rate matrix is column-conservative and the only source is y_init=1 on
         * the top stage). Clamping non-finite/negative fractions to 0 breaks that
         * sum; renormalize to restore exact stage conservation so the written ion
         * partition stays well-conditioned (no spurious all-zero element). */
        double ysum = 0.0;
        for (int j = 0; j < n_pops; j++) {
            if (!isfinite(yv[j]) || yv[j] < 0.0) yv[j] = 0.0;
            ysum += yv[j];
        }
        if (ysum > 0.0) {
            double inv = 1.0 / ysum;
            for (int j = 0; j < n_pops; j++) yv[j] *= inv;
        } else {
            /* degenerate solve: fall back to fully ionized (the IC) */
            yv[n_pops - 1] = 1.0;
        }

        double charge = 0.0;
        for (int j = 0; j < n_pops; j++) {
            double yj = yv[j];
            charge += (double)atom->ion_pop_stage[ip_start + j] * yj;
            if (write_pops) {
                double n_ion = n_element * yj;
                if (n_ion < 1e-300) n_ion = 1e-300;
                atom->ion_number_density[(ip_start + j) * n_shells + s] = n_ion;
            }
        }
        ne_sum += n_element * charge;
    }
    return ne_sum;
}

/* A2 separation test (read-only diagnostic, write_pops never set):
 * solve the charge fixed point n_e = charge(T_e_fix, n_e) at a FROZEN T_e for a
 * chosen closure (tdep = full-ion-IC backward-Euler eq.26; steady = no-IC nebular
 * Saha). Damped fixed-point. Lets us separate (a) IC bias from (b) phi_neb form
 * and from T_e drift by comparing the self-consistent n_e at the CMFGEN-like T_e
 * vs the converged T_e for both closures. No physics path is modified. */
static double solve_ne_fixed_te(AtomicData *atom, PlasmaState *plasma, int s,
                                double T_e_fix, double t_exp, int n_shells,
                                int tdep, const double *gamma_jnu) {
    double ne = plasma->n_electron[s];
    if (!(ne > 0.0)) ne = 1.0e8;
    for (int it = 0; it < 120; it++) {
        double c = tdep
            ? coupled_charge_density_tdep(atom, plasma, s, T_e_fix, ne, t_exp, n_shells, 0, gamma_jnu)
            : coupled_charge_density(atom, plasma, s, T_e_fix, ne, n_shells);
        if (!(c > 0.0)) break;
        double ne_new = 0.5 * (ne + c);       /* damped to avoid n_e oscillation */
        double rel = fabs(ne_new - ne) / (ne + 1e-300);
        ne = ne_new;
        if (rel < 1e-6) break;
    }
    return ne;
}

void coupled_newton_solve_all(PlasmaState *plasma, GammaDeposition *gamma_dep,
                              NLTEConfig *nlte, AtomicData *atom,
                              OpacityState *opacity, Geometry *geo,
                              double time_explosion, int n_shells) {
    if (nlte == NULL || nlte->nlte_level_populations == NULL ||
        nlte->J_nu == NULL) return;   /* no lagged radiation field yet */
    build_radeq_line_table(nlte, atom, opacity);

    const double GFF = 1.2;
    const double FF_COEF = 1.426e-27;
    double Gamma_ad = 2.0 / time_explosion;
    int use_cmfgen = atom->cmfgen_loaded &&
                     atom->cmfgen_n_freq_bins == nlte->n_freq_bins;

    int cool_nonneg = 1;
    { const char *cn = getenv("LUMINA_RADEQ_COOL_NONNEG"); if (cn) cool_nonneg = atoi(cn); }
    int cool_nlte_only = 1;
    { const char *co = getenv("LUMINA_RADEQ_COOL_NLTE_ONLY"); if (co) cool_nlte_only = atoi(co); }
    int cool_escape = 1;
    { const char *ce = getenv("LUMINA_RADEQ_COOL_ESCAPE"); if (ce) cool_escape = atoi(ce); }
    /* A3 (A): ETLA T_e-responsive bound-bound cooling (n_up recomputed in local SE
     * at the trial T_e inside the Newton). Default OFF → byte-identical. When ON it
     * overrides the escape branch (cooling is built from the responsive SE form). */
    int line_respond = 0;
    { const char *lr = getenv("LUMINA_RADEQ_LINE_RESPOND"); if (lr) line_respond = atoi(lr); }
    int use_escape = cool_escape && !line_respond;
    double hybrid_taurec_thr = 1.0;
    { const char *ht = getenv("LUMINA_RADEQ_HYBRID_TAUREC"); if (ht) hybrid_taurec_thr = atof(ht); }
    /* A2 increment-2: time-dependent (backward-Euler eq.26) charge residual.
     * Default ON; LUMINA_COUPLED_TDEP=0 reverts to increment-1 steady-state Saha. */
    int coupled_tdep = 1;
    { const char *td = getenv("LUMINA_COUPLED_TDEP"); if (td) coupled_tdep = atoi(td); }

    int nfb = nlte->n_freq_bins;
    double *nu_mid  = (double *)malloc((size_t)nfb * sizeof(double));
    for (int bb = 0; bb < nfb; bb++) {
        double lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
        nu_mid[bb] = exp(lo + 0.5 * nlte->d_log_nu);
    }
    size_t nl_alloc = (size_t)(radeq_n_lines > 0 ? radeq_n_lines : 1);
    /* mark shells the Newton owns so the trailing reconcile skips them */
    unsigned char *newton_owned = (unsigned char *)calloc((size_t)n_shells, 1);

    /* A2 fix (B): the shell loop is local/independent (depth coupling deferred
     * to A4) so it parallelizes bit-identically. Per-shell scratch (emit_nu,
     * ca/cb/cbet) is privatized per thread; nu_mid is read-only/shared.
     * LUMINA_COUPLED_NEWTON_OMP=0 forces serial (byte-identical A/B). */
    int newton_omp = 1;
    { const char *no = getenv("LUMINA_COUPLED_NEWTON_OMP"); if (no) newton_omp = atoi(no); }
    /* diagnostic: dump the per-iter Newton trace for ONE shell (id), default off */
    int trace_sh = -1;
    { const char *tr = getenv("LUMINA_COUPLED_NEWTON_TRACE"); if (tr) trace_sh = atoi(tr); }
    /* A2 separation test: if set to a frozen T_e (K), dump the self-consistent n_e
     * at that T_e vs the converged T_e, for BOTH the tdep(full-ion IC) and
     * steady(no-IC) closures, per Newton-owned shell. Default off (<=0). */
    double septest_Tfix = -1.0;
    { const char *st = getenv("LUMINA_COUPLED_SEPTEST"); if (st) septest_Tfix = atof(st); }
    /* A2 PRIMARY: direct J_ν photoionization closure for the charge residual,
     * replacing Γ=α·φ_neb (kills the φ_neb form bias + the exp(−χ/kT_e) T_e
     * runaway). Default off → byte-identical φ_neb path. Needs cmfgen σ_bf. */
    int use_jnu = 0;
    { const char *jp = getenv("LUMINA_COUPLED_JNU_PHOTOION"); if (jp) use_jnu = atoi(jp); }
    int n_ip = atom->elem_ion_offset[atom->n_elements];

    /* A3 incr-1: diagonal-Λ* radiation response (J follows W·B(T_e) inside the
     * Newton). Default OFF → byte-identical frozen-J path. Needs Geometry for the
     * local thermalization length L=v_outer·t_exp; disabled if absent. */
    int use_lstar = 0;
    { const char *ls = getenv("LUMINA_COUPLED_LAMBDA_STAR"); if (ls) use_lstar = atoi(ls); }
    double wbfloor_T = -1.0;
    { const char *wf = getenv("LUMINA_COUPLED_JNU_WBFLOOR"); if (wf) wbfloor_T = atof(wf); }
    /* line-eps split active: g_lre_chi_line holds the THERMAL line channel,
     * which then belongs in the local-Planck response fraction eps_b. */
    int line_eps_on = 0;
    { const char *le = getenv("LUMINA_CMFGEN_LINE_EPS");
      if (le && atof(le) > 0.0) line_eps_on = 1;
      const char *lp = getenv("LUMINA_CMFGEN_LINE_EPS_PHYS");
      if (lp && atoi(lp)) line_eps_on = 1; }
    double lstar_tauscale = 1.0;
    { const char *lt = getenv("LUMINA_COUPLED_LAMBDA_TAUSCALE"); if (lt) lstar_tauscale = atof(lt); }
    if (use_lstar && (geo == NULL || geo->v_outer == NULL)) use_lstar = 0;
    /* B3-1: route the diagonal-Λ* response INTO the J_ν photoionization integral
     * (not just RADEQ bf-heating). Requires both J_ν photoion and Λ* active. */
    int use_jnu_lstar = 0;
    { const char *jl = getenv("LUMINA_COUPLED_JNU_LSTAR"); if (jl) use_jnu_lstar = atoi(jl); }
    use_jnu_lstar = use_jnu_lstar && use_jnu && use_lstar;

    /* Option-2 integral radiative equilibrium (LUMINA_RADEQ_LINE_RE=1): add the
     * T_e-responsive radiative line term 4π∫χ_line(J−S_l)dν to the energy
     * residual and DROP the collisional/escape bound-bound cooling. Needs the
     * CMFGEN line opacity/source registered via radeq_set_line_re_source and a
     * matching shell count; else stays off (byte-identical collisional path). */
    int line_re = 0;
    { const char *lr = getenv("LUMINA_RADEQ_LINE_RE"); if (lr) line_re = atoi(lr); }
    line_re = line_re && g_lre_chi_line && g_lre_nshells == n_shells;
    if (line_re)
        printf("  [COUPLED-NEWTON] Option-2 line-RE term ON "
               "(drop collisional bb; %d bins)\n", g_lre_nbins);

    long n_solved = 0;
    /* Fallback-visibility counters (no physics change): a shell that exits the
     * Newton without satisfying the descent test (conv==0) committed a STALE
     * iterate, and a shell whose T_e lands on the 1000K hard floor is
     * under-determined. Both were silently committed and marked owned; surface
     * them so the Option-2 A/B is not contaminated by hidden not-converged
     * commits. */
    long n_stall = 0;     /* Newton exited without descent-test convergence */
    long n_floor_cn = 0;  /* T_e pinned to the 1000K hard floor */
    /* (0) coupled-Newton profiler (LUMINA_CN_PROF=1): wall-time of the parallel
     * region + per-thread shell distribution + heaviest single shell, to split
     * "tail load-imbalance" from "per-shell cost". Off by default, zero overhead. */
    int cn_prof = 0;
    { const char *p = getenv("LUMINA_CN_PROF"); if (p) cn_prof = atoi(p); }
#ifdef _OPENMP
    int cn_nthr = newton_omp ? omp_get_max_threads() : 1;
#else
    int cn_nthr = 1;
#endif
    double *cn_tsum = NULL, *cn_tmax = NULL; long *cn_scnt = NULL; int *cn_smax = NULL;
    double cn_wt0 = 0.0;
    if (cn_prof) {
        cn_tsum = (double *)calloc(cn_nthr, sizeof(double));
        cn_tmax = (double *)calloc(cn_nthr, sizeof(double));
        cn_scnt = (long *)calloc(cn_nthr, sizeof(long));
        cn_smax = (int *)calloc(cn_nthr, sizeof(int));
#ifdef _OPENMP
        cn_wt0 = omp_get_wtime();
#endif
    }
#ifdef _OPENMP
    #pragma omp parallel if(newton_omp)
#endif
    {
    double *emit_nu = (double *)malloc((size_t)nfb * sizeof(double));
    double *ca   = (double *)malloc(nl_alloc * sizeof(double));
    double *cb   = (double *)malloc(nl_alloc * sizeof(double));
    double *cbet = (double *)malloc(nl_alloc * sizeof(double));
    /* A3 (A) ETLA scratch: A=coeff/g_lo, B=coeff/g_up, beta, dE, nlo (lagged lower),
     * and the lagged Sobolev-escape radiative rates Rlu/Rul. NULL when OFF. */
    double *et_A   = line_respond ? (double *)malloc(nl_alloc * sizeof(double)) : NULL;
    double *et_B   = line_respond ? (double *)malloc(nl_alloc * sizeof(double)) : NULL;
    double *et_bet = line_respond ? (double *)malloc(nl_alloc * sizeof(double)) : NULL;
    double *et_dE  = line_respond ? (double *)malloc(nl_alloc * sizeof(double)) : NULL;
    double *et_nlo = line_respond ? (double *)malloc(nl_alloc * sizeof(double)) : NULL;
    double *et_nup = line_respond ? (double *)malloc(nl_alloc * sizeof(double)) : NULL;
    double *et_Rlu = line_respond ? (double *)malloc(nl_alloc * sizeof(double)) : NULL;
    double *et_Rul = line_respond ? (double *)malloc(nl_alloc * sizeof(double)) : NULL;
    double *gamma_jnu = (use_jnu && n_ip > 0)
        ? (double *)malloc((size_t)n_ip * sizeof(double)) : NULL;
    /* A3 incr-1 per-bin scratch: gbin=photo-weight, chi_loc=bf opacity,
     * lstar=1−e^{−τ_bf}, blag=B_ν at the lagged seed T_e. NULL when OFF. */
    double *gbin    = use_lstar ? (double *)malloc((size_t)nfb * sizeof(double)) : NULL;
    double *chi_loc = use_lstar ? (double *)malloc((size_t)nfb * sizeof(double)) : NULL;
    double *lstar   = use_lstar ? (double *)malloc((size_t)nfb * sizeof(double)) : NULL;
    double *blag    = use_lstar ? (double *)malloc((size_t)nfb * sizeof(double)) : NULL;
#ifdef _OPENMP
    #pragma omp for schedule(dynamic, 1) reduction(+:n_solved,n_stall,n_floor_cn)
#endif
    for (int s = 0; s < n_shells; s++) {
        double cn_ts0 = 0.0;
#ifdef _OPENMP
        if (cn_prof) cn_ts0 = omp_get_wtime();
#endif
        double T_rad = plasma->T_rad[s];
        double W     = plasma->W[s];
        if (T_rad <= 0.0 || plasma->n_electron[s] <= 0.0) continue;

        /* Shell partition vs the frozen-in cascade:
         *  - increment-2 (tdep): the backward-Euler residual reproduces the frozen
         *    limit itself, so the Newton can own everything frozen-in did NOT
         *    freeze (incl. the transition zone). Skip ONLY frozen-in-owned shells.
         *  - increment-1 (steady): keep the old τ_rec/t_exp gate (inner only),
         *    leaving the transition zone to steady-state Saha (A/B baseline). */
        if (coupled_tdep) {
            if (frozenin_is_frozen && s < frozenin_is_frozen_n &&
                frozenin_is_frozen[s]) continue;
        } else {
            double alpha_rec = 2.6e-13 * pow(T_rad / 1.0e4, -0.8);
            double n_e0 = plasma->n_electron[s];
            double tau_rec = (alpha_rec > 0.0) ? 1.0 / (n_e0 * alpha_rec) : 0.0;
            if (tau_rec >= hybrid_taurec_thr * time_explosion) continue;
        }

        /* ---- assemble the per-shell, n_e/T_e-independent radiative-equilibrium
         *      tables from the lagged NLTE pops + J_ν (same as the bisection path) ---- */
        double H_photo = 0.0;
        double beta_rad_h = 1.0 / (K_BOLTZMANN * T_rad);
        for (int bb = 0; bb < nfb; bb++) emit_nu[bb] = 0.0;
        if (gbin) for (int bb = 0; bb < nfb; bb++) { gbin[bb] = 0.0; chi_loc[bb] = 0.0; }
        for (int i = 0; i < nlte->n_nlte_ions; i++) {
            int Z = nlte->nlte_Z[i];
            int ion_stage = nlte->nlte_ion[i];
            double chi_erg = find_ioniz_energy(atom, Z, ion_stage) * EV_TO_ERG;
            if (chi_erg <= 0.0) continue;
            double sigma0 = get_bf_sigma0(Z, ion_stage);
            if (sigma0 <= 0.0) { int Zeff = Z - ion_stage; if (Zeff < 1) Zeff = 1;
                sigma0 = 7.91e-18 / ((double)Zeff * (double)Zeff); }
            int ls = nlte->nlte_ion_level_offset[i];
            int le = nlte->nlte_ion_level_offset[i + 1];
            for (int l = ls; l < le; l++) {
                int g = nlte->nlte_to_global_level[l];
                double E_lev = atom->level_energy_eV[g] * EV_TO_ERG;
                double nu_th = (chi_erg - E_lev) / H_PLANCK;
                if (nu_th <= 0.0) continue;
                double n_lev = nlte->nlte_level_populations[(size_t)l * n_shells + s];
                if (n_lev <= 0.0) continue;
                int has = use_cmfgen && atom->cmfgen_has_sigma[g];
                const double *srow = has ?
                    &atom->cmfgen_sigma_bf[(size_t)g * atom->cmfgen_n_freq_bins] : NULL;
                double integ = 0.0;
                for (int bb = 0; bb < nfb; bb++) {
                    double nu_bin = nu_mid[bb];
                    if (nu_bin < nu_th) continue;
                    double lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
                    double dnu = exp(lo + nlte->d_log_nu) - exp(lo);
                    double sig = srow ? srow[bb] : sigma0 * pow(nu_th / nu_bin, 3.0);
                    if (sig <= 0.0) continue;
                    double f_above = 1.0 - nu_th / nu_bin;
                    double geom = 4.0 * M_PI_VAL * sig * f_above * dnu;
                    double bnu_pref = 2.0 * H_PLANCK * nu_bin * nu_bin * nu_bin /
                                      (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT);
                    double J = nlte->J_nu[(size_t)s * nlte->n_freq_bins + bb];
                    integ += geom * J;
                    (void)beta_rad_h;
                    emit_nu[bb] += n_lev * geom * bnu_pref;
                    if (gbin) { gbin[bb] += n_lev * geom; chi_loc[bb] += n_lev * sig; }
                }
                H_photo += n_lev * integ;
            }
        }
        double H_gamma = (gamma_dep && gamma_dep->heating_rate &&
                          gamma_dep->heating_rate[s] > 0.0) ?
                          gamma_dep->heating_rate[s] : 0.0;
        if (H_photo <= 0.0 && H_gamma <= 0.0) continue;  /* no anchor → leave fallback */

        double u_rad = 4.0 * W * SIGMA_SB * T_rad * T_rad * T_rad * T_rad /
                       C_SPEED_OF_LIGHT;
        double Gamma_C = 8.0 * SIGMA_THOMSON * u_rad /
                         (3.0 * M_ELECTRON * C_SPEED_OF_LIGHT);
        double compton0 = 1.5 * K_BOLTZMANN * Gamma_C;   /* ×n_e ×(T_rad−T_e) */
        double ff0 = FF_COEF * GFF;                      /* ×n_e^2 ×sqrt(T_e) */

        double beta_rad = 1.0 / (K_BOLTZMANN * T_rad);
        double C_bb_esc = 0.0;
        long n_active = 0;
        /* Option-2: skip the collisional bound-bound assembly entirely — the
         * radiative line term (RADEQ_LINE_RE_TERM) is the sole bb owner, so
         * C_bb_esc and n_active stay 0 and radeq_net's collisional cooling
         * vanishes. */
        for (long k = 0; !line_re && k < radeq_n_lines; k++) {
            const RadEqLine *rl = &radeq_lines[k];
            if (cool_nlte_only && (rl->nlte_lo < 0 || rl->nlte_up < 0)) continue;
            double n_ion = atom->ion_number_density[(size_t)rl->ip * n_shells + s];
            double U     = atom->partition_functions[(size_t)rl->ip * n_shells + s];
            if (U <= 0.0) U = 1.0;
            double nlo_k, nup_k;
            if (rl->nlte_lo >= 0) nlo_k = nlte->nlte_level_populations[(size_t)rl->nlte_lo * n_shells + s];
            else { double bz = atom->level_energy_eV[rl->lo_g] * EV_TO_ERG * beta_rad;
                   double wt = atom->level_metastable[rl->lo_g] ? 1.0 : W;
                   nlo_k = (bz < 500.0) ? n_ion * wt * rl->g_lo * exp(-bz) / U : 0.0; }
            if (rl->nlte_up >= 0) nup_k = nlte->nlte_level_populations[(size_t)rl->nlte_up * n_shells + s];
            else { double bz = atom->level_energy_eV[rl->up_g] * EV_TO_ERG * beta_rad;
                   double wt = atom->level_metastable[rl->up_g] ? 1.0 : W;
                   nup_k = (bz < 500.0) ? n_ion * wt * rl->g_up * exp(-bz) / U : 0.0; }
            if (nlo_k <= 0.0 && nup_k <= 0.0) continue;
            if (use_escape) {
                if (nup_k > 0.0 && rl->A_ul > 0.0 && opacity->tau_sobolev) {
                    double tau = opacity->tau_sobolev[(size_t)rl->line * n_shells + s];
                    C_bb_esc += radeq_beta_esc(tau) * rl->A_ul * nup_k * rl->dE;
                }
                continue;
            }
            ca[n_active]   = rl->dE * rl->coeff * nlo_k / rl->g_lo;
            cb[n_active]   = rl->dE * rl->coeff * nup_k / rl->g_up;
            cbet[n_active] = rl->beta;
            if (line_respond) {
                /* lagged Sobolev-escape radiative rates for the ETLA SE n_up */
                double nu_l = rl->dE / H_PLANCK;
                double beta_esc = 1.0;
                if (opacity->tau_sobolev)
                    beta_esc = radeq_beta_esc(opacity->tau_sobolev[(size_t)rl->line * n_shells + s]);
                double Jbar = nlte_get_J_at_nu(nlte, s, nu_l);
                double B_ul = (nu_l > 0.0) ? rl->A_ul * C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT /
                              (2.0 * H_PLANCK * nu_l * nu_l * nu_l) : 0.0;
                double B_lu = (rl->g_lo > 0) ? B_ul * (double)rl->g_up / (double)rl->g_lo : 0.0;
                et_A[n_active]   = rl->coeff / rl->g_lo;
                et_B[n_active]   = rl->coeff / rl->g_up;
                et_bet[n_active] = rl->beta;
                et_dE[n_active]  = rl->dE;
                et_nlo[n_active] = nlo_k;
                et_nup[n_active] = nup_k;
                et_Rlu[n_active] = B_lu * Jbar * beta_esc;
                et_Rul[n_active] = (rl->A_ul + B_ul * Jbar) * beta_esc;
            }
            n_active++;
        }

        /* ---- 2×2 Newton on x=(T_e, n_e) ---- */
        double T_e = plasma->T_e[s];
        double n_e = plasma->n_electron[s];
        /* Te_lag = the assemble-time T_e the CMFGEN line opacity/source were
         * built at: the registration SNAPSHOT when present. plasma->T_e[s] at
         * Newton entry is WRONG here — the pre-Newton bisection rewrote it
         * after cmfgen_assemble (the Te_lag-capture defect). */
        double Te_lag = (g_lre_te_lag && s < g_lre_te_lag_n &&
                         g_lre_te_lag[s] > 100.0) ? g_lre_te_lag[s]
                      : (plasma->T_e[s] > 100.0) ? plasma->T_e[s]
                        : plasma->T_e_T_rad_ratio * T_rad;
        if (T_e <= 100.0) T_e = plasma->T_e_T_rad_ratio * T_rad;

        /* A3 incr-1: freeze the local thermalization fraction Λ*_bb and the lagged
         * Planck B_ν(T_e^lag) once per shell. τ_bf = χ_bf·L with L=v_outer·t_exp
         * the homologous depth scale (knob LUMINA_COUPLED_LAMBDA_TAUSCALE). Λ*→1
         * where the gas is thick to its own bf continuum (inner), →0 thin (outer),
         * so J tracks W·B(T_e) only where it should. B_lag is the response anchor. */
        if (gbin) {
            /* Phase-1 faithful response: use the formal solve's diagonal Λ*_b
             * (registered via radeq_set_line_re_source) weighted by the thermal
             * absorption fraction ε_b=χ_abs/χ_tot, so the stored lstar[bb] is the
             * true ∂J_b/∂S_b·∂S_b/∂B = Λ*_b·ε_b. Falls back to the bf τ-proxy only
             * when no CMFGEN Λ* is registered (non-pure-CMFGEN callers). */
            int have_real = (g_lre_lambda_star && g_lre_nshells == n_shells
                             && g_lre_nbins == nfb);
            double Lscale = lstar_tauscale * geo->v_outer[s] * time_explosion;
            for (int bb = 0; bb < nfb; bb++) {
                double Ld, eps;
                if (have_real) {
                    size_t idx = (size_t)s * nfb + bb;
                    double ct = g_lre_chi_tot[idx];
                    Ld  = g_lre_lambda_star[idx];
                    eps = (ct > 0.0)
                        ? (g_lre_chi_abs[idx] +
                           (line_eps_on ? g_lre_chi_line[idx] : 0.0)) / ct
                        : 0.0;
                    lstar[bb] = Ld * eps;
                    /* Faithful ALI: anchor the response at the FROZEN per-bin J*_b
                     * (the SAME binned mean intensity nlte->J_nu that built H_photo),
                     * NOT B_nu(Te_lag). With blag=J*, the bf heating-cooling pair plus
                     * H_resp collapse to Sum gbin*(1-Lambda*eps)*(J*-B(T_e)): the
                     * trapped fraction Lambda*eps lets J follow B(T_e) while the
                     * streaming fraction (1-Lambda*eps) keeps the (J*-B(T_e)) restoring
                     * slope, so the thick limit drives T_e -> color temp of J* (= T_rad)
                     * with dr/dT_e<0. Anchoring at B(Te_lag) made the pair cancel to 0
                     * identically (response inert, T_e pinned at the seed). */
                    blag[bb] = nlte->J_nu[idx];
                } else {
                    double tau = chi_loc[bb] * Lscale;
                    Ld  = (tau > 1e-8) ? -expm1(-tau) : tau;
                    eps = 1.0;          /* proxy already bf-only */
                    lstar[bb] = Ld * eps;
                    blag[bb]  = planck_bnu(Te_lag, nu_mid[bb]);
                }
            }
            static int lstar_diag_once = 0;
            if (s == 0 && !lstar_diag_once) {
                lstar_diag_once = 1;
                double lm = 0.0, dj = 0.0; int nb = 0;
                for (int bb = 0; bb < nfb; bb++)
                    if (gbin[bb] > 0.0) {
                        lm += lstar[bb]; nb++;
                        double jb = nlte->J_nu[(size_t)s * nfb + bb];
                        if (jb > 0.0) dj += fabs(blag[bb] - jb) / jb;
                    }
#ifdef _OPENMP
                #pragma omp critical
#endif
                fprintf(stderr, "[LSTAR-DIAG] s=0 have_real=%d nbins=%d "
                        "mean(Lambda*eps)=%.4f mean|blag-J*|/J*=%.3e "
                        "g_lre(ptr=%p ns=%d/%d nb=%d/%d)\n",
                        have_real, nb, nb ? lm / nb : 0.0, nb ? dj / nb : -1.0,
                        (void *)g_lre_lambda_star, g_lre_nshells, n_shells,
                        g_lre_nbins, nfb);
            }
            if (trace_sh == s) {
                double lm = 0.0; int nb = 0;
                for (int bb = 0; bb < nfb; bb++)
                    if (gbin[bb] > 0.0) { lm += lstar[bb]; nb++; }
#ifdef _OPENMP
                #pragma omp critical
#endif
                printf("    [LSTAR s=%d] mean Lambda*=%.3f over %d photo-bins "
                       "L=%.3e cm T_lag=%.0f W=%.3f\n",
                       s, nb ? lm / nb : 0.0, nb, Lscale, T_e, W);
            }
        }

        /* PRIMARY: precompute per-ion J_ν photoionization rate ONCE at the LAGGED
         * T_e (frozen across this shell's Newton solve). Freezing removes the weak
         * excitation-Boltzmann T_e dependence from the residual, leaving only the
         * recombination α(T_e); J_ν carries no exp(−χ/kT_e) lever, so the runaway
         * cannot re-form. Falls back to φ_neb (gamma_jnu[ip]=−1) for ions w/o σ_bf. */
        if (gamma_jnu) {
            double T_lag = (T_e > 100.0) ? T_e : plasma->T_e_T_rad_ratio * T_rad;
            /* B3-1: blend J_ν toward W·B(T_e^lag) by Λ* inside the photoion
             * integral. lstar/blag are this shell's diagonal-Λ* and B_ν(T_e^lag)
             * (computed just above when gbin active). NULL → bare lagged J. */
            const double *jbl = use_jnu_lstar ? lstar : NULL;
            const double *jbb = use_jnu_lstar ? blag  : NULL;
            for (int ip = 0; ip < n_ip; ip++)
                gamma_jnu[ip] = coupled_photoion_rate_jnu(atom, nlte, ip, s,
                                                          T_lag, n_shells,
                                                          jbl, jbb, W, wbfloor_T);
        }
        double T_lo = 0.03 * T_rad, T_hi = 3.0 * T_rad;
        int conv = 0;
        /* A3 (A): replace radeq_net's internal lagged line cooling with the
         * T_e-responsive ETLA form. radeq_net still adds radeq_line_cool(ca,cb,...);
         * adding (old − etla) cancels it and substitutes the SE-responsive cooling. */
#define RADEQ_LINE_DELTA(TT,NN) (line_respond ? \
        radeq_line_cool((TT),(NN),ca,cb,cbet,n_active,cool_nonneg) - \
        radeq_line_cool_etla((TT),(NN),et_A,et_B,et_bet,et_dE,et_nlo,et_nup,et_Rlu,et_Rul,n_active,(line_respond==2)) : 0.0)
        /* Option-2 radiative line term (heating side), T_e-responsive at the
         * trial T_e with the lagged source frozen at Te_lag. 0 when line_re off. */
#define RADEQ_LINE_RE_TERM(TT) (line_re ? radeq_line_re((TT), Te_lag, s) : 0.0)
        /* DIAGNOSTIC (no logic change): residual-at-truth probe. Scans r1(T_e)
         * over a grid with the bound-bound ESCAPE coolant ON (C_bb_esc) vs OFF
         * (collisional-net only) to expose whether the RE equation even HAS a
         * root. LUMINA_CN_RTRUTH=1 + trace shell. Reference T_e (CMFGEN ~4434 at
         * photosphere) probed alongside fractions of T_rad. */
        if (getenv("LUMINA_CN_RTRUTH") &&
            (s == 0 || s == n_shells/4 || s == n_shells/2 ||
             s == (3*n_shells)/4 || s == n_shells-1)) {
            double frac[] = {0.3,0.5,0.7,0.9,1.0,1.1,1.3,1.6,2.0};
            int NF = 9;
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            {
                printf("    [CN-RTRUTH s=%d] H_photo=%.3e H_gamma=%.3e C_esc=%.3e n_e=%.3e T_rad=%.1f n_act=%ld\n",
                       s, H_photo, H_gamma, C_bb_esc, n_e, T_rad, n_active);
                double prev_on = 0, prev_off = 0; int roots_on = 0, roots_off = 0;
                for (int pi = 0; pi < NF; pi++) {
                    double TT = frac[pi] * T_rad;
                    double C_ff  = ff0 * n_e * n_e * sqrt(TT);
                    double C_ad  = 1.5 * n_e * K_BOLTZMANN * TT * Gamma_ad;
                    double C_rec = radeq_recomb_cool(TT, emit_nu, nu_mid, nfb);
                    double C_col = radeq_line_cool(TT, n_e, ca, cb, cbet, n_active, cool_nonneg);
                    double Hresp = radeq_Hresp(TT, gbin, lstar, blag, 1.0, nu_mid, nfb);
                    double dlt   = RADEQ_LINE_DELTA(TT, n_e);
                    double lre   = RADEQ_LINE_RE_TERM(TT);
                    double r_on = radeq_net(TT, T_rad, n_e, H_photo, H_gamma,
                                            compton0*n_e, ff0*n_e*n_e, Gamma_ad,
                                            ca, cb, cbet, n_active, cool_nonneg, C_bb_esc,
                                            emit_nu, nu_mid, nfb) + Hresp + dlt + lre;
                    double r_off = radeq_net(TT, T_rad, n_e, H_photo, H_gamma,
                                             compton0*n_e, ff0*n_e*n_e, Gamma_ad,
                                             ca, cb, cbet, n_active, cool_nonneg, 0.0,
                                             emit_nu, nu_mid, nfb) + Hresp + dlt + lre;
                    if (pi > 0) { if (prev_on*r_on < 0) roots_on++; if (prev_off*r_off < 0) roots_off++; }
                    prev_on = r_on; prev_off = r_off;
                    printf("    [CN-RTRUTH s=%d] T_e=%8.1f C_ff=%.2e C_ad=%.2e C_rec=%.2e C_col=%.2e | r1ON=%+.3e r1OFF=%+.3e\n",
                           s, TT, C_ff, C_ad, C_rec, C_col, r_on, r_off);
                }
                printf("    [CN-RTRUTH s=%d] roots_on=%d roots_off=%d (sign-changes over T_e grid)\n",
                       s, roots_on, roots_off);
            }
        }
        for (int it = 0; it < 60; it++) {
            double r1 = radeq_net(T_e, T_rad, n_e, H_photo, H_gamma,
                                  compton0 * n_e, ff0 * n_e * n_e, Gamma_ad,
                                  ca, cb, cbet, n_active, cool_nonneg, C_bb_esc,
                                  emit_nu, nu_mid, nfb)
                       + radeq_Hresp(T_e, gbin, lstar, blag, 1.0, nu_mid, nfb)
                       + RADEQ_LINE_DELTA(T_e, n_e)
                       + RADEQ_LINE_RE_TERM(T_e);
            double r2 = n_e - (coupled_tdep
                ? coupled_charge_density_tdep(atom, plasma, s, T_e, n_e, time_explosion, n_shells, 0, gamma_jnu)
                : coupled_charge_density(atom, plasma, s, T_e, n_e, n_shells));

            /* (2) solve in (T_e, x=ln n_e): the n_e column is differentiated and
             * stepped in log space, so n_e=exp(x) is strictly positive by
             * construction (the line-search positivity guard can no longer fail
             * for the descent direction). dx is a relative perturbation; the log
             * column rescales the Jacobian by n_e, improving conditioning of the
             * near-singular case that previously produced the runaway step. */
            double dT = 1e-4 * T_e, dx = 1e-4;
            double ne_x = n_e * exp(dx);
            double r1_T = radeq_net(T_e + dT, T_rad, n_e, H_photo, H_gamma,
                                    compton0 * n_e, ff0 * n_e * n_e, Gamma_ad,
                                    ca, cb, cbet, n_active, cool_nonneg, C_bb_esc,
                                    emit_nu, nu_mid, nfb)
                         + radeq_Hresp(T_e + dT, gbin, lstar, blag, 1.0, nu_mid, nfb)
                         + RADEQ_LINE_DELTA(T_e + dT, n_e)
                         + RADEQ_LINE_RE_TERM(T_e + dT);
            double r1_n = radeq_net(T_e, T_rad, ne_x,
                                    H_photo, H_gamma, compton0 * ne_x,
                                    ff0 * ne_x * ne_x, Gamma_ad,
                                    ca, cb, cbet, n_active, cool_nonneg, C_bb_esc,
                                    emit_nu, nu_mid, nfb)
                         + radeq_Hresp(T_e, gbin, lstar, blag, 1.0, nu_mid, nfb)
                         + RADEQ_LINE_DELTA(T_e, ne_x)
                         + RADEQ_LINE_RE_TERM(T_e);
            double r2_T = n_e - (coupled_tdep
                ? coupled_charge_density_tdep(atom, plasma, s, T_e + dT, n_e, time_explosion, n_shells, 0, gamma_jnu)
                : coupled_charge_density(atom, plasma, s, T_e + dT, n_e, n_shells));
            double r2_n = ne_x - (coupled_tdep
                ? coupled_charge_density_tdep(atom, plasma, s, T_e, ne_x, time_explosion, n_shells, 0, gamma_jnu)
                : coupled_charge_density(atom, plasma, s, T_e, ne_x, n_shells));

            double J11 = (r1_T - r1) / dT, J12 = (r1_n - r1) / dx;
            double J21 = (r2_T - r2) / dT, J22 = (r2_n - r2) / dx;
            double det = J11 * J22 - J12 * J21;
            if (!isfinite(det) || fabs(det) < 1e-300) break;
            double dTe = -( J22 * r1 - J12 * r2) / det;
            double dlx = -(-J21 * r1 + J11 * r2) / det;  /* Newton step in ln(n_e) */

            /* damped line search: accept ONLY a valid (n_e>0, finite) AND
             * improving trial. If no descent point is found in 25 halvings,
             * take NO step (keep previous T_e,n_e) and stall — never commit a
             * rejected nonphysical trial (the negative-n_e bug). */
            double rn0 = fabs(r1) / (fabs(H_photo) + 1e-300) + fabs(r2) / (n_e + 1e-300);
            double lam = 1.0;
            double Tn = T_e, nn = n_e;
            int accepted = 0;
            for (int ls = 0; ls < 25; ls++) {
                double Tt = T_e + lam * dTe;
                double nt = n_e * exp(lam * dlx);
                if (Tt < T_lo) Tt = T_lo; if (Tt > T_hi) Tt = T_hi;
                if (!isfinite(Tt) || !isfinite(nt) || nt <= 0.0) { lam *= 0.5; continue; }
                double q1 = radeq_net(Tt, T_rad, nt, H_photo, H_gamma,
                                      compton0 * nt, ff0 * nt * nt, Gamma_ad,
                                      ca, cb, cbet, n_active, cool_nonneg, C_bb_esc,
                                      emit_nu, nu_mid, nfb)
                          + radeq_Hresp(Tt, gbin, lstar, blag, 1.0, nu_mid, nfb)
                          + RADEQ_LINE_DELTA(Tt, nt)
                          + RADEQ_LINE_RE_TERM(Tt);
                double q2 = nt - (coupled_tdep
                    ? coupled_charge_density_tdep(atom, plasma, s, Tt, nt, time_explosion, n_shells, 0, gamma_jnu)
                    : coupled_charge_density(atom, plasma, s, Tt, nt, n_shells));
                double rn1 = fabs(q1) / (fabs(H_photo) + 1e-300) + fabs(q2) / (nt + 1e-300);
                if (isfinite(rn1) && rn1 < rn0) { Tn = Tt; nn = nt; accepted = 1; break; }
                lam *= 0.5;
            }
            if (!accepted) break;   /* no valid descent step: stall, keep prior state */
            double relT = fabs(Tn - T_e) / (T_e + 1e-300);
            double reln = fabs(nn - n_e) / (n_e + 1e-300);
            T_e = Tn; n_e = nn;
            if (trace_sh == s) {
#ifdef _OPENMP
                #pragma omp critical
#endif
                printf("    [CN-TRACE s=%d it=%2d] rn0=%.3e r1=%.3e r2=%.3e lam=%.2e "
                       "dTe=%.3e dlnne=%.3e -> T_e=%.1f n_e=%.4e relT=%.2e reln=%.2e\n",
                       s, it, rn0, r1, r2, lam, dTe, dlx, T_e, n_e, relT, reln);
            }
            if (relT < 1e-5 && reln < 1e-5) { conv = 1; break; }
        }
#undef RADEQ_LINE_DELTA
        if (!conv) n_stall++;          /* committed a stale (non-converged) iterate */
        if (T_e < 1000.0) { T_e = 1000.0; n_floor_cn++; }  /* under-determined: floor */
        plasma->T_e[s] = T_e;
        plasma->n_electron[s] = n_e;
        newton_owned[s] = 1;
        /* write this shell's ion partition consistent with the converged (T_e,n_e):
         * tdep → the time-dependent tridiagonal; steady → nebular Saha. */
        if (coupled_tdep)
            (void)coupled_charge_density_tdep(atom, plasma, s, T_e, n_e,
                                              time_explosion, n_shells, 1, gamma_jnu);
        else
            compute_ion_populations_shell(atom, plasma, s, n_shells);
        if (septest_Tfix > 0.0) {
            double ne_td_fix = solve_ne_fixed_te(atom, plasma, s, septest_Tfix,
                                                 time_explosion, n_shells, 1, gamma_jnu);
            double ne_st_fix = solve_ne_fixed_te(atom, plasma, s, septest_Tfix,
                                                 time_explosion, n_shells, 0, gamma_jnu);
            double ne_td_cnv = solve_ne_fixed_te(atom, plasma, s, T_e,
                                                 time_explosion, n_shells, 1, gamma_jnu);
            double ne_st_cnv = solve_ne_fixed_te(atom, plasma, s, T_e,
                                                 time_explosion, n_shells, 0, gamma_jnu);
#ifdef _OPENMP
            #pragma omp critical
#endif
            printf("    [SEPTEST s=%2d] T_conv=%.0f n_e_conv=%.4e | Tfix=%.0f: "
                   "tdep=%.4e steady=%.4e | Tconv: tdep=%.4e steady=%.4e\n",
                   s, T_e, n_e, septest_Tfix, ne_td_fix, ne_st_fix,
                   ne_td_cnv, ne_st_cnv);
        }
        n_solved += conv ? 1 : 0;
#ifdef _OPENMP
        if (cn_prof) {
            int tid = omp_get_thread_num();
            double dt = omp_get_wtime() - cn_ts0;
            cn_tsum[tid] += dt; cn_scnt[tid] += 1;
            if (dt > cn_tmax[tid]) { cn_tmax[tid] = dt; cn_smax[tid] = s; }
        }
#endif
    }
    free(emit_nu); free(ca); free(cb); free(cbet); free(gamma_jnu);
    free(gbin); free(chi_loc); free(lstar); free(blag);
    free(et_A); free(et_B); free(et_bet); free(et_dE);
    free(et_nlo); free(et_nup); free(et_Rlu); free(et_Rul);
    }  /* end omp parallel */
    if (cn_prof) {
        double wall = 0.0;
#ifdef _OPENMP
        wall = omp_get_wtime() - cn_wt0;
#endif
        double busy = 0.0, tmax_all = 0.0; long shells_tot = 0;
        int smax_all = -1, tid_max = -1;
        for (int t = 0; t < cn_nthr; t++) {
            busy += cn_tsum[t]; shells_tot += cn_scnt[t];
            if (cn_tmax[t] > tmax_all) { tmax_all = cn_tmax[t]; smax_all = cn_smax[t]; tid_max = t; }
        }
        printf("  [CN-PROF] wall=%.3fs busy_sum=%.3fs threads=%d shells=%ld "
               "speedup=%.2fx eff=%.0f%%\n",
               wall, busy, cn_nthr, shells_tot,
               wall > 0 ? busy / wall : 0.0,
               wall > 0 ? 100.0 * busy / (wall * cn_nthr) : 0.0);
        printf("  [CN-PROF] heaviest shell s=%d (tid=%d) %.3fs = %.0f%% of wall "
               "(tail-imbalance indicator)\n",
               smax_all, tid_max, tmax_all, wall > 0 ? 100.0 * tmax_all / wall : 0.0);
        for (int t = 0; t < cn_nthr; t++)
            if (cn_scnt[t] > 0)
                printf("  [CN-PROF]   tid=%2d shells=%3ld busy=%.3fs maxshell=s%d(%.3fs)\n",
                       t, cn_scnt[t], cn_tsum[t], cn_smax[t], cn_tmax[t]);
        free(cn_tsum); free(cn_tmax); free(cn_scnt); free(cn_smax);
    }
    free(nu_mid);

    /* Reconcile ONLY the shells the Newton does not own (steady-state Saha at the
     * current T_e,n_e). Newton-owned shells keep their consistent partition above;
     * frozen-in shells keep the cascade result written in compute_plasma_state. */
    init_ml_phi_neb_correction();
    init_zeta_override();
    init_twocomp_lock();
    for (int s = 0; s < n_shells; s++) {
        if (newton_owned[s]) continue;
        if (frozenin_is_frozen && s < frozenin_is_frozen_n && frozenin_is_frozen[s])
            continue;
        compute_ion_populations_shell(atom, plasma, s, n_shells);
    }
    free(newton_owned);

    printf("  [COUPLED-NEWTON] %ld/%ld shells converged (%s); T_e[0]=%.0f n_e[0]=%.3e\n",
           n_solved, (long)n_shells, coupled_tdep ? "tdep eq26" : "steady Saha",
           plasma->T_e[0], plasma->n_electron[0]);
    printf("  [COUPLED-NEWTON] NOT-CONVERGED: %ld stall (stale iterate committed), "
           "%ld floor-pinned (T_e=1000K, under-determined) of %ld shells\n",
           n_stall, n_floor_cn, (long)n_shells);
}

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

    /* ---- CMFGEN super-level collapse maps ---- */
    /* Pass A: count super-levels per ion (super idx is per-ion 0-based,
     * contiguous, from the levels.csv "super_level" column; = level_num for
     * non-collapsed ions). */
    nlte->nlte_ion_super_offset[0] = 0;
    for (int i = 0; i < NLTE_MAX_IONS; i++) {
        int Z = nlte->nlte_Z[i];
        int ion = nlte->nlte_ion[i];
        int max_s = -1;
        for (int l = 0; l < atom->n_levels; l++) {
            if (atom->level_Z[l] == Z && atom->level_ion[l] == ion) {
                int s = atom->level_super[l];
                if (s > max_s) max_s = s;
            }
        }
        int n_super_i = (max_s >= 0) ? (max_s + 1) : 0;
        nlte->nlte_ion_super_offset[i + 1] = nlte->nlte_ion_super_offset[i] + n_super_i;
    }
    nlte->n_super_total = nlte->nlte_ion_super_offset[NLTE_MAX_IONS];

    /* Pass B: FL nlte idx -> global SL solve idx, and lowest-E anchor per SL. */
    nlte->fl_to_super = (int *)malloc(nlte->n_nlte_levels_total * sizeof(int));
    for (int g = 0; g < nlte->n_nlte_levels_total; g++) nlte->fl_to_super[g] = g;
    nlte->super_anchor_global = (int *)malloc(
        (nlte->n_super_total > 0 ? nlte->n_super_total : 1) * sizeof(int));
    for (int s = 0; s < nlte->n_super_total; s++) nlte->super_anchor_global[s] = -1;
    for (int l = 0; l < atom->n_levels; l++) {
        for (int i = 0; i < NLTE_MAX_IONS; i++) {
            if (atom->level_Z[l] == nlte->nlte_Z[i] &&
                atom->level_ion[l] == nlte->nlte_ion[i]) {
                int g_nlte = nlte->global_to_nlte_level[l];
                int sl_global = nlte->nlte_ion_super_offset[i] + atom->level_super[l];
                nlte->fl_to_super[g_nlte] = sl_global;
                int prev = nlte->super_anchor_global[sl_global];
                if (prev < 0 ||
                    atom->level_energy_eV[l] < atom->level_energy_eV[prev])
                    nlte->super_anchor_global[sl_global] = l;
                break;
            }
        }
    }

    /* Activate super mode only when the env knob is on AND the data actually
     * collapses (otherwise stay on the byte-identical FL path). */
    {
        const char *e = getenv("LUMINA_SUPER_LEVELS");
        int env_on = (e && atoi(e) != 0);
        nlte->super_mode = (env_on && nlte->n_super_total < nlte->n_nlte_levels_total) ? 1 : 0;
    }
    nlte->within_sl_frac = (double *)malloc(
        (size_t)nlte->n_nlte_levels_total * n_shells * sizeof(double));
    for (size_t k = 0; k < (size_t)nlte->n_nlte_levels_total * n_shells; k++)
        nlte->within_sl_frac[k] = 1.0;
    printf("  [NLTE] Super-levels: %s (%d FL -> %d SL across ions)\n",
           nlte->super_mode ? "ACTIVE" : "off (identity)",
           nlte->n_nlte_levels_total, nlte->n_super_total);

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
    free(nlte->fl_to_super);
    free(nlte->super_anchor_global);
    free(nlte->within_sl_frac);
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

/* Cap super-Planckian J_nu in UV (λ < lambda_max) at W_cap * B_nu(T_rad).
 * UV reprocessing in inner shells drives W to 1.7–2.7 (see
 * project_pathB_prime_jnu_diagnosis.md), pumping bound-free rates 1.7–2.7×
 * above what an undiluted Planck would give. Clipping J_nu directly attacks
 * the σ_bf saturation gap without touching ion-lock or atomic data.
 * Controlled by:
 *   LUMINA_J_NU_UV_CAP            (0/1, default 0)
 *   LUMINA_J_NU_UV_CAP_LAMBDA_MAX (Å,   default 3500)
 *   LUMINA_J_NU_UV_W_CAP          (-,   default 1.0 = Planckian) */
void nlte_apply_uv_jnu_cap(NLTEConfig *nlte, PlasmaState *plasma, int n_shells) {
    static int initialized = 0;
    static int enabled = 0;
    static double lambda_max_aa = 3500.0;
    static double lambda_min_aa = 2000.0;
    static double W_cap = 1.0;
    if (!initialized) {
        const char *env_en   = getenv("LUMINA_J_NU_UV_CAP");
        enabled = (env_en != NULL && atoi(env_en) != 0);
        const char *env_lmax = getenv("LUMINA_J_NU_UV_CAP_LAMBDA_MAX");
        if (env_lmax) lambda_max_aa = atof(env_lmax);
        const char *env_lmin = getenv("LUMINA_J_NU_UV_CAP_LAMBDA_MIN");
        if (env_lmin) lambda_min_aa = atof(env_lmin);
        const char *env_w    = getenv("LUMINA_J_NU_UV_W_CAP");
        if (env_w)   W_cap = atof(env_w);
        if (enabled) {
            printf("  [J_nu UV cap] enabled: %.0f A <= lambda <= %.0f A, W <= %.2f\n",
                   lambda_min_aa, lambda_max_aa, W_cap);
        }
        initialized = 1;
    }
    if (!enabled) return;

    /* λ_min < λ < λ_max  ⇔  nu_lo < nu < nu_hi  (note swap) */
    double nu_lo = C_SPEED_OF_LIGHT / (lambda_max_aa * 1e-8); /* Hz */
    double nu_hi = C_SPEED_OF_LIGHT / (lambda_min_aa * 1e-8); /* Hz */
    long long n_capped = 0;
    double sum_ratio = 0.0;
    double max_ratio = 0.0;

    for (int s = 0; s < n_shells; s++) {
        double T = plasma->T_rad[s];
        if (T <= 0.0) continue;
        for (int b = 0; b < nlte->n_freq_bins; b++) {
            double nu_mid = nlte->nu_min * exp((b + 0.5) * nlte->d_log_nu);
            if (nu_mid < nu_lo || nu_mid > nu_hi) continue; /* out of band */
            int idx = s * nlte->n_freq_bins + b;
            double J = nlte->J_nu[idx];
            double Bnu = planck_bnu(T, nu_mid);
            double J_max = W_cap * Bnu;
            if (J_max > 0.0 && J > J_max) {
                double ratio = J / J_max;
                sum_ratio += ratio;
                if (ratio > max_ratio) max_ratio = ratio;
                nlte->J_nu[idx] = J_max;
                n_capped++;
            }
        }
    }
    if (n_capped > 0) {
        printf("  [J_nu UV cap] iter %d: capped %lld bins, "
               "mean J/J_max = %.2f, max = %.2f\n",
               nlte->current_iter, n_capped,
               sum_ratio / (double)n_capped, max_ratio);
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
                                GammaDeposition *gamma_dep,
                                const NLTERateLookup *lookup,
                                int pair_idx) {
    int lev_start = nlte->nlte_ion_level_offset[ion_idx_lo];
    int n_shells = plasma->n_shells;
    double T_rad = plasma->T_rad[shell];
    double T_e   = plasma->T_e[shell];
    double n_e   = plasma->n_electron[shell];

    /* Column-major access: ACM(i,j) = A_cm[j*N + i] */
    #define ACM(i,j) A_cm[(j) * N + (i)]

    /* Super-level solve indexing. In identity mode fl_to_super[g]==g and
     * super_start==lev_start, so SOLVE_OF reduces to the FL nlte index and
     * FRAC_OF==1 — the matrix is byte-identical to the FL solve. In super mode
     * the matrix dim N is the super-level count and FL contributions aggregate
     * into their SL row/col, weighted by the within-SL Boltzmann fraction. */
    int super_start = nlte->nlte_ion_super_offset[ion_idx_lo];
    int n_lo_super  = nlte->nlte_ion_super_offset[ion_idx_lo + 1] - super_start;
    #define SOLVE_OF(fl_g) (nlte->fl_to_super[(fl_g)] - super_start)
    #define FRAC_OF(fl_g)  (nlte->within_sl_frac[(size_t)(fl_g) * n_shells + shell])

    /* α #286 floor-pop regularization: track bb-connectivity per level so we
     * can detect bb-isolated upper-ion levels (e.g. Cr/Fe/Co III top) that
     * cause conservation-row collapse (#219e). Allocated only when knob is on. */
    static int floor_reg_init = 0;
    static int floor_reg_mode = 0;
    if (!floor_reg_init) {
        const char *e = getenv("LUMINA_NLTE_FLOOR_REG");
        if (e && atoi(e) != 0) floor_reg_mode = 1;
        floor_reg_init = 1;
    }
    int *bb_connected = floor_reg_mode ? (int *)calloc(N, sizeof(int)) : NULL;

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

        int fl_lo_g = nlte->global_to_nlte_level[lower_global];
        int fl_up_g = nlte->global_to_nlte_level[upper_global];
        int i_lo = SOLVE_OF(fl_lo_g);
        int i_up = SOLVE_OF(fl_up_g);
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

        /* Within-SL Boltzmann fractions: only the fraction of the lower SL
         * residing in this FL absorbs, only the fraction of the upper SL in
         * this FL emits. Intra-SL transitions (i_lo==i_up) cancel — correct,
         * since within an SL the populations are pinned to Boltzmann. */
        double f_lo = FRAC_OF(fl_lo_g);
        double f_up = FRAC_OF(fl_up_g);
        ACM(i_up, i_lo) += total_up   * f_lo;
        ACM(i_lo, i_up) += total_down * f_up;
        ACM(i_lo, i_lo) -= total_up   * f_lo;
        ACM(i_up, i_up) -= total_down * f_up;

        if (bb_connected && (total_up + total_down) > 1e-30) {
            bb_connected[i_lo] = 1;
            bb_connected[i_up] = 1;
        }
    }

    /* ---- Photoionization / Recombination ---- */
    int Z_elem = nlte->nlte_Z[ion_idx_lo];
    double chi_eV = find_ioniz_energy(atom, Z_elem, nlte->nlte_ion[ion_idx_lo]);
    double chi_erg = chi_eV * EV_TO_ERG;
    double nu_edge = chi_erg / H_PLANCK;

    int n_lo_levels = nlte->nlte_ion_level_offset[ion_idx_lo + 1] -
                      nlte->nlte_ion_level_offset[ion_idx_lo];
    /* ground_hi is the matrix row/col of the upper-ion ground: it is the SL
     * count of the lower ion (== n_lo_levels in identity mode). The upper ion
     * is not collapsed here, so its FL == SL. */
    int ground_hi = n_lo_super;
    int hi_ground_global_nlte = nlte->nlte_ion_level_offset[ion_idx_hi];

    /* Diagnostic accumulators for LUMINA_NLTE_RATE_DUMP */
    double sum_R_bf = 0.0, sum_R_rec = 0.0;
    double sum_R_bf_ground = 0.0, sum_R_rec_ground = 0.0;
    double n_star_ground = 0.0;
    int sum_R_bf_levels = 0;

    if (ground_hi < N && nu_edge > 0.0 && nu_edge < nlte->nu_max) {
        /* Task #138 Heavy.1: per-level CMFGEN σ_bf where available, Kramers fallback.
         * Same grid as NLTE J_ν (NLTE_N_FREQ_BINS, log-spaced) → direct index. */
        int Z_ion = nlte->nlte_Z[ion_idx_lo];
        int ion_lo_stage = nlte->nlte_ion[ion_idx_lo];
        double sigma_0 = get_bf_sigma0(Z_ion, ion_lo_stage);
        if (sigma_0 <= 0.0) {
            int Z_eff = Z_ion - ion_lo_stage;
            if (Z_eff < 1) Z_eff = 1;
            sigma_0 = 7.91e-18 / ((double)Z_eff * (double)Z_eff);
        }
        const int use_cmfgen = atom->cmfgen_loaded &&
                               atom->cmfgen_n_freq_bins == nlte->n_freq_bins;

        /* Task #40 (A)+(B): GPU lookup path. R_bf_table is col-major
         * [L_phot_total × n_shells]; pair_idx selects the row offset. */
        int use_gpu_R_bf = (lookup != NULL && lookup->R_bf_table != NULL &&
                            lookup->phot_offset != NULL && pair_idx >= 0);
        int phot_base    = use_gpu_R_bf ? lookup->phot_offset[pair_idx] : 0;
        int L_phot_total = use_gpu_R_bf ? lookup->L_phot_total : 0;

        for (int lev = 0; lev < n_lo_levels; lev++) {
            int global_lev = nlte->nlte_to_global_level[lev_start + lev];
            double E_lev = atom->level_energy_eV[global_lev] * EV_TO_ERG;
            double nu_thresh = (chi_erg - E_lev) / H_PLANCK;
            if (nu_thresh <= 0.0) continue;

            int level_has_cmfgen = use_cmfgen && atom->cmfgen_has_sigma[global_lev];
            const double *sigma_row = level_has_cmfgen ?
                &atom->cmfgen_sigma_bf[(size_t)global_lev *
                                       (size_t)atom->cmfgen_n_freq_bins] : NULL;

            double R_bf = 0.0;
            if (use_gpu_R_bf) {
                int idx = phot_base + lev;
                R_bf = lookup->R_bf_table[(size_t)shell * L_phot_total + idx];
            } else {
                for (int bb = 0; bb < nlte->n_freq_bins; bb++) {
                    double log_nu_lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
                    double nu_bin = exp(log_nu_lo + 0.5 * nlte->d_log_nu);
                    if (nu_bin < nu_thresh) continue;
                    double delta_nu = exp(log_nu_lo + nlte->d_log_nu) - exp(log_nu_lo);
                    double J_bin = nlte->J_nu[shell * nlte->n_freq_bins + bb];
                    double sigma;
                    if (sigma_row) {
                        sigma = sigma_row[bb];
                        if (sigma <= 0.0) continue;
                    } else {
                        sigma = sigma_0 * pow(nu_thresh / nu_bin, 3.0);
                    }
                    R_bf += 4.0 * M_PI_VAL * J_bin * sigma / (H_PLANCK * nu_bin) * delta_nu;
                }
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
                        int global_ghi = nlte->nlte_to_global_level[hi_ground_global_nlte];
                        g_ion = atom->level_g[global_ghi];
                        if (g_ion < 1) g_ion = 1;
                    }
                    n_star_ratio = n_e * thermal_deBroglie *
                        (double)g_lev / (2.0 * (double)g_ion) * exp_factor;
                    if (n_star_ratio > 1e30) n_star_ratio = 1e30;
                }
            }
            double R_rec = R_bf * n_star_ratio;

            /* Collapse this FL to its SL solve index. Ionization out is weighted
             * by the FL's within-SL fraction (only that fraction of the SL pop
             * sits in this FL and ionizes); recombination in is unweighted and
             * sums over the SL's FL (total recomb captured by the SL). */
            int sl = SOLVE_OF(lev_start + lev);
            double f_lev = FRAC_OF(lev_start + lev);
            if (R_bf > 0.0 && sl >= 0 && sl < N && ground_hi < N) {
                ACM(ground_hi, sl) += R_bf * f_lev;
                ACM(sl, sl)        -= R_bf * f_lev;
                ACM(sl, ground_hi) += R_rec;
                ACM(ground_hi, ground_hi) -= R_rec;

                sum_R_bf  += R_bf;
                sum_R_rec += R_rec;
                sum_R_bf_levels++;
                if (lev == 0) {
                    sum_R_bf_ground  = R_bf;
                    sum_R_rec_ground = R_rec;
                    n_star_ground    = n_star_ratio;
                }
            }
        }
    }

    /* Diagnostic dump: per-(Z, ion_pair, shell) rate balance summary.
     * Writes to nlte_rate_balance.csv (append mode, header on first call). */
    {
        const char *env = getenv("LUMINA_NLTE_RATE_DUMP");
        if (env && env[0] == '1') {
            static int header_written = 0;
            #ifdef _OPENMP
            #pragma omp critical(nlte_rate_dump)
            #endif
            {
                FILE *fp = fopen("nlte_rate_balance.csv", header_written ? "a" : "w");
                if (fp) {
                    if (!header_written) {
                        fprintf(fp, "Z,ion_lo,shell,N_levels,T_e,T_rad,W,n_e,"
                                    "n_ion_lo_neb,n_ion_hi_neb,"
                                    "sum_R_bf,sum_R_rec,sum_R_bf_lev_count,"
                                    "R_bf_ground,R_rec_ground,n_star_ground,"
                                    "ratio_RrecRbf,ratio_NLTE_predicted_lo_to_hi\n");
                        header_written = 1;
                    }
                    int Z_d = Z_elem;
                    int ion_lo_d = nlte->nlte_ion[ion_idx_lo];
                    int ip_lo = find_ion_pop_idx(atom, Z_d, ion_lo_d);
                    int ip_hi = find_ion_pop_idx(atom, Z_d, nlte->nlte_ion[ion_idx_hi]);
                    double n_neb_lo = (ip_lo >= 0) ?
                        atom->ion_number_density[ip_lo * n_shells + shell] : 0.0;
                    double n_neb_hi = (ip_hi >= 0) ?
                        atom->ion_number_density[ip_hi * n_shells + shell] : 0.0;
                    double ratio_RR = (sum_R_bf > 0.0) ? sum_R_rec / sum_R_bf : 0.0;
                    fprintf(fp, "%d,%d,%d,%d,%.2f,%.2f,%.6e,%.6e,"
                                "%.6e,%.6e,"
                                "%.6e,%.6e,%d,"
                                "%.6e,%.6e,%.6e,"
                                "%.6e,%.6e\n",
                            Z_d, ion_lo_d, shell, N, T_e, T_rad,
                            plasma->W[shell], n_e,
                            n_neb_lo, n_neb_hi,
                            sum_R_bf, sum_R_rec, sum_R_bf_levels,
                            sum_R_bf_ground, sum_R_rec_ground, n_star_ground,
                            ratio_RR, n_star_ground);
                    fclose(fp);
                }
            }
        }
    }

    /* ---- Step 1.5: Charge Exchange rates ---- */
    int Z_pair = nlte->nlte_Z[ion_idx_lo]; /* element Z for this ion pair */
    int ion_lo_stage = nlte->nlte_ion[ion_idx_lo];   /* e.g. 1 for II */
    int ion_hi_stage = nlte->nlte_ion[ion_idx_hi];   /* e.g. 2 for III */

    /* ---- Heavy.2 / Task #139: Dielectronic recombination ----
     * α_DR(T_e) is added as a non-Milne recombination channel from
     * upper-ion ground (ground_hi) into lower-ion ground (lev=0). This
     * lifts R_rec where Milne-from-CMFGEN-σ_bf alone gives Saha-LTE-at-T_e
     * over-ionization. Cascades through the lower-ion bb network repopulate
     * excited levels self-consistently. Requires the ion_recomb=ion_hi_stage
     * entry in DR_TABLE; if missing the rate is zero (legacy behavior).
     *
     * LUMINA_DR_FLOOR_CMS env var (default 0): empirical phenomenological
     * α_DR floor in cm³/s, applied uniformly to *every* NLTE (II,III) pair.
     * Probes "would any extra recombination close the gap?" independent of
     * T-dependence. Mazzotta LS-coupling misses low-T near-threshold
     * resonances; this lets us decouple "is more recomb needed" from
     * "is the T-shape right". Use ~1e-12 to 1e-10 cm³/s as test range. */
    {
        const DRCoefficient *coef = dr_lookup(Z_pair, ion_hi_stage);
        double alpha_dr = (coef && n_e > 0.0) ? dr_alpha_eval(coef, T_e) : 0.0;

        static int floor_init = 0;
        static double alpha_dr_floor = 0.0;
        if (!floor_init) {
            const char *fenv = getenv("LUMINA_DR_FLOOR_CMS");
            if (fenv) alpha_dr_floor = atof(fenv);
            floor_init = 1;
        }
        if (alpha_dr_floor > 0.0 && alpha_dr < alpha_dr_floor)
            alpha_dr = alpha_dr_floor;

        double R_dr = alpha_dr * n_e;   /* [s⁻¹] per upper-ion ion */
        if (R_dr > 0.0 && ground_hi < N) {
            ACM(0, ground_hi)         += R_dr;
            ACM(ground_hi, ground_hi) -= R_dr;
        }
    }

    for (int r = 0; r < CE_N_REACTIONS; r++) {
        const ChargeExchangeReaction *ce = &CE_REACTIONS[r];
        double k_fwd = ce->rate_coeff * pow(T_e / 1e4, ce->alpha);
        double k_rev = k_fwd * exp(ce->delta_E_eV * EV_TO_ERG /
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

            /* Forward: all A^ion_A levels → A^(ion_A+1) ground.
             * Level-independent rate: applied once per super-level. */
            for (int sl = 0; sl < n_lo_super; sl++) {
                ACM(ground_hi, sl) += R_fwd;
                ACM(sl, sl)        -= R_fwd;
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
            /* Reverse: all B^(ion_B-1) levels → B^ion_B ground.
             * Level-independent rate: applied once per super-level. */
            for (int sl = 0; sl < n_lo_super; sl++) {
                ACM(ground_hi, sl) += R_rev;
                ACM(sl, sl)        -= R_rev;
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

    /* ---- Time-dependent ionization: backward-Euler dn_i/dt term (option A) ----
     *
     * Steady-state assumes dn_i/dt = 0, i.e. infinite time to equilibrate, so at
     * the cold outer ejecta (T~2500K) it recombines to near-neutral and n_e
     * collapses. CMFGEN keeps the ions frozen because recombination there is far
     * slower than the expansion age (tau_rec/t_exp ~ 300). We restore that by
     * solving the implicit (backward-Euler) step
     *     (n^new - n^old)/Dt = A n^new   =>   (A - I/Dt) n^new = -n^old/Dt,
     * with Dt = time_explosion (single step from t=0) and n^old the fully-ionized
     * initial condition (all pair mass in the hi-ion ground). The 1/Dt diagonal
     * (= 1.19e-5 s^-1 at 0.976d) sits exactly at the tau_rec = t_exp crossover:
     * rows whose net rate >> 1/Dt relax to steady state, rows whose rate << 1/Dt
     * stay at n^old. The conservation row below still pins Sum n to the mass-based
     * nebular total, so only the ion PARTITION becomes time-dependent. The
     * conservation/anchor rows are overwritten afterwards, dropping the time term
     * from exactly those rows (correct: they are closures, not rate equations). */
    if (nlte_timedep_active() && time_explosion > 0.0) {
        double inv_dt = 1.0 / time_explosion;
        double n_pair_total = nlte_pair_total_density(nlte, atom, plasma,
                                  nlte->nlte_Z[ion_idx_lo],
                                  ion_idx_lo, ion_idx_hi, shell);
        for (int i = 0; i < N; i++) {
            double n_old_i = (i == ground_hi) ? n_pair_total : 0.0;
            ACM(i, i) -= inv_dt;
            b[i]      -= n_old_i * inv_dt;
        }
    }

    /* ---- Conservation equation(s) ----
     *
     * Default: single combined row (sum over ALL levels of (lo,hi) = nebular total
     * ion-pair density). This lets the bf+rec+CE+DR matrix entries determine
     * the ion partitioning self-consistently. Works only if those rates are
     * physically reasonable.
     *
     * LUMINA_NLTE_ION_LOCK=1: Mihalas-Lucy hybrid. Replace TWO rows
     * (last row of lower-ion block, last row of upper-ion block) with
     * per-ion conservation. n_lo_total and n_hi_total are pinned to nebular
     * W·ζ-Saha values. NLTE then only redistributes excited levels within
     * each ion. Avoids the Milne-T_e-vs-T_rad over-ionization trap that DR
     * (any magnitude) cannot fix. Standard SN-code approach (TARDIS exact).
     */
    int Z_nl = nlte->nlte_Z[ion_idx_lo];
    int ion_lock_mode = nlte_ion_lock_active(nlte->current_iter);

    /* α #286 floor-pop regularization: choose alternate conservation rows for
     * upper- and lower-ion blocks when the default top row is bb-isolated, and
     * write Boltzmann@T_rad anchors on every remaining bb-isolated row in the
     * upper-ion block. Resolves #219e (Cr/Fe/Co III top-level pop collapse). */
    int alt_row_hi = N - 1;
    int alt_row_lo = n_lo_super - 1;
    if (bb_connected) {
        /* Pick the last bb-connected upper-ion row as conservation destination.
         * Falls back to N-1 if every row is bb-isolated (shouldn't happen). */
        for (int k = N - 1; k >= n_lo_super; k--) {
            if (bb_connected[k]) { alt_row_hi = k; break; }
        }
        /* Same for lower-ion block (ion-lock mode only). */
        for (int k = n_lo_super - 1; k >= 0; k--) {
            if (bb_connected[k]) { alt_row_lo = k; break; }
        }

        /* For each remaining bb-isolated upper-ion level (excluding the
         * conservation row), overwrite the row with a dilute (nebular)
         * Boltzmann anchor:
         *   x[k] - W * (g_k/g_ref) * exp(-(E_k-E_ref)/kT_rad) * x[ref] = 0
         * where ref = upper-ion ground (n_lo_levels) and W = plasma dilution
         * factor for this shell. The W factor depletes high-E anchor pops
         * in outer (dilute) shells per the standard nebular approximation,
         * fixing the NIR over-emission from W=1 LTE-at-T_rad. */
        if (ground_hi < N) {
            double W_shell = plasma->W[shell];
            if (!isfinite(W_shell) || W_shell <= 0.0) W_shell = 1.0;
            int ref_global = nlte->super_anchor_global[super_start + ground_hi];
            double E_ref = atom->level_energy_eV[ref_global] * EV_TO_ERG;
            int g_ref = atom->level_g[ref_global];
            if (g_ref < 1) g_ref = 1;
            for (int k = n_lo_super; k < N; k++) {
                if (bb_connected[k]) continue;
                if (k == alt_row_hi) continue; /* reserved for conservation */
                if (k == ground_hi) continue;  /* anchor reference itself */
                int gk_global = nlte->super_anchor_global[super_start + k];
                double E_k = atom->level_energy_eV[gk_global] * EV_TO_ERG;
                int g_k = atom->level_g[gk_global];
                if (g_k < 1) g_k = 1;
                double dE = E_k - E_ref;
                if (dE < 0.0) dE = 0.0;
                double boltz_ratio = W_shell * (double)g_k / (double)g_ref *
                                     exp(-dE / (K_BOLTZMANN * T_rad));
                if (!isfinite(boltz_ratio)) boltz_ratio = 0.0;
                for (int j = 0; j < N; j++) ACM(k, j) = 0.0;
                ACM(k, k) = 1.0;
                ACM(k, ground_hi) = -boltz_ratio;
                b[k] = 0.0;
            }
        }
    }

    if (ion_lock_mode && n_lo_super > 0 && n_lo_super < N) {
        double n_lo_total = 0.0;
        double n_hi_total = 0.0;
        int ip_lo = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[ion_idx_lo]);
        int ip_hi = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[ion_idx_hi]);
        if (ip_lo >= 0)
            n_lo_total = atom->ion_number_density[ip_lo * n_shells + shell];
        if (ip_hi >= 0)
            n_hi_total = atom->ion_number_density[ip_hi * n_shells + shell];

        int row_lo = alt_row_lo;
        int row_hi = alt_row_hi;
        for (int j = 0; j < N; j++) {
            ACM(row_lo, j) = (j < n_lo_super) ? 1.0 : 0.0;
            ACM(row_hi, j) = (j >= n_lo_super) ? 1.0 : 0.0;
        }
        b[row_lo] = n_lo_total;
        b[row_hi] = n_hi_total;
    } else {
        double n_total = nlte_pair_total_density(nlte, atom, plasma, Z_nl,
                                                  ion_idx_lo, ion_idx_hi, shell);
        int row = bb_connected ? alt_row_hi : (N - 1);
        for (int j = 0; j < N; j++)
            ACM(row, j) = 1.0;
        b[row] = n_total;
    }

    if (bb_connected) free(bb_connected);

    #undef ACM
    #undef SOLVE_OF
    #undef FRAC_OF
}

/* CPU NLTE solver: assemble + Gauss elimination for one ion pair in one shell */
static void nlte_solve_ion_shell(NLTEConfig *nlte, AtomicData *atom,
                                  PlasmaState *plasma, OpacityState *opacity,
                                  int ion_idx_lo, int ion_idx_hi,
                                  int shell, double time_explosion,
                                  GammaDeposition *gamma_dep,
                                  int pair_shares_slot) {
    int lev_start = nlte->nlte_ion_level_offset[ion_idx_lo];
    int super_start = nlte->nlte_ion_super_offset[ion_idx_lo];
    /* Matrix is assembled and solved on super-levels (N); full-level pops are
     * then redistributed within each SL by the within-SL Boltzmann fraction.
     * In identity mode N == N_fl and the SL->FL expansion is a no-op. */
    int N    = nlte->nlte_ion_super_offset[ion_idx_hi + 1] - super_start;
    int N_fl = nlte->nlte_ion_level_offset[ion_idx_hi + 1] - lev_start;
    if (N <= 0 || N_fl <= 0) return;
    int n_shells = plasma->n_shells;
    int n_lo_super = nlte->nlte_ion_super_offset[ion_idx_lo + 1] - super_start;

    double *A_cm = (double *)calloc((size_t)N * N, sizeof(double));
    double *b = (double *)calloc((size_t)N, sizeof(double));

    /* Dead-pair skip (mirror of the CUDA assembly skip): no atoms of this
     * element here -> ~0 pops either way; route straight to the Boltzmann
     * fallback below, skipping the costly assemble+solve. Env-gated. */
    int ret = 0;
    int has_nonfinite = 0;
    if (nlte_skip_dead_pairs()) {
        int Z_dead = nlte->nlte_Z[ion_idx_lo];
        double n_tot = nlte_pair_total_density(nlte, atom, plasma, Z_dead,
                                               ion_idx_lo, ion_idx_hi, shell);
        if (n_tot < 1e-10) has_nonfinite = 1;
    }

    if (!has_nonfinite) {
        nlte_assemble_rate_matrix(nlte, atom, plasma, opacity,
                                   ion_idx_lo, ion_idx_hi, shell, time_explosion,
                                   A_cm, b, N, gamma_dep,
                                   NULL, -1);

        ret = gauss_solve(A_cm, b, N);
    }

    /* Detect non-finite output: gauss_solve may succeed but produce NaN/Inf
     * when the rate matrix is ill-conditioned at high T_e. */
    if (!has_nonfinite && ret == 0) {
        for (int i = 0; i < N; i++) {
            if (!isfinite(b[i])) { has_nonfinite = 1; break; }
        }
        /* Boltzmann-ceiling sanity gate: a near-singular matrix can yield a
         * FINITE but inverted solution (excited levels 1e9-1e11x ground).
         * Reject when any level exceeds its ion ground pop by more than
         * (g_i/g_ground)*margin, routing into the Boltzmann fallback below. */
        double inv_ceil = nlte_inv_ceiling();
        if (!has_nonfinite && inv_ceil > 0.0) {
            int n_lo = n_lo_super;   /* SL-space ground/excited split */
            int g0_lo = atom->level_g[nlte->super_anchor_global[super_start]];
            int g0_hi = (n_lo < N) ?
                atom->level_g[nlte->super_anchor_global[super_start + n_lo]] : 1;
            double b0_lo = b[0];
            double b0_hi = (n_lo < N) ? b[n_lo] : 1.0;
            for (int i = 0; i < N; i++) {
                double bg = (i < n_lo) ? b0_lo : b0_hi;
                int gg = (i < n_lo) ? g0_lo : g0_hi;
                if (bg <= 0.0) {
                    /* Empty/negative ground with a populated excited level is
                     * itself an inversion — the garbage solve drained ground. */
                    if (b[i] > 0.0) { has_nonfinite = 1; break; }
                    continue;
                }
                int gi = atom->level_g[nlte->super_anchor_global[super_start + i]];
                double ceil_ratio = ((double)gi / (double)(gg > 0 ? gg : 1)) * inv_ceil;
                if (b[i] / bg > ceil_ratio) { has_nonfinite = 1; break; }
            }
        }
    }

    /* LUMINA_NLTE_FORCE_LTE_LEVELS=1: bypass rate-solve result, use Boltzmann@T_rad. */
    static int cpu_force_lte_init = 0;
    static int cpu_force_lte_mode = 0;
    if (!cpu_force_lte_init) {
        const char *e = getenv("LUMINA_NLTE_FORCE_LTE_LEVELS");
        if (e && atoi(e) != 0) cpu_force_lte_mode = 1;
        cpu_force_lte_init = 1;
    }
    if (cpu_force_lte_mode) has_nonfinite = 1;

    if (ret == 0 && !has_nonfinite) {
        /* Clamp negatives + rescale to enforce conservation.
         * Default: combined Σ x_i = n_pair_total.
         * LUMINA_NLTE_ION_LOCK=1: per-ion rescale to (n_lo_total, n_hi_total)
         * to preserve the ion-lock from the matrix. */
        int Z_nl = nlte->nlte_Z[ion_idx_lo];
        /* pair_shares_slot: overlapping O pairs ({28,29} & {29,30}) must
         * pin each ion to its own nebular total. Combined renorm scales O III
         * by the O II-dominated pair sum → O III 1e13x / O I 500x blowup. */
        int lock = nlte_ion_lock_active(nlte->current_iter) ||
                   nlte_per_ion_rescale_active() || pair_shares_slot;

        int n_lo_levels = nlte->nlte_ion_level_offset[ion_idx_lo + 1] -
                          nlte->nlte_ion_level_offset[ion_idx_lo];

        for (int i = 0; i < N; i++) {
            if (b[i] < 0.0) b[i] = 1e-30;
        }

        /* Redistribute super-level solution to full levels:
         *   n_FL = x_SL[SL(FL)] * f_FL,   f_FL = within-SL Boltzmann fraction.
         * Identity mode: SL(FL)==FL nlte idx and f_FL==1, so xfl == b. */
        double *xfl = (double *)malloc((size_t)N_fl * sizeof(double));
        for (int i = 0; i < N_fl; i++) {
            int sl = nlte->fl_to_super[lev_start + i] - super_start;
            double f = nlte->within_sl_frac[(size_t)(lev_start + i) * n_shells + shell];
            xfl[i] = b[sl] * f;
        }

        if (lock && n_lo_levels > 0 && n_lo_levels < N_fl) {
            double n_lo_total = 0.0, n_hi_total = 0.0;
            int ip_lo = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[ion_idx_lo]);
            int ip_hi = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[ion_idx_hi]);
            if (ip_lo >= 0)
                n_lo_total = atom->ion_number_density[ip_lo * n_shells + shell];
            if (ip_hi >= 0)
                n_hi_total = atom->ion_number_density[ip_hi * n_shells + shell];

            double sum_lo = 0.0, sum_hi = 0.0;
            for (int i = 0; i < n_lo_levels; i++) sum_lo += xfl[i];
            for (int i = n_lo_levels; i < N_fl; i++) sum_hi += xfl[i];
            double scale_lo = (sum_lo > 0.0 && n_lo_total > 0.0) ? n_lo_total / sum_lo : 1.0;
            double scale_hi = (sum_hi > 0.0 && n_hi_total > 0.0) ? n_hi_total / sum_hi : 1.0;
            for (int i = 0; i < n_lo_levels; i++) {
                nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] =
                    xfl[i] * scale_lo;
            }
            for (int i = n_lo_levels; i < N_fl; i++) {
                nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] =
                    xfl[i] * scale_hi;
            }
        } else {
            double n_total = nlte_pair_total_density(nlte, atom, plasma, Z_nl,
                                                      ion_idx_lo, ion_idx_hi, shell);
            double sum = 0.0;
            for (int i = 0; i < N_fl; i++) sum += xfl[i];
            double scale = (sum > 0.0 && n_total > 0.0) ? n_total / sum : 1.0;
            for (int i = 0; i < N_fl; i++) {
                nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] =
                    xfl[i] * scale;
            }
        }
        free(xfl);
    } else {
        /* Singular or non-finite: fall back to Boltzmann at T_rad.
         * In ion-lock mode, rescale per-ion (Boltzmann shape, ion totals
         * pinned to nebular) so the lock invariant survives a failed solve. */
        double T_rad = plasma->T_rad[shell];
        int Z_nl = nlte->nlte_Z[ion_idx_lo];
        int lock = nlte_ion_lock_active(nlte->current_iter) ||
                   nlte_per_ion_rescale_active() || pair_shares_slot;
        int n_lo_levels = nlte->nlte_ion_level_offset[ion_idx_lo + 1] -
                          nlte->nlte_ion_level_offset[ion_idx_lo];

        static int fallback_warn = 0;
        if (fallback_warn < 16) {
            fprintf(stderr,
                "[NLTE-FALLBACK] CPU pair (Z=%d, ions %d/%d, N=%d) shell=%d "
                "ret=%d has_nonfinite=%d -> Boltzmann@T_rad\n",
                Z_nl, nlte->nlte_ion[ion_idx_lo], nlte->nlte_ion[ion_idx_hi],
                N, shell, ret, has_nonfinite);
            fallback_warn++;
        }

        double pop_buf_unused; (void)pop_buf_unused;
        double sum_lo = 0.0, sum_hi = 0.0;
        for (int i = 0; i < N_fl; i++) {
            int global = nlte->nlte_to_global_level[lev_start + i];
            double E = atom->level_energy_eV[global] * EV_TO_ERG;
            int g = atom->level_g[global];
            double pop = (double)g * exp(-E / (K_BOLTZMANN * T_rad));
            nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] = pop;
            if (i < n_lo_levels) sum_lo += pop;
            else sum_hi += pop;
        }

        if (lock && n_lo_levels > 0 && n_lo_levels < N_fl) {
            double n_lo_total = 0.0, n_hi_total = 0.0;
            int ip_lo = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[ion_idx_lo]);
            int ip_hi = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[ion_idx_hi]);
            if (ip_lo >= 0) n_lo_total = atom->ion_number_density[ip_lo * n_shells + shell];
            if (ip_hi >= 0) n_hi_total = atom->ion_number_density[ip_hi * n_shells + shell];
            double scale_lo = (sum_lo > 0.0 && n_lo_total > 0.0) ? n_lo_total / sum_lo : 1.0;
            double scale_hi = (sum_hi > 0.0 && n_hi_total > 0.0) ? n_hi_total / sum_hi : 1.0;
            for (int i = 0; i < n_lo_levels; i++)
                nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] *= scale_lo;
            for (int i = n_lo_levels; i < N_fl; i++)
                nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] *= scale_hi;
        } else {
            double n_total = nlte_pair_total_density(nlte, atom, plasma, Z_nl,
                                                      ion_idx_lo, ion_idx_hi, shell);
            double sum = sum_lo + sum_hi;
            if (sum > 0.0 && n_total > 0.0) {
                double scale = n_total / sum;
                for (int i = 0; i < N_fl; i++)
                    nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] *= scale;
            }
        }
    }

    free(A_cm);
    free(b);
}

/* Update tau_sobolev for NLTE lines using NLTE level populations.
 * Floor the result at the nebular tau already in opacity->tau_sobolev:
 * NLTE rate matrices can collapse populations of dominant ions (e.g. Si II)
 * in inner shells where photoion pumping is unbalanced, producing tau values
 * many orders of magnitude below the Saha-Boltzmann nebular estimate.
 * Using max(nlte, nebular) preserves NLTE refinements for ions/levels where
 * NLTE actually increases opacity (e.g. Fe II in outer UV-forming shells)
 * while preventing pathological under-population in the inner photosphere. */
/* Per-Z skip mask, parsed once from LUMINA_NLTE_SKIP_Z (comma list, e.g. "14,16"). */
static int nlte_skip_z[100];
static int nlte_skip_z_init = 0;
static void nlte_skip_z_load(void) {
    if (nlte_skip_z_init) return;
    nlte_skip_z_init = 1;
    const char *e = getenv("LUMINA_NLTE_SKIP_Z");
    if (!e || !*e) return;
    char buf[256]; strncpy(buf, e, sizeof(buf)-1); buf[sizeof(buf)-1]=0;
    char *tok = strtok(buf, ", \t");
    while (tok) {
        int z = atoi(tok);
        if (z > 0 && z < 100) nlte_skip_z[z] = 1;
        tok = strtok(NULL, ", \t");
    }
    printf("  [NLTE] LUMINA_NLTE_SKIP_Z active: ");
    for (int i = 1; i < 100; i++) if (nlte_skip_z[i]) printf("%d ", i);
    printf("(these elements keep nebular tau)\n");
}

static void nlte_update_tau_sobolev(NLTEConfig *nlte, AtomicData *atom,
                                     OpacityState *opacity,
                                     double time_explosion, int n_shells) {
    int n_lines = opacity->n_lines;
    nlte_skip_z_load();

    for (int line = 0; line < n_lines; line++) {
        int ion_idx = nlte->nlte_line_map[line];
        if (ion_idx < 0) continue; /* not an NLTE line */

        int Z     = atom->line_atomic_number[line];
        if (Z > 0 && Z < 100 && nlte_skip_z[Z]) continue; /* keep nebular tau */
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

        double nu_l = C_SPEED_OF_LIGHT / lam_cm;
        double src_prefac = 2.0 * H_PLANCK * nu_l * nu_l * nu_l
                            / (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT);
        for (int s = 0; s < n_shells; s++) {
            double n_lower = nlte->nlte_level_populations[nlte_lo * n_shells + s];
            double n_upper = nlte->nlte_level_populations[nlte_up * n_shells + s];

            /* Stimulated emission correction */
            double stim_corr = 1.0;
            if (n_lower > 0.0 && n_upper > 0.0 && g_lo > 0 && g_up > 0) {
                stim_corr = 1.0 - ((double)g_lo * n_upper) / ((double)g_up * n_lower);
                if (stim_corr < 0.0) stim_corr = 0.0;
            }

            double tau_nlte = SOBOLEV_COEFF * f_lu * lam_cm * time_explosion *
                              n_lower * stim_corr;
            if (tau_nlte < 1e-100) tau_nlte = 1e-100;
            opacity->tau_sobolev[line * n_shells + s] = tau_nlte;

            /* CMF NLTE line source function (paper-method, fluorescence-bearing):
             * S_l = (2hv^3/c^2) / (g_u n_l / (g_l n_u) - 1), from the NLTE level
             * pops. Stored for the CMF formal solver; <=0 left for fallback. */
            double S_l = 0.0;
            if (n_lower > 0.0 && n_upper > 0.0 && g_lo > 0 && g_up > 0) {
                double ratio = ((double)g_up * n_lower) / ((double)g_lo * n_upper);
                double denom = ratio - 1.0;
                if (denom > 1e-30) S_l = src_prefac / denom;
            }
            opacity->line_source_S[line * n_shells + s] = S_l;
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

    /* Super-level mode: precompute the within-SL Boltzmann fractions f_i from
     * the current T_e (energies measured relative to each SL's lowest-E anchor
     * to avoid overflow). f_i is the fraction of a super-level's population that
     * sits in full level i — used to weight bb/bf rates and to redistribute the
     * SL solution back to full levels. Identity SLs (1 FL) trivially get f=1. */
    if (nlte->super_mode) {
        double *Zsl = (double *)malloc(
            (nlte->n_super_total > 0 ? nlte->n_super_total : 1) * sizeof(double));
        for (int s = 0; s < n_shells; s++) {
            double T_e = plasma->T_e[s];
            double kT = K_BOLTZMANN * (T_e > 0.0 ? T_e : 1.0);
            for (int sl = 0; sl < nlte->n_super_total; sl++) Zsl[sl] = 0.0;
            for (int g = 0; g < nlte->n_nlte_levels_total; g++) {
                int gl = nlte->nlte_to_global_level[g];
                int sl = nlte->fl_to_super[g];
                int anchor = nlte->super_anchor_global[sl];
                double E_rel = (atom->level_energy_eV[gl] -
                                atom->level_energy_eV[anchor]) * EV_TO_ERG;
                if (E_rel < 0.0) E_rel = 0.0;
                double w = (double)atom->level_g[gl] * exp(-E_rel / kT);
                if (!isfinite(w) || w < 0.0) w = 0.0;
                nlte->within_sl_frac[(size_t)g * n_shells + s] = w;
                Zsl[sl] += w;
            }
            for (int g = 0; g < nlte->n_nlte_levels_total; g++) {
                int sl = nlte->fl_to_super[g];
                double Z = Zsl[sl];
                size_t idx = (size_t)g * n_shells + s;
                nlte->within_sl_frac[idx] = (Z > 0.0) ?
                    nlte->within_sl_frac[idx] / Z : 1.0;
            }
        }
        free(Zsl);
    }

    /* #281: 16 pairs (last two overlap on slot 29 = O II) for full O triplet. */
    int n_pairs = NLTE_PAIR_COUNT;
    int pairs[][2] = { {0,1}, {2,3}, {4,5}, {6,7}, {8,9}, {10,11},
                       {12,13}, {14,15}, {16,17}, {18,19},
                       {20,21}, {22,23}, {24,25}, {26,27},
                       {28,29}, {29,30} };
    const char *names[] = { "Si", "Ca", "Fe", "S", "Co", "Ni",
                            "C", "Mg", "Ti", "Cr",
                            "Al", "Sc", "V", "Mn",
                            "O(I-II)", "O(II-III)" };

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

            /* C1: Preserve overlapping lo-ion populations across pairs that
             * share a slot (e.g. pair 14={28,29}=O(I-II) and pair 15={29,30}
             * =O(II-III) both touch slot 29 = O II). Without this, pair 15
             * silently overwrites the O II populations pair 14 just set,
             * with a Saha-derived ratio that ignores the O I↔O II coupling.
             * Save before, restore after — pair p's solve sees consistent
             * O II as the lower-boundary population, and the prior pair's
             * answer survives. */
            int lo_overlaps_prior = 0;
            int pair_shares_slot = 0;
            for (int pp = 0; pp < n_pairs; pp++) {
                if (pp == p) continue;
                if (pairs[pp][0] == lo || pairs[pp][1] == lo) {
                    pair_shares_slot = 1;
                    if (pp < p) lo_overlaps_prior = 1;
                }
                if (pairs[pp][0] == hi || pairs[pp][1] == hi)
                    pair_shares_slot = 1;
            }
            double *saved_lo = NULL;
            int saved_lev_s = 0, saved_lev_e = 0;
            if (lo_overlaps_prior) {
                saved_lev_s = nlte->nlte_ion_level_offset[lo];
                saved_lev_e = nlte->nlte_ion_level_offset[lo + 1];
                int n_save = (saved_lev_e - saved_lev_s) * n_shells;
                saved_lo = (double *)malloc((size_t)n_save * sizeof(double));
                memcpy(saved_lo,
                       &nlte->nlte_level_populations[(size_t)saved_lev_s * n_shells],
                       (size_t)n_save * sizeof(double));
            }

            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1)
            #endif
            for (int s = 0; s < n_shells; s++) {
                nlte_solve_ion_shell(nlte, atom, plasma, opacity,
                                     lo, hi, s, time_explosion, gamma_dep,
                                     pair_shares_slot);
            }

            if (saved_lo) {
                int n_save = (saved_lev_e - saved_lev_s) * n_shells;
                memcpy(&nlte->nlte_level_populations[(size_t)saved_lev_s * n_shells],
                       saved_lo, (size_t)n_save * sizeof(double));
                free(saved_lo);
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

    /* Probe-B fix (task #29): feed the NLTE-solved ion stage back into opacity. */
    nlte_writeback_ion_stage(nlte, atom, plasma, opacity, time_explosion,
                             n_shells, pairs, n_pairs);

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

    /* [NLTE-DUMP] Per-shell, per-level population dump for UV diagnostic.
     * Activated by env LUMINA_NLTE_LEVEL_DUMP=1. Counter increments per
     * call so successive iterations land in distinct files. */
    {
        const char *env = getenv("LUMINA_NLTE_LEVEL_DUMP");
        if (env && env[0] == '1') {
            static int dump_counter = 0;
            char path[256];
            snprintf(path, sizeof(path),
                     "nlte_levels_iter%03d.csv", dump_counter++);
            FILE *fp = fopen(path, "w");
            if (!fp) {
                fprintf(stderr, "[NLTE-DUMP] failed to open %s\n", path);
            } else {
                fprintf(fp, "Z,ion,shell,level_idx,global_idx,E_eV,g,n_pop,T_e,T_rad,W,n_ion_total\n");
                for (int ii = 0; ii < nlte->n_nlte_ions; ii++) {
                    int Zv  = nlte->nlte_Z[ii];
                    int ion = nlte->nlte_ion[ii];
                    int lev_s = nlte->nlte_ion_level_offset[ii];
                    int lev_e = nlte->nlte_ion_level_offset[ii + 1];
                    int ip = find_ion_pop_idx(atom, Zv, ion);
                    for (int l = lev_s; l < lev_e; l++) {
                        int gi = nlte->nlte_to_global_level[l];
                        double E_eV = atom->level_energy_eV[gi];
                        int gw = atom->level_g[gi];
                        int local_l = l - lev_s;
                        for (int s = 0; s < n_shells; s++) {
                            double n_pop = nlte->nlte_level_populations[
                                (size_t)l * n_shells + s];
                            double T_e   = plasma->T_e ? plasma->T_e[s] :
                                           plasma->T_e_T_rad_ratio * plasma->T_rad[s];
                            double T_rad = plasma->T_rad[s];
                            double W     = plasma->W[s];
                            double n_ion = (ip >= 0) ?
                                atom->ion_number_density[ip * n_shells + s] : 0.0;
                            fprintf(fp, "%d,%d,%d,%d,%d,%.6f,%d,%.6e,%.2f,%.2f,%.6e,%.6e\n",
                                    Zv, ion, s, local_l, gi, E_eV, gw, n_pop,
                                    T_e, T_rad, W, n_ion);
                        }
                    }
                }
                fclose(fp);
                printf("  [NLTE-DUMP] wrote %s\n", path);
            }
        }
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

    /* C7: build a Z -> mass_amu lookup once instead of approximating A ≈ 2Z.
     * 2Z understates heavy elements (Fe Z=26 → A=52 vs true ~56, Ni Z=28 →
     * 56 vs 58.7) and overstates light ones (He Z=2 → 4 vs 4.0 ≈ ok, but C
     * Z=6 → 12 vs 12.01 ok; lopsided for Si Z=14 → 28 vs 28.09 ok; only
     * heavy/odd-Z and H differ — H Z=1 → 2 vs 1.008 = 2× error). Doppler
     * width v_th ∝ 1/√A, so a 2× mass error gives √2 ≈ 1.4× wrong v_th. */
    double Z_to_amu[100];
    for (int z = 0; z < 100; z++) Z_to_amu[z] = (double)(2 * z); /* fallback */
    for (int e = 0; e < atom->n_elements; e++) {
        int z = atom->element_Z[e];
        if (z > 0 && z < 100) Z_to_amu[z] = atom->element_mass_amu[e];
    }

    for (int s = 0; s < n_shells; s++) {
        double T_rad = plasma->T_rad[s];

        for (int i = 0; i < n_lines; i++) {
            double tau_i = tau_orig[i * n_shells + s];
            if (tau_i < 1e-10) continue; /* skip negligible lines */

            double nu_i = opacity->line_list_nu[i];
            int Z_i = atom->line_atomic_number[i];
            double mass_amu = (Z_i > 0 && Z_i < 100) ? Z_to_amu[Z_i] : (2.0 * Z_i);
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

    /* === Formal-integral ablation knobs (Task #227) === */
    const char *_env_cut  = getenv("LUMINA_FI_TAU_CUTOFF");
    const char *_env_cont = getenv("LUMINA_FI_CONT_OPACITY");
    const char *_env_idil = getenv("LUMINA_FI_INNER_DILUTE");
    double fi_tau_cutoff = _env_cut  ? atof(_env_cut)  : 1.0e-5;
    int    fi_use_cont   = _env_cont ? atoi(_env_cont) : 0;
    double fi_W_inner    = _env_idil ? atof(_env_idil) : 1.0;

    printf("\n=== Formal Integral Spectrum ===\n");
    printf("  Impact parameters: %d, beta_max=%.4f\n", n_impact, beta_max);
    if (fi_tau_cutoff != 1.0e-5)
        printf("  [FI] tau_sob cutoff = %.2e (LUMINA_FI_TAU_CUTOFF)\n", fi_tau_cutoff);
    if (fi_use_cont)
        printf("  [FI] continuum e-scatter opacity ON (LUMINA_FI_CONT_OPACITY=1)\n");
    if (fi_W_inner != 1.0)
        printf("  [FI] inner Planck dilution W=%.3f (LUMINA_FI_INNER_DILUTE)\n", fi_W_inner);

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
            double z_prev = z_max_ejecta; /* observer-side end of current segment */

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
                if (tau_sob < fi_tau_cutoff) continue; /* env-gated cutoff */

                /* Continuum segment from z_prev to z (env-gated).
                 * Adds e-scatter attenuation + thermal continuum emission. */
                if (fi_use_cont && z_prev > z) {
                    double dz = z_prev - z;
                    double z_mid = 0.5 * (z_prev + z);
                    double r_mid = sqrt(p2 + z_mid * z_mid);
                    int shell_mid = -1;
                    for (int s = 0; s < n_shells; s++) {
                        if (r_mid >= geo->r_inner[s] && r_mid < geo->r_outer[s]) {
                            shell_mid = s; break;
                        }
                    }
                    if (shell_mid >= 0) {
                        double chi_cont = opacity->electron_density[shell_mid] * SIGMA_THOMSON;
                        double dtau_c = chi_cont * dz;
                        double S_cont = plasma->W[shell_mid] *
                                        planck_bnu(plasma->T_rad[shell_mid], nu_obs);
                        double oz = (dtau_c > 500.0) ? 1.0 : (1.0 - exp(-dtau_c));
                        I_nu   += S_cont * oz * exp(-tau_acc);
                        tau_acc += dtau_c;
                    }
                }

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
                z_prev = z;
            }

            /* Final continuum segment from z_prev to inner boundary (env-gated) */
            if (fi_use_cont) {
                double z_inner_b = (p < r_phot) ? z_phot : -z_max_ejecta;
                if (z_prev > z_inner_b) {
                    double dz = z_prev - z_inner_b;
                    double z_mid = 0.5 * (z_prev + z_inner_b);
                    double r_mid = sqrt(p2 + z_mid * z_mid);
                    int shell_mid = -1;
                    for (int s = 0; s < n_shells; s++) {
                        if (r_mid >= geo->r_inner[s] && r_mid < geo->r_outer[s]) {
                            shell_mid = s; break;
                        }
                    }
                    if (shell_mid >= 0) {
                        double chi_cont = opacity->electron_density[shell_mid] * SIGMA_THOMSON;
                        double dtau_c = chi_cont * dz;
                        double S_cont = plasma->W[shell_mid] *
                                        planck_bnu(plasma->T_rad[shell_mid], nu_obs);
                        double oz = (dtau_c > 500.0) ? 1.0 : (1.0 - exp(-dtau_c));
                        I_nu   += S_cont * oz * exp(-tau_acc);
                        tau_acc += dtau_c;
                    }
                }
            }

            /* Inner boundary: dilute Planck at T_inner (env-gated W_inner) */
            if (p < r_phot) {
                I_nu += fi_W_inner * planck_bnu(T_inner, nu_obs) * exp(-tau_acc);
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

/* ============================================================ */
/* CMF (comoving-frame) formal solver — paper-method line transfer */
/*                                                                  */
/* Local line absorption coefficient in homologous flow:            */
/*   chi_l(z) = tau_S / (sqrt(pi)*sigma_z) * exp(-((z-z_res)/sigma_z)^2) */
/* with sigma_z = v_Dopp * t_exp and z_res = c*t*(1 - nu_l/nu_obs).  */
/* Exact identity int chi_l dz = tau_S, so a single thin line        */
/* reproduces the Sobolev result; the new physics is line OVERLAP.   */
/* Per cell we deposit the integral-preserving fraction              */
/*   dtau_k = tau_S * 0.5*[erf((z_hi-z_res)/sigma) - erf((z_lo-z_res)/sigma)] */
/* so sub-cell-narrow lines drop their full tau into the nearest cell.*/
/* ============================================================ */
void compute_cmf_formal_spectrum(
    Geometry *geo, PlasmaState *plasma, OpacityState *opacity,
    AtomicData *atom, NLTEConfig *nlte, BFOpacity *bf, double T_inner,
    Spectrum *spec, int n_impact, int n_zstep, double v_turb_cms)
{
    int n_shells = geo->n_shells;
    int n_lines  = opacity->n_lines;
    double t_exp = geo->time_explosion;
    double ct    = C_SPEED_OF_LIGHT * t_exp;
    double r_phot  = geo->r_inner[0];
    double r_outer = geo->r_outer[n_shells - 1];
    double beta_max = r_outer / ct;
    double dp = r_outer / n_impact;

    /* Line element atomic numbers, indexed to match line_list_nu / tau_sobolev
     * (atom->line_* share the global nu-descending ordering of line_list_nu). */
    const int *line_Z = atom->line_atomic_number;

    /* Z -> amu lookup from the element table (fallback 2*Z) */
    double amu_by_Z[120];
    for (int z = 0; z < 120; z++) amu_by_Z[z] = 2.0 * (double)z;
    amu_by_Z[1] = 1.008;
    for (int e = 0; e < atom->n_elements; e++) {
        int z = atom->element_Z[e];
        if (z >= 0 && z < 120) amu_by_Z[z] = atom->element_mass_amu[e];
    }

    /* Ablation knobs (mirror the Sobolev formal integral plus CMF-specific) */
    const char *_env_cut  = getenv("LUMINA_CMF_TAU_CUTOFF");
    const char *_env_nsig = getenv("LUMINA_CMF_NSIGMA");
    const char *_env_idil = getenv("LUMINA_CMF_INNER_DILUTE");
    double cmf_tau_cutoff = _env_cut  ? atof(_env_cut)  : 1.0e-4;
    double cmf_nsigma     = _env_nsig ? atof(_env_nsig) : 4.0;
    double cmf_W_inner    = _env_idil ? atof(_env_idil) : 1.0;

    printf("\n=== CMF Formal Solver (comoving-frame line transfer) ===\n");
    printf("  Impact params: %d  z-cells/ray: %d  beta_max=%.4f\n",
           n_impact, n_zstep, beta_max);
    printf("  v_turb=%.2f km/s  tau cutoff=%.1e  profile half-width=%.1f sigma\n",
           v_turb_cms * 1.0e-5, cmf_tau_cutoff, cmf_nsigma);

    for (int b = 0; b < spec->n_bins; b++) spec->flux[b] = 0.0;

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        /* Thread-private per-ray accumulators */
        double *dtau  = (double *)malloc((size_t)n_zstep * sizeof(double));
        double *sdtau = (double *)malloc((size_t)n_zstep * sizeof(double));

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic, 8)
        #endif
        for (int bin = 0; bin < spec->n_bins; bin++) {
            double lambda_cm = spec->wavelength[bin] * 1.0e-8;
            double nu_obs = C_SPEED_OF_LIGHT / lambda_cm;

            double nu_line_min = nu_obs * (1.0 - beta_max);
            double nu_line_max = nu_obs * (1.0 + beta_max);
            int l_first = bsearch_descending_le(opacity->line_list_nu, n_lines, nu_line_max);
            int l_last  = bsearch_descending_ge(opacity->line_list_nu, n_lines, nu_line_min);

            double L_nu_integral = 0.0;

            for (int ip = 0; ip < n_impact; ip++) {
                double p = dp * (ip + 0.5);
                if (p >= r_outer) continue;
                double p2 = p * p;

                double z_max = sqrt(r_outer * r_outer - p2);
                double z_in  = (p < r_phot) ? sqrt(r_phot * r_phot - p2) : -z_max;
                double span  = z_max - z_in;
                if (span <= 0.0) continue;
                double dz = span / n_zstep;

                for (int k = 0; k < n_zstep; k++) { dtau[k] = 0.0; sdtau[k] = 0.0; }

                /* --- Continuum (e-scatter + bound-free) per cell --- */
                for (int k = 0; k < n_zstep; k++) {
                    double z_mid = z_in + (k + 0.5) * dz;
                    double r = sqrt(p2 + z_mid * z_mid);
                    if (r < r_phot || r > r_outer) continue;
                    int shell = -1;
                    for (int s = 0; s < n_shells; s++)
                        if (r >= geo->r_inner[s] && r < geo->r_outer[s]) { shell = s; break; }
                    if (shell < 0) continue;

                    double nu_cmf = nu_obs * (1.0 - z_mid / ct); /* comoving freq here */

                    double chi_es = opacity->electron_density[shell] * SIGMA_THOMSON;
                    double dtau_es = chi_es * dz;
                    double S_es;
                    if (nlte != NULL && nlte->enabled) {
                        S_es = nlte_get_J_at_nu(nlte, shell, nu_cmf);
                        if (S_es <= 0.0)
                            S_es = plasma->W[shell] * planck_bnu(plasma->T_rad[shell], nu_cmf);
                    } else {
                        S_es = plasma->W[shell] * planck_bnu(plasma->T_rad[shell], nu_cmf);
                    }
                    dtau[k]  += dtau_es;
                    sdtau[k] += S_es * dtau_es;

                    if (bf != NULL && bf->enabled) {
                        double chi_bf = bf_get_chi(bf, shell, nu_cmf);
                        if (chi_bf > 0.0) {
                            double dtau_bf = chi_bf * dz;
                            double S_bf = planck_bnu(plasma->T_e[shell], nu_cmf); /* thermal */
                            dtau[k]  += dtau_bf;
                            sdtau[k] += S_bf * dtau_bf;
                        }
                    }
                }

                /* --- Lines: deposit Gaussian Doppler profiles onto the grid --- */
                for (int l = l_last; l >= l_first; l--) {
                    double nu_l = opacity->line_list_nu[l];
                    double z_res = ct * (1.0 - nu_l / nu_obs);

                    /* Resonance shell sets T_e and the line's Doppler width */
                    double r_res = sqrt(p2 + z_res * z_res);
                    int shell = -1;
                    for (int s = 0; s < n_shells; s++)
                        if (r_res >= geo->r_inner[s] && r_res < geo->r_outer[s]) { shell = s; break; }
                    if (shell < 0) continue;

                    double tau_S = opacity->tau_sobolev[l * n_shells + shell];
                    if (tau_S < cmf_tau_cutoff) continue;

                    int Z = (line_Z != NULL) ? line_Z[l] : 0;
                    double A = (Z > 0 && Z < 120) ? amu_by_Z[Z] : 2.0 * (Z > 0 ? Z : 28);
                    if (A <= 0.0) A = 56.0;
                    double v_th2 = 2.0 * K_BOLTZMANN * plasma->T_e[shell] / (A * AMU);
                    double v_dopp = sqrt(v_th2 + v_turb_cms * v_turb_cms);
                    double sigma_z = v_dopp * t_exp;
                    if (sigma_z <= 0.0) continue;
                    double inv_sigma = 1.0 / sigma_z;

                    /* Cells covered by +/- cmf_nsigma sigma around z_res */
                    double z_lo_w = z_res - cmf_nsigma * sigma_z;
                    double z_hi_w = z_res + cmf_nsigma * sigma_z;
                    if (z_hi_w <= z_in || z_lo_w >= z_max) continue;
                    int k0 = (int)floor((z_lo_w - z_in) / dz);
                    int k1 = (int)ceil ((z_hi_w - z_in) / dz);
                    if (k0 < 0) k0 = 0;
                    if (k1 > n_zstep) k1 = n_zstep;

                    /* Paper-method line source function: the NLTE two-level
                     * source S_l = (2hv^3/c^2)/(g_u n_l/(g_l n_u) - 1) computed
                     * from the NLTE level populations (carries fluorescence /
                     * thermalization). This replaces the old S_l = J (coherent
                     * scatter) which made CMF degenerate with Sobolev/scatter.
                     * Lines outside the NLTE network (S<=0) fall back to the
                     * dilute-LTE source W*B(T_rad) -- still not coherent scatter. */
                    double S_l = (opacity->line_source_S != NULL)
                               ? opacity->line_source_S[l * n_shells + shell] : 0.0;
                    if (S_l <= 0.0)
                        S_l = plasma->W[shell] * planck_bnu(plasma->T_rad[shell], nu_l);

                    double e_lo = erf((z_in + k0 * dz - z_res) * inv_sigma);
                    for (int k = k0; k < k1; k++) {
                        double e_hi = erf((z_in + (k + 1) * dz - z_res) * inv_sigma);
                        double frac = 0.5 * (e_hi - e_lo); /* fraction of profile in cell */
                        e_lo = e_hi;
                        if (frac <= 0.0) continue;
                        double dtau_l = tau_S * frac;
                        dtau[k]  += dtau_l;
                        sdtau[k] += S_l * dtau_l;
                    }
                }

                /* --- Formal solution, inner boundary -> observer (z increasing) --- */
                double I_nu = (p < r_phot)
                            ? cmf_W_inner * planck_bnu(T_inner, nu_obs) : 0.0;
                for (int k = 0; k < n_zstep; k++) {
                    double dt = dtau[k];
                    if (dt <= 0.0) continue;
                    double S_eff = sdtau[k] / dt;
                    double one_minus_exp = (dt > 500.0) ? 1.0 : (1.0 - exp(-dt));
                    I_nu = I_nu * (1.0 - one_minus_exp) + S_eff * one_minus_exp;
                }

                L_nu_integral += I_nu * p * dp;
            }

            double L_nu = 8.0 * M_PI_VAL * M_PI_VAL * L_nu_integral;
            spec->flux[bin] = L_nu * C_SPEED_OF_LIGHT / (lambda_cm * lambda_cm);
        }

        free(dtau);
        free(sdtau);
    }

    printf("  CMF formal spectrum computed.\n");
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

/* ============================================================ */
/* [MA-FATE] Macro-atom packet fate histogram                    */
/* Counts (entry_band, exit_band) for each macro-atom cascade.   */
/* ============================================================ */
static unsigned long long g_ma_fate_hist[MA_FATE_NBANDS * MA_FATE_NBANDS] = {0};

int macro_atom_fate_band_from_nu(double nu_comov) {
    if (nu_comov <= 0.0) return 7;
    double lam_A = (C_SPEED_OF_LIGHT / nu_comov) * 1.0e8;
    if (lam_A >= 1700.0  && lam_A <  3000.0) return 0;  /* UV-blanket  */
    if (lam_A >= 3000.0  && lam_A <  3300.0) return 1;  /* CaIIK-blue  */
    if (lam_A >= 3300.0  && lam_A <  3700.0) return 2;  /* UV-target   */
    if (lam_A >= 3700.0  && lam_A <  4400.0) return 3;  /* blue+fluor  */
    if (lam_A >= 4400.0  && lam_A <  5500.0) return 4;  /* green       */
    if (lam_A >= 5500.0  && lam_A <  7000.0) return 5;  /* red         */
    if (lam_A >= 7000.0  && lam_A < 10000.0) return 6;  /* NIR1        */
    return 7;                                            /* NIR2/far    */
}

void macro_atom_fate_record(double entry_nu_comov, double exit_nu_comov) {
    int eb = macro_atom_fate_band_from_nu(entry_nu_comov);
    int xb = macro_atom_fate_band_from_nu(exit_nu_comov);
    int idx = eb * MA_FATE_NBANDS + xb;
#pragma omp atomic
    g_ma_fate_hist[idx]++;
}

void macro_atom_fate_reset(void) {
    for (int i = 0; i < MA_FATE_NBANDS * MA_FATE_NBANDS; i++) g_ma_fate_hist[i] = 0;
}

void macro_atom_fate_add_counts(const unsigned long long add[MA_FATE_NBANDS * MA_FATE_NBANDS]) {
    for (int i = 0; i < MA_FATE_NBANDS * MA_FATE_NBANDS; i++) g_ma_fate_hist[i] += add[i];
}

void macro_atom_fate_print(const char *label) {
    static const char *band_name[MA_FATE_NBANDS] = {
        "UVblnk", "CaIIKb", "UVtgt ", "fluor ",
        "green ", "red   ", "NIR1  ", "NIR2  "
    };
    unsigned long long row_tot[MA_FATE_NBANDS] = {0};
    unsigned long long col_tot[MA_FATE_NBANDS] = {0};
    unsigned long long grand = 0;
    for (int e = 0; e < MA_FATE_NBANDS; e++) {
        for (int x = 0; x < MA_FATE_NBANDS; x++) {
            unsigned long long n = g_ma_fate_hist[e * MA_FATE_NBANDS + x];
            row_tot[e] += n;
            col_tot[x] += n;
            grand     += n;
        }
    }
    printf("\n[MA-FATE] Macro-atom packet fate histogram (%s)\n",
           label ? label : "");
    printf("  Bands: UVblnk=1700-3000 CaIIKb=3000-3300 UVtgt=3300-3700 fluor=3700-4400\n"
           "         green =4400-5500 red   =5500-7000 NIR1 =7000-10000 NIR2 =>10000\n");
    if (grand == 0) {
        printf("  (no macro-atom interactions recorded)\n");
        return;
    }
    printf("              | exit->");
    for (int x = 0; x < MA_FATE_NBANDS; x++) printf("    %s ", band_name[x]);
    printf("|    row tot\n");
    for (int e = 0; e < MA_FATE_NBANDS; e++) {
        printf("  entry %s |", band_name[e]);
        for (int x = 0; x < MA_FATE_NBANDS; x++) {
            unsigned long long n = g_ma_fate_hist[e * MA_FATE_NBANDS + x];
            double pct = row_tot[e] > 0 ? 100.0 * (double)n / (double)row_tot[e] : 0.0;
            printf(" %8llu(%4.1f%%)", n, pct);
        }
        printf(" | %10llu\n", row_tot[e]);
    }
    printf("  col tot     |       ");
    for (int x = 0; x < MA_FATE_NBANDS; x++) printf(" %14llu", col_tot[x]);
    printf(" | %10llu\n", grand);
    /* Diagnostic: UV-blanket-entry redistribution. The key physics question:
     * does Fe II UV-line absorbed energy emerge at UVtgt+fluor (3300-4400, the
     * Mazzali-Lucy fluorescence peaks observed in HST 2011fe), or does it
     * preferentially leak to NIR via cascade through low-lying levels? */
    unsigned long long uv_row = row_tot[0];
    if (uv_row > 0) {
        double pct_self = 100.0 * (double)g_ma_fate_hist[0 * MA_FATE_NBANDS + 0] / (double)uv_row;
        double pct_uvtgt = 100.0 * (double)g_ma_fate_hist[0 * MA_FATE_NBANDS + 2] / (double)uv_row;
        double pct_fluor = 100.0 * (double)g_ma_fate_hist[0 * MA_FATE_NBANDS + 3] / (double)uv_row;
        double pct_opt   = 100.0 * (double)(g_ma_fate_hist[0 * MA_FATE_NBANDS + 4]
                                          + g_ma_fate_hist[0 * MA_FATE_NBANDS + 5]) / (double)uv_row;
        double pct_nir   = 100.0 * (double)(g_ma_fate_hist[0 * MA_FATE_NBANDS + 6]
                                          + g_ma_fate_hist[0 * MA_FATE_NBANDS + 7]) / (double)uv_row;
        printf("  UVblnk-entry fate: -> UVblnk %.1f%% | UVtgt %.1f%% | fluor %.1f%% |"
               " opt(grn+red) %.1f%% | NIR %.1f%%\n",
               pct_self, pct_uvtgt, pct_fluor, pct_opt, pct_nir);
        printf("  (HST 2011fe physics: UVtgt+fluor should dominate; large NIR fraction\n"
               "   means cascade terminates at low-lying levels instead of emitting at 3300-4400 A.)\n");
    }
}

/* ============================================================ */
/* [H3] Per-(Z, ion, entry_band, exit_band) attribution           */
/* ============================================================ */
const int MA_FATE_Z_LIST[MA_FATE_NZ] = {
    6, 8, 12, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28
};
static const char *MA_FATE_Z_NAME[MA_FATE_NZ] = {
    "C","O","Mg","Al","Si","S","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni"
};
static unsigned long long g_ma_fate_zihist[MA_FATE_ZI_LEN] = {0};

void macro_atom_fate_zi_reset(void) {
    for (int i = 0; i < MA_FATE_ZI_LEN; i++) g_ma_fate_zihist[i] = 0;
}
void macro_atom_fate_zi_add_counts(const unsigned long long add[MA_FATE_ZI_LEN]) {
    for (int i = 0; i < MA_FATE_ZI_LEN; i++) g_ma_fate_zihist[i] += add[i];
}
void macro_atom_fate_zi_dump_csv(const char *path, const char *label) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[MA-FATE-ZI] could not open %s for write\n", path);
        return;
    }
    fprintf(f, "# label=%s\n", label ? label : "");
    fprintf(f, "# bands: 0=UVblnk[1700,3000) 1=CaIIKb[3000,3300) "
               "2=UVtgt[3300,3700) 3=fluor[3700,4400) 4=green[4400,5500) "
               "5=red[5500,7000) 6=NIR1[7000,10000) 7=NIR2[>=10000)\n");
    fprintf(f, "Z,Z_name,ion,entry_band,exit_band,count\n");
    for (int zi = 0; zi < MA_FATE_NZ; zi++) {
        for (int io = 0; io < MA_FATE_NION; io++) {
            for (int eb = 0; eb < MA_FATE_NBANDS; eb++) {
                for (int xb = 0; xb < MA_FATE_NBANDS; xb++) {
                    int idx = ((zi*MA_FATE_NION + io)*MA_FATE_NBANDS + eb)
                              *MA_FATE_NBANDS + xb;
                    unsigned long long n = g_ma_fate_zihist[idx];
                    if (n == 0) continue;
                    fprintf(f, "%d,%s,%d,%d,%d,%llu\n",
                            MA_FATE_Z_LIST[zi], MA_FATE_Z_NAME[zi],
                            io, eb, xb, n);
                }
            }
        }
    }
    fclose(f);
    printf("[MA-FATE-ZI] CSV written to %s\n", path);
}

/* ============================================================ */
/* [MA-CYCLE] Macro-atom internal cycle count histogram          */
/* ma_iter is the number of branching iterations per packet      */
/* before it picks an emission (BB or BF) transition.            */
/* ============================================================ */
static unsigned long long g_ma_cycle_hist[MA_CYCLE_BINS] = {0};

void macro_atom_cycle_record(int n_cycles) {
    int b = n_cycles;
    if (b < 0) b = 0;
    if (b >= MA_CYCLE_BINS) b = MA_CYCLE_BINS - 1;
#pragma omp atomic
    g_ma_cycle_hist[b]++;
}

void macro_atom_cycle_reset(void) {
    for (int i = 0; i < MA_CYCLE_BINS; i++) g_ma_cycle_hist[i] = 0;
}

void macro_atom_cycle_add_counts(const unsigned long long add[MA_CYCLE_BINS]) {
    for (int i = 0; i < MA_CYCLE_BINS; i++) g_ma_cycle_hist[i] += add[i];
}

void macro_atom_cycle_print(const char *label) {
    unsigned long long total = 0;
    for (int i = 0; i < MA_CYCLE_BINS; i++) total += g_ma_cycle_hist[i];
    if (total == 0) {
        printf("\n[MA-CYCLE] %s: no data\n", label);
        return;
    }
    double mean = 0.0;
    unsigned long long cumul = 0;
    int p50 = -1, p90 = -1, p99 = -1;
    for (int i = 0; i < MA_CYCLE_BINS; i++) {
        mean  += (double)i * (double)g_ma_cycle_hist[i];
        cumul += g_ma_cycle_hist[i];
        if (p50 < 0 && cumul * 2 >= total) p50 = i;
        if (p90 < 0 && cumul * 10 >= total * 9) p90 = i;
        if (p99 < 0 && cumul * 100 >= total * 99) p99 = i;
    }
    mean /= (double)total;
    unsigned long long cap_hits = g_ma_cycle_hist[MA_CYCLE_BINS - 1];

    printf("\n[MA-CYCLE] %s\n", label);
    printf("  total=%llu  mean=%.2f  p50=%d  p90=%d  p99=%d  cap@%d=%llu (%.4f%%)\n",
           total, mean, p50, p90, p99, MA_CYCLE_BINS - 1,
           cap_hits, 100.0 * (double)cap_hits / (double)total);
    int bins[] = {0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000};
    int n_bins = (int)(sizeof(bins) / sizeof(bins[0]));
    printf("  cycle range  count           fraction\n");
    for (int b = 0; b < n_bins - 1; b++) {
        unsigned long long sub = 0;
        for (int i = bins[b]; i < bins[b+1]; i++) sub += g_ma_cycle_hist[i];
        printf("  [%4d,%4d)  %14llu  %6.2f%%\n",
               bins[b], bins[b+1], sub,
               100.0 * (double)sub / (double)total);
    }
    printf("  [%4d    ]  %14llu  %6.2f%%\n",
           MA_CYCLE_BINS - 1, cap_hits,
           100.0 * (double)cap_hits / (double)total);
}

#ifdef __cplusplus   /* Phase 6 - Step 9: close extern C guard */
}                    /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */
