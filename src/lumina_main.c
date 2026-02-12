/* lumina_main.c — Phase 5: Main driver
 * Runs LUMINA MC transport with TARDIS reference plasma state.
 * Compares output W, T_rad, spectrum vs TARDIS ground truth. */

#include "lumina.h" /* Phase 5 - Step 1 */
#ifdef _OPENMP
#include <omp.h>    /* Phase 5 - Step 1: OpenMP support */
#endif

/* ============================================================ */
/* Phase 5 - Step 2: Packet initialization (TARDIS style)       */
/* ============================================================ */

/* Phase 5 - Step 2: Initialize packet at inner boundary */
static void initialize_packet(RPacket *pkt, Geometry *geo, MCConfig *config,
                               double packet_energy, RNG *rng) {
    /* Phase 5 - Step 2: Start at inner boundary */
    pkt->r = geo->r_inner[0]; /* Phase 5 - Step 2 */
    /* Phase 5 - Step 2: Isotropic emission from photosphere */
    /* TARDIS: mu = sqrt(random()) for limb-darkened outward emission */
    pkt->mu = sqrt(rng_uniform(rng)); /* Phase 5 - Step 2 */
    pkt->current_shell_id = 0; /* Phase 5 - Step 2 */
    pkt->status = PACKET_IN_PROCESS; /* Phase 5 - Step 2 */
    pkt->next_line_id = 0; /* Phase 5 - Step 2: will be set in single_packet_loop */

    /* Phase 5 - Step 2: Frequency from blackbody at T_inner */
    /* TARDIS samples nu from Planck distribution: */
    /* Use inverse CDF method: sample x from P(x) where x = h*nu/(k*T) */
    /* Simplified: sample from BB using von Neumann rejection */
    double T = config->T_inner; /* Phase 5 - Step 2 */
    double kT_h = K_BOLTZMANN * T / H_PLANCK; /* Phase 5 - Step 2 */

    /* Phase 5 - Step 2: Sample from Planck distribution */
    /* Bjorkman & Wood 2001 (TARDIS method): */
    /* 1) Sample xi0, find l_min: sum(i^-4, i=1..l) >= (pi^4/90)*xi0 */
    /* 2) Sample xi1-xi4, compute x = -ln(xi1*xi2*xi3*xi4) / l_min */
    /* 3) nu = x * kT/h */
    double nu; /* Phase 5 - Step 2 */
    { /* Phase 5 - Step 2: Bjorkman-Wood scope */
        double xi0 = rng_uniform(rng); /* Phase 5 - Step 2 */
        double l_coef = M_PI_VAL * M_PI_VAL * M_PI_VAL * M_PI_VAL / 90.0; /* Phase 5 - Step 2 */
        double target = xi0 * l_coef; /* Phase 5 - Step 2 */
        double cumsum = 0.0; /* Phase 5 - Step 2 */
        double l_min = 1.0; /* Phase 5 - Step 2 */
        for (int l = 1; l <= 1000; l++) { /* Phase 5 - Step 2 */
            double l_inv4 = 1.0 / ((double)l * (double)l * (double)l * (double)l); /* Phase 5 - Step 2 */
            cumsum += l_inv4; /* Phase 5 - Step 2 */
            if (cumsum >= target) { /* Phase 5 - Step 2 */
                l_min = (double)l; /* Phase 5 - Step 2 */
                break; /* Phase 5 - Step 2 */
            }
        }
        double r1 = rng_uniform(rng); /* Phase 5 - Step 2 */
        double r2 = rng_uniform(rng); /* Phase 5 - Step 2 */
        double r3 = rng_uniform(rng); /* Phase 5 - Step 2 */
        double r4 = rng_uniform(rng); /* Phase 5 - Step 2 */
        if (r1 < 1e-300) r1 = 1e-300; /* Phase 5 - Step 2 */
        if (r2 < 1e-300) r2 = 1e-300; /* Phase 5 - Step 2 */
        if (r3 < 1e-300) r3 = 1e-300; /* Phase 5 - Step 2 */
        if (r4 < 1e-300) r4 = 1e-300; /* Phase 5 - Step 2 */
        double x = -log(r1 * r2 * r3 * r4) / l_min; /* Phase 5 - Step 2 */
        nu = x * kT_h; /* Phase 5 - Step 2 */
    }

    pkt->nu = nu; /* Phase 5 - Step 2: comoving frame frequency */
    pkt->energy = packet_energy; /* Phase 5 - Step 2: uniform energy packets */
}

/* ============================================================ */
/* Phase 5 - Step 3: Main simulation loop                       */
/* ============================================================ */

int main(int argc, char *argv[]) {
    setbuf(stdout, NULL); /* Phase 5 - Step 3: unbuffered output */
    printf("============================================================\n"); /* Phase 5 - Step 3 */
    printf("LUMINA-SN v2.0 — TARDIS-Faithful Reimplementation\n"); /* Phase 5 - Step 3 */
    printf("============================================================\n"); /* Phase 5 - Step 3 */

    /* Phase 5 - Step 3: Load TARDIS reference data */
    Geometry geo; /* Phase 5 - Step 3 */
    OpacityState opacity; /* Phase 5 - Step 3 */
    PlasmaState plasma; /* Phase 5 - Step 3 */
    MCConfig config; /* Phase 5 - Step 3 */
    AtomicData atom_data; /* Task #072 */
    memset(&config, 0, sizeof(config)); /* Phase 5 - Step 3 */

    /* Default T_e/T_rad ratio (will be overridden by config.json if present) */
    plasma.T_e_T_rad_ratio = 0.9;

    /* Phase 5 - Step 3: Set defaults matching TARDIS sn2011fe.yml */
    config.enable_full_relativity = false; /* Phase 5 - Step 3 */
    config.disable_line_scattering = false; /* Phase 5 - Step 3 */
    config.line_interaction_type = LINE_MACROATOM; /* Phase 5 - Step 3 */
    config.damping_constant = 0.5; /* Phase 5 - Step 3 */
    config.hold_iterations = 3; /* Phase 5 - Step 3 */

    const char *ref_dir = "data/tardis_reference"; /* Phase 5 - Step 3 */
    if (argc > 1) ref_dir = argv[1]; /* Phase 5 - Step 3 */

    if (load_tardis_reference_data(ref_dir, &geo, &opacity, &plasma, &config) != 0) { /* Phase 5 - Step 3 */
        fprintf(stderr, "Failed to load reference data\n"); /* Phase 5 - Step 3 */
        return 1; /* Phase 5 - Step 3 */
    }

    /* Task #072: Load atomic data for plasma solver */
    if (load_atomic_data(&atom_data, ref_dir, geo.n_shells) != 0) {
        fprintf(stderr, "Failed to load atomic data\n");
        return 1;
    }
    /* Task #072: Initialize n_electron from TARDIS reference */
    plasma.n_electron = (double *)malloc(geo.n_shells * sizeof(double));
    for (int i = 0; i < geo.n_shells; i++)
        plasma.n_electron[i] = opacity.electron_density[i];

    /* Phase 5 - Step 3: Override with command-line packets if given */
    int n_packets = config.n_packets; /* Phase 5 - Step 3 */
    if (argc > 2) n_packets = atoi(argv[2]); /* Phase 5 - Step 3 */
    int n_iterations = config.n_iterations; /* Phase 5 - Step 3 */
    if (argc > 3) n_iterations = atoi(argv[3]); /* Phase 5 - Step 3 */

    /* Spectrum mode: "real" (default), "rotation", "all" */
    int enable_rotation = 0;
    if (argc > 4) {
        if (strcmp(argv[4], "rotation") == 0) enable_rotation = 1;
        else if (strcmp(argv[4], "all") == 0) enable_rotation = 1;
    }

    /* NLTE mode: argv[5] == "nlte" or env LUMINA_NLTE=1 */
    int enable_nlte = 0;
    if (argc > 5 && strcmp(argv[5], "nlte") == 0) enable_nlte = 1;
    if (getenv("LUMINA_NLTE") && atoi(getenv("LUMINA_NLTE")) > 0) enable_nlte = 1;
    config.enable_nlte = enable_nlte;

    /* NLTE start iteration: default 0 (all iters), env LUMINA_NLTE_START_ITER=N */
    int nlte_start_iter = 0;
    if (getenv("LUMINA_NLTE_START_ITER"))
        nlte_start_iter = atoi(getenv("LUMINA_NLTE_START_ITER"));

    /* Dynamic transition probability update: default OFF, enable with LUMINA_DYNAMIC_TRANSPROB=1 */
    int enable_transprob_update = 0;
    if (getenv("LUMINA_DYNAMIC_TRANSPROB"))
        enable_transprob_update = 1;

    printf("\nSimulation parameters:\n"); /* Phase 5 - Step 3 */
    printf("  Packets: %d, Iterations: %d\n", n_packets, n_iterations); /* Phase 5 - Step 3 */
    printf("  Line interaction: MACROATOM\n"); /* Phase 5 - Step 3 */
    printf("  Spectrum mode: %s\n", enable_rotation ? "real + rotation" : "real only");
    if (enable_nlte && nlte_start_iter > 0)
        printf("  NLTE: ENABLED from iter %d (first %d non-NLTE)\n",
               nlte_start_iter + 1, nlte_start_iter);
    else
        printf("  NLTE: %s\n", enable_nlte ? "ENABLED (all iters)" : "disabled");
    printf("  T_inner: %.2f K\n", config.T_inner); /* Phase 5 - Step 3 */
    printf("  Transition probs: %s\n", enable_transprob_update ? "DYNAMIC" : "FROZEN");

    /* Phase 5 - Step 4: Compute shell volumes */
    double *volume = (double *)malloc(geo.n_shells * sizeof(double)); /* Phase 5 - Step 4 */
    for (int i = 0; i < geo.n_shells; i++) { /* Phase 5 - Step 4 */
        volume[i] = (4.0 / 3.0) * M_PI_VAL * /* Phase 5 - Step 4 */
            (geo.r_outer[i] * geo.r_outer[i] * geo.r_outer[i] - /* Phase 5 - Step 4 */
             geo.r_inner[i] * geo.r_inner[i] * geo.r_inner[i]); /* Phase 5 - Step 4 */
    }

    /* Phase 5 - Step 4: Create estimators and spectrum */
    Estimators *est = create_estimators(geo.n_shells, opacity.n_lines); /* Phase 5 - Step 4 */
    Spectrum *spec = create_spectrum(500.0, 20000.0, 2000); /* Phase 5 - Step 4 */

    Spectrum *spec_rot = enable_rotation ? create_spectrum(500.0, 20000.0, 2000) : NULL;

    /* NLTE: Initialize if enabled */
    NLTEConfig nlte;
    memset(&nlte, 0, sizeof(nlte));
    if (enable_nlte) {
        printf("\n--- NLTE Initialization ---\n");
        nlte_init(&nlte, &atom_data, &opacity, geo.n_shells);
    }

    /* Phase 5 - Step 4: Time of simulation (TARDIS: 1 / L_inner) */
    /* TARDIS: L_inner = 4 * pi * sigma_sb * r_inner^2 * T_inner^4 */
    double L_inner = 4.0 * M_PI_VAL * geo.r_inner[0] * geo.r_inner[0] * /* Phase 5 - Step 4 */
                     SIGMA_SB * pow(config.T_inner, 4); /* Phase 5 - Step 4 */
    /* Phase 5 - Step 4: TARDIS: time_of_simulation = 1.0 / L_inner */
    double time_simulation = 1.0 / L_inner; /* Phase 5 - Step 4 */
    /* Phase 5 - Step 4: packet_energy = 1.0 (unit energy) */
    /* TARDIS uses E_packet = 1/n_packets in internal units */
    double packet_energy = 1.0 / (double)n_packets; /* Phase 5 - Step 4 */

    printf("  L_inner: %.6e erg/s\n", L_inner); /* Phase 5 - Step 4 */
    printf("  time_simulation: %.6e s\n", time_simulation); /* Phase 5 - Step 4 */
    printf("  Packet energy: %.6e (internal units)\n", packet_energy); /* Phase 5 - Step 4 */

    /* ============================================================ */
    /* Phase 5 - Step 5: Iteration loop                             */
    /* ============================================================ */

    for (int iter = 0; iter < n_iterations; iter++) { /* Phase 5 - Step 5 */
        printf("\n--- Iteration %d/%d ---\n", iter + 1, n_iterations); /* Phase 5 - Step 5 */

        /* Phase 5 - Step 5: Reset estimators */
        reset_estimators(est); /* Phase 5 - Step 5 */
        reset_spectrum(spec); /* Phase 5 - Step 5 */
        if (spec_rot) reset_spectrum(spec_rot);
        if (enable_nlte)
            memset(nlte.j_nu_estimator, 0,
                   (size_t)geo.n_shells * nlte.n_freq_bins * sizeof(double));

        /* Phase 5 - Step 5: Recompute L_inner and time_simulation */
        L_inner = 4.0 * M_PI_VAL * geo.r_inner[0] * geo.r_inner[0] * /* Phase 5 - Step 5 */
                  SIGMA_SB * pow(config.T_inner, 4); /* Phase 5 - Step 5 */
        time_simulation = 1.0 / L_inner; /* Phase 5 - Step 5 */
        packet_energy = 1.0 / (double)n_packets; /* Phase 5 - Step 5 */

        /* Phase 5 - Step 5: Transport all packets (OpenMP-ready) */
        /* Store escaped packet data for spectrum binning after parallel section */
        double *escaped_nu = (double *)malloc(n_packets * sizeof(double)); /* Phase 5 - Step 5 */
        double *escaped_energy = (double *)malloc(n_packets * sizeof(double)); /* Phase 5 - Step 5 */
        int *escaped_flag = (int *)calloc(n_packets, sizeof(int)); /* Phase 5 - Step 5 */
        double *escaped_r = enable_rotation ? (double *)malloc(n_packets * sizeof(double)) : NULL;
        double *escaped_mu = enable_rotation ? (double *)malloc(n_packets * sizeof(double)) : NULL;
        int n_escaped = 0; /* Phase 5 - Step 5 */
        int n_reabsorbed = 0; /* Phase 5 - Step 5 */

        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        { /* Phase 5 - Step 5: thread-parallel block */
            int tid = 0; /* Phase 5 - Step 5 */
            #ifdef _OPENMP
            tid = omp_get_thread_num(); /* Phase 5 - Step 5 */
            #endif
            RNG rng; /* Phase 5 - Step 5 */
            rng_init(&rng, config.seed + (uint64_t)iter * 1000 + (uint64_t)tid); /* Phase 5 - Step 5 */

            /* Phase 5 - Step 5: Per-thread local estimators (lightweight: no j_blue) */
            Estimators *local_est = create_estimators(geo.n_shells, 0); /* Phase 5 - Step 5 */
            /* NLTE: attach J_nu histogram to thread-local estimators */
            if (enable_nlte) {
                local_est->nlte_n_freq_bins = nlte.n_freq_bins;
                local_est->nlte_nu_min = nlte.nu_min;
                local_est->nlte_d_log_nu = nlte.d_log_nu;
                local_est->j_nu_estimator = (double *)calloc(
                    (size_t)geo.n_shells * nlte.n_freq_bins, sizeof(double));
            }
            int local_escaped = 0, local_reabsorbed = 0; /* Phase 5 - Step 5 */

            #ifdef _OPENMP
            #pragma omp for schedule(dynamic, 64)
            #endif
            for (int p = 0; p < n_packets; p++) { /* Phase 5 - Step 5 */
                RPacket pkt; /* Phase 5 - Step 5 */
                pkt.index = p; /* Phase 5 - Step 5 */
                initialize_packet(&pkt, &geo, &config, packet_energy, &rng); /* Phase 5 - Step 5 */

                single_packet_loop(&pkt, &geo, &opacity, local_est, &config, &rng); /* Phase 5 - Step 5 */

                /* Phase 5 - Step 5: Store results (per-packet, no race) */
                if (pkt.status == PACKET_EMITTED) { /* Phase 5 - Step 5 */
                    local_escaped++; /* Phase 5 - Step 5 */
                    escaped_flag[p] = 1; /* Phase 5 - Step 5 */
                    escaped_nu[p] = pkt.nu; /* Phase 5 - Step 5 */
                    escaped_energy[p] = pkt.energy; /* Phase 5 - Step 5 */
                    if (enable_rotation) {
                        escaped_r[p] = pkt.r;
                        escaped_mu[p] = pkt.mu;
                    }
                } else if (pkt.status == PACKET_REABSORBED) { /* Phase 5 - Step 5 */
                    local_reabsorbed++; /* Phase 5 - Step 5 */
                }

                /* Phase 5 - Step 5: Progress report (thread 0 only) */
                if (tid == 0 && (p + 1) % (n_packets / 10 > 0 ? n_packets / 10 : 1) == 0) { /* Phase 5 - Step 5 */
                    printf("  Packets: ~%d/%d\r", p + 1, n_packets); /* Phase 5 - Step 5 */
                    fflush(stdout); /* Phase 5 - Step 5 */
                }
            }

            /* Phase 5 - Step 5: Reduce per-thread estimators into global */
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            { /* Phase 5 - Step 5: reduction block */
                for (int i = 0; i < geo.n_shells; i++) { /* Phase 5 - Step 5 */
                    est->j_estimator[i] += local_est->j_estimator[i]; /* Phase 5 - Step 5 */
                    est->nu_bar_estimator[i] += local_est->nu_bar_estimator[i]; /* Phase 5 - Step 5 */
                }
                /* NLTE: reduce J_nu histograms */
                if (enable_nlte && local_est->j_nu_estimator) {
                    size_t j_nu_size = (size_t)geo.n_shells * nlte.n_freq_bins;
                    for (size_t i = 0; i < j_nu_size; i++)
                        nlte.j_nu_estimator[i] += local_est->j_nu_estimator[i];
                }
                /* Phase 5 - Step 5: j_blue/Edotlu not tracked per-thread (too large) */
                n_escaped += local_escaped; /* Phase 5 - Step 5 */
                n_reabsorbed += local_reabsorbed; /* Phase 5 - Step 5 */
            }
            if (local_est->j_nu_estimator) free(local_est->j_nu_estimator);
            local_est->j_nu_estimator = NULL;
            free_estimators(local_est); /* Phase 5 - Step 5 */
        } /* Phase 5 - Step 5: end parallel block */

        /* Phase 5 - Step 5b: Spectrum binning + L_emitted from actual packets */
        double L_emitted = 0.0;
        double rot_weight_sum = 0.0;
        int rot_count = 0;
        for (int p = 0; p < n_packets; p++) { /* Phase 5 - Step 5b */
            if (escaped_flag[p]) { /* Phase 5 - Step 5b */
                bin_escaped_packet(spec, escaped_nu[p], escaped_energy[p] * L_inner);
                L_emitted += escaped_energy[p] * L_inner;
                if (enable_rotation) {
                    double beta = escaped_r[p] / (C_SPEED_OF_LIGHT * geo.time_explosion);
                    double D_pkt = 1.0 - beta * escaped_mu[p];
                    double D_obs = 1.0 - beta * 1.0; /* mu_obs = 1 (face-on) */
                    double w = (D_obs / D_pkt) * (D_obs / D_pkt);
                    bin_escaped_packet(spec_rot, escaped_nu[p],
                                        escaped_energy[p] * L_inner * w);
                    rot_weight_sum += w;
                    rot_count++;
                }
            }
        }
        free(escaped_nu); /* Phase 5 - Step 5b */
        free(escaped_energy); /* Phase 5 - Step 5b */
        free(escaped_flag); /* Phase 5 - Step 5b */
        free(escaped_r);
        free(escaped_mu);

        double escape_fraction = (double)n_escaped / n_packets; /* Phase 5 - Step 5 */
        printf("  Packets: %d/%d done. Escaped: %d (%.2f%%), Reabsorbed: %d (%.2f%%)\n", /* Phase 5 - Step 5 */
               n_packets, n_packets, /* Phase 5 - Step 5 */
               n_escaped, 100.0 * escape_fraction, /* Phase 5 - Step 5 */
               n_reabsorbed, 100.0 * n_reabsorbed / n_packets); /* Phase 5 - Step 5 */

        /* Phase 5 - Step 6: Solve radiation field */
        solve_radiation_field(est, geo.time_explosion, time_simulation, volume, /* Phase 5 - Step 6 */
                               &opacity, &plasma, config.damping_constant); /* Phase 5 - Step 6 */

        /* Task #072: Recompute tau_sobolev from updated W, T_rad */
        if (iter > 0) {
            compute_plasma_state(&atom_data, &plasma, &opacity, geo.time_explosion);

            /* NLTE: solve rate equations and update tau for NLTE lines */
            if (enable_nlte && iter >= nlte_start_iter) {
                nlte_normalize_j_nu(&nlte, time_simulation, volume, geo.n_shells);
                nlte_solve_all(&nlte, &atom_data, &plasma, &opacity,
                               geo.time_explosion, geo.n_shells);
            }

            /* Dynamic transition probability recomputation */
            if (enable_transprob_update && iter >= config.hold_iterations) {
                compute_transition_probabilities(&atom_data, &plasma, &opacity,
                    config.damping_constant,
                    (iter > config.hold_iterations) ? 1 : 0);
            }
        }

        /* Task #072: Validation mode — compute plasma state with reference values on iter 0 */
        if (iter == 0 && getenv("LUMINA_VALIDATE_PLASMA")) {
            printf("  [Validation] Computing plasma state with reference W, T_rad...\n");
            /* Save current W, T_rad (reference values) */
            double *save_W = (double *)malloc(geo.n_shells * sizeof(double));
            double *save_T = (double *)malloc(geo.n_shells * sizeof(double));
            memcpy(save_W, plasma.W, geo.n_shells * sizeof(double));
            memcpy(save_T, plasma.T_rad, geo.n_shells * sizeof(double));
            /* Restore reference for plasma computation */
            char ref_path[512];
            snprintf(ref_path, sizeof(ref_path), "%s/plasma_state.csv", ref_dir);
            FILE *ref_f = fopen(ref_path, "r");
            if (ref_f) {
                char buf2[1024];
                fgets(buf2, sizeof(buf2), ref_f); /* skip header */
                int si = 0;
                while (fgets(buf2, sizeof(buf2), ref_f) && si < geo.n_shells) {
                    int sid;
                    double w, t;
                    sscanf(buf2, "%d,%lf,%lf", &sid, &w, &t);
                    plasma.W[si] = w;
                    plasma.T_rad[si] = t;
                    si++;
                }
                fclose(ref_f);
            }
            /* Restore reference n_e */
            for (int i = 0; i < geo.n_shells; i++)
                plasma.n_electron[i] = opacity.electron_density[i];
            compute_plasma_state(&atom_data, &plasma, &opacity, geo.time_explosion);
            /* Write validation tau to file */
            FILE *vf = fopen("lumina_tau_validation.csv", "w");
            if (vf) {
                fprintf(vf, "line,tau_lumina_s0,tau_tardis_s0\n");
                /* Load TARDIS tau for comparison (already in opacity.tau_sobolev before overwrite) */
                /* Actually tau_sobolev was already overwritten. Load from file */
                int tr, tc;
                char tau_path[512];
                snprintf(tau_path, sizeof(tau_path), "%s/tau_sobolev.npy", ref_dir);
                /* Can't easily reload npy in C. Print what we have */
                fprintf(vf, "# LUMINA tau computed with TARDIS reference W/T_rad\n");
                fprintf(vf, "# Compare with tardis_reference/tau_sobolev.npy\n");
                for (int l = 0; l < opacity.n_lines; l++) {
                    fprintf(vf, "%d,%.10e\n", l, opacity.tau_sobolev[l * geo.n_shells + 0]);
                }
                fclose(vf);
                printf("  [Validation] tau written to lumina_tau_validation.csv\n");
                (void)tr; (void)tc; (void)tau_path;
            }
            /* Restore the LUMINA-computed values for continued iteration */
            memcpy(plasma.W, save_W, geo.n_shells * sizeof(double));
            memcpy(plasma.T_rad, save_T, geo.n_shells * sizeof(double));
            free(save_W);
            free(save_T);
        }

        /* Phase 5 - Step 6: Print plasma state comparison */
        printf("  Shell  W_LUMINA   T_rad_LUM  nubar/j\n"); /* Phase 5 - Step 6 */
        for (int i = 0; i < geo.n_shells; i += 5) { /* Phase 5 - Step 6: print every 5th */
            double ratio = est->nu_bar_estimator[i] / est->j_estimator[i]; /* Phase 5 - Step 6 */
            printf("  %3d    %.6f   %.2f K   %.4e\n", /* Phase 5 - Step 6 */
                   i, plasma.W[i], plasma.T_rad[i], ratio); /* Phase 5 - Step 6 */
        }

        /* Phase 5 - Step 7: Update T_inner (after hold iterations) */
        if (iter >= config.hold_iterations) { /* Phase 5 - Step 7 */
            double old_T = config.T_inner; /* Phase 5 - Step 7 */
            update_t_inner(&config, L_emitted); /* Phase 5 - Step 7 */
            printf("  T_inner: %.2f K -> %.2f K (L_em=%.3e, L_req=%.3e)\n",
                   old_T, config.T_inner, L_emitted, config.luminosity_requested);
        } else { /* Phase 5 - Step 7 */
            printf("  T_inner: %.2f K (hold iteration %d/%d)\n", /* Phase 5 - Step 7 */
                   config.T_inner, iter + 1, config.hold_iterations); /* Phase 5 - Step 7 */
        }
    }

    /* ============================================================ */
    /* Phase 5 - Step 8: Output results                             */
    /* ============================================================ */

    printf("\n============================================================\n"); /* Phase 5 - Step 8 */
    printf("Final Results\n"); /* Phase 5 - Step 8 */
    printf("============================================================\n"); /* Phase 5 - Step 8 */

    /* Phase 5 - Step 8: Load TARDIS reference for comparison */
    char path[512]; /* Phase 5 - Step 8 */
    snprintf(path, sizeof(path), "%s/plasma_state.csv", ref_dir); /* Phase 5 - Step 8 */

    /* Phase 5 - Step 8: Read TARDIS W and T_rad */
    FILE *ref_fp = fopen(path, "r"); /* Phase 5 - Step 8 */
    double tardis_W[30], tardis_T_rad[30]; /* Phase 5 - Step 8 */
    if (ref_fp) { /* Phase 5 - Step 8 */
        char buf[1024]; /* Phase 5 - Step 8 */
        fgets(buf, sizeof(buf), ref_fp); /* Phase 5 - Step 8: skip header */
        int i = 0; /* Phase 5 - Step 8 */
        while (fgets(buf, sizeof(buf), ref_fp) && i < 30) { /* Phase 5 - Step 8 */
            int sid; /* Phase 5 - Step 8 */
            sscanf(buf, "%d,%lf,%lf", &sid, &tardis_W[i], &tardis_T_rad[i]); /* Phase 5 - Step 8 */
            i++; /* Phase 5 - Step 8 */
        }
        fclose(ref_fp); /* Phase 5 - Step 8 */

        printf("\nShell  W_LUMINA   W_TARDIS   W_err%%   T_rad_LUM  T_rad_TAR  T_err%%\n"); /* Phase 5 - Step 8 */
        printf("-----  --------   --------   ------   ---------  ---------  ------\n"); /* Phase 5 - Step 8 */
        for (int i = 0; i < geo.n_shells; i++) { /* Phase 5 - Step 8 */
            double w_err = (plasma.W[i] - tardis_W[i]) / tardis_W[i] * 100.0; /* Phase 5 - Step 8 */
            double t_err = (plasma.T_rad[i] - tardis_T_rad[i]) / tardis_T_rad[i] * 100.0; /* Phase 5 - Step 8 */
            printf("  %3d  %8.6f   %8.6f   %+6.1f   %9.2f  %9.2f  %+6.1f\n", /* Phase 5 - Step 8 */
                   i, plasma.W[i], tardis_W[i], w_err, /* Phase 5 - Step 8 */
                   plasma.T_rad[i], tardis_T_rad[i], t_err); /* Phase 5 - Step 8 */
        }

        /* Phase 5 - Step 8: Compute mean absolute errors */
        double sum_w_err = 0.0, sum_t_err = 0.0; /* Phase 5 - Step 8 */
        for (int i = 0; i < geo.n_shells; i++) { /* Phase 5 - Step 8 */
            sum_w_err += fabs((plasma.W[i] - tardis_W[i]) / tardis_W[i]); /* Phase 5 - Step 8 */
            sum_t_err += fabs((plasma.T_rad[i] - tardis_T_rad[i]) / tardis_T_rad[i]); /* Phase 5 - Step 8 */
        }
        printf("\nMean |W error|: %.2f%%\n", sum_w_err / geo.n_shells * 100.0); /* Phase 5 - Step 8 */
        printf("Mean |T_rad error|: %.2f%%\n", sum_t_err / geo.n_shells * 100.0); /* Phase 5 - Step 8 */
        printf("T_inner final: %.2f K (TARDIS: 10521.52 K, err: %.2f%%)\n", /* Phase 5 - Step 8 */
               config.T_inner, /* Phase 5 - Step 8 */
               (config.T_inner - 10521.52) / 10521.52 * 100.0); /* Phase 5 - Step 8 */
    }

    /* Phase 5 - Step 9: Write spectrum to CSV */
    const char *output_file = "lumina_spectrum.csv"; /* Phase 5 - Step 9 */
    FILE *out = fopen(output_file, "w"); /* Phase 5 - Step 9 */
    if (out) { /* Phase 5 - Step 9 */
        fprintf(out, "wavelength_angstrom,flux\n"); /* Phase 5 - Step 9 */
        for (int i = 0; i < spec->n_bins; i++) { /* Phase 5 - Step 9 */
            fprintf(out, "%.6f,%.6e\n", spec->wavelength[i], spec->flux[i]); /* Phase 5 - Step 9 */
        }
        fclose(out); /* Phase 5 - Step 9 */
        printf("\nSpectrum written to %s\n", output_file); /* Phase 5 - Step 9 */
    }

    /* Write rotation spectrum */
    if (spec_rot) {
        FILE *rf = fopen("lumina_spectrum_rotation.csv", "w");
        if (rf) {
            fprintf(rf, "wavelength_angstrom,flux\n");
            for (int i = 0; i < spec_rot->n_bins; i++) {
                fprintf(rf, "%.6f,%.6e\n", spec_rot->wavelength[i], spec_rot->flux[i]);
            }
            fclose(rf);
            printf("Rotation spectrum written to lumina_spectrum_rotation.csv\n");
        }
    }

    /* Phase 5 - Step 9b: Write final plasma state */
    out = fopen("lumina_plasma_state.csv", "w"); /* Phase 5 - Step 9b */
    if (out) { /* Phase 5 - Step 9b */
        fprintf(out, "shell_id,W,T_rad\n"); /* Phase 5 - Step 9b */
        for (int i = 0; i < geo.n_shells; i++) { /* Phase 5 - Step 9b */
            fprintf(out, "%d,%.10f,%.6f\n", i, plasma.W[i], plasma.T_rad[i]); /* Phase 5 - Step 9b */
        }
        fclose(out); /* Phase 5 - Step 9b */
        printf("Plasma state written to lumina_plasma_state.csv\n"); /* Phase 5 - Step 9b */
    }

    /* Phase 5 - Step 10: Cleanup */
    free_geometry(&geo); /* Phase 5 - Step 10 */
    free_opacity_state(&opacity); /* Phase 5 - Step 10 */
    free_plasma_state(&plasma); /* Phase 5 - Step 10 */
    free_estimators(est); /* Phase 5 - Step 10 */
    free_spectrum(spec); /* Phase 5 - Step 10 */
    if (spec_rot) free_spectrum(spec_rot);
    free(volume); /* Phase 5 - Step 10 */
    free_atomic_data(&atom_data); /* Task #072 */
    if (enable_nlte) nlte_free(&nlte);

    printf("\nDone.\n"); /* Phase 5 - Step 10 */
    return 0; /* Phase 5 - Step 10 */
}
