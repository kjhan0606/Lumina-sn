/* lumina_main.c — Phase 5: Main driver
 * Runs LUMINA MC transport with TARDIS reference plasma state.
 * Compares output W, T_rad, spectrum vs TARDIS ground truth. */

#include "lumina.h"
#include "line_jbar.h" /* Phase 5 - Step 1 */
#include "lumina_cmfgen.h"  /* pure-CMFGEN parallel radiation path */
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
    /* A6: enforce C locale so "." is the decimal separator in sscanf/atof
     * regardless of LANG (ko_KR.UTF-8 etc. would parse "1.5" as 1). */
    setlocale(LC_NUMERIC, "C");
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
    /* ★2026-08-07: plasma 는 자동 변수인데 영초기화되지 않았고, 개별 필드만 대입됐다.
     * 그래서 plasma.te_publication (A2-10 소유 구조체)이 쓰레기값이었다 —
     * 첫 radeq 의 `old=*pub; *pub=c; a210_publication_free(&old)` 가
     * **쓰레기 포인터를 해제**한다.  영초기화하면 free(NULL) 이므로 무해하다. */
    memset(&plasma, 0, sizeof(plasma));
    /* ★2026-08-07: OpacityState 도 같은 부류였다.  GPU 쪽(lumina_cuda.cu)에는 이미
     * memset 이 있고 그 주석이 SIGSEGV 사고를 기록해 두었는데 **CPU 쪽에만 없었다**.
     * 로더가 필드별로 채우므로 새로 늘어난 필드(발행 세대 등)가 stack garbage 로 남는다. */
    memset(&opacity, 0, sizeof(opacity));

    /* Phase 5 - Step 3: Set defaults matching TARDIS sn2011fe.yml */
    config.enable_full_relativity = false; /* Phase 5 - Step 3 */
    config.disable_line_scattering = false; /* Phase 5 - Step 3 */
    config.line_interaction_type = LINE_MACROATOM; /* Phase 5 - Step 3 */
    {
        const char *li_env = getenv("LUMINA_LINE_INTERACTION");
        if (li_env) {
            if      (strcmp(li_env, "scatter")    == 0 || strcmp(li_env, "0") == 0) config.line_interaction_type = LINE_SCATTER;
            else if (strcmp(li_env, "downbranch") == 0 || strcmp(li_env, "1") == 0) config.line_interaction_type = LINE_DOWNBRANCH;
            else if (strcmp(li_env, "macroatom")  == 0 || strcmp(li_env, "macro") == 0 || strcmp(li_env, "2") == 0) config.line_interaction_type = LINE_MACROATOM;
            else fprintf(stderr, "[WARN] unknown LUMINA_LINE_INTERACTION=%s, keeping macroatom\n", li_env);
        }
    }
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
    /* Task #38: Optional pre-baked CMFGEN sigma_bf grid.
     * LUMINA_CMFGEN_SIGMA_BF semantics:
     *   unset / "1" / "on" / "yes" → load default path (data/atomic/cmfgen_sigma_bf.bin)
     *   "0" / "off" / "no"          → skip load (Kramers fallback)
     *   anything containing '/'     → explicit path override */
    {
        const char *cmf_env = getenv("LUMINA_CMFGEN_SIGMA_BF");
        const char *cmf_path = "data/atomic/cmfgen_sigma_bf.bin";
        int cmf_enable = 1;
        if (cmf_env) {
            if (!strcmp(cmf_env, "0") || !strcmp(cmf_env, "off") || !strcmp(cmf_env, "no"))
                cmf_enable = 0;
            else if (strchr(cmf_env, '/'))
                cmf_path = cmf_env;
        }
        if (cmf_enable) load_cmfgen_sigma_bf(&atom_data, cmf_path);
    }
    /* Top-stage continuum anchor: inject synthetic IV ground levels (gated,
     * default off). After cmfgen_sigma_bf load, before nlte_init (mirrors GPU). */
    inject_topstage_continuum_levels(&atom_data, &opacity);
    /* Task #072: Initialize n_electron from TARDIS reference */
    plasma.n_electron = (double *)malloc(geo.n_shells * sizeof(double));
    for (int i = 0; i < geo.n_shells; i++)
        plasma.n_electron[i] = opacity.electron_density[i];

    /* P6: Initialize per-shell electron temperature */
    plasma.T_e = (double *)malloc(geo.n_shells * sizeof(double));
    for (int i = 0; i < geo.n_shells; ++i)
        plasma.T_e[i] = opacity.t_electrons[i];
    /* This is an unqualified generation-zero material seed. Only the A2-10
     * radiative-equilibrium supplier below may mint a production generation. */
    plasma.T_e_generation = 0;

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
    /* A3: previously presence-only check → "=0" enabled the feature opposite of intent. */
    int enable_transprob_update = 0;
    if (getenv("LUMINA_DYNAMIC_TRANSPROB") &&
        atoi(getenv("LUMINA_DYNAMIC_TRANSPROB")) > 0)
        enable_transprob_update = 1;

    /* Fe scatter = macro-atom.  [스크랩 2026-08-07] LUMINA_FE_SCATTER (attic/knobs) */
    config.fe_scatter_mode = 0;
    config.line_atomic_number = atom_data.line_atomic_number;
    config.line_ion_number = atom_data.line_ion_number;

    /* Gamma-ray deposition: LUMINA_GAMMA_DEP=1 */
    int gamma_dep_enabled = 0;
    if (getenv("LUMINA_GAMMA_DEP") && atoi(getenv("LUMINA_GAMMA_DEP")) > 0)
        gamma_dep_enabled = 1;

    /* [스크랩 2026-08-07] LUMINA_OVERLAP_CORR — 선중첩 보정은 꺼진 채였다 (attic/knobs) */

    /* Bound-free opacity: LUMINA_BF_OPACITY=1 */
    int bf_opacity_enabled = (getenv("LUMINA_BF_OPACITY") &&
                               atoi(getenv("LUMINA_BF_OPACITY")) > 0);

    /* [스크랩 2026-08-07] LUMINA_SELF_CONSISTENT_TE — radeq 와 OR 로 묶인 구 경로 (attic/knobs) */
    /* Task #20: real radiative-equilibrium T_e: LUMINA_RADEQ_TE=1 */
    int radeq_te = (getenv("LUMINA_RADEQ_TE") &&
                     atoi(getenv("LUMINA_RADEQ_TE")) > 0);

    printf("\nSimulation parameters:\n"); /* Phase 5 - Step 3 */
    printf("  Packets: %d [source=%s], Iterations: %d [source=%s]\n",
           n_packets, argc > 2 ? "argv[2]" : "config.json:n_packets",
           n_iterations, argc > 3 ? "argv[3]" : "config.json:n_iterations"); /* CONFIG-PREC */
    printf("  Line interaction: %s\n",
        config.line_interaction_type == LINE_SCATTER    ? "SCATTER" :
        config.line_interaction_type == LINE_DOWNBRANCH ? "DOWNBRANCH" : "MACROATOM");
    printf("  Spectrum mode: %s\n", enable_rotation ? "real + rotation" : "real only");
    if (enable_nlte && nlte_start_iter > 0)
        printf("  NLTE: ENABLED from iter %d (first %d non-NLTE)\n",
               nlte_start_iter + 1, nlte_start_iter);
    else
        printf("  NLTE: %s\n", enable_nlte ? "ENABLED (all iters)" : "disabled");
    printf("  T_inner: %.2f K (resolved and logged by CONFIG-PREC loader)\n",
           config.T_inner); /* Phase 5 - Step 3 */
    printf("  Transition probs: %s\n", enable_transprob_update ? "DYNAMIC" : "FROZEN");
    printf("  Fe scatter: %s\n", config.fe_scatter_mode == 2 ? "ALL Fe TWO-LEVEL" :
                                  config.fe_scatter_mode == 1 ? "Fe II TWO-LEVEL" : "MACRO-ATOM");
    printf("  Gamma-ray deposition: %s\n", gamma_dep_enabled ? "ENABLED" : "disabled");
    printf("  BF+FF opacity: %s\n", bf_opacity_enabled ? "ENABLED" : "disabled");
    if (radeq_te)
        printf("  Self-consistent T_e: ENABLED (full radiative-equilibrium balance)\n");
    else
        printf("  Self-consistent T_e: disabled (generation-zero material seed)\n");

    /* [스크랩 2026-08-07] LUMINA_TIME_EXPLOSION 다중 epoch 재척도 (attic/knobs) */

    /* K-FRESH: tau is solver-owned.  The deck NPY is only an epoch-validated
     * seed and must be overwritten before transport or pure-CMFGEN consumes it. */
    /* 안 B(user 판정 2026-08-07): 아래 K-FRESH 가 플라즈마를 풀려면 **발행된 T_e** 가
     * 있어야 하는데, 이 지점엔 복사장이 없어 radeq 가 돌 수 없다.  덱 seed T_e 를
     * 1세대로 발행해 고리를 끊는다 — 첫 상태는 seed 온도의 LTE(CMFGEN·ARTIS 방식).
     * radeq 발행이 아니며 A2-10 대장에 seed 로 계수된다. */
    if (lumina_publish_seed_te(&plasma,
            "deck seed T_e — bootstrap before first transport (CPU)") != 0)
        return EXIT_FAILURE;

    /* L1-1: 반복 0 에는 복사장이 없어 rate-SE 가 원리적으로 불가능하다.  그 한 번만
     * seed-T_e LTE(Saha)로 물질을 공급하는 창을 연다.  창은 **바로 아래에서 닫으므로**
     * 반복 >=1 은 여전히 fail-closed 다(사전등록 게이트 G3). */
    if (lumina_bootstrap_window_open("iteration-0 material (CPU)") != 0)
        return EXIT_FAILURE;
    int _kfresh_rc = lumina_prepare_solver_owned_tau(&atom_data, &plasma, &opacity,
            geo.time_explosion, "CPU transport/CMFGEN");
    lumina_bootstrap_window_close();
    if (_kfresh_rc != 0)
        return EXIT_FAILURE;

    /* Phase 5 - Step 4: Compute shell volumes */
    double *volume = (double *)malloc(geo.n_shells * sizeof(double)); /* Phase 5 - Step 4 */
    for (int i = 0; i < geo.n_shells; i++) { /* Phase 5 - Step 4 */
        volume[i] = (4.0 / 3.0) * M_PI_VAL * /* Phase 5 - Step 4 */
            (geo.r_outer[i] * geo.r_outer[i] * geo.r_outer[i] - /* Phase 5 - Step 4 */
             geo.r_inner[i] * geo.r_inner[i] * geo.r_inner[i]); /* Phase 5 - Step 4 */
    }

    /* Phase 5 - Step 4: Create estimators and spectrum */
    Estimators *est = create_estimators(geo.n_shells, opacity.n_lines); /* Phase 5 - Step 4 */
    double spec_min = 500.0, spec_max = 20000.0;
    int spec_bins = 2000;
    /* [스크랩 2026-08-07] LUMINA_SPEC_RANGE — 스펙트럼 범위는 위 기본값 고정 (attic/knobs) */
    Spectrum *spec = create_spectrum(spec_min, spec_max, spec_bins);

    Spectrum *spec_rot = enable_rotation ? create_spectrum(spec_min, spec_max, spec_bins) : NULL;

    /* NLTE: Initialize if enabled */
    NLTEConfig nlte;
    LineJbarQSet line_qset; memset(&line_qset, 0, sizeof(line_qset)); /* A2-06 */
    LineJbarAccumulator line_acc; memset(&line_acc, 0, sizeof(line_acc)); /* A2-06 */
    uint64_t *line_ids_u64 = NULL; /* A2-06: commit-request line_id form */
    memset(&nlte, 0, sizeof(nlte));
    if (enable_nlte) {
        printf("\n--- NLTE Initialization ---\n");
        if (nlte_init(&nlte, &atom_data, &opacity, geo.n_shells) != 0 || radiation_field_owner_init(&nlte.radiation_field, (size_t)geo.n_shells) != 0) { fprintf(stderr, "[RADIATION-FIELD][FATAL] initialization failed\n"); return EXIT_FAILURE; }
        bf_set_nlte_pops(&nlte);
        /* A2-06: Q_g = enabled bound-bound rate-graph lines (nlte_line_map>=0),
         * frozen + hashed before any accumulation (SPEC_A2_06_V5). */
        if (line_jbar_qset_build(&line_qset, opacity.n_lines,
                                 opacity.line_list_nu, nlte.nlte_line_map,
                                 NULL) != 0) {
            fprintf(stderr, "[A2-06][FATAL] Q_g build failed\n");
            return EXIT_FAILURE;
        }
        printf("  [A2-06] Q_g lines=%zu q_set_hash=%.16s... profile=%llu\n",
               line_qset.n_q, line_qset.q_set_hash,
               (unsigned long long)line_qset.profile_id);
        line_ids_u64 = (uint64_t *)malloc(line_qset.n_q * sizeof(uint64_t));
        if (!line_ids_u64) { fprintf(stderr, "[A2-06][FATAL] oom\n"); return EXIT_FAILURE; }
        for (size_t qi_ = 0; qi_ < line_qset.n_q; qi_++)
            line_ids_u64[qi_] = (uint64_t)line_qset.line_id[qi_];
        /* R6: main owns the one frozen Q_g used by both publication arms. */
        nlte.line_qset = &line_qset;
    }

    /* Gamma-ray deposition: initialize if enabled */
    GammaDeposition gamma_dep;
    memset(&gamma_dep, 0, sizeof(gamma_dep));
    if (gamma_dep_enabled) {
        gamma_deposition_init(&gamma_dep, geo.n_shells);
        printf("\n--- Gamma-ray Deposition Initialized ---\n");
    }

    /* BF opacity: initialize if enabled */
    BFOpacity bf;
    memset(&bf, 0, sizeof(bf));
    if (bf_opacity_enabled) {
        bf_opacity_init(&bf, geo.n_shells);
        /* Initial BF computation from reference plasma state */
        compute_bf_opacity(&bf, &atom_data, &plasma, geo.n_shells);
        printf("\n--- BF+FF Opacity Initialized (%d freq bins) ---\n", bf.n_freq_bins);
    }

    /* ============================================================ */
    /* PURE-CMFGEN parallel path (LUMINA_PURE_CMFGEN=1): bypass the   */
    /* Monte-Carlo loop, fill J_nu deterministically, run downstream  */
    /* solvers, dump plasma state, and skip to cleanup.               */
    /* ============================================================ */
    {
        const char *_pure = getenv("LUMINA_PURE_CMFGEN");
        if (_pure && atoi(_pure)) {
            const char *_ni = getenv("LUMINA_PURE_CMFGEN_ITER");
            int pc_iter = _ni ? atoi(_ni) : n_iterations;
            if (pc_iter < 1) pc_iter = 1;
            printf("\n=== PURE-CMFGEN deterministic radiation path "
                   "(MC transport bypassed) ===\n");
            if (cmfgen_run(&geo, &opacity,
                           bf_opacity_enabled ? &bf : NULL,
                           &plasma, enable_nlte ? &nlte : NULL, &atom_data,
                           gamma_dep_enabled ? &gamma_dep : NULL,
                           config.T_inner, pc_iter) != 0) {
                fprintf(stderr, "[CMFGEN][FATAL] deterministic path failed\n");
                return EXIT_FAILURE;
            }

            FILE *pf = fopen("lumina_plasma_state.csv", "w");
            if (pf) {
                fprintf(pf, "shell_id,T_e,n_e\n");
                for (int i = 0; i < geo.n_shells; i++)
                    fprintf(pf, "%d,%.6f,%.6e\n", i, plasma.T_e[i],
                            plasma.n_electron ? plasma.n_electron[i]
                                              : opacity.electron_density[i]);
                fclose(pf);
                printf("Pure-CMFGEN plasma state written to "
                       "lumina_plasma_state.csv\n");
            }

            free_geometry(&geo);
            free_opacity_state(&opacity);
            free_plasma_state(&plasma);
            free_estimators(est);
            free_spectrum(spec);
            if (spec_rot) free_spectrum(spec_rot);
            free(volume);
            free_atomic_data(&atom_data);
            if (enable_nlte) { radiation_field_owner_free(&nlte.radiation_field); nlte_free(&nlte); }
            line_jbar_accumulator_free(&line_acc);
            line_jbar_qset_free(&line_qset);
            free(line_ids_u64);
            if (gamma_dep_enabled) gamma_deposition_free(&gamma_dep);
            if (bf_opacity_enabled) bf_opacity_free(&bf);
            printf("\nDone (pure-CMFGEN).\n");
            return 0;
        }
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
        int te_qualified = 0;
        int material_locked = iter > 0 && nlte_ion_lock_active(iter);

        printf("\n--- Iteration %d/%d ---\n", iter + 1, n_iterations); /* Phase 5 - Step 5 */

        /* Phase 5 - Step 5: Reset estimators */
        reset_estimators(est); /* Phase 5 - Step 5 */
        reset_spectrum(spec); /* Phase 5 - Step 5 */
        if (spec_rot) reset_spectrum(spec_rot);
        if (enable_nlte)
            memset(nlte.j_nu_estimator, 0,
                   (size_t)geo.n_shells * nlte.n_freq_bins * sizeof(double));
        /* [MA-FATE] reset hist; only the final iteration retains counts. */
        if (iter == n_iterations - 1) {
            macro_atom_fate_reset();
            macro_atom_cycle_reset();
        }

        /* Phase 5 - Step 5: Recompute L_inner and time_simulation */
        L_inner = 4.0 * M_PI_VAL * geo.r_inner[0] * geo.r_inner[0] * /* Phase 5 - Step 5 */
                  SIGMA_SB * pow(config.T_inner, 4); /* Phase 5 - Step 5 */
        time_simulation = 1.0 / L_inner; /* Phase 5 - Step 5 */
        packet_energy = 1.0 / (double)n_packets; if (enable_nlte && radiation_field_begin_mc(&nlte.radiation_field, geo.v_inner, geo.v_outer, (size_t)geo.n_shells, geo.time_explosion, (uint64_t)(iter + 1)) != 0) { fprintf(stderr, "[RADIATION-FIELD][FATAL] MC work reset failed\n"); return EXIT_FAILURE; } a2_02c_capture_begin((unsigned long long)(iter + 1), (unsigned long long)n_packets, &geo, volume, time_simulation); /* A2-02C + A2-04 producers */

        /* Phase 5 - Step 5: Transport all packets (OpenMP-ready) */
        /* Store escaped packet data for spectrum binning after parallel section */
        double *escaped_nu = (double *)malloc(n_packets * sizeof(double)); /* Phase 5 - Step 5 */
        double *escaped_energy = (double *)malloc(n_packets * sizeof(double)); /* Phase 5 - Step 5 */
        int *escaped_flag = (int *)calloc(n_packets, sizeof(int)); /* Phase 5 - Step 5 */
        double *escaped_r = enable_rotation ? (double *)malloc(n_packets * sizeof(double)) : NULL;
        double *escaped_mu = enable_rotation ? (double *)malloc(n_packets * sizeof(double)) : NULL;
        int n_escaped = 0; /* Phase 5 - Step 5 */
        int n_reabsorbed = 0; int radiation_field_commit_error = 0; /* Phase 5 - Step 5; A2-04 */
        if (enable_nlte && line_qset.n_q > 0) { /* A2-06: fresh per generation */
            line_jbar_accumulator_free(&line_acc);
            if (line_jbar_accumulator_init(&line_acc, line_qset.n_q,
                                           (size_t)geo.n_shells) != 0)
                radiation_field_commit_error = 1;
        }

        #ifdef _OPENMP
        #pragma omp parallel reduction(|:radiation_field_commit_error)
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
                    (size_t)geo.n_shells * nlte.n_freq_bins, sizeof(double)); local_est->radiation_field_accumulator = radiation_field_accumulator_create((size_t)geo.n_shells); if (!local_est->radiation_field_accumulator) radiation_field_commit_error = 1;
            }
            int local_escaped = 0, local_reabsorbed = 0; /* Phase 5 - Step 5 */
            LineJbarPacketPartial line_partial; /* A2-06 thread-local */
            memset(&line_partial, 0, sizeof(line_partial));
            if (enable_nlte && line_qset.n_q > 0) {
                if (line_jbar_partial_init(&line_partial) != 0)
                    radiation_field_commit_error = 1;
                else {
                    local_est->line_jbar_qset = &line_qset;
                    local_est->line_jbar_accumulator = &line_acc;
                    local_est->line_jbar_partial = &line_partial;
                }
            }

            #ifdef _OPENMP
            #pragma omp for schedule(dynamic, 64)
            #endif
            for (int p = 0; p < n_packets; p++) { /* Phase 5 - Step 5 */
                RPacket pkt; /* Phase 5 - Step 5 */
                pkt.index = p; /* Phase 5 - Step 5 */
                initialize_packet(&pkt, &geo, &config, packet_energy, &rng); /* Phase 5 - Step 5 */

                single_packet_loop(&pkt, &geo, &opacity, local_est, &config,
                                   bf_opacity_enabled ? &bf : NULL, &plasma, &rng);
                /* A2-06: packet-population flush (variance needs y_p complete) */
                if (local_est->line_jbar_partial &&
                    line_jbar_packet_flush(&line_acc, &line_partial) != 0)
                    radiation_field_commit_error = 1;

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
                } if (local_est->radiation_field_accumulator && radiation_field_accumulator_reduce(&nlte.radiation_field.accumulator, local_est->radiation_field_accumulator) != 0) radiation_field_commit_error = 1;
                /* Phase 5 - Step 5: j_blue/Edotlu not tracked per-thread (too large) */
                n_escaped += local_escaped; /* Phase 5 - Step 5 */
                n_reabsorbed += local_reabsorbed; /* Phase 5 - Step 5 */
            }
            if (local_est->j_nu_estimator) free(local_est->j_nu_estimator);
            local_est->j_nu_estimator = NULL; radiation_field_accumulator_free(local_est->radiation_field_accumulator); local_est->radiation_field_accumulator = NULL;
            if (local_est->line_jbar_partial) { /* A2-06 */
                line_jbar_partial_free(&line_partial);
                local_est->line_jbar_partial = NULL;
                local_est->line_jbar_qset = NULL;
                local_est->line_jbar_accumulator = NULL;
            }
            free_estimators(local_est); /* Phase 5 - Step 5 */
        } a2_02c_capture_end(); if (radiation_field_commit_error || (enable_nlte && radiation_field_commit(&nlte.radiation_field, &(RadiationFieldCommitRequest){ .provenance_kind=RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH, .producer="CPU_MC_COMOVING_PATH_LENGTH_BIN_AVERAGE", .generation=(uint64_t)(iter+1), .epoch=geo.time_explosion, .n_shells=(size_t)geo.n_shells, .v_inner=geo.v_inner, .v_outer=geo.v_outer, .source_n_bins=LUMINA_RADFIELD_N_BINS, .statistic_kind=RADIATION_FIELD_ESTIMATOR_COUNT, .source_count=nlte.radiation_field.accumulator.contribution_count, .raw_path_length=nlte.radiation_field.accumulator.raw_path_length, .volume=volume, .time_simulation=time_simulation, .out_of_grid_contribution_count=nlte.radiation_field.accumulator.out_of_grid_contribution_count, .line_n=line_qset.n_q, .line_id=(const uint64_t*)line_ids_u64, .line_q_set_hash=line_qset.q_set_hash, .line_profile_id=line_qset.profile_id, .line_profile_hash=line_qset.profile_hash, .line_sum=line_acc.sum, .line_sumsq=line_acc.sumsq, .line_count=line_acc.count, .line_n_packets=(uint64_t)n_packets, .line_error_latch=line_acc.error_latch }) != 0)) { fprintf(stderr, "[RADIATION-FIELD][FATAL] MC commit failed\n"); return EXIT_FAILURE; } /* A2-02C/A2-04 + A2-06 dual-view commit */
        if (enable_nlte) { nlte.radfield_view_status = radiation_field_read_view(&nlte.radiation_field, geo.time_explosion, (size_t)geo.n_shells, (uint64_t)(iter+1), &nlte.radfield_view); if (nlte.radfield_view_status != RADIATION_FIELD_VIEW_OK) { fprintf(stderr, "[RADIATION-FIELD][FATAL] view refresh failed after MC commit status=%d\n", nlte.radfield_view_status); return EXIT_FAILURE; } } /* A2-05: the only MC-lane view refresh point */
        if (enable_nlte && line_qset.n_q > 0) { /* A2-06: line view refresh */
            nlte.line_qset = &line_qset;
            nlte.line_view_status = radiation_field_line_jbar_view(
                &nlte.radiation_field, geo.time_explosion, (size_t)geo.n_shells,
                (uint64_t)(iter+1), line_qset.q_set_hash, line_qset.profile_id,
                line_qset.profile_hash,
                &nlte.line_view);
            if (nlte.line_view_status != LINE_JBAR_VIEW_OK) {
                fprintf(stderr, "[A2-06][FATAL] line view refresh failed "
                        "status=%d\n", nlte.line_view_status);
                return EXIT_FAILURE;
            }
        }

        /* Gamma publication owns the immutable physical epoch.  It follows the
         * R7 commit/view barrier and precedes A2-10 and every material update. */
        if (!material_locked && gamma_dep_enabled &&
            gamma_dep.generation == 0) {
            int gamma_rc = gamma_deposition_publish(
                &gamma_dep, GAMMA_PROVENANCE_INTERNAL_BATEMAN,
                geo.time_explosion, &atom_data, &plasma, &geo, NULL);
            if (gamma_rc != 0) {
                fprintf(stderr,
                        "[GAMMA][FATAL] lane=MC iter=%d rc=%d\n",
                        iter, gamma_rc);
                return EXIT_FAILURE;
            }
            printf("  [Gamma] heating_rate[0]=%.2e, [%d]=%.2e erg/s/cm3\n",
                   gamma_dep.heating_rate[0], geo.n_shells - 1,
                   gamma_dep.heating_rate[geo.n_shells - 1]);
        }

        {
            int r7_rc = lumina_r7_publish_and_solve_te(
                &opacity, bf_opacity_enabled ? &bf : NULL,
                &atom_data, &plasma, enable_nlte ? &nlte : NULL,
                gamma_dep_enabled ? &gamma_dep : NULL,
                geo.time_explosion, geo.n_shells,
                radeq_te && !material_locked, "MC", iter);
            if (r7_rc != 0) {
                fprintf(stderr,
                        "[R7][FATAL] lane=MC iter=%d rc=%d\n",
                        iter, r7_rc);
                return EXIT_FAILURE;
            }
            te_qualified = radeq_te && !material_locked;
        }

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

        /* Binned-J estimator reads the frequency-resolved histogram, which is
         * reduced into nlte.j_nu_estimator (not est). Expose it (raw, this
         * iteration) on est so solve_radiation_field can fit a dilute Planck.
         * Safe: free/reset_estimators never touch est->j_nu_estimator. */
        if (enable_nlte) {
            est->j_nu_estimator   = nlte.j_nu_estimator;
            est->nlte_n_freq_bins = nlte.n_freq_bins;
            est->nlte_nu_min      = nlte.nu_min;
            est->nlte_d_log_nu    = nlte.d_log_nu;
        }

        /* Option (8): freeze W/T_rad too once ion-lock activates — true
         * transport-only iteration; plasma state from converged free-NLTE iter. */
        if (!material_locked) {
            /* Phase 5 - Step 6: Solve radiation field */
            solve_radiation_field(est, geo.time_explosion, time_simulation, volume,
                                   &opacity, &plasma, config.damping_constant);
        }

        if (material_locked) {
            printf("  [plasma frozen by ion-lock; transport-only iter %d]\n", iter);
        } else if (iter > 0 || te_qualified) {
            if (compute_plasma_state(&atom_data, &plasma, &opacity,
                                     geo.time_explosion) != 0) {
                fprintf(stderr, "[A2-07][FATAL] population transaction failed at iter=%d\n",
                        iter);
                return EXIT_FAILURE;
            }
            if(te_qualified&&(atom_data.partition_stamp.te_generation!=plasma.te_publication.committed_te_generation||strcmp(atom_data.partition_stamp.te_manifest_sha256,plasma.te_publication.te_manifest_sha256))){fprintf(stderr,"[A2-10][FATAL] A2-07 Te/population stamp mismatch iter=%d\n",iter);return EXIT_FAILURE;}

            /* Recompute BF opacity grid after plasma update */
            if (bf_opacity_enabled)
                compute_bf_opacity(&bf, &atom_data, &plasma, geo.n_shells);

            /* NLTE: solve rate equations and update tau for NLTE lines */
            if (enable_nlte && iter >= nlte_start_iter) {
                nlte.current_iter = iter;
                nlte_normalize_j_nu(&nlte, time_simulation, volume, geo.n_shells);
                if (nlte_solve_all(&nlte, &atom_data, &plasma, &opacity,
                                   geo.time_explosion, geo.n_shells,
                                   gamma_dep_enabled ? &gamma_dep : NULL) != 0) {
                    fprintf(stderr, "[NLTE][FATAL] solve failed at iter=%d\n", iter);
                    return EXIT_FAILURE;
                }

            }

            /* Dynamic transition probability recomputation */
            if (enable_transprob_update && iter >= config.hold_iterations) {
                compute_transition_probabilities(&atom_data, &plasma, &opacity,
                    (enable_nlte && iter >= nlte_start_iter) ? &nlte : NULL,
                    config.damping_constant,
                    (iter > config.hold_iterations) ? 1 : 0, &geo);
            }
        }

        /* Phase 5 - Step 6: Print canonical estimator diagnostics. */
        printf("  Shell  T_e         nubar/j\n");
        for (int i = 0; i < geo.n_shells; i += 5) { /* Phase 5 - Step 6: print every 5th */
            double ratio = est->nu_bar_estimator[i] / est->j_estimator[i]; /* Phase 5 - Step 6 */
            printf("  %3d    %.2f K   %.4e\n", i, plasma.T_e[i], ratio);
        }

        /* Phase 5 - Step 7: Update T_inner (after hold iterations) */
        if (iter >= config.hold_iterations) { /* Phase 5 - Step 7 */
            double old_T = config.T_inner; /* Phase 5 - Step 7 */
            int t_inner_frozen = nlte_ion_lock_active(iter);
            /* [스크랩 2026-08-07] LUMINA_DIFFUSION_INNER_BC · LUMINA_T_INNER_FIX
             * — 고정-L 확산 BC 와 T_inner 핀. 둘 다 미설정이 기본이었다 (attic/knobs) */
            if (!t_inner_frozen) {
                update_t_inner(&config, L_emitted);
                printf("  T_inner: %.2f K -> %.2f K (L_em=%.3e, L_req=%.3e)\n",
                       old_T, config.T_inner, L_emitted, config.luminosity_requested);
            } else {
                printf("  T_inner: %.2f K [frozen-by-lock] (L_em=%.3e, L_req=%.3e)\n",
                       config.T_inner, L_emitted, config.luminosity_requested);
            }
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

    /* [MA-FATE] Macro-atom packet fate histogram (final iteration) */
    macro_atom_fate_print("final iteration, CPU transport");
    macro_atom_cycle_print("final iteration, CPU transport");

    printf("T_inner final: %.2f K\n", config.T_inner);

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

    /* P5: Formal integral spectrum (noise-free) */
    {
        Spectrum *spec_fi = create_spectrum(spec_min, spec_max, spec_bins);
        compute_formal_integral_spectrum(
            &geo, &plasma, &opacity, &atom_data,
            nlte.enabled ? &nlte : NULL, config.T_inner,
            spec_fi, 100, 0.0);
        FILE *ff = fopen("lumina_spectrum_formal.csv", "w");
        if (ff) {
            fprintf(ff, "wavelength_angstrom,flux\n");
            for (int i = 0; i < spec_fi->n_bins; i++)
                fprintf(ff, "%.6f,%.6e\n", spec_fi->wavelength[i], spec_fi->flux[i]);
            fclose(ff);
            printf("Formal integral spectrum written to lumina_spectrum_formal.csv\n");
        }
        free_spectrum(spec_fi);
    }

    /* [스크랩 2026-08-07] LUMINA_TRANSPORT=cmf 진입점과 LUMINA_CMF_{NZ,NIMPACT,VTURB_KMS}
     * — 결정론 팔의 정본은 lumina_cmfgen.c 이며 이쪽은 구 진입점이다 (attic/knobs) */

    /* Phase 5 - Step 9b: Write material state; radiation has its own schema. */
    out = fopen("lumina_plasma_state.csv", "w"); /* Phase 5 - Step 9b */
    if (out) { /* Phase 5 - Step 9b */
        fprintf(out, "shell_id,T_e,n_e\n");
        for (int i = 0; i < geo.n_shells; i++) { /* Phase 5 - Step 9b */
            fprintf(out, "%d,%.6f,%.6e\n", i, plasma.T_e[i],
                    plasma.n_electron[i]);
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
    if (enable_nlte) { radiation_field_owner_free(&nlte.radiation_field); nlte_free(&nlte); }
    line_jbar_accumulator_free(&line_acc);
    line_jbar_qset_free(&line_qset);
    free(line_ids_u64);
    if (gamma_dep_enabled) gamma_deposition_free(&gamma_dep);
    if (bf_opacity_enabled && bf.event_enabled)
        printf("[BF-PHIXS-FALLBACK] CPU upper-ground fallback activations=%llu\n",
               bf.event_target_fallback_activations);
    if (bf_opacity_enabled) bf_opacity_free(&bf);

    printf("\nDone.\n"); /* Phase 5 - Step 10 */
    return 0; /* Phase 5 - Step 10 */
}
