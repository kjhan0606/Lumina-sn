/* selftest_ioniz_saha.c — KNOWN-ANSWER (constrained) test of the IONIZATION path.
 *
 * The transport side already has four known-answer self-tests
 * (cmf_plasma_selftest / cmf_nlte_selftest / cmf_fsolve_selftest / cmf_obs_selftest).
 * The ionization/photoionization path had none. This is that test.
 *
 * IDENTITY UNDER TEST
 *   Impose J_nu = B_nu(T_e).  With detailed balance intact, the production
 *   photoionization/recombination equilibrium MUST reproduce Saha exactly:
 *       q      = Gamma_phot / alpha_rec                          [cm^-3]
 *       q_saha = 2 (U_{i+1}/U_i) (2 pi m_e k T / h^2)^{3/2} exp(-chi/kT)
 *   and n(i+1)/n(i) = q/n_e, i.e. the r of the brief.  ratio = q/q_saha must be 1.
 *
 * WHAT IS EXERCISED — all PRODUCTION code, nothing re-implemented:
 *   radeq_simul_all()            Gph assembly (lumina_plasma.c)
 *   frozenin_alpha_rr()          Milne recombination coefficient
 *   coupled_photoion_rate_jnu()  the NLTE-route photoion rate
 *   simul_ladder()               the ion ladder + n_e fixed point
 * reached through the public entry compute_radiative_equilibrium_te() with
 * LUMINA_RADEQ_SIMUL=1.  The probe itself lives behind LUMINA_IONIZ_SELFTEST
 * (default OFF => production byte-identical).
 *
 * One shell per test temperature; T_e[s] is what the Gph loop and the Milne
 * integral both read, and Gph is built BEFORE the radeq T bisection, so the
 * printed identity is evaluated exactly at the requested T.
 *
 * Build: make selftest_ioniz_saha      (CPU only, gcc, no GPU needed)
 * Run:   LUMINA_IONIZ_SELFTEST=1 LUMINA_RADEQ_SIMUL=1 \
 *        ./selftest_ioniz_saha data/tardis_reference 5000,10000,20000,50000
 */
#include "src/lumina.h"
#include <math.h>

static double planck(double nu, double T) {
    const double h = H_PLANCK, k = K_BOLTZMANN, c = 2.99792458e10;
    double x = h * nu / (k * T);
    if (x > 500.0) return 0.0;
    return 2.0 * h * nu * nu * nu / (c * c) / (exp(x) - 1.0);
}

/* [RATES-FIX F3] production Milne integral, exported by lumina_plasma.c */
double lumina_rates_alpha_probe(AtomicData *atom, int Z, int stage, double T);

/* A3: low-T NaN probe. exp(+chi_l/kT) overflows to inf below T ~ chi_l/(700k)
 * while the x>700 bin skip has already zeroed Rbf => 0 x inf = NaN. Run with
 * LUMINA_ALPHA_LOWT_PROBE=1 (with and without LUMINA_RATES_FIX=1). */
static int alpha_lowt_probe(AtomicData *atom) {
    const int zs[3][2] = {{13, 2}, {12, 1}, {26, 3}};   /* Al III, Mg II, Fe IV */
    const char *nm[3]  = {"Al III->IV", "Mg II->III", "Fe IV->V"};
    const double Tp[3] = {200.0, 500.0, 1000.0};
    int nbad = 0;
    printf("=== [RATES-FIX F3] low-T alpha probe (frozenin_alpha_rr) ===\n");
    for (int i = 0; i < 3; i++)
        for (int t = 0; t < 3; t++) {
            double a = lumina_rates_alpha_probe(atom, zs[i][0], zs[i][1], Tp[t]);
            int bad = !isfinite(a);
            if (bad) nbad++;
            printf("  %-12s Z=%d stage=%d T=%6.0f  alpha=%-14.6e %s\n",
                   nm[i], zs[i][0], zs[i][1], Tp[t], a, bad ? "*** NOT FINITE ***" : "finite");
        }
    printf("=== low-T alpha probe: %d/9 non-finite ===\n", nbad);
    fflush(stdout);
    return nbad;
}

int main(int argc, char **argv) {
    const char *ref_dir = (argc > 1) ? argv[1] : "data/tardis_reference_toy06_19p48d";
    const char *tlist   = (argc > 2) ? argv[2] : "5000,10000,20000,50000";
    double rho = (argc > 3) ? atof(argv[3]) : 1e-13;

    double T[32]; int n_shells = 0;
    { const char *p = tlist;
      while (*p && n_shells < 32) { T[n_shells++] = atof(p);
          while (*p && *p != ',') p++; if (*p == ',') p++; } }
    if (n_shells < 1) { fprintf(stderr, "no temperatures\n"); return 1; }

    AtomicData atom; PlasmaState plasma; OpacityState opacity; NLTEConfig nlte;
    memset(&plasma, 0, sizeof plasma); memset(&opacity, 0, sizeof opacity);
    memset(&nlte, 0, sizeof nlte);
    if (load_atomic_data(&atom, ref_dir, n_shells) != 0) {
        fprintf(stderr, "load_atomic_data(%s) failed\n", ref_dir); return 1; }

    /* the bf cross sections both halves of the identity integrate */
    char sigbuf[512];
    const char *sig = getenv("LUMINA_CMFGEN_SIGMA_BF");   /* production semantics */
    if (!sig || !strchr(sig, '/')) {
        snprintf(sigbuf, sizeof sigbuf, "%s/cmfgen_sigma_bf.bin", ref_dir);
        sig = sigbuf;
    }
    if (load_cmfgen_sigma_bf(&atom, sig) != 0) {
        fprintf(stderr, "FATAL: cmfgen sigma_bf not loaded (%s) — the identity is "
                        "untestable without the bf cross sections\n", sig);
        return 2;
    }

    /* A3 probe: needs only the atomic data + sigma_bf loaded above. Exits
     * before the (minutes-long) radeq pass; env unset => never runs. */
    if (getenv("LUMINA_ALPHA_LOWT_PROBE")) return alpha_lowt_probe(&atom) > 0 ? 3 : 0;

    plasma.n_shells = n_shells;
    plasma.T_e   = malloc(n_shells * sizeof(double));
    plasma.n_electron = malloc(n_shells * sizeof(double));
    plasma.rho   = malloc(n_shells * sizeof(double));
    for (int s = 0; s < n_shells; s++) {
        plasma.T_e[s] = T[s];
        plasma.rho[s] = rho; plasma.n_electron[s] = 1e8;
    }
    /* Uniform composition: the identity is PER ION, and the model's inner shells
     * are pure IGE, so an equal-mass mix is what makes every element's ladder get
     * evaluated at every test temperature. Harness-side only. */
    for (int e = 0; e < atom.n_elements; e++)
        for (int s = 0; s < n_shells; s++)
            atom.abundances[e * n_shells + s] = 1.0 / (double)atom.n_elements;
    for (int ip = 0; ip < atom.n_ion_pops; ip++)
        for (int s = 0; s < n_shells; s++) {
            atom.ion_number_density[ip * n_shells + s] =
                (atom.ion_pop_stage[ip] == 1) ? 1e8 : 0.0;
            atom.partition_functions[ip * n_shells + s] = 1.0;
        }

    opacity.n_lines  = atom.n_lines;
    opacity.n_shells = n_shells;
    opacity.tau_sobolev = calloc((size_t)atom.n_lines * n_shells, sizeof(double));
    if (nlte_init(&nlte, &atom, &opacity, n_shells) != 0) {
        fprintf(stderr, "nlte_init failed\n"); return 1; }

    /* impose TE: J_nu = B_nu(T_e[s]) on the production bin grid (the gate also
     * forces this at every field read; setting it here makes the test valid even
     * with LUMINA_IONIZ_SELFTEST=2). */
    for (int s = 0; s < n_shells; s++)
        for (int b = 0; b < nlte.n_freq_bins; b++) {
            double nu = exp(log(nlte.nu_min) + (b + 0.5) * nlte.d_log_nu);
            nlte.J_nu[(size_t)s * nlte.n_freq_bins + b] = planck(nu, T[s]);
        }
    nlte.enabled = 1;

    printf("=== IONIZATION KNOWN-ANSWER TEST (J_nu = B_nu(T_e)) ===\n");
    printf("ref_dir=%s  rho=%.3e  n_shells=%d  nfb=%d  nu=[%.4e,%.4e]  "
           "cmfgen_sigma: loaded=%d nfreq=%d nu=[%.4e,%.4e]\n",
           ref_dir, rho, n_shells, nlte.n_freq_bins, nlte.nu_min, nlte.nu_max,
           atom.cmfgen_loaded, atom.cmfgen_n_freq_bins,
           atom.cmfgen_nu_min, atom.cmfgen_nu_max);
    if (atom.cmfgen_loaded &&
        (atom.cmfgen_n_freq_bins != nlte.n_freq_bins ||
         fabs(atom.cmfgen_nu_min / nlte.nu_min - 1.0) > 1e-12 ||
         fabs(atom.cmfgen_nu_max / nlte.nu_max - 1.0) > 1e-12))
        printf("*** GRID MISMATCH: sigma_bf grid != J_nu grid — the two halves of "
               "the identity do not share a quadrature ***\n");
    for (int s = 0; s < n_shells; s++) printf("  s=%d T_e=%.0f K\n", s, T[s]);
    fflush(stdout);

    /* PRODUCTION entry; LUMINA_RADEQ_SIMUL=1 routes into radeq_simul_all. */
    compute_radiative_equilibrium_te(&plasma, NULL, &nlte, &atom, &opacity,
                                     86400.0, n_shells);
    printf("=== done ===\n");
    return 0;
}
