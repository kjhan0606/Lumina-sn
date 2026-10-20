/* field_branch_ioniz_selftest.c
 *
 * One-cell radiation-field branch comparison using the production
 * radeq_simul_all() Gph/Milne consumer.  The atomic deck still carries the
 * original 50-shell shape; FIELD_SHELL selects the reported shell and all
 * other shells receive the same controlled field so the loader/consumer
 * contract remains unchanged.
 *
 * Branches (LUMINA_FIELD_BRANCH):
 *   B_TE          J_nu = B_nu(T_e)
 *   W_B           J_nu = W(s) B_nu(T_rad)
 *   CMFGEN_TABLE J_nu = one row of the sealed CMFGEN J table
 *
 * The production ionization self-test is set to mode 2 by this harness: it
 * prints the live field result and does not replace the supplied J_nu.
 */
#include "../src/lumina.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Keep the expression identical to the production Planck helper. */
static double planck_bnu(double nu, double T)
{
    const double x = H_PLANCK * nu / (K_BOLTZMANN * T);
    if (x > 500.0) return 0.0;
    return (2.0 * H_PLANCK * nu * nu * nu /
            (2.99792458e10 * 2.99792458e10)) / (exp(x) - 1.0);
}

static int load_jtable(const char *path, int n_shells, int nfb, double **out)
{
    FILE *fp = fopen(path, "rb");
    int hdr[4] = {0, 0, 0, 0};
    size_t source_count, target_count;
    double *source, *data;
    if (!fp) return -1;
    if (fread(hdr, sizeof(hdr[0]), 4, fp) != 4 ||
        hdr[0] != 0x4A544142 || hdr[2] != n_shells ||
        (hdr[3] != 1000 && hdr[3] != nfb)) {
        fclose(fp);
        return -2;
    }
    source_count = (size_t)n_shells * (size_t)hdr[3];
    source = (double *)malloc(source_count * sizeof(*source));
    if (!source || fread(source, sizeof(*source), source_count, fp) != source_count) {
        free(source);
        fclose(fp);
        return -3;
    }
    fclose(fp);
    target_count = (size_t)n_shells * (size_t)nfb;
    data = (double *)malloc(target_count * sizeof(*data));
    if (!data) {
        free(source);
        return -4;
    }
    if (hdr[3] == nfb) {
        memcpy(data, source, target_count * sizeof(*data));
    } else {
        /* The historical CMFGEN J-table is on the same log-frequency span
         * with 1000 bin centers; interpolate its positive spectrum onto the
         * current closed 1234-bin grid.  Zero coverage remains zero. */
        const int src_nfb = hdr[3];
        for (int s = 0; s < n_shells; ++s)
            for (int b = 0; b < nfb; ++b) {
                const double x = ((double)b + 0.5) / (double)nfb;
                const double u = x * (double)src_nfb - 0.5;
                int j = (int)floor(u);
                double f = u - floor(u);
                if (j < 0) { j = 0; f = 0.0; }
                if (j >= src_nfb - 1) { j = src_nfb - 1; f = 0.0; }
                const double a = source[(size_t)s * src_nfb + (size_t)j];
                const double z = source[(size_t)s * src_nfb + (size_t)(j + (f > 0.0 && j < src_nfb - 1 ? 1 : 0))];
                double value = 0.0;
                if (a > 0.0 && z > 0.0 && f > 0.0)
                    value = exp((1.0 - f) * log(a) + f * log(z));
                else if (a > 0.0)
                    value = a;
                data[(size_t)s * nfb + (size_t)b] = value;
            }
    }
    free(source);
    *out = data;
    return 0;
}

static int parse_shell(void)
{
    const char *e = getenv("FIELD_SHELL");
    int s = e ? atoi(e) : 30;
    return s;
}

int main(int argc, char **argv)
{
    const char *ref_dir = argc > 1 ? argv[1] : "data/tardis_reference_toy06_19p48d";
    const double T = argc > 2 ? atof(argv[2]) : 9865.5;
    const double rho = argc > 3 ? atof(argv[3]) : 1.0e-13;
    const int n_shells = 50; /* sealed toy06 abundance-table contract */
    const int field_shell = parse_shell();
    const char *branch = getenv("LUMINA_FIELD_BRANCH");
    const char *table_path = getenv("LUMINA_FIELD_TABLE");
    const double W = getenv("FIELD_W") ? atof(getenv("FIELD_W"))
                                        : 0.005611760422201795;
    const double T_rad = getenv("FIELD_TRAD") ? atof(getenv("FIELD_TRAD"))
                                               : 11610.792206;
    AtomicData atom;
    PlasmaState plasma;
    OpacityState opacity;
    NLTEConfig nlte;
    double *table = NULL;
    char sigbuf[512];

    if (!branch || !*branch) branch = "B_TE";
    if (field_shell < 0 || field_shell >= n_shells) {
        fprintf(stderr, "invalid FIELD_SHELL=%d\n", field_shell);
        return 2;
    }
    if (strcmp(branch, "B_TE") != 0 && strcmp(branch, "W_B") != 0 &&
        strcmp(branch, "CMFGEN_TABLE") != 0) {
        fprintf(stderr, "unknown LUMINA_FIELD_BRANCH=%s\n", branch);
        return 2;
    }
    if (strcmp(branch, "CMFGEN_TABLE") == 0) {
        if (!table_path) table_path = "data/cmfgen_jtable_toy06_19p48d.bin";
        /* nfb is checked after nlte_init; defer the read below. */
    }

    memset(&atom, 0, sizeof(atom));
    memset(&plasma, 0, sizeof(plasma));
    memset(&opacity, 0, sizeof(opacity));
    memset(&nlte, 0, sizeof(nlte));
    if (load_atomic_data(&atom, ref_dir, n_shells) != 0) {
        fprintf(stderr, "load_atomic_data(%s) failed\n", ref_dir);
        return 3;
    }
    if (!getenv("LUMINA_CMFGEN_SIGMA_BF") ||
        !strchr(getenv("LUMINA_CMFGEN_SIGMA_BF"), '/')) {
        snprintf(sigbuf, sizeof(sigbuf), "%s/cmfgen_sigma_bf.bin", ref_dir);
        setenv("LUMINA_CMFGEN_SIGMA_BF", sigbuf, 1);
    }
    if (load_cmfgen_sigma_bf(&atom, getenv("LUMINA_CMFGEN_SIGMA_BF")) != 0) {
        fprintf(stderr, "load_cmfgen_sigma_bf failed\n");
        return 4;
    }
    /* Diagnostic-only perturbation.  Apply it after loading, in memory, so the
     * sealed sigma asset is never overwritten.  This deliberately scales the
     * same sigma table consumed by both Gph and the Milne alpha path; it is a
     * detailed-balance sensitivity control, not an independent alpha test. */
    double sigma_scale = 1.0;
    const char *sigma_scale_env = getenv("LUMINA_FIELD_SIGMA_SCALE");
    if (sigma_scale_env && *sigma_scale_env) {
        sigma_scale = atof(sigma_scale_env);
        if (!isfinite(sigma_scale) || sigma_scale <= 0.0) {
            fprintf(stderr, "invalid LUMINA_FIELD_SIGMA_SCALE=%s\n",
                    sigma_scale_env);
            return 4;
        }
        if (sigma_scale != 1.0) {
            size_t n_sigma = (size_t)atom.n_levels *
                             (size_t)atom.cmfgen_n_freq_bins;
            for (size_t i = 0; i < n_sigma; ++i)
                atom.cmfgen_sigma_bf[i] *= sigma_scale;
        }
    }
#ifdef LUMINA_SELFTEST_ALPHA_SCALE
    const double alpha_scale = (double)LUMINA_SELFTEST_ALPHA_SCALE;
#else
    const double alpha_scale = 1.0;
#endif

    plasma.n_shells = n_shells;
    plasma.T_e = (double *)malloc((size_t)n_shells * sizeof(*plasma.T_e));
    plasma.n_electron = (double *)malloc((size_t)n_shells * sizeof(*plasma.n_electron));
    plasma.rho = (double *)malloc((size_t)n_shells * sizeof(*plasma.rho));
    for (int s = 0; s < n_shells; ++s) {
        plasma.T_e[s] = T;
        plasma.n_electron[s] = 1.0e8;
        plasma.rho[s] = rho;
    }
    for (int e = 0; e < atom.n_elements; ++e)
        for (int s = 0; s < n_shells; ++s)
            atom.abundances[e * n_shells + s] = 1.0 / (double)atom.n_elements;
    for (int ip = 0; ip < atom.n_ion_pops; ++ip)
        for (int s = 0; s < n_shells; ++s) {
            atom.ion_number_density[(size_t)ip * n_shells + s] =
                atom.ion_pop_stage[ip] == 1 ? 1.0e8 : 0.0;
            atom.partition_functions[(size_t)ip * n_shells + s] = 1.0;
        }
    opacity.n_lines = atom.n_lines;
    opacity.n_shells = n_shells;
    opacity.tau_sobolev = (double *)calloc((size_t)atom.n_lines * n_shells,
                                           sizeof(*opacity.tau_sobolev));
    if (nlte_init(&nlte, &atom, &opacity, n_shells) != 0) {
        fprintf(stderr, "nlte_init failed\n");
        return 5;
    }
    if (strcmp(branch, "CMFGEN_TABLE") == 0 &&
        load_jtable(table_path, n_shells, nlte.n_freq_bins, &table) != 0) {
        fprintf(stderr, "CMFGEN table load failed: %s\n", table_path);
        return 6;
    }

    for (int s = 0; s < n_shells; ++s)
        for (int b = 0; b < nlte.n_freq_bins; ++b) {
            const double nu = exp(log(nlte.nu_min) +
                                  ((double)b + 0.5) * nlte.d_log_nu);
            double J = planck_bnu(nu, T);
            if (strcmp(branch, "W_B") == 0) J = W * planck_bnu(nu, T_rad);
            if (strcmp(branch, "CMFGEN_TABLE") == 0)
                J = table[(size_t)s * nlte.n_freq_bins + (size_t)b];
            nlte.J_nu[(size_t)s * nlte.n_freq_bins + (size_t)b] = J;
        }
    nlte.enabled = 1;
    setenv("LUMINA_IONIZ_SELFTEST", "2", 1);
    printf("[FIELD-BRANCH] branch=%s reported_shell=%d T_e=%.6f rho=%.6e "
           "W=%.17g T_rad=%.6f sigma_scale=%.17g alpha_scale=%.17g table=%s\n",
           branch, field_shell, T, rho, W, T_rad,
           sigma_scale,
           alpha_scale,
           table_path ? table_path : "none");
    fflush(stdout);

    if (lumina_ioniz_selftest_run(&plasma, &nlte, &atom, &opacity,
                                  86400.0, n_shells) != 0) {
        fprintf(stderr, "ionization self-test core did not run\n");
        return 7;
    }
    free(table);
    return 0;
}
