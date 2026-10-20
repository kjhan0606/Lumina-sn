/* Gate-B Phase-1.6 frozen-cell oracle.
 *
 * This executable imports a recorded Lumina cell, then calls the production
 * opacity and NLTE rate-assembly functions.  LUMINA_FROZEN_ORACLE only adds
 * read-only probes at their consumption points; normal production targets do
 * not compile those probes.
 */
#include "src/lumina.h"

#include <errno.h>
#include <sys/stat.h>

#define ORACLE_COARSE_BINS 24
#define ORACLE_TARGETS 8

static const int target_Z[ORACLE_TARGETS] = {14,14,16,16,26,26,26,27};
static const int target_stage[ORACLE_TARGETS] = {1,2,1,2,1,2,3,2};

typedef struct {
    int consume_iter;
    int producer_iter;
    int c1_bins;
    int c2_bins;
    int jbar_rows;
    int jbar_exact_ions;
    int tail_color_active;
    int tri_response_active;
    int line_re_active;
    int photoion_mc_active;
    double photoion_mc_alpha;
    unsigned long long ma_terminals_global;
    unsigned long long ma_destroyed_global;
    int ma_cell_rate_available;
    double ma_cell_heating;
} FrozenReplay;

static int g_ew_sigma_available = 0;
int nlte_precompute_within_sl_frac_checked(
    NLTEConfig *, AtomicData *, PlasmaState *, int);

static void path_join(char *out, size_t n, const char *dir, const char *name) {
    snprintf(out, n, "%s/%s", dir, name);
}

static int load_resolved_environment(const char *run_dir) {
    char path[4096], line[8192];
    path_join(path, sizeof(path), run_dir, "stdout.log");
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    int in_config = 0, loaded = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "=== RESOLVED CONFIG")) { in_config = 1; continue; }
        if (!in_config) continue;
        if (strstr(line, "argv:")) break;
        char key[256], value[7680];
        if (sscanf(line, " %255[^=]=%7679[^\n]", key, value) == 2) {
            if (!strcmp(key, "LUMINA_BIN") || !strcmp(key, "OMP_NUM_THREADS"))
                continue;
            setenv(key, value, 1);
            loaded++;
        }
    }
    fclose(fp);
    setenv("OMP_NUM_THREADS", "1", 1);
    return loaded;
}

static int env_enabled(const char *name) {
    const char *value = getenv(name);
    return value && atoi(value) != 0;
}

/* jbar dumps are ordered by iteration and can be tens of gigabytes.  Position
 * at the requested iteration instead of reparsing every older frozen block. */
static int seek_csv_iteration(FILE *fp, int wanted_iter) {
    char line[2048];
    if (fseeko(fp,0,SEEK_END)) return -1;
    off_t end=ftello(fp),lo=0,hi=end;
    while(hi-lo>(off_t)(1<<20)){
        off_t mid=lo+(hi-lo)/2;
        if(fseeko(fp,mid,SEEK_SET))return -1;
        if(mid>0)(void)fgets(line,sizeof(line),fp);
        off_t pos=ftello(fp);
        int it=-1;
        if(!fgets(line,sizeof(line),fp)) { hi=mid; continue; }
        if(sscanf(line,"%d,",&it)!=1||it<wanted_iter)lo=pos;
        else hi=mid;
    }
    off_t start=lo>(off_t)(2<<20)?lo-(off_t)(2<<20):0;
    if(fseeko(fp,start,SEEK_SET))return -1;
    if(start>0)(void)fgets(line,sizeof(line),fp);
    while(1){
        off_t pos=ftello(fp);int it=-1;
        if(!fgets(line,sizeof(line),fp))break;
        if(sscanf(line,"%d,",&it)!=1)continue;
        if(it==wanted_iter){return fseeko(fp,pos,SEEK_SET)?-1:0;}
        if(it>wanted_iter)break;
    }
    return -1;
}

static int find_consume_iteration(const char *run_dir, int shell) {
    (void)shell; /* every archived iteration contains the complete shell set */
    char path[4096], line[2048];
    path_join(path, sizeof(path), run_dir, "lumina_jbar_dump.csv");
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    if(fseeko(fp,0,SEEK_END)){fclose(fp);return -1;}
    off_t end=ftello(fp),start=end>(off_t)(2<<20)?end-(off_t)(2<<20):0;
    if(fseeko(fp,start,SEEK_SET)){fclose(fp);return -1;}
    if(start>0)(void)fgets(line,sizeof(line),fp);
    int last_iter = -1;
    while (fgets(line, sizeof(line), fp)) {
        int it, s;
        if (sscanf(line, "%d,%d", &it, &s) == 2 && it > last_iter)
            last_iter = it;
    }
    fclose(fp);
    return last_iter;
}

static int load_ma_global_counts(const char *run_dir, int producer_iter,
                                 FrozenReplay *replay) {
    char path[4096], line[8192];
    path_join(path, sizeof(path), run_dir, "stdout.log");
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    int found = 0;
    while (fgets(line, sizeof(line), fp)) {
        int it;
        unsigned long long terminals, destroyed;
        if (sscanf(line,
                   "[MA-LINE-DESTRUCT] it %d: terminals=%llu destroyed=%llu",
                   &it, &terminals, &destroyed) == 3 ||
            sscanf(line,
                   "[MA-LINE-DESTRUCT] it%d: terminals=%llu destroyed=%llu",
                   &it, &terminals, &destroyed) == 3) {
            if (it == producer_iter) {
                replay->ma_terminals_global = terminals;
                replay->ma_destroyed_global = destroyed;
                found = 1;
            }
        }
    }
    fclose(fp);
    return found ? 0 : -1;
}

static void load_ma_cell_rate(const char *run_dir, int producer_iter, int shell,
                              FrozenReplay *replay) {
    char path[4096], line[1024];
    path_join(path, sizeof(path), run_dir, "lumina_ma_line_destruct.csv");
    FILE *fp = fopen(path, "r");
    if (!fp) return;
    (void)fgets(line, sizeof(line), fp);
    while (fgets(line, sizeof(line), fp)) {
        int it, s;
        unsigned long long terminals, destroyed;
        double heating;
        if (sscanf(line, "%d,%d,%llu,%llu,%lf", &it, &s, &terminals,
                   &destroyed, &heating) == 5 &&
            it == producer_iter && s == shell && isfinite(heating)) {
            replay->ma_cell_rate_available = 1;
            replay->ma_cell_heating = heating;
            break;
        }
    }
    fclose(fp);
}

static int find_ion(const AtomicData *atom, int Z, int stage) {
    for (int i = 0; i < atom->n_ion_pops; i++)
        if (atom->ion_pop_Z[i] == Z && atom->ion_pop_stage[i] == stage) return i;
    return -1;
}

static int find_global_level(const AtomicData *atom, int Z, int stage, int levnum) {
    int ip = find_ion(atom, Z, stage);
    if (ip < 0) return -1;
    for (int g = atom->level_offset[ip]; g < atom->level_offset[ip + 1]; g++)
        if (atom->level_num[g] == levnum) return g;
    return -1;
}

static double planck_bnu_input(double T, double nu) {
    if (!(T > 0.0) || !(nu > 0.0)) return 0.0;
    double x = H_PLANCK * nu / (K_BOLTZMANN * T);
    if (x > 700.0) return 0.0;
    return 2.0 * H_PLANCK * nu * nu * nu /
           (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT) / expm1(x);
}

static double tau_from_beta(double beta) {
    if (!(beta > 0.0) || beta >= 1.0) return 0.0;
    double lo = 0.0, hi = 1.0;
    while ((1.0 - exp(-hi)) / hi > beta && hi < 1.0e16) hi *= 2.0;
    for (int it = 0; it < 100; it++) {
        double mid = 0.5 * (lo + hi);
        double bm = (1.0 - exp(-mid)) / mid;
        if (bm > beta) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

static int load_plasma_cell(const char *run_dir, const char *model_dir, int shell,
                            PlasmaState *plasma, double *time_explosion) {
    char path[4096], line[2048];
    path_join(path, sizeof(path), run_dir, "lumina_plasma_state.csv");
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    (void)fgets(line,sizeof(line),fp);
    int found = 0;
    while (fgets(line, sizeof(line), fp)) {
        int s; double W, Tr, ne, Te;
        if (sscanf(line, "%d,%lf,%lf,%lf,%lf", &s, &W, &Tr, &ne, &Te) == 5 &&
            s == shell) {
            plasma->W[0] = W; plasma->T_rad[0] = Tr;
            plasma->n_electron[0] = ne; plasma->T_e[0] = Te;
            found = 1; break;
        }
    }
    fclose(fp);
    if (!found) return -1;

    path_join(path, sizeof(path), model_dir, "density.csv");
    fp = fopen(path, "r");
    if (!fp) return -1;
    (void)fgets(line,sizeof(line),fp);
    while (fgets(line, sizeof(line), fp)) {
        int s; double rho;
        if (sscanf(line, "%d,%lf", &s, &rho) == 2 && s == shell) {
            plasma->rho[0] = rho; break;
        }
    }
    fclose(fp);

    path_join(path, sizeof(path), model_dir, "geometry.csv");
    fp = fopen(path, "r");
    if (!fp) return -1;
    fgets(line, sizeof(line), fp);
    while (fgets(line, sizeof(line), fp)) {
        int s; double ri, ro, vi, vo;
        if (sscanf(line, "%d,%lf,%lf,%lf,%lf", &s, &ri, &ro, &vi, &vo) == 5 &&
            s == shell) {
            double ti = (vi > 0.0) ? ri / vi : 0.0;
            double to = (vo > 0.0) ? ro / vo : 0.0;
            *time_explosion = 0.5 * (ti + to);
            break;
        }
    }
    fclose(fp);
    return (*time_explosion > 0.0 && plasma->rho[0] > 0.0) ? 0 : -1;
}

static int load_ion_pops(const char *run_dir, int shell, AtomicData *atom) {
    char path[4096], line[1024];
    path_join(path, sizeof(path), run_dir, "lumina_ion_pops.csv");
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    fgets(line, sizeof(line), fp);
    int n = 0;
    while (fgets(line, sizeof(line), fp)) {
        int s, Z, stage; double pop;
        if (sscanf(line, "%d,%d,%d,%lf", &s, &Z, &stage, &pop) != 4 || s != shell)
            continue;
        int ip = find_ion(atom, Z, stage);
        if (ip >= 0) { atom->ion_number_density[ip] = pop; n++; }
    }
    fclose(fp);
    return n;
}

static int load_level_pops(const char *run_dir, int shell, AtomicData *atom,
                           NLTEConfig *nlte, double *bk) {
    char path[4096], line[2048];
    path_join(path, sizeof(path), run_dir, "lumina_levelpop.csv");
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    fgets(line, sizeof(line), fp);
    int n = 0;
    while (fgets(line, sizeof(line), fp)) {
        int s, Z, stage, lev, g, hs, nsp;
        double E, nk, ng, bkv;
        if (sscanf(line, "%d,%d,%d,%d,%lf,%d,%lf,%lf,%lf,%d,%d",
                   &s, &Z, &stage, &lev, &E, &g, &nk, &ng, &bkv, &hs, &nsp) != 11 ||
            s != shell) continue;
        int gl = find_global_level(atom, Z, stage, lev);
        if (gl < 0) continue;
        int nl = nlte->global_to_nlte_level[gl];
        if (nl >= 0) {
            nlte->nlte_level_populations[nl] = nk;
            bk[nl] = bkv;
            n++;
        }
    }
    fclose(fp);
    return n;
}

static int load_c1_field(const char *run_dir, int shell, int wanted_iter,
                         NLTEConfig *nlte) {
    char path[4096], line[2048], mode[32];
    double W[ORACLE_COARSE_BINS] = {0}, T[ORACLE_COARSE_BINS] = {0};
    char modes[ORACLE_COARSE_BINS][32] = {{0}};
    int seen[ORACLE_COARSE_BINS] = {0}, n = 0;
    path_join(path, sizeof(path), run_dir, "lumina_c1_bins.csv");
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    fgets(line, sizeof(line), fp);
    while (fgets(line, sizeof(line), fp)) {
        int it, s, b; double l0, l1, J, w, t;
        if (sscanf(line, "%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%31s",
                   &it, &s, &b, &l0, &l1, &J, &w, &t, mode) != 9 ||
            s != shell || it != wanted_iter ||
            b < 0 || b >= ORACLE_COARSE_BINS) continue;
        if (!seen[b]) {
            W[b] = w; T[b] = t; seen[b] = 1;
            snprintf(modes[b], sizeof(modes[b]), "%s", mode);
            n++;
        }
    }
    fclose(fp);
    if (n != ORACLE_COARSE_BINS) return -1;
    for (int f = 0; f < nlte->n_freq_bins; f++) {
        int c = (int)((long)f * ORACLE_COARSE_BINS / nlte->n_freq_bins);
        if (!seen[c] || !strcmp(modes[c], "degen")) return -2;
        double nu = nlte->nu_min * exp(((double)f + 0.5) * nlte->d_log_nu);
        nlte->J_nu[f] = (!strcmp(modes[c], "empty") || W[c] <= 0.0 || T[c] <= 0.0)
                      ? 0.0 : W[c] * planck_bnu_input(T[c], nu);
    }
    return n;
}

static int load_jbar(const char *run_dir, int shell, int wanted_iter,
                     OpacityState *opacity,
                     int *ion_complete) {
    char path[4096], line[2048], modebuf[32];
    path_join(path, sizeof(path), run_dir, "lumina_jbar_dump.csv");
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    if(seek_csv_iteration(fp,wanted_iter)){fclose(fp);return -1;}
    int n = 0;
    memset(ion_complete, 0, ORACLE_TARGETS * sizeof(*ion_complete));
    while (fgets(line, sizeof(line), fp)) {
        int it, s, Z, ion, line_id, count, mode;
        double lam, jbar, beta, B;
        if (sscanf(line, "%d,%d,%d,%d,%d,%lf,%lf,%d,%lf,%d,%lf",
                   &it, &s, &Z, &ion, &line_id, &lam, &jbar, &count,
                   &beta, &mode, &B) != 11) continue;
        if(it>wanted_iter)break;
        if(s != shell ||
            it != wanted_iter || line_id < 0 ||
            line_id >= opacity->n_lines) continue;
        (void)Z; (void)ion; (void)lam; (void)mode; (void)B; (void)modebuf;
        opacity->jbar_line[line_id] = jbar;
        opacity->jbar_count[line_id] = count;
        opacity->tau_sobolev[line_id] = tau_from_beta(beta);
        for (int q = 0; q < ORACLE_TARGETS; q++)
            if (target_Z[q] == Z && target_stage[q] == ion)
                ion_complete[q] = 1;
        n++;
    }
    fclose(fp);
    return n;
}

static int load_bf_rate_estimator(const char *run_dir, int shell,
                                  int wanted_iter, NLTEConfig *nlte,
                                  double *mc_J) {
    char path[4096], line[2048];
    path_join(path, sizeof(path), run_dir, "lumina_c2_bfr_dump.csv");
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    fgets(line, sizeof(line), fp);
    int n = 0;
    unsigned char *seen = (unsigned char *)calloc(
        (size_t)nlte->n_freq_bins, sizeof(unsigned char));
    if (!seen) { fclose(fp); return -1; }
    while (fgets(line, sizeof(line), fp)) {
        int it, s, b, count;
        double nu_mid, Jraw, bfr;
        if (sscanf(line, "%d,%d,%d,%lf,%lf,%lf,%d",
                   &it, &s, &b, &nu_mid, &Jraw, &bfr, &count) != 7 ||
            s != shell || it != wanted_iter ||
            b < 0 || b >= nlte->n_freq_bins) continue;
        double expect = nlte->nu_min *
            exp(((double)b + 0.5) * nlte->d_log_nu);
        if (fabs(nu_mid / expect - 1.0) > 1.0e-6 ||
            !isfinite(Jraw) || !isfinite(bfr) || count < 0) {
            free(seen); fclose(fp); return -2;
        }
        nlte->bf_rate_estimator[b] = bfr;
        mc_J[b] = Jraw;
        nlte->j_nu_count[b] = count;
        if (!seen[b]) { seen[b] = 1; n++; }
    }
    free(seen);
    fclose(fp);
    return n;
}

static int load_deposition_cell(const char *model_dir, int shell,
                                GammaDeposition *gamma_dep) {
    char path[4096], line[512];
    path_join(path, sizeof(path), model_dir, "deposition_cmfgen.csv");
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    fgets(line, sizeof(line), fp);
    int found = 0;
    while (fgets(line, sizeof(line), fp)) {
        int s; double h;
        if (sscanf(line, "%d,%lf", &s, &h) == 2 && s == shell) {
            gamma_dep->heating_rate[0] = h;
            found = 1;
            break;
        }
    }
    fclose(fp);
    if (found) gamma_deposition_compute_nonthermal(gamma_dep);
    return found ? 0 : -1;
}

static void csv_row(FILE *fp, int shell, const char *category,
                    const char *quantity, int Z, int stage,
                    const char *transition, double nu, double value,
                    const char *unit, const char *producer,
                    const char *status, const char *note) {
    fprintf(fp, "%d,%s,%s,%d,%d,%s,%.17e,",
            shell, category, quantity, Z, stage, transition ? transition : "", nu);
    if (isfinite(value)) fprintf(fp, "%.17e", value);
    fprintf(fp, ",%s,%s,%s,%s\n", unit ? unit : "", producer, status, note);
}

static void write_state_rows(FILE *fp, int shell, PlasmaState *plasma,
                             AtomicData *atom, NLTEConfig *nlte,
                             const double *bk) {
    csv_row(fp, shell, "state", "T_e", 0, -1, "", 0.0, plasma->T_e[0],
            "K", "input:lumina_plasma_state.csv", "available", "writer header verified");
    csv_row(fp, shell, "state", "n_e", 0, -1, "", 0.0, plasma->n_electron[0],
            "cm^-3", "input:lumina_plasma_state.csv", "available", "writer header verified");
    csv_row(fp, shell, "state", "T_rad", 0, -1, "", 0.0, plasma->T_rad[0],
            "K", "input:lumina_plasma_state.csv", "available", "writer header verified");
    csv_row(fp, shell, "state", "W", 0, -1, "", 0.0, plasma->W[0],
            "dimensionless", "input:lumina_plasma_state.csv", "available",
            "writer header verified");
    for (int q = 0; q < ORACLE_TARGETS; q++) {
        int ip = find_ion(atom, target_Z[q], target_stage[q]);
        double ntot = 0.0;
        for (int j = 0; j < atom->n_ion_pops; j++)
            if (atom->ion_pop_Z[j] == target_Z[q]) ntot += atom->ion_number_density[j];
        double ni = (ip >= 0) ? atom->ion_number_density[ip] : NAN;
        csv_row(fp, shell, "state", "n_ion", target_Z[q], target_stage[q], "",
                0.0, ni, "cm^-3", "input:lumina_ion_pops.csv",
                ip >= 0 ? "available" : "unavailable", "supplementary production writer");
        csv_row(fp, shell, "state", "ion_fraction", target_Z[q], target_stage[q], "",
                0.0, (ip >= 0 && ntot > 0.0) ? ni / ntot : NAN, "dimensionless",
                "input:lumina_ion_pops.csv", (ip >= 0 && ntot > 0.0) ?
                "available" : "unavailable",
                (ip >= 0 && ntot > 0.0)
                    ? "all recorded stages summed"
                    : "recorded element-stage census sums to zero in this frozen cell");

        int best = -1; double best_pop = -1.0;
        for (int n = 0; n < nlte->n_nlte_levels_total; n++) {
            int gl = nlte->nlte_to_global_level[n];
            if (atom->level_Z[gl] != target_Z[q] ||
                atom->level_ion[gl] != target_stage[q]) continue;
            double p = nlte->nlte_level_populations[n];
            if (p > 0.0 && isfinite(bk[n]) && bk[n] >= 0.0 && p > best_pop) {
                best_pop = p; best = n;
            }
        }
        char lev[64] = "";
        if (best >= 0) {
            int gl = nlte->nlte_to_global_level[best];
            snprintf(lev, sizeof(lev), "level%d", atom->level_num[gl]);
        }
        csv_row(fp, shell, "state", "b_k_representative",
                target_Z[q], target_stage[q], lev, 0.0,
                best >= 0 ? bk[best] : NAN, "dimensionless",
                "input:lumina_levelpop.csv", (best >= 0 && isfinite(bk[best])) ?
                "available" : "unavailable",
                best >= 0 ? "maximum-population recorded level" :
                "no positive-population level with a non-sentinel recorded b_k");
    }
}

static void write_replay_rows(FILE *fp, int shell, const FrozenReplay *r) {
    csv_row(fp, shell, "input", "rate_consume_iteration", 0, -1, "", 0.0,
            r->consume_iter, "iteration", "input:lumina_jbar_dump.csv",
            "available", "authoritative NLTE rate-assembly consumer block");
    csv_row(fp, shell, "input", "field_producer_iteration", 0, -1, "", 0.0,
            r->producer_iter, "iteration",
            "input:lumina_c1_bins.csv+lumina_c2_bfr_dump.csv",
            "available", "lagged field published for the next iteration");
    csv_row(fp, shell, "input", "producer_to_consumer_lag", 0, -1, "", 0.0,
            r->consume_iter - r->producer_iter, "iteration",
            "bench_frozen_oracle:epoch_contract", "available",
            "required value is exactly one");
    csv_row(fp, shell, "input", "c1_coarse_bins_loaded", 0, -1, "", 0.0,
            r->c1_bins, "count", "input:lumina_c1_bins.csv", "available",
            "selected producer iteration only");
    csv_row(fp, shell, "input", "jbar_rows_loaded", 0, -1, "", 0.0,
            r->jbar_rows, "count", "input:lumina_jbar_dump.csv", "available",
            "selected consumer iteration only");
    csv_row(fp, shell, "input", "raw_jbar_exact_ions", 0, -1, "", 0.0,
            r->jbar_exact_ions, "count", "input:lumina_jbar_dump.csv", "available",
            "remaining ions use an explicitly labelled C1 fallback replay");
    csv_row(fp, shell, "input", "thermal_tail_color_active", 0, -1, "", 0.0,
            r->tail_color_active, "boolean", "resolved_config+radeq_set_tail_color",
            "available", "inactive registrations are restored as NULL");
    csv_row(fp, shell, "input", "thermal_tri_response_active", 0, -1, "", 0.0,
            r->tri_response_active, "boolean", "resolved_config+radeq_set_tri_response",
            "available", "consumed only by the disabled coupled/global Newton path");
    csv_row(fp, shell, "input", "thermal_line_re_active", 0, -1, "", 0.0,
            r->line_re_active, "boolean", "resolved_config+radeq_set_line_re_source",
            "available", "inactive registrations are restored as NULL");
    csv_row(fp, shell, "input", "thermal_photoion_mc_active", 0, -1, "", 0.0,
            r->photoion_mc_active, "boolean",
            "resolved_config+plasma_set_photoion_mc_field", "available",
            r->photoion_mc_active
                ? "lagged raw MC J and occupancy restored from selected C2 producer block"
                : "production gate disabled; NULL is the exact registered state");
    csv_row(fp, shell, "input", "ma_line_destruct_terminals_global", 0, -1, "",
            0.0, (double)r->ma_terminals_global, "count",
            "input:stdout.log", "available",
            "global transport context only; not a per-cell volumetric rate");
    csv_row(fp, shell, "input", "ma_line_destruct_destroyed_global", 0, -1, "",
            0.0, (double)r->ma_destroyed_global, "count",
            "input:stdout.log", "available",
            "global transport context only; per-shell energy ledger was not archived");
}

static int run_cell(const char *run_dir, const char *out_dir, int shell,
                    AtomicData *atom, PlasmaState *plasma, NLTEConfig *nlte,
                    OpacityState *opacity, BFOpacity *bf, double *bk,
                    const char *model_dir, GammaDeposition *gamma_dep) {
    memset(atom->ion_number_density, 0,
           (size_t)atom->n_ion_pops * sizeof(double));
    memset(nlte->nlte_level_populations, 0,
           (size_t)nlte->n_nlte_levels_total * sizeof(double));
    memset(opacity->jbar_line, 0, (size_t)opacity->n_lines * sizeof(double));
    memset(opacity->jbar_count, 0, (size_t)opacity->n_lines * sizeof(int));
    memset(opacity->tau_sobolev, 0, (size_t)opacity->n_lines * sizeof(double));
    memset(nlte->J_nu, 0, (size_t)nlte->n_freq_bins * sizeof(double));
    memset(nlte->bf_rate_estimator, 0,
           (size_t)nlte->n_freq_bins * sizeof(double));
    memset(nlte->j_nu_count, 0, (size_t)nlte->n_freq_bins * sizeof(int));
    for (int n = 0; n < nlte->n_nlte_levels_total; n++) bk[n] = NAN;
    double texp = 0.0;
    int jbar_complete[ORACLE_TARGETS];
    FrozenReplay replay;
    memset(&replay, 0, sizeof(replay));
    replay.consume_iter = find_consume_iteration(run_dir, shell);
    replay.producer_iter = replay.consume_iter - 1;
    replay.tail_color_active = env_enabled("LUMINA_A4_TAIL_COLOR");
    replay.tri_response_active = env_enabled("LUMINA_COUPLED_NEWTON");
    replay.line_re_active = env_enabled("LUMINA_RADEQ_LINE_RE");
    replay.photoion_mc_active = env_enabled("LUMINA_COEVOLVE_PHOTOION_MC");
    replay.photoion_mc_alpha = getenv("LUMINA_COEVOLVE_PHOTOION_ALPHA")
        ? atof(getenv("LUMINA_COEVOLVE_PHOTOION_ALPHA")) : 0.5;
    double *mc_J = (double *)calloc((size_t)nlte->n_freq_bins, sizeof(double));
    if (!mc_J) return -1;
    if (replay.consume_iter <= 0 ||
        replay.tail_color_active || replay.tri_response_active ||
        replay.line_re_active) {
        fprintf(stderr,
                "s%d: active thermal registration lacks an archived Phase-1.6 "
                "snapshot (consume=%d tail=%d tri=%d line=%d)\n",
                shell, replay.consume_iter, replay.tail_color_active,
                replay.tri_response_active, replay.line_re_active);
        free(mc_J);
        return -1;
    }
    if (load_plasma_cell(run_dir, model_dir, shell, plasma, &texp) ||
        load_ion_pops(run_dir, shell, atom) <= 0 ||
        load_level_pops(run_dir, shell, atom, nlte, bk) <= 0 ||
        (replay.c1_bins = load_c1_field(
            run_dir, shell, replay.producer_iter, nlte)) != ORACLE_COARSE_BINS ||
        (replay.jbar_rows = load_jbar(
            run_dir, shell, replay.consume_iter, opacity,
            jbar_complete)) <= 0 ||
        (replay.c2_bins = load_bf_rate_estimator(
            run_dir, shell, replay.producer_iter, nlte, mc_J)) !=
                nlte->n_freq_bins ||
        load_deposition_cell(model_dir, shell, gamma_dep) ||
        load_ma_global_counts(run_dir, replay.producer_iter, &replay)) {
        fprintf(stderr, "s%d: frozen input load failed\n", shell);
        free(mc_J);
        return -1;
    }
    for (int q = 0; q < ORACLE_TARGETS; q++)
        replay.jbar_exact_ions += jbar_complete[q] ? 1 : 0;
    load_ma_cell_rate(run_dir, replay.producer_iter, shell, &replay);
    if (replay.consume_iter - replay.producer_iter != 1) {
        fprintf(stderr, "s%d: invalid producer/consumer lag %d -> %d\n",
                shell, replay.producer_iter, replay.consume_iter);
        free(mc_J);
        return -1;
    }

    /* Restore the registration state actually consumed before the production
     * thermal call.  The parity50 resolved configuration disables tail color,
     * global-Newton tri-response, line-RE, and lagged photoion MC; NULL is
     * therefore an exact state, not a guessed zero.  A future active photoion
     * configuration is recoverable from C2's raw-J/count columns. */
    radeq_set_tail_color(NULL, 0);
    radeq_set_tri_response(NULL, NULL, NULL, 0, 0);
    radeq_set_line_re_source(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                             NULL, NULL, NULL, 0, 0);
    if (replay.photoion_mc_active)
        plasma_set_photoion_mc_field(mc_J, replay.photoion_mc_alpha, 1,
                                     nlte->n_freq_bins, nlte->j_nu_count);
    else
        plasma_set_photoion_mc_field(NULL, 0.0, 0, 0, NULL);

    lumina_oracle_prepare_partitions(atom, plasma, 1);
    if (nlte_precompute_within_sl_frac_checked(
            nlte, atom, plasma, 1) != 0) {
        free(mc_J);
        return -1;
    }
    /* Wave-3 shadow observer: use the recorded shell label while the frozen
     * replay stores that cell at local index zero.  No-op unless the EW gate is
     * explicitly armed; the legacy oracle rows remain authoritative. */
    nlte->current_iter = replay.consume_iter;
    int baseline_sigma_loaded=atom->cmfgen_loaded;
    int ew_failed = 0;
    if (g_ew_sigma_available) atom->cmfgen_loaded=1;
    if (nlte_element_wide_matches(16, shell))
        if (nlte_element_wide_run_labeled(nlte, atom, plasma, opacity,
                                           16, 0, shell, texp, gamma_dep) < 0)
            ew_failed = 1;
    if (nlte_element_wide_matches(26, shell))
        if (nlte_element_wide_run_labeled(nlte, atom, plasma, opacity,
                                           26, 0, shell, texp, gamma_dep) < 0)
            ew_failed = 1;
    atom->cmfgen_loaded=baseline_sigma_loaded;
    if (ew_failed) {
        fprintf(stderr, "s%d: element-wide observer failed\n", shell);
        free(mc_J);
        return -1;
    }

    char path[4096];
    snprintf(path, sizeof(path), "%s/lumina_oracle_cell_s%d.csv", out_dir, shell);
    FILE *fp = fopen(path, "w");
    if (!fp) { perror(path); return -1; }
    fprintf(fp, "shell,category,quantity,Z,stage,transition,frequency_Hz,value,"
                "unit,producer,status,note\n");
    write_state_rows(fp, shell, plasma, atom, nlte, bk);
    write_replay_rows(fp, shell, &replay);

    lumina_oracle_trace_begin(fp, shell);
    lumina_oracle_set_frozen_input_coverage(
        replay.c2_bins, nlte->n_freq_bins, jbar_complete, ORACLE_TARGETS);
    lumina_oracle_set_transport_thermal(replay.ma_cell_heating,
                                        replay.ma_cell_rate_available);
    compute_bf_opacity(bf, atom, plasma, 1);

    int oracle_pairs[NLTE_PAIR_COUNT][2];
    const char *oracle_pair_names[NLTE_PAIR_COUNT];
    int oracle_n_pairs=nlte_get_pairs(oracle_pairs,oracle_pair_names);
    for (int p = 0; p < oracle_n_pairs; p++) {
        int lo=oracle_pairs[p][0],hi=oracle_pairs[p][1];
        int ss = nlte->nlte_ion_super_offset[lo];
        int N = nlte->nlte_ion_super_offset[hi + 1] - ss;
        if (N <= 0) continue;
        double *A = (double *)calloc((size_t)N * N, sizeof(double));
        double *rhs = (double *)calloc((size_t)N, sizeof(double));
        if (!A || !rhs) { free(A); free(rhs); fclose(fp); return -1; }
        nlte_assemble_rate_matrix(nlte, atom, plasma, opacity, lo, hi, 0, texp,
                                  A, rhs, N, NULL, NULL, -1);
        free(A); free(rhs);
    }
    compute_radiative_equilibrium_te(plasma, gamma_dep, nlte, atom, opacity,
                                     texp, 1);
    lumina_oracle_trace_end();
    plasma_set_photoion_mc_field(NULL, 0.0, 0, 0, NULL);
    free(mc_J);
    fclose(fp);
    printf("[ORACLE] wrote %s\n", path);
    return 0;
}

int main(int argc, char **argv) {
    const char *run_dir = (argc > 1) ? argv[1] : "logs/coevolve_consume_parity50";
    const char *model_dir = (argc > 2) ? argv[2] :
        "data/tardis_reference_toy06_19p48d_sivcaiv";
    const char *out_dir = (argc > 3) ? argv[3] :
        "validation/gate_b_dual_oracle/phase1_6";
    if (mkdir("validation/gate_b_dual_oracle", 0775) && errno != EEXIST) {
        perror("validation/gate_b_dual_oracle"); return 1;
    }
    if (mkdir(out_dir, 0775) && errno != EEXIST) { perror(out_dir); return 1; }

    int nenv = load_resolved_environment(run_dir);
    if (nenv <= 0) {
        fprintf(stderr, "cannot recover resolved environment from %s/stdout.log\n", run_dir);
        return 1;
    }
    printf("[ORACLE] recovered %d production settings; CPU single-thread path\n", nenv);

    AtomicData atom;
    PlasmaState plasma; memset(&plasma, 0, sizeof(plasma));
    OpacityState opacity; memset(&opacity, 0, sizeof(opacity));
    NLTEConfig nlte; memset(&nlte, 0, sizeof(nlte));
    BFOpacity bf; memset(&bf, 0, sizeof(bf));
    if (load_atomic_data(&atom, model_dir, 1) != 0) return 1;
    if (nlte_element_wide_enabled()) {
        char sigma_path[4096];
        snprintf(sigma_path, sizeof(sigma_path), "%s/cmfgen_sigma_bf.bin", model_dir);
        (void)load_cmfgen_sigma_bf(&atom, sigma_path);
        g_ew_sigma_available=atom.cmfgen_loaded;
        atom.cmfgen_loaded=0; /* observer baseline remains the OFF arithmetic */
    }

    plasma.n_shells = 1;
    plasma.W = (double *)calloc(1, sizeof(double));
    plasma.T_rad = (double *)calloc(1, sizeof(double));
    plasma.T_e = (double *)calloc(1, sizeof(double));
    plasma.n_electron = (double *)calloc(1, sizeof(double));
    plasma.rho = (double *)calloc(1, sizeof(double));
    plasma.T_e_T_rad_ratio = 1.0;

    opacity.n_lines = atom.n_lines; opacity.n_shells = 1;
    opacity.line_list_nu = atom.line_nu;
    opacity.tau_sobolev = (double *)calloc((size_t)atom.n_lines, sizeof(double));
    opacity.jbar_line = (double *)calloc((size_t)atom.n_lines, sizeof(double));
    opacity.jbar_count = (int *)calloc((size_t)atom.n_lines, sizeof(int));
    if (nlte_init(&nlte, &atom, &opacity, 1) != 0) return 1;
    double *bk = (double *)malloc((size_t)nlte.n_nlte_levels_total * sizeof(double));
    bf_opacity_init(&bf, 1);
    bf_set_nlte_pops(&nlte);
    GammaDeposition gamma_dep;
    gamma_deposition_init(&gamma_dep, 1);

    int rc = 0;
    const int shells[] = {0, 8, 43};
    const char *only_shell_env = getenv("LUMINA_FROZEN_ORACLE_ONLY_SHELL");
    if (only_shell_env && *only_shell_env) {
        int only_shell = atoi(only_shell_env);
        if (only_shell != 0 && only_shell != 8 && only_shell != 20 &&
            only_shell != 43) {
            fprintf(stderr, "invalid LUMINA_FROZEN_ORACLE_ONLY_SHELL=%s\n",
                    only_shell_env);
            rc = 1;
        } else if (run_cell(run_dir, out_dir, only_shell, &atom, &plasma,
                            &nlte, &opacity, &bf, bk, model_dir, &gamma_dep)) {
            rc = 1;
        }
    } else {
        for (size_t i = 0; i < sizeof(shells) / sizeof(shells[0]); i++)
            if (run_cell(run_dir, out_dir, shells[i], &atom, &plasma, &nlte,
                         &opacity, &bf, bk, model_dir, &gamma_dep)) rc = 1;
    }
    /* Wave-3 reverse-axis preparation: keep the authoritative OFF oracle at
     * its original three cells, but make s20 available in an explicitly armed
     * S-only shadow run.  s0 Fe is already one of the three authority cells. */
    if (!only_shell_env && nlte_element_wide_enabled() &&
        nlte_element_wide_matches(16, 20))
        if (run_cell(run_dir, out_dir, 20, &atom, &plasma, &nlte,
                     &opacity, &bf, bk, model_dir, &gamma_dep)) rc = 1;

    gamma_deposition_free(&gamma_dep);
    free(bk);
    bf_opacity_free(&bf);
    nlte_free(&nlte);
    free(opacity.tau_sobolev); free(opacity.jbar_line); free(opacity.jbar_count);
    free(plasma.W); free(plasma.T_rad); free(plasma.T_e);
    free(plasma.n_electron); free(plasma.rho);
    free_atomic_data(&atom);
    return rc;
}
