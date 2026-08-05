/* A2-05 L-1bf gate fixture: deterministic (ORACLE_INPUT) or MC (CHAIN) commit
 * of a supplied J field into the canonical RadiationField, then per-level
 * Gamma through the SAME shared entry point production uses
 * (bf_rate_gamma_legacy_grid) -- byte-identical arithmetic, no forked math.
 *
 * Input (little-endian binary):
 *   magic   "A205IN01" (8)
 *   mode    u64   1 = deterministic, 2 = MC path-length
 *   n_shells u64, generation u64, epoch f64
 *   v_inner[n_shells] f64, v_outer[n_shells] f64
 *   mode 1: J[n_shells*4000] f64, validity[n_shells*4000] i32
 *   mode 2: raw_path_length[n_shells*4000] f64, counts[n_shells*4000] u64,
 *           volume[n_shells] f64, time_simulation f64
 *   nfb u64, nu_min f64, d_log_nu f64
 *   n_levels u64
 *   per level: nu_thresh f64, sigma[nfb] f64
 *
 * Output: one text line per (level, shell):
 *   GAMMA <level> <shell> <state> <gamma> <w_miss> <sample_count> <poisson_var>
 */
#include "bf_rate_jnu.h"
#include "radiation_field.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_exact(FILE *f, void *p, size_t n)
{
    return fread(p, 1, n, f) == n ? 0 : -1;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s INPUT\n", argv[0]);
        return 2;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) return 2;
    char magic[8];
    uint64_t mode, n_shells_u, generation, nfb_u, n_levels;
    double epoch;
    if (read_exact(in, magic, 8) || memcmp(magic, "A205IN01", 8) != 0 ||
        read_exact(in, &mode, 8) || read_exact(in, &n_shells_u, 8) ||
        read_exact(in, &generation, 8) || read_exact(in, &epoch, 8) ||
        (mode != 1 && mode != 2) || n_shells_u == 0 || n_shells_u > 1000)
        return 2;
    size_t n_shells = (size_t)n_shells_u;
    size_t cells = n_shells * (size_t)LUMINA_RADFIELD_N_BINS;

    double *v_inner = malloc(n_shells * sizeof(double));
    double *v_outer = malloc(n_shells * sizeof(double));
    if (!v_inner || !v_outer ||
        read_exact(in, v_inner, n_shells * sizeof(double)) ||
        read_exact(in, v_outer, n_shells * sizeof(double)))
        return 2;

    RadiationFieldOwner owner;
    if (radiation_field_owner_init(&owner, n_shells) != 0) return 2;

    RadiationFieldCommitRequest req;
    memset(&req, 0, sizeof(req));
    req.generation = generation;
    req.epoch = epoch;
    req.n_shells = n_shells;
    req.v_inner = v_inner;
    req.v_outer = v_outer;
    req.source_n_bins = LUMINA_RADFIELD_N_BINS;

    double *values = NULL, *raw = NULL, *volume = NULL;
    RadiationFieldValidityState *validity = NULL;
    int32_t *validity32 = NULL;
    uint64_t *counts = NULL;
    if (mode == 1) {
        values = malloc(cells * sizeof(double));
        validity32 = malloc(cells * sizeof(int32_t));
        validity = malloc(cells * sizeof(*validity));
        if (!values || !validity32 || !validity ||
            read_exact(in, values, cells * sizeof(double)) ||
            read_exact(in, validity32, cells * sizeof(int32_t)))
            return 2;
        for (size_t i = 0; i < cells; ++i)
            validity[i] = (RadiationFieldValidityState)validity32[i];
        req.provenance_kind = RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
        req.producer = "A2_05_L1BF_ORACLE_INPUT";
        req.source_frequency_bin_edges = owner.field.frequency_bin_edges.values;
        req.source_J_nu = values;
        req.source_validity = validity;
        req.statistic_kind = RADIATION_FIELD_DETERMINISTIC;
    } else {
        raw = malloc(cells * sizeof(double));
        counts = malloc(cells * sizeof(uint64_t));
        volume = malloc(n_shells * sizeof(double));
        double time_simulation;
        if (!raw || !counts || !volume ||
            read_exact(in, raw, cells * sizeof(double)) ||
            read_exact(in, counts, cells * sizeof(uint64_t)) ||
            read_exact(in, volume, n_shells * sizeof(double)) ||
            read_exact(in, &time_simulation, sizeof(double)))
            return 2;
        req.provenance_kind = RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH;
        req.producer = "A2_05_L1BF_CHAIN";
        req.statistic_kind = RADIATION_FIELD_ESTIMATOR_COUNT;
        req.raw_path_length = raw;
        req.source_count = counts;
        req.volume = volume;
        req.time_simulation = time_simulation;
        uint64_t total = 0;
        for (size_t i = 0; i < cells; ++i) total += counts[i];
        req.contribution_count = total;
    }
    if (radiation_field_commit(&owner, &req) != 0) {
        fprintf(stderr, "A2_05_L1BF commit FAILED\n");
        return 1;
    }

    RadiationFieldView view;
    int st = radiation_field_read_view(&owner, epoch, n_shells, generation, &view);
    if (st != RADIATION_FIELD_VIEW_OK) {
        fprintf(stderr, "A2_05_L1BF read_view FAILED status=%d\n", st);
        return 1;
    }

    uint64_t nu64;
    double nu_min, d_log_nu;
    if (read_exact(in, &nfb_u, 8) || read_exact(in, &nu_min, 8) ||
        read_exact(in, &d_log_nu, 8) || read_exact(in, &n_levels, 8) ||
        nfb_u == 0 || nfb_u > 100000 || n_levels > 1000000)
        return 2;
    (void)nu64;
    int nfb = (int)nfb_u;
    double *sigma_row = malloc((size_t)nfb * sizeof(double));
    double *node_nu = malloc(2 * (size_t)nfb * sizeof(double));
    double *node_sg = malloc(2 * (size_t)nfb * sizeof(double));
    if (!sigma_row || !node_nu || !node_sg) return 2;

    printf("A2_05_L1BF_FIXTURE mode=%llu shells=%zu generation=%llu levels=%llu\n",
           (unsigned long long)mode, n_shells, (unsigned long long)generation,
           (unsigned long long)n_levels);
    for (uint64_t lev = 0; lev < n_levels; ++lev) {
        double nu_thresh;
        if (read_exact(in, &nu_thresh, 8) ||
            read_exact(in, sigma_row, (size_t)nfb * sizeof(double)))
            return 2;
        for (size_t s = 0; s < n_shells; ++s) {
            BfRateResult r;
            if (bf_rate_gamma_legacy_grid(&view, s, nfb, nu_min, d_log_nu,
                                          sigma_row, 0.0, nu_thresh,
                                          node_nu, node_sg, &r) != 0) {
                fprintf(stderr, "A2_05_L1BF integrator rc!=0 lev=%llu s=%zu\n",
                        (unsigned long long)lev, s);
                return 1;
            }
            printf("GAMMA %llu %zu %d %.17e %.6e %llu %.17e\n",
                   (unsigned long long)lev, s, (int)r.state, r.gamma, r.w_miss,
                   (unsigned long long)r.sample_count, r.gamma_poisson_var);
        }
    }
    if (fgetc(in) != EOF) return 2;
    printf("A2_05_L1BF_FIXTURE DONE\n");
    return 0;
}
