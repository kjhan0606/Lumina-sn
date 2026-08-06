/* lumina_nlte_assemble.cu — GPU port of the dominant bound-bound (bb) radiative
 * + collisional assembly loop in nlte_assemble_rate_matrix (lumina_plasma.c).
 *
 * Profiling: assemble 24.3s (99.6%) vs GPU solve 0.09s (0.4%); the 99.6% is the
 * per-(pair,shell) loop over all n_lines (~137k) that filters by pair and adds
 * each in-pair line's Einstein bb rates (R_lu=B_lu*J, R_ul=A_ul+B_ul*J) plus the
 * van Regemorter / Axelrod collisional pair into the rate matrix.
 *
 * Design
 * ------
 *  - The CPU still assembles EVERYTHING EXCEPT the bb loop (bf/rec from the
 *    prebaked GPU lookup, charge-exchange, dielectronic recomb, top-stage drain,
 *    time-dep, conservation rows). The driver sets nlte_assemble_set_skip_bb(1)
 *    so the CPU bb loop is skipped; this kernel adds the bb+collisional terms on
 *    top of the CPU-assembled remainder.
 *  - Per-line quantities that are shell-INDEPENDENT (pair map, the lower/upper
 *    NLTE full-level indices, dE, g_lo, g_up, f_lu, nu, B_lu, B_ul, A_ul) are
 *    precomputed ONCE on the host in init (this replaces the CPU per-line level
 *    scan, itself a big part of the cost) and uploaded to the device.
 *  - Kernel parallelization axis: one thread per (line, shell). gridDim.y =
 *    n_shells, x covers n_lines. Each thread filters by the pair, maps the FL
 *    indices to super-level solve indices (SOLVE_OF), computes the four matrix
 *    contributions and atomicAdds them into the column-major device matrix
 *    d[s*N*N + j*N + i] (ACM(i,j) = d[j*N+i]).
 *  - All arithmetic is FP64 (matching the CPU); atomicAdd(double) only reorders
 *    the summation, so the match to the CPU is at round-off (~1e-12), far better
 *    than the 1e-5 tolerance.
 *
 * J_bar sources supported (mirror nlte_assemble_rate_matrix exactly):
 *   - default binned ambient field  J = nlte_get_J_at_nu(shell, nu_line)
 *   - LUMINA_NLTE_DILUTE_FIELD       J = W(s) * B(nu_line, T_R)
 * Collisions: LUMINA_NLTE_COLL_FIX (Upsilon form) or legacy van Regemorter /
 *   Axelrod, plus LUMINA_NLTE_COLL_FLOOR. Within-super-level Boltzmann fractions
 *   (FRAC_OF) are applied identically.
 *
 * NOT ported (caller must stay on CPU; nlte_assemble_gpu_supported()==0): the
 * sealed/experimental bb modes LUMINA_NLTE_JBAR_POPS, LUMINA_MALI,
 * LUMINA_NLTE_JEQB, LUMINA_JBAR_SRC_BINNED, LUMINA_CMF_LINERES_CONSUME. The
 * per-line LUMINA_NLTE_BUDGET_DUMP CSV is also CPU-only (diagnostic). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C" {
#include "lumina.h"
#include "gpu_radiation_field_contract.h"
int nlte_coll_fix_enabled(void);   /* defined in lumina_plasma.c */
}

static GpuRadiationFieldCounters g_a2_12_nlte_asm_counters;

#define ACUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                    \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
} while(0)

/* Collisional constants (mirror lumina_plasma.c). */
#define A_VAN_REGEMORTER_COEFF  2.16e-6
#define A_AXELROD_OMEGA         1.0

typedef struct {
    int initialized;
    int n_shells;
    int n_lines;
    int n_freq;
    int n_nlte_levels_total;

    /* Per-line packed arrays (shell-independent), uploaded once. */
    int    *d_map;        /* nlte_line_map (pair index) or sentinel <0 to skip  */
    int    *d_fl_lo;      /* lower-level NLTE full-level idx (fl), or -1 to skip */
    int    *d_fl_up;      /* upper-level NLTE full-level idx (fl), or -1 to skip */
    double *d_dE;         /* |E_up - E_lo| in erg                                */
    int    *d_g_lo;
    int    *d_g_up;
    double *d_f_lu;
    double *d_nu;
    double *d_B_lu;
    double *d_B_ul;
    double *d_A_ul;

    /* Mapping + per-shell varying state (re-uploaded each CE iter). */
    int    *d_fl_to_super;            /* [n_nlte_levels_total]                   */
    double *d_within_sl_frac;         /* [n_nlte_levels_total * n_shells]        */
    double *d_J_nu;                   /* [n_shells * n_freq] (row-major as CPU)  */
    double *d_T_e;                    /* [n_shells]                              */
    double *d_n_e;                    /* [n_shells]                              */
    double *d_W;                      /* [n_shells]                              */

    /* Frequency-grid params for the binned J lookup. */
    double nu_min, nu_max, d_log_nu;

    /* Scratch device matrix buffer [n_shells * maxN * maxN], grown on demand. */
    double *d_mat;
    size_t  d_mat_elems;

    /* Per-shell active mask (dead-pair skip mirror). */
    int    *d_active;

    /* Cached config (read at init / refresh; constant during a solve). */
    int    coll_fix;
    double coll_floor;
    int    dilute_on;
    double dilute_TR;
} NLTEAsmState;

static NLTEAsmState g_asm = {0};

/* ---- device helpers (mirror planck_bnu / nlte_get_J_at_nu) ---- */
__device__ static inline double a_planck_bnu(double T, double nu) {
    double x = H_PLANCK * nu / (K_BOLTZMANN * T);
    if (x > 500.0) return 0.0;
    return (2.0 * H_PLANCK * nu * nu * nu /
            (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT)) / (exp(x) - 1.0);
}

__device__ static inline double a_J_at_nu(const double *J_nu, int s, int n_freq,
                                          double nu, double nu_min, double nu_max,
                                          double d_log_nu) {
    if (nu <= nu_min || nu >= nu_max) return 1e-30;
    double log_ratio = log(nu / nu_min);
    int bin = (int)(log_ratio / d_log_nu);
    if (bin < 0) bin = 0;
    if (bin >= n_freq) bin = n_freq - 1;
    return J_nu[(size_t)s * n_freq + bin];
}

/* ---- the assembly kernel: one thread per (line, shell) ---- */
__global__ void nlte_assemble_bb_kernel(
        double *d_mat, int N, int n_shells, int n_lines,
        int pair_lo, int pair_hi, int super_start, int n_lo_super,
        const int *d_map, const int *d_fl_lo, const int *d_fl_up,
        const double *d_dE, const int *d_g_lo, const int *d_g_up,
        const double *d_f_lu, const double *d_nu,
        const double *d_B_lu, const double *d_B_ul, const double *d_A_ul,
        const int *d_fl_to_super, const double *d_within_sl_frac,
        const double *d_J_nu, int n_freq,
        const double *d_T_e, const double *d_n_e, const double *d_W,
        double nu_min, double nu_max, double d_log_nu,
        int coll_fix, double coll_floor, int dilute_on, double dilute_TR,
        const int *d_active)
{
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    int s    = blockIdx.y;
    if (line >= n_lines || s >= n_shells) return;
    if (!d_active[s]) return;

    int map = d_map[line];
    if (map < pair_lo || map > pair_hi) return;   /* line not in this pair */

    int fl_lo = d_fl_lo[line];
    int fl_up = d_fl_up[line];
    if (fl_lo < 0 || fl_up < 0) return;            /* unmapped level -> skip */

    int i_lo = d_fl_to_super[fl_lo] - super_start;
    int i_up = d_fl_to_super[fl_up] - super_start;
    if (i_lo < 0 || i_lo >= N || i_up < 0 || i_up >= N) return;

    double T_e = d_T_e[s];
    double n_e = d_n_e[s];
    double nu  = d_nu[line];

    /* J_bar source */
    double J_line;
    if (dilute_on) {
        J_line = d_W[s] * a_planck_bnu(dilute_TR, nu);
    } else {
        J_line = a_J_at_nu(d_J_nu, s, n_freq, nu, nu_min, nu_max, d_log_nu);
    }

    double B_lu = d_B_lu[line];
    double B_ul = d_B_ul[line];
    double A_ul = d_A_ul[line];
    double R_absorb = B_lu * J_line;   /* beta_use = 1 in the supported domain */
    double R_stim   = B_ul * J_line;
    double R_spont  = A_ul;

    /* Collisional pair */
    double dE   = d_dE[line];
    int    g_lo = d_g_lo[line];
    int    g_up = d_g_up[line];
    double f_lu = d_f_lu[line];
    double C_up = 0.0, C_down = 0.0;
    if (T_e > 0.0 && dE > 0.0 && g_lo > 0 && g_up > 0) {
        double exp_factor = exp(-dE / (K_BOLTZMANN * T_e));
        if (coll_fix) {
            double dE_eV = dE / EV_TO_ERG;
            double ry_over_dE = 13.6057 / (dE_eV > 0.0 ? dE_eV : 1e30);
            double gbar = 0.2;
            double ups_vr = 14.5 * f_lu * ry_over_dE * ry_over_dE * gbar;
            double ups = (ups_vr > A_AXELROD_OMEGA) ? ups_vr : A_AXELROD_OMEGA;
            C_up   = n_e * 8.629e-6 / (g_lo * sqrt(T_e)) * ups * exp_factor;
            C_down = n_e * 8.629e-6 / (g_up * sqrt(T_e)) * ups;
        } else {
            if (f_lu > 1e-10) {
                C_up = A_VAN_REGEMORTER_COEFF * n_e * f_lu *
                       exp_factor / (g_lo * sqrt(T_e)) * 0.2;
            } else {
                C_up = 8.63e-6 * n_e * A_AXELROD_OMEGA *
                       exp_factor / (g_lo * sqrt(T_e));
            }
            C_down = C_up * ((double)g_lo / (double)g_up) *
                     exp(dE / (K_BOLTZMANN * T_e));
        }
    }
    if (coll_floor > 0.0 && T_e > 0.0 && g_lo > 0 && g_up > 0) {
        double cd_min = coll_floor * A_ul;
        if (C_down < cd_min) {
            C_down = cd_min;
            C_up = cd_min * ((double)g_up / (double)g_lo) *
                   exp(-dE / (K_BOLTZMANN * T_e));
        }
    }

    double total_up   = R_absorb + C_up;
    double total_down = R_stim + R_spont + C_down;

    double f_lo = d_within_sl_frac[(size_t)fl_lo * n_shells + s];
    double f_up = d_within_sl_frac[(size_t)fl_up * n_shells + s];

    /* Column-major: ACM(i,j) = d_mat[base + j*N + i]. Identical to the CPU. */
    size_t base = (size_t)s * N * N;
    atomicAdd(&d_mat[base + (size_t)i_lo * N + i_up],  total_up   * f_lo); /* (i_up,i_lo) */
    atomicAdd(&d_mat[base + (size_t)i_up * N + i_lo],  total_down * f_up); /* (i_lo,i_up) */
    atomicAdd(&d_mat[base + (size_t)i_lo * N + i_lo], -total_up   * f_lo); /* (i_lo,i_lo) */
    atomicAdd(&d_mat[base + (size_t)i_up * N + i_up], -total_down * f_up); /* (i_up,i_up) */
}

/* find_ion_pop_idx mirror (lumina_plasma.c is static). */
static int asm_find_ion_pop_idx(AtomicData *atom, int Z, int ion_stage) {
    for (int e = 0; e < atom->n_elements; e++) {
        if (atom->element_Z[e] != Z) continue;
        int start = atom->elem_ion_offset[e];
        int end   = atom->elem_ion_offset[e + 1];
        for (int ip = start; ip < end; ip++)
            if (atom->ion_pop_stage[ip] == ion_stage) return ip;
    }
    return -1;
}

extern "C" int nlte_assemble_gpu_init(NLTEConfig *nlte, AtomicData *atom,
                                      OpacityState *opacity, int n_shells)
{
    if (g_asm.initialized) return 0;
    if (!nlte || !nlte->enabled || n_shells <= 0) return -1;

    int n_lines = opacity->n_lines;
    int n_freq  = nlte->n_freq_bins;
    int n_lev   = nlte->n_nlte_levels_total;

    g_asm.n_shells = n_shells;
    g_asm.n_lines  = n_lines;
    g_asm.n_freq   = n_freq;
    g_asm.n_nlte_levels_total = n_lev;
    g_asm.nu_min   = nlte->nu_min;
    g_asm.nu_max   = nlte->nu_max;
    g_asm.d_log_nu = nlte->d_log_nu;

    /* ---- Host precompute of the per-line shell-independent packed arrays.
     * Mirrors the level-scan in nlte_assemble_rate_matrix, done ONCE. ---- */
    int    *h_map   = (int    *)malloc((size_t)n_lines * sizeof(int));
    int    *h_fl_lo = (int    *)malloc((size_t)n_lines * sizeof(int));
    int    *h_fl_up = (int    *)malloc((size_t)n_lines * sizeof(int));
    double *h_dE    = (double *)malloc((size_t)n_lines * sizeof(double));
    int    *h_g_lo  = (int    *)malloc((size_t)n_lines * sizeof(int));
    int    *h_g_up  = (int    *)malloc((size_t)n_lines * sizeof(int));
    double *h_f_lu  = (double *)malloc((size_t)n_lines * sizeof(double));
    double *h_nu    = (double *)malloc((size_t)n_lines * sizeof(double));
    double *h_B_lu  = (double *)malloc((size_t)n_lines * sizeof(double));
    double *h_B_ul  = (double *)malloc((size_t)n_lines * sizeof(double));
    double *h_A_ul  = (double *)malloc((size_t)n_lines * sizeof(double));
    if (!h_map || !h_fl_lo || !h_fl_up || !h_dE || !h_g_lo || !h_g_up ||
        !h_f_lu || !h_nu || !h_B_lu || !h_B_ul || !h_A_ul) {
        fprintf(stderr, "[NLTE-ASM] host precompute alloc failed\n");
        return -1;
    }

    int n_mapped = 0;
    for (int line = 0; line < n_lines; line++) {
        h_map[line]   = nlte->nlte_line_map[line];
        h_fl_lo[line] = -1; h_fl_up[line] = -1;
        h_dE[line] = 0.0; h_g_lo[line] = 0; h_g_up[line] = 0;
        h_f_lu[line] = atom->line_f_lu[line];
        h_nu[line]   = atom->line_nu[line];
        h_B_lu[line] = atom->line_B_lu[line];
        h_B_ul[line] = atom->line_B_ul[line];
        h_A_ul[line] = atom->line_A_ul[line];

        int map = h_map[line];
        if (map < 0) continue;   /* never in any pair */

        int ip = asm_find_ion_pop_idx(atom, atom->line_atomic_number[line],
                                      atom->line_ion_number[line]);
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
        if (fl_lo_g < 0 || fl_up_g < 0) continue;  /* non-NLTE level -> skip */

        h_fl_lo[line] = fl_lo_g;
        h_fl_up[line] = fl_up_g;
        h_dE[line] = fabs(atom->level_energy_eV[upper_global] -
                          atom->level_energy_eV[lower_global]) * EV_TO_ERG;
        h_g_lo[line] = atom->level_g[lower_global];
        h_g_up[line] = atom->level_g[upper_global];
        n_mapped++;
    }

    /* ---- upload the per-line arrays (constant for the whole solve) ---- */
    #define ASM_UPLOAD_I(dst, src) do {                                       \
        ACUDA_CHECK(cudaMalloc(&(dst), (size_t)n_lines * sizeof(int)));        \
        ACUDA_CHECK(cudaMemcpy((dst), (src), (size_t)n_lines * sizeof(int),    \
                               cudaMemcpyHostToDevice)); } while(0)
    #define ASM_UPLOAD_D(dst, src) do {                                       \
        ACUDA_CHECK(cudaMalloc(&(dst), (size_t)n_lines * sizeof(double)));     \
        ACUDA_CHECK(cudaMemcpy((dst), (src), (size_t)n_lines * sizeof(double), \
                               cudaMemcpyHostToDevice)); } while(0)
    ASM_UPLOAD_I(g_asm.d_map,   h_map);
    ASM_UPLOAD_I(g_asm.d_fl_lo, h_fl_lo);
    ASM_UPLOAD_I(g_asm.d_fl_up, h_fl_up);
    ASM_UPLOAD_D(g_asm.d_dE,    h_dE);
    ASM_UPLOAD_I(g_asm.d_g_lo,  h_g_lo);
    ASM_UPLOAD_I(g_asm.d_g_up,  h_g_up);
    ASM_UPLOAD_D(g_asm.d_f_lu,  h_f_lu);
    ASM_UPLOAD_D(g_asm.d_nu,    h_nu);
    ASM_UPLOAD_D(g_asm.d_B_lu,  h_B_lu);
    ASM_UPLOAD_D(g_asm.d_B_ul,  h_B_ul);
    ASM_UPLOAD_D(g_asm.d_A_ul,  h_A_ul);
    #undef ASM_UPLOAD_I
    #undef ASM_UPLOAD_D

    free(h_map); free(h_fl_lo); free(h_fl_up); free(h_dE); free(h_g_lo);
    free(h_g_up); free(h_f_lu); free(h_nu); free(h_B_lu); free(h_B_ul);
    free(h_A_ul);

    /* persistent device buffers for the per-iter state */
    ACUDA_CHECK(cudaMalloc(&g_asm.d_fl_to_super, (size_t)n_lev * sizeof(int)));
    ACUDA_CHECK(cudaMalloc(&g_asm.d_within_sl_frac,
                           (size_t)n_lev * n_shells * sizeof(double)));
    ACUDA_CHECK(cudaMalloc(&g_asm.d_J_nu,
                           (size_t)n_shells * n_freq * sizeof(double)));
    ACUDA_CHECK(cudaMalloc(&g_asm.d_T_e, (size_t)n_shells * sizeof(double)));
    ACUDA_CHECK(cudaMalloc(&g_asm.d_n_e, (size_t)n_shells * sizeof(double)));
    ACUDA_CHECK(cudaMalloc(&g_asm.d_W,   (size_t)n_shells * sizeof(double)));
    ACUDA_CHECK(cudaMalloc(&g_asm.d_active, (size_t)n_shells * sizeof(int)));

    g_asm.d_mat = NULL;
    g_asm.d_mat_elems = 0;
    g_asm.initialized = 1;

    printf("  [NLTE-ASM] GPU bb-assembly init: %d lines (%d NLTE-mapped) x %d shells, "
           "%d freq, %d NLTE levels\n", n_lines, n_mapped, n_shells, n_freq, n_lev);
    return 0;
}

extern "C" int nlte_assemble_gpu_supported(void)
{
    /* Sealed/experimental bb modes the GPU path does NOT reproduce -> stay CPU.
     * LUMINA_ARTIS_PARITY is a blocker: the parity collision network (A2/A5
     * artis_col_rates vR+Bethe+Gaunt, A3 real close-coupling Omega + per-line
     * proxy suppression, A4 collisional ioniz/recomb, A1 metastable full-connect)
     * lives ENTIRELY in the CPU assembler; the GPU bb kernel only knows the legacy
     * vR / COLL_FIX forms and would both mis-rate and double-count the col_data
     * passes. Force CPU assembly under parity. */
    const char *blockers[] = {
        "LUMINA_NLTE_JBAR_POPS", "LUMINA_MALI", "LUMINA_NLTE_JEQB",
        "LUMINA_JBAR_SRC_BINNED", "LUMINA_CMF_LINERES_CONSUME",
        "LUMINA_ARTIS_PARITY", NULL
    };
    for (int i = 0; blockers[i]; i++) {
        const char *e = getenv(blockers[i]);
        if (e && atoi(e) != 0) {
            fprintf(stderr, "[NLTE-ASM] %s active -> GPU bb-assembly unsupported, "
                            "falling back to CPU assembly.\n", blockers[i]);
            return 0;
        }
    }
    return 1;
}

extern "C" void nlte_assemble_gpu_refresh(NLTEConfig *nlte, PlasmaState *plasma)
{
    if (gpu_rf_block_unmigrated(&g_a2_12_nlte_asm_counters,
            GPU_RATE_NOT_MIGRATED, "lumina_nlte_assemble.refresh") != 0)
        return;
    if (!g_asm.initialized) return;
    int n_shells = g_asm.n_shells;
    int n_freq   = g_asm.n_freq;
    int n_lev    = g_asm.n_nlte_levels_total;

    ACUDA_CHECK(cudaMemcpy(g_asm.d_fl_to_super, nlte->fl_to_super,
                           (size_t)n_lev * sizeof(int), cudaMemcpyHostToDevice));
    ACUDA_CHECK(cudaMemcpy(g_asm.d_within_sl_frac, nlte->within_sl_frac,
                           (size_t)n_lev * n_shells * sizeof(double),
                           cudaMemcpyHostToDevice));
    ACUDA_CHECK(cudaMemcpy(g_asm.d_J_nu, nlte->J_nu,
                           (size_t)n_shells * n_freq * sizeof(double),
                           cudaMemcpyHostToDevice));
    ACUDA_CHECK(cudaMemcpy(g_asm.d_T_e, plasma->T_e,
                           (size_t)n_shells * sizeof(double), cudaMemcpyHostToDevice));
    ACUDA_CHECK(cudaMemcpy(g_asm.d_n_e, plasma->n_electron,
                           (size_t)n_shells * sizeof(double), cudaMemcpyHostToDevice));
    ACUDA_CHECK(cudaMemcpy(g_asm.d_W, plasma->W,
                           (size_t)n_shells * sizeof(double), cudaMemcpyHostToDevice));

    /* Config gates (read once; they are getenv-cached on the CPU side too). */
    g_asm.coll_fix = nlte_coll_fix_enabled();
    {
        const char *e = getenv("LUMINA_NLTE_COLL_FLOOR");
        g_asm.coll_floor = e ? atof(e) : 0.0;
        if (g_asm.coll_floor < 0.0) g_asm.coll_floor = 0.0;
    }
    {
        const char *e = getenv("LUMINA_NLTE_DILUTE_FIELD");
        g_asm.dilute_on = (e && atoi(e) != 0) ? 1 : 0;
        const char *t = getenv("LUMINA_NLTE_DILUTE_TR_K");
        double over = t ? atof(t) : 0.0;
        g_asm.dilute_TR = (over > 0.0) ? over : plasma->T_rad[0];
    }
}

extern "C" void nlte_assemble_bb_gpu_pair(double *h_matrices, int N, int n_shells,
                                          int pair_lo, int pair_hi, int super_start,
                                          int n_lo_super, const int *active)
{
    if (gpu_rf_block_unmigrated(&g_a2_12_nlte_asm_counters,
            GPU_RATE_NOT_MIGRATED, "lumina_nlte_assemble.bb_pair") != 0)
        return;
    if (!g_asm.initialized || N <= 0) return;

    size_t elems = (size_t)n_shells * N * N;
    if (elems > g_asm.d_mat_elems) {
        if (g_asm.d_mat) cudaFree(g_asm.d_mat);
        ACUDA_CHECK(cudaMalloc(&g_asm.d_mat, elems * sizeof(double)));
        g_asm.d_mat_elems = elems;
    }

    /* Upload the CPU-assembled remainder, atomicAdd the bb contributions on top,
     * download the merged matrices back into h_matrices (then the existing
     * batched solve re-uploads + factorizes them, unchanged). */
    ACUDA_CHECK(cudaMemcpy(g_asm.d_mat, h_matrices, elems * sizeof(double),
                           cudaMemcpyHostToDevice));
    ACUDA_CHECK(cudaMemcpy(g_asm.d_active, active, (size_t)n_shells * sizeof(int),
                           cudaMemcpyHostToDevice));

    int threads = 256;
    dim3 grid((g_asm.n_lines + threads - 1) / threads, n_shells, 1);
    nlte_assemble_bb_kernel<<<grid, threads>>>(
        g_asm.d_mat, N, n_shells, g_asm.n_lines,
        pair_lo, pair_hi, super_start, n_lo_super,
        g_asm.d_map, g_asm.d_fl_lo, g_asm.d_fl_up,
        g_asm.d_dE, g_asm.d_g_lo, g_asm.d_g_up,
        g_asm.d_f_lu, g_asm.d_nu,
        g_asm.d_B_lu, g_asm.d_B_ul, g_asm.d_A_ul,
        g_asm.d_fl_to_super, g_asm.d_within_sl_frac,
        g_asm.d_J_nu, g_asm.n_freq,
        g_asm.d_T_e, g_asm.d_n_e, g_asm.d_W,
        g_asm.nu_min, g_asm.nu_max, g_asm.d_log_nu,
        g_asm.coll_fix, g_asm.coll_floor, g_asm.dilute_on, g_asm.dilute_TR,
        g_asm.d_active);
    ACUDA_CHECK(cudaGetLastError());
    ACUDA_CHECK(cudaDeviceSynchronize());

    ACUDA_CHECK(cudaMemcpy(h_matrices, g_asm.d_mat, elems * sizeof(double),
                           cudaMemcpyDeviceToHost));
}

extern "C" void nlte_assemble_gpu_free(void)
{
    if (!g_asm.initialized) return;
    cudaFree(g_asm.d_map); cudaFree(g_asm.d_fl_lo); cudaFree(g_asm.d_fl_up);
    cudaFree(g_asm.d_dE); cudaFree(g_asm.d_g_lo); cudaFree(g_asm.d_g_up);
    cudaFree(g_asm.d_f_lu); cudaFree(g_asm.d_nu);
    cudaFree(g_asm.d_B_lu); cudaFree(g_asm.d_B_ul); cudaFree(g_asm.d_A_ul);
    cudaFree(g_asm.d_fl_to_super); cudaFree(g_asm.d_within_sl_frac);
    cudaFree(g_asm.d_J_nu); cudaFree(g_asm.d_T_e); cudaFree(g_asm.d_n_e);
    cudaFree(g_asm.d_W); cudaFree(g_asm.d_active);
    if (g_asm.d_mat) cudaFree(g_asm.d_mat);
    memset(&g_asm, 0, sizeof(g_asm));
}
