#include "lumina.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Test-only link wrapper for the frozen EW entry point.  bench_frozen_oracle.c
 * is compiled with nlte_element_wide_run_labeled renamed to the wrapper while
 * the production objects are compiled unchanged. */
static uint64_t hash_bytes(uint64_t h, const void *ptr, size_t n) {
    const unsigned char *p = (const unsigned char *)ptr;
    if (!ptr) {
        const unsigned char marker = 0xa5;
        p = &marker;
        n = 1;
    }
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t hash_region(uint64_t h, const void *ptr, size_t count,
                            size_t width) {
    return hash_bytes(h, ptr, ptr ? count * width : 0);
}

static uint64_t nlte_owned_hash(const NLTEConfig *n, const AtomicData *a,
                                const PlasmaState *p) {
    const size_t ns = (size_t)p->n_shells;
    const size_t nf = ns * (size_t)n->n_freq_bins;
    const size_t nl = (size_t)n->n_nlte_levels_total;
    uint64_t h = UINT64_C(1469598103934665603);
    h = hash_bytes(h, n, sizeof(*n));
    h = hash_region(h, n->nlte_to_global_level, nl, sizeof(int));
    h = hash_region(h, n->global_to_nlte_level, (size_t)a->n_levels, sizeof(int));
    h = hash_region(h, n->nlte_line_map, (size_t)a->n_lines, sizeof(int));
    h = hash_region(h, n->drainless_metastable, (size_t)a->n_levels, sizeof(int));
    h = hash_region(h, n->nlte_level_populations, nl * ns, sizeof(double));
    h = hash_region(h, n->j_nu_estimator, nf, sizeof(double));
    h = hash_region(h, n->j_nu_count, nf, sizeof(int));
    h = hash_region(h, n->J_nu, nf, sizeof(double));
    h = hash_region(h, n->nu_bar_nu_estimator, nf, sizeof(double));
    h = hash_region(h, n->bf_rate_estimator, nf, sizeof(double));
    h = hash_region(h, n->fl_to_super, nl, sizeof(int));
    h = hash_region(h, n->super_anchor_global, (size_t)n->n_super_total, sizeof(int));
    h = hash_region(h, n->within_sl_frac, nl * ns, sizeof(double));
    h = hash_region(h, n->shell_tau, ns, sizeof(double));
    return h;
}

int nlte_element_wide_run_labeled(NLTEConfig *, AtomicData *, PlasmaState *,
                                  OpacityState *, int, int, int, double,
                                  GammaDeposition *);

int wave32_wrapped_nlte_element_wide_run_labeled(
    NLTEConfig *nlte, AtomicData *atom, PlasmaState *plasma,
    OpacityState *opacity, int Z, int shell_index, int shell_label,
    double time_explosion, GammaDeposition *gamma_dep) {
    const uint64_t before = nlte_owned_hash(nlte, atom, plasma);
    const int rc = nlte_element_wide_run_labeled(
        nlte, atom, plasma, opacity, Z, shell_index, shell_label,
        time_explosion, gamma_dep);
    const uint64_t after = nlte_owned_hash(nlte, atom, plasma);
    fprintf(stderr,
            "[W32-NLTE-STATE] Z=%d s=%d before=%016llx after=%016llx "
            "byte_unchanged=%d rc=%d\n",
            Z, shell_label, (unsigned long long)before,
            (unsigned long long)after, before == after, rc);
    return rc;
}
