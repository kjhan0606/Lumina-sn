#include "lumina.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static AtomicData *seed_atom;
static int *saved_ion;
static int seeded;

int nlte_bf_field_source(const NLTEConfig *, double, double, double, int,
                         int *, int *, double *);
int nlte_element_wide_run_labeled(NLTEConfig *, AtomicData *, PlasmaState *,
                                  OpacityState *, int, int, int, double,
                                  GammaDeposition *);

/* Called from a production EW object compiled with only the helper symbol
 * renamed.  The first bf target visit occurs after EWPrivateView construction;
 * corrupting its upper anchors at that point seeds an invalid Kramers target
 * and must increment continuum_deletion_firing_count. */
int wave32_seeded_bf_field_source(const NLTEConfig *nlte, double T_e,
                                  double nu, double J_default,
                                  int gpu_lookup_available,
                                  int *use_gpu_lookup,
                                  int *gpu_field_bypassed,
                                  double *J_selected) {
    if (!seeded && seed_atom) {
        seeded = 1;
        for (int sl = 0; sl < nlte->n_super_total; sl++) {
            int gl = nlte->super_anchor_global[sl];
            if (gl >= 0 && gl < seed_atom->n_levels)
                seed_atom->level_ion[gl] += 100;
        }
        fprintf(stderr, "[W32-CONTINUUM-SEED] corrupted private-view anchors\n");
    }
    return nlte_bf_field_source(nlte, T_e, nu, J_default,
                                gpu_lookup_available, use_gpu_lookup,
                                gpu_field_bypassed, J_selected);
}

int wave32_continuum_wrapped_nlte_element_wide_run_labeled(
    NLTEConfig *nlte, AtomicData *atom, PlasmaState *plasma,
    OpacityState *opacity, int Z, int shell_index, int shell_label,
    double time_explosion, GammaDeposition *gamma_dep) {
    int rc;
    saved_ion = (int *)malloc((size_t)atom->n_levels * sizeof(int));
    if (!saved_ion) return -1;
    memcpy(saved_ion, atom->level_ion, (size_t)atom->n_levels * sizeof(int));
    seed_atom = atom;
    seeded = 0;
    rc = nlte_element_wide_run_labeled(
        nlte, atom, plasma, opacity, Z, shell_index, shell_label,
        time_explosion, gamma_dep);
    memcpy(atom->level_ion, saved_ion, (size_t)atom->n_levels * sizeof(int));
    free(saved_ion);
    saved_ion = NULL;
    seed_atom = NULL;
    return rc;
}
