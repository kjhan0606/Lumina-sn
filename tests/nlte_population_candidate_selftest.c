#include "nlte_population_candidate.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int fail(const char *message)
{
    fprintf(stderr, "[NLTE-CANDIDATE][FAIL] %s\n", message);
    return 1;
}

int main(void)
{
    NLTEPopulationSolveDiagnostic diagnostic;
    memset(&diagnostic, 0xa5, sizeof(diagnostic));
    nlte_population_solve_diagnostic_reset(&diagnostic);
    if (diagnostic.stage != NLTE_POP_DIAG_STAGE_NONE ||
        diagnostic.ion_status != NLTE_ION_SOLVE_OK ||
        diagnostic.ce_iteration != -1 || diagnostic.pair_index != -1 ||
        diagnostic.pair_lo_slot != -1 || diagnostic.pair_hi_slot != -1 ||
        diagnostic.Z != -1 || diagnostic.ion_lo != -1 ||
        diagnostic.ion_hi != -1 || diagnostic.shell != -1 ||
        diagnostic.worst_ion_slot != -1 || diagnostic.worst_shell != -1 ||
        diagnostic.ion_lock_active != -1 ||
        diagnostic.solve_level_index != -1 ||
        diagnostic.super_level_index != -1 ||
        diagnostic.anchor_global_level != -1 ||
        diagnostic.level_number != -1 || diagnostic.statistical_weight != -1 ||
        diagnostic.negative_count != 0 ||
        !isnan(diagnostic.trial_temperature) ||
        !isnan(diagnostic.population_value) ||
        !isnan(diagnostic.negative_relative_scale) ||
        !isnan(diagnostic.pair_total_density) ||
        diagnostic.linear_rank != -1 ||
        diagnostic.equilibration_iterations != -1 ||
        diagnostic.refinement_iterations != -1 ||
        !isnan(diagnostic.pivot_growth) ||
        !isnan(diagnostic.final_backward_error))
        return fail("population solve diagnostic reset is incomplete");
    if (strcmp(nlte_population_solve_stage_name(
                   NLTE_POP_DIAG_STAGE_PAIR_SOLVE), "PAIR_SOLVE") != 0 ||
        strcmp(nlte_population_solve_stage_name(
                   NLTE_POP_DIAG_STAGE_IONIZATION), "IONIZATION") != 0 ||
        strcmp(nlte_ion_solve_status_name(
                   NLTE_ION_SOLVE_RANK_INCOMPLETE), "RANK_INCOMPLETE") != 0)
        return fail("population solve diagnostic positive names mismatch");
    if (strcmp(nlte_population_solve_stage_name(
                   (NLTEPopulationSolveStage)999), "UNKNOWN_STAGE") != 0 ||
        strcmp(nlte_ion_solve_status_name(
                   (NLTEIonSolveStatus)999), "UNKNOWN_ION_STATUS") != 0)
        return fail("population solve diagnostic invalid enum did not fail closed");

    enum { NS = 2, NI = 3, NL = 4 };
    NLTEConfig nlte;
    AtomicData atom;
    PlasmaState plasma;
    OpacityState opacity;
    memset(&nlte, 0, sizeof(nlte));
    memset(&atom, 0, sizeof(atom));
    memset(&plasma, 0, sizeof(plasma));
    memset(&opacity, 0, sizeof(opacity));

    double public_te[NS] = {9000.0, 11000.0};
    double public_ne[NS] = {2.0, 3.0};
    double public_ion[NI * NS] = {1, 2, 3, 4, 5, 6};
    double public_level[NL * NS] = {1, 2, 3, 4, 5, 6, 7, 8};
    double public_partition[NI * NS] = {2, 2, 3, 3, 4, 4};
    double public_within[NL * NS] = {0.5, 0.5, 0.25, 0.25,
                                     0.75, 0.75, 1.0, 1.0};
    plasma.n_shells = NS;
    plasma.T_e = public_te;
    plasma.n_electron = public_ne;
    plasma.T_e_generation = 4;
    atom.n_ion_pops = NI;
    atom.n_levels = NL;
    atom.ion_number_density = public_ion;
    atom.partition_functions = public_partition;
    int level_offset[NI + 1] = {0, 2, 3, 4};
    double level_energy_eV[NL] = {0.0, 1.0, 0.0, 0.0};
    int level_g[NL] = {2, 4, 3, 4};
    atom.level_offset = level_offset;
    atom.level_energy_eV = level_energy_eV;
    atom.level_g = level_g;
    atom.population_committed_generation = 7;
    nlte.n_nlte_levels_total = NL;
    nlte.super_mode = 1;
    nlte.n_super_total = 3;
    int nlte_to_global_level[NL] = {0, 1, 2, 3};
    int full_to_superlevel[NL] = {0, 0, 1, 2};
    int super_anchor_global[3] = {0, 2, 3};
    nlte.nlte_to_global_level = nlte_to_global_level;
    nlte.fl_to_super = full_to_superlevel;
    nlte.super_anchor_global = super_anchor_global;
    nlte.nlte_level_populations = public_level;
    nlte.within_sl_frac = public_within;
    nlte.population_committed_generation = 7;
    nlte.population_error_count = 9;

    enum { NLINE = 2, NLINE_VALUES = NLINE * NS };
    double public_tau[NLINE_VALUES] = {0.1, -0.2, 0.3, 0.4};
    double public_source[NLINE_VALUES] = {1.0, -2.0, 3.0, 4.0};
    A208Validity public_tau_validity[NLINE_VALUES] = {
        A208_VALID, A208_VALID, A208_VALID, A208_VALID
    };
    A208Validity public_source_validity[NLINE_VALUES] = {
        A208_VALID, A208_VALID, A208_VALID, A208_VALID
    };
    opacity.n_lines = NLINE;
    opacity.n_shells = NS;
    opacity.tau_sobolev = public_tau;
    opacity.line_source_S = public_source;
    opacity.tau_validity = public_tau_validity;
    opacity.line_source_validity = public_source_validity;
    opacity.tau_required_generation = 3;
    opacity.tau_computed_generation = 3;
    if (a208_publication_init(&opacity.cpu_opacity, NS, 2, 0, 0) != 0)
        return fail("public opacity publication fixture allocation failed");
    if (a209_publication_init(&opacity.cpu_emissivity, NS, 2) != 0) {
        a208_publication_free(&opacity.cpu_opacity);
        return fail("public emissivity publication fixture allocation failed");
    }
    opacity.cpu_opacity.generation_required = 11;
    opacity.cpu_opacity.generation_committed = 11;
    opacity.cpu_opacity.frequency_edges[0] = 123.0;
    opacity.cpu_emissivity.required_emissivity_generation = 11;
    opacity.cpu_emissivity.committed_emissivity_generation = 11;
    opacity.cpu_emissivity.nu_edge[0] = 456.0;

    NLTEConfig nlte_before = nlte;
    AtomicData atom_before = atom;
    PlasmaState plasma_before = plasma;
    OpacityState opacity_before = opacity;
    double te_before[NS], ne_before[NS], ion_before[NI * NS];
    double level_before[NL * NS], partition_before[NI * NS];
    double within_before[NL * NS];
    double tau_before[NLINE_VALUES], source_before[NLINE_VALUES];
    A208Validity tau_validity_before[NLINE_VALUES];
    A208Validity source_validity_before[NLINE_VALUES];
    memcpy(te_before, public_te, sizeof(te_before));
    memcpy(ne_before, public_ne, sizeof(ne_before));
    memcpy(ion_before, public_ion, sizeof(ion_before));
    memcpy(level_before, public_level, sizeof(level_before));
    memcpy(partition_before, public_partition, sizeof(partition_before));
    memcpy(within_before, public_within, sizeof(within_before));
    memcpy(tau_before, public_tau, sizeof(tau_before));
    memcpy(source_before, public_source, sizeof(source_before));
    memcpy(tau_validity_before, public_tau_validity,
           sizeof(tau_validity_before));
    memcpy(source_validity_before, public_source_validity,
           sizeof(source_validity_before));

    double trial_te[NS] = {10000.0, 12000.0};
    NLTEPopulationCandidate candidate;
    if (nlte_population_candidate_begin(
            &candidate, &nlte, &atom, &plasma, trial_te, NS, 5, 8) !=
        NLTE_CANDIDATE_OK)
        return fail("valid candidate begin rejected");
    if (!candidate.active || candidate.plasma.T_e != candidate.trial_te ||
        candidate.plasma.n_electron != candidate.electron_density ||
        candidate.atom.ion_number_density != candidate.ion_population ||
        candidate.atom.partition_functions != candidate.partition ||
        candidate.nlte.nlte_level_populations != candidate.level_population ||
        candidate.nlte.within_sl_frac != candidate.within_sl_fraction ||
        candidate.nlte.ew_runtime_counts_sink != &candidate.ew_runtime_counts ||
        !candidate.nlte.solve_effect_policy_explicit ||
        candidate.nlte.solve_effects_allowed != 0 ||
        nlte_solve_effect_allowed(
            &candidate.nlte, NLTE_SOLVE_EFFECT_DIAGNOSTIC_FILES))
        return fail("candidate pointer ownership is incomplete");
    if (candidate.trial_te == public_te ||
        candidate.ion_population == public_ion ||
        candidate.level_population == public_level ||
        candidate.electron_density == public_ne ||
        candidate.partition == public_partition ||
        candidate.within_sl_fraction == public_within)
        return fail("candidate aliases a public mutable array");
    if (candidate.plasma.T_e_generation != 5 ||
        candidate.nlte.population_required_generation != 8 ||
        candidate.nlte.population_error_count != 0 ||
        candidate.nlte.population_counters.pop_generation_required != 8 ||
        candidate.ionization_prepared ||
        candidate.ionization_worst_ne_shell != -1 ||
        candidate.ionization_worst_ion_index != -1 ||
        candidate.ionization_worst_ion_shell != -1 ||
        candidate.ionization_worst_charge_shell != -1 ||
        candidate.ionization_failure_status != POP_OK ||
        candidate.ionization_failure_shell != -1 ||
        candidate.ionization_failure_element != -1 ||
        candidate.ionization_failure_Z != -1 ||
        candidate.ionization_failure_ip_cur != -1 ||
        candidate.ionization_failure_ip_next != -1 ||
        candidate.ionization_failure_level != -1 ||
        candidate.ionization_failure_bf_state != -1 ||
        !isnan(candidate.ionization_failure_nu_threshold) ||
        !isnan(candidate.ionization_failure_sigma_max) ||
        candidate.solve_diagnostic.stage != NLTE_POP_DIAG_STAGE_NONE ||
        candidate.solve_diagnostic.shell != -1 ||
        candidate.solve_diagnostic.pair_index != -1)
        return fail("trial tokens/error ledger not candidate-local");
    if (nlte_population_candidate_prepare_opacity_view(
            &candidate, &opacity) != NLTE_CANDIDATE_OK)
        return fail("valid private opacity view rejected");
    if (!candidate.opacity_active ||
        candidate.opacity.tau_sobolev != candidate.tau_sobolev ||
        candidate.opacity.line_source_S != candidate.line_source ||
        candidate.opacity.tau_validity != candidate.tau_validity ||
        candidate.opacity.line_source_validity !=
            candidate.line_source_validity ||
        candidate.opacity.electron_density != candidate.electron_density ||
        candidate.opacity.t_electrons != candidate.trial_te ||
        candidate.opacity.cpu_opacity.generation_required != 11 ||
        candidate.opacity.cpu_opacity.generation_committed != 0 ||
        candidate.opacity.cpu_opacity.frequency_edges != NULL ||
        candidate.opacity.cpu_emissivity.nu_edge != NULL ||
        candidate.tau_sobolev == public_tau ||
        candidate.line_source == public_source ||
        candidate.tau_validity == public_tau_validity ||
        candidate.line_source_validity == public_source_validity)
        return fail("candidate opacity slab ownership is incomplete");

    if (nlte_population_candidate_prepare_thermodynamics(&candidate) !=
        NLTE_CANDIDATE_OK)
        return fail("private thermodynamic producer rejected valid state");
    if (candidate.atom.partition_stamp.te_generation != 5 ||
        candidate.atom.partition_stamp.computed_population_generation != 8 ||
        candidate.nlte.within_sl_stamp.te_generation != 5 ||
        candidate.nlte.within_sl_stamp.computed_population_generation != 8 ||
        candidate.nlte.within_sl_stamp.n_items != NL)
        return fail("thermodynamic stamps do not carry trial generations");
    for (int s = 0; s < NS; ++s) {
        double sum = candidate.within_sl_fraction[s] +
                     candidate.within_sl_fraction[NS + s];
        double x = 1.0 * 1.602176634e-12 /
                   (1.380649e-16 * trial_te[s]);
        double expected0 = 2.0 / (2.0 + 4.0 * exp(-x));
        if (fabs(sum - 1.0) > 1.0e-14 ||
            fabs(candidate.within_sl_fraction[s] - expected0) > 1.0e-14 ||
            candidate.within_sl_fraction[2 * NS + s] != 1.0 ||
            candidate.within_sl_fraction[3 * NS + s] != 1.0)
            return fail("within-super-level known answer mismatch");
    }

    candidate.trial_te[0] = 50000.0;
    candidate.electron_density[0] = 99.0;
    candidate.ion_population[0] = 88.0;
    candidate.level_population[0] = 77.0;
    candidate.partition[0] = 66.0;
    candidate.within_sl_fraction[0] = 0.123;
    candidate.atom.partition_stamp.te_generation = 1234;
    candidate.nlte.within_sl_stamp.te_generation = 5678;
    candidate.nlte.population_error_count = 42;
    candidate.nlte.population_counters.pop_nonfinite = 1;
    candidate.nlte.ew_runtime_counts_sink->save_restore_calls = 12;
    candidate.nlte.ew_runtime_counts_sink->per_ion_pin_calls = 34;
    candidate.nlte.ew_runtime_counts_sink->topstage_IV_calls = 56;
    candidate.atom.population_committed_generation = 999;
    candidate.nlte.population_committed_generation = 999;
    candidate.tau_sobolev[0] = -9.0;
    candidate.line_source[0] = -8.0;
    candidate.tau_validity[0] = A208_EXACT_ZERO;
    candidate.line_source_validity[0] = A208_NONFINITE;
    candidate.opacity.tau_required_generation = 77;

    if (memcmp(&nlte, &nlte_before, sizeof(nlte)) != 0 ||
        memcmp(&atom, &atom_before, sizeof(atom)) != 0 ||
        memcmp(&plasma, &plasma_before, sizeof(plasma)) != 0 ||
        memcmp(&opacity, &opacity_before, sizeof(opacity)) != 0 ||
        memcmp(public_te, te_before, sizeof(te_before)) != 0 ||
        memcmp(public_ne, ne_before, sizeof(ne_before)) != 0 ||
        memcmp(public_ion, ion_before, sizeof(ion_before)) != 0 ||
        memcmp(public_level, level_before, sizeof(level_before)) != 0 ||
        memcmp(public_partition, partition_before, sizeof(partition_before)) != 0 ||
        memcmp(public_within, within_before, sizeof(within_before)) != 0 ||
        memcmp(public_tau, tau_before, sizeof(tau_before)) != 0 ||
        memcmp(public_source, source_before, sizeof(source_before)) != 0 ||
        memcmp(public_tau_validity, tau_validity_before,
               sizeof(tau_validity_before)) != 0 ||
        memcmp(public_source_validity, source_validity_before,
               sizeof(source_validity_before)) != 0)
        return fail("candidate mutation escaped to a public object");

    double candidate_partition_before[NI * NS];
    double candidate_within_before[NL * NS];
    PopulationDerivedStamp candidate_partition_stamp_before =
        candidate.atom.partition_stamp;
    PopulationDerivedStamp candidate_within_stamp_before =
        candidate.nlte.within_sl_stamp;
    memcpy(candidate_partition_before, candidate.partition,
           sizeof(candidate_partition_before));
    memcpy(candidate_within_before, candidate.within_sl_fraction,
           sizeof(candidate_within_before));
    int invalid_anchor[3] = {NL, 2, 3};
    candidate.nlte.super_anchor_global = invalid_anchor;
    if (nlte_population_candidate_prepare_thermodynamics(&candidate) !=
            NLTE_CANDIDATE_THERMODYNAMIC_FAILED || candidate.active ||
        candidate.population_status != POP_ATOMIC_MISSING)
        return fail("invalid super-level anchor did not fail closed");
    if (memcmp(candidate.partition, candidate_partition_before,
               sizeof(candidate_partition_before)) != 0 ||
        memcmp(candidate.within_sl_fraction, candidate_within_before,
               sizeof(candidate_within_before)) != 0 ||
        memcmp(&candidate.atom.partition_stamp,
               &candidate_partition_stamp_before,
               sizeof(candidate_partition_stamp_before)) != 0 ||
        memcmp(&candidate.nlte.within_sl_stamp,
               &candidate_within_stamp_before,
               sizeof(candidate_within_stamp_before)) != 0)
        return fail("failed thermodynamic substage partially published");

    nlte_population_candidate_free(&candidate);
    nlte_population_candidate_free(&candidate);
    if (candidate.active)
        return fail("candidate free did not clear lifecycle state");
    if (opacity.cpu_opacity.frequency_edges == NULL ||
        opacity.cpu_opacity.frequency_edges[0] != 123.0 ||
        opacity.cpu_emissivity.nu_edge == NULL ||
        opacity.cpu_emissivity.nu_edge[0] != 456.0)
        return fail("candidate free consumed a public publication");

    double bad_te[NS] = {NAN, 12000.0};
    if (nlte_population_candidate_begin(
            &candidate, &nlte, &atom, &plasma, bad_te, NS, 5, 8) !=
            NLTE_CANDIDATE_INVALID_TEMPERATURE || candidate.active)
        return fail("nonfinite trial temperature accepted");
    if (memcmp(&nlte, &nlte_before, sizeof(nlte)) != 0 ||
        memcmp(&atom, &atom_before, sizeof(atom)) != 0 ||
        memcmp(&plasma, &plasma_before, sizeof(plasma)) != 0)
        return fail("failed begin changed a public object");

    a209_publication_free(&opacity.cpu_emissivity);
    a208_publication_free(&opacity.cpu_opacity);
    printf("[NLTE-CANDIDATE][SELFTEST] status=PASS shells=2 "
           "private_arrays=10 stamps=PRIVATE errors=PRIVATE generations=PRIVATE "
           "thermodynamics=PASS within_sl_rollback=PASS "
           "thermo_failure_bytes=PASS public_mutations=0 double_free=PASS "
           "ew_counters=PRIVATE opacity_slab=PRIVATE publications=DETACHED "
           "solver_core=SHARED_ENTRYPOINT\n");
    return 0;
}
