#include "nlte_population_candidate.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *message)
{
    fprintf(stderr, "[NLTE-CANDIDATE-TAU][FAIL] %s\n", message);
    return 1;
}

int main(void)
{
    enum { NS = 2, NI = 1, NL = 2, NLINE = 3 };
    NLTEConfig nlte;
    AtomicData atom;
    PlasmaState plasma;
    OpacityState opacity;
    memset(&nlte, 0, sizeof(nlte));
    memset(&atom, 0, sizeof(atom));
    memset(&plasma, 0, sizeof(plasma));
    memset(&opacity, 0, sizeof(opacity));

    double te[NS] = {10000.0, 12000.0};
    double ne[NS] = {1.0, 2.0};
    double ion[NI * NS] = {12.0, 24.0};
    double partition[NI * NS] = {2.0, 2.0};
    double level_population[NL * NS] = {10.0, 20.0, 2.0, 4.0};
    double within[NL * NS] = {1.0, 1.0, 1.0, 1.0};
    int level_offset[NI + 1] = {0, 2};
    int level_num[NL] = {0, 1};
    double level_energy[NL] = {0.0, 1.0};
    int level_g[NL] = {2, 4};
    int level_Z[NL] = {14, 14};
    int level_ion[NL] = {1, 1};
    int ion_Z[NI] = {14};
    int ion_stage[NI] = {1};
    int element_Z[1] = {14};
    double element_mass[1] = {28.0};
    int element_ion_offset[2] = {0, 1};
    double abundance[NS] = {1.0, 1.0};
    double rho[NS] = {12.0 * 28.0 * AMU, 24.0 * 28.0 * AMU};
    int line_Z[NLINE] = {14, 14, 8};
    int line_ion[NLINE] = {1, 1, 1};
    int line_lower[NLINE] = {0, 0, 0};
    int line_upper[NLINE] = {1, 1, 1};
    double line_f[NLINE] = {0.5, 0.25, 0.125};
    double line_lambda[NLINE] = {5.0e-5, 6.0e-5, 7.0e-5};
    double line_nu[NLINE] = {6.0e14, 5.0e14, 4.0e14};
    double line_A_ul[NLINE] = {1.0e6, 2.0e6, 3.0e6};
    int nlte_to_global[NL] = {0, 1};
    int global_to_nlte[NL] = {0, 1};
    int line_map[NLINE] = {0, -1, -1};

    atom.n_ion_pops = NI;
    atom.n_levels = NL;
    atom.n_lines = NLINE;
    atom.ion_number_density = ion;
    atom.partition_functions = partition;
    atom.level_offset = level_offset;
    atom.level_num = level_num;
    atom.level_energy_eV = level_energy;
    atom.level_g = level_g;
    atom.level_Z = level_Z;
    atom.level_ion = level_ion;
    atom.ion_pop_Z = ion_Z;
    atom.ion_pop_stage = ion_stage;
    atom.n_elements = 1;
    atom.element_Z = element_Z;
    atom.element_mass_amu = element_mass;
    atom.elem_ion_offset = element_ion_offset;
    atom.abundances = abundance;
    atom.line_atomic_number = line_Z;
    atom.line_ion_number = line_ion;
    atom.line_level_lower = line_lower;
    atom.line_level_upper = line_upper;
    atom.line_f_lu = line_f;
    atom.line_wavelength_cm = line_lambda;
    atom.line_nu = line_nu;
    atom.line_A_ul = line_A_ul;
    atom.population_committed_generation = 7;

    plasma.n_shells = NS;
    plasma.T_e = te;
    plasma.n_electron = ne;
    plasma.rho = rho;
    plasma.T_e_generation = 4;

    nlte.enabled = 1;
    nlte.n_nlte_ions = 2;
    nlte.n_nlte_levels_total = NL;
    nlte.nlte_Z[0] = 14;
    nlte.nlte_ion[0] = 1;
    nlte.nlte_Z[1] = 14;
    nlte.nlte_ion[1] = 2;
    nlte.nlte_ion_level_offset[0] = 0;
    nlte.nlte_ion_level_offset[1] = 2;
    nlte.nlte_ion_level_offset[2] = 2;
    nlte.nlte_to_global_level = nlte_to_global;
    nlte.global_to_nlte_level = global_to_nlte;
    nlte.nlte_line_map = line_map;
    nlte.nlte_level_populations = level_population;
    nlte.within_sl_frac = within;
    nlte.population_committed_generation = 7;
    nlte.radfield_view.generation = 2;
    /* A2-04 and R6 are one atomic radiation publication.  A candidate may
     * never mix continuum rates from R2 with line rates from R3. */
    nlte.line_view.generation = 2;

    double public_tau[NLINE * NS] = {0.01, 0.02, 7.0, 8.0, 9.0, 10.0};
    double public_source[NLINE * NS] = {1.0, 2.0, 9.0, 10.0, 11.0, 12.0};
    A208Validity public_tau_validity[NLINE * NS] = {
        A208_VALID, A208_VALID, A208_VALID, A208_VALID,
        A208_VALID, A208_VALID
    };
    A208Validity public_source_validity[NLINE * NS] = {
        A208_VALID, A208_VALID, (A208Validity)0, (A208Validity)0,
        (A208Validity)0, (A208Validity)0
    };
    opacity.n_lines = NLINE;
    opacity.n_shells = NS;
    opacity.tau_sobolev = public_tau;
    opacity.line_source_S = public_source;
    opacity.tau_validity = public_tau_validity;
    opacity.line_source_validity = public_source_validity;
    opacity.line_list_nu = line_nu;
    opacity.tau_required_generation = 3;
    opacity.tau_computed_generation = 3;

    {
        NLTEConfig zinert_nlte = nlte;
        AtomicData zinert_atom = atom;
        double zinert_levels[NL * NS] = {1.0, 2.0, 3.0, 4.0};
        double zinert_ions[NI * NS] = {5.0, 6.0};
        double zero_abundance[NS] = {0.0, 0.0};
        zinert_nlte.nlte_level_populations = zinert_levels;
        zinert_atom.ion_number_density = zinert_ions;
        zinert_atom.abundances = zero_abundance;
        if (nlte_zinert_pair_exact_zero(
                &zinert_nlte, &zinert_atom, &plasma, 0, 1, 1) != 1 ||
            zinert_levels[0] != 1.0 || zinert_levels[NS] != 3.0 ||
            zinert_levels[1] != 0.0 || zinert_levels[NS + 1] != 0.0 ||
            zinert_ions[0] != 5.0 || zinert_ions[1] != 0.0)
            return fail("Z-inert pair did not publish exact zero privately");

        double active_abundance[NS] = {0.0, 1.0};
        double active_levels[NL * NS] = {7.0, 8.0, 9.0, 10.0};
        zinert_nlte.nlte_level_populations = active_levels;
        zinert_atom.abundances = active_abundance;
        if (nlte_zinert_pair_exact_zero(
                &zinert_nlte, &zinert_atom, &plasma, 0, 1, 0) != 0 ||
            active_levels[0] != 7.0 || active_levels[NS] != 9.0 ||
            nlte_zinert_pair_exact_zero(
                &zinert_nlte, &zinert_atom, &plasma, 0, 2, 0) != -1)
            return fail("active/invalid Z-inert pair control did not fail closed");

        /* An active-only composition omits inactive Z entirely instead of
         * carrying an all-zero row.  The retained NLTE topology is still an
         * exact-zero graph and must close before its zero-level layout check. */
        int absent_element_Z[1] = {8};
        double absent_levels[NL * NS] = {11.0, 12.0, 13.0, 14.0};
        double absent_ions[NI * NS] = {15.0, 16.0};
        zinert_nlte.nlte_level_populations = absent_levels;
        zinert_atom.element_Z = absent_element_Z;
        zinert_atom.ion_number_density = absent_ions;
        int absent_rc = nlte_zinert_pair_exact_zero(
            &zinert_nlte, &zinert_atom, &plasma, 0, 1, 0);
        if (absent_rc != 1 ||
            absent_levels[0] != 0.0 || absent_levels[NS] != 0.0 ||
            absent_levels[1] != 12.0 || absent_levels[NS + 1] != 14.0 ||
            absent_ions[0] != 0.0 || absent_ions[1] != 16.0) {
            fprintf(stderr,
                    "[NLTE-CANDIDATE-TAU][ABSENT-Z] rc=%d "
                    "levels=%.17g,%.17g,%.17g,%.17g ions=%.17g,%.17g\n",
                    absent_rc, absent_levels[0], absent_levels[1],
                    absent_levels[2], absent_levels[3],
                    absent_ions[0], absent_ions[1]);
            return fail("composition-absent Z-inert pair was not exact zero");
        }

        /* An active element may have an exactly empty adjacent-ion pair in one
         * shell.  This is not Z-inert and must not use a small-value cutoff:
         * exact zero closes, positive total is untouched, invalid input blocks. */
        NLTEConfig zero_nlte = nlte;
        AtomicData zero_atom = atom;
        int zero_offsets[3] = {0, 1, 2};
        int zero_ion_Z[2] = {14, 14};
        int zero_ion_stage[2] = {1, 2};
        int zero_element_offset[2] = {0, 2};
        double zero_levels[NL * NS] = {21.0, 22.0, 23.0, 24.0};
        double zero_ions[2 * NS] = {0.0, 2.0, 0.0, 3.0};
        zero_nlte.nlte_ion_level_offset[0] = 0;
        zero_nlte.nlte_ion_level_offset[1] = 1;
        zero_nlte.nlte_ion_level_offset[2] = 2;
        zero_nlte.nlte_level_populations = zero_levels;
        zero_atom.n_ion_pops = 2;
        zero_atom.ion_number_density = zero_ions;
        zero_atom.ion_pop_Z = zero_ion_Z;
        zero_atom.ion_pop_stage = zero_ion_stage;
        zero_atom.elem_ion_offset = zero_element_offset;
        if (nlte_zero_total_pair_exact_zero(
                &zero_nlte, &zero_atom, &plasma, 0, 1, 0) != 1 ||
            zero_levels[0] != 0.0 || zero_levels[NS] != 0.0 ||
            zero_levels[1] != 22.0 || zero_levels[NS + 1] != 24.0 ||
            zero_ions[0] != 0.0 || zero_ions[NS] != 0.0)
            return fail("active zero-total pair was not exact zero");
        zero_levels[0] = 31.0;
        zero_levels[NS] = 33.0;
        zero_ions[0] = 1.0e-300;
        if (nlte_zero_total_pair_exact_zero(
                &zero_nlte, &zero_atom, &plasma, 0, 1, 0) != 0 ||
            zero_levels[0] != 31.0 || zero_levels[NS] != 33.0)
            return fail("positive subnormal pair total was treated as zero");
        zero_ions[0] = -1.0;
        if (nlte_zero_total_pair_exact_zero(
                &zero_nlte, &zero_atom, &plasma, 0, 1, 0) != -1)
            return fail("negative pair density did not fail closed");
    }

    OpacityState opacity_before = opacity;
    double public_tau_before[NLINE * NS], public_source_before[NLINE * NS];
    memcpy(public_tau_before, public_tau, sizeof(public_tau));
    memcpy(public_source_before, public_source, sizeof(public_source));
    A208Counters counters_before = *a208_counters();
    A209Counters emissivity_counters_before = *a209_counters();

    /* A2-10 pre-core lagged-tau falsifier: trial ionization plus trial LTE
     * excitation must replace the public-cloned tau before the mode-3 matrix,
     * while the public slab remains byte-identical.  This is intentionally a
     * seed test, not a population--tau fixed-point claim. */
    {
        NLTEPopulationCandidate seed_candidate;
        double seed_trial_te[NS] = {11000.0, 13000.0};
        if (nlte_population_candidate_begin(
                &seed_candidate, &nlte, &atom, &plasma, seed_trial_te,
                NS, 5, 8) != NLTE_CANDIDATE_OK ||
            nlte_population_candidate_prepare_opacity_view(
                &seed_candidate, &opacity) != NLTE_CANDIDATE_OK ||
            nlte_population_candidate_prepare_ionization(
                &seed_candidate) != NLTE_CANDIDATE_OK)
            return fail("pre-core tau seed setup rejected");
        if (setenv("LUMINA_NLTE_JBAR_POPS", "3", 1) != 0)
            return fail("pre-core tau seed mode setup failed");
        if (nlte_population_candidate_prepare_precore_tau_seed(
                &seed_candidate, 100.0) != NLTE_CANDIDATE_OK)
            return fail("pre-core trial-LTE tau seed rejected");
        unsetenv("LUMINA_NLTE_JBAR_POPS");
        double seed_delta0 = seed_candidate.ion_population[0] * 2.0 /
                             seed_candidate.partition[0] *
                             (1.0 - exp(-1.0 * 1.602176634e-12 /
                                        (1.380649e-16 * seed_trial_te[0])));
        double seed_delta1 = seed_candidate.ion_population[1] * 2.0 /
                             seed_candidate.partition[1] *
                             (1.0 - exp(-1.0 * 1.602176634e-12 /
                                        (1.380649e-16 * seed_trial_te[1])));
        double seed_expected0 = SOBOLEV_COEFF * 0.5 * 5.0e-5 * 100.0 *
                                seed_delta0;
        double seed_expected1 = SOBOLEV_COEFF * 0.5 * 5.0e-5 * 100.0 *
                                seed_delta1;
        if (fabs(seed_candidate.tau_sobolev[0] - seed_expected0) >
                1.0e-15 * fmax(1.0, fabs(seed_expected0)) ||
            fabs(seed_candidate.tau_sobolev[1] - seed_expected1) >
                1.0e-15 * fmax(1.0, fabs(seed_expected1)) ||
            seed_candidate.tau_sobolev[0] == public_tau[0] ||
            seed_candidate.tau_sobolev[1] == public_tau[1] ||
            seed_candidate.opacity.tau_required_generation != 4 ||
            seed_candidate.opacity.tau_computed_generation != 4 ||
            seed_candidate.tau_sobolev[2 * NS] != 0.0 ||
            seed_candidate.tau_sobolev[2 * NS + 1] != 0.0 ||
            seed_candidate.tau_validity[2 * NS] != A208_EXACT_ZERO ||
            seed_candidate.tau_validity[2 * NS + 1] != A208_EXACT_ZERO ||
            memcmp(public_tau, public_tau_before, sizeof(public_tau)) != 0 ||
            memcmp(public_source, public_source_before,
                   sizeof(public_source)) != 0)
            return fail("pre-core trial-LTE tau seed/public isolation mismatch");
        nlte_population_candidate_free(&seed_candidate);
    }

    NLTEPopulationCandidate candidate;
    double trial_te[NS] = {11000.0, 13000.0};
    if (nlte_population_candidate_begin(
            &candidate, &nlte, &atom, &plasma, trial_te,
            NS, 5, 8) != NLTE_CANDIDATE_OK ||
        nlte_population_candidate_prepare_opacity_view(
            &candidate, &opacity) != NLTE_CANDIDATE_OK)
        return fail("candidate setup rejected");

    candidate.ion_population[0] = 1.0;
    candidate.ion_population[1] = 2.0;
    if (nlte_population_candidate_prepare_ionization(&candidate) !=
            NLTE_CANDIDATE_OK || !candidate.ionization_prepared ||
        candidate.ion_population[0] != 12.0 ||
        candidate.ion_population[1] != 24.0 ||
        candidate.ionization_max_ion_relative_change < 0.9 ||
        candidate.ionization_max_ne_relative_change <= 0.0 ||
        candidate.ionization_max_charge_residual > 0.05 ||
        ion[0] != 12.0 || ion[1] != 24.0 || ne[0] != 1.0 || ne[1] != 2.0)
        return fail("trial-consistent private ionization owner mismatch");

    /* Simulate the already-verified shared core hand-off; this fixture tests
     * only the post-core private tau/source producer. */
    candidate.atom.population_committed_generation = 8;
    candidate.nlte.population_committed_generation = 8;
    if (nlte_population_candidate_produce_tau_source(
            &candidate, 100.0) != NLTE_CANDIDATE_OK)
        return fail("positive signed tau production rejected");
    double expected0 = SOBOLEV_COEFF * 0.5 * 5.0e-5 * 100.0 * 9.0;
    double expected1 = SOBOLEV_COEFF * 0.5 * 5.0e-5 * 100.0 * 18.0;
    double bulk_delta0 = candidate.ion_population[0] * 2.0 /
                         candidate.partition[0] *
                         (1.0 - exp(-1.0 * 1.602176634e-12 /
                                    (1.380649e-16 * candidate.trial_te[0])));
    double bulk_delta1 = candidate.ion_population[1] * 2.0 /
                         candidate.partition[1] *
                         (1.0 - exp(-1.0 * 1.602176634e-12 /
                                    (1.380649e-16 * candidate.trial_te[1])));
    double expected_bulk0 = SOBOLEV_COEFF * 0.25 * 6.0e-5 * 100.0 *
                            bulk_delta0;
    double expected_bulk1 = SOBOLEV_COEFF * 0.25 * 6.0e-5 * 100.0 *
                            bulk_delta1;
    if (fabs(candidate.tau_sobolev[0] - expected0) > 1.0e-15 ||
        fabs(candidate.tau_sobolev[1] - expected1) > 1.0e-15 ||
        candidate.tau_validity[0] != A208_VALID ||
        fabs(candidate.tau_sobolev[NS] - expected_bulk0) >
            1.0e-15 * fmax(1.0, fabs(expected_bulk0)) ||
        fabs(candidate.tau_sobolev[NS + 1] - expected_bulk1) >
            1.0e-15 * fmax(1.0, fabs(expected_bulk1)) ||
        candidate.tau_sobolev[NS] == public_tau[NS] ||
        candidate.tau_sobolev[NS + 1] == public_tau[NS + 1] ||
        candidate.tau_validity[NS] != A208_VALID ||
        candidate.tau_validity[NS + 1] != A208_VALID ||
        candidate.line_source[NS] != 0.0 ||
        candidate.line_source[NS + 1] != 0.0 ||
        candidate.line_source_validity[NS] != A208_UNSAMPLED ||
        candidate.line_source_validity[NS + 1] != A208_UNSAMPLED ||
        candidate.tau_sobolev[2 * NS] != 0.0 ||
        candidate.tau_sobolev[2 * NS + 1] != 0.0 ||
        candidate.line_source[2 * NS] != 0.0 ||
        candidate.line_source[2 * NS + 1] != 0.0 ||
        candidate.tau_validity[2 * NS] != A208_EXACT_ZERO ||
        candidate.tau_validity[2 * NS + 1] != A208_EXACT_ZERO ||
        candidate.line_source_validity[2 * NS] != A208_EXACT_ZERO ||
        candidate.line_source_validity[2 * NS + 1] != A208_EXACT_ZERO ||
        candidate.opacity.tau_required_generation != 5 ||
        candidate.opacity.tau_computed_generation != 5)
        return fail("positive tau known answer/generation mismatch");

    double identity_tau[NLINE * NS];
    memcpy(identity_tau, candidate.tau_sobolev, sizeof(identity_tau));
    if (nlte_population_candidate_produce_tau_source(
            &candidate, 100.0) != NLTE_CANDIDATE_OK ||
        memcmp(identity_tau, candidate.tau_sobolev,
               sizeof(identity_tau)) != 0 ||
        candidate.opacity.tau_required_generation != 7 ||
        candidate.opacity.tau_computed_generation != 7)
        return fail("same-state tau regeneration is not byte-identical");

    candidate.level_population[NS] = 40.0;
    candidate.level_population[NS + 1] = 80.0;
    if (nlte_population_candidate_produce_tau_source(
            &candidate, 100.0) != NLTE_CANDIDATE_OK ||
        !(candidate.tau_sobolev[0] < 0.0) ||
        !(candidate.line_source[0] < 0.0) ||
        candidate.opacity.tau_required_generation != 9 ||
        candidate.opacity.tau_computed_generation != 9)
        return fail("negative signed tau/source was not preserved");

    if (nlte_population_candidate_prepare_bf(&candidate) !=
            NLTE_CANDIDATE_OK || !candidate.bf_active ||
        !candidate.bf.chi_bf || !candidate.bf.eta_bf ||
        !isfinite(candidate.bf.chi_bf[0]))
        return fail("private CPU BF producer rejected");
    bf_opacity_free(&candidate.bf);
    candidate.bf_active = 0;

    if (bf_opacity_init_checked(&candidate.bf, NS) != 0)
        return fail("private BF fixture allocation failed");
    candidate.bf_active = 1;
    candidate.bf.event_measure_provenance =
        EVENT_MEASURE_LEGACY_ARGMAX;
    candidate.bf.event_measure_generation = 1;
    size_t n_bf = (size_t)NS * candidate.bf.n_freq_bins;
    for (size_t i = 0; i < n_bf; ++i) {
        candidate.bf.chi_bf[i] = 1.0e-20;
        candidate.bf.eta_bf[i] = 1.0e-20;
    }
    if (nlte_population_candidate_produce_publications(
            &candidate, 100.0) != NLTE_CANDIDATE_OK ||
        candidate.opacity.cpu_opacity.generation_committed == 0 ||
        candidate.opacity.cpu_emissivity.committed_emissivity_generation !=
            candidate.opacity.cpu_opacity.generation_committed ||
        candidate.opacity_counters.generation_committed == 0 ||
        candidate.emissivity_counters.generation_committed == 0)
        return fail("private A208/A209 publication rejected");

    if (memcmp(&opacity, &opacity_before, sizeof(opacity)) != 0 ||
        memcmp(public_tau, public_tau_before, sizeof(public_tau)) != 0 ||
        memcmp(public_source, public_source_before, sizeof(public_source)) != 0 ||
        memcmp(a208_counters(), &counters_before,
               sizeof(counters_before)) != 0 ||
        memcmp(a209_counters(), &emissivity_counters_before,
               sizeof(emissivity_counters_before)) != 0)
        return fail("candidate tau/publication escaped into public/global state");

    double radius[NS] = {1.0e14, 2.0e14};
    double velocity[NS] = {1.0e12, 2.0e12};
    NLTEPopulationBundleRequest request = {
        &nlte, &atom, &plasma, &opacity, trial_te,
        radius, velocity, NS, 100.0, 5, 8, NULL
    };
    NLTEPopulationCandidate failed_candidate;
    if (nlte_population_candidate_build_bundle(&failed_candidate, &request) !=
            NLTE_CANDIDATE_SOLVE_FAILED || failed_candidate.active ||
        failed_candidate.bundle_complete)
        return fail("bundle did not fail closed at missing gamma input");
    if (memcmp(&opacity, &opacity_before, sizeof(opacity)) != 0 ||
        memcmp(public_tau, public_tau_before, sizeof(public_tau)) != 0 ||
        memcmp(public_source, public_source_before, sizeof(public_source)) != 0 ||
        memcmp(a208_counters(), &counters_before,
               sizeof(counters_before)) != 0 ||
        memcmp(a209_counters(), &emissivity_counters_before,
               sizeof(emissivity_counters_before)) != 0)
        return fail("failed bundle mutated public/global state");
    nlte_population_candidate_free(&failed_candidate);

    /* Complete a synthetic final replay around the already real private
     * tau/BF/A208/A209 candidate.  The commit must move owners once, copy all
     * material arrays, and leave neither the root nor candidate with public
     * ownership authority. */
    double opacity_ne[NS] = {1.0, 2.0};
    double opacity_te[NS] = {10000.0, 12000.0};
    opacity.electron_density = opacity_ne;
    opacity.t_electrons = opacity_te;
    candidate.atom.partition_stamp.required_population_generation = 8;
    candidate.atom.partition_stamp.computed_population_generation = 8;
    candidate.atom.partition_stamp.te_generation = 5;
    candidate.atom.partition_stamp.n_shells = NS;
    candidate.atom.partition_stamp.n_items = NI;
    candidate.atom.partition_stamp.status = POP_OK;
    candidate.nlte.within_sl_stamp.required_population_generation = 8;
    candidate.nlte.within_sl_stamp.computed_population_generation = 8;
    candidate.nlte.within_sl_stamp.te_generation = 5;
    candidate.nlte.within_sl_stamp.n_shells = NS;
    candidate.nlte.within_sl_stamp.n_items = NL;
    candidate.nlte.within_sl_stamp.status = POP_OK;
    candidate.adiabatic = calloc(NS, sizeof(*candidate.adiabatic));
    if (!candidate.adiabatic) return fail("commit adiabatic allocation");
    candidate.adiabatic_active = 1;
    candidate.bundle_complete = 1;
    candidate.nlte.population_counters.pop_generation_required = 8;
    candidate.nlte.population_counters.pop_generation_committed = 8;

    ElectronTemperaturePublication te_candidate = {0};
    if (a210_publication_init(&te_candidate, NS) != 0)
        return fail("commit Te candidate allocation");
    te_candidate.required_te_generation = 5;
    te_candidate.producer_equation = A210_RE_INTEGRAL;
    memcpy(te_candidate.T_e, candidate.trial_te, sizeof(trial_te));
    memcpy(te_candidate.n_e, candidate.electron_density, sizeof(ne));
    if (population_te_manifest_sha256(
            te_candidate.T_e, NS, te_candidate.te_manifest_sha256) != POP_OK)
        return fail("commit Te manifest");
    for (int s = 0; s < NS; ++s) {
        te_candidate.shell_status[s] = RADEQ_OK;
        te_candidate.residual_status[s] = RADEQ_OK;
        te_candidate.ledger[s].equation_kind = A210_RE_INTEGRAL;
        te_candidate.ledger[s].adiabatic_model =
            A210_ADIABATIC_CMFGEN_COMPLETE;
        te_candidate.ledger[s].status = RADEQ_OK;
        te_candidate.ledger[s].m_line = 1;
    }
    BFOpacity public_bf;
    if (bf_opacity_init_checked(&public_bf, NS) != 0)
        return fail("public BF fixture allocation");
    NLTEConfig nlte_precommit = nlte;
    AtomicData atom_precommit = atom;
    PlasmaState plasma_precommit = plasma;
    OpacityState opacity_precommit = opacity;
    BFOpacity bf_precommit = public_bf;
    double te_precommit[NS], ne_precommit[NS], ion_precommit[NI * NS];
    memcpy(te_precommit, te, sizeof(te_precommit));
    memcpy(ne_precommit, ne, sizeof(ne_precommit));
    memcpy(ion_precommit, ion, sizeof(ion_precommit));
    te_candidate.residual_status[0] = RADEQ_HEAT_RESIDUAL;
    if (nlte_population_candidate_commit_bundle(
            &candidate, &te_candidate, &nlte, &atom, &plasma,
            &opacity, &public_bf) != NLTE_CANDIDATE_COMMIT_FAILED ||
        memcmp(&nlte, &nlte_precommit, sizeof(nlte)) != 0 ||
        memcmp(&atom, &atom_precommit, sizeof(atom)) != 0 ||
        memcmp(&plasma, &plasma_precommit, sizeof(plasma)) != 0 ||
        memcmp(&opacity, &opacity_precommit, sizeof(opacity)) != 0 ||
        memcmp(&public_bf, &bf_precommit, sizeof(public_bf)) != 0 ||
        memcmp(te, te_precommit, sizeof(te_precommit)) != 0 ||
        memcmp(ne, ne_precommit, sizeof(ne_precommit)) != 0 ||
        memcmp(ion, ion_precommit, sizeof(ion_precommit)) != 0)
        return fail("failed commit changed public state");
    te_candidate.residual_status[0] = RADEQ_OK;
    candidate.status = NLTE_CANDIDATE_OK;
    uint64_t published_opacity_generation =
        candidate.opacity.cpu_opacity.generation_committed;
    if (nlte_population_candidate_commit_bundle(
            &candidate, &te_candidate, &nlte, &atom, &plasma,
            &opacity, &public_bf) != NLTE_CANDIDATE_OK)
        return fail("single bundle commit rejected");
    if (plasma.T_e_generation != 5 ||
        atom.population_committed_generation != 8 ||
        nlte.population_committed_generation != 8 ||
        memcmp(plasma.T_e, trial_te, sizeof(trial_te)) != 0 ||
        memcmp(opacity.t_electrons, trial_te, sizeof(trial_te)) != 0 ||
        opacity.cpu_opacity.generation_committed !=
            published_opacity_generation ||
        opacity.cpu_emissivity.committed_emissivity_generation !=
            published_opacity_generation ||
        plasma.te_publication.committed_te_generation != 5 ||
        te_candidate.T_e || candidate.opacity.cpu_opacity.frequency_edges ||
        candidate.opacity.cpu_emissivity.nu_edge || candidate.bf.chi_bf ||
        candidate.active || candidate.bundle_complete)
        return fail("single bundle commit ownership/generation mismatch");
    nlte_population_candidate_free(&candidate);
    a208_publication_free(&opacity.cpu_opacity);
    a209_publication_free(&opacity.cpu_emissivity);
    a210_publication_free(&plasma.te_publication);
    bf_opacity_free(&public_bf);
    printf("[NLTE-CANDIDATE-TAU][SELFTEST] status=PASS signed_tau=POSITIVE+NEGATIVE "
           "source=SIGNED generations=PRIVATE ew_authority=EXPLICIT "
           "bulk_tau=TRIAL_REBUILT same_state_tau=BIT_EXACT "
           "a208_a209_counters=PRIVATE publications=PRIVATE "
           "zinert_pair=EXACT_ZERO zero_total_pair=EXACT_ZERO "
           "positive_subnormal=UNCHANGED invalid_pair=BLOCKED "
           "bundle_failure=ATOMIC single_commit=ATOMIC public_mutations=0\n");
    return 0;
}
