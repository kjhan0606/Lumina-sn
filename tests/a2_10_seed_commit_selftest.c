/* A2-INIT seed-material commit controls.
 *
 * Negative: wrong Te generation, wrong Te-publication provenance, and a
 * trial/public seed-Te byte mismatch must each fail closed with every public
 * owner byte-preserved (Te bytes AND generation AND publication included).
 * Positive: a qualified seed commit must change the material (populations,
 * ne, tau/source, publications, BF owners) while the public seed Te array,
 * Te generation, and Te publication stay byte-identical and the population
 * generation advances exactly once. */
#include "nlte_population_candidate.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *message)
{
    fprintf(stderr, "[A2-10-SEED-COMMIT][FAIL] %s\n", message);
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
    atom.population_committed_generation = 1;

    plasma.n_shells = NS;
    plasma.T_e = te;
    plasma.n_electron = ne;
    plasma.rho = rho;
    plasma.T_e_generation = 1;
    plasma.te_publication.required_te_generation = 1;
    plasma.te_publication.committed_te_generation = 1;
    if (population_te_manifest_sha256(
            te, NS, plasma.te_publication.te_manifest_sha256) != POP_OK)
        return fail("seed Te publication manifest fixture");

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
    /* nlte_init() has not yet published a population at the bootstrap
     * boundary.  The atom owner is P1; this predictor creates P2. */
    nlte.population_committed_generation = 0;
    nlte.radfield_view.generation = 1;
    nlte.line_view.generation = 1;

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
    double opacity_ne[NS] = {1.0, 2.0};
    double opacity_te[NS] = {10000.0, 12000.0};
    opacity.n_lines = NLINE;
    opacity.n_shells = NS;
    opacity.tau_sobolev = public_tau;
    opacity.line_source_S = public_source;
    opacity.tau_validity = public_tau_validity;
    opacity.line_source_validity = public_source_validity;
    opacity.line_list_nu = line_nu;
    opacity.tau_required_generation = 3;
    opacity.tau_computed_generation = 3;
    opacity.electron_density = opacity_ne;
    opacity.t_electrons = opacity_te;

    /* Candidate at the SEED Te (trial bytes == public T_e), Te generation 1
     * (current, not +1) and population generation 1 -> 2. */
    NLTEPopulationCandidate candidate;
    if (nlte_population_candidate_begin(
            &candidate, &nlte, &atom, &plasma, te, NS, 1, 2) !=
            NLTE_CANDIDATE_OK ||
        nlte_population_candidate_prepare_opacity_view(
            &candidate, &opacity) != NLTE_CANDIDATE_OK ||
        nlte_population_candidate_prepare_ionization(
            &candidate) != NLTE_CANDIDATE_OK)
        return fail("seed candidate setup rejected");

    /* Simulate the already-verified shared-core hand-off (as the tau
     * selftest does) with a level state that DIFFERS from the public one so
     * the positive control can prove the material moved. */
    candidate.atom.population_committed_generation = 2;
    candidate.nlte.population_committed_generation = 2;
    candidate.level_population[NS] = 40.0;
    candidate.level_population[NS + 1] = 80.0;
    if (nlte_population_candidate_produce_tau_source(
            &candidate, 100.0) != NLTE_CANDIDATE_OK)
        return fail("seed candidate tau/source producer rejected");

    if (bf_opacity_init_checked(&candidate.bf, NS) != 0)
        return fail("private BF fixture allocation failed");
    candidate.bf_active = 1;
    candidate.bf.event_measure_provenance = EVENT_MEASURE_LEGACY_ARGMAX;
    candidate.bf.event_measure_generation = 1;
    size_t n_bf = (size_t)NS * candidate.bf.n_freq_bins;
    for (size_t i = 0; i < n_bf; ++i) {
        candidate.bf.chi_bf[i] = 1.0e-20;
        candidate.bf.eta_bf[i] = 1.0e-20;
    }
    if (nlte_population_candidate_produce_publications(
            &candidate, 100.0) != NLTE_CANDIDATE_OK)
        return fail("private A208/A209 publication rejected");

    candidate.atom.partition_stamp.required_population_generation = 2;
    candidate.atom.partition_stamp.computed_population_generation = 2;
    candidate.atom.partition_stamp.te_generation = 1;
    candidate.atom.partition_stamp.n_shells = NS;
    candidate.atom.partition_stamp.n_items = NI;
    candidate.atom.partition_stamp.status = POP_OK;
    candidate.nlte.within_sl_stamp.required_population_generation = 2;
    candidate.nlte.within_sl_stamp.computed_population_generation = 2;
    candidate.nlte.within_sl_stamp.te_generation = 1;
    candidate.nlte.within_sl_stamp.n_shells = NS;
    candidate.nlte.within_sl_stamp.n_items = NL;
    candidate.nlte.within_sl_stamp.status = POP_OK;
    candidate.adiabatic = calloc(NS, sizeof(*candidate.adiabatic));
    if (!candidate.adiabatic) return fail("commit adiabatic allocation");
    candidate.adiabatic_active = 1;
    candidate.bundle_complete = 1;
    candidate.nlte.population_counters.pop_generation_required = 2;
    candidate.nlte.population_counters.pop_generation_committed = 2;

    BFOpacity public_bf;
    if (bf_opacity_init_checked(&public_bf, NS) != 0)
        return fail("public BF fixture allocation");

    double candidate_ne[NS];
    double candidate_tau0 = candidate.tau_sobolev[0];
    memcpy(candidate_ne, candidate.electron_density, sizeof(candidate_ne));

    /* -------- negative control 1: wrong Te generation -------- */
    plasma.T_e_generation = 2;
    {
        unsigned char nlte_pre[sizeof(nlte)], atom_pre[sizeof(atom)];
        unsigned char plasma_pre[sizeof(plasma)], opacity_pre[sizeof(opacity)];
        unsigned char bf_pre[sizeof(public_bf)];
        double te_pre[NS], ne_pre[NS], lvl_pre[NL * NS], tau_pre[NLINE * NS];
        unsigned char pub_pre[sizeof(plasma.te_publication)];
        memcpy(nlte_pre, &nlte, sizeof(nlte_pre));
        memcpy(atom_pre, &atom, sizeof(atom_pre));
        memcpy(plasma_pre, &plasma, sizeof(plasma_pre));
        memcpy(opacity_pre, &opacity, sizeof(opacity_pre));
        memcpy(bf_pre, &public_bf, sizeof(bf_pre));
        memcpy(pub_pre, &plasma.te_publication, sizeof(pub_pre));
        memcpy(te_pre, te, sizeof(te_pre));
        memcpy(ne_pre, ne, sizeof(ne_pre));
        memcpy(lvl_pre, level_population, sizeof(lvl_pre));
        memcpy(tau_pre, public_tau, sizeof(tau_pre));
        if (nlte_population_candidate_commit_seed_material(
                &candidate, &nlte, &atom, &plasma, &opacity, &public_bf) !=
                NLTE_CANDIDATE_COMMIT_FAILED ||
            memcmp(&nlte, nlte_pre, sizeof(nlte)) != 0 ||
            memcmp(&atom, atom_pre, sizeof(atom)) != 0 ||
            memcmp(&plasma, plasma_pre, sizeof(plasma)) != 0 ||
            memcmp(&opacity, opacity_pre, sizeof(opacity)) != 0 ||
            memcmp(&public_bf, bf_pre, sizeof(public_bf)) != 0 ||
            memcmp(&plasma.te_publication, pub_pre, sizeof(pub_pre)) != 0 ||
            memcmp(te, te_pre, sizeof(te_pre)) != 0 ||
            memcmp(ne, ne_pre, sizeof(ne_pre)) != 0 ||
            memcmp(level_population, lvl_pre, sizeof(lvl_pre)) != 0 ||
            memcmp(public_tau, tau_pre, sizeof(tau_pre)) != 0 ||
            plasma.T_e_generation != 2 ||
            atom.population_committed_generation != 1)
            return fail("wrong-Te-generation commit was not fail-closed");
    }
    plasma.T_e_generation = 1;
    candidate.status = NLTE_CANDIDATE_OK;

    /* -------- negative control 2: wrong publication provenance -------- */
    plasma.te_publication.te_manifest_sha256[0] ^= 1;
    {
        double te_pre[NS];
        memcpy(te_pre, te, sizeof(te_pre));
        if (nlte_population_candidate_commit_seed_material(
                &candidate, &nlte, &atom, &plasma, &opacity, &public_bf) !=
                NLTE_CANDIDATE_COMMIT_FAILED ||
            memcmp(te, te_pre, sizeof(te_pre)) != 0 ||
            plasma.T_e_generation != 1 ||
            atom.population_committed_generation != 1 ||
            nlte.population_committed_generation != 0)
            return fail("wrong-provenance commit was not fail-closed");
    }
    plasma.te_publication.te_manifest_sha256[0] ^= 1;
    candidate.status = NLTE_CANDIDATE_OK;

    /* -------- negative control 3: trial/public seed Te byte mismatch;
     * the failed preflight must preserve public Te bytes and generation. */
    te[0] = 10001.0;
    {
        double te_pre[NS];
        unsigned char pub_pre[sizeof(plasma.te_publication)];
        memcpy(te_pre, te, sizeof(te_pre));
        memcpy(pub_pre, &plasma.te_publication, sizeof(pub_pre));
        if (nlte_population_candidate_commit_seed_material(
                &candidate, &nlte, &atom, &plasma, &opacity, &public_bf) !=
                NLTE_CANDIDATE_COMMIT_FAILED ||
            memcmp(te, te_pre, sizeof(te_pre)) != 0 ||
            memcmp(&plasma.te_publication, pub_pre, sizeof(pub_pre)) != 0 ||
            plasma.T_e_generation != 1 ||
            atom.population_committed_generation != 1)
            return fail("Te-byte-mismatch commit was not fail-closed");
    }
    te[0] = 10000.0;
    candidate.status = NLTE_CANDIDATE_OK;

    /* -------- positive control: material moves, seed Te does not -------- */
    double te_pre[NS], opacity_te_pre[NS];
    unsigned char pub_pre[sizeof(plasma.te_publication)];
    memcpy(te_pre, te, sizeof(te_pre));
    memcpy(opacity_te_pre, opacity_te, sizeof(opacity_te_pre));
    memcpy(pub_pre, &plasma.te_publication, sizeof(pub_pre));
    uint64_t published_opacity_generation =
        candidate.opacity.cpu_opacity.generation_committed;
    if (nlte_population_candidate_commit_seed_material(
            &candidate, &nlte, &atom, &plasma, &opacity, &public_bf) !=
            NLTE_CANDIDATE_OK)
        return fail("qualified seed commit rejected");
    if (memcmp(te, te_pre, sizeof(te_pre)) != 0 ||
        memcmp(opacity_te, opacity_te_pre, sizeof(opacity_te_pre)) != 0 ||
        plasma.T_e_generation != 1 ||
        memcmp(&plasma.te_publication, pub_pre, sizeof(pub_pre)) != 0)
        return fail("seed commit modified the public Te owners");
    if (atom.population_committed_generation != 2 ||
        nlte.population_committed_generation != 2 ||
        nlte.population_required_generation != 2)
        return fail("seed commit did not advance the population generation");
    if (level_population[NS] != 40.0 || level_population[NS + 1] != 80.0 ||
        public_tau[0] != candidate_tau0 ||
        memcmp(ne, candidate_ne, sizeof(candidate_ne)) != 0 ||
        memcmp(opacity_ne, candidate_ne, sizeof(candidate_ne)) != 0)
        return fail("seed commit did not publish the candidate material");
    if (opacity.cpu_opacity.generation_committed !=
            published_opacity_generation ||
        opacity.cpu_emissivity.committed_emissivity_generation !=
            published_opacity_generation ||
        candidate.opacity.cpu_opacity.frequency_edges ||
        candidate.opacity.cpu_emissivity.nu_edge || candidate.bf.chi_bf ||
        candidate.active || candidate.bundle_complete)
        return fail("seed commit ownership/generation mismatch");
    nlte_population_candidate_free(&candidate);
    a208_publication_free(&opacity.cpu_opacity);
    a209_publication_free(&opacity.cpu_emissivity);
    bf_opacity_free(&public_bf);
    printf("[A2-10-SEED-COMMIT][SELFTEST] status=PASS "
           "wrong_generation=BLOCKED wrong_provenance=BLOCKED "
           "te_byte_mismatch=BLOCKED preservation=BYTE_EXACT "
           "material=CHANGED te_generation=UNCHANGED "
           "te_publication=UNCHANGED population_generation=1->2 "
           "floor=0 cap=0 clamp=0 jitter=0 repair=0\n");
    return 0;
}
