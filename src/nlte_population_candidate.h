#ifndef LUMINA_NLTE_POPULATION_CANDIDATE_H
#define LUMINA_NLTE_POPULATION_CANDIDATE_H

#include "lumina.h"
#include "atomic_internal_energy.h"
#include "cmfgen_adiabatic.h"

typedef enum {
    NLTE_CANDIDATE_OK = 0,
    NLTE_CANDIDATE_INVALID_ARGUMENT,
    NLTE_CANDIDATE_INVALID_LAYOUT,
    NLTE_CANDIDATE_INVALID_TEMPERATURE,
    NLTE_CANDIDATE_INVALID_GENERATION,
    NLTE_CANDIDATE_ALLOCATION_FAILED,
    NLTE_CANDIDATE_THERMODYNAMIC_FAILED,
    NLTE_CANDIDATE_SOLVE_FAILED,
    NLTE_CANDIDATE_OPACITY_FAILED,
    NLTE_CANDIDATE_INTERNAL_ENERGY_FAILED,
    NLTE_CANDIDATE_ADIABATIC_FAILED,
    NLTE_CANDIDATE_COMMIT_FAILED
} NLTEPopulationCandidateStatus;

/* Private population-solve failure provenance.  The public A2-07 path does
 * not need or own this state: it is attached only to an all-shell trial so a
 * failed A2-10 endpoint cannot collapse pair, shell and CE failures into the
 * same POP_SOLVE_FAILED result. */
typedef enum {
    NLTE_POP_DIAG_STAGE_NONE = 0,
    NLTE_POP_DIAG_STAGE_PRECONDITION,
    NLTE_POP_DIAG_STAGE_TRANSACTION,
    NLTE_POP_DIAG_STAGE_THERMODYNAMIC,
    NLTE_POP_DIAG_STAGE_IONIZATION,
    NLTE_POP_DIAG_STAGE_WORKSPACE,
    NLTE_POP_DIAG_STAGE_ELEMENT_WIDE,
    NLTE_POP_DIAG_STAGE_PAIR_SOLVE,
    NLTE_POP_DIAG_STAGE_CE_NONCONVERGED,
    NLTE_POP_DIAG_STAGE_PUBLISH,
    NLTE_POP_DIAG_STAGE_INTERNAL
} NLTEPopulationSolveStage;

typedef enum {
    NLTE_ION_SOLVE_OK = 0,
    NLTE_ION_SOLVE_INVALID_LAYOUT,
    NLTE_ION_SOLVE_ALLOCATION_FAILED,
    NLTE_ION_SOLVE_ASSEMBLY_FAILED,
    NLTE_ION_SOLVE_RANK_INCOMPLETE,
    NLTE_ION_SOLVE_LINEAR_FAILED,
    NLTE_ION_SOLVE_NONFINITE,
    NLTE_ION_SOLVE_NEGATIVE_POPULATION
} NLTEIonSolveStatus;

typedef struct {
    NLTEPopulationSolveStage stage;
    NLTEIonSolveStatus ion_status;
    int ce_iteration;       /* one-based; -1 when no CE pass was entered */
    int ce_max_iterations;
    int pair_index;
    int pair_lo_slot;
    int pair_hi_slot;
    int Z;
    int ion_lo;
    int ion_hi;
    int shell;
    int worst_ion_slot;
    int worst_shell;
    double max_rel_change;
    double ce_threshold;
    int ion_lock_active;
    int solve_level_index;  /* pair-local super-level index */
    int super_level_index;  /* NLTE-global super-level index */
    int anchor_global_level;
    int level_number;
    int statistical_weight;
    int negative_count;
    double trial_temperature;
    double electron_density;
    double level_energy_eV;
    double population_value;
    double vector_min;
    double vector_max;
    double vector_sum;
    double vector_abs_max;
    double negative_relative_scale;
    double pair_total_density;
    double lo_target_density;
    double hi_target_density;
    double solved_lo_sum;
    double solved_hi_sum;
    int linear_rank;
    int equilibration_iterations;
    int refinement_iterations;
    double pivot_growth;
    double initial_backward_error;
    double final_backward_error;
} NLTEPopulationSolveDiagnostic;

/* Shallow object views whose mutable material arrays are private.  Immutable
 * maps, atomic tables and radiation views remain borrowed from the committed
 * objects.  Never pass these shallow copies to a public-object free routine. */
typedef struct {
    NLTEConfig nlte;
    AtomicData atom;
    PlasmaState plasma;

    double *trial_te;
    double *ion_population;
    double *level_population;
    double *electron_density;
    double *partition;
    double *within_sl_fraction;
    int ionization_prepared;
    double ionization_max_ne_relative_change;
    double ionization_max_ion_relative_change;
    double ionization_max_charge_residual;
    int ionization_worst_ne_shell;
    int ionization_worst_ion_index;
    int ionization_worst_ion_shell;
    int ionization_worst_charge_shell;
    PopulationStatus ionization_failure_status;
    int ionization_failure_shell;
    int ionization_failure_element;
    int ionization_failure_Z;
    int ionization_failure_ip_cur;
    int ionization_failure_ip_next;
    int ionization_failure_stage;
    int ionization_failure_level;
    int ionization_failure_level_number;
    int ionization_failure_bf_state;
    int ionization_failure_has_sigma;
    double ionization_failure_trial_temperature;
    double ionization_failure_electron_density;
    double ionization_failure_chi_eV;
    double ionization_failure_level_energy_eV;
    double ionization_failure_nu_threshold;
    double ionization_failure_sigma_max;
    double ionization_failure_w_miss;
    NLTEEWRuntimeCounts ew_runtime_counts;
    A208Counters opacity_counters;
    A209Counters emissivity_counters;
    int *ew_tau_authority;
    size_t ew_tau_authority_count;

    OpacityState opacity;
    double *tau_sobolev;
    double *line_source;
    A208Validity *tau_validity;
    A208Validity *line_source_validity;
    size_t n_line_values;
    int opacity_active;

    BFOpacity bf;
    int bf_active;

    double *n_atom;
    double *internal_energy_density;
    double *internal_energy_atom;
    CmfgenAdiabaticCell *adiabatic;
    int adiabatic_active;
    int bundle_complete;

    size_t n_shells;
    size_t n_ion_values;
    size_t n_level_values;
    uint64_t required_te_generation;
    uint64_t required_population_generation;
    PopulationStatus population_status;
    NLTEPopulationSolveDiagnostic solve_diagnostic;
    NLTEPopulationCandidateStatus status;
    int active;
} NLTEPopulationCandidate;

typedef struct {
    const NLTEConfig *public_nlte;
    const AtomicData *public_atom;
    const PlasmaState *public_plasma;
    const OpacityState *public_opacity;
    const double *trial_te;
    const double *radius_cm;
    const double *velocity_cm_s;
    size_t n_shells;
    double time_explosion;
    uint64_t required_te_generation;
    uint64_t required_population_generation;
    GammaDeposition *gamma_dep;
} NLTEPopulationBundleRequest;

/* Creates a side-effect-free candidate view only; it does not run the solver
 * or authorize a generation commit.  On failure candidate is empty and all
 * public objects are byte-preserved. */
NLTEPopulationCandidateStatus nlte_population_candidate_begin(
    NLTEPopulationCandidate *candidate,
    const NLTEConfig *public_nlte,
    const AtomicData *public_atom,
    const PlasmaState *public_plasma,
    const double *trial_te,
    size_t n_shells,
    uint64_t required_te_generation,
    uint64_t required_population_generation);

/* Build partition functions and within-super-level fractions as one private
 * candidate substage.  A failure changes only candidate status; both private
 * output arrays and stamps retain their pre-call bytes. */
NLTEPopulationCandidateStatus nlte_population_candidate_prepare_thermodynamics(
    NLTEPopulationCandidate *candidate);

/* Recompute n_e and the external ion-stage owner at the trial temperature,
 * entirely inside the candidate.  This precedes the shared level-SE core so
 * its ion-lock conservation rows never consume totals cloned at another T_e.
 * No population generation is committed by this substage. */
NLTEPopulationCandidateStatus nlte_population_candidate_prepare_ionization(
    NLTEPopulationCandidate *candidate);

/* Diagnostic A/B only: replace the public-cloned pre-core Sobolev slab with
 * an LTE excitation seed built from the trial T_e and trial ionization.  This
 * is not the final population--tau fixed point and grants no commit authority;
 * it exists only to falsify or confirm the lagged-tau dependency in the
 * mode-3 Sobolev SE matrix. */
NLTEPopulationCandidateStatus nlte_population_candidate_prepare_precore_tau_seed(
    NLTEPopulationCandidate *candidate,
    double time_explosion);

/* Clone the mutable tau/source slab and scalar generations.  Immutable line
 * tables are borrowed.  The CPU opacity/emissivity publication owners are
 * detached: no public allocation is borrowed, consumed, or freed. */
NLTEPopulationCandidateStatus nlte_population_candidate_prepare_opacity_view(
    NLTEPopulationCandidate *candidate,
    const OpacityState *public_opacity);

/* Execute the shared A2-07 population/CE core on the private candidate view.
 * This stops before ion writeback, tau/source production, diagnostics, or any
 * public commit.  opacity and gamma are borrowed read-only producer inputs. */
NLTEPopulationCandidateStatus nlte_population_candidate_solve_core(
    NLTEPopulationCandidate *candidate,
    double time_explosion,
    GammaDeposition *gamma_dep);

/* Post-core private producer: optional ion-stage writeback followed by the
 * signed tau/source refresh using candidate-local EW authority. */
NLTEPopulationCandidateStatus nlte_population_candidate_produce_tau_source(
    NLTEPopulationCandidate *candidate,
    double time_explosion);

/* Build a candidate-owned BF(+FF) grid from the committed candidate
 * population.  The candidate route is CPU-only and suppresses observers; it
 * never consumes or mutates a public BFOpacity object. */
NLTEPopulationCandidateStatus nlte_population_candidate_prepare_bf(
    NLTEPopulationCandidate *candidate);

/* Publish A2-08/A2-09 into the detached candidate OpacityState owners.  Both
 * stages use candidate-local counters and explicit EW tau authority. */
NLTEPopulationCandidateStatus nlte_population_candidate_produce_publications(
    NLTEPopulationCandidate *candidate,
    double time_explosion);

/* Produce neutral-ground-reference internal energy and the complete signed
 * CMFGEN homologous adiabatic vector from the same all-shell population
 * generation.  radius and velocity are borrowed read-only for this call. */
NLTEPopulationCandidateStatus nlte_population_candidate_prepare_adiabatic(
    NLTEPopulationCandidate *candidate,
    const double *radius_cm,
    const double *velocity_cm_s,
    double epoch_s);

/* Execute the full private producer chain.  Success grants no public commit
 * authority; it only marks bundle_complete after every all-shell product is
 * generation-consistent. */
NLTEPopulationCandidateStatus nlte_population_candidate_build_bundle(
    NLTEPopulationCandidate *candidate,
    const NLTEPopulationBundleRequest *request);

/* A2-INIT seed-only material commit: publish the candidate material bundle
 * (ion/ne/level populations, tau/source, BF, A2-08, A2-09) at the CURRENT
 * published seed Te generation.  The preflight proves the trial Te is
 * byte-identical to the public seed Te and its publication manifest, and the
 * commit never writes plasma->T_e, opacity->t_electrons, the Te generation,
 * or plasma->te_publication — no radiative-equilibrium Te and no A2-10 ledger
 * is fabricated or consumed.  Failure preserves every public owner
 * byte-for-byte. */
NLTEPopulationCandidateStatus nlte_population_candidate_commit_seed_material(
    NLTEPopulationCandidate *candidate,
    NLTEConfig *public_nlte,
    AtomicData *public_atom,
    PlasmaState *public_plasma,
    OpacityState *public_opacity,
    BFOpacity *public_bf);

/* Preflight the complete replay bundle and the uncommitted A2-10 root, then
 * publish all material arrays and allocation owners with no fallible work
 * after the first public byte changes. */
NLTEPopulationCandidateStatus nlte_population_candidate_commit_bundle(
    NLTEPopulationCandidate *candidate,
    ElectronTemperaturePublication *te_candidate,
    NLTEConfig *public_nlte,
    AtomicData *public_atom,
    PlasmaState *public_plasma,
    OpacityState *public_opacity,
    BFOpacity *public_bf);

void nlte_population_candidate_free(NLTEPopulationCandidate *candidate);

const char *nlte_population_candidate_status_name(
    NLTEPopulationCandidateStatus status);

void nlte_population_solve_diagnostic_reset(
    NLTEPopulationSolveDiagnostic *diagnostic);

const char *nlte_population_solve_stage_name(
    NLTEPopulationSolveStage stage);

const char *nlte_ion_solve_status_name(NLTEIonSolveStatus status);

#endif
