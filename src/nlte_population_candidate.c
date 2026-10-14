#include "nlte_population_candidate.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int multiply_size(size_t a, size_t b, size_t *out)
{
    if (!out || (a != 0 && b > SIZE_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

void nlte_population_solve_diagnostic_reset(
    NLTEPopulationSolveDiagnostic *diagnostic)
{
    if (!diagnostic) return;
    memset(diagnostic, 0, sizeof(*diagnostic));
    diagnostic->ce_iteration = -1;
    diagnostic->pair_index = -1;
    diagnostic->pair_lo_slot = -1;
    diagnostic->pair_hi_slot = -1;
    diagnostic->Z = -1;
    diagnostic->ion_lo = -1;
    diagnostic->ion_hi = -1;
    diagnostic->shell = -1;
    diagnostic->worst_ion_slot = -1;
    diagnostic->worst_shell = -1;
    diagnostic->ion_lock_active = -1;
    diagnostic->solve_level_index = -1;
    diagnostic->super_level_index = -1;
    diagnostic->anchor_global_level = -1;
    diagnostic->level_number = -1;
    diagnostic->statistical_weight = -1;
    diagnostic->trial_temperature = NAN;
    diagnostic->electron_density = NAN;
    diagnostic->level_energy_eV = NAN;
    diagnostic->population_value = NAN;
    diagnostic->vector_min = NAN;
    diagnostic->vector_max = NAN;
    diagnostic->vector_sum = NAN;
    diagnostic->vector_abs_max = NAN;
    diagnostic->negative_relative_scale = NAN;
    diagnostic->pair_total_density = NAN;
    diagnostic->lo_target_density = NAN;
    diagnostic->hi_target_density = NAN;
    diagnostic->solved_lo_sum = NAN;
    diagnostic->solved_hi_sum = NAN;
    diagnostic->linear_rank = -1;
    diagnostic->equilibration_iterations = -1;
    diagnostic->refinement_iterations = -1;
    diagnostic->pivot_growth = NAN;
    diagnostic->initial_backward_error = NAN;
    diagnostic->final_backward_error = NAN;
}

const char *nlte_population_solve_stage_name(
    NLTEPopulationSolveStage stage)
{
    static const char *const names[] = {
        "NONE", "PRECONDITION", "TRANSACTION", "THERMODYNAMIC",
        "IONIZATION", "WORKSPACE", "ELEMENT_WIDE", "PAIR_SOLVE",
        "CE_NONCONVERGED", "PUBLISH", "INTERNAL"
    };
    size_t index = (size_t)stage;
    return index < sizeof(names) / sizeof(names[0]) ? names[index]
                                                    : "UNKNOWN_STAGE";
}

const char *nlte_ion_solve_status_name(NLTEIonSolveStatus status)
{
    static const char *const names[] = {
        "OK", "INVALID_LAYOUT", "ALLOCATION_FAILED", "ASSEMBLY_FAILED",
        "RANK_INCOMPLETE", "LINEAR_FAILED", "NONFINITE",
        "NEGATIVE_POPULATION"
    };
    size_t index = (size_t)status;
    return index < sizeof(names) / sizeof(names[0]) ? names[index]
                                                    : "UNKNOWN_ION_STATUS";
}

static double *copy_values(const double *source, size_t count)
{
    if (count == 0) return NULL;
    if (!source || count > SIZE_MAX / sizeof(double)) return NULL;
    double *copy = malloc(count * sizeof(*copy));
    if (copy) memcpy(copy, source, count * sizeof(*copy));
    return copy;
}

static A208Validity *copy_validity_values(
    const A208Validity *source, size_t count)
{
    if (count == 0) return NULL;
    if (!source || count > SIZE_MAX / sizeof(*source)) return NULL;
    A208Validity *copy = malloc(count * sizeof(*copy));
    if (copy) memcpy(copy, source, count * sizeof(*copy));
    return copy;
}

void nlte_physical_history_free(NLTEPhysicalHistory *history)
{
    if(!history)return;
    free(history->storage);free(history->level_global);
    free(history->ion_Z);free(history->ion_stage);
    memset(history,0,sizeof(*history));
}

NLTEPopulationCandidateStatus nlte_physical_history_capture(
    NLTEPhysicalHistory *history,const NLTEConfig *nlte,const AtomicData *atom,
    const PlasmaState *plasma,double epoch_s)
{
    if(!history || !nlte || !atom || !plasma || !isfinite(epoch_s) || epoch_s<=0 ||
       plasma->n_shells<=0 || atom->n_ion_pops<=0 || nlte->n_nlte_levels_total<=0 ||
       !plasma->rho || !plasma->T_e || !plasma->n_electron ||
       !nlte->nlte_to_global_level || !nlte->nlte_level_populations ||
       !atom->ion_pop_Z || !atom->ion_pop_stage || !atom->ion_number_density)
        return NLTE_CANDIDATE_INVALID_ARGUMENT;
    size_t ns=(size_t)plasma->n_shells,ni=(size_t)atom->n_ion_pops,
           nl=(size_t)nlte->n_nlte_levels_total,count;
    if(ni>SIZE_MAX-nl-5 || multiply_size(5+ni+nl,ns,&count) ||
       count>SIZE_MAX/sizeof(double) || ni>SIZE_MAX/sizeof(int) ||
       nl>SIZE_MAX/sizeof(int))return NLTE_CANDIDATE_INVALID_LAYOUT;
    NLTEPhysicalHistory trial={0};
    AtomicInternalEnergyCell *energy=calloc(ns,sizeof(*energy));
    trial.storage=malloc(count*sizeof(double));
    trial.level_global=malloc(nl*sizeof(int));
    trial.ion_Z=malloc(ni*sizeof(int));trial.ion_stage=malloc(ni*sizeof(int));
    NLTEPopulationCandidateStatus status=NLTE_CANDIDATE_ALLOCATION_FAILED;
    if(!energy || !trial.storage || !trial.level_global || !trial.ion_Z || !trial.ion_stage)
        goto history_done;
    status=NLTE_CANDIDATE_INTERNAL_ENERGY_FAILED;
    if(atomic_internal_energy_build(atom,nlte,plasma,ns,
           atom->population_committed_generation,plasma->T_e_generation,energy)!=
           ATOMIC_INTERNAL_ENERGY_OK)goto history_done;
    for(size_t s=0;s<ns;++s){
        if(!isfinite(plasma->rho[s]) || plasma->rho[s]<=0 ||
           !isfinite(plasma->n_electron[s]) || plasma->n_electron[s]<0)
            goto history_done;
        trial.storage[s]=plasma->rho[s];trial.storage[ns+s]=plasma->T_e[s];
        trial.storage[2*ns+s]=energy[s].n_atom_cm3;
        trial.storage[3*ns+s]=plasma->n_electron[s];
        trial.storage[4*ns+s]=energy[s].internal_energy_atom_erg;
    }
    trial.energy=(CmfgenMaterialEnergyState){ns,epoch_s,trial.storage,trial.storage+ns,
        trial.storage+2*ns,trial.storage+3*ns,trial.storage+4*ns};
    trial.n_ions=ni;trial.n_levels=nl;
    trial.population_generation=atom->population_committed_generation;
    trial.temperature_generation=plasma->T_e_generation;
    trial.ion_density=trial.storage+5*ns;trial.level_density=trial.ion_density+ni*ns;
    memcpy(trial.ion_density,atom->ion_number_density,ni*ns*sizeof(double));
    memcpy(trial.level_density,nlte->nlte_level_populations,nl*ns*sizeof(double));
    memcpy(trial.level_global,nlte->nlte_to_global_level,nl*sizeof(int));
    memcpy(trial.ion_Z,atom->ion_pop_Z,ni*sizeof(int));
    memcpy(trial.ion_stage,atom->ion_pop_stage,ni*sizeof(int));
    for(size_t i=0;i<ni*ns+nl*ns;++i)
        if(!(trial.ion_density[i]>=0) || !isfinite(trial.ion_density[i]))goto history_done;
    nlte_physical_history_free(history);*history=trial;
    memset(&trial,0,sizeof(trial));status=NLTE_CANDIDATE_OK;
history_done:
    free(energy);nlte_physical_history_free(&trial);return status;
}

static PopulationStatus physical_history_solve_impl(
    const NLTEConfig *nlte,const AtomicData *atom,const PlasmaState *plasma,
    size_t shell,double epoch_s,size_t first,size_t n,
    int terminal_ion,const double *generator,double *solution,PopulationGeneratorSolveDiagnostic *diagnostic)
{
    if(!nlte||!atom||!plasma||!plasma->physical_history||!plasma->rho||
       !generator||!solution||!n||n>SIZE_MAX/sizeof(double)||
       plasma->n_shells<=0||nlte->n_nlte_levels_total<=0||atom->n_ion_pops<=0||
       shell>=(size_t)plasma->n_shells||first>SIZE_MAX-n||
       !nlte->nlte_to_global_level||!nlte->fl_to_super||
       !atom->ion_pop_Z||!atom->ion_pop_stage)return POP_ATOMIC_MISSING;
    const NLTEPhysicalHistory *h=plasma->physical_history;
    size_t ns=(size_t)plasma->n_shells;
    if(h->energy.n_shells!=ns||h->n_levels!=(size_t)nlte->n_nlte_levels_total||
       h->n_ions!=(size_t)atom->n_ion_pops||!h->level_density||!h->level_global||
       !h->energy.density_g_cm3||!h->ion_Z||!h->ion_stage||
       !isfinite(epoch_s)||!(epoch_s>h->energy.epoch_s)||
       !(h->energy.epoch_s>0)||!isfinite(h->energy.epoch_s))return POP_ATOMIC_MISSING;
    if(memcmp(h->level_global,nlte->nlte_to_global_level,h->n_levels*sizeof(int))||
       memcmp(h->ion_Z,atom->ion_pop_Z,h->n_ions*sizeof(int))||
       memcmp(h->ion_stage,atom->ion_pop_stage,h->n_ions*sizeof(int)))return POP_ATOMIC_MISSING;
    double oldrho=h->energy.density_g_cm3[shell],newrho=plasma->rho[shell];
    if(!(oldrho>0)||!isfinite(oldrho)||!(newrho>0)||!isfinite(newrho))return POP_NONFINITE;
    double ratio=newrho/oldrho,expected=pow(h->energy.epoch_s/epoch_s,3);
    if(!(expected>0)||!isfinite(ratio)||fabs(ratio/expected-1)>1e-10)return POP_SOLVE_FAILED;
    double *old=calloc(n,sizeof(*old));unsigned char *covered=calloc(n,sizeof(*covered));
    if(!old||!covered){free(old);free(covered);return POP_SOLVE_FAILED;}
    PopulationStatus status=POP_OK;
    size_t bound_n=n;
    if(terminal_ion>=0){
        if((size_t)terminal_ion>=h->n_ions||!h->ion_density||!atom->level_offset||n<2||
           atom->level_offset[terminal_ion]!=atom->level_offset[terminal_ion+1]){
            status=POP_ATOMIC_MISSING;goto time_levels_done;
        }
        bound_n--;
        old[n-1]=h->ion_density[(size_t)terminal_ion*ns+shell];covered[n-1]=1;
        if(!isfinite(old[n-1])||old[n-1]<0){status=POP_NONFINITE;goto time_levels_done;}
    }
    for(size_t g=0;g<h->n_levels;g++){
        int sl=nlte->fl_to_super[g];
        if(sl<0){status=POP_ATOMIC_MISSING;goto time_levels_done;}
        if((size_t)sl<first||(size_t)sl>=first+bound_n)continue;
        double value=h->level_density[g*ns+shell];
        if(value<0||!isfinite(value)){status=POP_NONFINITE;goto time_levels_done;}
        old[(size_t)sl-first]+=value;covered[(size_t)sl-first]=1;
    }
    for(size_t i=0;i<n;i++)if(!covered[i]){status=POP_ATOMIC_MISSING;goto time_levels_done;}
    status=population_generator_implicit_step(generator,n,old,ratio,
        epoch_s-h->energy.epoch_s,solution,diagnostic);
time_levels_done:
    free(old);free(covered);return status;
}

PopulationStatus nlte_physical_history_solve_levels(
    const NLTEConfig *n,const AtomicData *a,const PlasmaState *p,
    size_t shell,double epoch,size_t first,size_t count,const double *A,double *x,
    PopulationGeneratorSolveDiagnostic *d){
    return physical_history_solve_impl(n,a,p,shell,epoch,first,count,-1,A,x,d);
}

PopulationStatus nlte_physical_history_solve_element(
    const NLTEConfig *n,const AtomicData *a,const PlasmaState *p,
    size_t shell,double epoch,size_t first,size_t count,int terminal_ion,
    const double *A,double *x){
    return physical_history_solve_impl(n,a,p,shell,epoch,first,count,terminal_ion,A,x,NULL);
}

static void candidate_bf_free(BFOpacity *bf)
{
    if (!bf) return;
    free(bf->chi_bf);
    free(bf->eta_bf);
    free(bf->activation_level);
    free(bf->event_level_offset);
    free(bf->event_element);
    free(bf->event_ion);
    free(bf->event_level);
    free(bf->event_target);
    free(bf->event_target_fallback);
    free(bf->event_has_sigma);
    free(bf->event_nu_edge);
    free(bf->event_sigma0);
    free(bf->event_weight);
    free(bf->event_stim_ratio);
    free(bf->event_chi_bf);
    free(bf->event_chi_bf_comparison);
    free(bf->event_Te);
    memset(bf, 0, sizeof(*bf));
}

static PopulationAtomicView candidate_atomic_view(const AtomicData *atom)
{
    PopulationAtomicView view;
    memset(&view, 0, sizeof(view));
    if (!atom) return view;
    view.n_ions = (size_t)atom->n_ion_pops;
    view.n_levels = (size_t)atom->n_levels;
    view.level_offset = atom->level_offset;
    view.energy_eV = atom->level_energy_eV;
    view.g = atom->level_g;
    view.level_Z = atom->level_Z;
    view.level_ion = atom->level_ion;
    view.topion_n = atom->topion_n;
    view.topion_ion_index = atom->topion_ion_index;
    view.topion_E_cm = atom->topion_E_cm;
    view.topion_g = atom->topion_g;
    return view;
}

void nlte_population_candidate_free(NLTEPopulationCandidate *candidate)
{
    if (!candidate) return;
    a208_publication_free(&candidate->opacity.cpu_opacity);
    a209_publication_free(&candidate->opacity.cpu_emissivity);
    candidate_bf_free(&candidate->bf);
    free(candidate->trial_te);
    free(candidate->ion_population);
    free(candidate->level_population);
    free(candidate->electron_density);
    free(candidate->partition);
    free(candidate->within_sl_fraction);
    free(candidate->ew_tau_authority);
    free(candidate->tau_sobolev);
    free(candidate->line_source);
    free(candidate->tau_validity);
    free(candidate->line_source_validity);
    free(candidate->n_atom);
    free(candidate->internal_energy_density);
    free(candidate->internal_energy_atom);
    free(candidate->adiabatic);
    free(candidate->time_work);
    memset(candidate, 0, sizeof(*candidate));
}

NLTEPopulationCandidateStatus nlte_population_candidate_begin(
    NLTEPopulationCandidate *candidate,
    const NLTEConfig *public_nlte,
    const AtomicData *public_atom,
    const PlasmaState *public_plasma,
    const double *trial_te,
    size_t n_shells,
    uint64_t required_te_generation,
    uint64_t required_population_generation)
{
    if (!candidate) return NLTE_CANDIDATE_INVALID_ARGUMENT;
    memset(candidate, 0, sizeof(*candidate));
    if (!public_nlte || !public_atom || !public_plasma || !trial_te ||
        n_shells == 0)
        return candidate->status = NLTE_CANDIDATE_INVALID_ARGUMENT;
    if (required_te_generation == 0 || required_population_generation == 0)
        return candidate->status = NLTE_CANDIDATE_INVALID_GENERATION;
    if (public_plasma->n_shells <= 0 ||
        (size_t)public_plasma->n_shells != n_shells ||
        public_atom->n_ion_pops <= 0 ||
        public_nlte->n_nlte_levels_total <= 0)
        return candidate->status = NLTE_CANDIDATE_INVALID_LAYOUT;

    size_t n_ion_values = 0;
    size_t n_level_values = 0;
    if (multiply_size((size_t)public_atom->n_ion_pops, n_shells,
                      &n_ion_values) != 0 ||
        multiply_size((size_t)public_nlte->n_nlte_levels_total, n_shells,
                      &n_level_values) != 0 ||
        !public_atom->ion_number_density ||
        !public_atom->partition_functions ||
        !public_nlte->nlte_level_populations ||
        !public_nlte->within_sl_frac ||
        !public_plasma->n_electron)
        return candidate->status = NLTE_CANDIDATE_INVALID_LAYOUT;
    for (size_t s = 0; s < n_shells; ++s)
        if (!isfinite(trial_te[s]) || trial_te[s] <= 0.0)
            return candidate->status = NLTE_CANDIDATE_INVALID_TEMPERATURE;

    candidate->trial_te = copy_values(trial_te, n_shells);
    candidate->ion_population = copy_values(
        public_atom->ion_number_density, n_ion_values);
    candidate->level_population = copy_values(
        public_nlte->nlte_level_populations, n_level_values);
    candidate->electron_density = copy_values(
        public_plasma->n_electron, n_shells);
    candidate->partition = copy_values(
        public_atom->partition_functions, n_ion_values);
    candidate->within_sl_fraction = copy_values(
        public_nlte->within_sl_frac, n_level_values);
    if (!candidate->trial_te || !candidate->ion_population ||
        !candidate->level_population || !candidate->electron_density ||
        !candidate->partition || !candidate->within_sl_fraction) {
        nlte_population_candidate_free(candidate);
        return candidate->status = NLTE_CANDIDATE_ALLOCATION_FAILED;
    }

    candidate->nlte = *public_nlte;
    candidate->atom = *public_atom;
    candidate->plasma = *public_plasma;
    candidate->n_shells = n_shells;
    candidate->n_ion_values = n_ion_values;
    candidate->n_level_values = n_level_values;
    candidate->required_te_generation = required_te_generation;
    candidate->required_population_generation = required_population_generation;
    candidate->population_status = POP_OK;
    nlte_population_solve_diagnostic_reset(&candidate->solve_diagnostic);
    candidate->ionization_worst_ne_shell = -1;
    candidate->ionization_worst_ion_index = -1;
    candidate->ionization_worst_ion_shell = -1;
    candidate->ionization_worst_charge_shell = -1;
    candidate->ionization_failure_shell = -1;
    candidate->ionization_failure_element = -1;
    candidate->ionization_failure_Z = -1;
    candidate->ionization_failure_ip_cur = -1;
    candidate->ionization_failure_ip_next = -1;
    candidate->ionization_failure_stage = -1;
    candidate->ionization_failure_level = -1;
    candidate->ionization_failure_level_number = -1;
    candidate->ionization_failure_bf_state = -1;
    candidate->ionization_failure_has_sigma = -1;
    candidate->ionization_failure_trial_temperature = NAN;
    candidate->ionization_failure_electron_density = NAN;
    candidate->ionization_failure_chi_eV = NAN;
    candidate->ionization_failure_level_energy_eV = NAN;
    candidate->ionization_failure_nu_threshold = NAN;
    candidate->ionization_failure_sigma_max = NAN;
    candidate->ionization_failure_w_miss = NAN;

    candidate->plasma.T_e = candidate->trial_te;
    candidate->plasma.n_electron = candidate->electron_density;
    candidate->plasma.T_e_generation = required_te_generation;
    candidate->plasma.population_last_error = POP_OK;
    candidate->plasma.population_error_count = 0;

    candidate->atom.ion_number_density = candidate->ion_population;
    candidate->atom.partition_functions = candidate->partition;

    candidate->nlte.nlte_level_populations = candidate->level_population;
    candidate->nlte.within_sl_frac = candidate->within_sl_fraction;
    candidate->nlte.population_required_generation =
        required_population_generation;
    candidate->nlte.population_first_error = POP_OK;
    candidate->nlte.population_error_count = 0;
    memset(&candidate->nlte.population_counters, 0,
           sizeof(candidate->nlte.population_counters));
    candidate->nlte.population_counters.pop_generation_required =
        required_population_generation;
    candidate->nlte.ew_runtime_counts_sink = &candidate->ew_runtime_counts;
    candidate->nlte.solve_effect_policy_explicit = 1;
    candidate->nlte.solve_effects_allowed = 0;

    /* Stamps remain private because they live in the shallow object copies.
     * The solve core replaces them with trial-token stamps before consuming
     * within_sl_fraction.  No public stamp is touched by this view. */
    candidate->status = NLTE_CANDIDATE_OK;
    candidate->active = 1;
    return NLTE_CANDIDATE_OK;
}

NLTEPopulationCandidateStatus nlte_population_candidate_prepare_opacity_view(
    NLTEPopulationCandidate *candidate,
    const OpacityState *public_opacity)
{
    if (!candidate || !candidate->active || !public_opacity ||
        candidate->opacity_active || public_opacity->n_lines < 0 ||
        public_opacity->n_shells <= 0 ||
        (size_t)public_opacity->n_shells != candidate->n_shells)
        return NLTE_CANDIDATE_INVALID_ARGUMENT;
    size_t n_lines = (size_t)public_opacity->n_lines;
    if (n_lines > 0 && candidate->n_shells > SIZE_MAX / n_lines) {
        candidate->status = NLTE_CANDIDATE_OPACITY_FAILED;
        candidate->active = 0;
        return candidate->status;
    }
    size_t count = n_lines * candidate->n_shells;
    if (count == 0 || !public_opacity->tau_sobolev ||
        !public_opacity->line_source_S || !public_opacity->tau_validity ||
        !public_opacity->line_source_validity) {
        candidate->status = NLTE_CANDIDATE_OPACITY_FAILED;
        candidate->active = 0;
        return candidate->status;
    }

    double *tau = copy_values(public_opacity->tau_sobolev, count);
    double *source = copy_values(public_opacity->line_source_S, count);
    A208Validity *tau_validity = copy_validity_values(
        public_opacity->tau_validity, count);
    A208Validity *source_validity = copy_validity_values(
        public_opacity->line_source_validity, count);
    if (!tau || !source || !tau_validity || !source_validity) {
        free(tau);
        free(source);
        free(tau_validity);
        free(source_validity);
        candidate->status = NLTE_CANDIDATE_ALLOCATION_FAILED;
        candidate->active = 0;
        return candidate->status;
    }

    uint64_t opacity_generation_base =
        public_opacity->cpu_opacity.generation_required;
    candidate->opacity = *public_opacity;
    /* OpacityState is otherwise a shallow immutable-input view.  These two
     * subobjects are allocation owners, so retaining their public pointers
     * would let a candidate commit/free destroy the committed publication. */
    memset(&candidate->opacity.cpu_opacity, 0,
           sizeof(candidate->opacity.cpu_opacity));
    memset(&candidate->opacity.cpu_emissivity, 0,
           sizeof(candidate->opacity.cpu_emissivity));
    candidate->opacity.cpu_opacity.generation_required =
        opacity_generation_base;
    candidate->tau_sobolev = tau;
    candidate->line_source = source;
    candidate->tau_validity = tau_validity;
    candidate->line_source_validity = source_validity;
    candidate->n_line_values = count;
    candidate->opacity.tau_sobolev = candidate->tau_sobolev;
    candidate->opacity.line_source_S = candidate->line_source;
    candidate->opacity.tau_validity = candidate->tau_validity;
    candidate->opacity.line_source_validity =
        candidate->line_source_validity;
    candidate->opacity.electron_density = candidate->electron_density;
    candidate->opacity.t_electrons = candidate->trial_te;
    candidate->opacity_active = 1;
    candidate->status = NLTE_CANDIDATE_OK;
    return NLTE_CANDIDATE_OK;
}

NLTEPopulationCandidateStatus nlte_population_candidate_prepare_thermodynamics(
    NLTEPopulationCandidate *candidate)
{
    if (!candidate || !candidate->active)
        return NLTE_CANDIDATE_INVALID_ARGUMENT;
    double *partition_work = copy_values(
        candidate->partition, candidate->n_ion_values);
    double *fraction_work = copy_values(
        candidate->within_sl_fraction, candidate->n_level_values);
    if (!partition_work || !fraction_work) {
        free(partition_work);
        free(fraction_work);
        candidate->population_status = POP_SOLVE_FAILED;
        candidate->status = NLTE_CANDIDATE_ALLOCATION_FAILED;
        candidate->active = 0;
        return candidate->status;
    }

    PopulationAtomicView view = candidate_atomic_view(&candidate->atom);
    PopulationDerivedStamp partition_stamp;
    PopulationDerivedStamp fraction_stamp;
    PopulationStatus status = population_partition_build(
        &view, candidate->trial_te, candidate->n_shells,
        candidate->required_population_generation,
        candidate->required_te_generation,
        partition_work, &partition_stamp);
    if (status == POP_OK) {
        status = population_within_superlevel_build(
            &partition_stamp, &view, candidate->trial_te,
            candidate->n_shells,
            candidate->required_population_generation,
            candidate->required_te_generation,
            (size_t)candidate->nlte.n_nlte_levels_total,
            candidate->nlte.super_mode,
            (size_t)(candidate->nlte.n_super_total > 0 ?
                     candidate->nlte.n_super_total : 0),
            candidate->nlte.nlte_to_global_level,
            candidate->nlte.fl_to_super,
            candidate->nlte.super_anchor_global,
            fraction_work, &fraction_stamp);
    }
    if (status != POP_OK) {
        free(partition_work);
        free(fraction_work);
        candidate->population_status = status;
        candidate->status = NLTE_CANDIDATE_THERMODYNAMIC_FAILED;
        candidate->active = 0;
        return candidate->status;
    }

    memcpy(candidate->partition, partition_work,
           candidate->n_ion_values * sizeof(*candidate->partition));
    memcpy(candidate->within_sl_fraction, fraction_work,
           candidate->n_level_values * sizeof(*candidate->within_sl_fraction));
    candidate->atom.partition_stamp = partition_stamp;
    candidate->nlte.within_sl_stamp = fraction_stamp;
    candidate->population_status = POP_OK;
    candidate->status = NLTE_CANDIDATE_OK;
    free(partition_work);
    free(fraction_work);
    return NLTE_CANDIDATE_OK;
}

NLTEPopulationCandidateStatus nlte_population_candidate_prepare_adiabatic(
        NLTEPopulationCandidate *candidate,
        const double *radius_cm,
        const double *velocity_cm_s,
        double epoch_s)
{
    if (!candidate || !candidate->active || candidate->adiabatic_active ||
        !radius_cm || !velocity_cm_s || !isfinite(epoch_s) ||
        epoch_s <= 0.0 || candidate->n_shells < 2 ||
        candidate->atom.population_committed_generation !=
            candidate->required_population_generation ||
        candidate->nlte.population_committed_generation !=
            candidate->required_population_generation) {
        if (candidate) {
            candidate->population_status = POP_STALE_DERIVED_TEMPERATURE;
            candidate->status = NLTE_CANDIDATE_ADIABATIC_FAILED;
            candidate->active = 0;
        }
        return NLTE_CANDIDATE_ADIABATIC_FAILED;
    }
    size_t n = candidate->n_shells;
    AtomicInternalEnergyCell *energy = calloc(n, sizeof(*energy));
    double *n_atom = calloc(n, sizeof(*n_atom));
    double *energy_density = calloc(n, sizeof(*energy_density));
    double *energy_atom = calloc(n, sizeof(*energy_atom));
    const NLTEPhysicalHistory *history=candidate->plasma.physical_history;
    CmfgenAdiabaticCell *adiabatic = history?NULL:calloc(n, sizeof(*adiabatic));
    CmfgenMaterialTimeWork *time_work=history?calloc(n,sizeof(*time_work)):NULL;
    if (!energy || !n_atom || !energy_density || !energy_atom ||
        (history?!time_work:!adiabatic)) {
        free(energy); free(n_atom); free(energy_density);
        free(energy_atom); free(adiabatic);free(time_work);
        candidate->population_status = POP_SOLVE_FAILED;
        candidate->status = NLTE_CANDIDATE_INTERNAL_ENERGY_FAILED;
        candidate->active = 0;
        return candidate->status;
    }
    AtomicInternalEnergyStatus energy_status = atomic_internal_energy_build(
        &candidate->atom, &candidate->nlte, &candidate->plasma, n,
        candidate->required_population_generation,
        candidate->required_te_generation, energy);
    if (energy_status != ATOMIC_INTERNAL_ENERGY_OK) {
        fprintf(stderr,
                "[NLTE-CANDIDATE-INTERNAL-ENERGY][INVALID] status=%s "
                "required_population_generation=%llu "
                "required_te_generation=%llu n_shells=%zu "
                "ionization_rows=%d reference_rows=%d\n",
                atomic_internal_energy_status_name(energy_status),
                (unsigned long long)candidate->required_population_generation,
                (unsigned long long)candidate->required_te_generation, n,
                candidate->atom.n_ionization,
                candidate->atom.n_ionization_reference);
        free(energy); free(n_atom); free(energy_density);
        free(energy_atom); free(adiabatic);free(time_work);
        candidate->population_status =
            energy_status == ATOMIC_INTERNAL_ENERGY_POPULATION_CLOSURE
          ? POP_SOLVE_FAILED : POP_ATOMIC_MISSING;
        candidate->status = NLTE_CANDIDATE_INTERNAL_ENERGY_FAILED;
        candidate->active = 0;
        return candidate->status;
    }
    for (size_t s = 0; s < n; ++s) {
        n_atom[s] = energy[s].n_atom_cm3;
        energy_density[s] = energy[s].energy_density_erg_cm3;
        energy_atom[s] = energy[s].internal_energy_atom_erg;
    }
    CmfgenAdiabaticInput input = {
        n, epoch_s, radius_cm, velocity_cm_s, candidate->trial_te,
        n_atom, candidate->electron_density, energy_atom
    };
    CmfgenAdiabaticStatus adiabatic_status;
    if(history){
        CmfgenMaterialEnergyState current={n,epoch_s,candidate->plasma.rho,
            candidate->trial_te,n_atom,candidate->electron_density,energy_atom};
        adiabatic_status=cmfgen_material_time_work(&current,&history->energy,time_work);
        for(size_t s=0;s<n && adiabatic_status==CMFGEN_ADIABATIC_OK;++s)
            if(!isfinite(radius_cm[s]) || radius_cm[s]<=0 ||
               !isfinite(velocity_cm_s[s]) || velocity_cm_s[s]<=0 ||
               (s && radius_cm[s]<=radius_cm[s-1]) ||
               fabs(velocity_cm_s[s]*epoch_s/radius_cm[s]-1)>1e-10)
                adiabatic_status=CMFGEN_ADIABATIC_NON_HOMOLOGOUS;
    }else adiabatic_status=cmfgen_adiabatic_v3_homologous_evaluate(&input, adiabatic);
    free(energy);
    if (adiabatic_status != CMFGEN_ADIABATIC_OK) {
        free(n_atom); free(energy_density); free(energy_atom); free(adiabatic);free(time_work);
        candidate->population_status = POP_SOLVE_FAILED;
        candidate->status = NLTE_CANDIDATE_ADIABATIC_FAILED;
        candidate->active = 0;
        return candidate->status;
    }
    candidate->n_atom = n_atom;
    candidate->internal_energy_density = energy_density;
    candidate->internal_energy_atom = energy_atom;
    candidate->adiabatic = adiabatic;
    candidate->time_work = time_work;
    candidate->adiabatic_active = 1;
    candidate->population_status = POP_OK;
    candidate->status = NLTE_CANDIDATE_OK;
    return NLTE_CANDIDATE_OK;
}

const char *nlte_population_candidate_status_name(
    NLTEPopulationCandidateStatus status)
{
    static const char *const names[] = {
        "NLTE_CANDIDATE_OK",
        "NLTE_CANDIDATE_INVALID_ARGUMENT",
        "NLTE_CANDIDATE_INVALID_LAYOUT",
        "NLTE_CANDIDATE_INVALID_TEMPERATURE",
        "NLTE_CANDIDATE_INVALID_GENERATION",
        "NLTE_CANDIDATE_ALLOCATION_FAILED",
        "NLTE_CANDIDATE_THERMODYNAMIC_FAILED",
        "NLTE_CANDIDATE_SOLVE_FAILED",
        "NLTE_CANDIDATE_OPACITY_FAILED",
        "NLTE_CANDIDATE_INTERNAL_ENERGY_FAILED",
        "NLTE_CANDIDATE_ADIABATIC_FAILED",
        "NLTE_CANDIDATE_COMMIT_FAILED"
    };
    if (status < NLTE_CANDIDATE_OK ||
        status > NLTE_CANDIDATE_COMMIT_FAILED)
        return "NLTE_CANDIDATE_UNKNOWN";
    return names[status];
}
