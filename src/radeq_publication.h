#ifndef LUMINA_RADEQ_PUBLICATION_H
#define LUMINA_RADEQ_PUBLICATION_H

#include "population_contract.h"
#include "cmfgen_adiabatic.h"
#include "opacity_publication.h"
#include "emissivity_publication.h"
#include "line_net_rate.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Qualified production material-solver range, shared with the repository's
 * ARTIS-mirror simultaneous thermal-balance path. */
#define A210_PRODUCTION_TE_MIN_K 3500.0
#define A210_PRODUCTION_TE_MAX_K 140000.0

typedef enum {
 RADEQ_OK=1,RADEQ_EXACT_ZERO_BALANCE,RADEQ_STALE_RF,RADEQ_STALE_BF,
 RADEQ_STALE_LINE,RADEQ_STALE_POP,RADEQ_STALE_OPACITY,RADEQ_STALE_EMISSIVITY,
 RADEQ_TERM_MISSING,RADEQ_TERM_SCHEMA,RADEQ_SIGN_MISMATCH,RADEQ_INVALID_TE_TRIAL,
 RADEQ_INVALID_NE,RADEQ_ATOMIC_MISSING,RADEQ_NO_BRACKET,RADEQ_NO_ROOT,
 RADEQ_NOT_CONVERGED,RADEQ_POPULATION_NOT_CONVERGED,RADEQ_CHARGE_NOT_CONVERGED,
 RADEQ_HEAT_RESIDUAL,RADEQ_TE_MANIFEST_MISMATCH,RADEQ_TE_CONTEXT_MISMATCH,
 RADEQ_UNQUALIFIED_TE,RADEQ_FIXED_T,RADEQ_FORBIDDEN_FALLBACK,
 RADEQ_PARTIAL_PUBLISH,RADEQ_INCOMPLETE_ADIABATIC,RADEQ_NONFINITE,
 RADEQ_GAMMA_UNPUBLISHED
} RadeqStatus;
typedef enum { A210_INCLUDED=1,A210_EXACT_ZERO,A210_REPLACED_NOT_APPLICABLE,
 A210_MISSING,A210_INCOMPLETE } A210TermStatus;
typedef enum { A210_EQUATION_NONE=0,A210_RE_INTEGRAL,
 A210_EHB_THERMAL } A210EquationKind;
typedef enum { A210_ADIABATIC_NONE=0,A210_ADIABATIC_ELECTRON_TRANSLATIONAL_ONLY,
 A210_ADIABATIC_CMFGEN_COMPLETE } A210AdiabaticModel;
typedef enum { A210_PHOTO=0,A210_LINE_ABS,A210_FF_ABS,A210_COMPTON_H,A210_GAMMA,A210_NONTHERMAL,A210_ADIABATIC_H,A210_NHEAT } A210HeatTerm;
typedef enum { A210_RECOMB=0,A210_LINE_EMIT,A210_COLL_LINE,A210_FF_EMIT,A210_COMPTON_C,A210_ADIABATIC,A210_NCOOL } A210CoolTerm;

typedef struct {
 A210EquationKind equation_kind;
 A210AdiabaticModel adiabatic_model;
 double heating[A210_NHEAT],cooling[A210_NCOOL];
 A210TermStatus heating_status[A210_NHEAT],cooling_status[A210_NCOOL];
 double A_line,E_line,Q_line_rad,C_line_ce;
 double photoionization_rate;
 double adiabatic_temperature_gradient,adiabatic_velocity_divergence;
 double adiabatic_electron_fraction_gradient,adiabatic_internal_energy_gradient;
 double adiabatic_signed_total;
 int m_line,radiative_line_included,collisional_or_escape_included;
 int line_owner_overlap;
 double line_owner_closure,normalized_line_owner_closure;
 double sum_heating,sum_cooling,residual,e_balance;
 RadeqStatus status;
} A210TermLedger;

typedef struct {
 double signed_rate;
 double absolute_uncertainty;
 double absolute_signed_rate_sum;
 double scaled_emission_rate;
 double scaled_absorption_rate;
 double cancellation_condition;
 uint64_t eligible_cells,cooling_cells,heating_cells,exact_zero_cells;
 uint64_t srce_chk_cells;
 LineNetStatus status;
} A210LineNetShell;

typedef struct {
 size_t n_shells;
 uint64_t population_generation,te_generation,tau_generation;
 uint64_t opacity_generation,radiation_generation;
 const A210LineNetShell *shell;
} A210LineNetPublication;

typedef enum {
 A210_TE_LANE_UNSET = 0,
 A210_TE_LANE_FREE_T,
 A210_TE_LANE_FIXED_T
} A210TeLane;

#define A210_RE_ROOT_REASON "RE-RESIDUAL-AT-PINNED-T"

static inline const char *a210_te_lane_name(A210TeLane lane)
{
    switch (lane) {
    case A210_TE_LANE_FREE_T:  return "FREE_T";
    case A210_TE_LANE_FIXED_T: return "FIXED_T";
    default:                   return "UNSET";
    }
}

static inline int a210_sha256_text_complete(const char *s)
{
    size_t i;

    if (!s)
        return 0;

    for (i = 0; i < 64; ++i) {
        const unsigned char c = (unsigned char)s[i];
        if (!((c >= '0' && c <= '9') ||
              (c >= 'a' && c <= 'f') ||
              (c >= 'A' && c <= 'F')))
            return 0;
    }
    return s[64] == '\0';
}

static inline const char *a210_fixed_te_profile_validate(
    const double *profile,
    size_t profile_shells,
    size_t expected_shells,
    double domain_min_K,
    double domain_max_K)
{
    size_t s;

    if (profile_shells != expected_shells)
        return "RADEQ_FIXED_T_SHELL_COUNT_MISMATCH";

    if (!profile && expected_shells != 0)
        return "RADEQ_FIXED_T_NONPHYSICAL_PROFILE";

    for (s = 0; s < profile_shells; ++s) {
        const double temperature_K = profile[s];

        if (!isfinite(temperature_K) || temperature_K <= 0.0)
            return "RADEQ_FIXED_T_NONPHYSICAL_PROFILE";

        if (temperature_K < domain_min_K ||
            temperature_K > domain_max_K)
            return "RADEQ_FIXED_T_PROFILE_OUT_OF_DOMAIN";
    }

    return NULL;
}

/* Production fixed-T profile loader, exposed for its focused selftest. */
const char *a210_fixed_te_profile_load(
    const char *path, size_t n, double **out_profile, char hash[65],
    double *out_min, double *out_max);

typedef struct {
 uint64_t solve_epoch,required_te_generation,committed_te_generation;
 A210EquationKind producer_equation;
 uint64_t radfield_generation,bf_rate_generation,line_view_generation;
 uint64_t population_generation,opacity_generation,emissivity_generation;
 char atomic_model_sha256[65],geometry_sha256[65],te_manifest_sha256[65];
 char te_context_sha256[65],term_manifest_sha256[65];

 A210TeLane te_lane;
 char te_profile_sha256[65];
 size_t pinned_shells;
 int re_root_required;

 size_t n_shells;double*T_e,*n_e;RadeqStatus*shell_status,*residual_status;
 A210TermLedger*ledger;
} ElectronTemperaturePublication;

static inline const char *a210_temperature_publication_validate(
    const ElectronTemperaturePublication *te,
    size_t expected_shells)
{
    if (!te)
        return "RADEQ_TEMPERATURE_PUBLICATION_MISSING";

    if (te->te_lane == A210_TE_LANE_FIXED_T) {
        if (!a210_sha256_text_complete(te->te_profile_sha256) ||
            te->re_root_required != 0)
            return "RADEQ_FIXED_T_PUBLICATION_INCOMPLETE";

        if (te->pinned_shells != expected_shells)
            return "RADEQ_FIXED_T_SHELL_COUNT_MISMATCH";

        return NULL;
    }

    if (te->te_lane == A210_TE_LANE_FREE_T) {
        if (te->te_profile_sha256[0] != '\0' ||
            te->pinned_shells != 0 ||
            te->re_root_required != 1)
            return "RADEQ_FREE_T_PUBLICATION_LEAK";

        return NULL;
    }

    return "RADEQ_TEMPERATURE_LANE_MISSING";
}

static inline int a210_te_manifest_has_fixed_fields(
    const ElectronTemperaturePublication *te)
{
    return te && te->te_lane == A210_TE_LANE_FIXED_T;
}

typedef struct {
 uint64_t solve_epoch,te_generation_required,te_generation_committed;
 uint64_t shells_attempted,shells_converged,shells_published,trials;
 uint64_t population_trials,opacity_trials,emissivity_trials;
 uint64_t photo_heat_terms,line_heat_terms,ff_heat_terms,compton_heat_terms;
 uint64_t gamma_heat_terms,nonthermal_heat_terms,adiabatic_heat_terms;
 uint64_t recomb_cool_terms,line_cool_terms;
 uint64_t collisional_cool_terms,ff_cool_terms,compton_cool_terms,adiabatic_cool_terms;
 uint64_t exact_zero_balance,no_bracket,no_root,nonconverged,charge_nonconverged;
 uint64_t blocked_stale,blocked_missing_term,blocked_gamma_unpublished;
 uint64_t blocked_schema,blocked_sign,blocked_incomplete_adiabatic;
 uint64_t te_manifest_mismatch,te_context_mismatch,fixed_te_attempts,seed_generation_attempts;
 uint64_t line_radiative_owner_shells,line_collisional_escape_owner_shells;
 uint64_t line_replaced_collisional_terms,line_replaced_radiative_terms;
 uint64_t line_owner_overlap_shells,line_owner_closure_failures;
 uint64_t diagnostic_seed_trials,diagnostic_requested_te_trials;
 uint64_t pin_attempts,floor_attempts,neighbor_attempts,old_te_attempts;
 uint64_t fallback_attempts,partial_publish_attempts,nonfinite_failures;
 A210TeLane te_lane;
 double max_line_owner_closure,max_heat_residual;
} A210Counters;

typedef RadeqStatus (*A210ResidualFunction)(size_t shell,double trial_te,
                                            A210TermLedger*ledger,void*context);
typedef RadeqStatus (*A210VectorResidualFunction)(
    const double *trial_te, size_t n_shells,
    A210TermLedger *ledger, void *context);

/* Parse the optional diagnostic-only uniform temperature.  Return 0 when
 * absent, 1 for one finite positive value, and -1 for an invalid token. */
int a210_requested_diagnostic_te(double *temperature_K);
typedef struct {
 const CpuOpacityPublication *opacity;
 const CpuEmissivityPublication *emissivity;
 const double *j_nu;
 const double *temperature_K;
 const double *electron_density_cm3;
 const double *gamma_heating_rate;
 const CmfgenAdiabaticCell *adiabatic;
 const A210LineNetPublication *line_net;
 size_t n_shells;
} A210TrialLedgerInput;
int a210_publication_init(ElectronTemperaturePublication*p,size_t n_shells);
void a210_publication_free(ElectronTemperaturePublication*p);
RadeqStatus a210_line_owner_finalize(A210TermLedger*l);
RadeqStatus a210_apply_cmfgen_adiabatic(
    A210TermLedger *ledger, const CmfgenAdiabaticCell *cell);
/* Legacy safety API retained for focused negative controls.  Production A2-10
 * consumes A210LineNetPublication and therefore does not call this guard. */
RadeqStatus a210_signed_tau_energy_preflight(
    const A208ValueView *values, size_t count,
    uint64_t *blocked_negative_heating, size_t *first_negative);
/* Build one coherent RE_INTEGRAL ledger vector from publications produced by
 * the same private material trial.  On failure ledger_out is byte-preserved. */
RadeqStatus a210_trial_ledger_build(
    const A210TrialLedgerInput *input, A210TermLedger *ledger_out);
int a210_solve_transaction(ElectronTemperaturePublication*published,
 const double*lower,const double*upper,const double*ne,size_t n_shells,
 uint64_t solve_epoch,uint64_t required_generation,const char*geometry_sha256,
 A210ResidualFunction residual,void*context,double*public_te,double*public_ne);
/* All shells are evaluated at one coherent trial vector.  The safeguarded
 * component brackets are updated only from complete vector evaluations; no
 * callback may read a committed-neighbor temperature. */
int a210_solve_vector_transaction(
 const double *lower,const double *upper,const double *ne,size_t n_shells,
 uint64_t solve_epoch,uint64_t required_generation,
 const char *geometry_sha256,A210VectorResidualFunction residual,
 void *context,ElectronTemperaturePublication *published,
 double *public_te,double *public_ne);
/* Same coherent vector solve, but returns an uncommitted publication candidate
 * and does not advance public-publish counters or generation authority. */
int a210_solve_vector_candidate(
 const double *lower,const double *upper,const double *ne,size_t n_shells,
 uint64_t solve_epoch,uint64_t required_generation,
 const char *geometry_sha256,A210VectorResidualFunction residual,
 void *context,ElectronTemperaturePublication *candidate,
 double *candidate_te,double *candidate_ne);
RadeqStatus a210_te_context_sha256(const char te_manifest[65],
 const char geometry[65],uint64_t solve_epoch,char out[65]);
RadeqStatus a210_geometry_sha256(const double *shell_boundaries,
                                 size_t n_boundaries,char out[65]);
A210Counters*a210_counters(void);void a210_counters_reset(void);
void a210_counters_print(FILE*stream);const char*a210_status_name(RadeqStatus s);
const char*a210_equation_name(A210EquationKind kind);
const char*a210_adiabatic_model_name(A210AdiabaticModel model);

#endif
