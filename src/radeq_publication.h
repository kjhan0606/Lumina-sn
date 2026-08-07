#ifndef LUMINA_RADEQ_PUBLICATION_H
#define LUMINA_RADEQ_PUBLICATION_H

#include "population_contract.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum {
 RADEQ_OK=1,RADEQ_EXACT_ZERO_BALANCE,RADEQ_STALE_RF,RADEQ_STALE_BF,
 RADEQ_STALE_LINE,RADEQ_STALE_POP,RADEQ_STALE_OPACITY,RADEQ_STALE_EMISSIVITY,
 RADEQ_TERM_MISSING,RADEQ_TERM_SCHEMA,RADEQ_SIGN_MISMATCH,RADEQ_INVALID_TE_TRIAL,
 RADEQ_INVALID_NE,RADEQ_ATOMIC_MISSING,RADEQ_NO_BRACKET,RADEQ_NO_ROOT,
 RADEQ_NOT_CONVERGED,RADEQ_POPULATION_NOT_CONVERGED,RADEQ_CHARGE_NOT_CONVERGED,
 RADEQ_HEAT_RESIDUAL,RADEQ_TE_MANIFEST_MISMATCH,RADEQ_TE_CONTEXT_MISMATCH,
 RADEQ_UNQUALIFIED_TE,RADEQ_FIXED_T,RADEQ_FORBIDDEN_FALLBACK,
 RADEQ_PARTIAL_PUBLISH,RADEQ_NONFINITE,RADEQ_GAMMA_UNPUBLISHED
} RadeqStatus;
typedef enum { A210_INCLUDED=1,A210_EXACT_ZERO,A210_REPLACED_NOT_APPLICABLE,A210_MISSING } A210TermStatus;
typedef enum { A210_PHOTO=0,A210_LINE_ABS,A210_FF_ABS,A210_COMPTON_H,A210_GAMMA,A210_NONTHERMAL,A210_NHEAT } A210HeatTerm;
typedef enum { A210_RECOMB=0,A210_LINE_EMIT,A210_COLL_LINE,A210_FF_EMIT,A210_COMPTON_C,A210_ADIABATIC,A210_NCOOL } A210CoolTerm;

typedef struct {
 double heating[A210_NHEAT],cooling[A210_NCOOL];
 A210TermStatus heating_status[A210_NHEAT],cooling_status[A210_NCOOL];
 double A_line,E_line,Q_line_rad,C_line_ce;
 double photoionization_rate;
 int m_line,radiative_line_included,collisional_or_escape_included;
 int line_owner_overlap;
 double line_owner_closure,normalized_line_owner_closure;
 double sum_heating,sum_cooling,residual,e_balance;
 RadeqStatus status;
} A210TermLedger;

typedef struct {
 uint64_t solve_epoch,required_te_generation,committed_te_generation;
 uint64_t radfield_generation,bf_rate_generation,line_view_generation;
 uint64_t population_generation,opacity_generation,emissivity_generation;
 char atomic_model_sha256[65],geometry_sha256[65],te_manifest_sha256[65];
 char te_context_sha256[65],term_manifest_sha256[65];
 size_t n_shells;double*T_e,*n_e;RadeqStatus*shell_status,*residual_status;
 A210TermLedger*ledger;
} ElectronTemperaturePublication;

typedef struct {
 uint64_t solve_epoch,te_generation_required,te_generation_committed;
 uint64_t shells_attempted,shells_converged,shells_published,trials;
 uint64_t population_trials,opacity_trials,emissivity_trials;
 uint64_t photo_heat_terms,line_heat_terms,ff_heat_terms,compton_heat_terms;
 uint64_t gamma_heat_terms,nonthermal_heat_terms,recomb_cool_terms,line_cool_terms;
 uint64_t collisional_cool_terms,ff_cool_terms,compton_cool_terms,adiabatic_cool_terms;
 uint64_t exact_zero_balance,no_bracket,no_root,nonconverged,charge_nonconverged;
 uint64_t blocked_stale,blocked_missing_term,blocked_gamma_unpublished;
 uint64_t blocked_schema,blocked_sign;
 uint64_t te_manifest_mismatch,te_context_mismatch,fixed_te_attempts,seed_generation_attempts;
 uint64_t line_radiative_owner_shells,line_collisional_escape_owner_shells;
 uint64_t line_replaced_collisional_terms,line_replaced_radiative_terms;
 uint64_t line_owner_overlap_shells,line_owner_closure_failures;
 uint64_t pin_attempts,floor_attempts,neighbor_attempts,old_te_attempts;
 uint64_t fallback_attempts,partial_publish_attempts,nonfinite_failures;
 double max_line_owner_closure,max_heat_residual;
} A210Counters;

typedef RadeqStatus (*A210ResidualFunction)(size_t shell,double trial_te,
                                            A210TermLedger*ledger,void*context);
int a210_publication_init(ElectronTemperaturePublication*p,size_t n_shells);
void a210_publication_free(ElectronTemperaturePublication*p);
RadeqStatus a210_line_owner_finalize(A210TermLedger*l);
int a210_solve_transaction(ElectronTemperaturePublication*published,
 const double*lower,const double*upper,const double*ne,size_t n_shells,
 uint64_t solve_epoch,uint64_t required_generation,const char*geometry_sha256,
 A210ResidualFunction residual,void*context,double*public_te,double*public_ne);
RadeqStatus a210_te_context_sha256(const char te_manifest[65],
 const char geometry[65],uint64_t solve_epoch,char out[65]);
RadeqStatus a210_geometry_sha256(const double *shell_boundaries,
                                 size_t n_boundaries,char out[65]);
A210Counters*a210_counters(void);void a210_counters_reset(void);
void a210_counters_print(FILE*stream);const char*a210_status_name(RadeqStatus s);

#endif
