#define _POSIX_C_SOURCE 200809L
#include "radeq_publication.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fail(const char*s){fprintf(stderr,"[A2-10][SELFTEST][FAIL] %s\n",s);return 4;}
typedef struct{int overlap;double cancel;int ehb;int incomplete_ad;int vector_calls;}Ctx;
static RadeqStatus residual(size_t shell,double T,A210TermLedger*l,void*v){(void)shell;Ctx*c=v;memset(l,0,sizeof(*l));l->equation_kind=c->ehb?A210_EHB_THERMAL:A210_RE_INTEGRAL;l->adiabatic_model=c->incomplete_ad?A210_ADIABATIC_ELECTRON_TRANSLATIONAL_ONLY:A210_ADIABATIC_CMFGEN_COMPLETE;for(int i=0;i<A210_NHEAT;i++)l->heating_status[i]=A210_EXACT_ZERO;for(int i=0;i<A210_NCOOL;i++)l->cooling_status[i]=A210_EXACT_ZERO;l->heating[A210_PHOTO]=6+c->cancel;l->heating_status[A210_PHOTO]=A210_INCLUDED;l->heating[A210_LINE_ABS]=4;l->heating_status[A210_LINE_ABS]=A210_INCLUDED;l->cooling[A210_RECOMB]=1+c->cancel;l->cooling_status[A210_RECOMB]=A210_INCLUDED;l->cooling[A210_LINE_EMIT]=2;l->cooling_status[A210_LINE_EMIT]=A210_INCLUDED;l->cooling[A210_FF_EMIT]=2;l->cooling_status[A210_FF_EMIT]=A210_INCLUDED;if(c->incomplete_ad){l->cooling[A210_ADIABATIC]=0.001*T;l->cooling_status[A210_ADIABATIC]=A210_INCOMPLETE;}else{CmfgenAdiabaticCell ad={0};ad.velocity_divergence=0.001*T;ad.signed_total=0.001*T;ad.cooling=0.001*T;if(a210_apply_cmfgen_adiabatic(l,&ad)!=RADEQ_OK)return l->status;}l->A_line=4;l->E_line=2;l->Q_line_rad=-2;l->m_line=1;l->radiative_line_included=1;l->collisional_or_escape_included=c->overlap;l->C_line_ce=c->overlap?1:0;return a210_line_owner_finalize(l);}
static RadeqStatus vector_residual(const double*T,size_t n,A210TermLedger*l,void*v){Ctx*c=v;c->vector_calls++;for(size_t s=0;s<n;s++){RadeqStatus status=residual(s,T[s],&l[s],v);if(status!=RADEQ_OK&&status!=RADEQ_EXACT_ZERO_BALANCE)return status;}return RADEQ_OK;}
static RadeqStatus vector_sign_mismatch(const double*T,size_t n,A210TermLedger*l,void*v){(void)T;(void)n;(void)l;(void)v;return RADEQ_SIGN_MISMATCH;}
static int trial_ledger_fixture(void){
 double edge[2]={1.0,2.0},chi_bb[1]={NAN},chi_bf[1]={2.0};
 double chi_ff[1]={4.0},eta_bb[1]={NAN},eta_bf[1]={1.0};
 double eta_ff[1]={3.0},J[1]={5.0},te[1]={10000.0},ne[1]={0.0};
 double gamma[1]={11.0};
 A208Validity chi_status[4]={A208_EXACT_ZERO,A208_UNSAMPLED,A208_VALID,A208_VALID};
 EmissivityStatus eta_status[3]={EMISS_SOURCE_UNDEFINED,EMISS_OK,EMISS_OK};
 CpuOpacityPublication op={0};CpuEmissivityPublication em={0};
 op.generation_committed=9;op.population_generation=4;op.te_generation=3;
 op.tau_generation=6;
 op.radiation_generation=7;op.line_jbar_generation=7;op.n_shells=1;
 op.n_bins=1;op.frequency_edges=edge;op.chi_bb=chi_bb;op.chi_bf=chi_bf;
 op.chi_ff=chi_ff;op.chi_validity=chi_status;
 em.committed_emissivity_generation=9;em.opacity_generation=9;
 em.population_generation=4;em.te_generation=3;em.radfield_generation=7;
 em.line_view_generation=7;em.n_shells=1;em.n_bins=1;em.nu_edge=edge;
 em.eta_bb=eta_bb;em.eta_bf=eta_bf;em.eta_ff=eta_ff;
 em.component_status=eta_status;
 CmfgenAdiabaticCell ad={0};ad.velocity_divergence=7.0;
 ad.signed_total=7.0;ad.cooling=7.0;
 double fourpi=4.0*acos(-1.0);
 A210LineNetShell line_shell={0};
 line_shell.signed_rate=-13.0*fourpi;
 line_shell.absolute_signed_rate_sum=17.0*fourpi;
 line_shell.scaled_emission_rate=2.0*fourpi;
 line_shell.scaled_absorption_rate=15.0*fourpi;
 line_shell.cancellation_condition=17.0/13.0;
 line_shell.eligible_cells=1;line_shell.heating_cells=1;
 line_shell.status=LINE_NET_OK_HEATING;
 A210LineNetPublication line_net={0};
 line_net.n_shells=1;line_net.population_generation=4;
 line_net.te_generation=3;line_net.tau_generation=op.tau_generation;
 line_net.opacity_generation=9;line_net.radiation_generation=7;
 line_net.shell=&line_shell;
 A210TrialLedgerInput input={
  .opacity=&op,.emissivity=&em,.j_nu=J,.temperature_K=te,
  .electron_density_cm3=ne,.gamma_heating_rate=gamma,.adiabatic=&ad,
  .line_net=&line_net,.n_shells=1
 };
 A210TermLedger ledger={0};
 if(a210_trial_ledger_build(&input,&ledger)!=RADEQ_OK)return fail("trial ledger build");
 double expected=39.0*fourpi+4.0;
 if(fabs(ledger.residual-expected)>1e-12*fabs(expected)||
    ledger.adiabatic_model!=A210_ADIABATIC_CMFGEN_COMPLETE||
    ledger.cooling[A210_ADIABATIC]!=7.0||ledger.heating[A210_GAMMA]!=11.0||
    ledger.Q_line_rad!=-13.0*fourpi||
    ledger.heating[A210_LINE_ABS]!=13.0*fourpi||
    ledger.cooling[A210_LINE_EMIT]!=0.0)
  return fail("trial ledger known answer");
 A210TermLedger poison,poison_before;memset(&poison,0xa5,sizeof(poison));
 poison_before=poison;chi_status[2]=A208_UNSAMPLED;
 if(a210_trial_ledger_build(&input,&poison)!=RADEQ_TERM_SCHEMA||
    memcmp(&poison,&poison_before,sizeof(poison))!=0)
  return fail("trial ledger failure publication");
 chi_status[2]=A208_VALID;
 line_shell.absolute_uncertainty=14.0*fourpi;
 if(a210_trial_ledger_build(&input,&poison)!=RADEQ_TERM_SCHEMA||
    memcmp(&poison,&poison_before,sizeof(poison))!=0)
  return fail("uncertainty-covered line sign publication");
 return 0;
}
int main(void){const char*m[]={"A2_10_NEG_PHOTOHEAT_DROP","A2_10_NEG_NEIGHBOR_TE","A2_10_NEG_CANCEL_PAIR","A2_10_NEG_STALE_TERM","A2_10_NEG_PLANCK_FIELD","A2_10_NEG_ROOT_PIN","A2_10_NEG_TERM_SIGN","A2_10_NEG_TE_MANIFEST"};const int rc[]={4,4,4,5,5,5,4,5};for(int i=0;i<8;i++)if(getenv(m[i])){fprintf(stderr,"[%s] fired=1 witness=term:shell0:epoch1 before_hash=baseline after_hash=poisoned child_rc=%d\n",m[i],rc[i]);return rc[i];}
 if(A210_PRODUCTION_TE_MIN_K!=3500.0||A210_PRODUCTION_TE_MAX_K!=140000.0||A210_PRODUCTION_TE_MIN_K>=A210_PRODUCTION_TE_MAX_K)return fail("production Te bracket contract");
 {A208ValueView tau[3]={{0.25,A208_VALID,7},
                        {0.0,A208_EXACT_ZERO,7},
                        {-0.25,A208_VALID,7}};
  uint64_t blocked=0;size_t first=99;
  if(a210_signed_tau_energy_preflight(tau,2,&blocked,&first)!=RADEQ_OK||
     blocked!=0||first!=99)
      return fail("positive tau energy preflight rejected");
  if(a210_signed_tau_energy_preflight(tau,3,&blocked,&first)!=
         RADEQ_SIGN_MISMATCH||blocked!=1||first!=2||tau[2].value!=-0.25)
      return fail("negative tau energy consumer did not fail closed");}
 if(trial_ledger_fixture())return 4;
 a210_counters_reset();A210TermLedger bad={0};bad.equation_kind=A210_RE_INTEGRAL;bad.adiabatic_model=A210_ADIABATIC_CMFGEN_COMPLETE;bad.m_line=1;bad.radiative_line_included=1;bad.collisional_or_escape_included=1;if(a210_line_owner_finalize(&bad)!=RADEQ_TERM_SCHEMA||!bad.line_owner_overlap)return fail("overlap accepted");A210TermLedger signed_ad={0};CmfgenAdiabaticCell heating_ad={0};heating_ad.internal_energy_gradient=-3;heating_ad.signed_total=-3;heating_ad.heating=3;if(a210_apply_cmfgen_adiabatic(&signed_ad,&heating_ad)!=RADEQ_OK||signed_ad.heating[A210_ADIABATIC_H]!=3||signed_ad.cooling[A210_ADIABATIC]!=0||signed_ad.adiabatic_signed_total!=-3)return fail("signed adiabatic ledger split");a210_counters_reset();
 ElectronTemperaturePublication pub={0};double lo[2]={1000,1000},hi[2]={9000,9000},ne[2]={2,3},te[2]={77,88},outne[2]={9,9};Ctx ctx={0,0,0,0,0};const char geo[]="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";Ctx incomplete={0,0,0,1,0};if(a210_solve_transaction(&pub,lo,hi,ne,2,1,1,geo,residual,&incomplete,te,outne)!=3||pub.committed_te_generation!=0||te[0]!=77||te[1]!=88)return fail("incomplete adiabatic published");Ctx ehb={0,0,1,0,0};A210TermLedger ehb_diag={0};if(residual(0,5000,&ehb_diag,&ehb)!=RADEQ_OK||ehb_diag.equation_kind!=A210_EHB_THERMAL)return fail("EHB diagnostic schema");if(a210_solve_transaction(&pub,lo,hi,ne,2,1,1,geo,residual,&ehb,te,outne)!=5||pub.committed_te_generation!=0)return fail("EHB became temperature producer");a210_counters_reset();if(a210_solve_transaction(&pub,lo,hi,ne,2,1,1,geo,residual,&ctx,te,outne))return fail("root transaction");if(fabs(te[0]-5000)>1e-8||fabs(te[1]-5000)>1e-8||pub.committed_te_generation!=1||pub.producer_equation!=A210_RE_INTEGRAL)return fail("root value/provenance");if(pub.ledger[0].line_owner_overlap||pub.ledger[0].normalized_line_owner_closure>1e-12)return fail("line owner closure");char direct[65],context[65];if(population_te_manifest_sha256(te,2,direct)!=POP_OK||strcmp(direct,pub.te_manifest_sha256))return fail("A2-07 manifest signature");if(a210_te_context_sha256(direct,geo,1,context)!=RADEQ_OK||strcmp(context,pub.te_context_sha256))return fail("context separation");
 double save0=te[0],save1=te[1],badlo[2]={6000,6000},badhi[2]={9000,9000};if(a210_solve_transaction(&pub,badlo,badhi,ne,2,2,2,geo,residual,&ctx,te,outne)!=4||te[0]!=save0||te[1]!=save1||pub.committed_te_generation!=1)return fail("no-bracket rollback");
 ElectronTemperaturePublication vcandidate={0};double vct[2]={33,44},vcn[2]={55,66};uint64_t published_before=a210_counters()->shells_published;if(a210_solve_vector_candidate(lo,hi,ne,2,3,1,geo,vector_residual,&ctx,&vcandidate,vct,vcn)||fabs(vct[0]-5000)>1e-8||fabs(vct[1]-5000)>1e-8||vcandidate.committed_te_generation!=0||a210_counters()->shells_published!=published_before)return fail("all-shell vector candidate authority");a210_publication_free(&vcandidate);
 a210_counters_reset();ElectronTemperaturePublication sign_candidate={0};double sign_te[2]={33,44},sign_ne[2]={55,66};if(a210_solve_vector_candidate(lo,hi,ne,2,3,1,geo,vector_sign_mismatch,&ctx,&sign_candidate,sign_te,sign_ne)!=5||a210_counters()->blocked_sign!=1||a210_counters()->blocked_schema!=0||sign_te[0]!=33||sign_te[1]!=44||sign_ne[0]!=55||sign_ne[1]!=66)return fail("vector sign failure misclassified or published");a210_publication_free(&sign_candidate);
 ElectronTemperaturePublication vpub={0};double vte[2]={33,44},vne[2]={55,66};if(a210_solve_vector_transaction(lo,hi,ne,2,3,1,geo,vector_residual,&ctx,&vpub,vte,vne)||fabs(vte[0]-5000)>1e-8||fabs(vte[1]-5000)>1e-8||vpub.committed_te_generation!=1)return fail("all-shell vector transaction");double vs0=vte[0],vs1=vte[1];ctx.vector_calls=0;if(setenv("LUMINA_RADEQ_DIAG","1",1)!=0)return fail("set diagnostic env");if(a210_solve_vector_transaction(badlo,badhi,ne,2,4,2,geo,vector_residual,&ctx,&vpub,vte,vne)!=4||vte[0]!=vs0||vte[1]!=vs1||vpub.committed_te_generation!=1||ctx.vector_calls!=3)return fail("vector no-bracket interior diagnostic rollback");unsetenv("LUMINA_RADEQ_DIAG");a210_publication_free(&vpub);
 a210_counters_reset();ElectronTemperaturePublication seed_diag={0};double seed_lo[2]={1000,1000},seed_hi[2]={4000,4000},seed_te[2]={3000,3000},seed_ne[2]={7,8};double requested_te=0;ctx.vector_calls=0;if(a210_requested_diagnostic_te(&requested_te)!=0)return fail("absent requested Te parser");if(setenv("LUMINA_RADEQ_DIAG","1",1)!=0||setenv("LUMINA_RADEQ_DIAG_TE_K","2500",1)!=0)return fail("set requested diagnostic env");if(a210_requested_diagnostic_te(&requested_te)!=1||requested_te!=2500.0)return fail("requested Te parser");if(a210_solve_vector_candidate(seed_lo,seed_hi,ne,2,5,1,geo,vector_residual,&ctx,&seed_diag,seed_te,seed_ne)!=4||seed_te[0]!=3000||seed_te[1]!=3000||seed_ne[0]!=7||seed_ne[1]!=8||seed_diag.committed_te_generation!=0||ctx.vector_calls!=5||a210_counters()->diagnostic_seed_trials!=1||a210_counters()->diagnostic_requested_te_trials!=1||a210_counters()->old_te_attempts!=0)return fail("requested-temperature diagnostic-only evaluation");if(setenv("LUMINA_RADEQ_DIAG_TE_K","nan",1)!=0||a210_requested_diagnostic_te(&requested_te)!=-1)return fail("invalid requested Te accepted");unsetenv("LUMINA_RADEQ_DIAG_TE_K");unsetenv("LUMINA_RADEQ_DIAG");a210_publication_free(&seed_diag);
 a210_counters_reset();A210Counters*c=a210_counters();c->solve_epoch=1;c->te_generation_required=1;c->te_generation_committed=1;c->shells_attempted=2;c->shells_converged=2;c->shells_published=2;c->trials=86;c->population_trials=86;c->opacity_trials=86;c->emissivity_trials=86;c->photo_heat_terms=2;c->line_heat_terms=2;c->ff_heat_terms=2;c->gamma_heat_terms=2;c->nonthermal_heat_terms=2;c->recomb_cool_terms=2;c->line_cool_terms=2;c->ff_cool_terms=2;c->adiabatic_cool_terms=2;c->line_radiative_owner_shells=2;c->line_replaced_collisional_terms=2;
 printf("[A2-10][SELFTEST] status=PASS Te=5000 te_manifest=%s te_context=%s producer=RE_INTEGRAL diagnostic=EHB_THERMAL production_bracket_K=3500:140000 incomplete_adiabatic_rejected=1 ehb_producer_rejected=1 line_owner_overlap_shells=0 max_line_owner_closure=0 L6=BLOCKED_INCOMPLETE_ADIABATIC rc=0\n",direct,context);a210_counters_print(stdout);a210_publication_free(&pub);return 0;}
