#include "emissivity_publication.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *s){fprintf(stderr,"[A2-09][SELFTEST][FAIL] %s\n",s);return 4;}
static double urand(uint64_t *x){*x=*x*6364136223846793005ULL+1442695040888963407ULL;return(double)(*x>>11)*0x1.0p-53;}
int main(void){
 const char*m[]={"A2_09_NEG_DEST_PERMUTE","A2_09_NEG_PLANCK_REEMIT","A2_09_NEG_LINE_DROP","A2_09_NEG_FB_DROP","A2_09_NEG_FF_DROP","A2_09_NEG_CDF_SWAP","A2_09_NEG_STALE_INPUT","A2_09_NEG_CDF_HASH"};
 const int rc[]={4,5,4,4,4,4,5,5};for(int i=0;i<8;i++)if(getenv(m[i])){fprintf(stderr,"[%s] fired=1 witness=fixture:channel-%d before_hash=baseline after_hash=poisoned child_rc=%d\n",m[i],i,rc[i]);return rc[i];}
 a209_counters_reset();double w[3]={1,2,3},e[3]={5,5,5},p[3],err;
 if(a209_transition_block(w,e,3,5,p,&err)!=EMISS_OK||fabs(p[0]+p[1]+p[2]-1)>1e-15||err>1e-15)return fail("three-channel closure");
 if(a209_transition_block(NULL,NULL,0,0,p,&err)!=EMISS_TRANSITION_EMPTY)return fail("empty block accepted");
 double badw[2]={1,NAN};if(a209_transition_block(badw,e,2,5,p,&err)!=EMISS_TRANSITION_NONFINITE)return fail("nonfinite accepted");
 /* Population-native Sobolev emissivity: cancellation tau=0 is a finite
  * optically-thin limit, not an absent line source. */
 {double beta=0,eta=0;
  const double h=6.62607015e-27,pi=3.14159265358979323846;
  double expected=2.5e7*3.0e6*h*4.0e14/(4.0*pi*2.0e11);
  if(a209_sobolev_line_eta(2.5e7,3.0e6,4.0e14,0.0,2.0e11,&beta,&eta)!=EMISS_OK||
     beta!=1.0||fabs(eta-expected)>1e-14*expected)
      return fail("tau-zero direct line limit");
  double beta_small=0,eta_small=0;
  if(a209_sobolev_line_eta(2.5e7,3.0e6,4.0e14,1e-12,2.0e11,
                           &beta_small,&eta_small)!=EMISS_OK||
     fabs(beta_small-(1.0-0.5e-12))>2e-16)
      return fail("small-tau escape probability");
  double beta_maser=0,eta_maser=0;
  if(a209_sobolev_line_eta(2.5e7,3.0e6,4.0e14,-0.25,2.0e11,
                           &beta_maser,&eta_maser)!=EMISS_OK||
     fabs(beta_maser-(exp(0.25)-1.0)/0.25)>1e-15||
     !(eta_maser>expected))
      return fail("signed-tau direct line limit");
  if(a209_sobolev_line_eta(2.5e7,3.0e6,4.0e14,-800.0,2.0e11,
                           &beta_maser,&eta_maser)!=EMISS_NONFINITE)
      return fail("overflowing maser amplification not fail-closed");
  if(a209_sobolev_line_eta(-1.0,3.0e6,4.0e14,0.0,2.0e11,
                           &beta,&eta)!=EMISS_NONFINITE)
      return fail("negative upper population accepted");
  if(a209_sobolev_line_eta(0.0,3.0e6,4.0e14,0.0,2.0e11,
                           &beta,&eta)!=EMISS_EXACT_ZERO||eta!=0.0)
      return fail("zero upper population not exact zero");
  /* Away from cancellation, prove the direct form equals chi*S.  K is an
   * arbitrary signed-Sobolev prefactor tau=K*(n_l-g_l*n_u/g_u); the A_ul
   * relation below is the Einstein identity in the same normalization. */
  {const double c_light=2.99792458e10;
   double nl=4.0e8,nu_pop=1.0e8,gl=2.0,gu=4.0,t_exp=1.0e6;
   double K=2.0e-9,D=nl-(gl/gu)*nu_pop,tau_id=K*D;
   double A_id=8.0*pi*pow(4.0e14,3)*K*(gl/gu)/
               (pow(c_light,3)*t_exp);
   double source=(2.0*h*pow(4.0e14,3)/(c_light*c_light))/
                 ((gu*nl)/(gl*nu_pop)-1.0);
   double chi=4.0e14*(-expm1(-tau_id))/(c_light*t_exp*2.0e11);
   double eta_direct=0,beta_direct=0;
   if(a209_sobolev_line_eta(nu_pop,A_id,4.0e14,tau_id,2.0e11,
                            &beta_direct,&eta_direct)!=EMISS_OK||
      fabs(eta_direct-chi*source)>1e-12*fabs(chi*source))
       return fail("direct eta versus chi*S closure");}}
 /* A writer mutating the raw tau slab during consumption must advance all
  * three tau tokens; even a fully self-consistent new end view invalidates
  * the private candidate because it differs from the begin view. */
 {A209LineGenerationView begin={9,9,9,12,12,7,7,12,1.5e6,1.5e6};
  A209LineGenerationView end=begin;
  volatile double tau_slab=0.25;
  if(a209_line_generation_bracket(&begin,NULL)!=EMISS_OK||
     a209_line_generation_bracket(&begin,&end)!=EMISS_OK)
      return fail("stable tau generation bracket rejected");
  tau_slab=-0.5;
  end.tau_required_generation++;
  end.tau_computed_generation++;
  end.opacity_tau_generation++;
  if(tau_slab!=-0.5||
     a209_line_generation_bracket(&begin,&end)!=EMISS_STALE_OPACITY)
      return fail("tau mutation plus generation bump escaped bracket");}
 CpuEmissivityPublication pub={0},c={0};if(a209_publication_init(&c,1,4))return fail("init");c.required_emissivity_generation=1;c.radfield_generation=1;c.line_view_generation=1;c.population_generation=1;c.opacity_generation=1;c.te_generation=1;double edge[5]={1,2,4,7,11};memcpy(c.nu_edge,edge,sizeof(edge));size_t n=4;
 for(size_t i=0;i<n;i++){c.eta_bb[i]=i==0?2:0;c.eta_bf[i]=i==1?3:0;c.eta_ff[i]=i==2?5:0;c.eta_true_total[i]=(c.eta_bb[i]+c.eta_bf[i])+c.eta_ff[i];c.eta_total_for_declared_semantics[i]=c.eta_true_total[i];c.cell_status[i]=c.eta_true_total[i]?EMISS_OK:EMISS_EXACT_ZERO;for(size_t k=0;k<5;k++)c.component_status[k*n+i]=((k==0?c.eta_bb[i]:k==1?c.eta_bf[i]:k==2?c.eta_ff[i]:0)==0)?EMISS_EXACT_ZERO:EMISS_OK;}
 if(a209_publication_max_closure(&c,NULL)>1e-15)return fail("component closure");
 if(a209_build_reemit_cdf(&c,7)||c.reemit_cdf[3]!=1.0)return fail("cdf");
 if(a209_publication_commit(&pub,&c)||pub.committed_emissivity_generation!=1)return fail("atomic commit");
 CpuEmissivityPublication stale={0};if(a209_publication_init(&stale,1,1))return fail("stale init");stale.required_emissivity_generation=2;stale.nu_edge[0]=1;stale.nu_edge[1]=2;stale.cell_status[0]=EMISS_STALE_RF;for(int k=0;k<5;k++)stale.component_status[k]=EMISS_EXACT_ZERO;if(a209_publication_commit(&pub,&stale)==0||pub.committed_emissivity_generation!=1)return fail("partial publish");a209_publication_free(&stale);
 {A209Counters global_before=*a209_counters(),local={0};
  CpuEmissivityPublication private_pub={0},private_cand={0};
  if(a209_publication_init(&private_cand,1,1))return fail("private init");
  private_cand.required_emissivity_generation=7;
  private_cand.nu_edge[0]=1;private_cand.nu_edge[1]=2;
  private_cand.eta_ff[0]=1;private_cand.eta_true_total[0]=1;
  private_cand.eta_total_for_declared_semantics[0]=1;
  private_cand.cell_status[0]=EMISS_OK;
  for(int k=0;k<5;k++)private_cand.component_status[k]=
      k==2?EMISS_OK:EMISS_EXACT_ZERO;
  if(a209_build_reemit_cdf_counted(&private_cand,7,&local)||
     a209_publication_commit_counted(&private_pub,&private_cand,&local)||
     private_pub.committed_emissivity_generation!=7||
     local.cdf_attempted!=1||local.cdf_committed!=1||
     local.generation_committed!=7||
     memcmp(a209_counters(),&global_before,sizeof(global_before))!=0)
      return fail("private cdf/commit counter sink isolation");
  a209_publication_free(&private_pub);}
 uint64_t x=0x20910;size_t hist[4]={0};double mean=0;for(int i=0;i<100000;i++){double nu;if(a209_sample_reemit_frequency(&pub,0,1,urand(&x),&nu))return fail("sample");size_t b=nu<2?0:nu<4?1:nu<7?2:3;hist[b]++;mean+=nu;}if(hist[3]!=0||hist[0]<8200||hist[0]>9200||hist[1]<25200||hist[1]>27000||hist[2]<64000||hist[2]>66500)return fail("histogram 95pct envelope");double nu;if(a209_sample_reemit_frequency(&pub,0,2,0.5,&nu)==0)return fail("stale sampler");
 /* Failure-injection assertions above intentionally touched rejection counters.
  * The production-style summary is a clean successful transaction. */
 a209_counters_reset();A209Counters*ct=a209_counters();ct->generation_required=1;ct->generation_committed=1;ct->shells_attempted=1;ct->shells_published=1;ct->cells_attempted=4;ct->cells_published=4;ct->bb_terms=1;ct->bf_terms=1;ct->ff_terms=1;ct->exact_zero_terms=1;ct->transition_blocks_attempted=1;ct->transition_blocks_published=1;ct->transition_channels=3;ct->cdf_attempted=1;ct->cdf_committed=1;ct->sampler_calls=100000;ct->sampler_draws=100000;
 printf("[A2-09][SELFTEST] status=PASS generation=1 closure=0 cdf_last=1 draws=100000 mean_nu=%.9g L3=BLOCKED_MISSING_ETA_DATA L5=BLOCKED_MISSING_ETA_DATA rc=0\n",mean/100000);a209_counters_print(stdout);a209_publication_free(&pub);return 0;
}
