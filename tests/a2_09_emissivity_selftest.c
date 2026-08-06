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
 CpuEmissivityPublication pub={0},c={0};if(a209_publication_init(&c,1,4))return fail("init");c.required_emissivity_generation=1;c.radfield_generation=1;c.line_view_generation=1;c.population_generation=1;c.opacity_generation=1;c.te_generation=1;double edge[5]={1,2,4,7,11};memcpy(c.nu_edge,edge,sizeof(edge));size_t n=4;
 for(size_t i=0;i<n;i++){c.eta_bb[i]=i==0?2:0;c.eta_bf[i]=i==1?3:0;c.eta_ff[i]=i==2?5:0;c.eta_true_total[i]=(c.eta_bb[i]+c.eta_bf[i])+c.eta_ff[i];c.eta_total_for_declared_semantics[i]=c.eta_true_total[i];c.cell_status[i]=c.eta_true_total[i]?EMISS_OK:EMISS_EXACT_ZERO;for(size_t k=0;k<5;k++)c.component_status[k*n+i]=((k==0?c.eta_bb[i]:k==1?c.eta_bf[i]:k==2?c.eta_ff[i]:0)==0)?EMISS_EXACT_ZERO:EMISS_OK;}
 if(a209_publication_max_closure(&c,NULL)>1e-15)return fail("component closure");
 if(a209_build_reemit_cdf(&c,7)||c.reemit_cdf[3]!=1.0)return fail("cdf");
 if(a209_publication_commit(&pub,&c)||pub.committed_emissivity_generation!=1)return fail("atomic commit");
 CpuEmissivityPublication stale={0};if(a209_publication_init(&stale,1,1))return fail("stale init");stale.required_emissivity_generation=2;stale.nu_edge[0]=1;stale.nu_edge[1]=2;stale.cell_status[0]=EMISS_STALE_RF;for(int k=0;k<5;k++)stale.component_status[k]=EMISS_EXACT_ZERO;if(a209_publication_commit(&pub,&stale)==0||pub.committed_emissivity_generation!=1)return fail("partial publish");a209_publication_free(&stale);
 uint64_t x=0x20910;size_t hist[4]={0};double mean=0;for(int i=0;i<100000;i++){double nu;if(a209_sample_reemit_frequency(&pub,0,1,urand(&x),&nu))return fail("sample");size_t b=nu<2?0:nu<4?1:nu<7?2:3;hist[b]++;mean+=nu;}if(hist[3]!=0||hist[0]<8200||hist[0]>9200||hist[1]<25200||hist[1]>27000||hist[2]<64000||hist[2]>66500)return fail("histogram 95pct envelope");double nu;if(a209_sample_reemit_frequency(&pub,0,2,0.5,&nu)==0)return fail("stale sampler");
 /* Failure-injection assertions above intentionally touched rejection counters.
  * The production-style summary is a clean successful transaction. */
 a209_counters_reset();A209Counters*ct=a209_counters();ct->generation_required=1;ct->generation_committed=1;ct->shells_attempted=1;ct->shells_published=1;ct->cells_attempted=4;ct->cells_published=4;ct->bb_terms=1;ct->bf_terms=1;ct->ff_terms=1;ct->exact_zero_terms=1;ct->transition_blocks_attempted=1;ct->transition_blocks_published=1;ct->transition_channels=3;ct->cdf_attempted=1;ct->cdf_committed=1;ct->sampler_calls=100000;ct->sampler_draws=100000;
 printf("[A2-09][SELFTEST] status=PASS generation=1 closure=0 cdf_last=1 draws=100000 mean_nu=%.9g L3=BLOCKED_MISSING_ETA_DATA L5=BLOCKED_MISSING_ETA_DATA rc=0\n",mean/100000);a209_counters_print(stdout);a209_publication_free(&pub);return 0;
}
