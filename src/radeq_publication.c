#include "radeq_publication.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define A210_H_PLANCK 6.62607015e-27
#define A210_K_BOLTZMANN 1.380649e-16
#define A210_C_LIGHT 2.99792458e10
#define A210_M_ELECTRON 9.1093837015e-28
#define A210_SIGMA_THOMSON 6.6524587321e-25
#define A210_FOUR_PI 12.56637061435917295385

typedef struct{uint32_t h[8];uint64_t bits;unsigned char b[64];size_t used;}Sh;
static A210Counters g;
static uint32_t ro(uint32_t x,unsigned n){return(x>>n)|(x<<(32-n));}
static void sb(Sh*s,const unsigned char*b){static const uint32_t k[64]={0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};uint32_t w[64];for(int i=0;i<16;i++)w[i]=((uint32_t)b[4*i]<<24)|((uint32_t)b[4*i+1]<<16)|((uint32_t)b[4*i+2]<<8)|b[4*i+3];for(int i=16;i<64;i++){uint32_t x=w[i-15],y=w[i-2];w[i]=w[i-16]+(ro(x,7)^ro(x,18)^(x>>3))+w[i-7]+(ro(y,17)^ro(y,19)^(y>>10));}uint32_t a=s->h[0],bb=s->h[1],c=s->h[2],d=s->h[3],e=s->h[4],f=s->h[5],gg=s->h[6],h=s->h[7];for(int i=0;i<64;i++){uint32_t t1=h+(ro(e,6)^ro(e,11)^ro(e,25))+((e&f)^(~e&gg))+k[i]+w[i],t2=(ro(a,2)^ro(a,13)^ro(a,22))+((a&bb)^(a&c)^(bb&c));h=gg;gg=f;f=e;e=d+t1;d=c;c=bb;bb=a;a=t1+t2;}s->h[0]+=a;s->h[1]+=bb;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;s->h[5]+=f;s->h[6]+=gg;s->h[7]+=h;}
static void si(Sh*s){uint32_t h[8]={0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};memcpy(s->h,h,sizeof(h));s->bits=0;s->used=0;}static void su(Sh*s,const void*v,size_t n){const unsigned char*p=v;s->bits+=(uint64_t)n*8;while(n){size_t r=64-s->used,t=n<r?n:r;memcpy(s->b+s->used,p,t);s->used+=t;p+=t;n-=t;if(s->used==64){sb(s,s->b);s->used=0;}}}static void u64(Sh*s,uint64_t x){unsigned char b[8];for(int i=0;i<8;i++)b[7-i]=x>>(8*i);su(s,b,8);}static void sd(Sh*s,char out[65]){uint64_t bits=s->bits;unsigned char x=0x80,z=0,l[8],d[32];su(s,&x,1);while(s->used!=56)su(s,&z,1);for(int i=0;i<8;i++)l[7-i]=bits>>(8*i);su(s,l,8);for(int i=0;i<8;i++){d[4*i]=s->h[i]>>24;d[4*i+1]=s->h[i]>>16;d[4*i+2]=s->h[i]>>8;d[4*i+3]=s->h[i];}static const char h[]="0123456789abcdef";for(int i=0;i<32;i++){out[2*i]=h[d[i]>>4];out[2*i+1]=h[d[i]&15];}out[64]=0;}
static double sum(const double*x,size_t n){double s=0,c=0;for(size_t i=0;i<n;i++){double y=x[i]-c,t=s+y;c=(t-s)-y;s=t;}return s;}
int a210_requested_diagnostic_te(double *temperature_K){
 const char*text=getenv("LUMINA_RADEQ_DIAG_TE_K");
 if(temperature_K)*temperature_K=0.0;
 if(!text)return 0;
 errno=0;char*end=NULL;double value=strtod(text,&end);
 if(errno!=0||end==text||!end||*end!='\0'||!isfinite(value)||value<=0.0)
  return-1;
 if(temperature_K)*temperature_K=value;
 return 1;
}
int a210_publication_init(ElectronTemperaturePublication*p,size_t n){if(!p||!n)return-1;memset(p,0,sizeof(*p));p->n_shells=n;p->T_e=calloc(n,sizeof(double));p->n_e=calloc(n,sizeof(double));p->shell_status=calloc(n,sizeof(*p->shell_status));p->residual_status=calloc(n,sizeof(*p->residual_status));p->ledger=calloc(n,sizeof(*p->ledger));if(!p->T_e||!p->n_e||!p->shell_status||!p->residual_status||!p->ledger){a210_publication_free(p);return-1;}return 0;}
void a210_publication_free(ElectronTemperaturePublication*p){if(!p)return;free(p->T_e);free(p->n_e);free(p->shell_status);free(p->residual_status);free(p->ledger);memset(p,0,sizeof(*p));}
RadeqStatus a210_apply_cmfgen_adiabatic(
 A210TermLedger*l,const CmfgenAdiabaticCell*c){
 if(!l||!c)return RADEQ_TERM_SCHEMA;
 double component_sum=((c->temperature_gradient+c->velocity_divergence)+
                       c->electron_fraction_gradient)+
                      c->internal_energy_gradient;
 double split=c->cooling-c->heating;
 double scale=fmax(fabs(c->temperature_gradient)+
                   fabs(c->velocity_divergence)+
                   fabs(c->electron_fraction_gradient)+
                   fabs(c->internal_energy_gradient),DBL_MIN);
 if(!isfinite(c->temperature_gradient)||
    !isfinite(c->velocity_divergence)||
    !isfinite(c->electron_fraction_gradient)||
    !isfinite(c->internal_energy_gradient)||
    !isfinite(c->signed_total)||!isfinite(c->cooling)||
    !isfinite(c->heating)||c->cooling<0.0||c->heating<0.0||
    fabs(component_sum-c->signed_total)>1e-12*scale||
    fabs(split-c->signed_total)>1e-12*scale||
    (c->cooling>0.0&&c->heating>0.0)){
  g.blocked_sign++;
  return l->status=RADEQ_SIGN_MISMATCH;
 }
 l->adiabatic_model=A210_ADIABATIC_CMFGEN_COMPLETE;
 l->adiabatic_temperature_gradient=c->temperature_gradient;
 l->adiabatic_velocity_divergence=c->velocity_divergence;
 l->adiabatic_electron_fraction_gradient=c->electron_fraction_gradient;
 l->adiabatic_internal_energy_gradient=c->internal_energy_gradient;
 l->adiabatic_signed_total=c->signed_total;
 l->cooling[A210_ADIABATIC]=c->cooling;
 l->cooling_status[A210_ADIABATIC]=c->cooling?
     A210_INCLUDED:A210_EXACT_ZERO;
 l->heating[A210_ADIABATIC_H]=c->heating;
 l->heating_status[A210_ADIABATIC_H]=c->heating?
     A210_INCLUDED:A210_EXACT_ZERO;
 return RADEQ_OK;
}

static int a210_opacity_component_valid(A208Validity validity){
 return validity==A208_VALID||validity==A208_EXACT_ZERO;
}

static int a210_emissivity_component_valid(EmissivityStatus status){
 return status==EMISS_OK||status==EMISS_EXACT_ZERO;
}

RadeqStatus a210_trial_ledger_build(
 const A210TrialLedgerInput*in,A210TermLedger*out){
 if(!in||!out||!in->opacity||!in->emissivity||!in->j_nu||
    !in->temperature_K||!in->electron_density_cm3||!in->adiabatic||
    !in->line_net||!in->n_shells)return RADEQ_TERM_SCHEMA;
 const CpuOpacityPublication*op=in->opacity;
 const CpuEmissivityPublication*em=in->emissivity;
 const A210LineNetPublication*ln=in->line_net;
 size_t ns=in->n_shells,nb=em->n_bins,cells=0;
 if(!nb||ns>SIZE_MAX/nb)return RADEQ_TERM_SCHEMA;
 cells=ns*nb;
 if(op->n_shells!=ns||em->n_shells!=ns||op->n_bins!=nb||
    !op->generation_committed||
    em->committed_emissivity_generation!=op->generation_committed||
    em->opacity_generation!=op->generation_committed||
    em->population_generation!=op->population_generation||
    em->te_generation!=op->te_generation||
    em->radfield_generation!=op->radiation_generation||
    em->line_view_generation!=op->line_jbar_generation||
    ln->n_shells!=ns||!ln->shell||
    ln->population_generation!=op->population_generation||
    ln->te_generation!=op->te_generation||
    ln->tau_generation!=op->tau_generation||
    ln->opacity_generation!=op->generation_committed||
    ln->radiation_generation!=op->radiation_generation||
    !op->frequency_edges||!op->chi_bf||!op->chi_ff||
    !op->chi_validity||!em->nu_edge||!em->eta_bf||
    !em->eta_ff||!em->component_status)return RADEQ_TERM_SCHEMA;
 for(size_t b=0;b<=nb;b++)
  if(!isfinite(op->frequency_edges[b])||
     !isfinite(em->nu_edge[b])||
     op->frequency_edges[b]!=em->nu_edge[b]||
     (b&&em->nu_edge[b]<=em->nu_edge[b-1]))return RADEQ_TERM_SCHEMA;
 A210TermLedger*candidate=calloc(ns,sizeof(*candidate));
 if(!candidate)return RADEQ_NONFINITE;
 RadeqStatus result=RADEQ_OK;
 for(size_t s=0;s<ns;s++){
  A210TermLedger*l=&candidate[s];
  l->equation_kind=A210_RE_INTEGRAL;
  for(int k=0;k<A210_NHEAT;k++)l->heating_status[k]=A210_EXACT_ZERO;
  for(int k=0;k<A210_NCOOL;k++)l->cooling_status[k]=A210_EXACT_ZERO;
  double te=in->temperature_K[s],ne=in->electron_density_cm3[s];
  if(!isfinite(te)||te<=0.0||!isfinite(ne)||ne<0.0){
   result=!isfinite(te)||!isfinite(ne)?RADEQ_NONFINITE:
          (te<=0.0?RADEQ_INVALID_TE_TRIAL:RADEQ_INVALID_NE);break;
  }
  double photo_abs=0.0,photo_rate=0.0,ff_abs=0.0;
  double recomb=0.0,ff_emit=0.0,j_int=0.0,jnu_int=0.0;
  for(size_t b=0;b<nb;b++){
   size_t i=s*nb+b;
   double lo=em->nu_edge[b],hi=em->nu_edge[b+1],dnu=hi-lo;
   double nu=sqrt(lo*hi),J=in->j_nu[i];
   if(!isfinite(J)||J<0.0||!isfinite(nu)||nu<=0.0||!isfinite(dnu)||
      dnu<=0.0||
      !a210_opacity_component_valid(op->chi_validity[2*cells+i])||
      !a210_opacity_component_valid(op->chi_validity[3*cells+i])||
      !a210_emissivity_component_valid(em->component_status[cells+i])||
      !a210_emissivity_component_valid(em->component_status[2*cells+i])||
      !isfinite(op->chi_bf[i])||
      !isfinite(op->chi_ff[i])||!isfinite(em->eta_bf[i])||
      !isfinite(em->eta_ff[i])||
      em->eta_bf[i]<0.0||em->eta_ff[i]<0.0){
    result=RADEQ_TERM_SCHEMA;break;
   }
   photo_abs+=op->chi_bf[i]*J*dnu;
   photo_rate+=op->chi_bf[i]*J*dnu/(A210_H_PLANCK*nu);
   ff_abs+=op->chi_ff[i]*J*dnu;
   recomb+=em->eta_bf[i]*dnu;
   ff_emit+=em->eta_ff[i]*dnu;
   j_int+=J*dnu;jnu_int+=J*nu*dnu;
  }
  if(result!=RADEQ_OK)break;
  photo_abs*=A210_FOUR_PI;photo_rate*=A210_FOUR_PI;
  ff_abs*=A210_FOUR_PI;
  recomb*=A210_FOUR_PI;ff_emit*=A210_FOUR_PI;
  if(photo_abs>=0.0){l->heating[A210_PHOTO]=photo_abs;}
  else{recomb-=photo_abs;}
  l->heating_status[A210_PHOTO]=photo_abs==0.0?
      A210_EXACT_ZERO:A210_INCLUDED;
  l->photoionization_rate=photo_rate;
  const A210LineNetShell*line=&ln->shell[s];
  if(!isfinite(line->signed_rate)||
     !isfinite(line->absolute_uncertainty)||line->absolute_uncertainty<0.0||
     !isfinite(line->absolute_signed_rate_sum)||
     line->absolute_signed_rate_sum<fabs(line->signed_rate)||
     !isfinite(line->scaled_emission_rate)||line->scaled_emission_rate<0.0||
     !isfinite(line->scaled_absorption_rate)||
     !isfinite(line->cancellation_condition)||
     (line->status!=LINE_NET_OK_COOLING&&
      line->status!=LINE_NET_OK_HEATING&&
      line->status!=LINE_NET_EXACT_ZERO)||
     (line->status==LINE_NET_OK_COOLING&&
      !(line->signed_rate>line->absolute_uncertainty))||
     (line->status==LINE_NET_OK_HEATING&&
      !(-line->signed_rate>line->absolute_uncertainty))||
     (line->status==LINE_NET_EXACT_ZERO&&
      (line->signed_rate!=0.0||line->absolute_uncertainty!=0.0))){
   result=RADEQ_TERM_SCHEMA;break;
  }
  l->A_line=line->scaled_absorption_rate;
  l->E_line=line->scaled_emission_rate;
  l->Q_line_rad=line->signed_rate;l->m_line=1;
  l->radiative_line_included=1;l->collisional_or_escape_included=0;
  if(line->status==LINE_NET_OK_COOLING){
   l->cooling[A210_LINE_EMIT]=line->signed_rate;
   l->cooling_status[A210_LINE_EMIT]=A210_INCLUDED;
  }else if(line->status==LINE_NET_OK_HEATING){
   l->heating[A210_LINE_ABS]=-line->signed_rate;
   l->heating_status[A210_LINE_ABS]=A210_INCLUDED;
  }
  if(ff_abs>=0.0)l->heating[A210_FF_ABS]=ff_abs;
  else ff_emit-=ff_abs;
  l->heating_status[A210_FF_ABS]=ff_abs==0.0?
      A210_EXACT_ZERO:A210_INCLUDED;
  l->cooling[A210_RECOMB]=recomb;
  l->cooling_status[A210_RECOMB]=recomb==0.0?
      A210_EXACT_ZERO:A210_INCLUDED;
  l->cooling[A210_FF_EMIT]=ff_emit;
  l->cooling_status[A210_FF_EMIT]=ff_emit==0.0?
      A210_EXACT_ZERO:A210_INCLUDED;
  if(j_int>0.0){
   double trad=A210_H_PLANCK*(jnu_int/j_int)/(4.0*A210_K_BOLTZMANN);
   double q=4.0*A210_K_BOLTZMANN*A210_SIGMA_THOMSON*ne/
            (A210_M_ELECTRON*A210_C_LIGHT*A210_C_LIGHT)*
            A210_FOUR_PI*j_int*(trad-te);
   if(!isfinite(q)){result=RADEQ_NONFINITE;break;}
   if(q>=0.0){l->heating[A210_COMPTON_H]=q;
    l->heating_status[A210_COMPTON_H]=q==0.0?A210_EXACT_ZERO:A210_INCLUDED;}
   else{l->cooling[A210_COMPTON_C]=-q;
    l->cooling_status[A210_COMPTON_C]=A210_INCLUDED;}
  }
  double qg=in->gamma_heating_rate?in->gamma_heating_rate[s]:0.0;
  if(!isfinite(qg)||qg<0.0){result=RADEQ_SIGN_MISMATCH;break;}
  l->heating[A210_GAMMA]=qg;
  l->heating_status[A210_GAMMA]=qg==0.0?A210_EXACT_ZERO:A210_INCLUDED;
  l->heating_status[A210_NONTHERMAL]=A210_EXACT_ZERO;
  l->cooling_status[A210_COLL_LINE]=A210_REPLACED_NOT_APPLICABLE;
  result=a210_apply_cmfgen_adiabatic(l,&in->adiabatic[s]);
  if(result!=RADEQ_OK)break;
  result=a210_line_owner_finalize(l);
  if(result!=RADEQ_OK&&result!=RADEQ_EXACT_ZERO_BALANCE)break;
  result=RADEQ_OK;
 }
 if(result==RADEQ_OK)memcpy(out,candidate,ns*sizeof(*out));
 free(candidate);return result;
}
RadeqStatus a210_line_owner_finalize(A210TermLedger*l){if(!l)return RADEQ_TERM_SCHEMA;if(l->equation_kind!=A210_RE_INTEGRAL&&l->equation_kind!=A210_EHB_THERMAL){g.blocked_schema++;return l->status=RADEQ_TERM_SCHEMA;}l->line_owner_overlap=l->radiative_line_included&&l->collisional_or_escape_included;if(l->m_line!=0&&l->m_line!=1){g.blocked_schema++;return l->status=RADEQ_TERM_SCHEMA;}if(l->line_owner_overlap){g.line_owner_overlap_shells++;g.blocked_schema++;return l->status=RADEQ_TERM_SCHEMA;}if((l->m_line==1)!=!!l->radiative_line_included||(l->m_line==0)!=!!l->collisional_or_escape_included){g.blocked_schema++;return l->status=RADEQ_TERM_SCHEMA;}if(l->m_line){l->cooling_status[A210_COLL_LINE]=A210_REPLACED_NOT_APPLICABLE;}else{l->heating_status[A210_LINE_ABS]=A210_REPLACED_NOT_APPLICABLE;l->cooling_status[A210_LINE_EMIT]=A210_REPLACED_NOT_APPLICABLE;}
 l->sum_heating=sum(l->heating,A210_NHEAT);l->sum_cooling=sum(l->cooling,A210_NCOOL);l->residual=l->sum_heating-l->sum_cooling;double hn=l->heating[A210_LINE_ABS],cn=l->cooling[A210_LINE_EMIT]+l->cooling[A210_COLL_LINE];double non=(l->sum_heating-hn)-(l->sum_cooling-cn);double own=l->m_line*(-l->Q_line_rad)-(1-l->m_line)*l->C_line_ce;l->line_owner_closure=l->residual-(non+own);double scale=0;for(int i=0;i<A210_NHEAT;i++)scale+=fabs(l->heating[i]);for(int i=0;i<A210_NCOOL;i++)scale+=fabs(l->cooling[i]);scale+=fabs(own);l->normalized_line_owner_closure=scale?fabs(l->line_owner_closure)/scale:fabs(l->line_owner_closure);if((scale==0&&l->line_owner_closure!=0)||l->normalized_line_owner_closure>1e-12){g.line_owner_closure_failures++;g.blocked_schema++;if(l->normalized_line_owner_closure>g.max_line_owner_closure)g.max_line_owner_closure=l->normalized_line_owner_closure;return l->status=RADEQ_TERM_SCHEMA;}double den=l->sum_heating+l->sum_cooling;l->e_balance=den?fabs(l->residual)/den:(l->residual==0?0:INFINITY);if(l->adiabatic_model!=A210_ADIABATIC_CMFGEN_COMPLETE||l->cooling_status[A210_ADIABATIC]==A210_INCOMPLETE)return l->status=RADEQ_INCOMPLETE_ADIABATIC;l->status=(den==0)?RADEQ_EXACT_ZERO_BALANCE:RADEQ_OK;return l->status;}
RadeqStatus a210_te_context_sha256(const char te[65],const char geo[65],uint64_t ep,char out[65]){if(!te||!geo||!out||strlen(te)!=64||strlen(geo)!=64)return RADEQ_TE_CONTEXT_MISMATCH;Sh s;si(&s);const char d[]="A2-10:Te-context:v1";su(&s,d,sizeof(d)-1);su(&s,te,64);su(&s,geo,64);u64(&s,ep);sd(&s,out);return RADEQ_OK;}
RadeqStatus a210_geometry_sha256(const double*x,size_t n,char out[65]){if(!x||n<2||!out)return RADEQ_TE_CONTEXT_MISMATCH;Sh s;si(&s);const char d[]="A2-10:shell-geometry:cm-s^-1:IEEE754:v1";su(&s,d,sizeof(d)-1);u64(&s,n);for(size_t i=0;i<n;i++){if(!isfinite(x[i]))return RADEQ_TE_CONTEXT_MISMATCH;uint64_t v;memcpy(&v,&x[i],8);u64(&s,v);}sd(&s,out);return RADEQ_OK;}
int a210_solve_transaction(ElectronTemperaturePublication*pub,const double*lo,const double*hi,const double*ne,size_t n,uint64_t ep,uint64_t gen,const char*geo,A210ResidualFunction fn,void*ctx,double*outte,double*outne){if(!pub||!lo||!hi||!ne||!geo||!fn||!outte||!outne||!n||!gen)return 2;ElectronTemperaturePublication c={0};if(a210_publication_init(&c,n))return 2;c.solve_epoch=ep;c.required_te_generation=gen;memcpy(c.geometry_sha256,geo,65);g.solve_epoch=ep;g.te_generation_required=gen;g.shells_attempted+=n;
 for(size_t s=0;s<n;s++){if(!(isfinite(lo[s])&&isfinite(hi[s])&&lo[s]>0&&hi[s]>lo[s]&&isfinite(ne[s])&&ne[s]>=0)){g.nonfinite_failures++;a210_publication_free(&c);return 5;}A210TermLedger a={0},b={0};RadeqStatus sa=fn(s,lo[s],&a,ctx),sbv=fn(s,hi[s],&b,ctx);g.trials+=2;if(sa==RADEQ_INCOMPLETE_ADIABATIC||sbv==RADEQ_INCOMPLETE_ADIABATIC){g.blocked_incomplete_adiabatic++;a210_publication_free(&c);return 3;}if((sa!=RADEQ_OK&&sa!=RADEQ_EXACT_ZERO_BALANCE)||(sbv!=RADEQ_OK&&sbv!=RADEQ_EXACT_ZERO_BALANCE)||a.equation_kind!=A210_RE_INTEGRAL||b.equation_kind!=A210_RE_INTEGRAL){g.blocked_schema++;a210_publication_free(&c);return 5;}if(c.producer_equation==A210_EQUATION_NONE)c.producer_equation=A210_RE_INTEGRAL;double fa=a.residual,fb=b.residual;if(fa==0){c.T_e[s]=lo[s];c.ledger[s]=a;}else if(fb==0){c.T_e[s]=hi[s];c.ledger[s]=b;}else if(signbit(fa)==signbit(fb)){{static int _said=0; if(!_said){_said=1;static const char*HN[]={"photo","line_abs","ff_abs","compton_h","gamma","nonthermal","adiabatic_h"};static const char*CN[]={"recomb","line_emit","coll_line","ff_emit","compton_c","adiabatic"};fprintf(stderr,"[A2-10][NOBRACKET-DIAG] shell=%zu T_lo=%.6g T_hi=%.6g "
  "res_lo=%.6e res_hi=%.6e heat_lo=%.6e cool_lo=%.6e heat_hi=%.6e cool_hi=%.6e\n",
  s,lo[s],hi[s],a.residual,b.residual,a.sum_heating,a.sum_cooling,b.sum_heating,b.sum_cooling);for(int _k=0;_k<A210_NHEAT;_k++)fprintf(stderr,"[A2-10][NOBRACKET-DIAG]   H %-10s lo=%.6e hi=%.6e\n",HN[_k],a.heating[_k],b.heating[_k]);for(int _k=0;_k<A210_NCOOL;_k++)fprintf(stderr,"[A2-10][NOBRACKET-DIAG]   C %-10s lo=%.6e hi=%.6e\n",CN[_k],a.cooling[_k],b.cooling[_k]);}}g.no_bracket++;a210_publication_free(&c);return 4;}else{double l=lo[s],h=hi[s];A210TermLedger m={0};for(int it=0;it<160;it++){double t=0.5*(l+h);RadeqStatus sm=fn(s,t,&m,ctx);g.trials++;if(sm!=RADEQ_OK&&sm!=RADEQ_EXACT_ZERO_BALANCE){g.nonconverged++;a210_publication_free(&c);return 5;}if(fabs(m.residual)<=1e-12*fmax(m.sum_heating+m.sum_cooling,DBL_MIN)||fabs(h-l)<=1e-12*t){c.T_e[s]=t;c.ledger[s]=m;break;}if(signbit(m.residual)==signbit(fa)){l=t;fa=m.residual;}else h=t;}if(c.T_e[s]==0){g.nonconverged++;a210_publication_free(&c);return 4;}}c.n_e[s]=ne[s];c.shell_status[s]=RADEQ_OK;c.residual_status[s]=c.ledger[s].e_balance<=1e-3?RADEQ_OK:RADEQ_HEAT_RESIDUAL;if(c.residual_status[s]!=RADEQ_OK){g.max_heat_residual=fmax(g.max_heat_residual,c.ledger[s].e_balance);a210_publication_free(&c);return 4;}if(c.ledger[s].m_line){g.line_radiative_owner_shells++;g.line_replaced_collisional_terms++;}else{g.line_collisional_escape_owner_shells++;g.line_replaced_radiative_terms+=2;}g.shells_converged++;}
 if(population_te_manifest_sha256(c.T_e,n,c.te_manifest_sha256)!=POP_OK){g.te_manifest_mismatch++;a210_publication_free(&c);return 5;}if(a210_te_context_sha256(c.te_manifest_sha256,geo,ep,c.te_context_sha256)!=RADEQ_OK){g.te_context_mismatch++;a210_publication_free(&c);return 5;}Sh sh;si(&sh);const char d[]="A2-10:term-ledger:v1";su(&sh,d,sizeof(d)-1);u64(&sh,ep);for(size_t s=0;s<n;s++)su(&sh,&c.ledger[s],sizeof(c.ledger[s]));sd(&sh,c.term_manifest_sha256);c.te_lane=A210_TE_LANE_FREE_T;c.re_root_required=1;c.committed_te_generation=gen;memcpy(outte,c.T_e,n*sizeof(double));memcpy(outne,c.n_e,n*sizeof(double));ElectronTemperaturePublication old=*pub;*pub=c;memset(&c,0,sizeof(c));a210_publication_free(&old);g.te_generation_committed=gen;g.shells_published+=n;return 0;}

static int a210_vector_ledgers_valid(
        RadeqStatus status,const A210TermLedger *ledger,size_t n){
 if(status!=RADEQ_OK&&status!=RADEQ_EXACT_ZERO_BALANCE)return 0;
 if(!ledger)return 0;
 for(size_t s=0;s<n;s++){
  if((ledger[s].status!=RADEQ_OK&&
      ledger[s].status!=RADEQ_EXACT_ZERO_BALANCE)||
     ledger[s].equation_kind!=A210_RE_INTEGRAL||
     ledger[s].adiabatic_model!=A210_ADIABATIC_CMFGEN_COMPLETE||
     ledger[s].cooling_status[A210_ADIABATIC]==A210_INCOMPLETE||
     !isfinite(ledger[s].residual)||!isfinite(ledger[s].sum_heating)||
     !isfinite(ledger[s].sum_cooling))return 0;
 }
 return 1;
}

static void a210_vector_ledger_diagnostic(
        const char *phase,RadeqStatus status,
        const A210TermLedger *ledger,size_t n){
 size_t first=n;
 if(ledger)for(size_t s=0;s<n;s++)if(
    (ledger[s].status!=RADEQ_OK&&
     ledger[s].status!=RADEQ_EXACT_ZERO_BALANCE)||
    ledger[s].equation_kind!=A210_RE_INTEGRAL||
    ledger[s].adiabatic_model!=A210_ADIABATIC_CMFGEN_COMPLETE||
    ledger[s].cooling_status[A210_ADIABATIC]==A210_INCOMPLETE||
    !isfinite(ledger[s].residual)||
    !isfinite(ledger[s].sum_heating)||
    !isfinite(ledger[s].sum_cooling)){first=s;break;}
 if(first<n){
  const A210TermLedger*l=&ledger[first];
  fprintf(stderr,
      "[A2-10][VECTOR-LEDGER-BLOCKED] phase=%s vector_status=%s "
      "first_shell=%zu ledger_status=%s equation_kind=%d "
      "adiabatic_model=%d adiabatic_status=%d residual=%.17g "
      "heating=%.17g cooling=%.17g\n",phase,a210_status_name(status),
      first,a210_status_name(l->status),(int)l->equation_kind,
      (int)l->adiabatic_model,(int)l->cooling_status[A210_ADIABATIC],
      l->residual,l->sum_heating,l->sum_cooling);
 }else{
  fprintf(stderr,
      "[A2-10][VECTOR-LEDGER-BLOCKED] phase=%s vector_status=%s "
      "first_shell=NONE ledger=%s n_shells=%zu\n",phase,
      a210_status_name(status),ledger?"PRESENT":"NULL",n);
 }
}

/* Preserve the callback's actual failure class when a complete ledger vector
 * cannot be published.  The old path unconditionally charged blocked_schema,
 * which turned a witnessed sign/cancellation failure into RADEQ_TERM_SCHEMA at
 * the R7 boundary.  This function changes counters only; candidate bytes,
 * return codes, and every numerical value remain untouched. */
static void a210_note_vector_callback_failure(
        RadeqStatus a,RadeqStatus b,const A210Counters *before){
 int sign=a==RADEQ_SIGN_MISMATCH||b==RADEQ_SIGN_MISMATCH;
 int nonfinite=a==RADEQ_NONFINITE||b==RADEQ_NONFINITE;
 int gamma=a==RADEQ_GAMMA_UNPUBLISHED||b==RADEQ_GAMMA_UNPUBLISHED;
 int missing=a==RADEQ_TERM_MISSING||b==RADEQ_TERM_MISSING||
             a==RADEQ_ATOMIC_MISSING||b==RADEQ_ATOMIC_MISSING;
 int stale=a==RADEQ_STALE_RF||a==RADEQ_STALE_BF||
           a==RADEQ_STALE_LINE||a==RADEQ_STALE_POP||
           a==RADEQ_STALE_OPACITY||a==RADEQ_STALE_EMISSIVITY||
           b==RADEQ_STALE_RF||b==RADEQ_STALE_BF||
           b==RADEQ_STALE_LINE||b==RADEQ_STALE_POP||
           b==RADEQ_STALE_OPACITY||b==RADEQ_STALE_EMISSIVITY;
 int charge=a==RADEQ_CHARGE_NOT_CONVERGED||
            b==RADEQ_CHARGE_NOT_CONVERGED;
 int nonconverged=a==RADEQ_POPULATION_NOT_CONVERGED||
                  a==RADEQ_NOT_CONVERGED||a==RADEQ_NO_ROOT||
                  b==RADEQ_POPULATION_NOT_CONVERGED||
                  b==RADEQ_NOT_CONVERGED||b==RADEQ_NO_ROOT;
 if(sign){
  if(g.blocked_sign==before->blocked_sign)g.blocked_sign++;
  return;
 }
 if(nonfinite){
  if(g.nonfinite_failures==before->nonfinite_failures)
   g.nonfinite_failures++;
  return;
 }
 if(gamma){
  if(g.blocked_gamma_unpublished==before->blocked_gamma_unpublished)
   g.blocked_gamma_unpublished++;
  return;
 }
 if(missing){
  if(g.blocked_missing_term==before->blocked_missing_term)
   g.blocked_missing_term++;
  return;
 }
 if(stale){
  if(g.blocked_stale==before->blocked_stale)g.blocked_stale++;
  return;
 }
 if(charge){
  if(g.charge_nonconverged==before->charge_nonconverged)
   g.charge_nonconverged++;
  return;
 }
 if(nonconverged){
  if(g.nonconverged==before->nonconverged)g.nonconverged++;
  return;
 }
 if(g.blocked_schema==before->blocked_schema)g.blocked_schema++;
}

static void a210_vector_no_bracket_diagnostic(
        const double *lower,const double *upper,
        const A210TermLedger *llo,const A210TermLedger *lhi,size_t n){
 static const char*HN[]={"photo","line_abs","ff_abs","compton_h",
                         "gamma","nonthermal","adiabatic_h"};
 static const char*CN[]={"recomb","line_emit","coll_line","ff_emit",
                         "compton_c","adiabatic"};
 size_t first=n,count=0,same_positive=0,same_negative=0,endpoint_zero=0;
 for(size_t s=0;s<n;s++){
  double flo=llo[s].residual,fhi=lhi[s].residual;
  if(flo==0.0||fhi==0.0||signbit(flo)==signbit(fhi)){
   if(first==n)first=s;
   count++;
   if(flo==0.0||fhi==0.0)endpoint_zero++;
   else if(signbit(flo))same_negative++;
   else same_positive++;
  }
 }
 if(first==n)return;
 const A210TermLedger *a=&llo[first],*b=&lhi[first];
 fprintf(stderr,
   "[A2-10][VECTOR-NOBRACKET] count=%zu first_shell=%zu "
   "same_positive=%zu same_negative=%zu endpoint_zero=%zu "
   "T_lo=%.17g T_hi=%.17g res_lo=%.17g res_hi=%.17g "
   "heat_lo=%.17g cool_lo=%.17g heat_hi=%.17g cool_hi=%.17g\n",
   count,first,same_positive,same_negative,endpoint_zero,
   lower[first],upper[first],a->residual,b->residual,
   a->sum_heating,a->sum_cooling,b->sum_heating,b->sum_cooling);
 for(int k=0;k<A210_NHEAT;k++)fprintf(stderr,
   "[A2-10][VECTOR-NOBRACKET] first_shell=%zu H_%s_lo=%.17g H_%s_hi=%.17g\n",
   first,HN[k],a->heating[k],HN[k],b->heating[k]);
 for(int k=0;k<A210_NCOOL;k++)fprintf(stderr,
   "[A2-10][VECTOR-NOBRACKET] first_shell=%zu C_%s_lo=%.17g C_%s_hi=%.17g\n",
   first,CN[k],a->cooling[k],CN[k],b->cooling[k]);
}

/* Endpoint signs alone do not exclude a non-monotone interior root.  In the
 * diagnostic lane, inspect explicitly selected private material states before
 * returning the unchanged fail-closed NO_BRACKET result.  This routine only
 * reports the result; it never edits the solve brackets or publication
 * candidate. */
static void a210_vector_interior_diagnostic(
        const A210TermLedger *llo,const A210TermLedger *lhi,
        const double *middle,const A210TermLedger *lmid,
        RadeqStatus middle_status,size_t n,const char *phase){
 if(!phase)phase="UNSPECIFIED";
 int valid=a210_vector_ledgers_valid(middle_status,lmid,n);
 if(!valid){
  fprintf(stderr,
      "[A2-10][VECTOR-INTERIOR-SCAN] phase=%s "
      "status=%s valid=0 action=DIAGNOSTIC_ONLY\n",
      phase,a210_status_name(middle_status));
  return;
 }
 size_t endpoint_no_bracket=0,interior_bracket=0,still_same_sign=0;
 for(size_t s=0;s<n;s++){
  double flo=llo[s].residual,fhi=lhi[s].residual,fmid=lmid[s].residual;
  if(flo!=0.0&&fhi!=0.0&&signbit(flo)!=signbit(fhi))continue;
  int lo_mid=(flo==0.0||fmid==0.0||signbit(flo)!=signbit(fmid));
  int mid_hi=(fmid==0.0||fhi==0.0||signbit(fmid)!=signbit(fhi));
  endpoint_no_bracket++;
  if(lo_mid||mid_hi)interior_bracket++;else still_same_sign++;
  fprintf(stderr,
      "[A2-10][VECTOR-INTERIOR-SCAN] phase=%s shell=%zu "
      "T_mid=%.17g res_lo=%.17g res_mid=%.17g res_hi=%.17g "
      "heat_mid=%.17g cool_mid=%.17g line_emit_mid=%.17g "
      "lo_mid_bracket=%d mid_hi_bracket=%d action=DIAGNOSTIC_ONLY\n",
      phase,s,middle[s],flo,fmid,fhi,lmid[s].sum_heating,
      lmid[s].sum_cooling,lmid[s].cooling[A210_LINE_EMIT],
      lo_mid,mid_hi);
 }
 fprintf(stderr,
     "[A2-10][VECTOR-INTERIOR-SCAN] phase=%s valid=1 "
     "endpoint_no_bracket=%zu interior_bracket=%zu still_same_sign=%zu "
     "action=DIAGNOSTIC_ONLY solver_result=RADEQ_NO_BRACKET\n",
     phase,endpoint_no_bracket,interior_bracket,still_same_sign);
}

static int a210_solve_vector_impl(
 const double*lower,const double*upper,const double*ne,size_t n,
 uint64_t ep,uint64_t gen,const char*geo,A210VectorResidualFunction fn,
 void*ctx,ElectronTemperaturePublication*pub,double*outte,double*outne,
 int authorize_publication){
 if(!lower||!upper||!ne||!geo||!fn||!pub||!outte||!outne||!n||!gen)
  return 2;
 for(size_t s=0;s<n;s++)
  if(!isfinite(lower[s])||!isfinite(upper[s])||lower[s]<=0.0||
     upper[s]<=lower[s]||!isfinite(ne[s])||ne[s]<0.0){
   g.nonfinite_failures++;return 5;
  }
 ElectronTemperaturePublication c={0};
 if(a210_publication_init(&c,n))return 2;
 double*lo=malloc(n*sizeof(*lo)),*hi=malloc(n*sizeof(*hi));
 double*mid=malloc(n*sizeof(*mid)),*flo=malloc(n*sizeof(*flo));
 double*fhi=malloc(n*sizeof(*fhi));
 A210TermLedger*llo=calloc(n,sizeof(*llo));
 A210TermLedger*lhi=calloc(n,sizeof(*lhi));
 A210TermLedger*lmid=calloc(n,sizeof(*lmid));
 if(!lo||!hi||!mid||!flo||!fhi||!llo||!lhi||!lmid){
  free(lo);free(hi);free(mid);free(flo);free(fhi);
  free(llo);free(lhi);free(lmid);a210_publication_free(&c);return 2;
 }
 memcpy(lo,lower,n*sizeof(*lo));memcpy(hi,upper,n*sizeof(*hi));
 c.solve_epoch=ep;c.required_te_generation=gen;memcpy(c.geometry_sha256,geo,65);
 g.solve_epoch=ep;g.te_generation_required=gen;g.shells_attempted+=n;
 A210Counters endpoint_counters_before=g;
 RadeqStatus slo=fn(lo,n,llo,ctx),shi=fn(hi,n,lhi,ctx);g.trials+=2;
 if(slo==RADEQ_INCOMPLETE_ADIABATIC||shi==RADEQ_INCOMPLETE_ADIABATIC){
  g.blocked_incomplete_adiabatic++;
  free(lo);free(hi);free(mid);free(flo);free(fhi);
  free(llo);free(lhi);free(lmid);a210_publication_free(&c);return 3;
 }
 int valid_lo=a210_vector_ledgers_valid(slo,llo,n);
 int valid_hi=a210_vector_ledgers_valid(shi,lhi,n);
 if(!valid_lo||!valid_hi){
  if(!valid_lo)a210_vector_ledger_diagnostic("LOWER",slo,llo,n);
  if(!valid_hi)a210_vector_ledger_diagnostic("UPPER",shi,lhi,n);
  a210_note_vector_callback_failure(
      valid_lo?RADEQ_OK:slo,valid_hi?RADEQ_OK:shi,
      &endpoint_counters_before);
  free(lo);free(hi);free(mid);free(flo);free(fhi);
  free(llo);free(lhi);free(lmid);a210_publication_free(&c);return 5;
 }
 int all_lo_exact=1,all_hi_exact=1;
 for(size_t s=0;s<n;s++){
  flo[s]=llo[s].residual;fhi[s]=lhi[s].residual;
  if(flo[s]!=0.0)all_lo_exact=0;
  if(fhi[s]!=0.0)all_hi_exact=0;
 }
 if(all_lo_exact){memcpy(c.T_e,lo,n*sizeof(*lo));memcpy(c.ledger,llo,n*sizeof(*llo));}
 else if(all_hi_exact){memcpy(c.T_e,hi,n*sizeof(*hi));memcpy(c.ledger,lhi,n*sizeof(*lhi));}
 else{
  for(size_t s=0;s<n;s++)if(flo[s]==0.0||fhi[s]==0.0||
      signbit(flo[s])==signbit(fhi[s])){
   a210_vector_no_bracket_diagnostic(lower,upper,llo,lhi,n);
   if(getenv("LUMINA_RADEQ_DIAG")){
    int old_valid=1;
    for(size_t q=0;q<n;q++)
     if(!isfinite(outte[q])||outte[q]<=lower[q]||outte[q]>=upper[q]){
      old_valid=0;break;
     }
    if(old_valid){
     memcpy(mid,outte,n*sizeof(*mid));
     memset(lmid,0,n*sizeof(*lmid));
     RadeqStatus sold=fn(mid,n,lmid,ctx);g.trials++;
     g.diagnostic_seed_trials++;
     a210_vector_interior_diagnostic(
         llo,lhi,mid,lmid,sold,n,"PUBLIC_SEED");
    }else{
     fprintf(stderr,
         "[A2-10][VECTOR-INTERIOR-SCAN] phase=PUBLIC_SEED valid=0 "
         "reason=OUTSIDE_OPEN_BRACKET action=DIAGNOSTIC_ONLY\n");
    }
    double requested_te=0.0;
    int requested_state=a210_requested_diagnostic_te(&requested_te);
    if(requested_state<0){
     fprintf(stderr,
         "[A2-10][VECTOR-INTERIOR-SCAN] phase=REQUESTED_TE valid=0 "
         "reason=INVALID_REQUESTED_TE value=%s action=DIAGNOSTIC_ONLY "
         "physical_values_modified=0\n",
         getenv("LUMINA_RADEQ_DIAG_TE_K"));
    }else if(requested_state>0){
     int requested_valid=1;
     for(size_t q=0;q<n;q++)
      if(!(requested_te>lower[q]&&requested_te<upper[q])){
       requested_valid=0;break;
      }
     if(requested_valid){
      for(size_t q=0;q<n;q++)mid[q]=requested_te;
      memset(lmid,0,n*sizeof(*lmid));
      RadeqStatus srequested=fn(mid,n,lmid,ctx);g.trials++;
      g.diagnostic_requested_te_trials++;
      a210_vector_interior_diagnostic(
          llo,lhi,mid,lmid,srequested,n,"REQUESTED_TE");
     }else{
      fprintf(stderr,
          "[A2-10][VECTOR-INTERIOR-SCAN] phase=REQUESTED_TE valid=0 "
          "reason=OUTSIDE_OPEN_BRACKET T_requested=%.17g "
          "action=DIAGNOSTIC_ONLY physical_values_modified=0\n",
          requested_te);
     }
    }
    for(size_t q=0;q<n;q++)
     mid[q]=exp(0.5*(log(lower[q])+log(upper[q])));
    memset(lmid,0,n*sizeof(*lmid));
    RadeqStatus smid=fn(mid,n,lmid,ctx);g.trials++;
    a210_vector_interior_diagnostic(
        llo,lhi,mid,lmid,smid,n,"GEOMETRIC_MID");
   }
   g.no_bracket++;
   free(lo);free(hi);free(mid);free(flo);free(fhi);
   free(llo);free(lhi);free(lmid);a210_publication_free(&c);return 4;
  }
  int converged=0,vector_failure=0;
  for(int iteration=0;iteration<160;iteration++){
   for(size_t s=0;s<n;s++)mid[s]=0.5*(lo[s]+hi[s]);
   memset(lmid,0,n*sizeof(*lmid));
   A210Counters mid_counters_before=g;
   RadeqStatus smid=fn(mid,n,lmid,ctx);g.trials++;
   if(smid==RADEQ_INCOMPLETE_ADIABATIC){
    g.blocked_incomplete_adiabatic++;vector_failure=3;break;
   }
   if(!a210_vector_ledgers_valid(smid,lmid,n)){
    a210_vector_ledger_diagnostic("MID",smid,lmid,n);
    a210_note_vector_callback_failure(
        smid,RADEQ_OK,&mid_counters_before);
    vector_failure=5;break;
   }
   int all=1;
   for(size_t s=0;s<n;s++){
    double scale=fmax(lmid[s].sum_heating+lmid[s].sum_cooling,DBL_MIN);
    if(fabs(lmid[s].residual)>1e-12*scale&&
       fabs(hi[s]-lo[s])>1e-12*mid[s])all=0;
   }
   if(all){
    memcpy(c.T_e,mid,n*sizeof(*mid));
    memcpy(c.ledger,lmid,n*sizeof(*lmid));converged=1;break;
   }
   for(size_t s=0;s<n;s++){
    double fm=lmid[s].residual;
    if(fm==0.0)continue;
    if(signbit(fm)==signbit(flo[s])){lo[s]=mid[s];flo[s]=fm;}
    else{hi[s]=mid[s];fhi[s]=fm;}
   }
  }
  if(!converged){
   if(!vector_failure){g.nonconverged++;vector_failure=4;}
   free(lo);free(hi);free(mid);free(flo);free(fhi);
   free(llo);free(lhi);free(lmid);a210_publication_free(&c);
   return vector_failure;
  }
 }
 for(size_t s=0;s<n;s++){
  c.n_e[s]=ne[s];c.shell_status[s]=RADEQ_OK;
  c.residual_status[s]=c.ledger[s].e_balance<=1e-3?
      RADEQ_OK:RADEQ_HEAT_RESIDUAL;
  if(c.residual_status[s]!=RADEQ_OK){
   g.max_heat_residual=fmax(g.max_heat_residual,c.ledger[s].e_balance);
   free(lo);free(hi);free(mid);free(flo);free(fhi);
   free(llo);free(lhi);free(lmid);a210_publication_free(&c);return 4;
  }
 }
 if(population_te_manifest_sha256(c.T_e,n,c.te_manifest_sha256)!=POP_OK){
  g.te_manifest_mismatch++;
  free(lo);free(hi);free(mid);free(flo);free(fhi);
  free(llo);free(lhi);free(lmid);a210_publication_free(&c);return 5;
 }
 if(a210_te_context_sha256(c.te_manifest_sha256,geo,ep,
                           c.te_context_sha256)!=RADEQ_OK){
  g.te_context_mismatch++;
  free(lo);free(hi);free(mid);free(flo);free(fhi);
  free(llo);free(lhi);free(lmid);a210_publication_free(&c);return 5;
 }
 Sh sh;si(&sh);const char domain[]="A2-10:term-ledger:v1";
 su(&sh,domain,sizeof(domain)-1);u64(&sh,ep);
 for(size_t s=0;s<n;s++)su(&sh,&c.ledger[s],sizeof(c.ledger[s]));
 sd(&sh,c.term_manifest_sha256);c.producer_equation=A210_RE_INTEGRAL;
 c.te_lane=A210_TE_LANE_FREE_T;c.re_root_required=1;
 c.committed_te_generation=authorize_publication?gen:0;
 memcpy(outte,c.T_e,n*sizeof(*outte));memcpy(outne,c.n_e,n*sizeof(*outne));
 ElectronTemperaturePublication old=*pub;*pub=c;memset(&c,0,sizeof(c));
 a210_publication_free(&old);
 if(authorize_publication)g.te_generation_committed=gen;
 g.shells_converged+=n;
 if(authorize_publication)g.shells_published+=n;
 free(lo);free(hi);free(mid);free(flo);free(fhi);
 free(llo);free(lhi);free(lmid);return 0;
}
int a210_solve_vector_transaction(
 const double*lower,const double*upper,const double*ne,size_t n,
 uint64_t ep,uint64_t gen,const char*geo,A210VectorResidualFunction fn,
 void*ctx,ElectronTemperaturePublication*pub,double*outte,double*outne){
 return a210_solve_vector_impl(lower,upper,ne,n,ep,gen,geo,fn,ctx,pub,
                               outte,outne,1);
}
int a210_solve_vector_candidate(
 const double*lower,const double*upper,const double*ne,size_t n,
 uint64_t ep,uint64_t gen,const char*geo,A210VectorResidualFunction fn,
 void*ctx,ElectronTemperaturePublication*pub,double*outte,double*outne){
 return a210_solve_vector_impl(lower,upper,ne,n,ep,gen,geo,fn,ctx,pub,
                               outte,outne,0);
}
A210Counters*a210_counters(void){return &g;}void a210_counters_reset(void){memset(&g,0,sizeof(g));}
const char*a210_status_name(RadeqStatus s){static const char*n[]={"INVALID","RADEQ_OK","RADEQ_EXACT_ZERO_BALANCE","RADEQ_STALE_RF","RADEQ_STALE_BF","RADEQ_STALE_LINE","RADEQ_STALE_POP","RADEQ_STALE_OPACITY","RADEQ_STALE_EMISSIVITY","RADEQ_TERM_MISSING","RADEQ_TERM_SCHEMA","RADEQ_SIGN_MISMATCH","RADEQ_INVALID_TE_TRIAL","RADEQ_INVALID_NE","RADEQ_ATOMIC_MISSING","RADEQ_NO_BRACKET","RADEQ_NO_ROOT","RADEQ_NOT_CONVERGED","RADEQ_POPULATION_NOT_CONVERGED","RADEQ_CHARGE_NOT_CONVERGED","RADEQ_HEAT_RESIDUAL","RADEQ_TE_MANIFEST_MISMATCH","RADEQ_TE_CONTEXT_MISMATCH","RADEQ_UNQUALIFIED_TE","RADEQ_FIXED_T","RADEQ_FORBIDDEN_FALLBACK","RADEQ_PARTIAL_PUBLISH","RADEQ_INCOMPLETE_ADIABATIC","RADEQ_NONFINITE","RADEQ_GAMMA_UNPUBLISHED"};return s>=RADEQ_OK&&s<=RADEQ_GAMMA_UNPUBLISHED?n[s]:n[0];}
const char*a210_equation_name(A210EquationKind k){return k==A210_RE_INTEGRAL?"RE_INTEGRAL":k==A210_EHB_THERMAL?"EHB_THERMAL":"EQUATION_NONE";}
const char*a210_adiabatic_model_name(A210AdiabaticModel m){return m==A210_ADIABATIC_CMFGEN_COMPLETE?"CMFGEN_COMPLETE":m==A210_ADIABATIC_ELECTRON_TRANSLATIONAL_ONLY?"ELECTRON_TRANSLATIONAL_ONLY":"ADIABATIC_NONE";}
void a210_counters_print(FILE*f){if(!f)f=stdout;fprintf(f,"[A2-10][RADEQ] solve_epoch=%llu te_generation_required=%llu te_generation_committed=%llu shells_attempted=%llu shells_converged=%llu shells_published=%llu trials=%llu population_trials=%llu opacity_trials=%llu emissivity_trials=%llu photo_heat_terms=%llu line_heat_terms=%llu ff_heat_terms=%llu compton_heat_terms=%llu gamma_heat_terms=%llu nonthermal_heat_terms=%llu adiabatic_heat_terms=%llu recomb_cool_terms=%llu line_cool_terms=%llu collisional_cool_terms=%llu ff_cool_terms=%llu compton_cool_terms=%llu adiabatic_cool_terms=%llu exact_zero_balance=%llu no_bracket=%llu no_root=%llu nonconverged=%llu charge_nonconverged=%llu blocked_stale=%llu blocked_missing_term=%llu blocked_gamma_unpublished=%llu blocked_schema=%llu blocked_sign=%llu blocked_incomplete_adiabatic=%llu te_manifest_mismatch=%llu te_context_mismatch=%llu fixed_te_attempts=%llu seed_generation_attempts=%llu line_radiative_owner_shells=%llu line_collisional_escape_owner_shells=%llu line_replaced_collisional_terms=%llu line_replaced_radiative_terms=%llu line_owner_overlap_shells=%llu line_owner_closure_failures=%llu diagnostic_seed_trials=%llu diagnostic_requested_te_trials=%llu pin_attempts=%llu floor_attempts=%llu neighbor_attempts=%llu old_te_attempts=%llu fallback_attempts=%llu partial_publish_attempts=%llu nonfinite_failures=%llu max_line_owner_closure=%.17g te_lane=%s max_heat_residual=%.17g\n",(unsigned long long)g.solve_epoch,(unsigned long long)g.te_generation_required,(unsigned long long)g.te_generation_committed,(unsigned long long)g.shells_attempted,(unsigned long long)g.shells_converged,(unsigned long long)g.shells_published,(unsigned long long)g.trials,(unsigned long long)g.population_trials,(unsigned long long)g.opacity_trials,(unsigned long long)g.emissivity_trials,(unsigned long long)g.photo_heat_terms,(unsigned long long)g.line_heat_terms,(unsigned long long)g.ff_heat_terms,(unsigned long long)g.compton_heat_terms,(unsigned long long)g.gamma_heat_terms,(unsigned long long)g.nonthermal_heat_terms,(unsigned long long)g.adiabatic_heat_terms,(unsigned long long)g.recomb_cool_terms,(unsigned long long)g.line_cool_terms,(unsigned long long)g.collisional_cool_terms,(unsigned long long)g.ff_cool_terms,(unsigned long long)g.compton_cool_terms,(unsigned long long)g.adiabatic_cool_terms,(unsigned long long)g.exact_zero_balance,(unsigned long long)g.no_bracket,(unsigned long long)g.no_root,(unsigned long long)g.nonconverged,(unsigned long long)g.charge_nonconverged,(unsigned long long)g.blocked_stale,(unsigned long long)g.blocked_missing_term,(unsigned long long)g.blocked_gamma_unpublished,(unsigned long long)g.blocked_schema,(unsigned long long)g.blocked_sign,(unsigned long long)g.blocked_incomplete_adiabatic,(unsigned long long)g.te_manifest_mismatch,(unsigned long long)g.te_context_mismatch,(unsigned long long)g.fixed_te_attempts,(unsigned long long)g.seed_generation_attempts,(unsigned long long)g.line_radiative_owner_shells,(unsigned long long)g.line_collisional_escape_owner_shells,(unsigned long long)g.line_replaced_collisional_terms,(unsigned long long)g.line_replaced_radiative_terms,(unsigned long long)g.line_owner_overlap_shells,(unsigned long long)g.line_owner_closure_failures,(unsigned long long)g.diagnostic_seed_trials,(unsigned long long)g.diagnostic_requested_te_trials,(unsigned long long)g.pin_attempts,(unsigned long long)g.floor_attempts,(unsigned long long)g.neighbor_attempts,(unsigned long long)g.old_te_attempts,(unsigned long long)g.fallback_attempts,(unsigned long long)g.partial_publish_attempts,(unsigned long long)g.nonfinite_failures,g.max_line_owner_closure,a210_te_lane_name(g.te_lane),g.max_heat_residual);}
RadeqStatus a210_signed_tau_energy_preflight(
        const A208ValueView *values, size_t count,
        uint64_t *blocked_negative_heating, size_t *first_negative) {
    int rc = a208_capability_check(
        A208_BLOCK_UNSUPPORTED, values, count,
        "P09-A2-10-LINE-ENERGY", blocked_negative_heating,
        first_negative);
    if (rc == 0) return RADEQ_OK;
    if (rc == 3) return RADEQ_SIGN_MISMATCH;
    return RADEQ_TERM_SCHEMA;
}
