/* Stage 3.2 Rung 1 model-free writer fixture.  No GPU/model/solver run. */
#include "lumina_cmfgen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NS 9
#define NB 10
#define NL 8
#define EPS_FLOOR 0.2
#define EPS_CAP 0.7

static double fixture_eps0(int l,double n) {
    static const double by_line[NL]={0.05,0.09,0.14,0.22,0.31,0.07,0.45,0.37};
    double shell_scale=(n < 1e-6) ? 0.25 : (n < 1e-3) ? 1.0 : 2.0;
    return by_line[l]*shell_scale;
}
double radeq_line_eps_phys(int l,double n,double T,double t) {
    (void)T;
    double eps0=fixture_eps0(l,n),beta=-expm1(-t)/t;
    return eps0/(eps0+(1.0-eps0)*beta);
}
int radeq_line_local_response(int line,double ne,double Te,double tau,
                              double *beta,double *eps0) {
    (void)line;(void)Te;
    if (!beta || !eps0 || !(tau > 0.0) || !(ne >= 0.0) || !(ne <= 1.0)) return -1;
    *beta=-expm1(-tau)/tau;
    if (getenv("S32_SEED_BETA_DEFECT")) *beta*=0.5;
    *eps0=fixture_eps0(line,ne);
    return (isfinite(*beta) && *beta > 0.0 && *beta <= 1.0) ? 0 : -1;
}

#define EV_REACHED 1U
#define EV_ELIGIBLE 2U
#define EV_THICK 4U
#define EV_EPAY2 8U
#define EV_ACCW 16U
#define EV_HOT 32U
#define EV_BRANCH 64U

int main(int argc,char **argv) {
    if (argc != 2) { fprintf(stderr,"usage: %s OUTPUT_BASE\n",argv[0]); return 2; }
    const double C=2.99792458e10, texp=1683072.0;
    const double wave[NL]={3050.0,3000.0,2000.0,1990.0,1500.0,1000.0,600.0,590.0};
    const double tau_axis[NL]={1e-3,1e-16,1e-6,1e-3,1.0,1e2,2e16,1e4};
    const double eps_axis[NS]={1e-8,3e-8,1e-6,1e-5,1e-4,1e-3,3e-3,1e-2,3e-2};
    double nu_min=C/(3300e-8),nu_max=C/(550e-8);
    double dl=log(nu_max/nu_min)/NB,nu[NB],dnu[NB];
    for(int b=0;b<NB;b++){nu[b]=nu_min*exp(((double)b+0.5)*dl);dnu[b]=nu[b]*dl;}
    double line_nu[NL],tau[NL*NS],line_source[NL*NS];
    double chi[NS*NB],chi_es[NS*NB],chi_tot[NS*NB],lambda_star[NS*NB];
    double eta[NS*NB],boundary[NS*NB],line_eta[NL*NS];
    int slot[NL],nsel=0;
    unsigned char disp[NS*NB],evidence[NS*NB];
    memset(chi,0,sizeof chi);memset(eta,0,sizeof eta);
    memset(boundary,0,sizeof boundary);memset(line_eta,0,sizeof line_eta);
    double win_lo=C/(3000e-8),win_hi=C/(600e-8);
    int blo=(int)floor(log(win_lo/nu_min)/dl),bhi=(int)floor(log(win_hi/nu_min)/dl);
    for(int l=0;l<NL;l++){
        line_nu[l]=C/(wave[l]*1e-8);
        slot[l]=(line_nu[l]>=win_lo && line_nu[l]<=win_hi)?nsel++:-1;
    }
    int thin_defect=getenv("S32_SEED_THIN_NUMERATOR_DEFECT")!=NULL;
    int eps_phys=!getenv("S32_FIXTURE_EPS_PHYS") ||
                 atoi(getenv("S32_FIXTURE_EPS_PHYS"))!=0;
    int row_unscaled=getenv("S32_SEED_ROW_UNSCALED_DEFECT")!=NULL;
    int auth_unscaled=getenv("S32_SEED_AUTHORITATIVE_UNSCALED_DEFECT")!=NULL;
    int both_unscaled=getenv("S32_SEED_BOTH_UNSCALED_DEFECT")!=NULL;
    int floor_hits=0,interior_hits=0,cap_hits=0;
    for(int s=0;s<NS;s++) for(int l=0;l<NL;l++) {
        double t=tau_axis[l];tau[l*NS+s]=t;
        double S=(1.0+l)*(1.0+s)*1e-7;line_source[l*NS+s]=S;
        if (t<=1e-12) continue; /* production assembly predicate, not row predicate */
        int b=(int)floor(log(line_nu[l]/nu_min)/dl);
        double frac=(t>1e-6 || thin_defect)?-expm1(-t):t;
        double w=frac*line_nu[l]/(C*texp*dnu[b]);
        double eta_l=w*S;
        if(eps_phys) {
            double el=radeq_line_eps_phys(l,eps_axis[s],1e4,t);
            if(el < EPS_FLOOR) { el=EPS_FLOOR;floor_hits++; }
            else if(el > EPS_CAP) { el=EPS_CAP;cap_hits++; }
            else interior_hits++;
            eta_l=w*el*S;
        }
        double row_eta=(row_unscaled || both_unscaled)?w*S:eta_l;
        double auth_eta=(auth_unscaled || both_unscaled)?w*S:eta_l;
        chi[s*NB+b]+=w;eta[s*NB+b]+=auth_eta;
        if(slot[l]>=0) line_eta[slot[l]*NS+s]+=row_eta;
        else if(b==blo || b==bhi) boundary[s*NB+b]+=row_eta;
    }
    for(int s=0;s<NS;s++) for(int b=0;b<NB;b++) {
        size_t q=(size_t)s*NB+b;
        if(b==6) { disp[q]=3;evidence[q]=EV_REACHED|EV_ELIGIBLE|EV_EPAY2|
                                      EV_HOT|EV_BRANCH; }
        else switch(b%4) {
        case 0: disp[q]=0;evidence[q]=EV_REACHED|EV_BRANCH;break;
        case 1: disp[q]=1;evidence[q]=EV_REACHED|EV_ELIGIBLE|EV_THICK|
                                    EV_EPAY2|EV_ACCW|EV_HOT|EV_BRANCH;break;
        case 2: disp[q]=2;evidence[q]=EV_REACHED|EV_ELIGIBLE|EV_EPAY2|
                                    EV_ACCW|EV_HOT|EV_BRANCH;break;
        default:disp[q]=3;evidence[q]=EV_REACHED|EV_ELIGIBLE|EV_EPAY2|
                                    EV_HOT|EV_BRANCH;break; /* acc_w==0 */
        }
        if(getenv("S32_SEED_DISPOSITION_DEFECT") && disp[q]==3) disp[q]=2;
    }
    if(getenv("S32_SEED_OPACITY_SHARE_DEFECT")) {
        for(int s=0;s<NS;s++) for(int l=0;l<NL;l++) if(slot[l]>=0 && tau[l*NS+s]>1e-12) {
            int b=(int)floor(log(line_nu[l]/nu_min)/dl);
            double t=tau[l*NS+s];double frac=(t>1e-6)?-expm1(-t):t;
            double w=frac*line_nu[l]/(C*texp*dnu[b]);
            line_eta[slot[l]*NS+s]=eta[s*NB+b]*(w/chi[s*NB+b]);
        }
    }
    for(int s=0;s<NS;s++) for(int b=0;b<NB;b++) {
        size_t q=(size_t)s*NB+b;
        double ratio=0.94+0.01*(double)(b%5);
        chi_tot[q]=1.0+chi[q];
        chi_es[q]=ratio*chi_tot[q];
        /* Cover the thick-cell limit explicitly: lambda_star == 1.0 is a valid
         * binary64 diagonal (escape underflows).  The absence of this case is
         * exactly why the half-open guard survived into v4 and killed run
         * 189065 at iter 10.  A fixture that cannot express the boundary cannot
         * defend it. */
        lambda_star[q]=(s==0 && b==0) ? 1.0 : 0.98-0.002*(double)(s%4);
    }
    if(getenv("S32_SEED_CHI_TOT_ZERO")) chi_tot[(size_t)8*NB+6]=0.0;
    double rin[NS],rout[NS],Te[NS],ne[NS];
    for(int s=0;s<NS;s++){rin[s]=1e14*(s+1);rout[s]=rin[s]+1e14;Te[s]=1e4;ne[s]=eps_axis[s];}
    Geometry geo;CMFGENState cs;OpacityState op;PlasmaState pl;
    memset(&geo,0,sizeof geo);memset(&cs,0,sizeof cs);memset(&op,0,sizeof op);memset(&pl,0,sizeof pl);
    geo.n_shells=NS;geo.time_explosion=texp;geo.r_inner=rin;geo.r_outer=rout;
    cs.n_shells=NS;cs.n_bins=NB;cs.nu_min=nu_min;cs.nu_max=nu_max;cs.d_log_nu=dl;
    cs.nu=nu;cs.dnu=dnu;cs.chi_line=chi;cs.chi_es=chi_es;cs.chi_tot=chi_tot;
    cs.lambda_star=lambda_star;cs.stage32_eta_pre_epay=eta;
    cs.stage32_boundary_eta=boundary;cs.stage32_line_eta=line_eta;
    cs.stage32_line_slot=slot;cs.stage32_line_slot_n=NL;cs.stage32_selected_lines=nsel;
    cs.stage32_source_nlte=1;cs.stage32_field_generation=37;
    cs.stage32_lambda_generation=getenv("S32_SEED_LAMBDA_GENERATION_DEFECT")?36:37;
    cs.stage32_epay_disposition=disp;cs.stage32_epay_evidence=evidence;
    op.n_shells=NS;op.n_lines=NL;op.line_list_nu=line_nu;op.tau_sobolev=tau;
    op.electron_density=ne;op.line_source_S=line_source;
    pl.T_e=Te;pl.n_electron=ne;
    const char *ie=getenv("S32_FIXTURE_ITER");int it=ie?atoi(ie):10;
    fprintf(stderr,"[fixture] eps_phys=%d floor_hits=%d interior_hits=%d cap_hits=%d\n",
            eps_phys,floor_hits,interior_hits,cap_hits);
    return cmfgen_dump_stage32_rung1(&cs,&geo,&op,&pl,it,argv[1])==0?0:1;
}
