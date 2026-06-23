/* B1: line-resolved CMF emergent P-Cygni (standalone, gcc).
 * Goal: the REAL CMFGEN method (full comoving-frame, frequency-coupled,
 * line-RESOLVED on a fine adaptive grid) — NOT Sobolev (that's S4/MC) — for the
 * R2 toy (single thick scattering Ca II line). Produce the emergent observer
 * spectrum and check it makes a P-Cygni matching the MC (trough ~0.62 @ 8178A,
 * blueshifted from rest 8542 by ~0.66*beta_phot).
 *
 * Kernel = the validated cmf_formal (lumina_cmf_selftest.c gates 2a/4a/2c):
 * tangent rays, each marched blue->red with PHOENIX/Hauschildt conservative
 * upwind frequency coupling (homologous advection a_lam=1/(t_exp c)). Here we
 * add (i) an ADAPTIVE wavelength grid that resolves the thermal line width,
 * (ii) PROFILE line opacity from tau_Sobolev (not the defanged expansion bin),
 * (iii) emergent I+(p) extraction -> L_lambda.
 *
 * Build: gcc -O2 -o cmf_pcygni_b1 src/cmf_pcygni_b1.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define C_CGS  2.99792458e10
#define H_CGS  6.62607015e-27
#define K_CGS  1.380649e-16

static double planck_nu(double nu, double T) {
    double x = H_CGS*nu/(K_CGS*T);
    if (x > 700.0) return 0.0;
    return (2.0*H_CGS*nu*nu*nu/(C_CGS*C_CGS))/(exp(x)-1.0);
}

/* ---- model ---- */
typedef struct {
    int NR, NF;
    double t_exp;
    double *r;        /* [NR] radii ascending (cm) */
    double *lam;      /* [NF] comoving wavelength ascending (cm) */
    double *chi;      /* [NR*NF] total opacity (cm^-1) */
    double *Ssrc;     /* [NR*NF] source S_nu (per-cm^2-Hz-sr units) */
    double Iin_core;  /* inner-boundary incoming intensity (photosphere) */
    int    *is_line;  /* [NF] 1 if this freq is inside a line core (scattering) */
} Model;

/* validated cmf_formal kernel + emergent extraction.
 * J[NR*NF] (mean intensity), Iemg[NP*NF] (emergent I+ per ray k at outer edge),
 * pP[NP] impact parameters. Returns NP. */
static int cmf_formal_emergent(const Model *m, double *J, double **Iemg_out,
                               double **pP_out)
{
    int NR=m->NR, NF=m->NF;
    double a_lam=1.0/(m->t_exp*C_CGS);
    int NCORE=24, NP=NR+NCORE;
    double *p=malloc(NP*sizeof(double));
    for (int k=0;k<NCORE;++k) p[k]=m->r[0]*k/(double)NCORE;
    for (int s=0;s<NR;++s) p[NCORE+s]=m->r[s];
    int *rn=calloc(NP,sizeof(int));
    int *rsh=malloc((size_t)NP*(NR+1)*sizeof(int));
    double *rz=malloc((size_t)NP*(NR+1)*sizeof(double));
    int *rcore=calloc(NP,sizeof(int)); double *rzin=calloc(NP,sizeof(double));
    for (int k=0;k<NP;++k){ double pk=p[k]; int n=0;
        for (int s=NR-1;s>=0;--s){ if (m->r[s]<=pk) break;
            rsh[(size_t)k*(NR+1)+n]=s; rz[(size_t)k*(NR+1)+n]=sqrt(m->r[s]*m->r[s]-pk*pk); ++n; }
        rn[k]=n; rcore[k]=(pk<m->r[0]); rzin[k]=rcore[k]?sqrt(m->r[0]*m->r[0]-pk*pk):0.0; }
    double *Iin_p=calloc((size_t)NP*(NR+1),sizeof(double)),*Iout_p=calloc((size_t)NP*(NR+1),sizeof(double));
    double *Iin_c=calloc((size_t)NP*(NR+1),sizeof(double)),*Iout_c=calloc((size_t)NP*(NR+1),sizeof(double));
    double *muL=malloc((size_t)NR*NP*sizeof(double)),*IpL=malloc((size_t)NR*NP*sizeof(double)),*ImL=malloc((size_t)NR*NP*sizeof(double));
    int *cnt=malloc(NR*sizeof(int));
    double *Iemg=calloc((size_t)NP*NF,sizeof(double));

    for (int l=0;l<NF;++l){
        double Dlam=(l>0)?(m->lam[l]-m->lam[l-1]):0, lam_l=m->lam[l], lam_b=(l>0)?m->lam[l-1]:m->lam[l];
        double adv=(l>0)?a_lam*(lam_l/Dlam):0.0;
        memset(cnt,0,NR*sizeof(int));
        for (int k=0;k<NP;++k){
            int n=rn[k]; if(n==0)continue; size_t kb=(size_t)k*(NR+1);
            double I=0.0;
            for (int i=0;i<n;++i){ int s=rsh[kb+i]; double ds=(i+1<n)?(rz[kb+i]-rz[kb+i+1]):(rz[kb+i]-rzin[k]);
                double chi=m->chi[(size_t)s*NF+l], chih=chi+a_lam*4.0+adv;
                double Su=(m->Ssrc[(size_t)s*NF+l]*chi+((l>0)?a_lam*(lam_b/Dlam)*Iin_p[kb+i]:0.0))/(chih>0?chih:1);
                double dtau=chih*ds, ex=exp(-dtau), e0,e1,wu,wd;
                if(dtau>1e-4){e0=1-ex;e1=dtau-e0;wu=e0-e1/dtau;wd=e1/dtau;}else{wu=0.5*dtau;wd=0.5*dtau;}
                I=I*ex+wu*Su+wd*Su; Iin_c[kb+i]=I;
                int c=cnt[s]; muL[(size_t)s*NP+c]=rz[kb+i]/m->r[s]; ImL[(size_t)s*NP+c]=I; }
            if(rcore[k]) I=m->Iin_core;
            for (int i=n-1;i>=0;--i){ int s=rsh[kb+i]; double ds=(i+1<n)?(rz[kb+i]-rz[kb+i+1]):(rz[kb+i]-rzin[k]);
                double chi=m->chi[(size_t)s*NF+l], chih=chi+a_lam*4.0+adv;
                double Su=(m->Ssrc[(size_t)s*NF+l]*chi+((l>0)?a_lam*(lam_b/Dlam)*Iout_p[kb+i]:0.0))/(chih>0?chih:1);
                double dtau=chih*ds, ex=exp(-dtau), e0,e1,wu,wd;
                if(dtau>1e-4){e0=1-ex;e1=dtau-e0;wu=e0-e1/dtau;wd=e1/dtau;}else{wu=0.5*dtau;wd=0.5*dtau;}
                I=I*ex+wu*Su+wd*Su; Iout_c[kb+i]=I;
                int c=cnt[s]; IpL[(size_t)s*NP+c]=I; cnt[s]=c+1; }
            Iemg[(size_t)k*NF+l] = I;     /* emergent I+ at outer edge of ray k */
        }
        for (int s=0;s<NR;++s){ int c=cnt[s]; if(c<1){J[(size_t)s*NF+l]=m->Ssrc[(size_t)s*NF+l];continue;}
            for(int a=1;a<c;++a){double mk=muL[(size_t)s*NP+a],ip=IpL[(size_t)s*NP+a],im=ImL[(size_t)s*NP+a];int b=a-1;
                while(b>=0&&muL[(size_t)s*NP+b]>mk){muL[(size_t)s*NP+b+1]=muL[(size_t)s*NP+b];IpL[(size_t)s*NP+b+1]=IpL[(size_t)s*NP+b];ImL[(size_t)s*NP+b+1]=ImL[(size_t)s*NP+b];--b;}
                muL[(size_t)s*NP+b+1]=mk;IpL[(size_t)s*NP+b+1]=ip;ImL[(size_t)s*NP+b+1]=im;}
            double mu[600],jv[600];int q=0; mu[q]=0;jv[q]=0.5*(IpL[(size_t)s*NP+0]+ImL[(size_t)s*NP+0]);q++;
            for(int a=0;a<c;++a){mu[q]=muL[(size_t)s*NP+a];jv[q]=0.5*(IpL[(size_t)s*NP+a]+ImL[(size_t)s*NP+a]);q++;}
            double Js=0; for(int a=0;a+1<q;++a)Js+=0.5*(jv[a]+jv[a+1])*(mu[a+1]-mu[a]); J[(size_t)s*NF+l]=Js; }
        double *t1=Iin_p;Iin_p=Iin_c;Iin_c=t1; double *t2=Iout_p;Iout_p=Iout_c;Iout_c=t2;
    }
    *Iemg_out = Iemg; *pP_out = p;
    free(rn);free(rsh);free(rz);free(rcore);free(rzin);
    free(Iin_p);free(Iout_p);free(Iin_c);free(Iout_c);free(muL);free(IpL);free(ImL);free(cnt);
    return NP;
}

int main(void)
{
    /* ---- R2 conditions ---- */
    double t_exp = 84326.4;
    double T_inner = 4430.0, T_e = 4430.0;
    double r_in = 1.6e14;
    double beta_phot = r_in/(C_CGS*t_exp);          /* ~0.0633 */
    double W = 0.5;                                  /* dilution -> scatter source */
    /* Ca II NIR triplet (rest A) + Sobolev tau (R2 dump: tau_max~8.5e6) */
    double line_lam0[3] = {8498e-8, 8542e-8, 8662e-8};
    double line_tauS[3] = {8.5e6,   8.5e6,   8.5e6};
    int NL3 = 3;
    /* thermal width of Ca (mass 40 amu) at T_e */
    double v_th = sqrt(2.0*K_CGS*T_e/(40.0*1.66054e-24));   /* ~1.36e5 cm/s */

    /* ---- radial grid: the ejecta from r_in outward (resolve velocity field) ---- */
    int NR = 60;
    double r_out = r_in * (1.0 + 8.0*beta_phot);     /* extend to ~+8*v_phot so the
                                                        disk projection + the radial
                                                        line region spans the P-Cygni */
    double *r = malloc(NR*sizeof(double));
    for (int s=0;s<NR;++s) r[s] = r_in + (r_out-r_in)*s/(double)(NR-1);

    /* ---- adaptive wavelength grid: fine near each line core (thermal width),
     * coarse continuum. Cover 7600-9300 A (P-Cygni span). ---- */
    double lam_min=7600e-8, lam_max=9300e-8;
    /* build sorted list: coarse base + fine windows around each (blueshifted) line */
    int CAP=200000; double *lamv=malloc(CAP*sizeof(double)); int NF=0;
    double dl_coarse = 4e-8;                          /* 4 A continuum step */
    for (double L=lam_min; L<=lam_max; L+=dl_coarse) lamv[NF++]=L;
    /* fine windows: each line, from rest down to rest*(1-beta) (blueshift range),
     * resolve thermal width */
    double dl_fine = 0.3*v_th/C_CGS*8542e-8;          /* ~0.3 thermal widths */
    for (int j=0;j<NL3;++j){
        double l0=line_lam0[j];
        double lo=l0*(1.0-1.2*beta_phot), hi=l0*(1.0+0.3*beta_phot);
        for (double L=lo; L<=hi; L+=dl_fine){ if(NF<CAP) lamv[NF++]=L; }
    }
    /* sort ascending + dedup */
    for (int a=1;a<NF;++a){ double x=lamv[a]; int b=a-1; while(b>=0&&lamv[b]>x){lamv[b+1]=lamv[b];--b;} lamv[b+1]=x; }
    double *lam=malloc(NF*sizeof(double)); int NFu=0;
    for (int a=0;a<NF;++a){ if(a==0||lamv[a]-lamv[a-1]>1e-12*lamv[a]) lam[NFu++]=lamv[a]; }
    NF=NFu;
    printf("[B1] NR=%d NF=%d (adaptive), beta_phot=%.4f, v_th=%.3e cm/s, dl_fine=%.4f A\n",
           NR, NF, beta_phot, v_th, dl_fine*1e8);

    /* ---- opacity + source ---- */
    double sigT=6.6524587e-25;
    double n_e = 1.0e9;                               /* R2 fixed n_e */
    double chi_es = n_e*sigT;                         /* grey electron scatter */
    double *chi  = malloc((size_t)NR*NF*sizeof(double));
    double *Ssrc = malloc((size_t)NR*NF*sizeof(double));
    double *Jbar = malloc((size_t)NR*NF*sizeof(double));   /* mean intensity */
    int *is_line = calloc(NF,sizeof(int));
    /* line peak opacity from tau_Sobolev:
     *   tau_S = (pi e^2/m c) f n_l * (lambda * t_exp)  [Sobolev length=lambda*t_exp]
     *   integral chi_line dlambda = (pi e^2/m c) f n_l * lambda^2/c  (line area)
     * so chi_line(lambda) = [tau_S/(t_exp*lambda)] * phi(lambda-lambda0),
     * phi normalised Gaussian of std = lambda0*v_th/c. */
    for (int s=0;s<NR;++s) for (int l=0;l<NF;++l){
        double cl = chi_es;                          /* continuum (electron scatter) */
        for (int j=0;j<NL3;++j){
            double l0=line_lam0[j], sig=l0*v_th/C_CGS;
            double x=(lam[l]-l0)/sig;
            if (fabs(x)<8.0){
                double phi=exp(-0.5*x*x)/(sig*sqrt(2.0*M_PI));
                cl += (line_tauS[j]/(t_exp*l0))*phi;
                if (s==0 && fabs(x)<3.0) is_line[l]=1;
            }
        }
        chi[(size_t)s*NF+l]=cl;
    }
    double Bcont = planck_nu(C_CGS/(8542e-8), T_e);
    /* init Jbar = W*B (diluted), source = scattering */
    for (size_t i=0;i<(size_t)NR*NF;++i) Jbar[i]=W*Bcont;

    /* ---- self-consistent scattering ALI (S = (chi_es*J + chi_line*J)/chi = J for
     * pure scatter; here both continuum e-scatter and the line scatter -> S=Jbar) ---- */
    double *J=malloc((size_t)NR*NF*sizeof(double));
    double *Iemg=NULL,*pP=NULL; int NP=0;
    Model m={NR,NF,t_exp,r,lam,chi,Ssrc,planck_nu(C_CGS/(8542e-8),T_inner),is_line};
    for (int it=0; it<40; ++it){
        for (int s=0;s<NR;++s) for (int l=0;l<NF;++l){
            /* pure-scatter source: S = Jbar (electron + resonance line both scatter) */
            Ssrc[(size_t)s*NF+l]=Jbar[(size_t)s*NF+l];
        }
        if (Iemg){ free(Iemg); free(pP); Iemg=NULL; }
        NP=cmf_formal_emergent(&m,J,&Iemg,&pP);
        double md=0;
        for (size_t i=0;i<(size_t)NR*NF;++i){
            double d=fabs(J[i]-Jbar[i])/(fabs(Jbar[i])+1e-30); if(d>md)md=d; Jbar[i]=J[i];
        }
        if (it>2 && md<1e-3) { printf("[B1] ALI converged it=%d maxrel=%.2e\n",it,md); break; }
    }

    /* ---- emergent observer spectrum: L_lambda = 8 pi^2 int I+(p,l) p dp ----
     * (comoving lam ~ observer here; the advection already carried the Doppler) */
    FILE *fp=fopen("logs/toy/R2/cmf_lineres_emergent.csv","w");
    fprintf(fp,"wavelength_angstrom,flux\n");
    for (int l=0;l<NF;++l){
        double integ=0, pprev=0, fprev=0;
        for (int k=0;k<NP;++k){
            double f=Iemg[(size_t)k*NF+l]*pP[k];
            integ += 0.5*(fprev+f)*(pP[k]-pprev); pprev=pP[k]; fprev=f;
        }
        double Lnu = 8.0*M_PI*M_PI*integ;
        double lam_cm=lam[l], lam_A=lam_cm*1e8;
        double L_lam=Lnu*C_CGS/(lam_cm*lam_cm)*1e-8;
        fprintf(fp,"%.6f,%.6e\n", lam_A, L_lam);
    }
    fclose(fp);
    printf("[B1] emergent -> logs/toy/R2/cmf_lineres_emergent.csv (%d lambda)\n", NF);
    return 0;
}
