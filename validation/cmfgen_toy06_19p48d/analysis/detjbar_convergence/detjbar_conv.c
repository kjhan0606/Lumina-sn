/* detjbar_conv.c -- OFFLINE CPU A/B harness for the fine (line-resolved) CMF
 * solver's ALI/lag convergence, and its effect on jbar_line_det.
 *
 * NO GPU, NO production run.  Both solver paths are re-implemented here
 * VERBATIM from the production sources so the A/B is apples-to-apples:
 *
 *   REF  ("sequential-frequency")  == src/lumina_cmfgen.c  cmf_solve_J(), the
 *        run_cpu branch: the blue->red advection upstream Iin_p/Iout_p is the
 *        CURRENT sweep's value at bin b+1 (the b-loop is sequential), so the
 *        frequency coupling is resolved EXACTLY inside every ALI iteration and
 *        the only iteration is the ALI scattering source.
 *
 *   LAG  ("lagged-advection")      == src/lumina_cmf_solve.cu k_march/k_jint/
 *        k_update, i.e. what LUMINA_CMF_SOLVE_GPU=1 actually runs: bin b reads
 *        bin (b+1)'s per-segment intensities from the PREVIOUS ALI iteration
 *        (lag_in/lag_out), so frequency information advances 1 bin per iter.
 *        Same fixed point, different iteration path.  Early stop maxrel<tol &&
 *        it>0 with tol=1e-4 hardcoded (cmfgen.c:1657 / cmf_solve.cu:323).
 *
 * production (parity42, scripts/run_coevolve_s01.sh + launch_parity42_gphoff.sh):
 *   LUMINA_CMF_SOLVE_GPU=1, LUMINA_CMF_ADV_SPLIT=1, LUMINA_CMF_FINE_ALI=20000,
 *   LUMINA_CMF_ALAM unset -> a_lam = 1/(t_exp*c) (advection ON)  ==> LAG path.
 *   window 1000-4000 A, vdop=1e6 cm/s, ppd=12  -> NF = 498721 bins, NS=50.
 *
 *   EXACT ("drifting characteristic")  == the DISCRETISATION reference for the
 *        frequency-advection operator.  REF/LAG share a fixed point, but that
 *        fixed point is the fixed point of a FIRST-ORDER OPERATOR-SPLIT UPWIND
 *        step whose per-segment frequency Courant number is beta = adv*ds ~ 874
 *        bins at production resolution.  Written out, one REF/LAG segment is
 *            I_i,b = SUM_j w_j [ T_{b+j} I_{i-1,b+j} + W_{b+j} S_{b+j} ],
 *            w_j = beta^j/(1+beta)^{j+1},  T_m = exp(-chi_m ds),
 *        i.e. the contribution arriving at bin b from bin b+j is attenuated ONLY
 *        by bin b+j's own opacity and NOT by the j intervening bins the real
 *        characteristic must cross.  EXACT integrates the same piecewise-constant
 *        (shell,bin) chi and S along the true characteristic ln nu = -a_lam*z,
 *        so the segment transmission is exp(-INT chi dz) instead of the mixture
 *        <exp(-chi ds)>.  Same geometry, same angle quadrature, same jbar
 *        extraction; the ONLY difference is the (z,nu) coupling.
 *        --exact 1 = O(1)/bin sliding-window form (production-affordable)
 *        --exact 2 = O(beta)/bin direct cell march (independent reference)
 *        The two are algebraically identical; 1-vs-2 is the harness self-check.
 *
 * SIMILARITY SCALING (see REPORT.md).  Reduced problem with r = NB/498721:
 *   dlognu -> dlognu_prod/r,  vdop -> vdop_prod/r  (ppd=12 held),
 *   n_ali  -> r*20000,        line list subsampled by r,
 *   geometry / n_e / T_e / tau_sobolev UNCHANGED.
 * This holds invariant every dimensionless group that controls the lag
 * convergence:  ppd,  D/NB (total drift in bins / grid size),  n_ali/D,
 * and kappa = n_ali*chi/adv (optical depth reached per n_ali bins of drift).
 *
 * build:  gcc -O3 -march=native -fopenmp -std=gnu11 -o detjbar_conv \
 *              detjbar_conv.c -lm
 * run:    ./detjbar_conv --nb 24936 --nali 1000 --fabs 0.3 --out tag
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CM_C      2.99792458e10
#define CM_H      6.62606957e-27
#define CM_KB     1.3806488e-16
#define SIGMA_T   6.652458734e-25
#define SQRTPI    1.7724538509055159

/* production constants (data/tardis_reference_toy06_19p48d_sivcaiv/config.json) */
#define T_EXP     1683072.0
#define V_IN      3.90e8
#define V_OUT     4.03e9
#define T_INNER   10020.0
#define NB_PROD   498721       /* NF at vdop=1e6, ppd=12, 1000-4000 A */
#define NALI_PROD 20000        /* LUMINA_CMF_FINE_ALI */
#define VDOP_PROD 1.0e6
#define PPD       12.0
#define LAM_LO    1000.0
#define LAM_HI    4000.0

static double planck(double nu, double T) {
    if (T <= 0.0) return 0.0;
    double x = CM_H * nu / (CM_KB * T);
    if (x > 700.0) return 0.0;
    double e = expm1(x);
    if (e <= 0.0) return 0.0;
    return 2.0 * CM_H * nu * nu * nu / (CM_C * CM_C) / e;
}

/* ------------------------------------------------------------------ */
/* problem                                                             */
/* ------------------------------------------------------------------ */
typedef struct {
    int NS, NB, NP, NCORE;
    double dlognu, nu_lo, nu_hi, vdop, a_lam;
    double *nu, *lam, *rmid, *Te, *ne;
    double *chi_es, *chi_abs, *chi_line, *chi_tot, *S_fixed, *J0;  /* [NS*NB] */
    double *adv;                                                    /* [NB]    */
    /* ray tables (verbatim cmf_solve_J) */
    double *p; int *rn, *rsh; double *rz; int *rcore; double *rzin;
    int *shell_off, *shell_k, *shell_seg; double *shell_mu; int nsamp;
    /* line bookkeeping for the jbar extraction */
    int    n_line;
    double *l_nu;      /* [n_line] */
    double *l_tau;     /* [n_line*NS] */
    long   *l_id;      /* [n_line] index in the lines file (run-invariant key) */
} Prob;

/* per-(b,k,seg) exp(-dtau_rad) table; dtau on the fly is cheap, exp is not */
static double *EXT = NULL;      /* [NB*NP*(NS+1)] */

static size_t SEG1, SEGSZ;

static void build_rays(Prob *P) {
    int NS = P->NS, NCORE = P->NCORE, NP = P->NP;
    P->p    = malloc(NP * sizeof(double));
    P->rn   = calloc(NP, sizeof(int));
    P->rsh  = malloc((size_t)NP * (NS + 1) * sizeof(int));
    P->rz   = malloc((size_t)NP * (NS + 1) * sizeof(double));
    P->rcore= calloc(NP, sizeof(int));
    P->rzin = calloc(NP, sizeof(double));
    double *rmid = P->rmid;
    for (int k = 0; k < NCORE; ++k) P->p[k] = rmid[0] * k / (double)NCORE;
    for (int s = 0; s < NS; ++s) P->p[NCORE + s] = rmid[s];
    for (int k = 0; k < NP; ++k) {
        double pk = P->p[k]; int n = 0;
        for (int s = NS - 1; s >= 0; --s) {
            if (rmid[s] <= pk) break;
            P->rsh[(size_t)k*(NS+1)+n] = s;
            P->rz [(size_t)k*(NS+1)+n] = sqrt(rmid[s]*rmid[s] - pk*pk);
            ++n;
        }
        P->rn[k] = n; P->rcore[k] = (pk < rmid[0]);
        P->rzin[k] = P->rcore[k] ? sqrt(rmid[0]*rmid[0] - pk*pk) : 0.0;
    }
    /* per-shell mu-sorted sample tables (as the GPU aux build in cmf_solve_J) */
    int *sc = calloc(NS, sizeof(int));
    for (int k = 0; k < NP; ++k) { size_t kb=(size_t)k*(NS+1);
        for (int i = 0; i < P->rn[k]; ++i) sc[P->rsh[kb+i]]++; }
    P->shell_off = malloc((NS+1)*sizeof(int)); P->shell_off[0] = 0;
    for (int s = 0; s < NS; ++s) P->shell_off[s+1] = P->shell_off[s] + sc[s];
    P->nsamp = P->shell_off[NS];
    P->shell_k   = malloc((size_t)P->nsamp*sizeof(int));
    P->shell_seg = malloc((size_t)P->nsamp*sizeof(int));
    P->shell_mu  = malloc((size_t)P->nsamp*sizeof(double));
    int *fl = calloc(NS, sizeof(int));
    for (int k = 0; k < NP; ++k) { size_t kb=(size_t)k*(NS+1);
        for (int i = 0; i < P->rn[k]; ++i) { int s = P->rsh[kb+i];
            int pos = P->shell_off[s] + fl[s]++;
            P->shell_k[pos]=k; P->shell_seg[pos]=i;
            P->shell_mu[pos]=P->rz[kb+i]/rmid[s]; } }
    for (int s = 0; s < NS; ++s) { int o0=P->shell_off[s], o1=P->shell_off[s+1];
        for (int a=o0+1;a<o1;++a){ double mk=P->shell_mu[a]; int kk=P->shell_k[a], ss=P->shell_seg[a]; int q=a-1;
            while(q>=o0 && P->shell_mu[q]>mk){ P->shell_mu[q+1]=P->shell_mu[q];
                P->shell_k[q+1]=P->shell_k[q]; P->shell_seg[q+1]=P->shell_seg[q]; --q; }
            P->shell_mu[q+1]=mk; P->shell_k[q+1]=kk; P->shell_seg[q+1]=ss; } }
    free(sc); free(fl);
}

static void build_ext(Prob *P) {
    int NS=P->NS, NB=P->NB, NP=P->NP;
    EXT = malloc(SEGSZ * sizeof(double));
    if (!EXT) { fprintf(stderr,"EXT alloc %.1f GB failed\n", SEGSZ*8.0/1e9); exit(1); }
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < NB; ++b) {
        for (int k = 0; k < NP; ++k) {
            int n = P->rn[k]; size_t kb=(size_t)k*(NS+1);
            size_t s0 = (size_t)b*NP*(NS+1) + kb;
            for (int i = 0; i < n; ++i) {
                int s = P->rsh[kb+i];
                double ds = (i+1<n) ? (P->rz[kb+i]-P->rz[kb+i+1]) : (P->rz[kb+i]-P->rzin[k]);
                double chi_rad = P->chi_tot[(size_t)s*NB+b] + P->a_lam*4.0;
                EXT[s0+i] = exp(-chi_rad*ds);
            }
        }
    }
}

/* ================================================================== */
/* REF: sequential-frequency sweep  (cmf_solve_J run_cpu branch,       */
/*      adv_split=1)                                                   */
/* ================================================================== */
static int solve_ref(Prob *P, double *J, int maxit, double tol, double *hist)
{
    int NS=P->NS, NB=P->NB, NP=P->NP;
    size_t seg1 = (size_t)NP*(NS+1);
    double *Iin_p=calloc(seg1,sizeof(double)), *Iout_p=calloc(seg1,sizeof(double));
    double *Iin_c=calloc(seg1,sizeof(double)), *Iout_c=calloc(seg1,sizeof(double));
    double *muL=malloc(seg1*sizeof(double)), *IpL=malloc(seg1*sizeof(double)), *ImL=malloc(seg1*sizeof(double));
    int *cnt=malloc(NS*sizeof(int));
    double *S=malloc((size_t)NS*NB*sizeof(double));
    double *Jnew=malloc((size_t)NS*NB*sizeof(double));
    int used = maxit;
    for (int it = 0; it < maxit; ++it) {
        for (int s=0;s<NS;++s) for (int b=0;b<NB;++b) {
            size_t idx=(size_t)s*NB+b;
            double r = (P->chi_tot[idx]>0.0)? P->chi_es[idx]/P->chi_tot[idx] : 0.0;
            S[idx] = P->S_fixed[idx] + r*J[idx];
        }
        memset(Iin_p,0,seg1*sizeof(double)); memset(Iout_p,0,seg1*sizeof(double));
        for (int b = NB-1; b >= 0; --b) {
            double adv = P->adv[b];
            double Bin = planck(P->nu[b], T_INNER);
            memset(cnt,0,NS*sizeof(int));
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic,1)
            #endif
            for (int k = 0; k < NP; ++k) {
                int n=P->rn[k]; if(!n) continue; size_t kb=(size_t)k*(NS+1);
                size_t s0=(size_t)b*NP*(NS+1)+kb;
                int cloc[512]; double I=0.0;
                for (int i=0;i<n;++i) {
                    int s=P->rsh[kb+i];
                    double ds=(i+1<n)?(P->rz[kb+i]-P->rz[kb+i+1]):(P->rz[kb+i]-P->rzin[k]);
                    size_t idx=(size_t)s*NB+b;
                    double chi_rad=P->chi_tot[idx]+P->a_lam*4.0, Su=S[idx];
                    double dtau=chi_rad*ds, ex=EXT[s0+i], wsum;
                    if(dtau>1e-4) wsum=1.0-ex; else wsum=dtau;
                    double I_rad=I*ex+wsum*Su;
                    double bds=adv*ds;
                    double Iblue=(b<NB-1)?Iin_p[kb+i]:I_rad;
                    I=(I_rad+bds*Iblue)/(1.0+bds); Iin_c[kb+i]=I;
                    int c;
                    #ifdef _OPENMP
                    #pragma omp atomic capture
                    #endif
                    { c=cnt[s]; cnt[s]+=1; }
                    cloc[i]=c; muL[(size_t)s*NP+c]=P->rz[kb+i]/P->rmid[s]; ImL[(size_t)s*NP+c]=I;
                }
                if (P->rcore[k]) I=Bin;
                for (int i=n-1;i>=0;--i) {
                    int s=P->rsh[kb+i];
                    double ds=(i+1<n)?(P->rz[kb+i]-P->rz[kb+i+1]):(P->rz[kb+i]-P->rzin[k]);
                    size_t idx=(size_t)s*NB+b;
                    double chi_rad=P->chi_tot[idx]+P->a_lam*4.0, Su=S[idx];
                    double dtau=chi_rad*ds, ex=EXT[s0+i], wsum;
                    if(dtau>1e-4) wsum=1.0-ex; else wsum=dtau;
                    double I_rad=I*ex+wsum*Su;
                    double bds=adv*ds;
                    double Iblue=(b<NB-1)?Iout_p[kb+i]:I_rad;
                    I=(I_rad+bds*Iblue)/(1.0+bds); Iout_c[kb+i]=I;
                    IpL[(size_t)s*NP+cloc[i]]=I;
                }
            }
            for (int s=0;s<NS;++s) {
                int c=cnt[s]; size_t idx=(size_t)s*NB+b;
                if (c<1) { Jnew[idx]=S[idx]; continue; }
                for (int a=1;a<c;++a){ double mk=muL[(size_t)s*NP+a],ip=IpL[(size_t)s*NP+a],im=ImL[(size_t)s*NP+a]; int q=a-1;
                    while(q>=0&&muL[(size_t)s*NP+q]>mk){muL[(size_t)s*NP+q+1]=muL[(size_t)s*NP+q];
                        IpL[(size_t)s*NP+q+1]=IpL[(size_t)s*NP+q];ImL[(size_t)s*NP+q+1]=ImL[(size_t)s*NP+q];--q;}
                    muL[(size_t)s*NP+q+1]=mk;IpL[(size_t)s*NP+q+1]=ip;ImL[(size_t)s*NP+q+1]=im; }
                double mu[512],jv[512]; int q=0;
                mu[q]=0; jv[q]=0.5*(IpL[(size_t)s*NP+0]+ImL[(size_t)s*NP+0]); q++;
                for(int a=0;a<c;++a){mu[q]=muL[(size_t)s*NP+a];jv[q]=0.5*(IpL[(size_t)s*NP+a]+ImL[(size_t)s*NP+a]);q++;}
                double Js=0; for(int a=0;a+1<q;++a) Js+=0.5*(jv[a]+jv[a+1])*(mu[a+1]-mu[a]);
                Jnew[idx]=Js;
            }
            { double *t1=Iin_p;Iin_p=Iin_c;Iin_c=t1; double *t2=Iout_p;Iout_p=Iout_c;Iout_c=t2; }
        }
        double maxrel=0.0;
        for (size_t i=0;i<(size_t)NS*NB;++i){
            double d=fabs(Jnew[i]-J[i])/(fabs(J[i])+1e-30); if(d>maxrel)maxrel=d; J[i]=Jnew[i]; }
        if (hist) hist[it]=maxrel;
        if (maxrel<tol && it>0) { used=it+1; break; }
    }
    free(Iin_p);free(Iout_p);free(Iin_c);free(Iout_c);free(muL);free(IpL);free(ImL);
    free(cnt);free(S);free(Jnew);
    return used;
}

/* ================================================================== */
/* LAG: lagged-advection scheme  (cmf_solve_J_gpu, adv_split=1)        */
/* ================================================================== */
static int solve_lag(Prob *P, double *J, int n_ali, double tol, double *hist)
{
    int NS=P->NS, NB=P->NB, NP=P->NP;
    double *in0=calloc(SEGSZ,sizeof(double)), *in1=calloc(SEGSZ,sizeof(double));
    double *out0=calloc(SEGSZ,sizeof(double)), *out1=calloc(SEGSZ,sizeof(double));
    if(!in0||!in1||!out0||!out1){fprintf(stderr,"lag buffers %.1f GB failed\n",4.0*SEGSZ*8/1e9);exit(1);}
    double *Jnew=malloc((size_t)NS*NB*sizeof(double));
    double *lag_in=in0,*cur_in=in1,*lag_out=out0,*cur_out=out1;
    int used=n_ali;
    for (int it = 0; it < n_ali; ++it) {
        /* ---- k_march: thread (b,k) ---- */
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < NB; ++b) {
            int has_blue = (b < NB-1);
            double adv = has_blue ? P->adv[b] : 0.0;
            double Bin = planck(P->nu[b], T_INNER);
            for (int k = 0; k < NP; ++k) {
                int n=P->rn[k]; if(!n) continue;
                size_t kb=(size_t)k*(NS+1);
                size_t seg0=(size_t)b*NP*(NS+1)+kb;
                size_t blue0=has_blue?((size_t)(b+1)*NP*(NS+1)+kb):0;
                double I=0.0;
                for (int i=0;i<n;++i) {
                    int s=P->rsh[kb+i];
                    double ds=(i+1<n)?(P->rz[kb+i]-P->rz[kb+i+1]):(P->rz[kb+i]-P->rzin[k]);
                    size_t idx=(size_t)s*NB+b;
                    double chi=P->chi_tot[idx];
                    double rr=(chi>0.0)?P->chi_es[idx]/chi:0.0;
                    double Sv=P->S_fixed[idx]+rr*J[idx];
                    double chi_rad=chi+P->a_lam*4.0;
                    double dtau=chi_rad*ds, ex=EXT[seg0+i], wsum;
                    if(dtau>1e-4) wsum=1.0-ex; else wsum=dtau;
                    double I_rad=I*ex+wsum*Sv;
                    double bds=adv*ds;
                    double Iblue=has_blue?lag_in[blue0+i]:I_rad;
                    I=(I_rad+bds*Iblue)/(1.0+bds);
                    cur_in[seg0+i]=I;
                }
                if (P->rcore[k]) I=Bin;
                for (int i=n-1;i>=0;--i) {
                    int s=P->rsh[kb+i];
                    double ds=(i+1<n)?(P->rz[kb+i]-P->rz[kb+i+1]):(P->rz[kb+i]-P->rzin[k]);
                    size_t idx=(size_t)s*NB+b;
                    double chi=P->chi_tot[idx];
                    double rr=(chi>0.0)?P->chi_es[idx]/chi:0.0;
                    double Sv=P->S_fixed[idx]+rr*J[idx];
                    double chi_rad=chi+P->a_lam*4.0;
                    double dtau=chi_rad*ds, ex=EXT[seg0+i], wsum;
                    if(dtau>1e-4) wsum=1.0-ex; else wsum=dtau;
                    double I_rad=I*ex+wsum*Sv;
                    double bds=adv*ds;
                    double Iblue=has_blue?lag_out[blue0+i]:I_rad;
                    I=(I_rad+bds*Iblue)/(1.0+bds);
                    cur_out[seg0+i]=I;
                }
            }
        }
        /* ---- k_jint: thread (s,b) ---- */
        #pragma omp parallel for schedule(static)
        for (int s = 0; s < NS; ++s) {
            int o0=P->shell_off[s], o1=P->shell_off[s+1], c=o1-o0;
            for (int b = 0; b < NB; ++b) {
                size_t idx=(size_t)s*NB+b;
                if (c<1) { double chi=P->chi_tot[idx];
                    double rr=(chi>0.0)?P->chi_es[idx]/chi:0.0;
                    Jnew[idx]=P->S_fixed[idx]+rr*J[idx]; continue; }
                size_t bbase=(size_t)b*NP*(NS+1);
                int kk=P->shell_k[o0], ii=P->shell_seg[o0];
                size_t sg=bbase+(size_t)kk*(NS+1)+ii;
                double prev_mu=0.0, prev_jv=0.5*(cur_out[sg]+cur_in[sg]), Js=0.0;
                for (int a=o0;a<o1;++a) {
                    int k2=P->shell_k[a], i2=P->shell_seg[a];
                    double mu=P->shell_mu[a];
                    size_t sg2=bbase+(size_t)k2*(NS+1)+i2;
                    double jv=0.5*(cur_out[sg2]+cur_in[sg2]);
                    Js += 0.5*(prev_jv+jv)*(mu-prev_mu);
                    prev_mu=mu; prev_jv=jv;
                }
                Jnew[idx]=Js;
            }
        }
        /* ---- k_update ---- */
        double maxrel=0.0;
        long ncell=(long)NS*NB;
        #pragma omp parallel for schedule(static) reduction(max:maxrel)
        for (long i=0;i<ncell;++i){
            double d=fabs(Jnew[i]-J[i])/(fabs(J[i])+1e-30); if(d>maxrel)maxrel=d; J[i]=Jnew[i]; }
        if (hist) hist[it]=maxrel;
        if (maxrel<tol && it>0) { used=it+1; break; }
        { double *t1=lag_in; lag_in=cur_in; cur_in=t1;
          double *t2=lag_out; lag_out=cur_out; cur_out=t2; }
    }
    free(in0);free(in1);free(out0);free(out1);free(Jnew);
    return used;
}

/* ================================================================== */
/* EXACT: formal solution along the true drifting characteristic       */
/* ================================================================== */
/* Per-cell drift bookkeeping.  A_DRIFT = d(bin)/dz = a_lam/dlognu, so crossing
 * ONE bin of drift costs a path length 1/A_DRIFT and an optical depth
 * DT1 = chi_rad/A_DRIFT.  (The scheme's own adv[b] = a_lam/(1-exp(-dlognu))
 * differs from A_DRIFT by a factor 1+dlognu/2 = 1+7e-6 at production; that
 * O(dlognu) difference is 5 orders below anything measured here.) */
static double *DT1 = NULL;   /* [NS*NB] dtau per bin of drift */
static double *T1  = NULL;   /* [NS*NB] exp(-DT1)             */
static double  A_DRIFT;
static long    EX_CLAMP = 0; /* negative-R roundoff clamps (must stay ~0) */

static void build_dtau1(Prob *P) {
    int NS=P->NS, NB=P->NB;
    A_DRIFT = P->a_lam / P->dlognu;
    DT1=malloc((size_t)NS*NB*sizeof(double));
    T1 =malloc((size_t)NS*NB*sizeof(double));
    if(!DT1||!T1){fprintf(stderr,"DT1/T1 alloc failed\n");exit(1);}
    #pragma omp parallel for schedule(static)
    for (int s=0;s<NS;++s) for (int b=0;b<NB;++b) {
        size_t i=(size_t)s*NB+b;
        double d=(P->chi_tot[i]+P->a_lam*4.0)/A_DRIFT;
        DT1[i]=d; T1[i]=exp(-d);
    }
}

#define CLI(m,NB) ((m)<0?0:((m)>(NB)-1?(NB)-1:(m)))

/* One ray segment (shell s, geometric length ds -> drift beta = A_DRIFT*ds
 * bins).  Uin is indexed by the bin at the segment's BLUE (upstream) end, Uout
 * by the bin at its RED (downstream) end: the characteristic that ends at bin b
 * started at bin b+beta.  uin_zero=1 means "vacuum upstream".
 * mode 1 = sliding window (O(1)/bin), mode 2 = direct cell march (O(beta)/bin).
 * Bins whose upstream b+beta leaves the grid clamp to bin NB-1 (the window
 * truncation; those bins are exactly probe (b)'s contaminated zone). */
static void seg_exact(int NB, const double *dt1, const double *t1,
                      const double *Sc, const double *ac,
                      double beta, const double *Uin, int uin_zero,
                      double *Uout, int mode)
{
    int q = (int)floor(beta);
    double phi = beta - q;
    /* trivial case: the whole path stays inside the end cell */
    if (beta <= 0.5) {
        for (int b=0;b<NB;++b) {
            double I = 0.0;
            if (!uin_zero) { int i0=CLI(b+q,NB), i1=CLI(b+q+1,NB);
                             I = (1.0-phi)*Uin[i0] + phi*Uin[i1]; }
            double dt = dt1[b]*beta, ex = exp(-dt);
            Uout[b] = I*ex + (1.0-ex)*Sc[b];
        }
        return;
    }
    int qt; double psi;
    if (phi < 0.5) { qt = q;     psi = phi + 0.5; }
    else           { qt = q + 1; psi = phi - 0.5; }
    /* path = head half-cell at b | full cells b+1..b+qt-1 | psi of cell b+qt */

    if (mode == 2) {
        for (int b=0;b<NB;++b) {
            double x = b + beta, I = 0.0;
            if (!uin_zero) { int i0=CLI(b+q,NB), i1=CLI(b+q+1,NB);
                             I = (1.0-phi)*Uin[i0] + phi*Uin[i1]; }
            int m = (int)floor(x + 0.5);
            double xc = x;
            while (1) {
                double xlo = (m - 0.5 > (double)b) ? (m - 0.5) : (double)b;
                double len = xc - xlo;
                if (len > 0.0) { int mm = CLI(m,NB);
                    double ex = exp(-dt1[mm]*len);
                    I = I*ex + (1.0-ex)*Sc[mm]; }
                if (xlo <= (double)b + 1e-12) break;
                xc = xlo; --m;
            }
            Uout[b] = I;
        }
        return;
    }

    /* ---- mode 1: sliding window over the qt-1 full cells ------------------
     * W(b) = SUM_{m=b+1}^{b+qt-1} dt1_m           (window optical depth)
     * R(b) = SUM_{m=b+1}^{b+qt-1} a_m PROD_{m'=b+1}^{m-1} t_m'   (window source)
     *   W(b-1) = W(b) + dt1_b - dt1_{b+qt-1}
     *   R(b-1) = a_b + t_b*( R(b) - a_{b+qt-1}*exp(-(W(b)-dt1_{b+qt-1})) )
     * The subtracted term is the window's BLUEST (most attenuated) one, so the
     * cancellation is benign; R<0 from roundoff is clamped and counted. */
    int b = NB-1;
    double W = 0.0, R = 0.0;
    { double pr = 1.0;
      for (int m=b+1; m<=b+qt-1; ++m) { int mm=CLI(m,NB);
          R += ac[mm]*pr; pr *= t1[mm]; W += dt1[mm]; } }
    long clamps = 0;
    for (; b >= 0; --b) {
        double I = 0.0;
        if (!uin_zero) { int i0=CLI(b+q,NB), i1=CLI(b+q+1,NB);
                         I = (1.0-phi)*Uin[i0] + phi*Uin[i1]; }
        int mt = CLI(b+qt,NB);
        double ex = exp(-psi*dt1[mt]);
        I = I*ex + (1.0-ex)*Sc[mt];              /* psi of the top cell   */
        I = I*exp(-W) + R;                       /* the full cells        */
        double hb = sqrt(t1[b]);                 /* exp(-0.5*dt1_b)       */
        Uout[b] = I*hb + (1.0-hb)*Sc[b];         /* head half-cell        */
        if (b == 0) break;
        if (qt < 2) continue;            /* empty window: W=R=0 identically */
        int mo = CLI(b+qt-1,NB);
        double Wm = W - dt1[mo];
        R = ac[b] + t1[b]*(R - ac[mo]*exp(-Wm));
        if (R < 0.0) { R = 0.0; ++clamps; }
        W = Wm + dt1[b];                 /* W(b-1) = W(b) - dt1_{b+qt-1} + dt1_b */
    }
    if (clamps) {
        #pragma omp atomic
        EX_CLAMP += clamps;
    }
}

static int solve_exact(Prob *P, double *J, int maxit, double tol, double *hist, int mode)
{
    int NS=P->NS, NB=P->NB, NP=P->NP;
    size_t stride=(size_t)NP*(NS+1)*(size_t)NB;
    double *EXin =malloc(stride*sizeof(double));
    double *EXout=malloc(stride*sizeof(double));
    if(!EXin||!EXout){fprintf(stderr,"EXACT buffers %.1f GB failed\n",2.0*stride*8/1e9);exit(1);}
    double *S =malloc((size_t)NS*NB*sizeof(double));
    double *ac=malloc((size_t)NS*NB*sizeof(double));
    double *Jnew=malloc((size_t)NS*NB*sizeof(double));
    double *Bin=malloc(NB*sizeof(double));
    for (int b=0;b<NB;++b) Bin[b]=planck(P->nu[b],T_INNER);
    int used=maxit;
    for (int it=0; it<maxit; ++it) {
        #pragma omp parallel for schedule(static)
        for (int s=0;s<NS;++s) for (int b=0;b<NB;++b) {
            size_t i=(size_t)s*NB+b;
            double r=(P->chi_tot[i]>0.0)?P->chi_es[i]/P->chi_tot[i]:0.0;
            S[i]=P->S_fixed[i]+r*J[i];
            ac[i]=(1.0-T1[i])*S[i];
        }
        #pragma omp parallel for schedule(dynamic,1)
        for (int k=0;k<NP;++k) {
            int n=P->rn[k]; if(!n) continue;
            size_t kb=(size_t)k*(NS+1);
            for (int i=0;i<n;++i) {              /* inbound: outer -> inner */
                int s=P->rsh[kb+i];
                double ds=(i+1<n)?(P->rz[kb+i]-P->rz[kb+i+1]):(P->rz[kb+i]-P->rzin[k]);
                const double *U=(i>0)?(EXin+((size_t)(k*(NS+1)+i-1))*NB):NULL;
                seg_exact(NB,DT1+(size_t)s*NB,T1+(size_t)s*NB,S+(size_t)s*NB,ac+(size_t)s*NB,
                          A_DRIFT*ds,U,(i==0),EXin+((size_t)(k*(NS+1)+i))*NB,mode);
            }
            for (int i=n-1;i>=0;--i) {           /* outbound: inner -> outer */
                int s=P->rsh[kb+i];
                double ds=(i+1<n)?(P->rz[kb+i]-P->rz[kb+i+1]):(P->rz[kb+i]-P->rzin[k]);
                const double *U;
                if (i==n-1) U = P->rcore[k] ? Bin : (EXin+((size_t)(k*(NS+1)+n-1))*NB);
                else        U = EXout+((size_t)(k*(NS+1)+i+1))*NB;
                seg_exact(NB,DT1+(size_t)s*NB,T1+(size_t)s*NB,S+(size_t)s*NB,ac+(size_t)s*NB,
                          A_DRIFT*ds,U,0,EXout+((size_t)(k*(NS+1)+i))*NB,mode);
            }
        }
        /* J integration: identical quadrature to the LAG k_jint */
        #pragma omp parallel for schedule(static)
        for (int s=0;s<NS;++s) {
            int o0=P->shell_off[s], o1=P->shell_off[s+1], c=o1-o0;
            for (int b=0;b<NB;++b) {
                size_t idx=(size_t)s*NB+b;
                if (c<1) { Jnew[idx]=S[idx]; continue; }
                int kk=P->shell_k[o0], ii=P->shell_seg[o0];
                size_t sg=((size_t)(kk*(NS+1)+ii))*NB+b;
                double prev_mu=0.0, prev_jv=0.5*(EXout[sg]+EXin[sg]), Js=0.0;
                for (int a=o0;a<o1;++a) {
                    int k2=P->shell_k[a], i2=P->shell_seg[a];
                    double mu=P->shell_mu[a];
                    size_t sg2=((size_t)(k2*(NS+1)+i2))*NB+b;
                    double jv=0.5*(EXout[sg2]+EXin[sg2]);
                    Js += 0.5*(prev_jv+jv)*(mu-prev_mu);
                    prev_mu=mu; prev_jv=jv;
                }
                Jnew[idx]=Js;
            }
        }
        double maxrel=0.0; long ncell=(long)NS*NB;
        #pragma omp parallel for schedule(static) reduction(max:maxrel)
        for (long i=0;i<ncell;++i){
            double d=fabs(Jnew[i]-J[i])/(fabs(J[i])+1e-30); if(d>maxrel)maxrel=d; J[i]=Jnew[i]; }
        if (hist) hist[it]=maxrel;
        if (maxrel<tol && it>0) { used=it+1; break; }
    }
    free(EXin);free(EXout);free(S);free(ac);free(Jnew);free(Bin);
    return used;
}

/* ------------------------------------------------------------------ *
 * Operator-level transmission probe (no solve).  For one shell crossing of
 * length ds the REF/LAG step transmits the upstream field through the geometric
 * mixing kernel w_j = beta^j/(1+beta)^{j+1}:
 *      G_b = SUM_j w_j exp(-chi_{b+j} ds)   <->   G_b = (T_b + beta G_{b+1})/(1+beta)
 * while the true characteristic must cross every intervening bin:
 *      E_b = exp(-INT chi dz) = exp(-SUM_m dt1_m * len_m)
 * G/E is the per-segment transmission overestimate with NO solver in the loop.
 * ------------------------------------------------------------------ */
static int cmp_d(const void *a, const void *b) {
    double x=*(const double*)a, y=*(const double*)b;
    return (x<y)?-1:((x>y)?1:0);
}

static void trans_probe(Prob *P, const char *tag)
{
    int NS=P->NS, NB=P->NB;
    char fn[512]; snprintf(fn,sizeof fn,"transprobe_%s.csv",tag);
    FILE *f=fopen(fn,"w");
    fprintf(f,"shell,ds_cm,beta_bins,dtau_seg_mean,E_med,G_med,ratio_p10,ratio_p50,ratio_p90,"
              "ratio_mean,lnratio_mean,dtau_bin_med,frac_bin_thin,dtau_cont\n");
    double *G=malloc((size_t)NB*sizeof(double));
    double *E=malloc((size_t)NB*sizeof(double));
    double *pre=malloc((size_t)(NB+1)*sizeof(double));
    double *rat=malloc((size_t)NB*sizeof(double));
    for (int s=0;s<NS;++s) {
        /* representative segment: the radial chord of the shell on the core ray */
        double ds = (s+1<NS) ? (P->rmid[s+1]-P->rmid[s]) : (P->rmid[s]-P->rmid[s-1]);
        double beta = A_DRIFT*ds;
        const double *dt1=DT1+(size_t)s*NB;
        pre[0]=0.0; for (int b=0;b<NB;++b) pre[b+1]=pre[b]+dt1[b];
        double mean=0.0;
        G[NB-1]=exp(-dt1[NB-1]*beta);
        for (int b=NB-2;b>=0;--b) G[b]=(exp(-dt1[b]*beta)+beta*G[b+1])/(1.0+beta);
        for (int b=0;b<NB;++b) {
            /* exact: head 0.5 at b, full b+1..b+qt-1, psi of b+qt  (== seg_exact) */
            int q=(int)floor(beta); double phi=beta-q; int qt; double psi;
            if(phi<0.5){qt=q;psi=phi+0.5;}else{qt=q+1;psi=phi-0.5;}
            int hi=CLI(b+qt-1,NB), tp=CLI(b+qt,NB);
            double W=0.5*dt1[b]+psi*dt1[tp];
            if (qt>=2) W += pre[hi+1]-pre[CLI(b+1,NB)];
            E[b]=exp(-W); mean+=W;
            rat[b]=(E[b]>0.0)?(G[b]/E[b]):INFINITY;
        }
        mean/=NB;
        double *c=malloc((size_t)NB*sizeof(double)); memcpy(c,rat,(size_t)NB*sizeof(double));
        double *e=malloc((size_t)NB*sizeof(double)); memcpy(e,E,(size_t)NB*sizeof(double));
        double *g=malloc((size_t)NB*sizeof(double)); memcpy(g,G,(size_t)NB*sizeof(double));
        qsort(c,NB,sizeof(double),cmp_d);
        qsort(e,NB,sizeof(double),cmp_d);
        qsort(g,NB,sizeof(double),cmp_d);
        double rm=0.0, lm=0.0; int nfin=0;
        for (int b=0;b<NB;++b) if (isfinite(rat[b])) { rm+=rat[b]; lm+=log(rat[b]); ++nfin; }
        /* WHY G >> E: the per-BIN segment optical depth chi_b*ds is bimodal --
         * a large fraction of bins sit at the line-free continuum floor, and the
         * mixing kernel lets the radiation run down those channels.  Report the
         * per-bin (not window-integrated) depth so the claim is checkable. */
        double *pb=malloc((size_t)NB*sizeof(double));
        for (int b=0;b<NB;++b) pb[b]=dt1[b]*beta;              /* = chi_b*ds */
        double dcont=(P->chi_es[(size_t)s*NB]+P->chi_abs[(size_t)s*NB]+P->a_lam*4.0)*ds;
        long thin=0; for (int b=0;b<NB;++b) if (pb[b] < 2.0*dcont) ++thin;
        qsort(pb,NB,sizeof(double),cmp_d);
        fprintf(f,"%d,%.6e,%.4f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.4f,%.6e\n",
                s,ds,beta,mean,e[NB/2],g[NB/2],c[NB/10],c[NB/2],c[(9*NB)/10],
                nfin?rm/nfin:0.0,nfin?lm/nfin:0.0,pb[NB/2],(double)thin/NB,dcont);
        free(pb);
        free(c);free(e);free(g);
    }
    fclose(f); free(G);free(E);free(pre);free(rat);
    fprintf(stderr,"[harness] wrote %s\n",fn);
}

/* ================================================================== */
int main(int argc, char **argv)
{
    int    NB = 12469, NS = 50;
    int    nali = -1;              /* default: production-equivalent r*20000 */
    double fabs_frac = 0.3;        /* chi_abs = fabs_frac * chi_es (grey)    */
    double taumul = 1.0;
    /* chi_line multiplier.  The ONE group that the r-similarity cannot hold is
     * the frequency-AVERAGED line blanketing <chi_line>/adv: subsampling the
     * list by r makes <chi_line> scale as r while adv also scales as r ->
     * <chi_line>/adv is held, BUT n_ali*<chi_line>/adv (the line optical depth
     * reached within the lag front) scales as r.  chimul<0 means "restore the
     * production blanket" = multiply chi0 by 1/r.  chimul=0 = continuum only.
     * Run 1 / 1/r / 0 to bracket the verdict. */
    double chimul = 1.0;
    int deposit_all = 1;     /* 1 = every in-window line (production blanket) */
    int out_stride  = 0;     /* jbar CSV row stride (0 = auto ~20k lines)     */
    int out_sel     = 0;     /* 0 = stride kept index (legacy), 1 = file index*/
    int slb_on      = 1;     /* use the measured sub-thermal S_l/B profile     */
    double slbmul   = 1.0;   /* extra constant factor on S_l/B (bracket arm)   */
    int    lag_conv_check = 0;     /* also run LAG to full convergence       */
    int    ref_maxit = 1500;
    double ref_tol = 1e-9;
    int    do_ref = 1, do_lag = 1; /* 0 skips that path (time/memory)        */
    int    exact_mode = 0;         /* 0 off, 1 sliding O(1), 2 direct O(beta)*/
    int    transprobe = 0;
    double lam_lo_win = LAM_LO;    /* blue edge of the solved window (A)     */
    double lam_hi_win = LAM_HI;    /* red edge.  The whole solve is strictly
                                    * one-way blue->red (bin b reads only b+1),
                                    * so dropping the reddest bins is LOSSLESS
                                    * for every bin that is kept -- it just makes
                                    * a production-resolution FUV probe 4x cheaper. */
    const char *tag = "run";
    const char *lines_bin = "lines3.bin";
    const char *plasma_csv = "plasma_state.csv";
    for (int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"--nb")) NB=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ns")) NS=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--nali")) nali=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--fabs")) fabs_frac=atof(argv[++i]);
        else if(!strcmp(argv[i],"--taumul")) taumul=atof(argv[++i]);
        else if(!strcmp(argv[i],"--chimul")) chimul=atof(argv[++i]);
        else if(!strcmp(argv[i],"--depositall")) deposit_all=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--outstride")) out_stride=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--outsel")) out_sel=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--slb")) slb_on=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--slbmul")) slbmul=atof(argv[++i]);
        else if(!strcmp(argv[i],"--lagconv")) lag_conv_check=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ref")) do_ref=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--lag")) do_lag=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--exact")) exact_mode=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--transprobe")) transprobe=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--lamlo")) lam_lo_win=atof(argv[++i]);
        else if(!strcmp(argv[i],"--lamhi")) lam_hi_win=atof(argv[++i]);
        else if(!strcmp(argv[i],"--reftol")) ref_tol=atof(argv[++i]);
        else if(!strcmp(argv[i],"--refmax")) ref_maxit=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--out")) tag=argv[++i];
        else if(!strcmp(argv[i],"--lines")) lines_bin=argv[++i];
        else if(!strcmp(argv[i],"--plasma")) plasma_csv=argv[++i];
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }
    Prob P; memset(&P,0,sizeof P);
    P.NS=NS; P.NCORE=16; P.NP=NS+16;

    /* --- frequency grid + similarity factor ---------------------------------
     * --nb ALWAYS means "bins spanning the production window 1000-4000 A", so
     * dlognu is pinned by --nb alone.  --lamlo<1000 then only APPENDS bins on
     * the blue side: nu_lo and dlognu are untouched, hence nu[b] is bit-identical
     * on every shared bin and a narrow-vs-wide A/B differs by the added blue
     * reservoir ALONE (probe b).  With --lamlo 1000 (default) this reduces
     * exactly to the original grid. */
    double nu_lo_ref = CM_C/(LAM_HI*1e-8), nu_hi_ref = CM_C/(LAM_LO*1e-8);
    P.dlognu = log(nu_hi_ref/nu_lo_ref)/(double)(NB-1);   /* pinned by --nb alone */
    P.nu_lo  = CM_C/(lam_hi_win*1e-8);
    int NB_blue = (lam_lo_win < LAM_LO)
                ? (int)llround(log(LAM_LO/lam_lo_win)/P.dlognu) : 0;
    if (lam_hi_win < LAM_HI)                    /* drop the reddest bins */
        NB -= (int)llround(log(LAM_HI/lam_hi_win)/P.dlognu);
    NB += NB_blue;
    P.NB = NB;
    P.nu_hi = P.nu_lo*exp((double)(NB-1)*P.dlognu); /* line filter edge; == the
                                   * original c/1000A when NB_blue==0 */
    SEG1  = (size_t)P.NP*(NS+1);
    SEGSZ = SEG1*(size_t)NB;
    double dlognu_prod = (VDOP_PROD/CM_C)/PPD;
    double r = dlognu_prod/P.dlognu;              /* == NB/NB_prod (approx) */
    P.vdop = PPD*P.dlognu*CM_C;                   /* ppd = 12 held exactly  */
    P.a_lam = 1.0/(T_EXP*CM_C);
    if (nali < 0) { nali = (int)llround(NALI_PROD*r); if(nali<1) nali=1; }

    P.nu = malloc(NB*sizeof(double)); P.lam = malloc(NB*sizeof(double));
    for (int b=0;b<NB;++b){ P.nu[b]=P.nu_lo*exp((b+0.5)*P.dlognu); P.lam[b]=CM_C/P.nu[b]; }
    P.adv = malloc(NB*sizeof(double));
    for (int b=0;b<NB;++b){
        if (b<NB-1){ double Dlam=P.lam[b]-P.lam[b+1]; P.adv[b]=P.a_lam*P.lam[b]/Dlam; }
        else P.adv[b]=0.0; }

    /* --- geometry (uniform in v, production span) --- */
    P.rmid=malloc(NS*sizeof(double));
    double dv=(V_OUT-V_IN)/NS;
    for (int s=0;s<NS;++s) P.rmid[s]=(V_IN+(s+0.5)*dv)*T_EXP;
    build_rays(&P);

    /* --- plasma: n_e, T_e from the parity42 dump (50 shells) --- */
    P.ne=malloc(NS*sizeof(double)); P.Te=malloc(NS*sizeof(double));
    {   FILE *f=fopen(plasma_csv,"r"); if(!f){perror(plasma_csv);return 1;}
        char buf[512]; fgets(buf,sizeof buf,f);
        double ne50[64],te50[64]; int n50=0;
        while (fgets(buf,sizeof buf,f) && n50<64) {
            int sid; double W,Trad,ne,te;
            if (sscanf(buf,"%d,%lf,%lf,%lf,%lf",&sid,&W,&Trad,&ne,&te)==5){ ne50[n50]=ne; te50[n50]=te; n50++; } }
        fclose(f);
        for (int s=0;s<NS;++s) { double x=(NS==n50)?s:((double)s*(n50-1)/(NS-1));
            int i0=(int)x; if(i0>n50-2)i0=n50-2; double fr=x-i0;
            P.ne[s]=exp((1-fr)*log(ne50[i0])+fr*log(ne50[i0+1]));
            P.Te[s]=(1-fr)*te50[i0]+fr*te50[i0+1]; }
    }

    /* --- continuum --- */
    size_t cell=(size_t)NS*NB;
    P.chi_es=malloc(cell*sizeof(double)); P.chi_abs=malloc(cell*sizeof(double));
    P.chi_line=calloc(cell,sizeof(double)); P.chi_tot=malloc(cell*sizeof(double));
    P.S_fixed=malloc(cell*sizeof(double)); P.J0=malloc(cell*sizeof(double));
    double *eta=calloc(cell,sizeof(double));
    for (int s=0;s<NS;++s){ double ce=P.ne[s]*SIGMA_T;
        for (int b=0;b<NB;++b){ size_t i=(size_t)s*NB+b; P.chi_es[i]=ce; P.chi_abs[i]=fabs_frac*ce; } }

    /* --- line forest: real (lambda, tau@s8,s45,s49) subsampled by r --- */
    {   FILE *f=fopen(lines_bin,"rb"); if(!f){perror(lines_bin);return 1;}
        long ntot; if(fread(&ntot,sizeof(long),1,f)!=1){fprintf(stderr,"bad lines file\n");return 1;}
        double *L=malloc((size_t)ntot*4*sizeof(double));
        if(fread(L,sizeof(double),(size_t)ntot*4,f)!=(size_t)ntot*4){fprintf(stderr,"short read\n");return 1;}
        fclose(f);
        /* deposit=all  : keep EVERY in-window line.  <chi_line> (and therefore
         *                kappa_line = n_ali*<chi_line>/adv) is then EXACTLY the
         *                production value; individual Doppler cores are not
         *                resolved on the coarse grid, but production already
         *                blends ~125 lines per bin so the field structure is
         *                the blend anyway.  THIS IS THE REFERENCE VARIANT.
         * deposit=sub  : keep every (1/r)-th line -> resolved profiles at the
         *                production lines-per-bin density, but <chi_line>
         *                under-represented by r (unless --chimul -1). */
        double stride = deposit_all ? 1.0 : (1.0/r);
        long nkeep_max = (long)(ntot/stride)+2;
        P.l_nu = malloc(nkeep_max*sizeof(double));
        P.l_tau= malloc((size_t)nkeep_max*NS*sizeof(double));
        P.l_id = malloc(nkeep_max*sizeof(long));
        int nk=0;
        double acc=0.0;
        double rho8=0.0;  /* density ratio proxy for s<8 extrapolation */
        for (long li=0; li<ntot; ++li) {
            acc += 1.0;
            if (acc < stride) continue;
            acc -= stride;
            double lamA=L[li*4+0], t8=L[li*4+1], t45=L[li*4+2], t49=L[li*4+3];
            if (t8<=1e-12 && t45<=1e-12 && t49<=1e-12) continue;
            double nu_l=CM_C/(lamA*1e-8);
            if (nu_l<=P.nu_lo || nu_l>=P.nu_hi) continue;
            if (nk>=nkeep_max) break;
            double g8 =(t8 >1e-100)?log10(t8 ):-100.0;
            double g45=(t45>1e-100)?log10(t45):-100.0;
            double g49=(t49>1e-100)?log10(t49):-100.0;
            for (int s=0;s<NS;++s) {
                double ss=(NS==50)?(double)s:((double)s*49.0/(NS-1));
                double g;
                if (ss<=8.0)       g=g8 + (8.0-ss)*0.0;      /* flat below s8 */
                else if (ss<=45.0) g=g8 +(g45-g8 )*(ss-8.0)/37.0;
                else if (ss<=49.0) g=g45+(g49-g45)*(ss-45.0)/4.0;
                else               g=g49;
                double tv=(g<=-90.0)?0.0:pow(10.0,g)*taumul;
                P.l_tau[(size_t)nk*NS+s]=tv;
            }
            P.l_nu[nk]=nu_l; P.l_id[nk]=li; ++nk;
        }
        (void)rho8;
        P.n_line=nk; free(L);
        fprintf(stderr,"[harness] lines kept=%d (from %ld, stride=%.2f, deposit=%s)\n",
                nk,ntot,stride,deposit_all?"all":"sub");
        if (out_stride<=0) out_stride = (nk>20000)?(nk/20000):1;
    }

    /* --- per-shell line source S_l/B.  parity42 ran src_nlte (line_source_S),
     * whose measured medians in the preserved linedumps are
     *   s8 = 1.0000,  s45 = 0.5563,  s49 = 0.0597.
     * slb_on=0 -> thermal S_l=B everywhere (the optimistic arm: the outer
     * shells then have a strong LOCAL thermal source, i.e. a SHORT reach, which
     * UNDER-states the lag error).  slb_on=1 -> the measured sub-thermal
     * profile, which is what production actually solved. */
    double *slb = malloc(NS*sizeof(double));
    for (int s=0;s<NS;++s) {
        double ss=(NS==50)?(double)s:((double)s*49.0/(NS-1));
        double g;
        if (!slb_on) { slb[s]=slbmul; continue; }
        double g8=log10(1.0), g45=log10(0.5563), g49=log10(0.0597);
        if (ss<=8.0)       g=g8;
        else if (ss<=45.0) g=g8 +(g45-g8 )*(ss-8.0)/37.0;
        else if (ss<=49.0) g=g45+(g49-g45)*(ss-45.0)/4.0;
        else               g=g49;
        slb[s]=pow(10.0,g)*slbmul;
    }
    printf("# S_l/B profile: slb_on=%d  s0=%.4f s25=%.4f s45=%.4f s49=%.4f\n",
           slb_on,slb[0],slb[NS/2],slb[(45*NS)/50],slb[NS-1]);

    /* --- deposit the forest exactly as cmfgen_fine_jbar does --- */
    if (chimul < 0.0) chimul = 1.0/r;      /* restore production <chi_line> */
    double chi0_pref = chimul/(SQRTPI*P.vdop*T_EXP);
    printf("# chi_line multiplier=%.4g  (1=r-similar, 1/r=%.4g restores production blanket)\n",
           chimul, 1.0/r);
    #pragma omp parallel for schedule(dynamic)
    for (int s=0;s<NS;++s) {
        for (int l=0;l<P.n_line;++l) {
            double tau=P.l_tau[(size_t)l*NS+s];
            if (tau<=1e-12) continue;
            double nu_l=P.l_nu[l];
            double dnuD=nu_l*P.vdop/CM_C;
            double xc=log(nu_l/P.nu_lo)/P.dlognu-0.5;
            int half=(int)ceil(4.0*dnuD/(nu_l*P.dlognu))+1;
            int ic=(int)floor(xc+0.5);
            int i0=ic-half,i1=ic+half;
            if(i0<0)i0=0; if(i1>NB-1)i1=NB-1;
            if(i0>i1) continue;
            double frac=(tau>1e-6)?-expm1(-tau):tau;
            double chi0=frac*chi0_pref;
            double Bl=planck(nu_l,P.Te[s]);
            double emit=Bl*slb[s];    /* S_l = (S_l/B)(s) * B(Te); slb from the
                                       * parity42 linedump medians (s8/s45/s49) */
            for (int i=i0;i<=i1;++i) {
                double xv=(P.nu[i]-nu_l)/dnuD;
                double pr=exp(-xv*xv);
                double cl=chi0*pr;
                P.chi_line[(size_t)s*NB+i]+=cl;
                eta[(size_t)s*NB+i]+=cl*emit;
            }
        }
    }
    /* --- assemble (line_eps=1 thermal, exactly production parity42) --- */
    for (int s=0;s<NS;++s) {
        for (int b=0;b<NB;++b) {
            size_t idx=(size_t)s*NB+b;
            double ct=P.chi_es[idx]+P.chi_abs[idx]+P.chi_line[idx];
            P.chi_tot[idx]=ct;
            double Bnu=planck(P.nu[b],P.Te[s]);
            P.S_fixed[idx]=(ct>0.0)?(P.chi_abs[idx]*Bnu+eta[idx])/ct:0.0;
            P.J0[idx]=Bnu;
        }
    }
    free(eta);

    if (do_ref || do_lag) build_ext(&P);   /* EXACT does not use EXT (13 GB at
                                            * production NB) */

    /* --- diagnostics of the drift structure --- */
    double D_core=0.0, D_max=0.0;
    for (int k=0;k<P.NP;++k) {
        int n=P.rn[k]; size_t kb=(size_t)k*(NS+1); double d=0.0;
        for (int i=0;i<n;++i){ double ds=(i+1<n)?(P.rz[kb+i]-P.rz[kb+i+1]):(P.rz[kb+i]-P.rzin[k]);
            d += P.adv[0]*ds; }
        if (P.rcore[k]) { if(k==0) D_core=d; }
        else d*=2.0;                                /* tangent ray: in + out  */
        if (d>D_max) D_max=d;
    }
    printf("# NB=%d NS=%d NP=%d  dlognu=%.6e  r=NB/NB_prod=%.6f  vdop=%.4e cm/s (ppd=%.1f)\n",
           NB,NS,P.NP,P.dlognu,r,P.vdop,PPD);
    printf("# n_ali(LAG)=%d   production-equivalent (20000*r=%.1f)\n",nali,NALI_PROD*r);
    printf("# drift budget: D(core ray, inbound)=%.0f bins   D(max over rays)=%.0f bins\n",D_core,D_max);
    printf("# n_ali/D_core=%.3f   n_ali/D_max=%.3f   (production: 0.467 / 0.210)\n",
           nali/D_core, nali/D_max);
    printf("# chi_es[0]=%.4e adv=%.4e  kappa0=n_ali*chi_tot0/adv=%.3f\n",
           P.chi_es[0],P.adv[0],nali*P.chi_tot[0]/P.adv[0]);
    for (int s=0;s<NS;s+=(NS>=25?12:5)) {
        double ml=0.0; for(int b=0;b<NB;++b) ml+=P.chi_line[(size_t)s*NB+b]; ml/=NB;
        printf("# shell %2d: chi_es=%.3e <chi_line>=%.3e  kappa_es=%.3f kappa_line=%.3f\n",
               s,P.chi_es[(size_t)s*NB],ml,nali*P.chi_es[(size_t)s*NB]/P.adv[0],nali*ml/P.adv[0]);
    }

    build_dtau1(&P);
    printf("# EXACT drift: A=%.6e bins/cm  beta(core ray, per shell)=%.2f..%.2f bins"
           "  (= %.2f..%.2f Doppler widths at ppd=%.0f)\n",
           A_DRIFT, A_DRIFT*(P.rmid[1]-P.rmid[0]), A_DRIFT*(P.rmid[NS-1]-P.rmid[NS-2]),
           A_DRIFT*(P.rmid[1]-P.rmid[0])/PPD, A_DRIFT*(P.rmid[NS-1]-P.rmid[NS-2])/PPD, PPD);
    if (NB_blue) printf("# window widened: lam_lo=%.1f A -> %d extra blue bins "
                        "(shared bins bit-identical to the %.0f A grid)\n",
                        lam_lo_win, NB_blue, LAM_LO);
    if (transprobe) trans_probe(&P,tag);

    /* --- solve --- */
    double *Jref=malloc(cell*sizeof(double)), *Jlag=malloc(cell*sizeof(double));
    double *h1=calloc(ref_maxit,sizeof(double)), *h2=calloc(nali,sizeof(double));
    memcpy(Jref,P.J0,cell*sizeof(double));
    double t0=0.0;
    int itref=0, itlag=0;
    if (do_ref) {
#ifdef _OPENMP
        t0=omp_get_wtime();
#endif
        itref=solve_ref(&P,Jref,ref_maxit,ref_tol,h1);
#ifdef _OPENMP
        printf("# REF: ALI iters=%d (tol=%.1e) maxrel_last=%.3e  %.1f s\n",itref,ref_tol,h1[itref-1],omp_get_wtime()-t0);
#else
        printf("# REF: ALI iters=%d maxrel_last=%.3e\n",itref,h1[itref-1]);
#endif
    }
    if (do_lag) {
        memcpy(Jlag,P.J0,cell*sizeof(double));
#ifdef _OPENMP
        t0=omp_get_wtime();
#endif
        itlag=solve_lag(&P,Jlag,nali,1e-4,h2);
#ifdef _OPENMP
        printf("# LAG: iters used=%d / n_ali=%d (tol=1e-4 EARLY STOP %s) maxrel_last=%.3e  %.1f s\n",
               itlag,nali,(itlag<nali)?"FIRED":"not reached",h2[itlag-1],omp_get_wtime()-t0);
#else
        printf("# LAG: iters used=%d / %d\n",itlag,nali);
#endif
    } else memset(Jlag,0,cell*sizeof(double));

    double *Jex=NULL;
    if (exact_mode) {
        Jex=malloc(cell*sizeof(double)); memcpy(Jex,P.J0,cell*sizeof(double));
        double *h4=calloc(ref_maxit,sizeof(double));
#ifdef _OPENMP
        t0=omp_get_wtime();
#endif
        int itex=solve_exact(&P,Jex,ref_maxit,ref_tol,h4,exact_mode);
#ifdef _OPENMP
        printf("# EXACT(mode %d): ALI iters=%d (tol=%.1e) maxrel_last=%.3e  %.1f s  R-clamps=%ld\n",
               exact_mode,itex,ref_tol,h4[itex-1],omp_get_wtime()-t0,EX_CLAMP);
#else
        printf("# EXACT(mode %d): ALI iters=%d maxrel_last=%.3e R-clamps=%ld\n",exact_mode,itex,h4[itex-1],EX_CLAMP);
#endif
        if (do_ref) {
            double mx=0.0,l2n=0,l2d=0;
            for(size_t i=0;i<cell;++i){ double d=fabs(Jex[i]-Jref[i])/(fabs(Jref[i])+1e-30);
                if(d>mx)mx=d; l2n+=(Jex[i]-Jref[i])*(Jex[i]-Jref[i]); l2d+=Jref[i]*Jref[i]; }
            printf("# EXACT vs REF (all cells): max rel=%.3e  L2 rel=%.3e\n",mx,sqrt(l2n/l2d));
        }
        free(h4);
    }

    double *Jlagc=NULL;
    if (lag_conv_check) {
        Jlagc=malloc(cell*sizeof(double)); memcpy(Jlagc,P.J0,cell*sizeof(double));
        int nn=lag_conv_check; double *h3=calloc(nn,sizeof(double));
        int it3=solve_lag(&P,Jlagc,nn,1e-9,h3);
        double mx=0.0,l2n=0,l2d=0;
        for(size_t i=0;i<cell;++i){ double d=fabs(Jlagc[i]-Jref[i])/(fabs(Jref[i])+1e-30);
            if(d>mx)mx=d; l2n+=(Jlagc[i]-Jref[i])*(Jlagc[i]-Jref[i]); l2d+=Jref[i]*Jref[i]; }
        printf("# LAG-CONVERGED(n=%d used=%d tol=1e-9): max rel vs REF=%.3e  L2 rel=%.3e  [same fixed point check]\n",
               nn,it3,mx,sqrt(l2n/l2d));
        free(h3);
    }

    /* --- per-line jbar extraction, exactly as cmfgen_fine_jbar --- */
    char fn[512]; snprintf(fn,sizeof fn,"detjbar_%s.csv",tag);
    FILE *fo=fopen(fn,"w");
    fprintf(fo,"line_idx,shell,lambda_A,tau_sob,jbar_ref,jbar_lag,jbar_lagconv,B_Te,jbar_exact\n");
    for (int l=0;l<P.n_line;++l) {
        /* out_sel 0 (legacy) strides the KEPT index; 1 strides the lines-file
         * index so a narrow-vs-wide pair emits the SAME physical lines even
         * though the wide run's array also holds the blue extension. */
        if (out_sel ? (P.l_id[l] % out_stride) : (l % out_stride)) continue;
        double nu_l=P.l_nu[l];
        double dnuD=nu_l*P.vdop/CM_C;
        double xc=log(nu_l/P.nu_lo)/P.dlognu-0.5;
        int half=(int)ceil(4.0*dnuD/(nu_l*P.dlognu))+1;
        int ic=(int)floor(xc+0.5);
        int i0=ic-half,i1=ic+half;
        if(i0<0)i0=0; if(i1>NB-1)i1=NB-1;
        for (int s=0;s<NS;++s) {
            double tau=P.l_tau[(size_t)l*NS+s];
            if (tau<=1e-12) continue;
            double nr=0,nl=0,nc=0,ne=0,den=0;
            for (int i=i0;i<=i1;++i) {
                double xv=(P.nu[i]-nu_l)/dnuD, w=exp(-xv*xv)*P.nu[i]*P.dlognu;
                nr+=w*Jref[(size_t)s*NB+i]; nl+=w*Jlag[(size_t)s*NB+i];
                if(Jlagc) nc+=w*Jlagc[(size_t)s*NB+i];
                if(Jex)   ne+=w*Jex  [(size_t)s*NB+i];
                den+=w;
            }
            if (den<=0) continue;
            fprintf(fo,"%ld,%d,%.4f,%.6e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                    out_sel?P.l_id[l]:(long)l,s,CM_C/nu_l*1e8,tau,nr/den,nl/den,
                    Jlagc?nc/den:0.0,planck(nu_l,P.Te[s]),Jex?ne/den:0.0);
        }
    }
    fclose(fo);
    fprintf(stderr,"[harness] wrote %s\n",fn);

    /* --- iteration history --- */
    snprintf(fn,sizeof fn,"iterhist_%s.csv",tag);
    fo=fopen(fn,"w"); fprintf(fo,"iter,ref_maxrel,lag_maxrel\n");
    int nh=(itref>itlag)?itref:itlag;
    for (int i=0;i<nh;++i) fprintf(fo,"%d,%.6e,%.6e\n",i,(i<itref)?h1[i]:0.0,(i<itlag)?h2[i]:0.0);
    fclose(fo);
    fprintf(stderr,"[harness] wrote %s\n",fn);
    return 0;
}
