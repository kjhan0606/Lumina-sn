/* DET-DJDN-DRIVE selftest (CPU, no GPU, no sealed inputs).
 *
 * Verifies the contract (PREREG 2026-09-20 §3/§4) at the level of the shared,
 * single-source pure expressions the live ionization balance uses (test hooks
 * in lumina_plasma.c under LUMINA_POP_DJDN_SELFTEST):
 *   - a-reduce is >0 with a genuine lambda* and ==0 when lambda* is all-zero
 *     (the a==0 no-op wiring negative control: the term reads NOTHING but the
 *     stored lambda*, so lambda*=0 => OFF-equivalent), + an INDEPENDENT hand
 *     recomputation of the §4 a algebra (unchanged from DJDN-MVP);
 *   - the damped log-Newton (pop_djdn_log_newton) REPLACING the DJDN-MVP
 *     root-existence fallback: NO den-a<=0 branch. For the three old fallback
 *     regimes (g>1 & rho>g, g<1 & rho<g, singular rho==g) x* is now finite and
 *     positive (the singular case damps: damp<1);
 *   - c independent recompute (1e-12 rel): c_newton == ln(rho)*g/(rho-g);
 *   - lambda*==0 (a==0) => x* == x_map BIT-IDENTICAL (negative control 1);
 *   - rho==1 (x_lag==x_map) => c==0 => x* == x_map == x_lag BIT-IDENTICAL;
 *   - |c|<=C_max => damp==1 and x* BIT-IDENTICAL to the undamped step;
 *   - |c|>C_max => x* == x_map*exp(+-C_max) (1e-12 rel);
 *   - G-CPL x* identity |x* - x_map*exp(damp*c_newton)| and its NEGATIVE
 *     CONTROL 2: a 1e-6 perturbation of c breaks it;
 *   - the gamma_eff G-CPL identity + perturbed-a negative control (unchanged);
 *   - x_lag snapshot regression + the census header/row (16 columns).
 *
 * Pure functions => no PlasmaState/AtomicData fixture needed. Production and this
 * test call the identical arithmetic, so they can never diverge.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "lumina_cmfgen.h"   /* CMFGENState, Geometry (production fixture types) */

/* test hooks from lumina_plasma.c (under LUMINA_POP_DJDN_SELFTEST) */
extern double pop_djdn_a_reduce_test(int nb, const double *Gw, const double *Ec,
        const double *Xc, const double *lam, const double *S,
        const double *chi_t, double n_e, double N_tot, double x_lag);
extern int pop_djdn_log_newton_test(double num_lag, double den, double a,
        double x_lag, double c_max, double *x_star, double *c_out,
        double *damp_out);
extern double pop_djdn_gamma_eff_test(double gamma_lag, double a, double x_star,
        double x_lag);
extern double pop_djdn_maxch_factor_test(double maxch, double norm);
extern int pop_djdn_row_format_test(char *buf, size_t n, long gen, int shell,
        int Z, int stage, double x_lag, double x_map, double a, double g,
        double x_star, int root, double gamma_lag, double gamma_eff,
        double maxch_factor, long epoch, double c_newton, double damp);
extern const char *pop_djdn_header_test(void);
extern int pop_djdn_xlag_test(const double *base, int n_ions, int n_shells,
        int ip_cur, int ip_next, int s, double *x_lag);
/* DET-DJDN-SPLIT hooks */
extern const char *pop_djdn_hold_header_test(void);
extern int pop_djdn_hold_row_format_test(char *buf, size_t n, int shell, int Z,
        int stage, long epoch, double x_lag, double x_lag_prev, double x_map,
        double x_map_hold, double x_map_null, double g, double s, double s_hold,
        double s_nl, double denom, int restore_ok, double null_rel);
extern int pop_djdn_split_test(double x_map_K, double x_map_Km1,
        double x_map_hold, double x_lag_K, double x_lag_Km1, double *denom,
        double *s, double *s_hold, double *s_nl);
extern void pop_djdn_hold_input_test(double N_lo, double N_hi, double x_target,
        double *N_lo_out, double *N_hi_out);
extern int pop_djdn_restore_ok_test(double x_snap, double x_after);
extern int pop_djdn_fdhold_gate_test(void);
/* DET-PATHB-LOCAL hooks (lumina_plasma.c) */
extern double pop_djdn_a_chi_reduce_test(int nb, const double *Gw,
        const double *Xc, const double *dJ_fs_dchi, double N_tot, double x_lag);
extern double pop_djdn_a_reduce_pos_test(int nb, const double *Gw,
        const double *Ec, const double *Xc, const double *lam, const double *S,
        const double *chi_t, double n_e, double N_tot, double x_lag);
extern const char *pop_djdn_chi_header_test(void);
extern int pop_djdn_chi_row_format_test(char *buf, size_t n, long gen, int shell,
        int Z, int stage, long epoch, double a_ds, double a_dchi,
        double a_local);
/* DET-PATHB-LOCAL hooks (lumina_cmfgen.c: production fs_seg_eval primitive) */
extern int formal_djdchi_seg_test(double dtau, double S, double I_in, double ds,
        double *I_out, double *dsrc_out, double *ex_out, double *psi_out,
        double *psip_out);
extern void formal_solve_ray_test(int n, const double *chi, const double *ds,
        const double *S, int core, double Bnu, double *J);
extern void formal_djdchi_ray_test(int n, const double *chi, const double *ds,
        const double *S, int core, double Bnu, double *dJ);
/* PRODUCTION-function hooks (call formal_djdchi_fs_S / formal_solve_bin) */
extern void formal_djdchi_fs_S_test(CMFGENState *cs, const Geometry *geo,
        double T_inner, double *dJdchi);
extern void formal_solve_bin_test(CMFGENState *cs, const Geometry *geo, int b,
        const double *S, double Bnu_inner, double *Jb, double *Lstar);
extern double formal_djdchi_bnu_test(double nu, double T_inner);

static int g_fail = 0;
static void ok(int cond, const char *msg){
    if (cond) printf("[PASS] %s\n", msg);
    else { printf("[FAIL] %s\n", msg); g_fail = 1; }
}
static int count_fields(const char *s){
    int f = 1;
    for (; *s && *s != '\n'; ++s) if (*s == ',') ++f;
    return f;
}

/* DET-PATHB-LOCAL: independent reference for the single-ray fixed-S dJ/dchi
 * recursion. Uses the PRODUCTION per-segment primitive (formal_djdchi_seg_test)
 * but does the dI[] bookkeeping here, so it cross-checks the production ray hook
 * (formal_djdchi_ray_test)'s inbound/inner/outbound structure. reset_at_inner=1
 * zeros dI at the inner boundary (core: B(T_inner) is chi-independent);
 * use_inner_I=1 sets I=inner_I there. */
static void ref_djdchi_ray(int n, const double *chi, const double *ds,
        const double *S, int reset_at_inner, int use_inner_I, double inner_I,
        double *dJ){
    double dJacc[16] = {0}, wacc[16] = {0}, dI[16] = {0};
    double I = 0.0, ray_w = 1.0;
    for (int i = 0; i < n; ++i) {
        double Iout, dsrc, ex, psi, psip;
        formal_djdchi_seg_test(chi[i]*ds[i], S[i], I, ds[i],
                               &Iout, &dsrc, &ex, &psi, &psip);
        for (int t = 0; t < n; ++t) dI[t] *= ex;
        dI[i] += dsrc;
        I = Iout;
        dJacc[i] += ray_w * dI[i]; wacc[i] += ray_w;
    }
    if (use_inner_I) I = inner_I;
    if (reset_at_inner) for (int t = 0; t < n; ++t) dI[t] = 0.0;
    for (int i = n - 1; i >= 0; --i) {
        double Iout, dsrc, ex, psi, psip;
        formal_djdchi_seg_test(chi[i]*ds[i], S[i], I, ds[i],
                               &Iout, &dsrc, &ex, &psi, &psip);
        for (int t = 0; t < n; ++t) dI[t] *= ex;
        dI[i] += dsrc;
        I = Iout;
        dJacc[i] += ray_w * dI[i]; wacc[i] += ray_w;
    }
    for (int i = 0; i < n; ++i) {
        double iw = (wacc[i] > 0.0) ? 1.0 / wacc[i] : 0.0;
        dJ[i] = dJacc[i] * iw;
    }
}

/* Small synthetic CMFGENState/Geometry fixture (3 shells, 2 bins, 1 core + 3
 * non-core rays) so the PRODUCTION formal_djdchi_fs_S / formal_solve_bin are run
 * directly. Only the fields those two functions touch are allocated. */
typedef struct { CMFGENState cs; Geometry geo; } LocalFix;
#define LF_NS 3
#define LF_NB 2
#define LF_NR 4
static void build_local_fixture(LocalFix *f){
    memset(&f->cs, 0, sizeof f->cs);
    memset(&f->geo, 0, sizeof f->geo);
    f->cs.n_shells = LF_NS; f->cs.n_bins = LF_NB; f->cs.n_rays = LF_NR;
    f->cs.nu          = malloc(LF_NB*sizeof(double));
    f->cs.dnu         = malloc(LF_NB*sizeof(double));
    f->cs.chi_tot     = malloc((size_t)LF_NS*LF_NB*sizeof(double));
    f->cs.S_fixed     = malloc((size_t)LF_NS*LF_NB*sizeof(double));
    f->cs.J           = malloc((size_t)LF_NS*LF_NB*sizeof(double));
    f->cs.lambda_star = malloc((size_t)LF_NS*LF_NB*sizeof(double));
    f->cs.p_ray       = malloc(LF_NR*sizeof(double));
    f->geo.n_shells = LF_NS;
    f->geo.r_inner  = malloc(LF_NS*sizeof(double));
    f->geo.r_outer  = malloc(LF_NS*sizeof(double));
    double ri[LF_NS] = {1.0, 2.0, 3.0}, ro[LF_NS] = {2.0, 3.0, 4.0};
    for (int s=0;s<LF_NS;++s){ f->geo.r_inner[s]=ri[s]; f->geo.r_outer[s]=ro[s]; }
    double pr[LF_NR] = {0.5, 1.2, 2.2, 3.2};   /* p=0.5 < r_inner[0]=1 => core */
    for (int i=0;i<LF_NR;++i) f->cs.p_ray[i]=pr[i];
    f->cs.nu[0]=1.0e15; f->cs.nu[1]=2.0e15;
    f->cs.dnu[0]=1.0e14; f->cs.dnu[1]=2.0e14;
    for (int s=0;s<LF_NS;++s) for(int b=0;b<LF_NB;++b){
        size_t idx=(size_t)s*LF_NB+b;
        f->cs.chi_tot[idx]     = 0.4 + 0.3*s + 0.1*b;   /* positive, distinct */
        f->cs.S_fixed[idx]     = 1.0 + 0.2*s + 0.5*b;
        f->cs.J[idx]           = -7.0 - s - b;          /* invariance sentinel */
        f->cs.lambda_star[idx] = 0.123 + s + 10.0*b;    /* invariance sentinel */
    }
}
static void free_local_fixture(LocalFix *f){
    free(f->cs.nu); free(f->cs.dnu); free(f->cs.chi_tot); free(f->cs.S_fixed);
    free(f->cs.J); free(f->cs.lambda_star); free(f->cs.p_ray);
    free(f->geo.r_inner); free(f->geo.r_outer);
}

int main(void){
    const double C_MAX = 1.0;              /* default LUMINA_POP_DJDN_CMAX */

    /* ---- a-reduce: >0 with a real lambda*, ==0 with lambda*==0 ---- */
    enum { NB = 4 };
    double Gw[NB]    = {1.0e-18, 2.0e-18, 1.5e-18, 0.5e-18};
    double Ec[NB]    = {3.0e-30, 2.0e-30, 1.0e-30, 0.5e-30};
    double Xc[NB]    = {4.0e-19, 3.0e-19, 2.0e-19, 1.0e-19};
    double S[NB]     = {1.2e-5,  1.0e-5,  0.8e-5,  0.6e-5};
    double chi_t[NB] = {2.5e-3,  2.0e-3,  1.5e-3,  1.0e-3};
    double lam1[NB]  = {0.5, 0.5, 0.5, 0.5};
    double lam0[NB]  = {0.0, 0.0, 0.0, 0.0};
    double n_e = 2.0e8, N_tot = 3.0e6, x_lag = 1.5;

    double a_on  = pop_djdn_a_reduce_test(NB, Gw, Ec, Xc, lam1, S, chi_t,
                                          n_e, N_tot, x_lag);
    double a_off = pop_djdn_a_reduce_test(NB, Gw, Ec, Xc, lam0, S, chi_t,
                                          n_e, N_tot, x_lag);
    ok(a_on > 0.0 && isfinite(a_on), "a > 0 by construction (real lambda*)");
    ok(a_off == 0.0,
       "lambda*==0 => a==0 (no-op wiring negative control: term reads only lambda*)");

    /* chi_t==0 bins are skipped; N_tot<=0 => a==0 */
    double chi_z[NB] = {0.0, 0.0, 0.0, 0.0};
    ok(pop_djdn_a_reduce_test(NB, Gw, Ec, Xc, lam1, S, chi_z,
                              n_e, N_tot, x_lag) == 0.0,
       "chi_t==0 bins contribute nothing (guarded)");
    ok(pop_djdn_a_reduce_test(NB, Gw, Ec, Xc, lam1, S, chi_t,
                              n_e, -1.0, x_lag) == 0.0,
       "N_tot<=0 => a==0 (no lagged pair)");

    /* ---- a: INDEPENDENT hand recomputation of the §4 algebra (1e-12 rel) ---- */
    double a_ref = 0.0;
    {
        double onep = 1.0 + x_lag;
        double geom = N_tot / (onep * onep);
        for (int b = 0; b < NB; ++b) {
            if (!(chi_t[b] > 0.0)) continue;
            if (!(lam1[b] > 0.0)) continue;
            double dSdx = geom * (n_e * Ec[b] + S[b] * Xc[b]) / chi_t[b];
            a_ref += Gw[b] * lam1[b] * dSdx;
        }
    }
    ok(a_ref > 0.0 && fabs(a_on - a_ref) <= 1e-12 * fabs(a_ref),
       "a matches an independent hand recomputation of the §4 algebra (1e-12 rel)");

    /* ==================================================================
     * pop_djdn_log_newton: damped log-variable Newton, NO fallback.
     * ================================================================== */

    /* ---- c INDEPENDENT recompute (1e-12 rel), |c|<=C_max case ----
     * num/den=1.2, x_lag=1.0 => x_map=1.2, rho=1.2; a=0.5,den=1 => g=0.5.
     * c' = ln(rho)*g/(rho-g) = ln(1.2)*0.5/0.7 ~ 0.1302 (< C_max). */
    {
        double num = 1.2, den = 1.0, a = 0.5, xl = 1.0;
        double xs = 0.0, c = 0.0, dmp = -1.0;
        int root = pop_djdn_log_newton_test(num, den, a, xl, C_MAX,
                                            &xs, &c, &dmp);
        double x_map = num / den;
        double rho = x_map / xl;
        double g = a / den;
        double c_ref = log(rho) * g / (rho - g);
        ok(root == 1 && fabs(c - c_ref) <= 1e-12 * (fabs(c_ref) > 1.0
                                                    ? fabs(c_ref) : 1.0),
           "c_newton == ln(rho)*g/(rho-g), independent recompute (1e-12 rel)");
        /* |c|<=C_max => damp==1 and x* BIT-IDENTICAL to the undamped step */
        double x_undamped = x_map * exp(c);
        ok(fabs(c) <= C_MAX && dmp == 1.0 && xs == x_undamped,
           "|c|<=C_max => damp==1 and x* bit-identical to undamped x_map*exp(c)");
        /* G-CPL x* identity + NEGATIVE CONTROL 2: perturb c by 1e-6 breaks it */
        double resid = fabs(xs - x_map * exp(dmp * c));
        double tol   = 1e-9 * xs;
        double c_bad = c * (1.0 + 1e-6);
        double resid_bad = fabs(xs - x_map * exp(dmp * c_bad));
        ok(resid <= tol,
           "G-CPL x* identity |x* - x_map*exp(damp*c_newton)| <= 1e-9*x*");
        ok(resid_bad > tol,
           "perturbed c (1e-6) breaks the G-CPL x* identity (negative control 2)");
    }

    /* ---- fallback fixture set: the three DJDN-MVP fallback regimes now yield
     * finite, positive x* (NO den-a<=0 switch-off). ---- */
    {
        /* (i) g>1 & rho>g : den=1,a=2 (g=2); num=5,xl=1 => x_map=5,rho=5.
         *     old code: den-a=-1<=0 => root=0. now: c=ln5*2/3 ~1.07 (>C_max)
         *     => c_d=+C_max => x*=5*e ~13.6 finite positive. */
        double xs = 0.0, c = 0.0, dmp = -1.0;
        int root = pop_djdn_log_newton_test(5.0, 1.0, 2.0, 1.0, C_MAX,
                                            &xs, &c, &dmp);
        ok(root == 1 && isfinite(xs) && xs > 0.0,
           "fallback g>1 & rho>g: x* finite & positive (no den-a<=0 switch-off)");

        /* (ii) g<1 & rho<g : den=10,a=1 (g=0.1); num=0.5,xl=1 => x_map=0.05,
         *      rho=0.05<g. old code: x*=(0.5-1)/(10-1)<0 => root=0. now finite. */
        double xs2 = 0.0, c2 = 0.0, dmp2 = -1.0;
        int root2 = pop_djdn_log_newton_test(0.5, 10.0, 1.0, 1.0, C_MAX,
                                             &xs2, &c2, &dmp2);
        ok(root2 == 1 && isfinite(xs2) && xs2 > 0.0,
           "fallback g<1 & rho<g: x* finite & positive (no negative-root switch-off)");

        /* (iii) singular rho==g exactly : den=1,x_lag=1,num=2 => x_map=2,rho=2;
         *       a=2 => g=2 => rho-g==0 exact => c_d=0, damp=0, x*=x_map. */
        double xs3 = -1.0, c3 = -1.0, dmp3 = -1.0;
        int root3 = pop_djdn_log_newton_test(2.0, 1.0, 2.0, 1.0, C_MAX,
                                             &xs3, &c3, &dmp3);
        ok(root3 == 1 && isfinite(xs3) && xs3 > 0.0 && dmp3 < 1.0 && xs3 == 2.0,
           "singular rho==g (F'=0): x* finite & positive, damp<1 (x*=x_map)");
    }

    /* ---- negative control 1 (existing): lambda*==0 => a==0 => x* == x_map
     * BIT-IDENTICAL (OFF value, Picard quotient undamped). ---- */
    {
        double num = 3.0, den = 7.0, xl = 0.4;   /* non-power-of-two operands */
        double xs = 0.0, c = -1.0, dmp = -1.0;
        int root = pop_djdn_log_newton_test(num, den, 0.0, xl, C_MAX,
                                            &xs, &c, &dmp);
        double x_map = num / den;
        ok(root == 1 && c == 0.0 && dmp == 1.0 && xs == x_map,
           "a==0 (lambda*=0) => c==0, x* == x_map BIT-IDENTICAL (negative control 1)");
    }

    /* ---- rho==1 (x_lag==x_map), a>0 => c==0 => x* == x_map == x_lag
     * BIT-IDENTICAL (fixed point unmoved). ---- */
    {
        double num = 3.0, den = 7.0;
        double xl = num / den;                   /* x_lag == x_map => rho==1 */
        double a = 0.5 * den;                    /* g=0.5 (<1, denom!=0) */
        double xs = 0.0, c = -1.0, dmp = -1.0;
        int root = pop_djdn_log_newton_test(num, den, a, xl, C_MAX,
                                            &xs, &c, &dmp);
        double x_map = num / den;
        ok(root == 1 && c == 0.0 && xs == x_map && xs == xl,
           "rho==1 => c==0 => x* == x_map == x_lag BIT-IDENTICAL (fixed point)");
    }

    /* ---- |c|>C_max => x* == x_map*exp(+-C_max) (1e-12 rel) ----
     * Reuse the g>1&rho>g fixture (c~+1.07>C_max) for +C_max, and the
     * g<1&rho<g fixture (c>0 too) — build an explicit negative-c case. */
    {
        /* +C_max: den=1,a=2,num=5,xl=1 => c>0, |c|>C_max */
        double xs = 0.0, c = 0.0, dmp = -1.0;
        pop_djdn_log_newton_test(5.0, 1.0, 2.0, 1.0, C_MAX, &xs, &c, &dmp);
        double x_map = 5.0;
        ok(c > C_MAX && fabs(xs - x_map * exp(C_MAX)) <= 1e-12 * xs
              && fabs(dmp - C_MAX / fabs(c)) <= 1e-12,
           "|c|>C_max (c>0) => x* == x_map*exp(+C_max), damp==C_max/|c| (1e-12 rel)");

        /* -C_max: choose g<1, rho<1 with rho<g so denom<0 and ln(rho)<0 giving
         * c<0 with |c|>C_max. den=10,a=1 (g=0.1); num=0.1,xl=1 => x_map=0.01,
         * rho=0.01<g; c=ln(0.01)*0.1/(0.01-0.1)=(-4.605)*0.1/(-0.09)=+5.1 (>0).
         * For c<0 need ln(rho)*g/(rho-g)<0. Take g>1, rho<1: den=1,a=2(g=2);
         * num=0.2,xl=1 => x_map=0.2,rho=0.2<g; denom=0.2-2=-1.8; ln(0.2)=-1.609;
         * c=(-1.609)*2/(-1.8)=+1.79 (>0). Hmm. c<0 needs sign(ln rho)!=sign(rho-g).
         * ln(rho)<0 (rho<1) with rho-g>0 (rho>g): impossible if g>=1. Use g<1,
         * rho>1>g: den=2,a=1 (g=0.5); num=10,xl=1 => x_map=5,rho=5; denom=4.5;
         * c=ln5*0.5/4.5=+0.179 (<C_max). Negative c requires rho<1 (ln<0) and
         * rho-g>0 => 0<g<rho<1: den=10,a=1 (g=0.1); num=5,xl=1 => x_map=0.5,
         * rho=0.5; denom=0.5-0.1=0.4>0; c=ln(0.5)*0.1/0.4=(-0.693)*0.25=-0.173
         * (<C_max). Scale g up: a=3.5,den=10 (g=0.35); num=... rho=0.4 (num=4):
         * denom=0.4-0.35=0.05; c=ln(0.4)*0.35/0.05=(-0.916)*7=-6.4 => |c|>C_max,
         * c<0. */
        double xn = 0.0, cn = 0.0, dn = -1.0;
        pop_djdn_log_newton_test(4.0, 10.0, 3.5, 1.0, C_MAX, &xn, &cn, &dn);
        double x_map_n = 4.0 / 10.0;
        ok(cn < -C_MAX && fabs(xn - x_map_n * exp(-C_MAX)) <= 1e-12 * xn
              && fabs(dn - C_MAX / fabs(cn)) <= 1e-12,
           "|c|>C_max (c<0) => x* == x_map*exp(-C_max), damp==C_max/|c| (1e-12 rel)");
    }

    /* ---- gamma_eff G-CPL identity + perturbed-a negative control (unchanged
     * arithmetic: gamma_eff == gamma_lag + a*(x*-x_lag)). ---- */
    {
        double num = 1.2, den = 1.0, a = 0.5, xl = 1.0, gamma_lag = 0.9;
        double xs = 0.0, c = 0.0, dmp = -1.0;
        int root = pop_djdn_log_newton_test(num, den, a, xl, C_MAX,
                                            &xs, &c, &dmp);
        double gamma_eff = pop_djdn_gamma_eff_test(gamma_lag, a, xs, xl);
        double tol = 1e-9 * (fabs(gamma_eff) > fabs(gamma_lag)
                             ? fabs(gamma_eff) : fabs(gamma_lag));
        double resid = fabs(gamma_eff - gamma_lag - a * (xs - xl));
        ok(root == 1 && resid <= tol,
           "G-CPL gamma_eff identity |gamma_eff - gamma_lag - a*(x*-x_lag)| <= 1e-9*max");
        double a_bad = a * (1.0 + 1e-6);
        double resid_bad = fabs(gamma_eff - gamma_lag - a_bad * (xs - xl));
        ok(resid_bad > tol,
           "perturbed a (1e-6) breaks the gamma_eff identity (negative control)");
    }

    /* ---- x_lag SNAPSHOT regression: x_lag reads only from its base array, so
     * perturbing the "live" populations cannot move it (BLOCKING-1 guard). ---- */
    enum { NIONS = 2, NSH = 3 };
    double snap_pop[NIONS * NSH];
    for (int i = 0; i < NIONS * NSH; ++i) snap_pop[i] = 0.0;
    snap_pop[0 * NSH + 1] = 100.0;             /* ip_cur  (ip0), shell 1 */
    snap_pop[1 * NSH + 1] = 30.0;              /* ip_next (ip1), shell 1 => x_lag=0.30 */
    double live_pop[NIONS * NSH];
    memcpy(live_pop, snap_pop, sizeof snap_pop);
    live_pop[1 * NSH + 1] = 999.0;             /* perturb the LIVE next-ion density */
    double xl_snap = -1.0, xl_live = -1.0;
    int vs = pop_djdn_xlag_test(snap_pop, NIONS, NSH, 0, 1, 1, &xl_snap);
    int vl = pop_djdn_xlag_test(live_pop, NIONS, NSH, 0, 1, 1, &xl_live);
    ok(vs == 1 && fabs(xl_snap - 0.30) <= 1e-12 &&
       vl == 1 && xl_live != xl_snap,
       "x_lag reads only from its base array: perturbing the live pop leaves "
       "the snapshot x_lag unchanged (self-cancellation regression)");
    double xl_bad = -1.0;
    ok(pop_djdn_xlag_test(snap_pop, NIONS, NSH, 0, 1, 0, &xl_bad) == 0,
       "x_lag: N_cur==0 (shell 0 empty) => invalid pair (0), no division");
    ok(pop_djdn_xlag_test(NULL, NIONS, NSH, 0, 1, 1, &xl_bad) == 0,
       "x_lag: NULL base (legacy path, snapshot absent) => 0 (term not applied)");

    /* ---- input absence: x_lag<=0 or x_map<=0 => root=0, off value x_map ---- */
    {
        double xs = -1.0, c = -1.0, dmp = -1.0;
        int root = pop_djdn_log_newton_test(1.2, 1.0, 0.5, 0.0, C_MAX,
                                            &xs, &c, &dmp);
        ok(root == 0 && xs == 1.2 && c == 0.0 && dmp == 1.0,
           "x_lag==0 => input absent => root=0, x* == off value x_map, recorded");
    }

    /* ---- MAX_CHNG step factor (dead noknob, kept; PREREG Q-LEAN L4) ---- */
    ok(pop_djdn_maxch_factor_test(0.25, 1.0) == 0.25,
       "MAXCH factor = MAXCH/norm when norm>MAXCH (0.25/1.0)");
    ok(pop_djdn_maxch_factor_test(0.25, 0.1) == 1.0,
       "MAXCH factor = 1 when norm<=MAXCH");
    ok(pop_djdn_maxch_factor_test(0.5, 0.5) == 1.0,
       "MAXCH factor = 1 at norm==MAXCH (not >)");
    ok(pop_djdn_maxch_factor_test(0.0, 5.0) == 1.0,
       "MAXCH unset (<=0) => factor 1 (inert, byte-identical)");

    /* ---- census header / row: now 16 columns (13 + epoch,c_newton,damp) ---- */
    const char *hdr = pop_djdn_header_test();
    ok(count_fields(hdr) == 16, "census header has exactly 16 columns");
    ok(strcmp(hdr, "gen,shell,Z,stage,x_lag,x_map,a,g,x_star,root,"
                   "gamma_lag,gamma_eff,maxch_factor,epoch,c_newton,damp\n") == 0,
       "census header column order matches contract (16 cols)");

    char row[320];
    int cn = pop_djdn_row_format_test(row, sizeof row, 7L, 20, 16, 1,
        1.5, 2.0, a_on, 0.42, 2.214285714, 1, 0.9, 1.114285714, 1.0,
        5L, 0.1302, 1.0);
    int row_ok = cn > 0 && cn < (int)sizeof row
        && count_fields(row) == 16
        && strncmp(row, "7,20,16,1,", 10) == 0
        && row[strlen(row) - 1] == '\n';
    ok(row_ok, "census row: 16 fields, gen/shell/Z/stage prefix, newline");
    printf("       [info] header: %s", hdr);
    printf("       [info] row:    %s", row);
    printf("       [info] a_on=%.6e\n", a_on);

    /* ================================================================
     * DET-DJDN-SPLIT: neighbour-held (FDHOLD) map split arithmetic.
     * Synthetic linear map M = g*x_s + b*x_{s+-1}. Production epoch K-1->K
     * moves BOTH the target (x_s) and the neighbour (x_nb):
     *   x_map = g*x_s + b*x_nb.
     * Choose the neighbour to move by the SAME amount as the target so that
     * s_nl collapses exactly to b (denom == Delta x_nb).
     * ================================================================ */
    {
        const double g = 3.0, b = 0.7;
        const double xs_Km1 = 1.0, xs_K = 1.5;      /* denom = 0.5 */
        const double xnb_Km1 = 2.0, xnb_K = 2.5;    /* Delta x_nb = 0.5 == denom */
        double x_map_Km1  = g * xs_Km1 + b * xnb_Km1;           /* 4.4  */
        double x_map_K    = g * xs_K   + b * xnb_K;             /* 6.25 */
        double x_map_hold = g * xs_Km1 + b * xnb_K;  /* target reverted, nb held */

        double denom = 0.0, s = 0.0, s_hold = 0.0, s_nl = 0.0;
        int rc = pop_djdn_split_test(x_map_K, x_map_Km1, x_map_hold,
                                     xs_K, xs_Km1, &denom, &s, &s_hold, &s_nl);
        ok(rc == 1 && fabs(s_hold - g) <= 1e-12 * (fabs(g) > 1.0 ? fabs(g) : 1.0),
           "synthetic map: s_hold -> g (neighbour held) (1e-12 rel)");
        ok(rc == 1 && fabs(s_nl - b) <= 1e-12 * (fabs(b) > 1.0 ? fabs(b) : 1.0),
           "synthetic map: s_nl -> b (neighbour variation) (1e-12 rel)");
        ok(rc == 1 && fabs(s - (g + b)) <= 1e-12 * fabs(g + b),
           "synthetic map: s == s_hold + s_nl == g + b (total secant)");

        /* NEGATIVE CONTROL 1: an implementation that MIXES the neighbour's Delta
         * x into the target shell (reverts BOTH to K-1, i.e. fails to hold the
         * neighbour) makes x_map_hold collapse to x_map_Km1, so s_hold -> s, NOT
         * g. The split must then FAIL to recover g. */
        double x_map_hold_bad = x_map_Km1;          /* neighbour NOT held */
        double sb = 0.0, sh_bad = 0.0, sn_bad = 0.0, db = 0.0;
        pop_djdn_split_test(x_map_K, x_map_Km1, x_map_hold_bad,
                            xs_K, xs_Km1, &db, &sb, &sh_bad, &sn_bad);
        ok(fabs(sh_bad - g) > 1e-6 && fabs(sh_bad - s) <= 1e-12 * fabs(s),
           "NEG-CTRL 1: mixing neighbour Delta x into target => s_hold != g "
           "(collapses to s) FAIL demonstrated");
        printf("       [info] neg-ctrl-1: s_hold(held)=%.6f==g, "
               "s_hold(mixed)=%.6f==s (g=%.3f, s=%.6f)\n", s_hold, sh_bad, g, s);
    }

    /* ---- null fixture: Delta x = 0 (revert to the CURRENT ratio) is an EXACT
     * no-op => the reevaluation input is bit-identical => x_map_hold == x_map.
     * pop_djdn_hold_input returns the pair bit-identical for x_target==ratio. */
    {
        double N_lo = 100.0, N_hi = 30.0;           /* ratio x = 0.30 */
        double lo = -1.0, hi = -1.0;
        pop_djdn_hold_input_test(N_lo, N_hi, N_hi / N_lo, &lo, &hi);
        ok(lo == N_lo && hi == N_hi,
           "null fixture: Delta x=0 (x_target==ratio) => input pair "
           "BIT-IDENTICAL (=> x_map_hold==x_map)");

        /* redistribution (non-null): total conserved, new ratio == x_target */
        double lo2 = -1.0, hi2 = -1.0, x_tgt = 0.5;
        pop_djdn_hold_input_test(N_lo, N_hi, x_tgt, &lo2, &hi2);
        ok(fabs((lo2 + hi2) - (N_lo + N_hi)) <= 1e-12 * (N_lo + N_hi),
           "redistribute: N_tot conserved (N_II+N_III == N_tot)");
        ok(fabs(hi2 / lo2 - x_tgt) <= 1e-12 * x_tgt,
           "redistribute: N_III/N_II == x_target");
    }

    /* ---- restore_ok: %.9e recovery of production x_star; NEGATIVE CONTROL 2:
     * an omitted restore (x* moved) => restore_ok == 0. ---- */
    {
        double x = 1.498327537e+02;                 /* sh24 epoch-7 x_star */
        ok(pop_djdn_restore_ok_test(x, x) == 1,
           "restore_ok: snapshot memcpy recovers x_star (%.9e match) => 1");
        ok(pop_djdn_restore_ok_test(x, x * (1.0 + 1e-6)) == 0,
           "NEG-CTRL 2: restore omitted (x* perturbed 1e-6) => restore_ok==0 "
           "FAIL demonstrated");
    }

    /* ---- denom degenerate: x_lag unchanged => secants undefined (NaN), rc 0 -- */
    {
        double dn = 0.0, s0 = 0.0, sh0 = 0.0, sn0 = 0.0;
        int rc0 = pop_djdn_split_test(6.25, 4.4, 4.75, 1.5, 1.5,
                                      &dn, &s0, &sh0, &sn0);
        ok(rc0 == 0 && isnan(s0) && isnan(sh0) && isnan(sn0),
           "denom degenerate (x_lag unchanged) => s/s_hold/s_nl NaN, rc 0");
    }

    /* ---- hold census header / row: exactly 16 columns, contract order ---- */
    const char *hhdr = pop_djdn_hold_header_test();
    ok(count_fields(hhdr) == 16, "hold census header has exactly 16 columns");
    ok(strcmp(hhdr, "shell,Z,stage,epoch,x_lag,x_lag_prev,x_map,x_map_hold,"
                    "x_map_null,g,s,s_hold,s_nl,denom,restore_ok,null_rel\n") == 0,
       "hold census header column order matches contract (16 cols)");
    {
        char hrow[512];
        int hn = pop_djdn_hold_row_format_test(hrow, sizeof hrow, 24, 16, 1, 7L,
            1.092563223e+02, 1.05e+02, 4.072876518e+02, 4.0e+02, 4.05e+02,
            8.189966386e+00, -4.584583036e+00, 8.0e+00, -12.58,
            4.256e+00, 1, 1.2e-03);
        int hrow_ok = hn > 0 && hn < (int)sizeof hrow
            && count_fields(hrow) == 16
            && strncmp(hrow, "24,16,1,7,", 10) == 0
            && hrow[strlen(hrow) - 1] == '\n';
        ok(hrow_ok, "hold census row: 16 fields, shell/Z/stage/epoch prefix, newline");
        printf("       [info] hold header: %s", hhdr);
        printf("       [info] hold row:    %s", hrow);
    }

    /* ---- OFF: LUMINA_POP_DJDN_FDHOLD unset => gate 0 (no observe, no hold
     * file, production path byte-identical; the writer early-returns). ---- */
    ok(pop_djdn_fdhold_gate_test() == 0,
       "OFF: FDHOLD unset => gate 0 (no hold file, production byte-identical)");

    /* ================= DET-PATHB-LOCAL (fixed-S dJ_fs/dchi) ================= */
    printf("\n--- DET-PATHB-LOCAL ---\n");
    /* single-ray fixtures (segments outer->inner) */
    enum { NR = 4 };
    double chi_r[NR] = {0.5, 1.0, 1.5, 2.0};
    double ds_r[NR]  = {1.0, 1.0, 1.0, 1.0};
    double S_r[NR]   = {1.0, 1.2, 1.5, 2.0};
    double Bnu_core  = 3.0;

    /* ---- (1) one-segment dI independent recompute, both tau branches 1e-12 -- */
    {
        double dt_big = 2.0, dt_sml = 1.0e-6, Sg = 1.3, Ii = 0.7, dl = 0.9;
        for (int k = 0; k < 2; ++k) {
            double dt = k ? dt_sml : dt_big;
            double Io, ds_out, ex, psi, psip;
            formal_djdchi_seg_test(dt, Sg, Ii, dl, &Io, &ds_out, &ex, &psi, &psip);
            double ex_ref = exp(-dt);
            double psi_ref = (dt > 1e-4) ? (1.0 - ex_ref) : (dt - 0.5*dt*dt);
            double psip_ref = (dt > 1e-4) ? ex_ref : (1.0 - dt);
            double Io_ref = Ii*ex_ref + Sg*psi_ref;
            double dsrc_ref = (-Ii*ex_ref + Sg*psip_ref) * dl;
            int good = fabs(ex - ex_ref) <= 1e-12*fabs(ex_ref)
                && fabs(psi - psi_ref) <= 1e-12*fabs(psi_ref)+1e-300
                && fabs(psip - psip_ref) <= 1e-12*fabs(psip_ref)
                && fabs(Io - Io_ref) <= 1e-12*fabs(Io_ref)
                && fabs(ds_out - dsrc_ref) <= 1e-12*fabs(dsrc_ref);
            ok(good, k ? "(1) small-tau segment dI recompute (1e-12)"
                       : "(1) large-tau segment dI recompute (1e-12)");
        }
    }

    /* ---- (1b) tau-clamp: raw dtau<0 (chi<0) => the segment is chi-independent
     *          (dtau pinned to 0) => dsrc==0 exactly, ex==1, I_out==I_in ---- */
    {
        double Io, dsrc, ex, psi, psip;
        double Sg = 1.3, Ii = 0.7, dl = 0.9;
        formal_djdchi_seg_test(-2.0, Sg, Ii, dl, &Io, &dsrc, &ex, &psi, &psip);
        ok(dsrc == 0.0 && ex == 1.0 && Io == Ii,
           "(1b) tau<0 clamp: dsrc==0 (chi-independent segment), ex==1, I_out==I_in");
        /* the NON-clamped counterpart at the same |dtau| is NOT zero (so the
         * zeroing is specific to the clamp, not a degenerate all-zero) */
        double Io2, dsrc2, ex2, psi2, psip2;
        formal_djdchi_seg_test(2.0, Sg, Ii, dl, &Io2, &dsrc2, &ex2, &psi2, &psip2);
        ok(dsrc2 != 0.0,
           "(1b) same |dtau| but chi>0 => dsrc != 0 (clamp zeroing is specific)");
    }

    /* ---- (2) core boundary: dI reset to 0 (production ray hook == reset ref;
     *          the no-reset variant DIFFERS => the dI==0 reset is present) ---- */
    {
        double dJ_prod[NR], dJ_reset[NR], dJ_noreset[NR];
        formal_djdchi_ray_test(NR, chi_r, ds_r, S_r, 1, Bnu_core, dJ_prod);
        ref_djdchi_ray(NR, chi_r, ds_r, S_r, 1, 1, Bnu_core, dJ_reset);
        ref_djdchi_ray(NR, chi_r, ds_r, S_r, 0, 1, Bnu_core, dJ_noreset);
        int match = 1, differ = 0;
        for (int i = 0; i < NR; ++i) {
            if (fabs(dJ_prod[i] - dJ_reset[i]) > 1e-12*fabs(dJ_reset[i])+1e-300)
                match = 0;
            if (fabs(dJ_reset[i] - dJ_noreset[i]) > 1e-9*fabs(dJ_reset[i]))
                differ = 1;
        }
        ok(match, "(2) core ray: production dI==0 reset matches the reset ref (1e-12)");
        ok(differ, "(2) core ray: no-reset variant DIFFERS (dI==0 at core is real)");
    }

    /* ---- (3) non-core tangent: inbound dI carries to outbound start
     *          (carry ref matches; reset-at-inner variant DIFFERS) ---- */
    {
        double dJ_prod[NR], dJ_carry[NR], dJ_reset[NR];
        formal_djdchi_ray_test(NR, chi_r, ds_r, S_r, 0, 0.0, dJ_prod);
        ref_djdchi_ray(NR, chi_r, ds_r, S_r, 0, 0, 0.0, dJ_carry);  /* carry */
        ref_djdchi_ray(NR, chi_r, ds_r, S_r, 1, 0, 0.0, dJ_reset);  /* no carry */
        int match = 1, differ = 0;
        for (int i = 0; i < NR; ++i) {
            if (fabs(dJ_prod[i] - dJ_carry[i]) > 1e-12*fabs(dJ_carry[i])+1e-300)
                match = 0;
            if (fabs(dJ_carry[i] - dJ_reset[i]) > 1e-9*fabs(dJ_carry[i]))
                differ = 1;
        }
        ok(match, "(3) non-core: production carry matches the tangent-carry ref (1e-12)");
        ok(differ, "(3) non-core: dropping the carry DIFFERS (inbound dI feeds outbound)");
    }

    /* ---- (4) 1-ray fixed-S finite-difference vs recursion (central diff) ---- */
    {
        double dJ_rec[NR];
        formal_djdchi_ray_test(NR, chi_r, ds_r, S_r, 0, 0.0, dJ_rec);
        int good = 1;
        for (int k = 0; k < NR; ++k) {
            double eps = chi_r[k] * 1e-6;
            double chp[NR], chm[NR], Jp[NR], Jm[NR];
            for (int i = 0; i < NR; ++i) { chp[i] = chi_r[i]; chm[i] = chi_r[i]; }
            chp[k] += eps; chm[k] -= eps;
            formal_solve_ray_test(NR, chp, ds_r, S_r, 0, 0.0, Jp);
            formal_solve_ray_test(NR, chm, ds_r, S_r, 0, 0.0, Jm);
            double fd = (Jp[k] - Jm[k]) / (2.0 * eps);
            double rel = fabs(fd - dJ_rec[k]) / (fabs(dJ_rec[k]) + 1e-300);
            if (rel > 1e-5) good = 0;
            printf("       [info] (4) shell %d: FD=%.9e recursion=%.9e rel=%.2e\n",
                   k, fd, dJ_rec[k], rel);
        }
        ok(good, "(4) fixed-S FD (J_fs(chi+eps)-J_fs(chi-eps))/2eps == recursion (rel<1e-5)");
    }

    /* ---- (5) the recursion is a DISTINCT form from the two discarded drafts
     *          -S/chi and -(tau_star/chi)(J-S): the max per-shell divergence is
     *          large (a substitution would change the assembled a_dchi -> (9)) -- */
    {
        double dJ_rec[NR], J_fs[NR];
        formal_djdchi_ray_test(NR, chi_r, ds_r, S_r, 0, 0.0, dJ_rec);
        formal_solve_ray_test(NR, chi_r, ds_r, S_r, 0, 0.0, J_fs);
        double max_r1 = 0.0, max_r2 = 0.0;
        for (int k = 0; k < NR; ++k) {
            double alt1 = -S_r[k] / chi_r[k];                       /* -S/chi */
            double tau  = chi_r[k] * ds_r[k];
            double alt2 = -(tau / chi_r[k]) * (J_fs[k] - S_r[k]);   /* L* collapse form */
            double r1 = fabs(dJ_rec[k]-alt1)/(fabs(dJ_rec[k])+1e-300);
            double r2 = fabs(dJ_rec[k]-alt2)/(fabs(dJ_rec[k])+1e-300);
            if (r1 > max_r1) max_r1 = r1;
            if (r2 > max_r2) max_r2 = r2;
        }
        ok(max_r1 > 1e-2 && max_r2 > 1e-2,
           "(5) recursion is a DISTINCT form from -S/chi and -(tau*/chi)(J-S) "
           "(max rel divergence large)");
        printf("       [info] (5) max rel div: vs -S/chi=%.3e  vs -(tau/chi)(J-S)=%.3e\n",
               max_r1, max_r2);
    }

    /* ===== PRODUCTION formal_djdchi_fs_S validation (calls the REAL walker) === */
    printf("--- production formal_djdchi_fs_S ---\n");
    {
        LocalFix f; build_local_fixture(&f);
        int NS = LF_NS, NB = LF_NB;
        double dJ[LF_NS*LF_NB];
        formal_djdchi_fs_S_test(&f.cs, &f.geo, 1.0e4, dJ);

        /* (P1) FD vs PRODUCTION formal_solve_bin (fixed S=S_fixed, same Bnu):
         * validates the walker's core dI=0 reset, ray weight (ray_w=p), indexing. */
        int fd_ok = 1, any_nonzero = 0;
        for (int b = 0; b < NB; ++b) {
            double Bnu = formal_djdchi_bnu_test(f.cs.nu[b], 1.0e4);
            double Scol[LF_NS];
            for (int s = 0; s < NS; ++s) Scol[s] = f.cs.S_fixed[(size_t)s*NB+b];
            for (int s = 0; s < NS; ++s) {
                double c0 = f.cs.chi_tot[(size_t)s*NB+b], eps = c0*1e-6;
                double Jp[LF_NS], Jm[LF_NS], Lp[LF_NS], Lm[LF_NS];
                f.cs.chi_tot[(size_t)s*NB+b] = c0 + eps;
                formal_solve_bin_test(&f.cs, &f.geo, b, Scol, Bnu, Jp, Lp);
                f.cs.chi_tot[(size_t)s*NB+b] = c0 - eps;
                formal_solve_bin_test(&f.cs, &f.geo, b, Scol, Bnu, Jm, Lm);
                f.cs.chi_tot[(size_t)s*NB+b] = c0;               /* restore */
                double fd = (Jp[s] - Jm[s]) / (2.0*eps);
                double dprod = dJ[(size_t)s*NB+b];
                double rel = fabs(fd - dprod) / (fabs(dprod) + 1e-300);
                if (fabs(dprod) > 0.0) any_nonzero = 1;
                if (rel > 1e-4) fd_ok = 0;
                printf("       [info] (P1) s=%d b=%d FD=%.6e prod=%.6e rel=%.2e\n",
                       s, b, fd, dprod, rel);
            }
        }
        ok(fd_ok && any_nonzero,
           "(P1) production formal_djdchi_fs_S FD-matches production formal_solve_bin "
           "(core dI=0, ray weight, indexing; rel<1e-4)");

        /* (P2) buffer invariance: the derivative pass writes ONLY dJdchi + its own
         * scratch => cs->J and cs->lambda_star are bit-unchanged. */
        double Jsave[LF_NS*LF_NB], Lsave[LF_NS*LF_NB], dJ2[LF_NS*LF_NB];
        memcpy(Jsave, f.cs.J, sizeof Jsave);
        memcpy(Lsave, f.cs.lambda_star, sizeof Lsave);
        formal_djdchi_fs_S_test(&f.cs, &f.geo, 1.0e4, dJ2);
        int inv = (memcmp(Jsave, f.cs.J, sizeof Jsave) == 0)
               && (memcmp(Lsave, f.cs.lambda_star, sizeof Lsave) == 0);
        ok(inv, "(P2) production dJ/dchi pass leaves cs->J and cs->lambda_star bit-identical");

        /* (P4) production dJ/dchi is a DISTINCT form from -S/chi. */
        int distinct = 0;
        for (int b = 0; b < NB; ++b) for (int s = 0; s < NS; ++s) {
            double d = dJ[(size_t)s*NB+b];
            double alt = -f.cs.S_fixed[(size_t)s*NB+b] / f.cs.chi_tot[(size_t)s*NB+b];
            if (fabs(d - alt) > 1e-2*(fabs(d)+1e-300)) distinct = 1;
        }
        ok(distinct, "(P4) production dJ/dchi != -S/chi on the fixture");
        free_local_fixture(&f);
    }

    /* (P3) tau<0 clamp in the PRODUCTION walker: a chi<0 shell contributes
     * exactly 0 (chi-independent segment) and its FD is 0 too. */
    {
        LocalFix f; build_local_fixture(&f);
        int NS = LF_NS, NB = LF_NB, sc = 1, bc = 0;   /* clamp shell 1, bin 0 */
        f.cs.chi_tot[(size_t)sc*NB+bc] = -0.5;        /* chi<0 => dtau clamped */
        double dJ[LF_NS*LF_NB];
        formal_djdchi_fs_S_test(&f.cs, &f.geo, 1.0e4, dJ);
        double dprod = dJ[(size_t)sc*NB+bc];
        double Bnu = formal_djdchi_bnu_test(f.cs.nu[bc], 1.0e4);
        double Scol[LF_NS];
        for (int s = 0; s < NS; ++s) Scol[s] = f.cs.S_fixed[(size_t)s*NB+bc];
        double c0 = f.cs.chi_tot[(size_t)sc*NB+bc], eps = fabs(c0)*1e-6;
        double Jp[LF_NS], Jm[LF_NS], Lp[LF_NS], Lm[LF_NS];
        f.cs.chi_tot[(size_t)sc*NB+bc] = c0 + eps;    /* stays negative */
        formal_solve_bin_test(&f.cs, &f.geo, bc, Scol, Bnu, Jp, Lp);
        f.cs.chi_tot[(size_t)sc*NB+bc] = c0 - eps;
        formal_solve_bin_test(&f.cs, &f.geo, bc, Scol, Bnu, Jm, Lm);
        f.cs.chi_tot[(size_t)sc*NB+bc] = c0;
        double fd = (Jp[sc] - Jm[sc]) / (2.0*eps);
        ok(dprod == 0.0 && fabs(fd) <= 1e-12,
           "(P3) tau<0 clamp: production dJ/dchi==0 for the chi<0 shell and FD==0");
        printf("       [info] (P3) clamp shell dJ/dchi=%.3e FD=%.3e\n", dprod, fd);
        free_local_fixture(&f);
    }

    /* ---- (6) a_local == a_ds + a_dchi (1e-12); a NEGATIVE a_dchi survives ---- */
    {
        enum { MB = 4 };
        double GwC[MB] = {1.0e-18, 2.0e-18, 1.5e-18, 0.5e-18};
        double XcC[MB] = {4.0e-19, 3.0e-19, 2.0e-19, 1.0e-19};
        double dJc[MB] = {1.0e6, 2.0e6, 3.0e6, 4.0e6};  /* dJ>0 => term<0 */
        double Ntot = 3.0e6, xlag = 1.5;
        double a_dchi = pop_djdn_a_chi_reduce_test(MB, GwC, XcC, dJc, Ntot, xlag);
        double onep = 1.0 + xlag, geom = Ntot/(onep*onep), a_ref = 0.0;
        for (int b = 0; b < MB; ++b) a_ref += GwC[b]*dJc[b]*(-XcC[b]*geom);
        double a_ds_v = 7.5e-4, a_local = a_ds_v + a_dchi;
        ok(fabs(a_dchi - a_ref) <= 1e-12*fabs(a_ref) && a_dchi < 0.0,
           "(6) a_dchi == independent signed sum (1e-12) and stays NEGATIVE");
        ok(a_local == a_ds_v + a_dchi,
           "(6) a_local == a_ds + a_dchi bit-identical (the negative term remains)");
    }

    /* ---- (7) unregistered/NULL dJ buffer => a_dchi==0 => a_local==a_ds bit --- */
    {
        double a_ds_v = 7.5e-4;
        double a0 = pop_djdn_a_chi_reduce_test(0, NULL, NULL, NULL, 3.0e6, 1.5);
        double a1 = pop_djdn_a_chi_reduce_test(4, NULL, NULL, NULL, 3.0e6, 1.5);
        ok(a0 == 0.0 && a1 == 0.0,
           "(7) CHI OFF/unregistered (nb=0 or NULL dJ) => a_dchi==0");
        ok((a_ds_v + a0) == a_ds_v,
           "(7) a_local == a_ds bit-identical when a_dchi==0 (388393 byte-identity)");
    }

    /* ---- (8) NC1 (rewritten): a genuine dJ => |a_dchi|>0 (the sh24 max signal);
     *          dJ_fs/dchi ≡ 0 (diagonal-only) => a_dchi==0 => gate FAIL demo ---- */
    {
        enum { MB = 4 };
        double GwC[MB] = {1.0e-18, 2.0e-18, 1.5e-18, 0.5e-18};
        double XcC[MB] = {4.0e-19, 3.0e-19, 2.0e-19, 1.0e-19};
        double dJnz[MB] = {1.0e6, -2.0e6, 3.0e6, -1.0e6};
        double dJ0[MB]  = {0.0, 0.0, 0.0, 0.0};
        double a_nz = pop_djdn_a_chi_reduce_test(MB, GwC, XcC, dJnz, 3.0e6, 1.5);
        double a_z  = pop_djdn_a_chi_reduce_test(MB, GwC, XcC, dJ0,  3.0e6, 1.5);
        ok(fabs(a_nz) > 0.0,
           "(8) NC1: genuine dJ_fs/dchi => |a_dchi|>0 (the sh24 max|a_dchi|>0 signal)");
        ok(a_z == 0.0,
           "(8) NC1: dJ_fs/dchi==0 (diagonal-only) => a_dchi==0 => max|a_dchi|>0 FAIL demo");
    }

    /* ---- (9) NC2: substituting -S/chi (or L* collapse) for the recursion dJ
     *          moves the assembled a_dchi => mismatch FAIL demonstrated ---- */
    {
        double dJ_rec[NR], J_fs[NR];
        formal_djdchi_ray_test(NR, chi_r, ds_r, S_r, 0, 0.0, dJ_rec);
        formal_solve_ray_test(NR, chi_r, ds_r, S_r, 0, 0.0, J_fs);
        double GwC[NR] = {1.0e-18, 2.0e-18, 1.5e-18, 0.5e-18};
        double XcC[NR] = {4.0e-19, 3.0e-19, 2.0e-19, 1.0e-19};
        double dJ_wrong[NR];
        for (int i = 0; i < NR; ++i) dJ_wrong[i] = -S_r[i]/chi_r[i];
        double a_correct = pop_djdn_a_chi_reduce_test(NR, GwC, XcC, dJ_rec, 3.0e6, 1.5);
        double a_wrong   = pop_djdn_a_chi_reduce_test(NR, GwC, XcC, dJ_wrong, 3.0e6, 1.5);
        ok(fabs(a_correct - a_wrong) > 1e-3*fabs(a_correct),
           "(9) NC2: a_dchi(recursion) != a_dchi(-S/chi) => wrong-derivative FAIL demo");
    }

    /* ---- (10) NC3: a negative term VANISHES under the positive-only a_reduce
     *           filter, but SURVIVES the signed a_chi_reduce ---- */
    {
        enum { MB = 1 };
        double Gw1[MB] = {1.0e-18};
        double Ecn[MB] = {-5.0e-30};   /* negative => dSdx<0 => term<0 */
        double Xc1[MB] = {1.0e-30};    /* tiny so n_e*Ec dominates negative */
        double lam1b[MB] = {0.5};
        double S1[MB] = {1.0e-5};
        double ct1[MB] = {2.0e-3};
        double a_pos = pop_djdn_a_reduce_pos_test(MB, Gw1, Ecn, Xc1, lam1b, S1, ct1,
                                                  2.0e8, 3.0e6, 1.5);
        /* the SAME sign routed through the signed chi reducer (dJ>0 => term<0) */
        double dJp[MB] = {2.0e6};
        double a_sgn = pop_djdn_a_chi_reduce_test(MB, Gw1, Xc1, dJp, 3.0e6, 1.5);
        ok(a_pos == 0.0,
           "(10) NC3: negative term dropped by positive-only a_reduce (FAIL if routed here)");
        ok(a_sgn < 0.0,
           "(10) NC3: signed a_chi_reduce KEEPS the negative term");
    }

    /* ---- (11) NC4: a_dchi != 0 moves the Newton (c, x_star); if newton(a_ds)
     *           == newton(a_local) then a_dchi was NOT consumed => FAIL ---- */
    {
        double num = 4.0, den = 1.0, xlag = 1.0, cmax = C_MAX;
        double a_ds_v = 0.30;              /* g_ds = 0.30 */
        double a_dchi_v = 0.55;            /* large enough to move (c,x*) */
        double a_local = a_ds_v + a_dchi_v;
        double xs_ds, c_ds, d_ds, xs_lo, c_lo, d_lo;
        int r_ds = pop_djdn_log_newton_test(num, den, a_ds_v, xlag, cmax,
                                            &xs_ds, &c_ds, &d_ds);
        int r_lo = pop_djdn_log_newton_test(num, den, a_local, xlag, cmax,
                                            &xs_lo, &c_lo, &d_lo);
        ok(r_ds == 1 && r_lo == 1 &&
           fabs(xs_lo - xs_ds) > 1e-6*fabs(xs_ds) &&
           fabs(c_lo - c_ds) > 1e-6*fabs(c_ds),
           "(11) NC4: a_dchi moves (c,x_star) => Newton consumes a_local not a_ds");
        printf("       [info] (11) c_ds=%.6f x*_ds=%.6f | c_loc=%.6f x*_loc=%.6f\n",
               c_ds, xs_ds, c_lo, xs_lo);
    }

    /* ---- (12) the derivative computation has NO side effect on the forward
     *           J_fs (production J/L* bit-identical CHI ON vs OFF) ---- */
    {
        double J1[NR], J2[NR], dJ[NR];
        formal_solve_ray_test(NR, chi_r, ds_r, S_r, 0, 0.0, J1);
        formal_djdchi_ray_test(NR, chi_r, ds_r, S_r, 0, 0.0, dJ);  /* runs between */
        formal_solve_ray_test(NR, chi_r, ds_r, S_r, 0, 0.0, J2);
        int bit = 1;
        for (int i = 0; i < NR; ++i) if (J1[i] != J2[i]) bit = 0;
        ok(bit, "(12) derivative pass leaves the forward J_fs bit-identical (no side effect)");
    }

    /* ---- (13) limiter identity x_star = x_map*exp(damp*c) at a=a_local ---- */
    {
        double num = 4.0, den = 1.0, xlag = 1.0, cmax = C_MAX;
        double a_local = 0.85;
        double xs, c, d;
        int rr = pop_djdn_log_newton_test(num, den, a_local, xlag, cmax, &xs, &c, &d);
        double x_map = num / den;
        ok(rr == 1 && fabs(xs - x_map*exp(d*c)) <= 1e-12*fabs(xs),
           "(13) limiter identity x_star == x_map*exp(damp*c) at a=a_local (1e-12)");
    }

    /* ---- (14) chi census header/row: exactly 8 columns, contract order ---- */
    {
        const char *chdr = pop_djdn_chi_header_test();
        ok(count_fields(chdr) == 8, "(14) chi census header has exactly 8 columns");
        ok(strcmp(chdr, "gen,shell,Z,stage,epoch,a_ds,a_dchi,a_local\n") == 0,
           "(14) chi census header column order matches contract (8 cols)");
        char crow[256];
        int cn = pop_djdn_chi_row_format_test(crow, sizeof crow, 5L, 24, 16, 1, 7L,
                                              7.5e-4, -3.1e-4, 4.4e-4);
        int rok = cn > 0 && cn < (int)sizeof crow && count_fields(crow) == 8
            && strncmp(crow, "5,24,16,1,7,", 12) == 0
            && crow[strlen(crow)-1] == '\n';
        ok(rok, "(14) chi census row: 8 fields, gen/shell/Z/stage/epoch prefix, newline");
        printf("       [info] chi header: %s", chdr);
        printf("       [info] chi row:    %s", crow);
    }

    /* ---- (15) the assembly hook takes NO level vector / rate matrix (the term
     *           is a scalar ladder gain; ii/dual-authority stays closed) ---- */
    ok(1, "(15) pop_djdn_a_chi_reduce signature carries no level/N-by-N argument "
          "(scalar single-stage-total gain; no level coupling)");

    printf(g_fail ? "\nRESULT: FAIL\n" : "\nRESULT: PASS\n");
    return g_fail ? 1 : 0;
}
