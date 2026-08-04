#ifndef LUMINA_RADEQ_COL_PAIRS_H
#define LUMINA_RADEQ_COL_PAIRS_H
/* =====================================================================
 * withParityO — CMFGEN-faithful all-level-pair collisional bound-bound
 * cooling for ONE ion (the radeq thermal-ledger COL term).
 *
 * SINGLE SOURCE OF TRUTH shared verbatim by:
 *   - the runtime (src/lumina_plasma.c, gate LUMINA_RADEQ_COL_PAIRS), and
 *   - the offline certification bench (src/lumina_radeq_col_pairs_bench.c).
 *
 * Verbatim port of dig_F11 reproduce_gencool.py :: compute_ion()
 * (decode of CMFGEN subs/subcol_multi_v3.f + omega_gen_v2.f):
 *
 *   COOL = Sum_{i<j} (n_i*q_up - n_j*q_dn)*dE     [erg/cm^3/s], >0 = cooling
 *   q_up = COLK*ne*Om*exp(-X)/(g_lo*sqrt(T4)),  X = HDKT*(edge_lo-edge_hi)/T4
 *   q_dn = COLK*ne*Om/(g_hi*sqrt(T4)),          dE = HPL15*(edge_lo-edge_hi)
 *   Om(lo,hi): tabulated (split-J Omega, already log-log interp in T4) else
 *              van-Regemorter 47.972*scale*GBAR*f_lu*g_lo/FL (f_lu != 0) else
 *              OMEGA_SET (forbidden, f_lu == 0).
 *   GBAR = max( same-n?0.7:0.2 , 0.276*exp(X)*E1(X) ).
 *   NO beta-escape — CMFGEN's COL term has none; escape is carried by the
 *   NLTE populations that feed n_lo / n_hi (F10/F11 established).
 *
 * The build routine emits per-pair prefactors (a,b,beta) that the existing
 * radeq_line_cool consumer evaluates at the trial T_e in the bisection:
 *   COOL(T_e) = (ne/sqrt(T_e)) * Sum ( a*exp(-beta/T_e) - b ),  beta=HDKT*FL*1e4
 * with a = COLK6*dE*Om*n_lo/g_lo, b = COLK6*dE*Om*n_hi/g_hi, COLK6 = COLK*1e2
 * (absorbs the 1/sqrt(T4) -> 1/sqrt(T_e) rescale so exp(-beta/T_e)==exp(-X)).
 *
 * The "edge" array carries any quantity whose pairwise DIFFERENCE equals the
 * ionisation-frequency difference in 1e15 Hz (higher edge == lower energy).
 * The bench passes CMFGEN osc col-3 verbatim (bit-faithful to reproduce); the
 * runtime passes edge[k] = (BIG - E_eV[k])*EV_TO_ERG/HPL15 (constant BIG drops
 * out in every difference).
 * ===================================================================== */
#include <math.h>
#include <stdlib.h>

/* constants verbatim from reproduce_gencool.py (CMFGEN source values) */
#define RCP_COLK   8.63e-8            /* 8.63e-8 * ne * Om / (g*sqrt(T4))     */
#define RCP_COLK6  (RCP_COLK * 1.0e2) /* folds sqrt(T4)->sqrt(T_e): 8.63e-6   */
#define RCP_HDKT   4.7994145          /* h*1e15 / (k * 1e4)                   */
#define RCP_HPL15  6.62607015e-12     /* h * 1e15  [erg]                      */
#define RCP_VRCONST 47.972            /* (8pi/sqrt3)*Ry/(h*1e15) van-Reg      */

typedef struct {
    long   n_tab, n_vr, n_set;   /* pair counts by Omega source              */
    double c_tab, c_vr, c_set;   /* their cooling contribution at Tref [+cool]*/
    long   n_pairs;              /* total pairs emitted                       */
} RcpCensus;

/* exp(x)*E1(x), stable for all x>0 (Abramowitz & Stegun 5.1.53 / 5.1.56).
 * Only reached on the van-Regemorter fill path (< ~1% of the total). */
static inline double rcp_exe1(double x) {
    if (x <= 0.0) return 0.0;
    if (x < 1.0) {
        double e1 = -log(x) - 0.57721566
            + x * (0.99999193 + x * (-0.24991055 + x * (0.05519968
            + x * (-0.00976004 + x * 0.00107857))));
        return exp(x) * e1;
    }
    double num = x * x + 2.334733 * x + 0.250621;
    double den = x * x + 3.330657 * x + 1.681534;
    return num / (den * x);      /* == e^x * E1(x) */
}

/* Log-log interpolation of a tabulated Omega(T) vector, clamped to the ends.
 * Matches reproduce_gencool.py :: loglog_interp_scalar (abscissa units cancel,
 * so T/tgrid may be Kelvin or T4 as long as both share the same unit). Linear
 * fallback if any node is non-positive (log undefined). */
static inline double rcp_loglog(const double *tg, const double *om, int nt, double T) {
    if (nt < 1) return 0.0;
    if (nt == 1 || T <= tg[0])    return om[0];
    if (T >= tg[nt - 1])          return om[nt - 1];
    int L = 1; while (L < nt - 1 && T > tg[L]) L++;
    if (om[L] > 0.0 && om[L - 1] > 0.0 && tg[L] > 0.0 && tg[L - 1] > 0.0) {
        double a = log(om[L] / om[L - 1]) / log(tg[L] / tg[L - 1]);
        return om[L - 1] * pow(T / tg[L - 1], a);
    }
    return om[L - 1] + (om[L] - om[L - 1]) * (T - tg[L - 1]) / (tg[L] - tg[L - 1]);
}

/* Build (a,b,beta) prefactors for one ion's full pair census, appending to
 * a[]/b[]/beta[] starting at *n (caller guarantees capacity >= *n +
 * nlev*(nlev-1)/2). Om resolved at Tref4 = T_ref/1e4 (tab_om already at that
 * T4; vR GBAR uses X at Tref4). Returns per-source census evaluated at Tref.
 *
 *   nlev, edge[nlev], g[nlev], n_pop[nlev], pqn[nlev]  — level metadata
 *   pqn[k] = principal-quantum bucket for the same-n GBAR test (<0 = unknown
 *            -> treated as NOT same-n, the conservative 0.2 branch)
 *   tab_lo/tab_hi/tab_om[n_tab] — tabulated split-J Omega (energy-ordered:
 *            tab_lo = higher-edge/lower-energy index), already at Tref4
 *   f_lo/f_hi/f_val[n_f]        — oscillator strengths (energy-ordered)
 *   ne, oset, scale             — electron density, forbidden Omega, Omega scale
 * Returns 0 on success, -1 on allocation failure (caller: fail-loud). */
static inline int radeq_col_pairs_build(
        int nlev, const double *edge, const int *g,
        const double *n_pop, const int *pqn,
        double ne, double Tref4, double oset, double scale,
        int n_tab, const int *tab_lo, const int *tab_hi, const double *tab_om,
        int n_f,   const int *f_lo,   const int *f_hi,   const double *f_val,
        double *a, double *b, double *beta, long *n, RcpCensus *cen) {
    if (cen) { cen->n_tab = cen->n_vr = cen->n_set = 0;
               cen->c_tab = cen->c_vr = cen->c_set = 0.0; cen->n_pairs = 0; }
    if (nlev < 2) return 0;
    size_t nn = (size_t)nlev * (size_t)nlev;
    double *Om = (double *)malloc(nn * sizeof(double));
    double *fv = (double *)malloc(nn * sizeof(double));
    if (!Om || !fv) { free(Om); free(fv); return -1; }
    for (size_t i = 0; i < nn; i++) { Om[i] = -1.0; fv[i] = 0.0; }
    for (int t = 0; t < n_tab; t++) {
        int lo = tab_lo[t], hi = tab_hi[t];
        if (lo < 0 || hi < 0 || lo >= nlev || hi >= nlev) continue;
        Om[(size_t)lo * nlev + hi] = tab_om[t];   /* may legitimately be 0 */
    }
    for (int t = 0; t < n_f; t++) {
        int lo = f_lo[t], hi = f_hi[t];
        if (lo < 0 || hi < 0 || lo >= nlev || hi >= nlev) continue;
        fv[(size_t)lo * nlev + hi] = f_val[t];
    }
    double sqT4 = sqrt(Tref4);
    double Te   = Tref4 * 1.0e4;
    double sqTe = sqrt(Te);
    long   m    = *n;
    for (int i = 0; i < nlev; i++) {
        for (int j = i + 1; j < nlev; j++) {
            int lo, hi;
            if (edge[i] >= edge[j]) { lo = i; hi = j; } else { lo = j; hi = i; }
            double FL = edge[lo] - edge[hi];       /* 1e15 Hz, > 0 */
            if (!(FL > 0.0)) continue;
            double dE = RCP_HPL15 * FL;
            double X  = RCP_HDKT * FL / Tref4;
            double om = Om[(size_t)lo * nlev + hi];
            int    src;
            if (om >= 0.0) { src = 0; }            /* tabulated split-J */
            else {
                double f = fv[(size_t)lo * nlev + hi];
                if (f != 0.0) {                    /* van-Regemorter fill */
                    double g1 = (pqn && pqn[lo] >= 0 && pqn[lo] == pqn[hi]) ? 0.7 : 0.2;
                    double g2 = 0.276 * rcp_exe1(X);
                    double gbar = g1 > g2 ? g1 : g2;
                    om = RCP_VRCONST * scale * gbar * f * (double)g[lo] / FL;
                    src = 1;
                } else { om = oset; src = 2; }     /* forbidden default */
            }
            double coeff = RCP_COLK6 * om;
            double av = coeff * dE * n_pop[lo] / (double)g[lo];
            double bv = coeff * dE * n_pop[hi] / (double)g[hi];
            a[m] = av; b[m] = bv; beta[m] = RCP_HDKT * FL * 1.0e4;
            if (cen) {
                double term = (ne / sqTe) * (av * exp(-X) - bv);   /* == reproduce */
                (void)sqT4;
                if      (src == 0) { cen->n_tab++; cen->c_tab += term; }
                else if (src == 1) { cen->n_vr++;  cen->c_vr  += term; }
                else               { cen->n_set++; cen->c_set += term; }
            }
            m++;
        }
    }
    if (cen) cen->n_pairs = m - *n;
    *n = m;
    free(Om); free(fv);
    return 0;
}

#endif /* LUMINA_RADEQ_COL_PAIRS_H */
