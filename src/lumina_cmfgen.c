/* ============================================================ */
/* lumina_cmfgen.c                                             */
/*                                                              */
/* Pure-CMFGEN deterministic radiation field (see header).      */
/* Coexists with the MC path; nothing here is reachable unless   */
/* LUMINA_PURE_CMFGEN=1 dispatches cmfgen_run() from main.       */
/* ============================================================ */
#include "lumina_cmfgen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Physical constants (cgs) — local copies; planck_bnu in plasma.c is static. */
#define CM_H      6.62607015e-27   /* erg s            */
#define CM_KB     1.380649e-16     /* erg/K            */
#define CM_C      2.99792458e10    /* cm/s             */
#define CM_SIGMA_T 6.6524587e-25   /* Thomson cm^2     */

static inline double cm_planck(double nu, double T) {
    if (T <= 0.0) return 0.0;
    double x = CM_H * nu / (CM_KB * T);
    if (x > 7.0e2) return 0.0;            /* underflow guard */
    double denom = expm1(x);
    if (denom <= 0.0) return 0.0;
    return (2.0 * CM_H * nu * nu * nu) / (CM_C * CM_C * denom);
}

/* ------------------------------------------------------------ */
int cmfgen_init(CMFGENState *cs, const Geometry *geo)
{
    memset(cs, 0, sizeof(*cs));
    cs->n_shells = geo->n_shells;
    cs->n_bins   = NLTE_N_FREQ_BINS;
    cs->nu_min   = NLTE_NU_MIN;
    cs->nu_max   = NLTE_NU_MAX;
    cs->d_log_nu = log(cs->nu_max / cs->nu_min) / (double)cs->n_bins;

    int NB = cs->n_bins, NS = cs->n_shells;
    cs->nu          = malloc(sizeof(double) * NB);
    cs->dnu         = malloc(sizeof(double) * NB);
    cs->chi_es      = calloc((size_t)NS * NB, sizeof(double));
    cs->chi_abs     = calloc((size_t)NS * NB, sizeof(double));
    cs->chi_line    = calloc((size_t)NS * NB, sizeof(double));
    cs->chi_line_th = calloc((size_t)NS * NB, sizeof(double));
    cs->chi_line_cls= calloc((size_t)NS * NB, sizeof(double));
    cs->chi_tot     = calloc((size_t)NS * NB, sizeof(double));
    cs->S_fixed     = calloc((size_t)NS * NB, sizeof(double));
    cs->J           = calloc((size_t)NS * NB, sizeof(double));
    cs->lambda_star = calloc((size_t)NS * NB, sizeof(double));
    cs->t_color     = calloc((size_t)NS, sizeof(double));
    cs->tri_lo      = calloc((size_t)NS * NB, sizeof(double));
    cs->tri_up      = calloc((size_t)NS * NB, sizeof(double));
    cs->tri_r       = calloc((size_t)NS * NB, sizeof(double));
    if (!cs->nu || !cs->dnu || !cs->chi_es || !cs->chi_abs || !cs->chi_line ||
        !cs->chi_line_th || !cs->t_color ||
        !cs->tri_lo || !cs->tri_up || !cs->tri_r ||
        !cs->chi_tot || !cs->S_fixed || !cs->J || !cs->lambda_star) {
        fprintf(stderr, "[CMFGEN] init alloc failed\n");
        return -1;
    }
    for (int b = 0; b < NB; ++b) {
        cs->nu[b]  = cs->nu_min * exp((b + 0.5) * cs->d_log_nu);
        cs->dnu[b] = cs->nu[b] * cs->d_log_nu;   /* log-grid bin width */
    }

    /* Tangent rays: one grazing each shell's r_outer, plus core rays packed
     * inside r_inner[0] for the diffusive inner boundary. */
    int n_core = 8;
    cs->n_rays = NS + n_core;
    cs->p_ray  = malloc(sizeof(double) * cs->n_rays);
    if (!cs->p_ray) { fprintf(stderr, "[CMFGEN] ray alloc failed\n"); return -1; }
    double r_in0 = geo->r_inner[0];
    for (int k = 0; k < n_core; ++k)            /* Gauss-like core spacing */
        cs->p_ray[k] = r_in0 * (k + 0.5) / (double)n_core;
    for (int s = 0; s < NS; ++s)
        cs->p_ray[n_core + s] = geo->r_outer[s];

    const char *d = getenv("LUMINA_RADEQ_DIAG");
    cs->diag = (d && atoi(d)) ? 1 : 0;
    cs->frozen_morph_eps = -1.0;   /* off until the post-convergence pass sets it */
    cs->cont_only = 0;             /* continuum-only J_inc pass off by default */
    return 0;
}

void cmfgen_free(CMFGENState *cs)
{
    if (!cs) return;
    free(cs->nu); free(cs->dnu); free(cs->chi_es); free(cs->chi_abs);
    free(cs->chi_line); free(cs->chi_line_th); free(cs->chi_line_cls);
    free(cs->chi_tot); free(cs->S_fixed); free(cs->J);
    free(cs->lambda_star); free(cs->t_color);
    free(cs->tri_lo); free(cs->tri_up); free(cs->tri_r);
    free(cs->p_ray);
    memset(cs, 0, sizeof(*cs));
}

/* ------------------------------------------------------------ */
/* Assemble per (shell,bin): electron-scatter, thermal bf/ff absorption,
 * expansion line opacity, and the scattering-independent source S_fixed. */
void cmfgen_assemble(CMFGENState *cs, const Geometry *geo,
                     const OpacityState *opac, BFOpacity *bf,
                     const PlasmaState *plasma)
{
    int NB = cs->n_bins, NS = cs->n_shells;
    int n_lines = opac->n_lines;
    double t_exp = geo->time_explosion;
    double inv_ct = 1.0 / (CM_C * t_exp);   /* expansion-opacity prefactor */

    memset(cs->chi_line, 0, sizeof(double) * (size_t)NS * NB);
    memset(cs->chi_line_th, 0, sizeof(double) * (size_t)NS * NB);
    memset(cs->chi_line_cls, 0, sizeof(double) * (size_t)NS * NB);
    /* line emissivity accumulator reuses chi_tot scratch before it is summed */
    double *eta_line = cs->chi_tot;

    /* PHYSICAL per-line destruction probability (LUMINA_CMFGEN_LINE_EPS_PHYS=1):
     * eps_l = C_ul/(C_ul + A_ul*beta_esc(tau_l)) per line — the measured FUV
     * blockers are ground/metastable resonance lines (eps_phys~1e-3..1e-2) that
     * the legacy path treats as FULLY thermal, pinning inner FUV J to the local
     * cold B(T_e). The thermal channel chi_line_th and its emissivity are
     * accumulated PER LINE (no bin-average eps mixing, codex review); the
     * scattering remainder joins chi_es in the combine loop. Knobs:
     * LUMINA_CMFGEN_EPS_FLOOR (1e-5), LUMINA_CMFGEN_EPS_CAP (1.0). */
    int eps_phys = 0;
    double eps_floor = 1e-5, eps_cap = 1.0;
    /* NLTE line-source consumption gate (LUMINA_CMFGEN_SRC_NLTE=1, default
     * OFF). The pop-ratio S_l data is correct physics (fluorescence) but is
     * EXPLOSIVE under the operator split: saturated FUV lines in cold gas
     * carry S_l up to ~1e16 x the local J (run 165510: J[mid,500] 1.8e-18 ->
     * 4.2e-2 at the first NLTE-fed iter), eta_line -> J -> T_e no-root pins
     * at 2*T_rad. Until the eps-weighted thermal channel / A4 in-Newton
     * source absorbs it, the B(T_e) fallback (the de-facto thermostat the
     * champion metrics rest on) stays the default. */
    int src_nlte = 0;
    { const char *sn = getenv("LUMINA_CMFGEN_SRC_NLTE");
      if (sn && atoi(sn)) src_nlte = 1; }
    { const char *ep = getenv("LUMINA_CMFGEN_LINE_EPS_PHYS");
      if (ep && atoi(ep)) eps_phys = 1;
      const char *ef = getenv("LUMINA_CMFGEN_EPS_FLOOR");
      if (ef) eps_floor = atof(ef);
      const char *ec = getenv("LUMINA_CMFGEN_EPS_CAP");
      if (ec) eps_cap = atof(ec); }
    /* TRANSFER-ONLY eps (LUMINA_CMFGEN_LINE_EPS_UV, e.g. 0.03 = branch-like
     * interim; codex ruling 2026-06-11): the scattering share enters ONLY the
     * formal solve (chi_es), while the RE/Newton closure keeps the FULL
     * chi_line with the cooling-only form chi_line*(min(J,B)-B) — the
     * operator-split T_e anchor (audited: nothing else owns the root at the
     * first NLTE iters) without FUV line pumping heating the gas. */
    double eps_uv = -1.0;
    { const char *eu = getenv("LUMINA_CMFGEN_LINE_EPS_UV");
      if (eu) eps_uv = atof(eu); }
    if (eps_phys) eps_uv = -1.0;              /* phys mode wins (experimental) */
    cs->chi_line_re = (eps_uv > 0.0) ? cs->chi_line : cs->chi_line_th;
    memset(eta_line, 0, sizeof(double) * (size_t)NS * NB);

    /* Expansion (Sobolev-binned) line opacity + emissivity.
     *   chi_line[bin] = sum_{l in bin} (1-e^{-tau_l}) * nu_l/(c t_exp dnu_bin)
     *   eta_line[bin] = sum_l (1-e^{-tau_l}) * nu_l/(c t_exp dnu_bin) * S_l
     * S_l = line_source_S if >0 else B_nu(T_e) (thermalised fallback). */
    for (int s = 0; s < NS; ++s) {
        double Te = plasma->T_e[s];
        double ne_s = plasma->n_electron ? plasma->n_electron[s]
                                         : opac->electron_density[s];
        for (int l = 0; l < n_lines; ++l) {
            double tau = opac->tau_sobolev[(size_t)l * NS + s];
            if (tau <= 1e-12) continue;
            double nu_l = opac->line_list_nu[l];
            if (nu_l <= cs->nu_min || nu_l >= cs->nu_max) continue;
            int b = (int)floor(log(nu_l / cs->nu_min) / cs->d_log_nu);
            if (b < 0 || b >= NB) continue;
            double frac = (tau > 1e-6) ? -expm1(-tau) : tau;   /* 1-e^{-tau} */
            double w = frac * nu_l * inv_ct / cs->dnu[b];      /* cm^-1 */
            double Sl = src_nlte ? opac->line_source_S[(size_t)l * NS + s]
                                 : 0.0;
            if (Sl <= 0.0) Sl = cm_planck(nu_l, Te);
            size_t idx = (size_t)s * NB + b;
            cs->chi_line[idx] += w;
            if (eps_phys) {
                double el = radeq_line_eps_phys(l, ne_s, Te, tau);
                if (el < 0.0) el = 1.0;        /* table not built: thermal */
                if (el < eps_floor) el = eps_floor;
                if (el > eps_cap)   el = eps_cap;
                cs->chi_line_th[idx] += w * el;
                eta_line[idx]        += w * el * Sl;  /* thermal-channel eta */
                /* A4 SRC_BLEND closure weight: exact two-level+Sobolev gas
                 * coupling. Jbar_l=(1-beta)S_l+beta*J_bin with S=(1-eps)Jbar
                 * +eps*B gives net line heating chi_l*eps*beta/(eps+beta)
                 * *(J_bin-B): saturated lines (beta<<eps, trapped field
                 * relaxed to local B) drop out — the 1/eps over-count AND the
                 * hot-continuum-fed-to-trapped-lines defect both die here. */
                {
                    double be = (tau > 700.0) ? 1.0 / tau
                              : (tau > 1e-6) ? (1.0 - exp(-tau)) / tau : 1.0;
                    cs->chi_line_cls[idx] += w * (el * be) / (el + be - el * be + 1e-300);
                }
            } else {
                eta_line[idx] += w * Sl;
            }
        }
    }

    /* Line-forest scattering split (LUMINA_CMFGEN_LINE_EPS = eps in (0,1]):
     * two-level treatment of the binned forest, S_line=(1-eps)*J+eps*B. The
     * scattering remainder (1-eps)*chi_line joins chi_es so the ALI closure
     * transports the photospheric FUV color outward (root of the inner
     * Gamma(Mg I/Si I) 5-1000x deficit: J was thermalized to the LOCAL cold
     * B within <1 shell by the pure-absorption frozen-source treatment).
     * Applied only in bins with chi_line > GATE*chi_abs (line-dominated;
     * default gate 1.0). eps<=0 or unset -> byte-identical legacy path. */
    double line_eps = -1.0, line_gate = 1.0;
    { const char *le = getenv("LUMINA_CMFGEN_LINE_EPS");
      if (le) line_eps = atof(le);
      const char *lg = getenv("LUMINA_CMFGEN_LINE_EPS_GATE");
      if (lg) line_gate = atof(lg); }
    long n_split = 0;

    /* electron scattering + bf/ff thermal absorption + combine. */
    for (int s = 0; s < NS; ++s) {
        double Te  = plasma->T_e[s];
        double n_e = plasma->n_electron ? plasma->n_electron[s]
                                        : opac->electron_density[s];
        double chi_e = n_e * CM_SIGMA_T;
        for (int b = 0; b < NB; ++b) {
            size_t idx = (size_t)s * NB + b;
            double nu = cs->nu[b];
            double B  = cm_planck(nu, Te);
            double chi_bf = bf ? bf_get_chi(bf, s, nu) : 0.0;
            if (chi_bf < 0.0) chi_bf = 0.0;
            /* free-free (Kramers, gaunt~1): chi_ff = 3.69e8 Z^2 n_e n_i T^-1/2
             * nu^-3 (1-e^{-h nu/kT}); approximate n_i ~ n_e, Z^2~1. */
            double chi_ff = 0.0;
            if (Te > 0.0 && nu > 0.0) {
                double gaunt = 1.0;
                double stim  = -expm1(-CM_H * nu / (CM_KB * Te));
                chi_ff = 3.692e8 * gaunt * n_e * n_e /
                         (sqrt(Te) * nu * nu * nu) * stim;
                if (chi_ff < 0.0) chi_ff = 0.0;
            }
            double chi_a   = chi_bf + chi_ff;        /* thermal true abs */
            /* continuum-only J_inc pass: drop ALL line opacity + emissivity so
             * the solved field is the bare chi_es+bf/ff continuum (no forest
             * blanketing, not self-referential with the line source). */
            double chi_ln  = cs->cont_only ? 0.0 : cs->chi_line[idx];
            double eta_ln  = cs->cont_only ? 0.0 : eta_line[idx];  /* in scratch */
            double chi_t   = chi_e + chi_a + chi_ln;

            double chi_ln_th = chi_ln;               /* legacy: all thermal */
            if (cs->frozen_morph_eps >= 0.0 &&
                chi_ln > line_gate * chi_a) {
                /* frozen-plasma morphology pass: force the forest-dominated
                 * bins to scatter with destruction prob eps (0 = pure
                 * coherent). The scattering remainder (1-eps)*chi_line joins
                 * chi_es so the ALI solve carries the photospheric field out;
                 * the per-line scattering source S_l=(1-eps)*Jbar+eps*B is
                 * written into opacity->line_source_S AFTER this solve. */
                double fe = cs->frozen_morph_eps;
                chi_ln_th = fe * chi_ln;
                eta_ln   *= fe;
                n_split++;
            } else if (eps_phys) {
                chi_ln_th = cs->chi_line_th[idx];    /* per-line accumulated */
                if (chi_ln_th > 0.0 || chi_ln > 0.0) n_split++;
            } else if (eps_uv > 0.0 && eps_uv <= 1.0 &&
                       chi_ln > line_gate * chi_a) {
                /* transfer-only split: S_fixed/transfer see the eps_uv share,
                 * the RE closure sees FULL chi_line via cs->chi_line_re. */
                chi_ln_th = eps_uv * chi_ln;
                eta_ln   *= eps_uv;
                n_split++;
            } else if (line_eps > 0.0 && line_eps <= 1.0 &&
                       chi_ln > line_gate * chi_a) {
                chi_ln_th = line_eps * chi_ln;
                eta_ln   *= line_eps;   /* thermal share of the NLTE source */
                n_split++;
            }
            cs->chi_es[idx]      = chi_e + (chi_ln - chi_ln_th);
            cs->chi_abs[idx]     = chi_a;
            cs->chi_line_th[idx] = chi_ln_th;
            /* S_fixed = (chi_abs*B + thermal line emissivity)/chi_tot; the
             * scattering share enters the solve as r*J, NOT here (no double
             * count of the line source). */
            cs->S_fixed[idx] = (chi_t > 0.0)
                             ? (chi_a * B + eta_ln) / chi_t : 0.0;
            /* chi_tot scratch now overwritten with the real total */
            cs->chi_tot[idx] = chi_t;
        }
    }
    if ((line_eps > 0.0 || eps_phys || eps_uv > 0.0) && cs->diag) {
        static int eps_diag_once = 0;
        if (!eps_diag_once && n_split > 0) {   /* first assemble with lines */
            eps_diag_once = 1;
            double thsum = 0.0, lnsum = 0.0;
            for (size_t i = 0; i < (size_t)NS * NB; i++) {
                thsum += cs->chi_line_th[i]; lnsum += cs->chi_line[i];
            }
            printf("[CMFGEN-LINEEPS] %s eps=%.3f gate=%.2f split bins %ld/%ld "
                   "global chi_th/chi_line=%.4f\n",
                   eps_phys ? "PHYS" : (eps_uv > 0.0 ? "UV-TRANSFER" : "CONST"),
                   eps_uv > 0.0 ? eps_uv : line_eps, line_gate,
                   n_split, (long)NS * NB, lnsum > 0.0 ? thsum / lnsum : 1.0);
        }
    }
}

/* ------------------------------------------------------------ */
/* One short-characteristics formal solution along all tangent rays for a
 * single frequency bin b, given the current total source S[shell]. Accumulates
 * J[shell] (angle-averaged) and the diagonal lambda_star[shell]. */
static void formal_solve_bin(CMFGENState *cs, const Geometry *geo,
                             int b, const double *S, double Bnu_inner,
                             double *Jb, double *Lstar,
                             double *Tlo, double *Tup)
{
    /* Tlo/Tup (optional, NULL to skip): tridiagonal Lambda off-diagonals —
     * Tlo[s] = nearest-INNER-neighbour coefficient L[s,s-1], Tup[s] = L[s,s+1].
     * Accumulated like Lacc (one-segment-attenuated upstream psi), normalized
     * by wacc. Used as the A4 ALI preconditioner (LUMINA_CMFGEN_LAMBDA_TRI). */
    int NS = cs->n_shells, NB = cs->n_bins;
    double *Jacc = calloc(NS, sizeof(double));
    double *wacc = calloc(NS, sizeof(double));
    double *Lacc = calloc(NS, sizeof(double));
    double *TloA = Tlo ? calloc(NS, sizeof(double)) : NULL;
    double *TupA = Tup ? calloc(NS, sizeof(double)) : NULL;
    if (!Jacc || !wacc || !Lacc || (Tlo && !TloA) || (Tup && !TupA)) {
        free(Jacc); free(wacc); free(Lacc); free(TloA); free(TupA); return;
    }

    /* shell-midpoint radii for the source grid */
    /* For each ray of impact parameter p, find shells with r_outer > p and
     * integrate inward (mu<0) then reflect/emit and integrate outward (mu>0). */
    for (int ray = 0; ray < cs->n_rays; ++ray) {
        double p = cs->p_ray[ray];
        /* collect intersected shells (outer->in), store z and shell idx */
        int    *sh = malloc(sizeof(int) * (NS + 1));
        double *z  = malloc(sizeof(double) * (NS + 1));
        int n = 0;
        for (int s = NS - 1; s >= 0; --s) {
            double ro = geo->r_outer[s];
            if (ro <= p) break;          /* inner shells don't reach this p */
            double rmid = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
            if (rmid <= p) rmid = p * 1.0000001;
            sh[n] = s;
            z[n]  = sqrt(rmid * rmid - p * p);
            ++n;
        }
        if (n == 0) { free(sh); free(z); continue; }
        int core = (p < geo->r_inner[0]) ? 1 : 0;
        /* core rays terminate at the core SURFACE z_core=sqrt(r_in0^2-p^2),
         * not at z=0: the innermost segment is z[n-1]-z_core, else dtau is
         * overestimated ~r_in/dr (100x) and the core B(T_inner) never leaks
         * into shell 0 (artificial seed-pin of inner T_e). */
        double z_core = 0.0;
        if (core) {
            double ri0 = geo->r_inner[0];
            z_core = sqrt(ri0 * ri0 - p * p);
            if (z_core > z[n - 1]) z_core = z[n - 1];
        }

        /* angular weight: this ray represents mu-interval around mu=z/r at the
         * outermost shell; use simple dp annulus weight 2 p dp / r^2 ~ dmu.
         * For an even-handed first solver we use uniform ray weight then
         * renormalise J by total weight per shell (wacc). */
        double ray_w = p;   /* proportional to annulus area element p dp */

        /* ----- inbound leg (mu<0): from outer boundary inward ----- */
        double I = 0.0;                       /* outer BC: no incoming */
        double psi_prev = 0.0;                /* upstream segment's psi */
        for (int i = 0; i < n; ++i) {
            int s = sh[i];
            double S_s = S[s];
            double ds = (i + 1 < n) ? fabs(z[i] - z[i + 1])
                                    : (z[i] - z_core);  /* to core surface/origin */
            double dtau = cs->chi_tot[(size_t)s * NB + b] * ds;
            if (dtau < 0.0) dtau = 0.0;
            double ex = exp(-dtau);
            double psi = (dtau > 1e-4) ? (1.0 - ex) : (dtau - 0.5*dtau*dtau);
            I = I * ex + S_s * psi;
            /* inbound (mu<0) hemisphere sample; the matching outbound leg
             * supplies mu>0, so J = mean over both legs = mean intensity.
             * No extra 0.5: the leg average already gives (I_+ + I_-)/2. */
            Jacc[s] += ray_w * I;
            wacc[s] += ray_w;
            Lacc[s] += ray_w * psi;           /* diagonal local response */
            /* inbound upstream neighbour is the next-OUTER shell sh[i-1]=s+1 */
            if (TupA && i > 0) TupA[s] += ray_w * ex * psi_prev;
            psi_prev = psi;
        }
        double psi_turn = core ? 0.0 : psi_prev;  /* tangent carry (non-core) */
        /* ----- inner boundary ----- */
        if (core) I = Bnu_inner;              /* diffusive core emits B */
        /* (non-core grazing ray: I continues with whatever it accumulated) */
        /* ----- outbound leg (mu>0): from inner shell back out ----- */
        psi_prev = 0.0;
        for (int i = n - 1; i >= 0; --i) {
            int s = sh[i];
            double S_s = S[s];
            double ds = (i + 1 < n) ? fabs(z[i] - z[i + 1]) : (z[i] - z_core);
            double dtau = cs->chi_tot[(size_t)s * NB + b] * ds;
            if (dtau < 0.0) dtau = 0.0;
            double ex = exp(-dtau);
            double psi = (dtau > 1e-4) ? (1.0 - ex) : (dtau - 0.5*dtau*dtau);
            I = I * ex + S_s * psi;            /* outbound (mu>0) hemisphere */
            Jacc[s] += ray_w * I;
            wacc[s] += ray_w;
            Lacc[s] += ray_w * psi;
            if (i == n - 1) {
                /* first outbound visit: upstream is the SAME shell's inbound
                 * segment (tangent turn) — a tridiag-LOCAL (diagonal) term;
                 * core rays carry B(T_inner) instead (inhomogeneous, in J_fs).
                 * Only counted in tridiag mode (keeps the legacy diagonal-ALI
                 * Lambda* byte-identical when the preconditioner is off). */
                if (TloA) Lacc[s] += ray_w * ex * psi_turn;
            } else if (TloA) {
                /* outbound upstream neighbour is the next-INNER shell s-1 */
                TloA[s] += ray_w * ex * psi_prev;
            }
            psi_prev = psi;
        }
        free(sh); free(z);
    }

    for (int s = 0; s < NS; ++s) {
        double iw = (wacc[s] > 0.0) ? 1.0 / wacc[s] : 0.0;
        Jb[s]    = Jacc[s] * iw;
        Lstar[s] = Lacc[s] * iw;
        if (Tlo) Tlo[s] = TloA ? TloA[s] * iw : 0.0;
        if (Tup) Tup[s] = TupA ? TupA[s] * iw : 0.0;
    }
    free(Jacc); free(wacc); free(Lacc); free(TloA); free(TupA);
}

/* ------------------------------------------------------------ */
void cmfgen_solve_J(CMFGENState *cs, const Geometry *geo, double T_inner,
                    int n_ali_iter)
{
    int NS = cs->n_shells, NB = cs->n_bins;
    double *S    = malloc(sizeof(double) * NS);
    double *Jb   = malloc(sizeof(double) * NS);
    double *Lst  = malloc(sizeof(double) * NS);
    if (!S || !Jb || !Lst) { free(S); free(Jb); free(Lst); return; }

    /* A4 tridiagonal-Lambda ALI preconditioner (LUMINA_CMFGEN_LAMBDA_TRI=1):
     * solve (I - Lambda_tri*R) J_new = J_fs - Lambda_tri*R*J_old per ALI pass
     * (Thomas, R=diag(r_s)). Needed once the line forest carries a near-unity
     * scattering albedo (physical eps ~1e-3): the pure diagonal closure
     * re-floors (the thin-UV pathology). Early-stop at max|dJ|/J < ALI_TOL. */
    int use_tri = 0;
    { const char *tr = getenv("LUMINA_CMFGEN_LAMBDA_TRI"); if (tr) use_tri = atoi(tr); }
    double ali_tol = 1e-3;
    { const char *tl = getenv("LUMINA_CMFGEN_ALI_TOL"); if (tl) ali_tol = atof(tl); }
    /* near-unity albedo (line-eps runs): per-pass change underestimates the
     * true error (spectral radius -> 1), so enforce a minimum pass count
     * before the early-stop is trusted (A4 Stage-1 sets 16+). */
    int ali_minit = 1;
    { const char *mi = getenv("LUMINA_CMFGEN_ALI_MINIT"); if (mi) ali_minit = atoi(mi); }
    double *Tlo = use_tri ? malloc(sizeof(double) * NS) : NULL;
    double *Tup = use_tri ? malloc(sizeof(double) * NS) : NULL;
    double *rA  = use_tri ? malloc(sizeof(double) * NS) : NULL;
    double *aA  = use_tri ? malloc(sizeof(double) * NS) : NULL;
    double *dA  = use_tri ? malloc(sizeof(double) * NS) : NULL;
    double *cA  = use_tri ? malloc(sizeof(double) * NS) : NULL;
    double *rhs = use_tri ? malloc(sizeof(double) * NS) : NULL;
    if (use_tri && (!Tlo || !Tup || !rA || !aA || !dA || !cA || !rhs)) {
        free(Tlo); free(Tup); free(rA); free(aA); free(dA); free(cA); free(rhs);
        Tlo = Tup = rA = aA = dA = cA = rhs = NULL;
        use_tri = 0;
    }

    /* optional single-cell ALI trace: LUMINA_CMFGEN_CELLDIAG="s,b" */
    int cd_s = -1, cd_b = -1;
    const char *cd = getenv("LUMINA_CMFGEN_CELLDIAG");
    if (cd && sscanf(cd, "%d,%d", &cd_s, &cd_b) != 2) { cd_s = cd_b = -1; }

    for (int b = 0; b < NB; ++b) {
        double Bin = cm_planck(cs->nu[b], T_inner);
        /* (tridiagonal-)ALI Lambda iteration for the scattering channel */
        for (int it = 0; it < n_ali_iter; ++it) {
            for (int s = 0; s < NS; ++s) {
                size_t idx = (size_t)s * NB + b;
                double r = (cs->chi_tot[idx] > 0.0)
                         ? cs->chi_es[idx] / cs->chi_tot[idx] : 0.0;
                if (rA) rA[s] = r;
                S[s] = cs->S_fixed[idx] + r * cs->J[idx];
            }
            formal_solve_bin(cs, geo, b, S, Bin, Jb, Lst, Tlo, Tup);
            double maxrel = 0.0;
            if (use_tri) {
                for (int s = 0; s < NS; ++s) {
                    size_t idx = (size_t)s * NB + b;
                    double Jo  = cs->J[idx];
                    double Jom = (s > 0)      ? cs->J[idx - NB] : 0.0;
                    double Jop = (s < NS - 1) ? cs->J[idx + NB] : 0.0;
                    aA[s] = (s > 0)      ? -Tlo[s] * rA[s - 1] : 0.0;
                    cA[s] = (s < NS - 1) ? -Tup[s] * rA[s + 1] : 0.0;
                    dA[s] = 1.0 - Lst[s] * rA[s];
                    if (dA[s] < 1e-10) dA[s] = 1e-10;
                    rhs[s] = Jb[s] - Lst[s] * rA[s] * Jo
                           - ((s > 0)      ? Tlo[s] * rA[s - 1] * Jom : 0.0)
                           - ((s < NS - 1) ? Tup[s] * rA[s + 1] * Jop : 0.0);
                }
                /* Thomas forward sweep */
                for (int s = 1; s < NS; ++s) {
                    double m = aA[s] / dA[s - 1];
                    dA[s]  -= m * cA[s - 1];
                    if (dA[s] < 1e-10) dA[s] = 1e-10;
                    rhs[s] -= m * rhs[s - 1];
                }
                double Jn = rhs[NS - 1] / dA[NS - 1];
                for (int s = NS - 1; s >= 0; --s) {
                    if (s < NS - 1) Jn = (rhs[s] - cA[s] * Jb[s + 1]) / dA[s];
                    /* NOTE: back-substitution must use the NEW J of s+1; reuse
                     * Jb[] as the solved-J scratch to avoid another array. */
                    size_t idx = (size_t)s * NB + b;
                    double Jold = cs->J[idx];
                    if (!isfinite(Jn) || Jn < 0.0) Jn = 0.0;  /* finite guard:
                        warm-started J makes any one-iter accident permanent */
                    double rel = (Jn > 1e-300) ? fabs(Jn - Jold) / Jn : 0.0;
                    if (rel > maxrel) maxrel = rel;
                    Jb[s] = Jn;                 /* solved J for s (scratch) */
                }
                for (int s = 0; s < NS; ++s) {
                    size_t idx = (size_t)s * NB + b;
                    cs->J[idx] = Jb[s];
                    /* Stage-4 deferred: keep the FORMAL diagonal for the
                     * Newton response (semantics unchanged). */
                    cs->lambda_star[idx] = Lst[s];
                    /* A4 Stage-2.5: persist the tridiagonal response
                     * coefficients (last ALI pass wins) so the global Newton
                     * can apply delta-J = (I - Lambda_tri R)^-1 Lambda_tri
                     * delta-S without re-running the formal solve. */
                    cs->tri_lo[idx] = Tlo[s];
                    cs->tri_up[idx] = Tup[s];
                    cs->tri_r[idx]  = rA[s];
                }
            } else {
                for (int s = 0; s < NS; ++s) {
                    size_t idx = (size_t)s * NB + b;
                    double r = (cs->chi_tot[idx] > 0.0)
                             ? cs->chi_es[idx] / cs->chi_tot[idx] : 0.0;
                    /* local ALI accel: J = (J_fs - L* r J_old)/(1 - L* r) */
                    double Ldiag = Lst[s];
                    double denom = 1.0 - Ldiag * r;
                    double Jnew = (denom > 1e-10)
                                ? (Jb[s] - Ldiag * r * cs->J[idx]) / denom
                                : Jb[s];
                    if (b == cd_b && s == cd_s) {
                        size_t i2 = (size_t)cd_s * NB + cd_b;
                        printf("[CMFGEN-CELL] s=%d b=%d ali=%d chi_es=%.3e chi_abs=%.3e "
                               "chi_line=%.3e r=%.4f Sfix=%.3e S=%.3e Lst=%.4e Jfs=%.3e "
                               "Jold=%.3e Jnew=%.3e\n", cd_s, cd_b, it, cs->chi_es[i2],
                               cs->chi_abs[i2], cs->chi_line[i2], r, cs->S_fixed[i2],
                               S[s], Ldiag, Jb[s], cs->J[idx], Jnew < 0 ? 0 : Jnew);
                    }
                    if (!isfinite(Jnew) || Jnew < 0.0) Jnew = 0.0;
                    cs->J[idx] = Jnew;
                    /* Persist the diagonal ∂J/∂S operator for the RADEQ/Newton T_e
                     * solve (Phase-1 faithful radiation response). Last ALI iter wins. */
                    cs->lambda_star[idx] = Lst[s];
                }
            }
            if (use_tri && maxrel < ali_tol && it >= ali_minit) break;
        }
    }
    free(S); free(Jb); free(Lst);
    free(Tlo); free(Tup); free(rA); free(aA); free(dA); free(cA); free(rhs);
    if (cd_s >= 0) fflush(stdout);
}

/* ------------------------------------------------------------ */
void cmfgen_window_color(CMFGENState *cs)
{
    int NS = cs->n_shells, NB = cs->n_bins;
    /* Two CONTINUUM windows (path-4 redo of the sealed 165485 anchor). The
     * old 4150-4300A blue band was line-REPROCESSED in the outer shells
     * (chi_line/chi_cont up to 8.6e6): anchoring T_e brightened the locally
     * re-emitted trough J -> hotter color -> runaway to 24kK. These bands are
     * chosen from the 165584 JDUMP chi map: worst chi_line/(chi_es+chi_abs)
     * over sh40-48 is <0.10 in BOTH, so the window J is TRANSPORTED photo-
     * spheric light, not local re-emission — the feedback loop gain is dead.
     * Measured on the converged champion field: color = 2471.6K, FLAT across
     * sh38-48 (gold tail T_e 2505-2560K; treadmill extracts 2254K = -10%).
     * Override via LUMINA_CMFGEN_COLOR_BANDS="l1,l2,l3,l4" (Angstrom). */
    double LB1 = 5619e-8, LB2 = 6083e-8, LR1 = 9000e-8, LR2 = 11000e-8;
    { const char *cb = getenv("LUMINA_CMFGEN_COLOR_BANDS");
      if (cb) { double a, b2, c2, e;
                if (sscanf(cb, "%lf,%lf,%lf,%lf", &a, &b2, &c2, &e) == 4) {
                    LB1 = a * 1e-8; LB2 = b2 * 1e-8;
                    LR1 = c2 * 1e-8; LR2 = e * 1e-8; } } }
    for (int s = 0; s < NS; ++s) {
        double Jb = 0.0, wb = 0.0, Jr = 0.0, wr = 0.0;
        int nb_ = 0, nr_ = 0;
        for (int b = 0; b < NB; ++b) {
            double lam = CM_C / cs->nu[b];
            int blue = (lam >= LB1 && lam <= LB2);
            int red  = (lam >= LR1 && lam <= LR2);
            if (!blue && !red) continue;
            size_t idx = (size_t)s * NB + b;
            double J = cs->J[idx];
            if (J <= 0.0) continue;
            if (blue) { Jb += J * cs->dnu[b]; wb += cs->dnu[b]; nb_++; }
            else      { Jr += J * cs->dnu[b]; wr += cs->dnu[b]; nr_++; }
        }
        cs->t_color[s] = -1.0;
        if (nb_ < 2 || nr_ < 2 || Jr <= 0.0 || wb <= 0.0 || wr <= 0.0) continue;
        double target = (Jb / wb) / (Jr / wr);
        if (!(target > 0.0)) continue;
        /* Planck band-ratio is monotonically increasing in T: bisect. */
        double nub = CM_C / (0.5 * (LB1 + LB2)), nur = CM_C / (0.5 * (LR1 + LR2));
        double Tlo = 800.0, Thi = 30000.0;
        double rlo = cm_planck(nub, Tlo) / cm_planck(nur, Tlo);
        double rhi = cm_planck(nub, Thi) / cm_planck(nur, Thi);
        if (target <= rlo || target >= rhi) continue;
        for (int it = 0; it < 60; ++it) {
            double Tm = 0.5 * (Tlo + Thi);
            double rm = cm_planck(nub, Tm) / cm_planck(nur, Tm);
            if (rm < target) Tlo = Tm; else Thi = Tm;
        }
        cs->t_color[s] = 0.5 * (Tlo + Thi);
    }
}

/* ------------------------------------------------------------ */
/* Thick/thin limit validation. For chosen shells prints, at 3 bins, the
 * inward radial optical depth tau_r, B(T_e), the (converged) total source S,
 * the solved J, and the ratios J/B (->1 thick, ->W thin) and J/S (->1 thick).
 * A correct solver must show J/B->1 and J/S->1 in the most opaque cell, and
 * J/B ~ W (geometric dilution) in the optically-thin outer. */
void cmfgen_validate(const CMFGENState *cs, const Geometry *geo,
                     const PlasmaState *plasma)
{
    int NS = cs->n_shells, NB = cs->n_bins;
    int bins[3] = { NB / 5, NB / 2, (4 * NB) / 5 };   /* blue, mid, red */
    int shells[4] = { 0, NS / 4, NS / 2, NS - 1 };

    printf("[CMFGEN-VALID] thick(J/B->1,J/S->1) / thin(J/B->W) limit check\n");
    for (int si = 0; si < 4; ++si) {
        int s = shells[si];
        double Te = plasma->T_e[s];
        double rmid = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
        /* geometric dilution W = 0.5(1 - sqrt(1 - (r_in0/r)^2)) */
        double x = geo->r_inner[0] / rmid;
        double W = (x < 1.0) ? 0.5 * (1.0 - sqrt(1.0 - x * x)) : 0.5;
        for (int bi = 0; bi < 3; ++bi) {
            int b = bins[bi];
            size_t idx = (size_t)s * NB + b;
            /* inward radial optical depth from outer boundary to shell s */
            double tau_r = 0.0;
            for (int q = NS - 1; q >= s; --q) {
                double dr = geo->r_outer[q] - geo->r_inner[q];
                tau_r += cs->chi_tot[(size_t)q * NB + b] * dr;
            }
            double B = cm_planck(cs->nu[b], Te);
            double Sloc = cs->S_fixed[idx]
                        + ((cs->chi_tot[idx] > 0.0)
                           ? cs->chi_es[idx] / cs->chi_tot[idx] : 0.0) * cs->J[idx];
            double J = cs->J[idx];
            printf("[CMFGEN-VALID] s=%2d b=%4d nu=%.3e Te=%.0f tau_r=%.3e "
                   "W=%.4f B=%.3e S=%.3e J=%.3e J/B=%.3f J/S=%.3f\n",
                   s, b, cs->nu[b], Te, tau_r, W, B, Sloc, J,
                   B > 0 ? J / B : 0.0, Sloc > 0 ? J / Sloc : 0.0);
        }
    }

    /* MISSING-TERM PROBE: volumetric line RADIATIVE heating 4π∫(χ_line·J − η_line)dν
     * [erg/s/cm^3], the term absent from radeq_net (which carries lines only as the
     * collisional-net cooling). η_line is reconstructed from the stored source:
     *   S_fixed = (χ_abs·B + η_line)/χ_tot  ⇒  η_line = S_fixed·χ_tot − χ_abs·B.
     * Compare net against H_photo/C_bb to see if it can anchor outer T_e. */
    int dsh[5] = { 0, NS / 2, (3 * NS) / 4, (7 * NS) / 8, NS - 1 };
    printf("[CMFGEN-LINEHEAT] 4pi*Int(chi_line*J - eta_line)dnu  (>0 = net heating)\n");
    for (int di = 0; di < 5; ++di) {
        int s = dsh[di];
        double Te = plasma->T_e[s];
        double abs_r = 0.0, emi_r = 0.0;
        for (int b = 0; b < NB; ++b) {
            size_t idx = (size_t)s * NB + b;
            double B = cm_planck(cs->nu[b], Te);
            double eta_ln = cs->S_fixed[idx] * cs->chi_tot[idx]
                          - cs->chi_abs[idx] * B;
            if (eta_ln < 0.0) eta_ln = 0.0;
            abs_r += cs->chi_line[idx] * cs->J[idx] * cs->dnu[b];
            emi_r += eta_ln * cs->dnu[b];
        }
        abs_r *= 4.0 * M_PI;  emi_r *= 4.0 * M_PI;
        printf("[CMFGEN-LINEHEAT] s=%2d Te=%.0f  H_line_abs=%.3e  emis_line=%.3e  "
               "net=%.3e\n", s, Te, abs_r, emi_r, abs_r - emi_r);
    }

    /* Full deterministic-J dump (LUMINA_CMFGEN_JDUMP=1): per shell x bin, the
     * solved J plus the opacity split and ALI diagonal, so the thin-UV floor
     * (Defect A) can be located offline against the sigma_bf edges. */
    const char *jd = getenv("LUMINA_CMFGEN_JDUMP");
    if (jd && atoi(jd)) {
        FILE *jf = fopen("lumina_cmfgen_jnu.csv", "w");
        if (jf) {
            fprintf(jf, "shell,bin,nu,J,chi_es,chi_abs,chi_line,chi_tot,"
                        "S_fixed,lambda_star\n");
            for (int s = 0; s < NS; ++s)
                for (int b = 0; b < NB; ++b) {
                    size_t idx = (size_t)s * NB + b;
                    fprintf(jf, "%d,%d,%.6e,%.6e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e\n",
                            s, b, cs->nu[b], cs->J[idx], cs->chi_es[idx],
                            cs->chi_abs[idx], cs->chi_line[idx],
                            cs->chi_tot[idx], cs->S_fixed[idx],
                            cs->lambda_star[idx]);
                }
            fclose(jf);
            printf("[CMFGEN-JDUMP] wrote lumina_cmfgen_jnu.csv (%d shells x %d bins)\n",
                   NS, NB);
        }
    }
    fflush(stdout);
}

/* ------------------------------------------------------------ */
/* Emergent observer-frame spectrum from the converged deterministic field.
 *
 * For each tangent ray of impact parameter p we propagate the formal solution
 * (same inbound + inner-BC + outbound legs as formal_solve_bin) and keep only
 * the surface intensity I+(p) at the end of the outbound leg. The emergent
 * monochromatic luminosity is the p-z surface integral
 *      L_nu = 8 pi^2 Int_0^Rmax I+(p) p dp .
 * The expansion (Sobolev-binned) line opacity already subsumes the comoving
 * d/dnu, so each bin is an observer-frame-broadened quasi-static slab and the
 * binned L_nu is the line-blanketed emergent SED at bin resolution. */
int cmfgen_write_spectrum(const CMFGENState *cs, const Geometry *geo,
                          double T_inner, const char *path)
{
    int NS = cs->n_shells, NB = cs->n_bins, NR = cs->n_rays;

    /* ---- precompute per-ray intersected shells + segment lengths (b-indep) ---- */
    int    *ray_n    = malloc(sizeof(int) * NR);
    int    *ray_core = malloc(sizeof(int) * NR);
    int    *seg_sh   = malloc(sizeof(int) * (size_t)NR * NS);
    double *seg_ds   = malloc(sizeof(double) * (size_t)NR * NS);
    double *S        = malloc(sizeof(double) * NS);
    double *Lnu      = calloc(NB, sizeof(double));
    if (!ray_n || !ray_core || !seg_sh || !seg_ds || !S || !Lnu) {
        free(ray_n); free(ray_core); free(seg_sh); free(seg_ds);
        free(S); free(Lnu);
        fprintf(stderr, "[CMFGEN] spectrum alloc failed\n");
        return -1;
    }

    for (int ray = 0; ray < NR; ++ray) {
        double p = cs->p_ray[ray];
        int *sh = &seg_sh[(size_t)ray * NS];
        double *zz = malloc(sizeof(double) * (NS + 1));
        int nshell = 0;
        for (int s = NS - 1; s >= 0; --s) {
            double ro = geo->r_outer[s];
            if (ro <= p) break;
            double rmid = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
            if (rmid <= p) rmid = p * 1.0000001;
            sh[nshell] = s;
            zz[nshell] = sqrt(rmid * rmid - p * p);
            ++nshell;
        }
        ray_n[ray]    = nshell;
        ray_core[ray] = (p < geo->r_inner[0]) ? 1 : 0;
        double z_core = 0.0;
        if (ray_core[ray] && nshell > 0) {
            double ri0 = geo->r_inner[0];
            z_core = sqrt(ri0 * ri0 - p * p);
            if (z_core > zz[nshell - 1]) z_core = zz[nshell - 1];
        }
        double *ds = &seg_ds[(size_t)ray * NS];
        for (int i = 0; i < nshell; ++i)
            ds[i] = (i + 1 < nshell) ? fabs(zz[i] - zz[i + 1])
                                     : (ray_core[ray] ? zz[i] - z_core
                                                      : fabs(zz[i]));
        free(zz);
    }

    /* ---- per-bin emergent flux ---- */
    for (int b = 0; b < NB; ++b) {
        double Bin = cm_planck(cs->nu[b], T_inner);
        for (int s = 0; s < NS; ++s) {
            size_t idx = (size_t)s * NB + b;
            double r = (cs->chi_tot[idx] > 0.0)
                     ? cs->chi_es[idx] / cs->chi_tot[idx] : 0.0;
            S[s] = cs->S_fixed[idx] + r * cs->J[idx];   /* converged source */
        }
        /* integrate I+(p) p dp over the (ascending-p) ray grid, trapezoid,
         * with f(0)=0 at the origin. */
        double integ = 0.0, p_prev = 0.0, f_prev = 0.0;
        for (int ray = 0; ray < NR; ++ray) {
            int n = ray_n[ray];
            if (n == 0) continue;
            const int *sh = &seg_sh[(size_t)ray * NS];
            const double *ds = &seg_ds[(size_t)ray * NS];
            double I = 0.0;                       /* outer BC: no incoming */
            for (int i = 0; i < n; ++i) {         /* inbound (mu<0) */
                double dtau = cs->chi_tot[(size_t)sh[i] * NB + b] * ds[i];
                if (dtau < 0.0) dtau = 0.0;
                double ex = exp(-dtau);
                double psi = (dtau > 1e-4) ? (1.0 - ex) : (dtau - 0.5*dtau*dtau);
                I = I * ex + S[sh[i]] * psi;
            }
            if (ray_core[ray]) I = Bin;           /* diffusive core emits B */
            for (int i = n - 1; i >= 0; --i) {    /* outbound (mu>0) */
                double dtau = cs->chi_tot[(size_t)sh[i] * NB + b] * ds[i];
                if (dtau < 0.0) dtau = 0.0;
                double ex = exp(-dtau);
                double psi = (dtau > 1e-4) ? (1.0 - ex) : (dtau - 0.5*dtau*dtau);
                I = I * ex + S[sh[i]] * psi;
            }
            double p = cs->p_ray[ray];
            double f = I * p;                      /* integrand I+(p) p */
            integ += 0.5 * (f_prev + f) * (p - p_prev);
            p_prev = p; f_prev = f;
        }
        Lnu[b] = 8.0 * M_PI * M_PI * integ;        /* erg/s/Hz */
    }

    /* ---- write ascending-wavelength CSV: L_lambda = L_nu * c/lambda^2 ---- */
    FILE *fp = fopen(path, "w");
    if (!fp) {
        free(ray_n); free(ray_core); free(seg_sh); free(seg_ds);
        free(S); free(Lnu);
        fprintf(stderr, "[CMFGEN] cannot open %s\n", path);
        return -1;
    }
    fprintf(fp, "wavelength_angstrom,flux\n");
    for (int b = NB - 1; b >= 0; --b) {            /* nu desc -> lambda asc */
        double lam_cm = CM_C / cs->nu[b];
        double lam_A  = lam_cm * 1.0e8;
        double L_lam  = Lnu[b] * CM_C / (lam_cm * lam_cm) * 1.0e-8; /* erg/s/A */
        fprintf(fp, "%.6f,%.6e\n", lam_A, L_lam);
    }
    fclose(fp);
    printf("Pure-CMFGEN emergent spectrum written to %s (%d bins)\n", path, NB);

    free(ray_n); free(ray_core); free(seg_sh); free(seg_ds);
    free(S); free(Lnu);
    return 0;
}

/* ------------------------------------------------------------ */
/* Linear interpolation of y over the ASCENDING nu grid (cs->nu[0]=nu_min ..
 * nu[NB-1]=nu_max, see cmfgen_create nu[b]=nu_min*exp((b+0.5)*d_log_nu)).
 * Returns 0 (vacuum) for nu_q outside (nu[0], nu[NB-1]). */
static double cmf_interp_nu_asc(const double *nu, const double *y, int NB,
                                double nu_q)
{
    if (nu_q <= nu[0] || nu_q >= nu[NB - 1]) return 0.0;
    int lo = 0, hi = NB - 1;                     /* nu[lo] <= nu_q < nu[hi] */
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (nu[mid] <= nu_q) lo = mid; else hi = mid;
    }
    double t = (nu_q - nu[lo]) / (nu[lo + 1] - nu[lo]); /* nu[lo+1] > nu[lo] */
    return y[lo] + t * (y[lo + 1] - y[lo]);
}

/* Observer-frame Doppler q = gamma(1 - mu*beta) at signed path coordinate z
 * along a ray of impact parameter p (mu = z/r, beta = r/(c t_exp)). */
static double cmf_q_at_z(double p, double z, double inv_ct)
{
    double r = sqrt(p * p + z * z);
    double beta = r * inv_ct;
    double mu = (r > 0.0) ? z / r : 0.0;
    return (1.0 - mu * beta) / sqrt(1.0 - beta * beta);
}

/* March one ray segment [z0,z1] (signed z increasing toward observer) through
 * shell s, sub-splitting so the comoving frequency q*nu_obs sweeps <=0.5 bin
 * per sub-step (resolves line P-Cygni; continuum unaffected). Returns updated I. */
static double cmf_obs_march(double I, double p, double z0, double z1, int s,
                            const double *nu, const double *chi_tot,
                            const double *Sbin, int NB, double nu_obs,
                            double inv_ct, double d_log_nu)
{
    double q0 = cmf_q_at_z(p, z0, inv_ct);
    double q1 = cmf_q_at_z(p, z1, inv_ct);
    int nsub = (int)ceil(fabs(log(q1 / q0)) / (0.5 * d_log_nu));
    if (nsub < 1)  nsub = 1;
    if (nsub > 256) nsub = 256;
    for (int m = 0; m < nsub; ++m) {
        double za = z0 + (z1 - z0) * ((double)m / nsub);
        double zb = z0 + (z1 - z0) * ((double)(m + 1) / nsub);
        double zm = 0.5 * (za + zb);
        double ds = fabs(zb - za);
        double r  = sqrt(p * p + zm * zm);
        double mu = (r > 0.0) ? zm / r : 0.0;
        double beta = r * inv_ct;
        double q = (1.0 - mu * beta) / sqrt(1.0 - beta * beta);
        double D = 1.0 / q;
        double nucmf = q * nu_obs;
        double chi0 = cmf_interp_nu_asc(nu, &chi_tot[(size_t)s * NB], NB, nucmf);
        double S0   = cmf_interp_nu_asc(nu, &Sbin[(size_t)s * NB],    NB, nucmf);
        if (chi0 < 0.0) chi0 = 0.0;
        double alpha = q * chi0;
        double dtau  = alpha * ds; if (dtau < 0.0) dtau = 0.0;
        double ex  = (dtau > 700.0) ? 0.0 : exp(-dtau);
        double psi = (dtau > 1e-4) ? (1.0 - ex) : (dtau - 0.5 * dtau * dtau);
        double Sobs = (alpha > 0.0) ? (D * D * D) * S0 : 0.0;
        I = I * ex + Sobs * psi;
    }
    return I;
}

/* S4 line thermalisation share (LUMINA_CMF_LINE_SOB_EPS, default 0 = pure
 * resonance scatter S_l=Jbar). Set once per spectrum in cmfgen_write_spectrum_obs. */
static double g_sob_eps = 0.0;
static int    g_sob_noemit = 0;
static int    g_sob_diag = 0;
static double g_sob_rphot  = 0.0;   /* photosphere radius (r_inner[0]) for W(r) */
static double g_sob_Tinner = 0.0;   /* photosphere T for the diluted backlight */
static int    g_sob_srcj   = 0;     /* 1 = legacy cs->J source (contaminated) */

/* S4 — LINE-RESOLVED Sobolev observer march (gate LUMINA_CMF_OBS_SOBOLEV).
 * The binned expansion opacity defangs tau_Sobolev (frac=1-e^{-tau}->1, /dnu_bin
 * => integrated ray tau ~ O(1), no P-Cygni; codex 019ef207 + agent ac7845ec).
 * Here the CONTINUUM is transported from chi_es+chi_abs (NOT the binned
 * chi_line), and each line whose rest nu_l lies in the sub-step's swept comoving
 * interval applies its FULL Sobolev jump  I = I*e^{-tau_S} + D^3*S_l*(1-e^{-tau_S})
 * at the resonance — exactly the directional resonance the MC does. Requires
 * LINE_EPS off (so chi_es is pure electron, no double-count). */
static double cmf_obs_march_sob(double I, double p, double z0, double z1, int s,
                                const CMFGENState *cs, const OpacityState *opac,
                                double Te_s, double nu_obs, double inv_ct)
{
    int NB = cs->n_bins, NS = cs->n_shells, NL = opac->n_lines;
    double q0 = cmf_q_at_z(p, z0, inv_ct);
    double q1 = cmf_q_at_z(p, z1, inv_ct);
    /* Resolve the Sobolev resonance to a FINE comoving-velocity step. The bin
     * width 0.5*d_log_nu (~800 km/s) smears each line jump over ~800 km/s of
     * observer wavelength -> too-shallow trough. Resolve to ~30 km/s (env
     * LUMINA_CMF_OBS_DVRES, km/s) so the jump lands at the sharp resonance. */
    double dlq = 30.0 / 2.99792458e5;
    { const char *e = getenv("LUMINA_CMF_OBS_DVRES"); if (e) dlq = atof(e) / 2.99792458e5; }
    int nsub = (int)ceil(fabs(log(q1 / q0)) / dlq);
    if (nsub < 1)    nsub = 1;
    if (nsub > 4096) nsub = 4096;
    for (int m = 0; m < nsub; ++m) {
        double za = z0 + (z1 - z0) * ((double)m / nsub);
        double zb = z0 + (z1 - z0) * ((double)(m + 1) / nsub);
        double zm = 0.5 * (za + zb);
        double ds = fabs(zb - za);
        double r  = sqrt(p * p + zm * zm);
        double mu = (r > 0.0) ? zm / r : 0.0;
        double beta = r * inv_ct;
        double q = (1.0 - mu * beta) / sqrt(1.0 - beta * beta);
        double D = 1.0 / q;
        double nucmf = q * nu_obs;
        /* continuum (electron scatter + thermal bf/ff), NO binned line */
        double chi_es = cmf_interp_nu_asc(cs->nu, &cs->chi_es[(size_t)s * NB], NB, nucmf);
        double chi_ab = cmf_interp_nu_asc(cs->nu, &cs->chi_abs[(size_t)s * NB], NB, nucmf);
        if (chi_es < 0.0) chi_es = 0.0;
        if (chi_ab < 0.0) chi_ab = 0.0;
        double chi_c = chi_es + chi_ab;
        double Jv = cmf_interp_nu_asc(cs->nu, &cs->J[(size_t)s * NB], NB, nucmf);
        double B  = cm_planck(nucmf, Te_s);
        double S_c = (chi_c > 0.0) ? (chi_ab * B + chi_es * Jv) / chi_c : 0.0;
        double alpha = q * chi_c;
        double dtau  = alpha * ds; if (dtau < 0.0) dtau = 0.0;
        double ex  = (dtau > 700.0) ? 0.0 : exp(-dtau);
        double psi = (dtau > 1e-4) ? (1.0 - ex) : (dtau - 0.5 * dtau * dtau);
        I = I * ex + (D * D * D) * S_c * psi;
        /* line resonances crossed in [nlo,nhi] of this sub-step's comoving sweep */
        double nu_a = cmf_q_at_z(p, za, inv_ct) * nu_obs;
        double nu_b = cmf_q_at_z(p, zb, inv_ct) * nu_obs;
        double nlo = (nu_a < nu_b) ? nu_a : nu_b;
        double nhi = (nu_a < nu_b) ? nu_b : nu_a;
        /* line_list_nu DESCENDING: first index with nu_l <= nhi */
        int lo = 0, hi = NL;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (opac->line_list_nu[mid] > nhi) lo = mid + 1; else hi = mid;
        }
        /* half-open (nlo, nhi]: a line on a sub-step boundary fires once (codex #3) */
        for (int l = lo; l < NL && opac->line_list_nu[l] > nlo; ++l) {
            double tauS = opac->tau_sobolev[(size_t)l * NS + s];
            if (!(tauS > 1e-6)) continue;            /* skips NaN too (codex #6) */
            /* Resonance-line source: scattering S_l=Jbar (CLEAN continuum mean
             * intensity = diluted backlight, NOT thermal B which refills the
             * line, NOR cs->J which is contaminated by the binned line — codex
             * #1). Prefer the cont_only field opac->jbar_line (LUMINA_CMF_JINC_
             * CONT); fall back to the local binned J. eps blends a thermal B
             * share for collision-thermalised lines (default 0 = pure scatter). */
            /* Resonance-scatter source = J_bar = the INCIDENT continuum field,
             * i.e. the geometrically-DILUTED photospheric backlight W(r)*B(T_phot)
             * — NOT the local cs->J, which is contaminated by the line's own
             * trapped radiation (binned line traps in its bin -> J/B~0.64 vs the
             * 0.5 clean continuum, self-refilling the trough). W(r) is the point-
             * photosphere dilution. (LUMINA_CMF_OBS_SRCJ=1 reverts to cs->J.) */
            double Jbar;
            if (g_sob_srcj) {
                Jbar = Jv;
                if (opac->jbar_line) {
                    double jl = opac->jbar_line[(size_t)l * NS + s];
                    if (jl > 0.0 && isfinite(jl)) Jbar = jl;
                }
            } else {
                double Wd = 0.5;
                if (g_sob_rphot > 0.0 && r > g_sob_rphot) {
                    double a = 1.0 - (g_sob_rphot * g_sob_rphot) / (r * r);
                    Wd = 0.5 * (1.0 - sqrt(a > 0.0 ? a : 0.0));
                }
                Jbar = Wd * cm_planck(nucmf, g_sob_Tinner);
            }
            double Sl = (1.0 - g_sob_eps) * Jbar + g_sob_eps * B;
            if (g_sob_diag && tauS > 1e4) {
                static int nd = 0;
                if (nd < 8) { nd++;
                    printf("[SOB-SRC] s=%d nu_l=%.4e tauS=%.2e Jbar=%.4e B=%.4e "
                           "Sl/B=%.4f jbar_set=%d\n", s, opac->line_list_nu[l],
                           tauS, Jbar, B, (B>0?Sl/B:0.0), opac->jbar_line?1:0); }
            }
            double exl = (tauS > 700.0) ? 0.0 : exp(-tauS);
            I = I * exl + (g_sob_noemit ? 0.0 : (D * D * D) * Sl) * (1.0 - exl);
        }
    }
    return I;
}

/* Observer-frame emergent spectrum (gate LUMINA_CMF_OBSERVER_FRAME=1).
 *
 * Same tangent-ray geometry as cmfgen_write_spectrum, but a SEPARATE formal
 * solve per observer frequency nu_obs: along each ray the comoving frequency
 * that contributes is nu_cmf(z) = q*nu_obs with q = gamma(1 - mu*beta),
 * beta = r/(c t_exp), mu = +-z/r (signed: inbound far side mu<0, outbound near
 * side mu>0). Material coefficients are evaluated at nu_cmf by interpolation;
 * the moving frame enters as alpha_obs = q*chi_cmf and S_obs = D^3*S_cmf
 * (D=1/q), and the diffusive core emits D_core^3 * B(q_core*nu_obs, T_inner).
 * beta->0 reproduces the comoving cmfgen_write_spectrum. (codex 019eefe6) */
int cmfgen_write_spectrum_obs(const CMFGENState *cs, const Geometry *geo,
                              double T_inner, const OpacityState *opac,
                              const double *Te, const char *path)
{
    int NS = cs->n_shells, NB = cs->n_bins, NR = cs->n_rays;
    double t_exp  = geo->time_explosion;
    double inv_ct = 1.0 / (CM_C * t_exp);
    /* S4: line-resolved Sobolev line synthesis (default ON when line data is
     * available). Replaces the defanged binned chi_line with full-tau_S
     * resonance jumps. Set LUMINA_CMF_OBS_SOBOLEV=0 for the legacy binned path. */
    int sob = (opac && opac->line_list_nu && opac->tau_sobolev && Te);
    { const char *e = getenv("LUMINA_CMF_OBS_SOBOLEV");
      if (e) sob = sob && atoi(e); }
    { const char *e = getenv("LUMINA_CMF_LINE_SOB_EPS");
      g_sob_eps = e ? atof(e) : 0.0;
      if (g_sob_eps < 0.0) g_sob_eps = 0.0;
      if (g_sob_eps > 1.0) g_sob_eps = 1.0; }
    { const char *e = getenv("LUMINA_CMF_OBS_NOEMIT"); g_sob_noemit = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_DIAG"); g_sob_diag = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_SRCJ"); g_sob_srcj = e ? atoi(e) : 0; }
    g_sob_rphot  = geo->r_inner[0];
    g_sob_Tinner = T_inner;

    /* per-(shell,bin) source S = S_fixed + (chi_es/chi_tot) J for interpolation.
     * Interpolate chi_tot and S directly (not eta=chi*S then /chi) to avoid an
     * off-node 0/0 ratio mismatch between independent interpolations. */
    double *Sbin = malloc(sizeof(double) * (size_t)NS * NB);
    double *Lnu  = calloc(NB, sizeof(double));
    if (!Sbin || !Lnu) {
        free(Sbin); free(Lnu);
        fprintf(stderr, "[CMFGEN] obs-spectrum alloc failed\n");
        return -1;
    }
    for (int s = 0; s < NS; ++s)
        for (int b = 0; b < NB; ++b) {
            size_t idx = (size_t)s * NB + b;
            double r = (cs->chi_tot[idx] > 0.0)
                     ? cs->chi_es[idx] / cs->chi_tot[idx] : 0.0;
            Sbin[idx] = cs->S_fixed[idx] + r * cs->J[idx];
        }

    /* Dense observer disk-ray grid: a sharp line P-Cygni needs the iso-velocity
     * annuli (mu*v(r) = const) resolved across the disk; the ~9-ray plasma grid
     * (cs->p_ray) is far too coarse and clips the trough. Uniform in p (=equal
     * area weight p dp). Continuum is unaffected (smooth). LUMINA_CMF_OBS_NRAY. */
    int NRO = 256; { const char *e = getenv("LUMINA_CMF_OBS_NRAY"); if (e) NRO = atoi(e); }
    double r_max_obs = geo->r_outer[NS - 1];
    double *p_obs = malloc(sizeof(double) * NRO);
    for (int kk = 0; kk < NRO; ++kk) p_obs[kk] = r_max_obs * (kk + 0.5) / (double)NRO;

    /* per-observer-frequency formal solve */
    for (int k = 0; k < NB; ++k) {
        double nu_obs = cs->nu[k];
        double integ = 0.0, p_prev = 0.0, f_prev = 0.0;

        for (int ray = 0; ray < NRO; ++ray) {
            double p = p_obs[ray];
            /* intersected shells, outer -> inner (descending r) */
            int    sh[256]; double rmid[256]; int nshell = 0;
            for (int s = NS - 1; s >= 0 && nshell < 256; --s) {
                double ro = geo->r_outer[s];
                if (ro <= p) break;
                double rm = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
                if (rm <= p) rm = p * 1.0000001;
                sh[nshell] = s; rmid[nshell] = rm; ++nshell;
            }
            if (nshell == 0) continue;
            int core = (p < geo->r_inner[0]) ? 1 : 0;

            /* segment z-extents (|z| at shell midpoints), outer->inner */
            double zabs[256];
            for (int i = 0; i < nshell; ++i) {
                double a = rmid[i] * rmid[i] - p * p;
                zabs[i] = (a > 0.0) ? sqrt(a) : 0.0;
            }
            double z_core = 0.0;
            if (core) {
                double ri0 = geo->r_inner[0];
                z_core = sqrt(ri0 * ri0 - p * p);
                if (nshell > 0 && z_core > zabs[nshell - 1]) z_core = zabs[nshell - 1];
            }

            double I = 0.0;   /* outer BC: no incoming radiation */

            /* ---- inbound (far side, z<0): outer -> inner, z increasing ---- */
            for (int i = 0; i < nshell; ++i) {
                if (sob) {
                    /* full shell radial extent [r_inner,r_outer] (NOT the midpoint
                     * — the midpoint collapses the shell to half-thickness and
                     * clips the high-z/high-blueshift resonance covering). */
                    double ro = geo->r_outer[sh[i]], ri = geo->r_inner[sh[i]];
                    double z_hi = (ro > p) ? sqrt(ro * ro - p * p) : 0.0;
                    double z_lo = (ri > p) ? sqrt(ri * ri - p * p) : 0.0;
                    I = cmf_obs_march_sob(I, p, -z_hi, -z_lo, sh[i], cs, opac,
                                          Te[sh[i]], nu_obs, inv_ct);
                } else {
                    double z_hi = zabs[i];                       /* outer edge */
                    double z_lo = (i + 1 < nshell) ? zabs[i + 1]
                                : (core ? z_core : 0.0);          /* inner edge */
                    I = cmf_obs_march(I, p, -z_hi, -z_lo, sh[i], cs->nu, cs->chi_tot,
                                      Sbin, NB, nu_obs, inv_ct, cs->d_log_nu);
                }
            }

            /* ---- diffusive core: D_core^3 * B(q_core*nu_obs, T_inner) ---- */
            if (core) {
                double ri0 = geo->r_inner[0];
                double mu_c = z_core / ri0;
                double beta_in = ri0 * inv_ct;
                double gam_in = 1.0 / sqrt(1.0 - beta_in * beta_in);
                double q_c = gam_in * (1.0 - mu_c * beta_in);
                double D_c = 1.0 / q_c;
                I = (D_c * D_c * D_c) * cm_planck(q_c * nu_obs, T_inner);
            }

            /* ---- outbound (near side, z>0): inner -> outer, z increasing ---- */
            for (int i = nshell - 1; i >= 0; --i) {
                if (sob) {
                    double ro = geo->r_outer[sh[i]], ri = geo->r_inner[sh[i]];
                    double z_hi = (ro > p) ? sqrt(ro * ro - p * p) : 0.0;
                    double z_lo = (ri > p) ? sqrt(ri * ri - p * p) : 0.0;
                    I = cmf_obs_march_sob(I, p, +z_lo, +z_hi, sh[i], cs, opac,
                                          Te[sh[i]], nu_obs, inv_ct);
                } else {
                    double z_hi = zabs[i];                       /* outer edge */
                    double z_lo = (i + 1 < nshell) ? zabs[i + 1]
                                : (core ? z_core : 0.0);          /* inner edge */
                    I = cmf_obs_march(I, p, +z_lo, +z_hi, sh[i], cs->nu, cs->chi_tot,
                                      Sbin, NB, nu_obs, inv_ct, cs->d_log_nu);
                }
            }

            double f = I * p;
            integ += 0.5 * (f_prev + f) * (p - p_prev);
            p_prev = p; f_prev = f;
        }
        Lnu[k] = 8.0 * M_PI * M_PI * integ;
    }

    FILE *fp = fopen(path, "w");
    if (!fp) { free(Sbin); free(Lnu); free(p_obs);
        fprintf(stderr, "[CMFGEN] cannot open %s\n", path); return -1; }
    fprintf(fp, "wavelength_angstrom,flux\n");
    for (int b = NB - 1; b >= 0; --b) {
        double lam_cm = CM_C / cs->nu[b];
        double lam_A  = lam_cm * 1.0e8;
        double L_lam  = Lnu[b] * CM_C / (lam_cm * lam_cm) * 1.0e-8;
        fprintf(fp, "%.6f,%.6e\n", lam_A, L_lam);
    }
    fclose(fp);
    printf("Pure-CMFGEN OBSERVER-frame spectrum -> %s (%d bins, beta_in=%.4f, "
           "lines=%s)\n", path, NB, geo->r_inner[0] * inv_ct,
           sob ? "SOBOLEV-resolved" : "binned");

    free(Sbin); free(Lnu); free(p_obs);
    return 0;
}

/* ------------------------------------------------------------ */
void cmfgen_write_jnu(const CMFGENState *cs, NLTEConfig *nlte)
{
    if (!nlte || !nlte->J_nu) return;
    size_t n = (size_t)cs->n_shells * cs->n_bins;
    memcpy(nlte->J_nu, cs->J, sizeof(double) * n);
}

/* ------------------------------------------------------------ */
/* ============================================================ */
/* P1: line-resolved comoving-frame (CMF) J solver (gate LUMINA_CMF_LINERES=1).
 * Replaces the per-bin frequency-DECOUPLED formal_solve_bin with a single
 * frequency-COUPLED sweep over all bins (b descending = blue->red), using the
 * validated PHOENIX/Hauschildt-Baron conservative upwind (chih=chi+a_lam*(4+
 * lambda/Dlam)) + Olson-Kunasz linear SC + tangent-ray mu-quadrature. Operates
 * on the SAME 1000-bin grid as the binned solver; gate 1a (cont_only) must
 * reproduce cmfgen_solve_J to <0.5% L2. cmfgen_solve_J is the untouched fallback.
 *   LUMINA_CMF_ALAM=0 disables the homologous freq advection (static limit, the
 *   transport-only sub-gate); =1 (default) the full CMF coupling.
 * Source = S_fixed + (chi_es/chi_tot)*J (the ALI scattering source, same as the
 * binned solver). Validated standalone in lumina_cmf_selftest.c (gates 2a/4a/2c). */
static void cmf_solve_J(CMFGENState *cs, const Geometry *geo, double T_inner,
                        int n_ali_iter)
{
    int NS = cs->n_shells, NB = cs->n_bins;
    double t_exp = geo->time_explosion;
    double a_lam_on = 1.0;
    { const char *e = getenv("LUMINA_CMF_ALAM"); if (e) a_lam_on = atof(e); }
    double a_lam = a_lam_on / (t_exp * CM_C);

    double *rmid = malloc(NS * sizeof(double));
    double *lam  = malloc(NB * sizeof(double));
    for (int s = 0; s < NS; ++s) rmid[s] = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
    for (int b = 0; b < NB; ++b) lam[b] = CM_C / cs->nu[b];  /* nu asc -> lam desc */

    int NCORE = 16, NP = NS + NCORE;
    double *p = malloc(NP * sizeof(double));
    for (int k = 0; k < NCORE; ++k) p[k] = rmid[0] * k / (double)NCORE;
    for (int s = 0; s < NS; ++s) p[NCORE + s] = rmid[s];
    int    *rn   = calloc(NP, sizeof(int));
    int    *rsh  = malloc((size_t)NP * (NS + 1) * sizeof(int));
    double *rz   = malloc((size_t)NP * (NS + 1) * sizeof(double));
    int    *rcore= calloc(NP, sizeof(int));
    double *rzin = calloc(NP, sizeof(double));
    for (int k = 0; k < NP; ++k) {
        double pk = p[k]; int n = 0;
        for (int s = NS - 1; s >= 0; --s) {
            if (rmid[s] <= pk) break;
            rsh[(size_t)k*(NS+1)+n] = s; rz[(size_t)k*(NS+1)+n] = sqrt(rmid[s]*rmid[s]-pk*pk); ++n;
        }
        rn[k] = n; rcore[k] = (pk < rmid[0]); rzin[k] = rcore[k] ? sqrt(rmid[0]*rmid[0]-pk*pk) : 0.0;
    }
    double *Iin_p =calloc((size_t)NP*(NS+1),sizeof(double)),*Iout_p=calloc((size_t)NP*(NS+1),sizeof(double));
    double *Iin_c =calloc((size_t)NP*(NS+1),sizeof(double)),*Iout_c=calloc((size_t)NP*(NS+1),sizeof(double));
    double *muL=malloc((size_t)NS*NP*sizeof(double)),*IpL=malloc((size_t)NS*NP*sizeof(double)),*ImL=malloc((size_t)NS*NP*sizeof(double));
    int *cnt = malloc(NS * sizeof(int));
    double *S    = malloc((size_t)NS*NB*sizeof(double));
    double *Jnew = malloc((size_t)NS*NB*sizeof(double));

    for (int it = 0; it < n_ali_iter; ++it) {
        for (int s = 0; s < NS; ++s) for (int b = 0; b < NB; ++b) {
            size_t idx = (size_t)s*NB+b;
            double r = (cs->chi_tot[idx] > 0.0) ? cs->chi_es[idx]/cs->chi_tot[idx] : 0.0;
            S[idx] = cs->S_fixed[idx] + r * cs->J[idx];
        }
        memset(Iin_p, 0, (size_t)NP*(NS+1)*sizeof(double));
        memset(Iout_p,0, (size_t)NP*(NS+1)*sizeof(double));
        for (int b = NB - 1; b >= 0; --b) {                 /* bluest (high nu) first */
            double Dlam = (b < NB-1) ? (lam[b]-lam[b+1]) : 0.0;   /* >0 */
            double adv  = (b < NB-1) ? a_lam*(lam[b]/Dlam) : 0.0;
            double lam_b= (b < NB-1) ? lam[b+1] : lam[b];
            double Bin  = cm_planck(cs->nu[b], T_inner);
            memset(cnt, 0, NS*sizeof(int));
            for (int k = 0; k < NP; ++k) {
                int n = rn[k]; if (n == 0) continue; size_t kb = (size_t)k*(NS+1);
                double I = 0.0;
                for (int i = 0; i < n; ++i) {
                    int s = rsh[kb+i]; double ds = (i+1<n)?(rz[kb+i]-rz[kb+i+1]):(rz[kb+i]-rzin[k]);
                    size_t idx = (size_t)s*NB+b; double chi = cs->chi_tot[idx], chih = chi + a_lam*4.0 + adv;
                    double Su = (S[idx]*chi + ((b<NB-1)?a_lam*(lam_b/Dlam)*Iin_p[kb+i]:0.0))/(chih>0?chih:1.0);
                    double dtau=chih*ds, ex=exp(-dtau), e0,e1,wu,wd;
                    if(dtau>1e-4){e0=1-ex;e1=dtau-e0;wu=e0-e1/dtau;wd=e1/dtau;}else{wu=0.5*dtau;wd=0.5*dtau;}
                    I = I*ex + wu*Su + wd*Su; Iin_c[kb+i]=I;
                    int c=cnt[s]; muL[(size_t)s*NP+c]=rz[kb+i]/rmid[s]; ImL[(size_t)s*NP+c]=I;
                }
                if (rcore[k]) I = Bin;
                for (int i = n-1; i >= 0; --i) {
                    int s = rsh[kb+i]; double ds = (i+1<n)?(rz[kb+i]-rz[kb+i+1]):(rz[kb+i]-rzin[k]);
                    size_t idx = (size_t)s*NB+b; double chi = cs->chi_tot[idx], chih = chi + a_lam*4.0 + adv;
                    double Su = (S[idx]*chi + ((b<NB-1)?a_lam*(lam_b/Dlam)*Iout_p[kb+i]:0.0))/(chih>0?chih:1.0);
                    double dtau=chih*ds, ex=exp(-dtau), e0,e1,wu,wd;
                    if(dtau>1e-4){e0=1-ex;e1=dtau-e0;wu=e0-e1/dtau;wd=e1/dtau;}else{wu=0.5*dtau;wd=0.5*dtau;}
                    I = I*ex + wu*Su + wd*Su; Iout_c[kb+i]=I;
                    int c=cnt[s]; IpL[(size_t)s*NP+c]=I; cnt[s]=c+1;
                }
            }
            for (int s = 0; s < NS; ++s) {
                int c = cnt[s]; size_t idx=(size_t)s*NB+b;
                if (c < 1) { Jnew[idx] = S[idx]; continue; }
                for (int a=1;a<c;++a){double mk=muL[(size_t)s*NP+a],ip=IpL[(size_t)s*NP+a],im=ImL[(size_t)s*NP+a];int q=a-1;
                    while(q>=0&&muL[(size_t)s*NP+q]>mk){muL[(size_t)s*NP+q+1]=muL[(size_t)s*NP+q];IpL[(size_t)s*NP+q+1]=IpL[(size_t)s*NP+q];ImL[(size_t)s*NP+q+1]=ImL[(size_t)s*NP+q];--q;}
                    muL[(size_t)s*NP+q+1]=mk;IpL[(size_t)s*NP+q+1]=ip;ImL[(size_t)s*NP+q+1]=im;}
                double mu[300],jv[300];int q=0; mu[q]=0;jv[q]=0.5*(IpL[(size_t)s*NP+0]+ImL[(size_t)s*NP+0]);q++;
                for(int a=0;a<c;++a){mu[q]=muL[(size_t)s*NP+a];jv[q]=0.5*(IpL[(size_t)s*NP+a]+ImL[(size_t)s*NP+a]);q++;}
                double Js=0; for(int a=0;a+1<q;++a)Js+=0.5*(jv[a]+jv[a+1])*(mu[a+1]-mu[a]); Jnew[idx]=Js;
            }
            { double *t1=Iin_p;Iin_p=Iin_c;Iin_c=t1; double *t2=Iout_p;Iout_p=Iout_c;Iout_c=t2; }
        }
        double maxrel = 0.0;
        for (size_t i = 0; i < (size_t)NS*NB; ++i) {
            double d = fabs(Jnew[i]-cs->J[i])/(fabs(cs->J[i])+1e-30); if (d>maxrel) maxrel=d;
            cs->J[i] = Jnew[i];
        }
        if (maxrel < 1e-4 && it > 0) break;
    }
    free(rmid);free(lam);free(p);free(rn);free(rsh);free(rz);free(rcore);free(rzin);
    free(Iin_p);free(Iout_p);free(Iin_c);free(Iout_c);free(muL);free(IpL);free(ImL);free(cnt);free(S);free(Jnew);
}

int cmfgen_run(Geometry *geo, OpacityState *opac, BFOpacity *bf,
               PlasmaState *plasma, NLTEConfig *nlte, AtomicData *atom,
               GammaDeposition *gamma, double T_inner, int n_iter)
{
    CMFGENState cs;
    if (cmfgen_init(&cs, geo) != 0) return -1;

    /* P1 gate 1a: LUMINA_CMF_CONTONLY=1 forces continuum-only assemble (line
     * opacity zeroed) so the CMF J-producer can be compared to cmfgen_solve_J
     * on identical continuum opacity. */
    int cmf_lineres = 0;
    { const char *e = getenv("LUMINA_CMF_LINERES"); if (e) cmf_lineres = atoi(e); }
    { const char *e = getenv("LUMINA_CMF_CONTONLY"); if (e && atoi(e)) cs.cont_only = 1; }

    const char *ali_env = getenv("LUMINA_CMFGEN_ALI_ITER");
    int n_ali = ali_env ? atoi(ali_env) : 8;
    if (n_ali < 1) n_ali = 1;

    printf("[CMFGEN] pure deterministic radiation driver: %d shells, %d bins, "
           "%d rays, %d outer iters, %d ALI/iter, t_exp=%.4e s\n",
           cs.n_shells, cs.n_bins, cs.n_rays, n_iter, n_ali, geo->time_explosion);

    double t_exp = geo->time_explosion;
    for (int iter = 0; iter < n_iter; ++iter) {
        if (nlte) nlte->current_iter = iter;

        /* refresh bf opacity for current ionization/T_e */
        if (bf) compute_bf_opacity(bf, atom, plasma, cs.n_shells);

        cmfgen_assemble(&cs, geo, opac, bf, plasma);
        if (cmf_lineres) cmf_solve_J(&cs, geo, T_inner, n_ali);   /* P1 line-resolved CMF */
        else             cmfgen_solve_J(&cs, geo, T_inner, n_ali);/* binned (champion) */
        cmfgen_window_color(&cs);
        radeq_set_tail_color(cs.t_color, cs.n_shells);
        radeq_set_tri_response(cs.tri_lo, cs.tri_up, cs.tri_r,
                               cs.n_shells, cs.n_bins);
        if (cs.diag && iter == n_iter - 1)
            cmfgen_validate(&cs, geo, plasma);
        cmfgen_write_jnu(&cs, nlte);

        /* CORRECTED FALSIFIER (LUMINA_CMF_JINC_CONT=1, codex 2026-06-22): compute the
         * CLEAN external continuum field J_inc (cont_only solve, ALL line opacity zeroed
         * => no line self-emission, no super-thermal contamination, bounded) and sample
         * it per line into opac->jbar_line so the mode-3 bb-rate hook (LUMINA_NLTE_JBAR_
         * POPS=3) pumps R_lu=B_lu*beta_l*J_inc_cont. Decisive: max(S_l/B)<=1e6 => mode-3
         * FORM is correct (the sealed MC explosion was the contaminated input, NOT the
         * form); max(S_l/B)>1e10 => form unstable => full MALI needed. cs.J + opacity are
         * saved/restored so continuum rates, line-RE, and the spectrum are unaffected. */
        {
            static int jinc_cont = -1;
            if (jinc_cont < 0) { const char *e = getenv("LUMINA_CMF_JINC_CONT");
                jinc_cont = (e && atoi(e)) ? 1 : 0; }
            if (jinc_cont && opac->tau_sobolev && opac->line_list_nu) {
                size_t NS = cs.n_shells, NB = cs.n_bins; int NL = opac->n_lines;
                if (!opac->jbar_line)  opac->jbar_line  = (double*)calloc((size_t)NL*NS, sizeof(double));
                if (!opac->jbar_count) opac->jbar_count = (int*)   calloc((size_t)NL*NS, sizeof(int));
                static double *Jsave = NULL;
                if (!Jsave) Jsave = (double*)malloc(NS*NB*sizeof(double));
                memcpy(Jsave, cs.J, NS*NB*sizeof(double));     /* save full J */
                cs.cont_only = 1;
                cmfgen_assemble(&cs, geo, opac, bf, plasma);   /* line opacity zeroed */
                cmfgen_solve_J(&cs, geo, T_inner, n_ali);      /* cs.J = J_inc_cont (continuum) */
                for (size_t s = 0; s < NS; ++s)
                    for (int l = 0; l < NL; ++l) {
                        size_t k = (size_t)l*NS + s;
                        double tau = opac->tau_sobolev[k];
                        double nu_l = opac->line_list_nu[l];
                        if (tau <= 1e-12 || nu_l <= cs.nu_min || nu_l >= cs.nu_max) {
                            opac->jbar_count[k] = 0; continue; }
                        int b = (int)floor(log(nu_l / cs.nu_min) / cs.d_log_nu);
                        if (b < 0 || b >= (int)NB) { opac->jbar_count[k] = 0; continue; }
                        opac->jbar_line[k]  = cs.J[s*NB + (size_t)b]; /* clean external J_inc */
                        opac->jbar_count[k] = 1000;                   /* pass use_jbar guard */
                    }
                /* CROSS-LINE OVERLAP (4a, LUMINA_CMF_OVERLAP=1, codex 2026-06-22): per shell,
                 * blue->red sweep carrying the redshifted ESCAPING emission of bluer lines into
                 * redder lines' incident field = the UV->optical fluorescence carrier (form
                 * validated by self-test 4a, +175%). J_emit = f_out*beta_l*S_l_lag (escaping
                 * part, NOT the trapped (1-beta)S_l); J_overlap attenuates by exp(-dtau_cont)
                 * over the frequency gap (Sobolev dr_res=(dnu/nu)*c*t_exp). cs.chi_es/chi_abs
                 * are continuum-only here (cont_only state). Lagged: S_l from previous-pass pops.
                 * Order: use J_overlap for line l, THEN add l's emission (no self-pump). */
                {
                    static int ovl = -1; static double fout = 0.5;
                    if (ovl < 0) { const char *e = getenv("LUMINA_CMF_OVERLAP"); ovl = (e&&atoi(e))?1:0;
                        const char *fo = getenv("LUMINA_OVERLAP_FOUT"); if (fo) fout = atof(fo); }
                    if (ovl) {
                        double t_exp_l = geo->time_explosion;
                        for (size_t s = 0; s < NS; ++s) {
                            double Jov = 0.0, rmax = 1.0;
                            for (int l = 0; l < NL; ++l) {   /* line_list_nu DESCENDING = blue->red */
                                double nu_l = opac->line_list_nu[l];
                                if (nu_l <= cs.nu_min || nu_l >= cs.nu_max) continue;
                                int b = (int)floor(log(nu_l/cs.nu_min)/cs.d_log_nu);
                                if (b < 0 || b >= (int)NB) continue;
                                size_t k = (size_t)l*NS + s;
                                double tau = opac->tau_sobolev[k];
                                if (tau > 1e-12) {
                                    double Jcont = opac->jbar_line[k];          /* continuum J_inc */
                                    opac->jbar_line[k] = Jcont + Jov;           /* + overlap incident */
                                    opac->jbar_count[k] = 1000;
                                    if (Jcont > 0.0 && (Jcont+Jov)/Jcont > rmax) rmax = (Jcont+Jov)/Jcont;
                                    double beta = (tau > 1e-6) ? -expm1(-tau)/tau : 1.0;
                                    double S = opac->line_source_S[k];
                                    if (S > 0.0) Jov += fout * beta * S;        /* escaping emission */
                                }
                                if (l < NL-1) {                                  /* attenuate over gap */
                                    double dnu = nu_l - opac->line_list_nu[l+1];
                                    if (dnu > 0.0) {
                                        double chic = cs.chi_es[s*NB+(size_t)b] + cs.chi_abs[s*NB+(size_t)b];
                                        Jov *= exp(-chic * (dnu/nu_l) * CM_C * t_exp_l);
                                    }
                                }
                            }
                            if (rmax > 1e3)
                                printf("  [OVERLAP] shell %zu max(J_inc/J_cont)=%.1e (runaway watch)\n", s, rmax);
                        }
                    }
                }
                cs.cont_only = 0;
                cmfgen_assemble(&cs, geo, opac, bf, plasma);   /* restore full opacity */
                memcpy(cs.J, Jsave, NS*NB*sizeof(double));     /* restore full J */
            }
        }
        /* Option-2 integral RE: register the CMFGEN line opacity/source for the
         * RADEQ/Newton T_e solve (LUMINA_RADEQ_LINE_RE=1). */
        /* RE/Newton line channel: chi_line_th (physical thermal share) except
         * in transfer-only eps_uv mode, where the closure keeps FULL chi_line
         * (cooling-only form in radeq_line_re; codex ruling). */
        radeq_set_line_re_source(cs.chi_line_re, cs.chi_abs, cs.chi_tot,
                                 cs.S_fixed, cs.J, cs.nu, cs.dnu,
                                 cs.lambda_star, plasma->T_e,
                                 cs.chi_line, cs.chi_line_cls,
                                 cs.n_shells, cs.n_bins);

        /* downstream solvers reused unchanged */
        compute_radiative_equilibrium_te(plasma, gamma, nlte, atom, opac,
                                         t_exp, cs.n_shells);
        compute_plasma_state(atom, plasma, opac, t_exp);
        if (nlte && nlte->enabled) {
            /* UNDER-RELAXED lagged solve (codex requirement): when the cont_only-J_inc
             * mode-3 pump is active (LUMINA_CMF_JINC_CONT), damp the population update
             * between outer passes (pops = (1-a)*old + a*new, a=LUMINA_JBAR_POPS_DAMP,
             * default 0.3) so the lagged jbar->bb->pop loop cannot oscillate to runaway.
             * jbar_line is held fixed during the solve (computed above from current pops
             * = lagged); the damping tames the outer loop. */
            static int jdamp = -1; static double jeta = 0.3;
            if (jdamp < 0) { const char *e = getenv("LUMINA_CMF_JINC_CONT");
                jdamp = (e && atoi(e)) ? 1 : 0;
                const char *d = getenv("LUMINA_JBAR_POPS_DAMP"); if (d) jeta = atof(d); }
            if (jdamp && nlte->nlte_level_populations) {
                size_t npop = (size_t)nlte->n_nlte_levels_total * cs.n_shells;
                static double *pop_old = NULL; static size_t pop_n = 0;
                if (pop_n != npop) { free(pop_old);
                    pop_old = (double*)malloc(npop*sizeof(double)); pop_n = npop; }
                memcpy(pop_old, nlte->nlte_level_populations, npop*sizeof(double));
                nlte_solve_all(nlte, atom, plasma, opac, t_exp, cs.n_shells, gamma);
                for (size_t k = 0; k < npop; ++k)
                    nlte->nlte_level_populations[k] =
                        (1.0 - jeta)*pop_old[k] + jeta*nlte->nlte_level_populations[k];
                nlte_update_tau_sobolev(nlte, atom, opac, t_exp, cs.n_shells);
            } else {
                nlte_solve_all(nlte, atom, plasma, opac, t_exp, cs.n_shells, gamma);
            }
        }

        if (cs.diag) {
            int mid = cs.n_shells / 2;
            printf("[CMFGEN] iter %2d: T_e[0]=%.0fK T_e[%d]=%.0fK T_e[%d]=%.0fK "
                   "J[mid,bin500]=%.3e\n",
                   iter, plasma->T_e[0], mid, plasma->T_e[mid],
                   cs.n_shells - 1, plasma->T_e[cs.n_shells - 1],
                   cs.J[(size_t)mid * cs.n_bins + 500]);
        }
    }

    /* Default frame = observer (all comparisons to gold are observer-frame);
     * set LUMINA_CMF_OBSERVER_FRAME=0 for the legacy comoving spectrum. */
    const char *obs_env = getenv("LUMINA_CMF_OBSERVER_FRAME");
    int obs_frame = obs_env ? atoi(obs_env) : 1;
    if (obs_frame) {
        cmfgen_write_spectrum_obs(&cs, geo, T_inner, opac, plasma->T_e,
                                  "lumina_spectrum.csv");
        cmfgen_write_spectrum(&cs, geo, T_inner, "lumina_spectrum_comoving.csv");
    } else {
        cmfgen_write_spectrum(&cs, geo, T_inner, "lumina_spectrum.csv");
    }
    cmfgen_free(&cs);
    return 0;
}
