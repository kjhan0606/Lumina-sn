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

/* GPU comoving-frame formal solver (lumina_cmf_solve.cu); -1 -> CPU fallback. */
int cmf_solve_J_gpu(int NS, int NB, int NP, int adv_split, double a_lam,
    const double *chi_tot, const double *chi_es, const double *chi_abs,
    const double *S_fixed, double *J,
    const double *Bin, const double *adv_b, const double *advcoef_b,
    const int *rn, const int *rsh, const double *rz,
    const int *rcore, const double *rzin,
    const int *shell_off, const int *shell_k, const int *shell_seg,
    const double *shell_mu, int nsamp,
    int n_ali_iter, double tol, int *iters_out);

static int cmf_dcmp(const void *a, const void *b) {
    double d = *(const double*)a - *(const double*)b;
    return (d > 0) - (d < 0);
}

static inline double cm_planck(double nu, double T) {
    if (T <= 0.0) return 0.0;
    double x = CM_H * nu / (CM_KB * T);
    if (x > 7.0e2) return 0.0;            /* underflow guard */
    double denom = expm1(x);
    if (denom <= 0.0) return 0.0;
    return (2.0 * CM_H * nu * nu * nu) / (CM_C * CM_C * denom);
}

/* Inner blackbody amplitude dilution (LUMINA_INNER_BB_SCALE, default 1.0). Read
 * once, shared by the J solve AND the emergent spectrum writers so the inner BC
 * is consistent. Previously only cmfgen_solve_J applied it; the emergent writers
 * (cmfgen_write_spectrum, cmfgen_write_spectrum_obs) used the full T_inner BB, so
 * INNER_BB_SCALE=0 diluted the solved field but NOT the emergent spectrum. */
static double cmf_inner_bb_scale(void) {
    static int init = 0; static double s = 1.0;
    if (!init) { const char *e = getenv("LUMINA_INNER_BB_SCALE");
                 if (e) s = atof(e); init = 1; }
    return s;
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

/* Stage 1 (transport-coupled T_e): radioactive deposition heating per shell
 * [erg/s/cm^3], registered before cmfgen_assemble so the formal-solve emissivity
 * can carry it (gate LUMINA_CMF_DEP_SOURCE). Setter avoids changing the assemble
 * signature (5 call sites). NULL => no injection (byte-identical). */
static const double *g_dep_heating = NULL;
static int g_dep_heating_n = 0;
void cmfgen_set_deposition(const double *heating_rate, int n_shells) {
    g_dep_heating = heating_rate; g_dep_heating_n = n_shells;
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

    /* Stage 1: thermalised radioactive deposition into the transport source.
     * Inject eta_dep,nu = kappa*B_nu(T_e) so 4pi*Int eta_dep dnu = frac*H_gamma[s],
     * with kappa normalised on the ACTUAL bin grid (kappa = frac*H_gamma/
     * (4pi*Sum_b B_nu*dnu)) so the injected power equals frac*H_gamma exactly
     * (no analytic-sigma-T^4 grid-truncation error). Added to S_fixed as
     * eta_dep/chi_tot below. Gate LUMINA_CMF_DEP_SOURCE (default off => no-op). */
    static int dep_src = -1; static double dep_frac = 1.0;
    if (dep_src < 0) {
        const char *e = getenv("LUMINA_CMF_DEP_SOURCE");
        dep_src = (e && atoi(e)) ? 1 : 0;
        const char *fe = getenv("LUMINA_CMF_DEP_FRAC");
        if (fe) dep_frac = atof(fe);
    }

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

    /* LUMINA_CMF_EPAY=1: energy-paid thermal emission (deterministic kpkt
     * mirror / per-shell transport radiative equilibrium). The thermal source
     * (chi_a*B + eta_line) is rescaled per shell so its frequency-integrated
     * power equals what the gas actually absorbs from the (lagged) field:
     *   scale_s = [Sum_b (chi_a+chi_line_th) J dnu] / [Sum_b (chi_a B+eta_ln) dnu]
     * The deposition injection kappa_dep*B is already exactly H_dep and stays
     * unscaled. Fixed point = radiative equilibrium (emit = abs + dep), the
     * same closure CMFGEN's global linearization and ARTIS's e-packet
     * bookkeeping enforce — a conservation constraint, not a tuning knob.
     * iter0 (J=0) => thermal dark start (physical: the ejecta history never
     * visits the hot state); the unpaid mutual-illumination bath (hot band
     * s36-40, ledger 1e6-1e7x) cannot ignite because a shell can never emit
     * more than it absorbed + deposition. */
    static int epay = -1;
    static int epay_smin = 0;   /* diagnostic: EPAY only for s >= smin */
    static double epay_tau = 2.0;
    static double epay_taubin = 1.0;   /* per-bin thick exemption threshold */
    if (epay < 0) { const char *e = getenv("LUMINA_CMF_EPAY");
                    epay = e ? atoi(e) : 0;
                    if (epay < 0) epay = 0;
                    const char *et = getenv("LUMINA_CMF_EPAY_TAU");
                    if (et) epay_tau = atof(et);
                    const char *es = getenv("LUMINA_CMF_EPAY_SMIN");
                    if (es) epay_smin = atoi(es);
                    const char *tb = getenv("LUMINA_CMF_EPAY_TAUBIN");
                    if (tb) epay_taubin = atof(tb);   /* <=0: no exemption */
                    if (epay) printf("[CMF-EPAY] energy-paid thermal emission ON"
                                     " (tau_es < %.2f only; thick=LTE legacy)\n",
                                     epay_tau); }
    double epay_scale_dbg[4] = {0,0,0,0};
    /* inward electron-scattering optical depth: EPAY applies only to thin
     * shells (tau_es < EPAY_TAU). At depth LTE holds, chi*B IS the honest
     * emission and the books close naturally — enforcing the lagged-J scale
     * there just recreates Lambda-iteration slowness (epay1: s0 stuck 0.68x). */
    double *epay_tau_arr = NULL;
    if (epay) {
        /* LUMINA_CMF_EPAY_TAUEFF (default 1e4): Planck-weighted effective
         * absorption depth sqrt(chi_abs*chi_tot)*dr, accumulated inward.
         * The toy06 profile has a 26x cliff exactly at the diffusive-core
         * boundary (s4: 2.6e4 -> s5: 1.0e3): the core keeps the legacy LTE
         * chi*B source (books close at ~1.1 naturally; EPAY's lagged-J scale
         * would Lambda-stall there), everything outside is EPAY-enforced.
         * Replaces the diagnostic shell-index gate EPAY_SMIN. */
        static double epay_taueff = 0.0;   /* 0 = disabled (gate by EPAY_SMIN;
            * the toy06 diffusive-core boundary s4/s5 is a verified 26x cliff
            * in offline emission-weighted tau_eff — scripts note in design
            * doc). The in-code coarse Planck-weighted variant under-measures
            * the core (misses the UV/line-dominated bins): epay9 regression
            * s0-4 -40%. Opt-in via LUMINA_CMF_EPAY_TAUEFF until the measure
            * is emission-weighted. */
        { static int te_once = 0;
          if (!te_once) { te_once = 1;
              const char *tf = getenv("LUMINA_CMF_EPAY_TAUEFF");
              if (tf) epay_taueff = atof(tf); } }
        epay_tau_arr = (double *)calloc(NS, sizeof(double));
        double acc_tau = 0.0;
        if (epay_taueff > 0.0)
        for (int s2 = NS - 1; s2 >= 0; --s2) {
            double Te2 = plasma->T_e[s2];
            double dr2 = geo->r_outer[s2] - geo->r_inner[s2];
            double wsum = 0.0, ca = 0.0, ct = 0.0;
            for (int b2 = 0; b2 < NB; b2 += 8) {   /* coarse Planck weights */
                size_t i2 = (size_t)s2 * NB + b2;
                double wgt = cm_planck(cs->nu[b2], Te2) * cs->dnu[b2];
                wsum += wgt;
                ca += wgt * cs->chi_abs[i2];
                ct += wgt * cs->chi_tot[i2];
            }
            if (wsum > 0.0) { ca /= wsum; ct /= wsum; }
            acc_tau += sqrt((ca > 0 ? ca : 0) * (ct > 0 ? ct : 0)) *
                       (dr2 > 0.0 ? dr2 : 0.0);
            epay_tau_arr[s2] = acc_tau;
        }
        if (epay_taueff > 0.0) epay_tau = epay_taueff;
        /* else: arr stays 0 < epay_tau — tau gate passes, EPAY_SMIN gates */
    }

    /* electron scattering + bf/ff thermal absorption + combine. */
    for (int s = 0; s < NS; ++s) {
        double Te  = plasma->T_e[s];
        double n_e = plasma->n_electron ? plasma->n_electron[s]
                                        : opac->electron_density[s];
        double chi_e = n_e * CM_SIGMA_T;
        /* Stage 1 deposition: kappa_dep = frac*H_gamma / (4pi*Sum_b B_nu*dnu). */
        double kappa_dep = 0.0;
        double acc_emit = 0.0, acc_abs = 0.0;   /* EPAY per-shell books */
        double acc_w = 0.0, acc_dep = 0.0;      /* EPAY=2 rate-shape norm */
        if (dep_src && g_dep_heating && s < g_dep_heating_n &&
            g_dep_heating[s] > 0.0 && Te > 0.0) {
            double bnorm = 0.0;
            for (int b = 0; b < NB; ++b) bnorm += cm_planck(cs->nu[b], Te) * cs->dnu[b];
            if (bnorm > 0.0)
                kappa_dep = dep_frac * g_dep_heating[s] / (4.0 * M_PI * bnorm);
        }
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
            /* EPAY books: thermal emitted vs absorbed (lagged J), per shell.
             * Per-bin THICK exemption (Kirchhoff limit): a locally thick bin
             * (tau_bin = (chi_a+chi_l,th)*dr > EPAY_TAUBIN) reabsorbs its own
             * emission — it cannot export unpaid energy, and its source MUST
             * relax to B (detailed balance) or the diffusive field dies (the
             * measured s5 J(16eV) = B/340, x3e5 below pre-EPAY: Si/S/Co stuck
             * at stage II -> the 5700-6000A blanket). Thick bins keep the
             * legacy chi*B source and are excluded from the books; the unpaid
             * -lamp problem only ever lived in EXPORTING (thin) bins. */
            int bin_thick = 0;
            if (epay) {
                double dr_s = geo->r_outer[s] - geo->r_inner[s];
                bin_thick = ((chi_a + chi_ln_th) * dr_s > epay_taubin);
            }
            if (epay && !bin_thick) {
                acc_emit += (chi_a * B + eta_ln) * cs->dnu[b];
                acc_abs  += (chi_a + chi_ln_th) * cs->J[idx] * cs->dnu[b];
                if (epay >= 2) {
                    /* rate-side emission shape: Milne spontaneous recombination
                     * (bf_get_eta, LUMINA_CMF_BF_MILNE=2 builder) + collisional
                     * line emissivity chi_l,th*B (= n_l C_lu h nu by the
                     * two-level identity — NOT eta_ln, whose macroatom
                     * line_source_S garbage bins (S_l/B~1e48 saga) concentrate
                     * the whole paid budget into one scattering-trapped IR bin
                     * and close a gain>1 loop: the epay6 J=1e107 runaway). */
                    acc_w   += ((bf ? bf_get_eta(bf, s, nu) : 0.0)
                                + chi_ln_th * B) * cs->dnu[b];
                    acc_dep += kappa_dep * B * cs->dnu[b];
                }
            }
            /* Stage 1: add the thermalised deposition emissivity eta_dep=kappa*B
             * to the source (eta_dep/chi_tot). No-op when the gate is off.
             * Under EPAY this is deferred to the rescale pass below. */
            if ((!epay || s < epay_smin ||
                 (epay_tau_arr && epay_tau_arr[s] >= epay_tau)) &&
                kappa_dep > 0.0 && chi_t > 0.0)
                cs->S_fixed[idx] += kappa_dep * B / chi_t;
            /* chi_tot scratch now overwritten with the real total */
            cs->chi_tot[idx] = chi_t;
        }
        if (epay && s >= epay_smin && epay_tau_arr[s] < epay_tau) {
            double scale = (acc_emit > 0.0) ? acc_abs / acc_emit : 0.0;
            /* rate-shape only in the NLTE lamp regime T_e >> T_rad: there
             * chi*B(T_e) is the proven unpaid-lamp form and Milne+lines is
             * honest. Near LTE (T_e ~ T_rad) the two forms agree (b->1,
             * Milne->B) BUT the Milne shape inherits the valley's corrupted
             * n_+ (recombination edges overweighted, epay4: n_e +0.3 dex) —
             * so keep the legacy chi*B shape there. */
            static double epay_hotf = 1.5;
            { static int hf_once = 0;
              if (!hf_once) { hf_once = 1;
                  const char *hf = getenv("LUMINA_CMF_EPAY_HOTF");
                  if (hf) epay_hotf = atof(hf); } }
            int hot_regime = (Te > epay_hotf * plasma->T_rad[s]);
            double dr_s = geo->r_outer[s] - geo->r_inner[s];
            if (epay >= 2 && acc_w > 0.0 && hot_regime) {
                /* kpkt mirror proper: paid power E_pay = absorbed + deposition,
                 * spectrum = normalized rate-side emissivity w(nu). Thick bins
                 * (Kirchhoff) keep the pass-1 legacy chi*B source untouched. */
                double E_pay = acc_abs + acc_dep;
                double wn = E_pay / acc_w;
                for (int b = 0; b < NB; ++b) {
                    size_t idx = (size_t)s * NB + b;
                    double chi_t = cs->chi_tot[idx];
                    if ((cs->chi_abs[idx] + cs->chi_line_th[idx]) * dr_s
                        > epay_taubin) {
                        if (kappa_dep > 0.0 && chi_t > 0.0)
                            cs->S_fixed[idx] += kappa_dep *
                                cm_planck(cs->nu[b], Te) / chi_t;
                        continue;   /* legacy Kirchhoff source */
                    }
                    double w = (bf ? bf_get_eta(bf, s, cs->nu[b]) : 0.0)
                             + cs->chi_line_th[idx] * cm_planck(cs->nu[b], Te);
                    cs->S_fixed[idx] = (chi_t > 0.0) ? wn * w / chi_t : 0.0;
                }
                scale = wn;   /* debug: report the shape norm instead */
            } else {
            for (int b = 0; b < NB; ++b) {
                size_t idx = (size_t)s * NB + b;
                double chi_t = cs->chi_tot[idx];
                if (!((cs->chi_abs[idx] + cs->chi_line_th[idx]) * dr_s
                      > epay_taubin))
                    cs->S_fixed[idx] *= scale;
                if (kappa_dep > 0.0 && chi_t > 0.0)
                    cs->S_fixed[idx] += kappa_dep * cm_planck(cs->nu[b], Te) / chi_t;
            }
            }
            if (s == 0) epay_scale_dbg[0] = scale;
            if (s == NS/2) epay_scale_dbg[1] = scale;
            if (s == 38 && s < NS) epay_scale_dbg[2] = scale;
            if (s == NS-1) epay_scale_dbg[3] = scale;
        }
    }
    if (epay)
        printf("[CMF-EPAY] scale s0=%.3e s%d=%.3e s38=%.3e s%d=%.3e\n",
               epay_scale_dbg[0], NS/2, epay_scale_dbg[1],
               epay_scale_dbg[2], NS-1, epay_scale_dbg[3]);
    free(epay_tau_arr);
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

    /* Inner-BB energy-balance scale (LUMINA_INNER_BB_SCALE, default 1.0): dilute
     * the inner blackbody source by this factor. When radioactive deposition is
     * ON, the inner-BB (central luminosity) + distributed deposition double-count
     * the energy and over-heat the plasma (drives T_e into the partial-ionization
     * max-line-opacity regime → MC packet trapping). Reducing the inner-BB lets
     * deposition supply the shell energy without the double-count (ARTIS uses
     * distributed deposition with NO inner blackbody). Keeps T_inner's COLOR;
     * only scales the amplitude (a dilution W_inner). */
    double inner_bb_scale = cmf_inner_bb_scale();
    for (int b = 0; b < NB; ++b) {
        double Bin = inner_bb_scale * cm_planck(cs->nu[b], T_inner);
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
        double Bin = cmf_inner_bb_scale() * cm_planck(cs->nu[b], T_inner);
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
static int    g_sob_contonly = 0;
static double g_sob_taumin = 1e-6;  /* min Sobolev tau for a line to fire (diag) */
static int    g_pray_bin = -1;
static double g_sob_rphot  = 0.0;   /* photosphere radius (r_inner[0]) for W(r) */
static double g_sob_Tinner = 0.0;   /* photosphere T for the diluted backlight */
static int    g_sob_srcj   = 0;     /* 1 = legacy cs->J source (contaminated) */
static int    g_sob_jbardet = 0;    /* 1 = ENERGY-CONSERVING scatter source S_l=Jbar_l
                                     * (producer jbar_line_det). W*B backlight (below
                                     * incident I) destroys ~47% of the flux. */
static int    g_sob_sei_cmv = 1;    /* 1 = SEI Pass-1 accumulates COMOVING I*q^3 (correct);
                                     * 0 = old observer-frame I (the +180% double-beam bug) */
static int    g_sob_sei = 0;        /* 1 = 2-pass SEI: lines jump at TRUE tau_S sourced
                                     * from the Pass-1 beamed continuum mean (jc[s]),
                                     * conserving AND sharp P-Cygni (2026-06-26 fix). */
static int    g_sob_faithful = 0;   /* 1 = transport static chi_tot+Sbin (conserving),
                                     * no Sobolev jumps — the obs energy fix (2026-06-26). */
/* Unified-emergent: drive the resonance line source from the NLTE-solved
 * line_source_S (fluorescence) instead of the W*B scattering backlight.
 * g_sob_sl_ptr points at opac->line_source_S[l*NS+s]; clamp caps S_l<=clamp*B. */
static const double *g_sob_sl_ptr = NULL;   /* NULL = scattering (default) */
static double        g_sob_sl_clamp = 0.0;  /* 0 = off */

/* S4 — LINE-RESOLVED Sobolev observer march (gate LUMINA_CMF_OBS_SOBOLEV).
 * The binned expansion opacity defangs tau_Sobolev (frac=1-e^{-tau}->1, /dnu_bin
 * => integrated ray tau ~ O(1), no P-Cygni; codex 019ef207 + agent ac7845ec).
 * Here the CONTINUUM is transported from chi_es+chi_abs (NOT the binned
 * chi_line), and each line whose rest nu_l lies in the sub-step's swept comoving
 * interval applies its FULL Sobolev jump  I = I*e^{-tau_S} + D^3*S_l*(1-e^{-tau_S})
 * at the resonance — exactly the directional resonance the MC does. Requires
 * LINE_EPS off (so chi_es is pure electron, no double-count). */
static double cmf_obs_march_sob_jc(const double *jc,
                                double I, double p, double z0, double z1, int s,
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
        if (g_sob_faithful) {
            /* FAITHFUL observer-frame transform of the CONSERVING static extractor
             * (physics review 2026-06-26): transport the SAME fine-grid chi_tot +
             * full source Sbin=S_fixed+r·J the static uses (lines RESOLVED on the
             * fine grid -> no Sobolev-jump hack). Conserves by construction; the
             * D⁴ beaming + P-Cygni asymmetry emerge from the line-of-sight geometry.
             * dtau=q·chi_tot·ds, source D³·Sbin. No separate line loop. */
            double chit = cmf_interp_nu_asc(cs->nu, &cs->chi_tot[(size_t)s*NB], NB, nucmf);
            double ces  = cmf_interp_nu_asc(cs->nu, &cs->chi_es[(size_t)s*NB], NB, nucmf);
            double Sfx  = cmf_interp_nu_asc(cs->nu, &cs->S_fixed[(size_t)s*NB], NB, nucmf);
            double Jvf  = cmf_interp_nu_asc(cs->nu, &cs->J[(size_t)s*NB], NB, nucmf);
            if (chit < 0.0) chit = 0.0; if (ces < 0.0) ces = 0.0;
            double rf   = (chit > 0.0) ? ces/chit : 0.0;
            double Sbin = Sfx + rf * Jvf;
            double dt   = q * chit * ds; if (dt < 0.0) dt = 0.0;
            double exf  = (dt > 700.0) ? 0.0 : exp(-dt);
            double psf  = (dt > 1e-4) ? (1.0 - exf) : (dt - 0.5*dt*dt);
            I = I*exf + (D*D*D) * Sbin * psf;
            continue;
        }
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
        if (g_sob_contonly) continue;   /* continuum-only diagnostic (no line jumps) */
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
            if (!(tauS > g_sob_taumin)) continue;    /* skips NaN too (codex #6) */
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
            double Sl;
            if (g_sob_sei && jc) {
                /* 2-pass SEI: scatter the Pass-1 BEAMED continuum mean intensity
                 * jc[s] (carries the D⁴ beaming that comoving J̄_l lacked). Then the
                 * true-tau_S black line conserves: blue trough (deep absorption of
                 * beamed I) balanced by red emission D³·jc[s]. eps blends thermal B. */
                double jb = jc[s]; if (!(jb > 0.0) || !isfinite(jb)) jb = B;
                Sl = (1.0 - g_sob_eps) * jb + g_sob_eps * B;
            } else if (g_sob_sl_ptr) {
                /* UNIFIED: NLTE-solved line source (carries fluorescence once the
                 * UV pump populates the upper level above Boltzmann). Thermal
                 * fallback B(T_e) if unset/<=0; optional clamp vs garbage. */
                Sl = g_sob_sl_ptr[(size_t)l * NS + s];
                if (!(Sl > 0.0) || !isfinite(Sl)) Sl = B;
                if (g_sob_sl_clamp > 0.0 && B > 0.0 && Sl > g_sob_sl_clamp * B)
                    Sl = g_sob_sl_clamp * B;
            } else {
                double Jbar;
                if (g_sob_jbardet) {
                    /* ENERGY-CONSERVING: S_l = local line mean intensity. Sobolev
                     * scatter then conserves (absorbed=re-emitted, only redistributed).
                     * Strong lines: producer J̄_l (jbar_line_det); weak/out-of-window
                     * lines (sentinel<0): the local fine-field J (Jv) — BOTH are local
                     * mean intensities. (W*B external backlight was the −47% energy
                     * leak: it sits below the incident I.) */
                    double jl = (opac->jbar_line_det)
                              ? opac->jbar_line_det[(size_t)l * NS + s] : -1.0;
                    Jbar = (jl >= 0.0 && isfinite(jl)) ? jl : Jv;
                } else if (g_sob_srcj) {
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
                Sl = (1.0 - g_sob_eps) * Jbar + g_sob_eps * B;
            }
            if (g_sob_diag && tauS > 1e4) {
                static int nd = 0;
                if (nd < 8) { nd++;
                    printf("[SOB-SRC] s=%d nu_l=%.4e tauS=%.2e Sl=%.4e B=%.4e "
                           "Sl/B=%.4f src=%s\n", s, opac->line_list_nu[l],
                           tauS, Sl, B, (B>0?Sl/B:0.0),
                           g_sob_sl_ptr ? "NLTE" : "scatter"); }
            }
            double exl = (tauS > 700.0) ? 0.0 : exp(-tauS);
            I = I * exl + (g_sob_noemit ? 0.0 : (D * D * D) * Sl) * (1.0 - exl);
        }
    }
    return I;
}

/* Back-compat wrapper: no SEI beamed-continuum column (jc=NULL). */
static inline double cmf_obs_march_sob(double I, double p, double z0, double z1, int s,
                                const CMFGENState *cs, const OpacityState *opac,
                                double Te_s, double nu_obs, double inv_ct)
{ return cmf_obs_march_sob_jc(NULL, I, p, z0, z1, s, cs, opac, Te_s, nu_obs, inv_ct); }

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
    { const char *e = getenv("LUMINA_CMF_OBS_CONTONLY"); g_sob_contonly = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_TAUMIN"); g_sob_taumin = e ? atof(e) : 1e-6; }
    g_pray_bin = -1;
    { const char *e = getenv("LUMINA_CMF_OBS_PRAY_LAM");
      if (e) { double tl = atof(e), best = 1e99;
        for (int b = 0; b < NB; ++b) { double la = CM_C / cs->nu[b] * 1.0e8;
          if (fabs(la - tl) < best) { best = fabs(la - tl); g_pray_bin = b; } } } }
    g_sob_rphot  = geo->r_inner[0];
    g_sob_Tinner = T_inner;
    if (g_sob_diag && opac->tau_sobolev) {
        for (int l = 0; l < opac->n_lines; ++l) {
            double la = CM_C / opac->line_list_nu[l] * 1.0e8;
            if (la > 7800.0 && la < 9300.0 &&
                opac->tau_sobolev[(size_t)l * NS] > g_sob_taumin)
                printf("[THICKLINE] rest=%.3fA tau=%.4e\n", la,
                       opac->tau_sobolev[(size_t)l * NS]);
        }
    }

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
                I = cmf_inner_bb_scale() * (D_c * D_c * D_c) *
                    cm_planck(q_c * nu_obs, T_inner);
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
            if (g_pray_bin == k)
                printf("[PRAY] lam=%.0f ray=%d p=%.4e core=%d I=%.5e\n",
                       CM_C / cs->nu[k] * 1.0e8, ray, p,
                       (p < geo->r_inner[0]) ? 1 : 0, I);
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
    /* ADV_SPLIT (LUMINA_CMF_ADV_SPLIT=1): flux-limited operator split for the
     * comoving blue->red advection (physics review 2026-06-26). The HB upwind
     * form lumps beta into the spatial dtau=(chi+beta)*ds; at this grid the
     * frequency-advection Courant number beta*ds~80 (a shell spans ~80 fine
     * bins of redshift) so e^{-80} annihilates the radially-transported
     * upstream intensity -> the optical e-scatter continuum collapses to
     * J/B~1e-4 at outer shells. Fix: radial short-char with the REAL opacity
     * (dtau=chi*ds, retains radial memory) + a SEPARATE capped advection pass
     * I += min(beta*ds,1)*(I_bluer - I). Smooth continuum (I_bluer~=I) -> ~0
     * correction -> radial W*B restored; lines still redistribute (capped).
     * Off (default) = legacy lumped HB form (byte-identical). */
    int adv_split = 0;
    { const char *e = getenv("LUMINA_CMF_ADV_SPLIT"); if (e) adv_split = atoi(e); }
    /* GPU formal solver (lumina_cmf_solve.cu). 0=CPU/OMP (byte-identical legacy),
     * 1=GPU (lagged-advection NB*NP kernel), 2=run BOTH and report max rel diff
     * in the converged J (self-check). The producer's fine grid (NB~500k) is the
     * driver of this path; the GPU lags the blue->red advection across ALI iters
     * to expose all (bin,ray) as independent threads -- see lumina_cmf_solve.cu. */
    int use_gpu = 0;
    { const char *e = getenv("LUMINA_CMF_SOLVE_GPU"); if (e) use_gpu = atoi(e); }

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

    /* --- GPU aux tables (built only when LUMINA_CMF_SOLVE_GPU>=1) --------------
     * Bin/adv/advcoef are the per-bin scalars the kernel needs; the per-shell
     * mu-sorted sample tables let the J-integration kernel gather each shell's ray
     * crossings without atomics (geometry fixes which rays cross a shell + their
     * mu = rz/rmid). */
    double *Bin_h=NULL,*adv_h=NULL,*advc_h=NULL,*Jin=NULL,*Jcpu=NULL;
    int *shell_off=NULL,*shell_k=NULL,*shell_seg=NULL; double *shell_mu=NULL;
    int nsamp=0, cpu_iters=n_ali_iter, gpu_iters=0;
    if (use_gpu >= 1) {
        Bin_h=malloc(NB*sizeof(double)); adv_h=malloc(NB*sizeof(double)); advc_h=malloc(NB*sizeof(double));
        for (int b=0;b<NB;++b){ Bin_h[b]=cm_planck(cs->nu[b],T_inner);
            if (b<NB-1){ double Dlam=lam[b]-lam[b+1]; adv_h[b]=a_lam*lam[b]/Dlam; advc_h[b]=a_lam*lam[b+1]/Dlam; }
            else { adv_h[b]=0.0; advc_h[b]=0.0; } }
        int *sc=calloc(NS,sizeof(int));
        for (int k=0;k<NP;++k){ size_t kb=(size_t)k*(NS+1); for (int i=0;i<rn[k];++i) sc[rsh[kb+i]]++; }
        shell_off=malloc((NS+1)*sizeof(int)); shell_off[0]=0;
        for (int s=0;s<NS;++s) shell_off[s+1]=shell_off[s]+sc[s];
        nsamp=shell_off[NS];
        shell_k=malloc((size_t)nsamp*sizeof(int)); shell_seg=malloc((size_t)nsamp*sizeof(int));
        shell_mu=malloc((size_t)nsamp*sizeof(double));
        int *fl=calloc(NS,sizeof(int));
        for (int k=0;k<NP;++k){ size_t kb=(size_t)k*(NS+1);
            for (int i=0;i<rn[k];++i){ int s=rsh[kb+i]; int pos=shell_off[s]+fl[s]++;
                shell_k[pos]=k; shell_seg[pos]=i; shell_mu[pos]=rz[kb+i]/rmid[s]; } }
        for (int s=0;s<NS;++s){ int o0=shell_off[s],o1=shell_off[s+1];   /* sort each shell ascending mu */
            for (int a=o0+1;a<o1;++a){ double mk=shell_mu[a]; int kk=shell_k[a],ss=shell_seg[a]; int q=a-1;
                while(q>=o0&&shell_mu[q]>mk){ shell_mu[q+1]=shell_mu[q]; shell_k[q+1]=shell_k[q]; shell_seg[q+1]=shell_seg[q]; --q; }
                shell_mu[q+1]=mk; shell_k[q+1]=kk; shell_seg[q+1]=ss; } }
        free(sc); free(fl);
        if (use_gpu==2){ Jin=malloc((size_t)NS*NB*sizeof(double)); memcpy(Jin,cs->J,(size_t)NS*NB*sizeof(double)); }
    }

    /* The lagged-advection GPU scheme is BYTE-EXACT to the CPU and converges in
     * the SAME ALI count ONLY when the blue->red frequency coupling is off
     * (a_lam==0, i.e. LUMINA_CMF_ALAM=0 -- the static/transport-only limit, which
     * is also what the fine emergent step uses). With a_lam>0 the coupling is
     * Courant-dominant (adv*ds~10^2 at the producer's fine resolution -> I_b~=I_b+1)
     * and lagging advances the field only ~1 bin/ALI-iter, so it reaches the same
     * fixed point but needs O(NB) iters (verified: NB=400 -> 409 iters; impractical
     * at NB=500k vs the ~24 budget). Warn loudly so GPU=1 is not used blindly with
     * advection on. */
    if (use_gpu >= 1 && a_lam != 0.0)
        fprintf(stderr, "[cmf_gpu] WARNING: LUMINA_CMF_ALAM!=0 (advection on): the lagged "
            "GPU solver needs O(NB) ALI iters to converge -> the field at n_ali=%d is "
            "likely NOT converged. Use LUMINA_CMF_ALAM=0 (static limit) for the GPU path.\n",
            n_ali_iter);

    /* use_gpu==1: attempt the GPU solve first; on failure fall through to CPU. */
    int run_cpu = (use_gpu != 1);
    if (use_gpu == 1) {
        int rc = cmf_solve_J_gpu(NS, NB, NP, adv_split, a_lam,
            cs->chi_tot, cs->chi_es, cs->chi_abs, cs->S_fixed, cs->J,
            Bin_h, adv_h, advc_h, rn, rsh, rz, rcore, rzin,
            shell_off, shell_k, shell_seg, shell_mu, nsamp, n_ali_iter, 1e-4, &gpu_iters);
        if (rc != 0) { fprintf(stderr, "[cmf_gpu] GPU solve failed (rc=%d) -> CPU fallback\n", rc); run_cpu = 1; }
    }

    if (run_cpu)
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
            /* The rays (k) are INDEPENDENT within a frequency (the blue->red
             * advection couples only across b, which is the outer sequential
             * loop). Parallelize over rays. The only shared write target is the
             * per-shell accumulator cnt[s]/muL/ImL/IpL: each (ray,shell) crossing
             * reserves a unique slot via an atomic capture on cnt[s] (in the
             * inbound pass), stored in the per-ray local cloc[] and reused in the
             * outbound pass. Slot order across rays becomes arbitrary, but the
             * per-shell J integration below sorts by mu, so J is order-invariant.
             * Iin_c/Iout_c writes are per-ray (kb range) -> no race. */
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1)
            #endif
            for (int k = 0; k < NP; ++k) {
                int n = rn[k]; if (n == 0) continue; size_t kb = (size_t)k*(NS+1);
                int cloc[300];   /* reserved slot per segment (NS+1 <= 300, cf. mu[300]) */
                double I = 0.0;
                for (int i = 0; i < n; ++i) {
                    int s = rsh[kb+i]; double ds = (i+1<n)?(rz[kb+i]-rz[kb+i+1]):(rz[kb+i]-rzin[k]);
                    size_t idx = (size_t)s*NB+b; double chi = cs->chi_tot[idx], chih = chi + a_lam*4.0 + adv;
                    if (adv_split) {
                        /* radial transport with REAL opacity (+tiny sphericity), then
                         * a capped frequency-advection pass (operator split). */
                        double chi_rad = chi + a_lam*4.0;
                        double Su = S[idx];
                        double dtau=chi_rad*ds, ex=exp(-dtau), e0,e1,wu,wd;
                        if(dtau>1e-4){e0=1-ex;e1=dtau-e0;wu=e0-e1/dtau;wd=e1/dtau;}else{wu=0.5*dtau;wd=0.5*dtau;}
                        double I_rad = I*ex + (wu+wd)*Su;
                        /* implicit upwind frequency advection (stable at any Courant
                         * beta*ds): smooth continuum (Iblue~=I_rad) -> I_rad (W*B
                         * retained); line (Iblue low) -> propagates absorption. */
                        double bds = adv*ds;
                        double Iblue = (b<NB-1)?Iin_p[kb+i]:I_rad;
                        I = (I_rad + bds*Iblue)/(1.0 + bds); Iin_c[kb+i]=I;
                    } else {
                    double Su = (S[idx]*chi + ((b<NB-1)?a_lam*(lam_b/Dlam)*Iin_p[kb+i]:0.0))/(chih>0?chih:1.0);
                    double dtau=chih*ds, ex=exp(-dtau), e0,e1,wu,wd;
                    if(dtau>1e-4){e0=1-ex;e1=dtau-e0;wu=e0-e1/dtau;wd=e1/dtau;}else{wu=0.5*dtau;wd=0.5*dtau;}
                    I = I*ex + wu*Su + wd*Su; Iin_c[kb+i]=I;
                    }
                    int c;
                    #ifdef _OPENMP
                    #pragma omp atomic capture
                    #endif
                    { c = cnt[s]; cnt[s] += 1; }
                    cloc[i] = c; muL[(size_t)s*NP+c]=rz[kb+i]/rmid[s]; ImL[(size_t)s*NP+c]=I;
                }
                if (rcore[k]) I = Bin;
                for (int i = n-1; i >= 0; --i) {
                    int s = rsh[kb+i]; double ds = (i+1<n)?(rz[kb+i]-rz[kb+i+1]):(rz[kb+i]-rzin[k]);
                    size_t idx = (size_t)s*NB+b; double chi = cs->chi_tot[idx], chih = chi + a_lam*4.0 + adv;
                    if (adv_split) {
                        double chi_rad = chi + a_lam*4.0;
                        double Su = S[idx];
                        double dtau=chi_rad*ds, ex=exp(-dtau), e0,e1,wu,wd;
                        if(dtau>1e-4){e0=1-ex;e1=dtau-e0;wu=e0-e1/dtau;wd=e1/dtau;}else{wu=0.5*dtau;wd=0.5*dtau;}
                        double I_rad = I*ex + (wu+wd)*Su;
                        double bds = adv*ds;
                        double Iblue = (b<NB-1)?Iout_p[kb+i]:I_rad;
                        I = (I_rad + bds*Iblue)/(1.0 + bds); Iout_c[kb+i]=I;
                    } else {
                    double Su = (S[idx]*chi + ((b<NB-1)?a_lam*(lam_b/Dlam)*Iout_p[kb+i]:0.0))/(chih>0?chih:1.0);
                    double dtau=chih*ds, ex=exp(-dtau), e0,e1,wu,wd;
                    if(dtau>1e-4){e0=1-ex;e1=dtau-e0;wu=e0-e1/dtau;wd=e1/dtau;}else{wu=0.5*dtau;wd=0.5*dtau;}
                    I = I*ex + wu*Su + wd*Su; Iout_c[kb+i]=I;
                    }
                    IpL[(size_t)s*NP+cloc[i]]=I;
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
        if (maxrel < 1e-4 && it > 0) { cpu_iters = it + 1; break; }
    }

    /* --- self-check (use_gpu==2): re-solve on the GPU from the SAME input and
     * report the max relative difference in the converged J + ALI iter counts.
     * Keeps the CPU field as authoritative so a stray =2 never perturbs a run. */
    if (use_gpu == 2) {
        Jcpu = malloc((size_t)NS*NB*sizeof(double));
        memcpy(Jcpu, cs->J, (size_t)NS*NB*sizeof(double));   /* converged CPU field */
        memcpy(cs->J, Jin,  (size_t)NS*NB*sizeof(double));   /* restore solver input */
        int rc = cmf_solve_J_gpu(NS, NB, NP, adv_split, a_lam,
            cs->chi_tot, cs->chi_es, cs->chi_abs, cs->S_fixed, cs->J,
            Bin_h, adv_h, advc_h, rn, rsh, rz, rcore, rzin,
            shell_off, shell_k, shell_seg, shell_mu, nsamp, n_ali_iter, 1e-4, &gpu_iters);
        if (rc == 0) {
            double maxrel=0.0, l2n=0.0, l2d=0.0; size_t worst=0;
            for (size_t i=0;i<(size_t)NS*NB;++i){ double dn=fabs(cs->J[i]-Jcpu[i]);
                double d=dn/(fabs(Jcpu[i])+1e-30); if(d>maxrel){maxrel=d;worst=i;}
                l2n+=dn*dn; l2d+=Jcpu[i]*Jcpu[i]; }
            fprintf(stderr,
                "[cmf_gpu] SELF-CHECK NS=%d NB=%d NP=%d adv_split=%d: max rel diff(J_gpu vs J_cpu)=%.3e "
                "L2 rel=%.3e  (worst s=%d b=%d: cpu=%.4e gpu=%.4e)  ALI iters cpu=%d gpu=%d\n",
                NS, NB, NP, adv_split, maxrel, (l2d>0)?sqrt(l2n/l2d):0.0,
                (int)(worst/NB),(int)(worst%NB), Jcpu[worst], cs->J[worst], cpu_iters, gpu_iters);
        } else {
            fprintf(stderr, "[cmf_gpu] SELF-CHECK GPU solve failed (rc=%d)\n", rc);
        }
        memcpy(cs->J, Jcpu, (size_t)NS*NB*sizeof(double));   /* keep CPU as authoritative */
    }
    (void)cpu_iters;
    free(Bin_h);free(adv_h);free(advc_h);free(shell_off);free(shell_k);free(shell_seg);free(shell_mu);
    free(Jin);free(Jcpu);
    free(rmid);free(lam);free(p);free(rn);free(rsh);free(rz);free(rcore);free(rzin);
    free(Iin_p);free(Iout_p);free(Iin_c);free(Iout_c);free(muL);free(IpL);free(ImL);free(cnt);free(S);free(Jnew);
}

/* ===================================================================
 * STAGE-1 PROOF (LUMINA_CMF_FINE_EMERGENT=1): frequency-resolved emergent
 * spectrum on the fine mesh. Clones the static (no-Doppler) per-frequency
 * formal integral of cmfgen_write_spectrum but reads the FINE field
 * (fs nu, chi, S_fixed, J) so line windows stay OPEN and the warm photosphere
 * color is transported outward (binned-J collapses to grey-thermal; fine field
 * carries J/B>1 at cold shells -- run 169643). Reuses the binned ray grid
 * (csb->p_ray/n_rays); geometry is frequency-independent. Valid only over the
 * producer window; writes lumina_spectrum_freqres.csv. Color (SED peak) is the
 * gate vs gold 6630A -- the no-Doppler approximation drops P-Cygni profiles but
 * preserves the source-function color, which is what we test. */
static int cmfgen_fine_emergent(const CMFGENState *fs, const CMFGENState *csb,
                                const Geometry *geo, double T_inner,
                                const char *path)
{
    int NS = fs->n_shells, NF = fs->n_bins, NR = csb->n_rays;
    if (NR <= 0 || !csb->p_ray) {
        fprintf(stderr, "[cmf_fine] emergent: no binned ray grid\n"); return -1;
    }
    int    *ray_n    = malloc(sizeof(int) * NR);
    int    *ray_core = malloc(sizeof(int) * NR);
    int    *seg_sh   = malloc(sizeof(int) * (size_t)NR * NS);
    double *seg_ds   = malloc(sizeof(double) * (size_t)NR * NS);
    double *S        = malloc(sizeof(double) * NS);
    double *Lnu      = calloc(NF, sizeof(double));
    if (!ray_n||!ray_core||!seg_sh||!seg_ds||!S||!Lnu) {
        free(ray_n);free(ray_core);free(seg_sh);free(seg_ds);free(S);free(Lnu);
        fprintf(stderr,"[cmf_fine] emergent alloc failed\n"); return -1;
    }
    /* per-ray intersected shells + segment lengths (same as write_spectrum) */
    for (int ray = 0; ray < NR; ++ray) {
        double p = csb->p_ray[ray];
        int *sh = &seg_sh[(size_t)ray * NS];
        double *zz = malloc(sizeof(double) * (NS + 1));
        int nshell = 0;
        for (int s = NS - 1; s >= 0; --s) {
            double ro = geo->r_outer[s];
            if (ro <= p) break;
            double rmid = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
            if (rmid <= p) rmid = p * 1.0000001;
            sh[nshell] = s;
            zz[nshell] = sqrt(rmid*rmid - p*p);
            ++nshell;
        }
        ray_n[ray]    = nshell;
        ray_core[ray] = (p < geo->r_inner[0]) ? 1 : 0;
        double z_core = 0.0;
        if (ray_core[ray] && nshell > 0) {
            double ri0 = geo->r_inner[0];
            z_core = sqrt(ri0*ri0 - p*p);
            if (z_core > zz[nshell-1]) z_core = zz[nshell-1];
        }
        double *ds = &seg_ds[(size_t)ray * NS];
        for (int i = 0; i < nshell; ++i)
            ds[i] = (i+1 < nshell) ? fabs(zz[i]-zz[i+1])
                                   : (ray_core[ray] ? zz[i]-z_core : fabs(zz[i]));
        free(zz);
    }
    /* per-fine-frequency emergent flux (static formal integral) */
    for (int b = 0; b < NF; ++b) {
        double Bin = cm_planck(fs->nu[b], T_inner);
        for (int s = 0; s < NS; ++s) {
            size_t idx = (size_t)s * NF + b;
            double r = (fs->chi_tot[idx] > 0.0) ? fs->chi_es[idx]/fs->chi_tot[idx] : 0.0;
            S[s] = fs->S_fixed[idx] + r * fs->J[idx];   /* converged fine source */
        }
        double integ = 0.0, p_prev = 0.0, f_prev = 0.0;
        for (int ray = 0; ray < NR; ++ray) {
            int n = ray_n[ray];
            if (n == 0) continue;
            const int *sh = &seg_sh[(size_t)ray * NS];
            const double *ds = &seg_ds[(size_t)ray * NS];
            double I = 0.0;
            for (int i = 0; i < n; ++i) {            /* inbound */
                double dtau = fs->chi_tot[(size_t)sh[i]*NF + b] * ds[i];
                if (dtau < 0.0) dtau = 0.0;
                double ex = exp(-dtau);
                double psi = (dtau > 1e-4) ? (1.0-ex) : (dtau - 0.5*dtau*dtau);
                I = I*ex + S[sh[i]]*psi;
            }
            if (ray_core[ray]) I = Bin;              /* diffusive core */
            for (int i = n-1; i >= 0; --i) {         /* outbound */
                double dtau = fs->chi_tot[(size_t)sh[i]*NF + b] * ds[i];
                if (dtau < 0.0) dtau = 0.0;
                double ex = exp(-dtau);
                double psi = (dtau > 1e-4) ? (1.0-ex) : (dtau - 0.5*dtau*dtau);
                I = I*ex + S[sh[i]]*psi;
            }
            double p = csb->p_ray[ray];
            double f = I * p;
            integ += 0.5 * (f_prev + f) * (p - p_prev);
            p_prev = p; f_prev = f;
        }
        Lnu[b] = 8.0 * M_PI * M_PI * integ;
    }
    FILE *fp = fopen(path, "w");
    if (!fp) { free(ray_n);free(ray_core);free(seg_sh);free(seg_ds);free(S);free(Lnu);
        fprintf(stderr,"[cmf_fine] cannot open %s\n",path); return -1; }
    fprintf(fp, "wavelength_angstrom,flux\n");
    for (int b = NF-1; b >= 0; --b) {                /* nu desc -> lambda asc */
        double lam_cm = CM_C / fs->nu[b];
        double lam_A  = lam_cm * 1.0e8;
        double L_lam  = Lnu[b] * CM_C / (lam_cm*lam_cm) * 1.0e-8;
        fprintf(fp, "%.6f,%.6e\n", lam_A, L_lam);
    }
    fclose(fp);
    fprintf(stderr, "[cmf_fine] freq-resolved emergent -> %s (%d fine freqs)\n", path, NF);
    free(ray_n);free(ray_core);free(seg_sh);free(seg_ds);free(S);free(Lnu);
    return 0;
}

/* ===================================================================
 * UNIFIED emergent (LUMINA_CMF_FINE_EMERGENT_OBS=1): the deterministic
 * production spectrum that combines all three physics on one integrator:
 *   - CONTINUUM color    : the FINE field fs (chi_es/chi_abs/J), interpolated at
 *                          the local comoving frequency (cures binned-J grey).
 *   - P-Cygni profiles   : observer-frame Doppler march (q = gamma(1-mu beta))
 *                          with full-tau_Sobolev line resonances (cmf_obs_march_sob).
 *   - FLUORESCENCE source: line resonances emit the NLTE-solved line_source_S
 *                          (g_sob_sl_ptr), not the W*B scattering backlight.
 * Output is a moderate observer grid (LUMINA_CMF_FINE_OBS_NOBS, default 3000)
 * over the fine window; continuum is interpolated on the fine grid so the grid
 * need not be fine. Reuses cmf_obs_march_sob (fs passed as the field state).
 * beta->0 / no lines reduces to the static cmfgen_fine_emergent color. */
static int cmfgen_fine_emergent_obs(const CMFGENState *fs, const Geometry *geo,
                                    double T_inner, const OpacityState *opac,
                                    const double *Te, const char *path)
{
    int NS = fs->n_shells, NF = fs->n_bins;
    double inv_ct = 1.0 / (CM_C * geo->time_explosion);
    if (!opac || !opac->line_list_nu || !opac->tau_sobolev || !Te) {
        fprintf(stderr, "[cmf_obs] missing line data for unified emergent\n");
        return -1;
    }
    /* line source: NLTE line_source_S (fluorescence) by default; or the W*B
     * diluted-photosphere SCATTERING source (LUMINA_CMF_FINE_OBS_SCATTER=1),
     * which is what produces classical P-Cygni (resonance scatter of the hot
     * backlight). Thermal NLTE S_l=B gives NO P-Cygni (absorption=emission
     * cancel), so the scatter source is the right vehicle until fluorescence
     * lifts S_l above Boltzmann. */
    int obs_scatter = 0;
    { const char *e = getenv("LUMINA_CMF_FINE_OBS_SCATTER"); if (e) obs_scatter = atoi(e); }
    g_sob_sl_ptr   = obs_scatter ? NULL : opac->line_source_S;
    g_sob_sl_clamp = 0.0;
    { const char *e = getenv("LUMINA_CMF_FINE_SL_CLAMP"); if (e) g_sob_sl_clamp = atof(e); }
    { const char *e = getenv("LUMINA_CMF_OBS_TAUMIN");    g_sob_taumin = e ? atof(e) : 1e-6; }
    g_sob_eps = 0.0; g_sob_noemit = 0; g_sob_srcj = 0; g_sob_contonly = 0;
    /* F1 fine-tune: line scatter/absorption blend S_l=(1-eps)*W*B + eps*B(Te).
     * eps=0 = pure scatter (fills troughs, right color); eps>0 adds the cold
     * local-thermal absorbing fraction that DEEPENS the P-Cygni troughs
     * (Ca II H&K / Fe II / O I) toward gold without the full-thermal reddening. */
    { const char *e = getenv("LUMINA_CMF_LINE_SOB_EPS");
      g_sob_eps = e ? atof(e) : 0.0;
      if (g_sob_eps < 0.0) g_sob_eps = 0.0; if (g_sob_eps > 1.0) g_sob_eps = 1.0; }
    { const char *e = getenv("LUMINA_CMF_OBS_CONTONLY"); g_sob_contonly = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_JBARDET"); g_sob_jbardet = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_FAITHFUL"); g_sob_faithful = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_SEI"); g_sob_sei = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_SEI_CMV"); g_sob_sei_cmv = e ? atoi(e) : 1; }
    { const char *e = getenv("LUMINA_CMF_OBS_SRCJ"); g_sob_srcj = e ? atoi(e) : 0; }
    g_sob_rphot = geo->r_inner[0]; g_sob_Tinner = T_inner;

    int NObs = 3000; { const char *e = getenv("LUMINA_CMF_FINE_OBS_NOBS"); if (e) NObs = atoi(e); }
    if (NObs < 2) NObs = 2;
    int NRO = 256; { const char *e = getenv("LUMINA_CMF_OBS_NRAY"); if (e) NRO = atoi(e); }
    double r_max = geo->r_outer[NS - 1], r_in0 = geo->r_inner[0];
    double *p_obs = malloc(sizeof(double) * NRO);
    double *nuo   = malloc(sizeof(double) * NObs);
    double *Lnu   = calloc(NObs, sizeof(double));
    if (!p_obs || !nuo || !Lnu) { free(p_obs); free(nuo); free(Lnu);
        fprintf(stderr, "[cmf_obs] alloc failed\n"); return -1; }
    for (int kk = 0; kk < NRO; ++kk) p_obs[kk] = r_max * (kk + 0.5) / (double)NRO;
    /* observer output grid: log-uniform over the fine window [nu_lo,nu_hi] */
    double nu_lo = fs->nu[0], nu_hi = fs->nu[NF - 1];
    double dln = log(nu_hi / nu_lo) / (double)(NObs - 1);
    for (int k = 0; k < NObs; ++k) nuo[k] = nu_lo * exp(dln * k);

    /* SEI Pass 1: continuum-only march -> beamed continuum mean Jbar_C[k,s]
     * (ray-ensemble angle average, carries the D⁴ beaming). Pass 2 (main loop)
     * sources the true-tau_S line jumps from it -> conserving + sharp P-Cygni. */
    double *Jbar_C = NULL;
    if (g_sob_sei) {
        Jbar_C = calloc((size_t)NObs * NS, sizeof(double));
        int *cnt = calloc((size_t)NObs * NS, sizeof(int));
        int save_co = g_sob_contonly; g_sob_contonly = 1;
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < NObs; ++k) {
            double nu_obs = nuo[k];
            for (int ray = 0; ray < NRO; ++ray) {
                double p = p_obs[ray]; int sh[256], nshell = 0;
                for (int s = NS-1; s >= 0 && nshell < 256; --s) { if (geo->r_outer[s] <= p) break; sh[nshell++] = s; }
                if (nshell == 0) continue;
                int core = (p < r_in0) ? 1 : 0; double I = 0.0;
                /* CMV (default): accumulate the COMOVING intensity I_cmf = I_obs*q^3
                 * so jc[s] is the comoving mean J_bar; Pass-2 then beams it ONCE via
                 * D^3 (the old code stored observer-frame I -> D^3 double-beamed it,
                 * the +180% SEI bug). LUMINA_CMF_OBS_SEI_CMV=0 reverts. */
                for (int i = 0; i < nshell; ++i) {
                    double ro=geo->r_outer[sh[i]],ri=geo->r_inner[sh[i]];
                    double zh=(ro>p)?sqrt(ro*ro-p*p):0.0, zl=(ri>p)?sqrt(ri*ri-p*p):0.0;
                    I = cmf_obs_march_sob(I,p,-zh,-zl,sh[i],fs,opac,Te[sh[i]],nu_obs,inv_ct);
                    double w=1.0; if (g_sob_sei_cmv) { double zm=-0.5*(zh+zl),rm=sqrt(p*p+zm*zm),
                        mu=(rm>0)?zm/rm:0.0,be=rm*inv_ct,ga=1.0/sqrt(1.0-be*be),q=ga*(1.0-mu*be); w=q*q*q; }
                    Jbar_C[(size_t)k*NS+sh[i]] += I*w; cnt[(size_t)k*NS+sh[i]]++;
                }
                if (core) { double zc=sqrt(r_in0*r_in0-p*p),mc=zc/r_in0,bi=r_in0*inv_ct,gi=1.0/sqrt(1.0-bi*bi);
                            double qc=gi*(1.0-mc*bi),Dc=1.0/qc; I=(Dc*Dc*Dc)*cm_planck(qc*nu_obs,T_inner); }
                for (int i = nshell-1; i >= 0; --i) {
                    double ro=geo->r_outer[sh[i]],ri=geo->r_inner[sh[i]];
                    double zh=(ro>p)?sqrt(ro*ro-p*p):0.0, zl=(ri>p)?sqrt(ri*ri-p*p):0.0;
                    I = cmf_obs_march_sob(I,p,+zl,+zh,sh[i],fs,opac,Te[sh[i]],nu_obs,inv_ct);
                    double w=1.0; if (g_sob_sei_cmv) { double zm=0.5*(zh+zl),rm=sqrt(p*p+zm*zm),
                        mu=(rm>0)?zm/rm:0.0,be=rm*inv_ct,ga=1.0/sqrt(1.0-be*be),q=ga*(1.0-mu*be); w=q*q*q; }
                    Jbar_C[(size_t)k*NS+sh[i]] += I*w; cnt[(size_t)k*NS+sh[i]]++;
                }
            }
        }
        g_sob_contonly = save_co;
        for (size_t i = 0; i < (size_t)NObs*NS; ++i) if (cnt[i] > 0) Jbar_C[i] /= cnt[i];
        free(cnt);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < NObs; ++k) {
        double nu_obs = nuo[k];
        const double *jc = (g_sob_sei && Jbar_C) ? &Jbar_C[(size_t)k * NS] : NULL;
        double integ = 0.0, p_prev = 0.0, f_prev = 0.0;
        for (int ray = 0; ray < NRO; ++ray) {
            double p = p_obs[ray];
            int sh[256], nshell = 0;
            for (int s = NS - 1; s >= 0 && nshell < 256; --s) {
                if (geo->r_outer[s] <= p) break;
                sh[nshell++] = s;
            }
            if (nshell == 0) continue;
            int core = (p < r_in0) ? 1 : 0;
            double I = 0.0;
            /* inbound (far side z<0): outer -> inner */
            for (int i = 0; i < nshell; ++i) {
                double ro = geo->r_outer[sh[i]], ri = geo->r_inner[sh[i]];
                double z_hi = (ro > p) ? sqrt(ro*ro - p*p) : 0.0;
                double z_lo = (ri > p) ? sqrt(ri*ri - p*p) : 0.0;
                I = cmf_obs_march_sob_jc(jc, I, p, -z_hi, -z_lo, sh[i], fs, opac,
                                      Te[sh[i]], nu_obs, inv_ct);
            }
            if (core) {                          /* diffusive core */
                double z_core = sqrt(r_in0*r_in0 - p*p);
                double mu_c = z_core / r_in0, beta_in = r_in0 * inv_ct;
                double gam_in = 1.0 / sqrt(1.0 - beta_in*beta_in);
                double q_c = gam_in * (1.0 - mu_c * beta_in), D_c = 1.0 / q_c;
                I = (D_c*D_c*D_c) * cm_planck(q_c * nu_obs, T_inner);
            }
            /* outbound (near side z>0): inner -> outer */
            for (int i = nshell - 1; i >= 0; --i) {
                double ro = geo->r_outer[sh[i]], ri = geo->r_inner[sh[i]];
                double z_hi = (ro > p) ? sqrt(ro*ro - p*p) : 0.0;
                double z_lo = (ri > p) ? sqrt(ri*ri - p*p) : 0.0;
                I = cmf_obs_march_sob_jc(jc, I, p, +z_lo, +z_hi, sh[i], fs, opac,
                                      Te[sh[i]], nu_obs, inv_ct);
            }
            double f = I * p;
            integ += 0.5 * (f_prev + f) * (p - p_prev);
            p_prev = p; f_prev = f;
        }
        Lnu[k] = 8.0 * M_PI * M_PI * integ;
    }
    g_sob_sl_ptr = NULL;   /* reset global */
    free(Jbar_C);

    FILE *fp = fopen(path, "w");
    if (!fp) { free(p_obs); free(nuo); free(Lnu);
        fprintf(stderr, "[cmf_obs] cannot open %s\n", path); return -1; }
    fprintf(fp, "wavelength_angstrom,flux\n");
    for (int k = NObs - 1; k >= 0; --k) {       /* nu desc -> lambda asc */
        double lam_cm = CM_C / nuo[k];
        double L_lam  = Lnu[k] * CM_C / (lam_cm*lam_cm) * 1.0e-8;
        fprintf(fp, "%.6f,%.6e\n", lam_cm * 1.0e8, L_lam);
    }
    fclose(fp);
    fprintf(stderr, "[cmf_obs] unified Doppler emergent -> %s (%d obs freqs, "
            "src=%s)\n", path, NObs, opac->line_source_S ? "NLTE-Sl" : "scatter");
    free(p_obs); free(nuo); free(Lnu);
    return 0;
}

/* ===================================================================
 * P7 PRODUCER: fine-grid line-resolved J_bar_l (gate II producer).
 *
 * Reuses the VALIDATED frequency-coupled kernel cmf_solve_J on a fine
 * uniform-log frequency mesh spanning a wavelength window (default
 * 1000-4000 A, the fluorescence pump region). Each line is deposited as a
 * Gaussian profile with center opacity
 *     chi0 = (1-e^{-tau_S}) / (sqrt(pi) * vdop * t_exp)
 * so that  Int chi_line dnu = (1-e^{-tau_S}) * nu_l/(c t_exp), which ties
 * back exactly to the binned expansion opacity (the I-2 deposition).
 * Continuum (chi_es, chi_abs) is log-nu interpolated from the binned state.
 * Lines carry their lagged source S_l (line_source_S, fallback B(nu_l,Te))
 * in S_fixed; electron scattering is the ALI scattering channel (the outer
 * NLTE loop re-derives S_l from the J_bar this produces -- the lagged-J
 * scheme validated in gate 5d). After the solve, extracts per line
 *     J_bar_l = Int phi_l J dnu / Int phi_l dnu
 * into opac->jbar_line_det (sentinel -1 for lines outside the window, so
 * the plasma bb-rate falls back). Standalone-validated: gates 3a/4b/4c/5*.
 *
 * Caller allocates opac->jbar_line_det[n_lines*n_shells]. The plasma
 * bb-rate reads it under LUMINA_CMF_LINERES_JBAR. */
/* bf + atom registered by cuda.cu so the producer can build the fine bf continuum
 * opacity (LUMINA_CMF_FINE_BF_OPAC). NULL → keep the interpolated continuum. */
static BFOpacity   *g_fine_bf   = NULL;
static AtomicData  *g_fine_atom = NULL;
void cmfgen_fine_set_bf_atom(BFOpacity *bf, AtomicData *atom) {
    g_fine_bf = bf; g_fine_atom = atom;
}

void cmfgen_fine_jbar(CMFGENState *csb, const Geometry *geo,
                      OpacityState *opac, double T_inner, PlasmaState *plasma)
{
    int NS = csb->n_shells, NL = opac->n_lines;
    if (NL <= 0 || !opac->jbar_line_det) return;
    double t_exp = geo->time_explosion;
    int diag = 0; { const char *e=getenv("LUMINA_CMF_FINE_DIAG"); if(e) diag=atoi(e); }

    /* --- window + resolution (env-tunable) --- */
    double lam_lo = 1000.0, lam_hi = 4000.0;   /* Angstrom */
    { const char *e=getenv("LUMINA_CMF_FINE_LAMLO"); if(e) lam_lo=atof(e); }
    { const char *e=getenv("LUMINA_CMF_FINE_LAMHI"); if(e) lam_hi=atof(e); }
    double vdop = 1.0e6;                        /* cm/s, Doppler width  */
    { const char *e=getenv("LUMINA_CMF_FINE_VDOP"); if(e) vdop=atof(e); }
    double ppd = 12.0;                          /* fine points / vdop   */
    { const char *e=getenv("LUMINA_CMF_FINE_PPD"); if(e) ppd=atof(e); }
    int n_ali = 24; { const char *e=getenv("LUMINA_CMF_FINE_ALI"); if(e) n_ali=atoi(e); }
    /* Stage-2-pre stabilization: clamp lagged super-thermal S_l in the line
     * emissivity deposit. The cold-shell NLTE matrix is ill-conditioned ->
     * S_l/B can be huge (numerical artifact, rates are DB-correct); the closed
     * producer<->NLTE loop amplifies it (Jbar/B 454,659 at iter5,7 in 169651).
     * Clamp S_l <= sl_clamp*B(Te) (orthodox LTE@Te-floor analog). 0 = off. */
    double sl_clamp = 0.0; { const char *e=getenv("LUMINA_CMF_FINE_SL_CLAMP"); if(e) sl_clamp=atof(e); }
    /* Line-forest SCATTERING source (LUMINA_CMF_FINE_LINE_EPS = eps in (0,1]):
     * the ROOT CAUSE of the absent fluorescence (2026-06-26, triple-verified) is
     * that the producer emits the in-window forest as THERMAL emitters (eta =
     * chi_line*B(Te)), so the UV pump field thermalises to B(Te) in the green-
     * emitting shells (measured J_bar/B = 1.001) -> no super-thermal UV to pump.
     * With eps<1 the line source becomes S_l=(1-eps)*Jbar + eps*B: the thermal
     * fraction eps*chi_line emits B, the scattering remainder (1-eps)*chi_line
     * joins chi_es so the ALI carries the diluted-but-HOT photospheric UV field
     * outward (W*B(T_phot) > cold B(T_e)) -> super-thermal pump -> fluorescence.
     * eps=1.0 (default) = legacy pure-thermal (byte-identical). FROZEN-PLASMA
     * staged use: converge thermal first, then enable as a perturbation. */
    double line_eps = 1.0; { const char *e=getenv("LUMINA_CMF_FINE_LINE_EPS"); if(e) line_eps=atof(e);
        if (line_eps < 0.0) line_eps = 0.0; if (line_eps > 1.0) line_eps = 1.0; }

    double nu_lo = CM_C / (lam_hi * 1.0e-8);    /* red edge  = low nu   */
    double nu_hi = CM_C / (lam_lo * 1.0e-8);    /* blue edge = high nu  */
    double dlognu = (vdop / CM_C) / ppd;
    int NF = (int)(log(nu_hi / nu_lo) / dlognu) + 1;
    if (NF < 2) return;

    /* default ALL lines to the sentinel (out-of-window fall back) */
    for (size_t i = 0; i < (size_t)NL * NS; ++i) opac->jbar_line_det[i] = -1.0;

    if (diag) fprintf(stderr,
        "[cmf_fine] window %.0f-%.0f A  vdop=%.1f km/s ppd=%.0f  NF=%d (%.1fM cells)\n",
        lam_lo, lam_hi, vdop/1e5, ppd, NF, (double)NF*NS/1e6);

    /* --- fine state: only the fields cmf_solve_J reads, plus chi_line/dnu --- */
    CMFGENState fs; memset(&fs, 0, sizeof fs);
    fs.n_shells = NS; fs.n_bins = NF;
    fs.nu_min = nu_lo; fs.nu_max = nu_hi; fs.d_log_nu = dlognu;
    fs.nu       = malloc((size_t)NF * sizeof(double));
    fs.dnu      = malloc((size_t)NF * sizeof(double));
    fs.chi_es   = calloc((size_t)NS * NF, sizeof(double));
    fs.chi_abs  = calloc((size_t)NS * NF, sizeof(double));
    fs.chi_line = calloc((size_t)NS * NF, sizeof(double));
    fs.chi_tot  = calloc((size_t)NS * NF, sizeof(double));
    fs.S_fixed  = calloc((size_t)NS * NF, sizeof(double));
    fs.J        = calloc((size_t)NS * NF, sizeof(double));
    double *eta = calloc((size_t)NS * NF, sizeof(double));   /* line emissivity */
    if (!fs.nu||!fs.dnu||!fs.chi_es||!fs.chi_abs||!fs.chi_line||!fs.chi_tot||
        !fs.S_fixed||!fs.J||!eta) {
        fprintf(stderr, "[cmf_fine] alloc failed (NF=%d NS=%d)\n", NF, NS);
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);free(fs.chi_line);
        free(fs.chi_tot);free(fs.S_fixed);free(fs.J);free(eta); return;
    }
    for (int i = 0; i < NF; ++i) {
        fs.nu[i]  = nu_lo * exp((i + 0.5) * dlognu);
        fs.dnu[i] = fs.nu[i] * dlognu;
    }

    /* --- continuum: log-nu interpolate chi_es/chi_abs from the binned state --- */
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < NS; ++s) {
        for (int i = 0; i < NF; ++i) {
            double x = log(fs.nu[i] / csb->nu_min) / csb->d_log_nu - 0.5;
            int b = (int)floor(x); double f = x - b;
            if (b < 0) { b = 0; f = 0.0; }
            if (b >= csb->n_bins - 1) { b = csb->n_bins - 2; f = 1.0; }
            size_t ia = (size_t)s*csb->n_bins + b, ib = ia + 1;
            fs.chi_es [(size_t)s*NF+i] = (1-f)*csb->chi_es [ia] + f*csb->chi_es [ib];
            fs.chi_abs[(size_t)s*NF+i] = (1-f)*csb->chi_abs[ia] + f*csb->chi_abs[ib];
        }
    }

    /* --- CMFGEN-method fix (LUMINA_CMF_FINE_BF_OPAC): replace the smeared interpolated
     * bf part of the continuum with the fine-ν bf opacity (sharp edges at the exact
     * thresholds), so the solved field develops the across-edge frequency structure the
     * binned/interpolated continuum averages away. chi_abs_fine = interp_chi_abs
     * − bf_get_chi(binned bf @ fine ν, the smeared part) + chi_bf_fine(sharp). ff kept. */
    {
        static int fbf = -1;
        if (fbf < 0) { const char *e = getenv("LUMINA_CMF_FINE_BF_OPAC"); fbf = (e && atoi(e)) ? 1 : 0; }
        if (fbf && g_fine_bf && g_fine_atom) {
            double *chi_bf_fine = (double *)malloc((size_t)NS * NF * sizeof(double));
            if (chi_bf_fine && bf_gemm_compute_fine(g_fine_bf, g_fine_atom, plasma, NS,
                    fs.nu, NF, g_fine_bf->nu_min, g_fine_bf->d_log_nu, chi_bf_fine) == 0) {
                double sum_old = 0.0, sum_new = 0.0;
                #pragma omp parallel for schedule(static) reduction(+:sum_old,sum_new)
                for (int s = 0; s < NS; ++s)
                    for (int i = 0; i < NF; ++i) {
                        size_t k = (size_t)s*NF + i;
                        double smeared_bf = bf_get_chi(g_fine_bf, s, fs.nu[i]);
                        double newabs = fs.chi_abs[k] - smeared_bf + chi_bf_fine[k];
                        if (newabs < 0.0) newabs = chi_bf_fine[k];   /* ff floor guard */
                        sum_old += fs.chi_abs[k]; sum_new += newabs;
                        fs.chi_abs[k] = newabs;
                    }
                fprintf(stderr, "[FINE-BF-OPAC] sharp-edge bf continuum applied "
                        "(NS=%d NF=%d, Σchi_abs %.3e -> %.3e)\n", NS, NF, sum_old, sum_new);
                /* DIAG: for the outer shell, show how much the bf sharpening changed
                 * chi_abs locally AND chi_abs vs scattering — disambiguates "swamped"
                 * (chi_abs<<chi_es) vs "too weak" (chi_bf_fine~smeared). One-shot. */
                static int bfdiag = -1;
                if (bfdiag < 0) { const char *e=getenv("LUMINA_CMF_FINE_BF_DIAG"); bfdiag=(e&&atoi(e))?1:0; }
                if (bfdiag) {
                    int sd = NS-1;   /* outer shell */
                    fprintf(stderr, "[FINE-BF-DIAG] shell %d (outer): lam_A  chi_es  smeared_bf  fine_bf  fine/smeared\n", sd);
                    for (int j = 1; j <= 12; ++j) {
                        int i = (int)((double)j/13.0 * NF);
                        double lamA = 2.99792458e18 / fs.nu[i];
                        double ce = fs.chi_es[(size_t)sd*NF+i];
                        double sb = bf_get_chi(g_fine_bf, sd, fs.nu[i]);
                        double ff = chi_bf_fine[(size_t)sd*NF+i];
                        fprintf(stderr, "[FINE-BF-DIAG]  %.0f  %.3e  %.3e  %.3e  %.2f\n",
                                lamA, ce, sb, ff, sb>0?ff/sb:0.0);
                    }
                }
            } else {
                fprintf(stderr, "[FINE-BF-OPAC] compute failed -> interpolated continuum\n");
            }
            free(chi_bf_fine);
        }
    }

    /* --- deposit line profiles (chi_line) + emissivity (eta) --- */
    /* CONTONLY falsifier (2026-06-25): skip the line deposit entirely so the
     * emergent is pure continuum (S_fixed = chi_abs*B/(chi_es+chi_abs)).
     * A/B vs full isolates whether the grn/nir gap is continuum color
     * (blanketing) or thermal line re-emission (-> needs fluorescence). */
    int fine_contonly = 0;
    { const char *e = getenv("LUMINA_CMF_FINE_CONTONLY"); if (e && atoi(e)) fine_contonly = 1; }
    const double SQRTPI = 1.7724538509055160;
    double chi0_pref = 1.0 / (SQRTPI * vdop * t_exp);   /* chi0 = (1-e^-tau)*pref */
    int src_nlte = (opac->line_source_S != NULL);
    long n_inwin = 0, n_clamped = 0;
    double max_slb = 0.0;   /* max S_l/B(Te) over deposited lines (diagnostic) */
    /* Strong-line threshold: skip lines whose Sobolev tau is below fine_taumin in
     * every shell. The dense UV pump forest (1000-3000A) is otherwise intractable
     * to deposit on the fine mesh; for the fluorescence pump only the dominant
     * tau_S lines matter (design option-c). Default 1e-12 = deposit all (legacy).
     * Set LUMINA_CMF_FINE_TAUMIN ~0.1-1 for a tractable UV pump producer. */
    double fine_taumin = 1e-12;
    { const char *e = getenv("LUMINA_CMF_FINE_TAUMIN"); if (e) fine_taumin = atof(e); }
    long n_skip_weak = 0;
    /* OMP enabler: the per-line deposit accumulates into chi_line[s,:]/eta[s,:], so
     * parallelising over LINES would race. Instead precompute the per-line in-window
     * + weak-skip flag SERIALLY (cheap, read-only), then parallelise the deposit over
     * SHELLS (each thread owns one shell's chi_line/eta -> no race). Behaviour is
     * identical to the old line-serial loop (same deposit, same diagnostics). */
    char *line_use = NULL;
    if (!fine_contonly) {
        line_use = (char *)malloc((size_t)NL);
        for (int l = 0; l < NL; ++l) {
            double nu_l = opac->line_list_nu[l];
            char use = (nu_l > nu_lo && nu_l < nu_hi) ? 1 : 0;
            if (use) {
                ++n_inwin;
                if (fine_taumin > 1e-12) {        /* skip line if weak in ALL shells */
                    double tmax = 0.0;
                    for (int s = 0; s < NS; ++s) {
                        double t = opac->tau_sobolev[(size_t)l * NS + s];
                        if (t > tmax) tmax = t;
                    }
                    if (tmax < fine_taumin) { ++n_skip_weak; use = 0; }
                }
            }
            line_use[l] = use;
        }
    }
    /* [OWNER] load-balance redesign (user #1, 2026-07-06): the legacy loop is
     * shell-parallel (width 50, inner-shell tail-dominated -> ~18 cores).
     * Owner-computes: fine-grid CHUNKS own their bins; each (s,chunk) task
     * binary-searches the DESCENDING line_list_nu for lines whose +-(4 sigma
     * + guard) deposit window reaches its bins. Every (s,i) bin is owned by
     * exactly one task and contributing lines are processed in the same l
     * order as the legacy loop => bit-identical accumulation.
     * LUMINA_FINE_OWNER=0 restores the legacy loop; non-monotone line list
     * also falls back. */
    int fine_owner = 1;
    { const char *e = getenv("LUMINA_FINE_OWNER"); if (e) fine_owner = atoi(e); }
    if (fine_owner) {
        static int nu_desc_ok = -1;
        if (nu_desc_ok < 0) {
            nu_desc_ok = 1;
            for (int l = 1; l < NL; ++l)
                if (opac->line_list_nu[l] > opac->line_list_nu[l-1]) { nu_desc_ok = 0; break; }
            if (!nu_desc_ok)
                printf("[FINE-OWNER] line_list_nu not descending -> legacy loop\n");
        }
        if (!nu_desc_ok) fine_owner = 0;
    }
    if (fine_owner && !fine_contonly) {
        int nch = 512;
        { const char *e = getenv("LUMINA_FINE_NCHUNK"); if (e) nch = atoi(e); }
        if (nch < 1) nch = 1; if (nch > NF) nch = NF;
        int chunk_sz = (NF + nch - 1) / nch;
        double marg = 8.0 * vdop / CM_C + 4.0 * dlognu;   /* window guard */
        #pragma omp parallel for schedule(dynamic) collapse(2) \
                reduction(max:max_slb) reduction(+:n_clamped)
        for (int s = 0; s < NS; ++s) {
            for (int ch = 0; ch < nch; ++ch) {
                const int ICLO = ch * chunk_sz;
                int ichi_t = ICLO + chunk_sz - 1;
                if (ichi_t >= NF) ichi_t = NF - 1;
                const int ICHI = ichi_t;
                if (ICLO > ICHI) continue;
                double nu_min = fs.nu[ICLO] * (1.0 - marg);
                double nu_max = fs.nu[ICHI] * (1.0 + marg);
                /* descending list: la = first idx with nu <= nu_max,
                 *                  lb = first idx with nu <  nu_min */
                int la, lb;
                { int lo = 0, hi = NL;
                  while (lo < hi) { int mid = lo + (hi - lo) / 2;
                      if (opac->line_list_nu[mid] > nu_max) lo = mid + 1; else hi = mid; }
                  la = lo; lo = la; hi = NL;
                  while (lo < hi) { int mid = lo + (hi - lo) / 2;
                      if (opac->line_list_nu[mid] >= nu_min) lo = mid + 1; else hi = mid; }
                  lb = lo; }
                for (int l = la; l < lb; ++l) {
            if (!line_use[l]) continue;
            double nu_l = opac->line_list_nu[l];
            double tau = opac->tau_sobolev[(size_t)l * NS + s];
            if (tau <= 1e-12) continue;
            double dnuD = nu_l * vdop / CM_C;
            double xc = log(nu_l / nu_lo) / dlognu - 0.5;
            int half  = (int)ceil(4.0 * dnuD / (nu_l * dlognu)) + 1;
            int ic = (int)floor(xc + 0.5);
            int i0 = ic - half, i1 = ic + half;
            if (i0 < ICLO) i0 = ICLO; if (i1 > ICHI) i1 = ICHI;
            if (i0 > i1) continue;
            double frac = (tau > 1e-6) ? -expm1(-tau) : tau;   /* 1-e^{-tau} */
            double chi0 = frac * chi0_pref;
            double Bl = cm_planck(nu_l, plasma->T_e[s]);
            double Sl = src_nlte ? opac->line_source_S[(size_t)l*NS+s] : 0.0;
            if (Sl <= 0.0) Sl = Bl;
            if (Bl > 0.0) {
                double rb = Sl / Bl; if (rb > max_slb) max_slb = rb;
            }
            if (sl_clamp > 0.0 && Bl > 0.0 && Sl > sl_clamp * Bl) {
                Sl = sl_clamp * Bl;
                ++n_clamped;
            }
            double emit = (line_eps < 1.0) ? (line_eps * Bl) : Sl;
            for (int i = i0; i <= i1; ++i) {
                double xv = (fs.nu[i] - nu_l) / dnuD;
                double p  = exp(-xv * xv);
                double cl = chi0 * p;
                fs.chi_line[(size_t)s*NF+i] += cl;
                eta        [(size_t)s*NF+i] += cl * emit;
            }
                }
            }
        }
    } else if (!fine_contonly) {
    #pragma omp parallel for schedule(dynamic) reduction(max:max_slb) reduction(+:n_clamped)
    for (int s = 0; s < NS; ++s) {
        const int ICLO = 0, ICHI = NF - 1;
        for (int l = 0; l < NL; ++l) {
            if (!line_use[l]) continue;
            double nu_l = opac->line_list_nu[l];
            double tau = opac->tau_sobolev[(size_t)l * NS + s];
            if (tau <= 1e-12) continue;
            double dnuD = nu_l * vdop / CM_C;
            double xc = log(nu_l / nu_lo) / dlognu - 0.5;
            int half  = (int)ceil(4.0 * dnuD / (nu_l * dlognu)) + 1;
            int ic = (int)floor(xc + 0.5);
            int i0 = ic - half, i1 = ic + half;
            if (i0 < ICLO) i0 = ICLO; if (i1 > ICHI) i1 = ICHI;
            if (i0 > i1) continue;
            double frac = (tau > 1e-6) ? -expm1(-tau) : tau;   /* 1-e^{-tau} */
            double chi0 = frac * chi0_pref;
            double Bl = cm_planck(nu_l, plasma->T_e[s]);
            double Sl = src_nlte ? opac->line_source_S[(size_t)l*NS+s] : 0.0;
            if (Sl <= 0.0) Sl = Bl;
            if (Bl > 0.0) {
                double rb = Sl / Bl; if (rb > max_slb) max_slb = rb;
            }
            if (sl_clamp > 0.0 && Bl > 0.0 && Sl > sl_clamp * Bl) {
                Sl = sl_clamp * Bl;
                ++n_clamped;
            }
            double emit = (line_eps < 1.0) ? (line_eps * Bl) : Sl;
            for (int i = i0; i <= i1; ++i) {
                double xv = (fs.nu[i] - nu_l) / dnuD;
                double p  = exp(-xv * xv);
                double cl = chi0 * p;
                fs.chi_line[(size_t)s*NF+i] += cl;
                eta        [(size_t)s*NF+i] += cl * emit;
            }
        }
    }
    }
    free(line_use);

    /* --- assemble chi_tot + S_fixed.  eps<1: fold (1-eps)*chi_line into chi_es
     * (ALI scattering albedo) so the line forest scatters the photospheric UV
     * field instead of thermalising it; eta already holds only eps*chi_line*B.
     * Total opacity is preserved: chi_es+(1-eps)cl + chi_abs + eps*cl = orig. */
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < NS; ++s) {
        double Te = plasma->T_e[s];
        for (int i = 0; i < NF; ++i) {
            size_t idx = (size_t)s*NF+i;
            if (line_eps < 1.0) fs.chi_es[idx] += (1.0 - line_eps) * fs.chi_line[idx];
            double chi_ln_th = (line_eps < 1.0) ? (line_eps * fs.chi_line[idx]) : fs.chi_line[idx];
            double ct = fs.chi_es[idx] + fs.chi_abs[idx] + chi_ln_th;
            fs.chi_tot[idx] = ct;
            double Bnu = cm_planck(fs.nu[i], Te);
            fs.S_fixed[idx] = (ct > 0.0)
                ? (fs.chi_abs[idx]*Bnu + eta[idx]) / ct : 0.0;
            fs.J[idx] = Bnu;                              /* warm ALI start */
        }
    }

    if (diag) {   /* tie-back: Int chi_line dnu vs Sobolev expectation (shell NS/2) */
        int st = NS/2; double got=0.0, exp_=0.0;
        for (int i = 0; i < NF; ++i) got += fs.chi_line[(size_t)st*NF+i]*fs.dnu[i];
        for (int l = 0; l < NL; ++l) { double nu_l=opac->line_list_nu[l];
            if (nu_l<=nu_lo||nu_l>=nu_hi) continue;
            double tau=opac->tau_sobolev[(size_t)l*NS+st];
            double frac=(tau>1e-6)?-expm1(-tau):(tau>0?tau:0);
            exp_ += frac*nu_l/(CM_C*t_exp); }
        fprintf(stderr,
            "[cmf_fine] S_l deposit: max S_l/B=%.3e  clamped=%ld/%ld lines (sl_clamp=%.1f)  "
            "skipped weak(tau<%.2g)=%ld  line_eps=%.3g%s\n",
            max_slb, n_clamped, n_inwin, sl_clamp, fine_taumin, n_skip_weak,
            line_eps, (line_eps < 1.0) ? " [SCATTERING pump]" : " [thermal]");
        fprintf(stderr,
            "[cmf_fine] lines in window=%ld  tie-back shell %d: "
            "Int chi_line dnu=%.4e  Sobolev expect=%.4e  ratio=%.4f\n",
            n_inwin, st, got, exp_, (exp_>0)?got/exp_:0.0);
    }

    /* --- solve frequency-coupled J on the fine mesh (validated kernel) --- */
    cmf_solve_J(&fs, geo, T_inner, n_ali);

    /* --- DIAGNOSTIC: J/B(lambda, shell) map (LUMINA_CMF_FINE_JMAP=1).
     * Locates WHERE (wavelength, shell) the fine field is super-thermal -> the
     * only place a fluorescence pump can come from (physics-agent 2026-06-26).
     * read-only, no feedback. Coarse 100A bins, all shells. */
    { const char *e = getenv("LUMINA_CMF_FINE_JMAP");
      if (e && atoi(e)) {
        FILE *fp = fopen("lumina_fine_jmap.csv", "w");
        if (fp) {
            fprintf(fp, "shell,lambda_A,Te,J_over_B\n");
            double binA = 100.0;   /* 100 Angstrom bins */
            for (int s = 0; s < NS; ++s) {
                double Te = plasma->T_e[s];
                double lam0 = lam_lo, lam;
                for (lam = lam0; lam < lam_hi; lam += binA) {
                    double lam_c = lam + 0.5*binA;
                    double nu_c = CM_C / (lam_c * 1.0e-8);
                    /* nearest fine bin */
                    double xx = log(nu_c / nu_lo) / dlognu - 0.5;
                    int bi = (int)floor(xx + 0.5);
                    if (bi < 0 || bi >= NF) continue;
                    double Jv = fs.J[(size_t)s*NF+bi];
                    double Bv = cm_planck(fs.nu[bi], Te);
                    fprintf(fp, "%d,%.1f,%.0f,%.4e\n", s, lam_c, Te,
                            (Bv > 0.0) ? Jv/Bv : 0.0);
                }
            }
            fclose(fp);
            if (diag) fprintf(stderr, "[cmf_fine] wrote lumina_fine_jmap.csv (J/B vs lambda,shell)\n");
        }
      }
    }

    /* --- Stage-1 PROOF: frequency-resolved emergent from the fine field --- */
    { const char *e = getenv("LUMINA_CMF_FINE_EMERGENT");
      if (e && atoi(e))
          cmfgen_fine_emergent(&fs, csb, geo, T_inner, "lumina_spectrum_freqres.csv"); }

    /* --- UNIFIED Doppler emergent: P-Cygni + fine color + NLTE fluorescence --- */
    { const char *e = getenv("LUMINA_CMF_FINE_EMERGENT_OBS");
      if (e && atoi(e))
          cmfgen_fine_emergent_obs(&fs, geo, T_inner, opac, plasma->T_e,
                                   "lumina_spectrum_freqres_obs.csv"); }

    /* --- extract J_bar_l = Int phi_l J dnu / Int phi_l dnu per line --- */
    /* Per-line J_bar_l extraction: each line writes its own [l*NS .. ] slice of
     * jbar_line_det (read-only fine field fs.J) -> embarrassingly parallel. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (int l = 0; l < NL; ++l) {
        double nu_l = opac->line_list_nu[l];
        if (nu_l <= nu_lo || nu_l >= nu_hi) continue;
        double dnuD = nu_l * vdop / CM_C;
        double xc = log(nu_l / nu_lo) / dlognu - 0.5;
        int half  = (int)ceil(4.0 * dnuD / (nu_l * dlognu)) + 1;
        int ic = (int)floor(xc + 0.5);
        int i0 = ic - half, i1 = ic + half;
        if (i0 < 0) i0 = 0; if (i1 >= NF) i1 = NF - 1;
        for (int s = 0; s < NS; ++s) {
            double num = 0.0, den = 0.0;
            for (int i = i0; i <= i1; ++i) {
                double xv = (fs.nu[i] - nu_l) / dnuD;
                double p  = exp(-xv * xv) * fs.dnu[i];
                num += p * fs.J[(size_t)s*NF+i];
                den += p;
            }
            opac->jbar_line_det[(size_t)l*NS+s] = (den > 0.0) ? num/den : -1.0;
        }
    }

    if (diag) {   /* J_bar_l sanity + S_l/B (b_k proxy) per-iter at warm+cold shells.
                   * Loop over {0(inner warm), 6, NS/2(cold)} so an A/B + iteration
                   * trace shows whether the fluorescence converges (warm) vs the
                   * cold near-singular tail blows up. Also report S_l/J_bar (>>1 =
                   * multi-level fluorescence emission beyond the local field). */
        int diag_shells[3] = { 0, 6, NS/2 };
        for (int di = 0; di < 3; ++di) {
        int st = diag_shells[di]; if (st >= NS) continue; double Te = plasma->T_e[st];
        long nf=0; double jmin=1e300, jmax=-1e300, rsum=0.0; long rn2=0;
        double slsum=0.0; long sln=0, sl_hot=0; double sjmax=0.0; long sj_hi=0;
        double *slbuf = (double*)malloc((size_t)NL * sizeof(double));
        for (int l = 0; l < NL; ++l) {
            double v = opac->jbar_line_det[(size_t)l*NS+st];
            if (v < 0.0) continue;
            ++nf; if (v<jmin) jmin=v; if (v>jmax) jmax=v;
            double B = cm_planck(opac->line_list_nu[l], Te);
            if (B>0) { rsum += v/B; ++rn2;
                double Sl = opac->line_source_S ? opac->line_source_S[(size_t)l*NS+st] : 0.0;
                if (Sl > 0.0) { double r=Sl/B; slsum+=r;
                    if (slbuf) slbuf[sln]=r; ++sln; if (r>10.0) ++sl_hot;
                    double sj = Sl/v; if (sj>sjmax) sjmax=sj; if (sj>3.0) ++sj_hi; } }
        }
        double med=0.0, p90=0.0;
        if (slbuf && sln>0) {   /* sort for median / p90 */
            qsort(slbuf, sln, sizeof(double), cmf_dcmp);
            med = slbuf[sln/2]; p90 = slbuf[(long)(0.9*sln)];
        }
        free(slbuf);
        fprintf(stderr, "[cmf_fine] shell %d Te=%.0f: filled=%ld  Jbar_l in "
            "[%.3e,%.3e]  mean Jbar/B=%.3f | in-window S_l/B: n=%ld mean=%.3e "
            "MEDIAN=%.3f p90=%.3f hot(>10)=%ld(%.1f%%) | S_l/Jbar max=%.2e fluor(>3)=%ld\n",
            st, Te, nf, jmin, jmax,
            (rn2>0)?rsum/rn2:0.0, sln, (sln>0)?slsum/sln:0.0,
            med, p90, sl_hot, (sln>0)?100.0*sl_hot/sln:0.0, sjmax, sj_hi);
        }
    }

    /* Codex test E (LUMINA_CMF_FINE_LINEDUMP=1): per in-window line at mid shell,
     * dump J_fine (line-resolved jbar_line_det) vs J_binned (1000-bin csb->J at the
     * line) vs S_l/B. If high S_l/B correlates with J_binned/J_fine >> 1, the
     * super-thermal is a binned-J contrast-collapse ARTIFACT; if J_binned ~ J_fine,
     * the fluorescence is robust physics independent of the binning. */
    { const char *e = getenv("LUMINA_CMF_FINE_LINEDUMP");
      if (e && atoi(e)) {
        int st = NS/2; double Te = plasma->T_e[st];
        FILE *df = fopen("cmf_fine_linedump.csv", "w");
        if (df) {
            fprintf(df, "lambda_A,nu,J_fine,J_binned,S_l,B,SoverB,Jbin_over_Jfine\n");
            for (int l = 0; l < NL; ++l) {
                double jf = opac->jbar_line_det[(size_t)l*NS+st];
                if (jf < 0.0) continue;
                double nu_l = opac->line_list_nu[l];
                int b = (int)floor(log(nu_l/csb->nu_min)/csb->d_log_nu);
                double jb = (b>=0 && b<csb->n_bins) ? csb->J[(size_t)st*csb->n_bins+b] : -1.0;
                double B = cm_planck(nu_l, Te);
                double Sl = opac->line_source_S ? opac->line_source_S[(size_t)l*NS+st] : 0.0;
                fprintf(df, "%.3f,%.6e,%.6e,%.6e,%.6e,%.6e,%.4e,%.4e\n",
                    CM_C/nu_l*1e8, nu_l, jf, jb, Sl, B,
                    (B>0)?Sl/B:0.0, (jf>0&&jb>0)?jb/jf:0.0);
            }
            fclose(df);
            fprintf(stderr, "[cmf_fine] LINEDUMP wrote cmf_fine_linedump.csv (shell %d)\n", st);
        }
      }
    }

    /* FINE-ν PHOTOION (LUMINA_CMF_FINE_PHOTOION): retain the local fine-ν field
     * fs.J + fs.nu (transfer ownership to OpacityState; otherwise freed below) and
     * register it so coupled_photoion_rate_jnu integrates bf rates on the fine grid
     * instead of the binned J. */
    { static int fph = -1;
      if (fph < 0) { const char *e=getenv("LUMINA_CMF_FINE_PHOTOION"); fph=(e&&atoi(e))?1:0; }
      if (fph) {
          free(opac->jnu_fine); free(opac->nu_fine);
          opac->jnu_fine = fs.J;  fs.J  = NULL;   /* transfer (skip free below) */
          opac->nu_fine  = fs.nu; fs.nu = NULL;
          opac->n_fine = NF; opac->nu_lo_fine = nu_lo; opac->dlognu_fine = dlognu;
          coupled_set_fine_jnu(opac->jnu_fine, opac->nu_fine, NF,
                               nu_lo, dlognu, NS);
      }
    }

    free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);free(fs.chi_line);
    free(fs.chi_tot);free(fs.S_fixed);free(fs.J);free(eta);
}

/* ===================================================================
 * CONFIRMATION-LADDER T1: controlled single-line P-Cygni self-test.
 * Synthetic homologous ejecta + BB core + ONE pure-scatter resonance line.
 * Feeds the SAME cmfgen_fine_emergent_obs (whatever obs path the env selects:
 * scatter / SEI / faithful) and writes the profile. KNOWN ANSWER:
 *   - pure scatter ⇒ net equivalent width = 0 (exact photon conservation)
 *   - Castor/SEI P-Cygni shape (blue absorption, red emission)
 *   - tau_S→∞ ⇒ blue-edge flux → 0
 * Env: LUMINA_CMF_OBS_SELFTEST=1; LUMINA_CMF_SELFTEST_TAUS=tau_S (default 100). */
/* ===================================================================
 * CONFIRMATION-LADDER Stage N: NLTE level populations / line source S_l.
 *   N1 (two-level atom): analytic S_l = (J̄ + εB)/(1+ε), ε=(C_ul/A_ul)(1−e^{−hν/kT}).
 *       Solve the rate equation, verify num==ana AND limits (ε→0:S_l→J̄; ε→∞:S_l→B).
 *   N4 (three-level UV pump): ground(0)/mid(1)/upper(2). UV resonance pump 0→2 (2500A),
 *       fluorescence decay 2→1 (5000A), optical 1→0 (5000A). Feed the UV pump field
 *       either FREQ-RESOLVED (sees hot photospheric W·B(T_rad)) or BINNED (collapses to
 *       local-thermal B(T_e), the binned-J grey defect). KNOWN ANSWER: freq-resolved
 *       pumps b_2>1 → elevated optical S_l(2→1) = fluorescence; binned gives b_2≈1, none.
 *       This is the controlled capstone of the fluorescence saga. Env LUMINA_CMF_NLTE_SELFTEST=n1|n4. */
static void cmf_nlte_solve3(const double E[3], const double g[3],
                            const double A[3][3], const double Jbar[3][3],
                            double qcoef, double ne, double Te, double n_out[3])
{
    const double H=6.62607015e-27, C=2.99792458e10, K=1.380649e-16;
    double R[3][3]; for (int i=0;i<3;++i) for (int j=0;j<3;++j) R[i][j]=0.0;
    for (int lo=0; lo<3; ++lo) for (int up=lo+1; up<3; ++up) {
        if (A[up][lo]<=0.0) continue;
        double dE=E[up]-E[lo], nu=dE/H;
        double Bul=A[up][lo]*C*C/(2.0*H*nu*nu*nu);
        double Blu=(g[up]/g[lo])*Bul;
        double qul=qcoef*ne;                          /* de-excit rate s^-1 */
        double qlu=qul*(g[up]/g[lo])*exp(-dE/(K*Te));
        double J=Jbar[lo][up];
        R[up][lo]=A[up][lo]+Bul*J+qul;                /* up->lo (down) */
        R[lo][up]=Blu*J+qlu;                          /* lo->up (up)   */
    }
    /* statistical equilibrium M n = 0, replace row 0 with normalization */
    double M[3][3];
    for (int i=0;i<3;++i){ double out=0; for(int j=0;j<3;++j) if(j!=i) out+=R[i][j];
        for(int j=0;j<3;++j) M[i][j]=(i==j)?-out:R[j][i]; }
    for (int j=0;j<3;++j) M[0][j]=1.0; double rhs[3]={1.0,0.0,0.0};
    /* 3x3 Gaussian elimination */
    for (int c=0;c<3;++c){ int p=c; for(int r=c+1;r<3;++r) if(fabs(M[r][c])>fabs(M[p][c])) p=r;
        for(int j=0;j<3;++j){double t=M[c][j];M[c][j]=M[p][j];M[p][j]=t;} double t=rhs[c];rhs[c]=rhs[p];rhs[p]=t;
        for(int r=0;r<3;++r) if(r!=c){double f=M[r][c]/M[c][c]; for(int j=0;j<3;++j) M[r][j]-=f*M[c][j]; rhs[r]-=f*rhs[c];}}
    for (int i=0;i<3;++i) n_out[i]=rhs[i]/M[i][i];
}

/* ===================================================================
 * CONFIRMATION-LADDER Stage P: plasma (ionization + energy balance).
 *   P1 (Saha LTE limit): 2-ion ground system + e-. Collisional ion/3-body recomb
 *       (detailed-balance → Saha) + radiative recomb + photoionization (non-thermal
 *       field). KNOWN: ne→∞ ⇒ (n_{i+1}·n_e/n_i) → S_Saha(Te) DESPITE the photoion field.
 *   P2 (gray radiative equilibrium): J_ν = W·B_ν(T_rad). KNOWN: ∫(J−B(Te))dν=0 ⇒
 *       Te = W^{1/4}·T_rad. Tests the energy-balance root-finder. */
void cmf_plasma_selftest(const char *mode)
{
    const double H=6.62607015e-27, C=2.99792458e10, K=1.380649e-16;
    const double ME=9.1093837e-28, EV=1.602176634e-12, PI=3.14159265358979;
    if (!mode || strcmp(mode,"p1")==0) {
        double chi=13.6*EV, gi=9, gip1=4, Te=8000.0, Trad=10000.0, Wd=0.5;
        double S=2.0*(gip1/gi)*pow(2*PI*ME*K*Te/(H*H),1.5)*exp(-chi/(K*Te)); /* Saha RHS cm^-3 */
        double Cion=1e-8*sqrt(Te)*exp(-chi/(K*Te));            /* collisional ion coeff */
        double alpha=2e-13*pow(Te/1e4,-0.7);                   /* radiative recomb coeff */
        double Gamma=Wd*1e-10*exp(-chi/(K*Trad));              /* photoion rate (hot field, toy) */
        printf("=== CONFIRMATION-LADDER P1 (Saha LTE limit), Te=%.0f Trad=%.0f, photoion field ON ===\n",Te,Trad);
        printf("  %-10s %-13s %-13s %-10s\n","ne","(n+·ne/n)","S_Saha(Te)","ratio");
        for (double ne=1e6; ne<=1e16; ne*=100){
            double x=(Cion*ne+Gamma)/(alpha*ne + Cion*ne*ne/S);  /* n_{i+1}/n_i */
            printf("  %-10.0e %-13.4e %-13.4e %-10.4f\n",ne,x*ne,S,x*ne/S);
        }
        printf("  KNOWN: ne→∞ ⇒ (n+·ne/n)→S_Saha (ratio→1) DESPITE photoion; low ne ratio>1 (field over-ionizes, NLTE).\n");
        printf("  ⟹ P1 PASS if ratio→1 at high ne (ionization rate network has correct Saha/LTE limit).\n");
    }
    if (!mode || strcmp(mode,"p2")==0) {
        double Trad=10000.0;
        int NF=400; double nu_lo=C/(30000e-8), nu_hi=C/(1000e-8), dln=log(nu_hi/nu_lo)/NF;
        printf("=== CONFIRMATION-LADDER P2 (gray radiative equilibrium), Trad=%.0f ===\n",Trad);
        printf("  %-7s %-11s %-13s %-9s\n","W","Te(solve)","Te=W^.25·Tr","ratio");
        double Wlist[4]={1.0,0.5,0.25,0.1};
        for (int w=0; w<4; ++w){ double W=Wlist[w];
            double lo=1000, hi=30000;
            for (int it=0; it<60; ++it){ double Te=0.5*(lo+hi), r=0;
                for (int i=0;i<NF;++i){ double nu=nu_lo*exp((i+0.5)*dln), dnu=nu*dln;
                    r += (W*cm_planck(nu,Trad)-cm_planck(nu,Te))*dnu; }
                if (r>0) lo=Te; else hi=Te; }
            double Te=0.5*(lo+hi), Tek=pow(W,0.25)*Trad;
            printf("  %-7.2f %-11.1f %-13.1f %-9.4f\n",W,Te,Tek,Te/Tek);
        }
        printf("  KNOWN: gray radeq ∫(W·B(Trad)−B(Te))dν=0 ⇒ Te=W^{1/4}·Trad. ratio→1 ⇒ energy-balance root-finder correct.\n");
    }
}

void cmf_nlte_selftest(const char *mode)
{
    const double H=6.62607015e-27, C=2.99792458e10, K=1.380649e-16;
    double Te=8000.0, Trad=10000.0, W=0.5;
    if (!mode || strcmp(mode,"n1")==0) {
        double gl=2,gu=4, lam=5000e-8, nu=C/lam, dE=H*nu, Aul=1e8;
        double twohnu3c2=2*H*nu*nu*nu/(C*C), Bnu=cm_planck(nu,Te), Jbar=W*Bnu;
        printf("=== CONFIRMATION-LADDER N1 (two-level S_l), Te=%.0f lam=5000A, Jbar=W·B=%.2fB ===\n",Te,W);
        printf("  %-9s %-10s %-11s %-11s %-9s\n","ne","eps","S_l/B(num)","S_l/B(ana)","match");
        for (double ne=1e5; ne<=1e13; ne*=100){
            double Bul=Aul*C*C/(2*H*nu*nu*nu), Blu=(gu/gl)*Bul;
            double qul=1e-7*ne, qlu=qul*(gu/gl)*exp(-dE/(K*Te));
            double Rlu=Blu*Jbar+qlu, Rul=Aul+Bul*Jbar+qul, ratio=Rlu/Rul; /* nu/nl */
            double Sl=twohnu3c2/((gu/gl)/ratio-1.0);
            double eps=(qul/Aul)*(1-exp(-dE/(K*Te))), Sl_ana=(Jbar+eps*Bnu)/(1+eps);
            printf("  %-9.0e %-10.3e %-11.4f %-11.4f %-9s\n",ne,eps,Sl/Bnu,Sl_ana/Bnu,
                   (fabs(Sl-Sl_ana)/Sl_ana<1e-6)?"OK":"FAIL");
        }
        printf("  KNOWN: ε→0 S_l→Jbar(0.50B); ε→∞ S_l→B(1.0); num==ana ⇒ 2-level machinery PASS\n");
    }
    if (!mode || strcmp(mode,"n4")==0) {
        /* 3 levels: 0 ground, 1 mid(5000A from g), 2 upper(2500A from g) */
        double lam10=5000e-8, lam02=2500e-8;
        double E[3]={0.0, H*C/lam10, H*C/lam02}, g[3]={2,4,2};
        double lam21=H*C/(E[2]-E[1]);  /* fluorescence line 2->1 */
        double A[3][3]={{0}}; A[2][0]=1e8; A[2][1]=1e7; A[1][0]=1e7;  /* UV pump / fluor / optical */
        double nu02=(E[2]-E[0])/H, nu21=(E[2]-E[1])/H, nu10=(E[1]-E[0])/H;
        double ne=1e8, qcoef=1e-8;   /* nebular: weak collisions */
        printf("=== CONFIRMATION-LADDER N4 (3-level UV pump→fluorescence), Te=%.0f Trad=%.0f ===\n",Te,Trad);
        printf("  pump 0→2=%.0fA, fluor 2→1=%.0fA, optical 1→0=%.0fA; ne=%.0e\n",lam02*1e8,lam21*1e8,lam10*1e8,ne);
        for (int cs=0; cs<2; ++cs){
            int freq=(cs==0);
            double Jbar[3][3]={{0}};
            /* UV pump 0↔2: FREQ sees hot photospheric W·B(Trad); BINNED collapses to local B(Te) */
            Jbar[0][2]=Jbar[2][0]= freq ? W*cm_planck(nu02,Trad) : cm_planck(nu02,Te);
            Jbar[1][2]=Jbar[2][1]= W*cm_planck(nu21,Te);     /* optical: both ~W·B(Te) */
            Jbar[0][1]=Jbar[1][0]= W*cm_planck(nu10,Te);
            double n[3]; cmf_nlte_solve3(E,g,A,Jbar,qcoef,ne,Te,n);
            /* LTE pops at Te for departure b_i = n_i/n_i^LTE (normalized) */
            double zl=0; double nlte_lte[3]; for(int i=0;i<3;++i){nlte_lte[i]=g[i]*exp(-E[i]/(K*Te)); zl+=nlte_lte[i];}
            for(int i=0;i<3;++i) nlte_lte[i]/=zl;
            double ntot=n[0]+n[1]+n[2];
            double b2=(n[2]/ntot)/nlte_lte[2], b1=(n[1]/ntot)/nlte_lte[1];
            /* S_l of fluorescence line 2→1 */
            double twohnu3c2=2*H*nu21*nu21*nu21/(C*C);
            double Sl21=twohnu3c2/((g[2]/g[1])*(n[1]/n[2])-1.0);
            double B21=cm_planck(nu21,Te);
            printf("  [%s] UV J̄_02/B(Te)=%6.1f  b_2=%7.3f  b_1=%6.3f  S_l(2→1)/B(Te)=%7.3f %s\n",
                   freq?"FREQ ":"BIN  ", Jbar[0][2]/cm_planck(nu02,Te), b2, b1, Sl21/B21,
                   freq?"":" (binned baseline)");
        }
        printf("  KNOWN: FREQ pump (UV J̄≫B) → b_2>1 + S_l(2→1)>1 = FLUORESCENCE; BIN → b_2≈1, S_l≈1 (none).\n");
        printf("  ⟹ if FREQ shows fluorescence and BIN does not, binned-J STRUCTURALLY cannot pump (saga capstone).\n");
    }
    if (!mode || strcmp(mode,"n2")==0) {
        /* LTE limit: high collisions must thermalize pops → Boltzmann@Te DESPITE a
         * non-thermal (10×B UV) field. Same 3-level atom as N4, ramp n_e. */
        double lam10=5000e-8, lam02=2500e-8;
        double E[3]={0.0, H*C/lam10, H*C/lam02}, g[3]={2,4,2};
        double A[3][3]={{0}}; A[2][0]=1e8; A[2][1]=1e7; A[1][0]=1e7;
        double nu21=(E[2]-E[1])/H;
        double Jbar[3][3]={{0}};
        Jbar[0][2]=Jbar[2][0]=10.0*cm_planck((E[2]-E[0])/H,Te);  /* hot UV pump (non-thermal) */
        Jbar[1][2]=Jbar[2][1]=W*cm_planck(nu21,Te);
        Jbar[0][1]=Jbar[1][0]=W*cm_planck((E[1]-E[0])/H,Te);
        printf("=== CONFIRMATION-LADDER N2 (LTE limit: high collisions→Boltzmann), Te=%.0f, field NON-thermal(UV 10×B) ===\n",Te);
        printf("  %-10s %-9s %-9s %-9s %-12s\n","ne","b_0","b_1","b_2","S_l(2→1)/B");
        for (double ne=1e8; ne<=1e18; ne*=1000){
            double n[3]; cmf_nlte_solve3(E,g,A,Jbar,1e-7,ne,Te,n);
            double zl=0,nl[3]; for(int i=0;i<3;++i){nl[i]=g[i]*exp(-E[i]/(K*Te));zl+=nl[i];} for(int i=0;i<3;++i)nl[i]/=zl;
            double ntot=n[0]+n[1]+n[2];
            double twohnu3c2=2*H*nu21*nu21*nu21/(C*C);
            double Sl21=twohnu3c2/((g[2]/g[1])*(n[1]/n[2])-1.0), B21=cm_planck(nu21,Te);
            printf("  %-10.0e %-9.3f %-9.3f %-9.3f %-12.3f\n",ne,(n[0]/ntot)/nl[0],(n[1]/ntot)/nl[1],(n[2]/ntot)/nl[2],Sl21/B21);
        }
        printf("  KNOWN: ne→∞ (collisions dominate) ⇒ b_i→1 ALL levels (Boltzmann@Te), S_l→1, DESPITE the 10×B UV pump.\n");
        printf("  ⟹ N2 PASS if b_i→1 at high ne (rate matrix has correct LTE limit; collisions wash out non-thermal field).\n");
    }
    if (!mode || strcmp(mode,"n5")==0 || strcmp(mode,"cond")==0) {
        /* CONDITIONING (ARTIS recipe, nltepop.cc:733): an NLTE-like detailed-balance
         * rate matrix whose equilibrium x_true spans many orders → cond≫1/eps → raw LU
         * garbage. ARTIS fix = iterative ROW-COL equilibration (f=√(colL2/rowL2), 10×)
         * before LU. KNOWN ANSWER = x_true (Boltzmann). Raw vs equilibrated vs x_true. */
        enum { NN=10 };
        printf("=== CONFIRMATION-LADDER N5 (NLTE matrix conditioning, ARTIS row-col equilibration) ===\n");
        printf("  detailed-balance rate matrix, equilibrium x_true=Boltzmann; raw LU vs ARTIS-equilibrated LU vs x_true\n");
        printf("  %-8s %-12s %-13s %-13s %-6s\n","decade","span","RAW err","EQ(ARTIS) err","PASS");
        double decades[5]={28,56,84,140,200};   /* x_true span in orders of magnitude */
        for (int sp=0; sp<5; ++sp){
        double perlev = decades[sp]*log(10.0)/(NN-1);   /* exp(-perlev*i) gives the target span */
        double xt[NN]; for(int i=0;i<NN;++i) xt[i]=exp(-perlev*i);
        double M0[NN*NN]; for(int i=0;i<NN*NN;++i) M0[i]=0.0;
        for(int i=0;i<NN;++i){ double dout=0;
            for(int j=0;j<NN;++j) if(j!=i){
                double Rji=(j<i)?1.0:xt[i]/xt[j];   /* rate j→i: up=1, down=xt[i]/xt[j] */
                M0[i*NN+j]=Rji;
                double Rik=(i<j)?1.0:xt[j]/xt[i];   /* rate i→j (for diagonal) */
                dout+=Rik;
            }
            M0[i*NN+i]=-dout;
        }
        double sumxt=0; for(int i=0;i<NN;++i) sumxt+=xt[i];
        for(int j=0;j<NN;++j) M0[0*NN+j]=1.0;       /* row 0 = normalization Σx=sumxt */
        /* local LU solve (partial pivot) on row-major A[NN*NN], b[NN] -> x[NN] */
        #define LUSOLVE(Asrc,bsrc,xout) do{ double A[NN*NN],b[NN]; \
            for(int _i=0;_i<NN*NN;_i++)A[_i]=(Asrc)[_i]; for(int _i=0;_i<NN;_i++)b[_i]=(bsrc)[_i]; \
            for(int c=0;c<NN;c++){ int p=c; for(int r=c+1;r<NN;r++) if(fabs(A[r*NN+c])>fabs(A[p*NN+c]))p=r; \
              for(int k=0;k<NN;k++){double t=A[c*NN+k];A[c*NN+k]=A[p*NN+k];A[p*NN+k]=t;} double tb=b[c];b[c]=b[p];b[p]=tb; \
              for(int r=0;r<NN;r++) if(r!=c){double f=A[r*NN+c]/A[c*NN+c]; for(int k=0;k<NN;k++)A[r*NN+k]-=f*A[c*NN+k]; b[r]-=f*b[c];}} \
            for(int _i=0;_i<NN;_i++)(xout)[_i]=b[_i]/A[_i*NN+_i]; }while(0)
        double bvec[NN]={0}; bvec[0]=sumxt;
        double xraw[NN]; LUSOLVE(M0,bvec,xraw);
        /* ARTIS iterative row-col equilibration */
        double Meq[NN*NN],beq[NN],cscale[NN]; for(int i=0;i<NN*NN;i++)Meq[i]=M0[i];
        for(int i=0;i<NN;i++){beq[i]=bvec[i];cscale[i]=1.0;}
        for(int it=0;it<10;it++){ int changed=0;
            for(int i=0;i<NN;i++){ double rn=0,cn=0;
                for(int j=0;j<NN;j++){rn+=Meq[i*NN+j]*Meq[i*NN+j]; cn+=Meq[j*NN+i]*Meq[j*NN+i];}
                rn=sqrt(rn);cn=sqrt(cn); if(rn==0||cn==0)continue;
                double f=sqrt(cn/rn); if(fabs(f-1.0)<1e-3)continue; changed=1;
                for(int j=0;j<NN;j++)Meq[i*NN+j]*=f; beq[i]*=f;          /* row i ×f */
                for(int j=0;j<NN;j++)Meq[j*NN+i]/=f; cscale[i]/=f;       /* col i ÷f */
            }
            if(!changed)break;
        }
        double yeq[NN],xeq[NN]; LUSOLVE(Meq,beq,yeq); for(int i=0;i<NN;i++)xeq[i]=cscale[i]*yeq[i];
        #undef LUSOLVE
        double eraw=0,eeq=0; for(int i=0;i<NN;i++){
            eraw=fmax(eraw,fabs(xraw[i]-xt[i])/fabs(xt[i])); eeq=fmax(eeq,fabs(xeq[i]-xt[i])/fabs(xt[i])); }
        printf("  %-8.0f %.0e..%-6.0e %-13.3e %-13.3e %-6s\n", decades[sp], xt[0], xt[NN-1], eraw, eeq,
               (eeq < 1e-3 && eeq < eraw*0.01) ? "EQ✓" : (eraw<1e-3?"both ok":"—"));
        }
        printf("  KNOWN: equilibrated recovers x_true (err≪1) even as span→200 orders; raw LU degrades.\n");
        printf("  ⟹ N5 PASS = ARTIS row-col equilibration holds where raw LU fails (the conditioning fix).\n");
    }
    if (!mode || strcmp(mode,"n6")==0) {
        /* SUPER-LEVELS (ARTIS recipe, nltepop.cc:1411 superlevel_boltzmann): solve an
         * N-level atom (a) FULL explicit vs (b) K explicit + high levels lumped into ONE
         * Boltzmann(T_exc) super-level (s_renorm-weighted rates). KNOWN ANSWER = full
         * solution. Super-level should recover the low-level pops AND cut the matrix
         * dimension N→K+1. Validates the span-reduction conditioning fix before plasma.c. */
        enum { NL=24 };
        const double EV=1.602176634e-12;
        int Kx=6;  /* explicit low levels 0..Kx-1; Kx..NL-1 → super-level */
        double Eg[NL], gg[NL];
        for(int i=0;i<NL;++i){ Eg[i]=i*0.40*EV; gg[i]=2.0*i+1.0; }   /* ladder, g=2i+1 */
        double ne=1e9, qc=1e-8, Te_=8000.0, Texc=8000.0;
        /* radiative+collisional rates between all pairs (toy: A∝1/(E_u-E_l)^3-ish, here A=1e7) */
        /* --- build full NL×NL statistical-equilibrium matrix --- */
        double Jbar=0.5;  /* W; field = W·B(nu,Trad=10000) per pair */
        double Tr=10000.0;
        #define RATE_UP(lo,hi)  ( atom_blu(lo,hi)*Jbar*cm_planck((Eg[hi]-Eg[lo])/H,Tr) + qc*ne*(gg[hi]/gg[lo])*exp(-(Eg[hi]-Eg[lo])/(K*Te_)) )
        #define RATE_DN(lo,hi)  ( atom_aul(lo,hi) + atom_blu(lo,hi)*(gg[lo]/gg[hi])*Jbar*cm_planck((Eg[hi]-Eg[lo])/H,Tr) + qc*ne )
        /* helper lambdas via macros referencing A_ul=1e7, B_lu from A */
        double Aul=1e7;
        #define atom_aul(lo,hi) (Aul)
        #define atom_blu(lo,hi) (Aul*C*C/(2*H*pow((Eg[hi]-Eg[lo])/H,3.0))*(gg[hi]/gg[lo]))
        /* FULL solve */
        static double Mf[NL*NL]; for(int i=0;i<NL*NL;++i)Mf[i]=0;
        for(int lo=0;lo<NL;++lo)for(int hi=lo+1;hi<NL;++hi){
            double ru=RATE_UP(lo,hi), rd=RATE_DN(lo,hi);
            Mf[hi*NL+lo]+=ru; Mf[lo*NL+hi]+=rd;          /* into hi from lo; into lo from hi */
            Mf[lo*NL+lo]-=ru; Mf[hi*NL+hi]-=rd;          /* out of lo (up), out of hi (down) */
        }
        for(int j=0;j<NL;++j)Mf[0*NL+j]=1.0;             /* row0 = normalization */
        #define SOLVEGEN(N,Asrc,bsrc,xout) do{ static double A[NL*NL],b[NL]; \
            for(int _i=0;_i<(N)*(N);_i++)A[_i]=(Asrc)[_i]; for(int _i=0;_i<(N);_i++)b[_i]=(bsrc)[_i]; \
            for(int c=0;c<(N);c++){int p=c;for(int r=c+1;r<(N);r++)if(fabs(A[r*(N)+c])>fabs(A[p*(N)+c]))p=r; \
              for(int k=0;k<(N);k++){double t=A[c*(N)+k];A[c*(N)+k]=A[p*(N)+k];A[p*(N)+k]=t;}double tb=b[c];b[c]=b[p];b[p]=tb; \
              for(int r=0;r<(N);r++)if(r!=c){double f=A[r*(N)+c]/A[c*(N)+c];for(int k=0;k<(N);k++)A[r*(N)+k]-=f*A[c*(N)+k];b[r]-=f*b[c];}} \
            for(int _i=0;_i<(N);_i++)(xout)[_i]=b[_i]/A[_i*(N)+_i]; }while(0)
        double bf[NL]={0}; bf[0]=1.0; double nf[NL]; SOLVEGEN(NL,Mf,bf,nf);
        /* --- SUPER-LEVEL solve: dim = Kx+1 (levels 0..Kx-1 explicit, index Kx = super) --- */
        double zsl=0, sb[NL]; for(int i=Kx;i<NL;++i){ sb[i]=gg[i]*exp(-(Eg[i]-Eg[Kx])/(K*Texc)); zsl+=sb[i]; }
        double sren[NL]; for(int i=Kx;i<NL;++i) sren[i]=sb[i]/zsl;     /* Boltzmann fraction within SL */
        int DS=Kx+1; static double Ms[NL*NL]; for(int i=0;i<DS*DS;++i)Ms[i]=0;
        #define IDX(i) ((i)<Kx?(i):Kx)                    /* map physical level → super index */
        for(int lo=0;lo<NL;++lo)for(int hi=lo+1;hi<NL;++hi){
            double ru=RATE_UP(lo,hi), rd=RATE_DN(lo,hi);
            double wlo=(lo<Kx)?1.0:sren[lo], whi=(hi<Kx)?1.0:sren[hi];  /* SL members enter ×Boltzmann frac */
            int a=IDX(lo), c=IDX(hi);
            Ms[c*DS+a]+=ru*wlo; Ms[a*DS+c]+=rd*whi;
            Ms[a*DS+a]-=ru*wlo; Ms[c*DS+c]-=rd*whi;
        }
        for(int j=0;j<DS;++j)Ms[0*DS+j]=1.0; double bs[NL]={0}; bs[0]=1.0; double ns[NL]; SOLVEGEN(DS,Ms,bs,ns);
        #undef SOLVEGEN
        #undef RATE_UP
        #undef RATE_DN
        #undef atom_aul
        #undef atom_blu
        #undef IDX
        /* compare low-level pops (normalize full so Σ=1; super already Σ=1 over DS) */
        double sumf=0; for(int i=0;i<NL;++i)sumf+=nf[i]; for(int i=0;i<NL;++i)nf[i]/=sumf;
        double slfull=0; for(int i=Kx;i<NL;++i)slfull+=nf[i];
        printf("=== CONFIRMATION-LADDER N6 (super-levels, ARTIS recipe), NL=%d, K_explicit=%d → dim %d→%d ===\n",NL,Kx,NL,DS);
        printf("  %-5s %-13s %-13s %-8s\n","lev","n_full","n_super","rel.err");
        double maxe=0; for(int i=0;i<Kx;++i){ double e=fabs(ns[i]-nf[i])/fmax(nf[i],1e-30); maxe=fmax(maxe,e);
            printf("  %-5d %-13.4e %-13.4e %-8.1e\n",i,nf[i],ns[i],e); }
        double esl=fabs(ns[Kx]-slfull)/fmax(slfull,1e-30);
        printf("  SL    %-13.4e %-13.4e %-8.1e (super-level total vs Σ full high)\n",slfull,ns[Kx],esl);
        printf("  max low-level rel.err = %.2e\n",maxe);
        printf("  KNOWN: super-level (Boltzmann high-E lump) recovers explicit low-level pops; dim %d→%d.\n",NL,DS);
        printf("  ⟹ N6 PASS if low-level err≪1 (super-level approx valid) → span-reduction conditioning fix works.\n");
    }
}

/* ===================================================================
 * CONFIRMATION-LADDER Stage F: cmf_solve_J (comoving J̄ formal solve) on
 * known-answer cases. Env LUMINA_CMF_FSOLVE_SELFTEST = "absorb"|"dilute".
 *   F1 absorb: isothermal, pure absorption (chi_abs only), thick, S_fixed=B(T)
 *              → J → B(T) in the interior (thermalization). Known: J/B → 1.
 *   F3 dilute: thin e-scatter halo (tau_es~0.3) + BB core, chi_abs=0, S_fixed=0
 *              → J → W(r)·B(T_inner) at outer shells. Known: J/(W·B) → 1. */
void cmf_fsolve_selftest(const char *mode)
{
    int NS=40, NF=200; { const char *e=getenv("LUMINA_CMF_SELFTEST_NS"); if(e) NS=atoi(e); }
    double t_exp=86400.0, Tinner=10000.0;
    double vph=5.0e8, vmax=2.5e9;
    int absorb = (mode && strcmp(mode,"absorb")==0);
    Geometry geo; memset(&geo,0,sizeof geo); geo.n_shells=NS; geo.time_explosion=t_exp;
    geo.r_inner=malloc(NS*sizeof(double)); geo.r_outer=malloc(NS*sizeof(double));
    geo.v_inner=malloc(NS*sizeof(double)); geo.v_outer=malloc(NS*sizeof(double));
    for (int s=0;s<NS;++s){ double v0=vph+(vmax-vph)*s/(double)NS,v1=vph+(vmax-vph)*(s+1)/(double)NS;
        geo.v_inner[s]=v0;geo.v_outer[s]=v1;geo.r_inner[s]=v0*t_exp;geo.r_outer[s]=v1*t_exp; }
    double nu_min=CM_C/(6000e-8), nu_max=CM_C/(4000e-8), dlognu=log(nu_max/nu_min)/NF;
    CMFGENState fs; memset(&fs,0,sizeof fs);
    fs.n_shells=NS;fs.n_bins=NF;fs.nu_min=nu_min;fs.nu_max=nu_max;fs.d_log_nu=dlognu;
    fs.nu=malloc(NF*sizeof(double));fs.dnu=malloc(NF*sizeof(double));
    fs.chi_es=calloc((size_t)NS*NF,sizeof(double));fs.chi_abs=calloc((size_t)NS*NF,sizeof(double));
    fs.chi_line=calloc((size_t)NS*NF,sizeof(double));fs.chi_tot=calloc((size_t)NS*NF,sizeof(double));
    fs.S_fixed=calloc((size_t)NS*NF,sizeof(double));fs.J=calloc((size_t)NS*NF,sizeof(double));
    for (int i=0;i<NF;++i){ fs.nu[i]=nu_min*exp((i+0.5)*dlognu); fs.dnu[i]=fs.nu[i]*dlognu; }
    double dr=geo.r_outer[NS-1]-geo.r_inner[0];
    for (int s=0;s<NS;++s) for (int i=0;i<NF;++i){ size_t idx=(size_t)s*NF+i;
        double B=cm_planck(fs.nu[i],Tinner);
        if (absorb){ double chia=100.0/dr; fs.chi_abs[idx]=chia; fs.chi_tot[idx]=chia;
                     fs.S_fixed[idx]=B; fs.J[idx]=0.5*B; }            /* thick, S=B → J→B */
        else { double taues=0.3; { const char *e=getenv("LUMINA_CMF_FSOLVE_TAUES"); if(e) taues=atof(e); }
               double ces=taues/dr; fs.chi_es[idx]=ces; fs.chi_tot[idx]=ces;
               fs.S_fixed[idx]=0.0; fs.J[idx]=0.5*B; }                /* thin scatter+core */
    }
    int n_ali=24; { const char *e=getenv("LUMINA_CMFGEN_ALI_ITER"); if(e) n_ali=atoi(e); }
    cmf_solve_J(&fs,&geo,Tinner,n_ali);
    int ic=NF/2;  /* mid frequency */
    printf("=== CONFIRMATION-LADDER F (%s), cmf_solve_J ===\n", absorb?"F1 absorb J→B":"F3 dilute J→W·B");
    for (int s=0;s<NS;s+=(absorb?NS/8:NS/8)){
        double B=cm_planck(fs.nu[ic],Tinner), J=fs.J[(size_t)s*NF+ic];
        double r=0.5*(geo.r_inner[s]+geo.r_outer[s]);
        double a=1.0-(geo.r_inner[0]*geo.r_inner[0])/(r*r), W=0.5*(1.0-sqrt(a>0?a:0));
        if (absorb) printf("  s=%2d  J/B=%.4f  (known: →1.0 thick interior)\n", s, J/B);
        else        printf("  s=%2d  J/(W·B)=%.4f  W=%.4f  (known: →1.0 dilution)\n", s, (W>0)?J/(W*B):0.0, W);
    }
    free(geo.r_inner);free(geo.r_outer);free(geo.v_inner);free(geo.v_outer);
    free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);free(fs.chi_line);
    free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
}

/* ============================================================ */
/* GPU cmf_solve_J self-check (model-free; dispatched pre-model from main under
 * env LUMINA_CMF_SOLVE_SELFTEST). Builds a representative fine grid -- a thin
 * e-scatter halo + warm BB core (continuum) plus, unless mode=="cont", a strong
 * Gaussian SCATTERING line deposited into chi_es (the producer's UV-pump pattern,
 * which stresses the blue->red advection across the line). Then calls cmf_solve_J,
 * which -- with LUMINA_CMF_SOLVE_GPU=2 -- runs BOTH the CPU and GPU solvers from
 * the same input and prints the max relative J difference + ALI iter counts.
 *   LUMINA_CMF_SOLVE_SELFTEST_NS / _NF / _TAUS tune the grid. */
void cmf_solve_gpu_selftest(const char *mode)
{
    int NS=40, NF=400;
    { const char *e=getenv("LUMINA_CMF_SOLVE_SELFTEST_NS"); if(e) NS=atoi(e); }
    { const char *e=getenv("LUMINA_CMF_SOLVE_SELFTEST_NF"); if(e) NF=atoi(e); }
    int with_line = !(mode && strcmp(mode,"cont")==0);
    double tauS=50.0; { const char *e=getenv("LUMINA_CMF_SOLVE_SELFTEST_TAUS"); if(e) tauS=atof(e); }
    double t_exp=86400.0, Tinner=10000.0, vph=5.0e8, vmax=2.5e9;
    Geometry geo; memset(&geo,0,sizeof geo); geo.n_shells=NS; geo.time_explosion=t_exp;
    geo.r_inner=malloc(NS*sizeof(double)); geo.r_outer=malloc(NS*sizeof(double));
    geo.v_inner=malloc(NS*sizeof(double)); geo.v_outer=malloc(NS*sizeof(double));
    for (int s=0;s<NS;++s){ double v0=vph+(vmax-vph)*s/(double)NS,v1=vph+(vmax-vph)*(s+1)/(double)NS;
        geo.v_inner[s]=v0;geo.v_outer[s]=v1;geo.r_inner[s]=v0*t_exp;geo.r_outer[s]=v1*t_exp; }
    /* fine uniform-log mesh over 4500-5500 A (line at 5000 A) */
    double nu_min=CM_C/(5500e-8), nu_max=CM_C/(4500e-8), dlognu=log(nu_max/nu_min)/NF;
    double nu_l=CM_C/(5000e-8), vdop=1.0e6;
    CMFGENState fs; memset(&fs,0,sizeof fs);
    fs.n_shells=NS;fs.n_bins=NF;fs.nu_min=nu_min;fs.nu_max=nu_max;fs.d_log_nu=dlognu;
    fs.nu=malloc(NF*sizeof(double));fs.dnu=malloc(NF*sizeof(double));
    fs.chi_es=calloc((size_t)NS*NF,sizeof(double));fs.chi_abs=calloc((size_t)NS*NF,sizeof(double));
    fs.chi_line=calloc((size_t)NS*NF,sizeof(double));fs.chi_tot=calloc((size_t)NS*NF,sizeof(double));
    fs.S_fixed=calloc((size_t)NS*NF,sizeof(double));fs.J=calloc((size_t)NS*NF,sizeof(double));
    for (int i=0;i<NF;++i){ fs.nu[i]=nu_min*exp((i+0.5)*dlognu); fs.dnu[i]=fs.nu[i]*dlognu; }
    double dr=geo.r_outer[NS-1]-geo.r_inner[0];
    double ces=0.3/dr, cab=0.05/dr;     /* tau_es~0.3 halo + weak true absorption */
    const double SQRTPI=1.7724538509055160;
    double dnuD=nu_l*vdop/CM_C, chi0=( (tauS>1e-6)?-expm1(-tauS):tauS )/(SQRTPI*vdop*t_exp);
    for (int s=0;s<NS;++s) for (int i=0;i<NF;++i){ size_t idx=(size_t)s*NF+i;
        double B=cm_planck(fs.nu[i],Tinner);
        double cl=0.0;
        if (with_line){ double xv=(fs.nu[i]-nu_l)/dnuD; cl=chi0*exp(-xv*xv); }
        fs.chi_es[idx]=ces+cl;                       /* line scatters (folded into chi_es) */
        fs.chi_abs[idx]=cab; fs.chi_line[idx]=0.0;
        fs.chi_tot[idx]=fs.chi_es[idx]+fs.chi_abs[idx];
        fs.S_fixed[idx]=(fs.chi_tot[idx]>0)?(cab*B)/fs.chi_tot[idx]:0.0;
        fs.J[idx]=0.5*B; }
    int n_ali=80; { const char *e=getenv("LUMINA_CMFGEN_ALI_ITER"); if(e) n_ali=atoi(e); }
    fprintf(stderr,"[cmf_gpu] selftest grid: NS=%d NF=%d line=%s tauS=%.1f n_ali=%d\n",
            NS,NF,with_line?"on":"off",tauS,n_ali);
    cmf_solve_J(&fs,&geo,Tinner,n_ali);   /* honours LUMINA_CMF_SOLVE_GPU (set =2 for A/B) */
    free(geo.r_inner);free(geo.r_outer);free(geo.v_inner);free(geo.v_outer);
    free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);free(fs.chi_line);
    free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
}

/* ... T1 single-line obs P-Cygni self-test ... */
void cmf_obs_selftest(void)
{
    int NS = 40; { const char *e=getenv("LUMINA_CMF_SELFTEST_NS"); if(e) NS=atoi(e); }
    double t_exp = 86400.0;                 /* 1 day */
    double vph = 5.0e8, vmax = 2.5e9;       /* 5000, 25000 km/s [cm/s] */
    { const char *e=getenv("LUMINA_CMF_SELFTEST_VSCALE"); if(e){ double v=atof(e); vph*=v; vmax*=v; } }
    double Tinner = 10000.0;
    double tauS = 100.0; { const char *e=getenv("LUMINA_CMF_SELFTEST_TAUS"); if(e) tauS=atof(e); }

    Geometry geo; memset(&geo,0,sizeof geo);
    geo.n_shells=NS; geo.time_explosion=t_exp;
    geo.r_inner=malloc(NS*sizeof(double)); geo.r_outer=malloc(NS*sizeof(double));
    geo.v_inner=malloc(NS*sizeof(double)); geo.v_outer=malloc(NS*sizeof(double));
    for (int s=0;s<NS;++s){
        double v0=vph+(vmax-vph)*s/(double)NS, v1=vph+(vmax-vph)*(s+1)/(double)NS;
        geo.v_inner[s]=v0; geo.v_outer[s]=v1; geo.r_inner[s]=v0*t_exp; geo.r_outer[s]=v1*t_exp;
    }
    double lam_l=5000e-8, nu_l=CM_C/lam_l;
    double nu_min=CM_C/(5500e-8), nu_max=CM_C/(4500e-8);
    double vdop=1.0e6, dlognu=(vdop/CM_C)/4.0;
    int NF=(int)(log(nu_max/nu_min)/dlognu)+1;
    CMFGENState fs; memset(&fs,0,sizeof fs);
    fs.n_shells=NS; fs.n_bins=NF; fs.nu_min=nu_min; fs.nu_max=nu_max; fs.d_log_nu=dlognu;
    fs.nu=malloc(NF*sizeof(double)); fs.dnu=malloc(NF*sizeof(double));
    fs.chi_es=calloc((size_t)NS*NF,sizeof(double)); fs.chi_abs=calloc((size_t)NS*NF,sizeof(double));
    fs.chi_line=calloc((size_t)NS*NF,sizeof(double)); fs.chi_tot=calloc((size_t)NS*NF,sizeof(double));
    fs.S_fixed=calloc((size_t)NS*NF,sizeof(double)); fs.J=calloc((size_t)NS*NF,sizeof(double));
    for (int i=0;i<NF;++i){ fs.nu[i]=nu_min*exp((i+0.5)*dlognu); fs.dnu[i]=fs.nu[i]*dlognu; }
    double dr=geo.r_outer[NS-1]-geo.r_inner[0], chi_es=0.1/dr;   /* thin halo tau_es~0.1 */
    for (int s=0;s<NS;++s){
        double r=0.5*(geo.r_inner[s]+geo.r_outer[s]);
        double a=1.0-(geo.r_inner[0]*geo.r_inner[0])/(r*r); double W=0.5*(1.0-sqrt(a>0?a:0));
        for (int i=0;i<NF;++i){ size_t idx=(size_t)s*NF+i;
            fs.chi_es[idx]=chi_es; fs.chi_tot[idx]=chi_es;
            fs.J[idx]=W*cm_planck(fs.nu[i],Tinner); }
    }
    OpacityState opac; memset(&opac,0,sizeof opac);
    opac.n_lines=1; opac.line_list_nu=malloc(sizeof(double)); opac.line_list_nu[0]=nu_l;
    opac.tau_sobolev=malloc((size_t)NS*sizeof(double));
    for (int s=0;s<NS;++s) opac.tau_sobolev[s]=tauS;
    double *Te=malloc(NS*sizeof(double)); for(int s=0;s<NS;++s) Te[s]=8000.0;
    fprintf(stderr,"[OBS-SELFTEST] single line 5000A tauS=%.1f, NS=%d NF=%d, beta=%.3f-%.3f\n",
            tauS,NS,NF,vph/CM_C,vmax/CM_C);
    /* CONFIRMATION-LADDER T1b: self-consistent comoving J̄ source (physics-agent fix).
     * Deposit the line into chi_es (pure scatter), solve cmf_solve_J for the
     * line-coupled J̄ field (line scattering raises the halo J̄ that fixed W·B
     * misses → the -68A static leak), extract J̄_l, feed the obs-march via
     * jbar_line_det (g_sob_jbardet). Gate LUMINA_CMF_SELFTEST_JBAR=1. */
    if (getenv("LUMINA_CMF_SELFTEST_JBAR") && atoi(getenv("LUMINA_CMF_SELFTEST_JBAR"))) {
        const double SQRTPI=1.7724538509055160;
        double dnuD=nu_l*vdop/CM_C, chi0_pref=1.0/(SQRTPI*vdop*t_exp);
        double frac=(tauS>1e-6)?-expm1(-tauS):tauS, chi0=frac*chi0_pref;
        double *cl_save=calloc((size_t)NS*NF,sizeof(double));
        for (int s=0;s<NS;++s) for (int i=0;i<NF;++i){
            double xv=(fs.nu[i]-nu_l)/dnuD, cl=chi0*exp(-xv*xv);
            size_t idx=(size_t)s*NF+i; cl_save[idx]=cl;
            fs.chi_es[idx]+=cl; fs.chi_tot[idx]=fs.chi_es[idx]+fs.chi_abs[idx];
            fs.J[idx]=cm_planck(fs.nu[i],Tinner)*0.5;  /* warm start */
        }
        int n_ali=24; { const char *e=getenv("LUMINA_CMFGEN_ALI_ITER"); if(e) n_ali=atoi(e); }
        cmf_solve_J(&fs,&geo,Tinner,n_ali);            /* self-consistent line-coupled J̄ */
        opac.jbar_line_det=malloc((size_t)NS*sizeof(double));
        for (int s=0;s<NS;++s){ double num=0,den=0;
            for (int i=0;i<NF;++i){ double xv=(fs.nu[i]-nu_l)/dnuD, phi=exp(-xv*xv);
                num+=phi*fs.J[(size_t)s*NF+i]; den+=phi; }
            opac.jbar_line_det[s]=(den>0)?num/den:-1.0; }
        for (int s=0;s<NS;++s) for (int i=0;i<NF;++i){    /* restore continuum chi_es for obs */
            size_t idx=(size_t)s*NF+i; fs.chi_es[idx]-=cl_save[idx];
            fs.chi_tot[idx]=fs.chi_es[idx]+fs.chi_abs[idx]; }
        free(cl_save);
        setenv("LUMINA_CMF_OBS_JBARDET","1",1);          /* obs uses J̄_l source */
        fprintf(stderr,"[OBS-SELFTEST] JBAR mode: self-consistent J̄_l[sh0]=%.3e [shNS/2]=%.3e (W*B sh0~%.3e)\n",
                opac.jbar_line_det[0],opac.jbar_line_det[NS/2],0.5*cm_planck(nu_l,Tinner));
    }
    cmfgen_fine_emergent_obs(&fs,&geo,Tinner,&opac,Te,"lumina_obs_selftest.csv");
    fprintf(stderr,"[OBS-SELFTEST] wrote lumina_obs_selftest.csv (analytic: net EW=0 for pure scatter)\n");
    free(geo.r_inner);free(geo.r_outer);free(geo.v_inner);free(geo.v_outer);
    free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);free(fs.chi_line);
    free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
    free(opac.line_list_nu);free(opac.tau_sobolev);free(Te);
    if (opac.jbar_line_det) free(opac.jbar_line_det);
}

int cmfgen_run(Geometry *geo, OpacityState *opac, BFOpacity *bf,
               PlasmaState *plasma, NLTEConfig *nlte, AtomicData *atom,
               GammaDeposition *gamma, double T_inner, int n_iter)
{
    if (getenv("LUMINA_CMF_OBS_SELFTEST")) { cmf_obs_selftest(); return 0; }
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

        /* P7 PRODUCER (LUMINA_CMF_LINERES_JBAR=1): fine-grid line-resolved J_bar_l
         * over the UV pump window. Fills opac->jbar_line_det; the plasma bb-rate
         * reads it (top priority, sentinel -1 => fall back). The binned cs above
         * already supplied the smooth continuum that cmfgen_fine_jbar interpolates,
         * and line_source_S carries the lagged line source (5d lagged-J scheme). */
        {
            static int lineres_jbar = -1;
            if (lineres_jbar < 0) { const char *e = getenv("LUMINA_CMF_LINERES_JBAR");
                lineres_jbar = (e && atoi(e)) ? 1 : 0; }
            if (lineres_jbar && opac->n_lines > 0 && opac->tau_sobolev) {
                if (!opac->jbar_line_det)
                    opac->jbar_line_det = (double*)malloc(
                        (size_t)opac->n_lines * cs.n_shells * sizeof(double));
                if (opac->jbar_line_det)
                    cmfgen_fine_jbar(&cs, geo, opac, T_inner, plasma);
            }
        }

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
