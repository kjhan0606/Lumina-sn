/* ============================================================ */
/* lumina_cmfgen.h                                              */
/*                                                              */
/* PURE-CMFGEN parallel radiation solver (coexists with the MC */
/* transport path; selected at runtime by LUMINA_PURE_CMFGEN=1).*/
/*                                                              */
/* Replaces ONLY the Monte-Carlo radiation field with a         */
/* deterministic comoving-frame spherical formal solution:      */
/*   - tangent-ray (impact-parameter p,z) short-characteristics */
/*   - expansion (Sobolev-binned) line opacity subsumes the     */
/*     comoving d/dnu term so each bin is solved statically      */
/*   - total chi = electron-scatter + bf + ff + line-expansion   */
/*   - source S = (chi_es*J + chi_abs*B + chi_line*S_line)/chi   */
/*   - coherent isotropic e-scattering closed by diagonal ALI    */
/* The emergent J_nu(shell,bin) is written into NLTEConfig.J_nu  */
/* on the existing 1000-bin grid, then ALL downstream solvers    */
/* (RADEQ T_e, plasma ionization, bf opacity, NLTE pops) are     */
/* reused unchanged.  The MC code path is never touched.         */
/* ============================================================ */
#ifndef LUMINA_CMFGEN_H
#define LUMINA_CMFGEN_H

#include "lumina.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int     n_shells;
    int     n_bins;            /* = NLTE_N_FREQ_BINS (grid shared with NLTE) */
    double  nu_min, nu_max, d_log_nu;
    double *nu;                /* [n_bins] bin-centre frequency [Hz] */
    double *dnu;               /* [n_bins] bin width nu*d_log_nu [Hz] */

    /* Per (shell,bin) opacity/source decomposition [n_shells*n_bins]. */
    double *chi_es;            /* electron scattering (coherent)   cm^-1 */
    double *chi_abs;           /* thermal true absorption bf+ff    cm^-1 */
    double *chi_line;          /* expansion line opacity (FULL)    cm^-1 */
    double *chi_line_th;       /* thermal (destruction) part of chi_line;
                                  = chi_line when the eps-split is off. The
                                  scattering remainder (chi_line-chi_line_th)
                                  is folded into chi_es so the ALI closure
                                  transports it (LUMINA_CMFGEN_LINE_EPS). */
    double *chi_tot;           /* chi_es + chi_abs + chi_line       cm^-1 */
    double *chi_line_cls;      /* A4 SRC_BLEND closure weight: per-line
                                  eps*beta/(eps+beta) accumulation — the exact
                                  two-level+Sobolev gas-coupling chi (saturated
                                  lines drop out via beta; thin lines reduce to
                                  eps). Registered for the Newton closure of
                                  non-frozen shells. */
    const double *chi_line_re; /* line opacity the RE/Newton closure sees:
                                  = chi_line (FULL) in transfer-only eps_uv
                                  mode (cooling-only closure), else chi_line_th */
    double *S_fixed;           /* (chi_abs*B + chi_line_th*S_line)/chi_tot */
    double *J;                 /* mean intensity (output) erg/s/cm^2/Hz/sr*/
    double *lambda_star;       /* diagonal approximate Lambda operator    */
    double *tri_lo;            /* [ns*nb] tridiag Lambda off-diag L[s,s-1] (A4) */
    double *tri_up;            /* [ns*nb] tridiag Lambda off-diag L[s,s+1] (A4) */
    double *tri_r;             /* [ns*nb] scattering albedo r=chi_es/chi_tot (A4) */
    double *t_color;           /* [n_shells] continuum-window 2-band Planck
                                  color temperature of the solved J (A4
                                  frozen-tail anchor); -1 where undetermined */

    /* Tangent-ray geometry. */
    int     n_rays;            /* core rays + one per shell radius        */
    double *p_ray;             /* [n_rays] impact parameters [cm]         */

    int     diag;             /* LUMINA_RADEQ_DIAG echo                   */
    double  frozen_morph_eps; /* >=0: frozen-plasma morphology pass — force
                                 the forest line-dominated bins to scatter with
                                 destruction probability eps (0 = pure coherent
                                 scatter), overriding the env eps split. -1 =
                                 off (normal assemble). Set only for the final
                                 post-convergence J re-solve, plasma untouched. */
    int     cont_only;        /* 1: continuum-only assemble — zero ALL line
                                 opacity/emissivity (chi_line, eta_line) so the
                                 solved J is the continuum-incident field J_inc
                                 (chi_es + bf/ff only). Used by the frozen
                                 morphology pass to set a NON-self-referential
                                 line source S_l=(1-eps)*J_inc+eps*B(Te), which
                                 (unlike total-Jbar) can fall below the backlight
                                 and carve a P-Cygni trough. 0 = normal. */
} CMFGENState;

/* Allocate grid + tangent rays from geometry. Returns 0 on success. */
int  cmfgen_init(CMFGENState *cs, const Geometry *geo);

/* Fill cs->t_color: per-shell color temperature of the solved J from the
 * Planck ratio of two continuum-window bands (bins with chi_line below a
 * fraction of the continuum opacity). The faithful outer-T_e anchor: gold's
 * thin-zone T_e is the field's optical color temperature, and the window
 * color of OUR deterministic J carries it even where the line-trough
 * thermostat extracts a too-cold value. */
void cmfgen_window_color(CMFGENState *cs);
void cmfgen_free(CMFGENState *cs);

/* Build chi_es/chi_abs/chi_line/chi_tot/S_fixed for the current plasma +
 * opacity + bf state. T_e is taken from plasma->T_e. */
void cmfgen_assemble(CMFGENState *cs, const Geometry *geo,
                     const OpacityState *opac, BFOpacity *bf,
                     const PlasmaState *plasma);

/* Spherical tangent-ray short-characteristics formal solve with diagonal
 * ALI scattering iteration; fills cs->J. T_inner sets the diffusive inner
 * boundary B_nu(T_inner) on core rays. */
void cmfgen_solve_J(CMFGENState *cs, const Geometry *geo, double T_inner,
                    int n_ali_iter);

/* Copy cs->J into nlte->J_nu (same [n_shells*n_bins] layout/grid). */
void cmfgen_write_jnu(const CMFGENState *cs, NLTEConfig *nlte);

/* Thick/thin-limit sanity print (J/B, J/S, radial tau) for chosen shells. */
void cmfgen_validate(const CMFGENState *cs, const Geometry *geo,
                     const PlasmaState *plasma);

/* Observer-frame emergent spectrum from the converged field via the tangent-ray
 * surface integral  L_nu = 8*pi^2 * Int_0^Rmax I+(p) p dp, where I+(p) is the
 * emergent intensity at the top of each ray's outbound leg and the source is the
 * converged S = S_fixed + (chi_es/chi_tot) J. Writes "wavelength_angstrom,flux"
 * (flux = L_lambda, erg/s/Angstrom) to `path`. Returns 0 on success. */
int  cmfgen_write_spectrum(const CMFGENState *cs, const Geometry *geo,
                           double T_inner, const char *path);

/* Observer-frame emergent spectrum (gate LUMINA_CMF_OBSERVER_FRAME=1): per-
 * nu_obs formal solve with homologous Doppler mapping nu_cmf=gamma(1-mu*beta)*
 * nu_obs along each ray. beta->0 reproduces cmfgen_write_spectrum. */
int  cmfgen_write_spectrum_obs(const CMFGENState *cs, const Geometry *geo,
                               double T_inner, const char *path);

/* Top-level env-gated driver: replaces the MC iteration loop. Iterates
 * assemble -> formal-solve -> write J_nu -> RADEQ T_e -> plasma -> bf ->
 * NLTE for n_iter passes. Returns 0 on success. */
int  cmfgen_run(Geometry *geo, OpacityState *opac, BFOpacity *bf,
                PlasmaState *plasma, NLTEConfig *nlte, AtomicData *atom,
                GammaDeposition *gamma, double T_inner, int n_iter);

#ifdef __cplusplus
}
#endif

#endif /* LUMINA_CMFGEN_H */
