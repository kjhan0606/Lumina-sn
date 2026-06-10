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
    double *chi_line;          /* expansion line opacity           cm^-1 */
    double *chi_tot;           /* chi_es + chi_abs + chi_line       cm^-1 */
    double *S_fixed;           /* (chi_abs*B + chi_line*S_line)/chi_tot   */
    double *J;                 /* mean intensity (output) erg/s/cm^2/Hz/sr*/
    double *lambda_star;       /* diagonal approximate Lambda operator    */

    /* Tangent-ray geometry. */
    int     n_rays;            /* core rays + one per shell radius        */
    double *p_ray;             /* [n_rays] impact parameters [cm]         */

    int     diag;             /* LUMINA_RADEQ_DIAG echo                   */
} CMFGENState;

/* Allocate grid + tangent rays from geometry. Returns 0 on success. */
int  cmfgen_init(CMFGENState *cs, const Geometry *geo);
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
