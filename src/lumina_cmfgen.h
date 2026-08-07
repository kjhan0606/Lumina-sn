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
    double *eta_total_audit;   /* authoritative total emissivity snapshot for
                                  R7 decomposition audit; populated immediately
                                  before an armed dump, never used by transport */
    /* Stage 3.2 Rung 1 diagnostic-only snapshots.  Allocated only when the
     * path-valued gate is armed.  They observe, but are never consumed by,
     * source assembly or transport. */
    double *stage32_eta_pre_epay;          /* [ns*nb] eta_line entering EPAY */
    double *stage32_boundary_eta;          /* [ns*nb] non-window line eta in edge bins */
    double *stage32_line_eta;              /* [selected_line*ns] production eta_l */
    int *stage32_line_slot;                /* [n_lines], -1 outside 600--3000 A */
    int stage32_line_slot_n;
    int stage32_selected_lines;
    int stage32_source_nlte;
    int stage32_diag_failed;
    uint64_t stage32_field_generation;     /* independent assembly-snapshot lineage */
    uint64_t stage32_lambda_generation;    /* formal diagonal lineage; must equal field */
    unsigned char *stage32_epay_disposition; /* [ns*nb], written only at branch sites */
    unsigned char *stage32_epay_evidence;  /* [ns*nb], independent branch predicate witness */
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

/* E4 in-situ emissivity A/B audit.  The B lane keeps the production opacity,
 * continuum, binning and EPAY machinery, replacing only the pre-EPAY line
 * emissivity with h*nu*A_ul*n_u/(4*pi*dnu).  nlte_level_populations already
 * contains the production full-level reconstruction (including the existing
 * within-super-level member distribution). */
enum {
    CMF_EMISS_UNDEF_NO_NLTE_LINE = 1U,
    CMF_EMISS_UNDEF_UPPER_LEVEL  = 2U,
    CMF_EMISS_UNDEF_A_UL         = 4U,
    CMF_EMISS_UNDEF_POPULATION   = 8U
};

typedef struct {
    uint64_t active_transition_count;
    uint64_t defined_transition_count;
    uint64_t undefined_transition_count;
    uint64_t active_line_shell_count;
    uint64_t defined_line_shell_count;
    uint64_t undefined_line_shell_count;
    double   a_reference_line_power;
    double   a_reference_covered_line_power;
    double   a_reference_undefined_line_power;
    double   a_reference_contribution_fraction;
    double   a_reference_undefined_contribution_fraction;
    uint64_t retained_transition_count;
    uint64_t retained_line_shell_count;
    double   a_reference_retained_line_power;
    double   a_reference_retained_contribution_fraction;
    int      retain_undefined_a;
    int      n_shells;
    int      n_bins;
    double  *undefined_a_emissivity_by_band;  /* [n_bins], sum_s eta*dnu */
    double  *undefined_a_emissivity_by_shell; /* [n_shells], sum_b eta*dnu */
    int      n_lines;
    unsigned char *undefined_reason;       /* [n_lines], CMF_EMISS_UNDEF_* mask */
    uint32_t      *undefined_shell_count;  /* [n_lines] */
    int      seed_line;
    int      seed_shell;
    double   seed_factor;
    uint64_t seed_hits;
} CMFGENEmissABStats;

typedef struct {
    const char *lane;                /* "A-production" or "B-Aul-nu" */
    const char *common_state_sha256; /* exact shared assembly-input digest */
    const CMFGENEmissABStats *coverage;
} CMFGENChietaLaneMeta;

void cmf_obs_selftest(void);  /* confirmation-ladder T1 single-line P-Cygni test */
void cmf_fsolve_selftest(const char *mode);  /* confirmation-ladder F cmf_solve_J test */
void cmf_solve_gpu_selftest(const char *mode);  /* GPU cmf_solve_J A/B vs CPU (LUMINA_CMF_SOLVE_GPU=2) */
void cmf_nlte_selftest(const char *mode);  /* confirmation-ladder N populations/S_l test */
void cmf_plasma_selftest(const char *mode);  /* confirmation-ladder P ionization/energy test */

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

/* Assemble an E4/E5 direct-Aul lane.  retain_undefined_a=0 is the original B
 * lane (undefined direct emissivity is zero); retain_undefined_a=1 is B2 and
 * preserves the exact production A-lane contribution for undefined cells.
 * seed_line<0 disables the negative-control injection; otherwise the selected
 * local n_u is multiplied by seed_factor without mutating NLTE state. */
int cmfgen_assemble_aulnu(CMFGENState *cs, const Geometry *geo,
                          const OpacityState *opac, BFOpacity *bf,
                          const PlasmaState *plasma, const NLTEConfig *nlte,
                          const AtomicData *atom, int seed_line,
                          int seed_shell, double seed_factor,
                          int retain_undefined_a,
                          CMFGENEmissABStats *stats);
void cmfgen_emiss_ab_stats_free(CMFGENEmissABStats *stats);
int cmfgen_write_emiss_ab_undefined(const CMFGENEmissABStats *stats,
                                     const AtomicData *atom,
                                     const char *path);
int cmfgen_emiss_ab_state_sha256(const CMFGENState *cs,
                                 const Geometry *geo,
                                 const OpacityState *opac,
                                 const BFOpacity *bf,
                                 const PlasmaState *plasma,
                                 const NLTEConfig *nlte,
                                 const AtomicData *atom,
                                 char out_hex[65]);

/* Stage 1: register radioactive deposition heating [n_shells] erg/s/cm^3 so the
 * next cmfgen_assemble can inject it into S_fixed (gate LUMINA_CMF_DEP_SOURCE).
 * Pass NULL to disable. */
void cmfgen_set_deposition(const double *heating_rate, int n_shells);

/* Spherical tangent-ray short-characteristics formal solve with diagonal
 * ALI scattering iteration; fills cs->J. T_inner sets the diffusive inner
 * boundary B_nu(T_inner) on core rays. */
void cmfgen_solve_J(CMFGENState *cs, const Geometry *geo, double T_inner,
                    int n_ali_iter);

/* Wave-3.2 R7 frozen coarse-field capture.  Binary v1 is written field by
 * field in little-endian order; path.manifest.json contains the SHA-256. */
int cmfgen_dump_frozen_chieta(const CMFGENState *cs, const Geometry *geo,
                              int iter, int field_generation,
                              int post_damping, const char *path);
int cmfgen_dump_frozen_chieta_lane(const CMFGENState *cs,
                                   const Geometry *geo, int iter,
                                   int field_generation, int post_damping,
                                   const char *path,
                                   const CMFGENChietaLaneMeta *meta);

/* [CMF-LINEPOP T2] population-native per-line dump for the SAME generation as
 * the frozen chi/eta capture.  Read-only replay of the assemble line loop: it
 * writes nothing into cs/opac/plasma and records whether it reproduced
 * cs->chi_line BITWISE, so "same generation" is checkable rather than assumed.
 * Selection is mandatory (LUMINA_CMF_LINEPOP_SHELLS) and oversized selections
 * fail closed instead of truncating.  Binary LCMFLP01 v1 little-endian;
 * path.manifest.json carries the SHA-256, the round-trip result and the EPAY
 * disposition census.  Returns 0 on success. */
int cmfgen_dump_line_populations(const CMFGENState *cs, const Geometry *geo,
                                 const OpacityState *opac,
                                 const PlasmaState *plasma,
                                 const NLTEConfig *nlte, const AtomicData *atom,
                                 int iter, int field_generation,
                                 const char *path);

/* Stage 3.2 Rung 1: read-only formal-operator local-response artifact.  Primary
 * rho is (chi_es/chi_tot)*lambda_star from production arrays of one verified
 * generation.  Per-line beta/eps0/eps_prime/eps_applied are secondary only.
 * The manifest binds SHA-256 and both generation lineages and closes count and
 * energy censuses by disposition. */
int cmfgen_dump_stage32_rung1(const CMFGENState *cs, const Geometry *geo,
                              const OpacityState *opac,
                              const PlasmaState *plasma,
                              int iteration,
                              const char *path);
/* Runtime gate wrapper used by both CPU and CUDA drivers.  Unset path is a
 * strict no-op; an armed gate requires LUMINA_STAGE32_RUNG1_ITER. */
int cmfgen_stage32_rung1_maybe_dump(const CMFGENState *cs,
                                    const Geometry *geo,
                                    const OpacityState *opac,
                                    const PlasmaState *plasma,
                                    int iteration,
                                    int n_iterations);

/* Copy cs->J into nlte->J_nu (same [n_shells*n_bins] layout/grid). */
void cmfgen_write_jnu(const CMFGENState *cs, NLTEConfig *nlte);

/* A2-04 CPU pure-CMFGEN producer: publish cs->J through the same canonical
 * RadiationField commit API as MC, then refresh the temporary legacy view for
 * pre-A2-05 consumers. */
int cmfgen_commit_jnu(const CMFGENState *cs, NLTEConfig *nlte,
                      const Geometry *geo, const OpacityState *opac,
                      uint64_t generation);

/* P7/R6 PRODUCER: fine-grid line-resolved J_bar_l over a wavelength window
 * (default 1000-4000 A). Reuses cmf_solve_J on a fine Doppler-resolved mesh,
 * fills private opac->jbar_line_det[n_lines*n_shells] (sentinel -1 outside
 * window).  The canonical commit translates that sentinel to UNSAMPLED.
 * Env: LUMINA_CMF_FINE_{LAMLO,LAMHI,VDOP,PPD,ALI,DIAG}. */
void cmfgen_fine_jbar(CMFGENState *csb, const Geometry *geo,
                      OpacityState *opac, double T_inner, PlasmaState *plasma);

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
                               double T_inner, const OpacityState *opac,
                               const double *Te, const char *path);

/* Compute the lowercase SHA-256 of an existing file.  The frozen chi/eta and
 * transport-estimator writers share this implementation so their sidecars do
 * not depend on an external sha256sum executable. */
int  cmfgen_sha256_file(const char *path, char hex[65]);

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
