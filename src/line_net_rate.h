#ifndef LUMINA_LINE_NET_RATE_H
#define LUMINA_LINE_NET_RATE_H

/* Line-resolved radiative-equilibrium energy exchange.
 *
 * The signed convention matches CMFGEN STEQ_T:
 *   q > 0: matter cooling
 *   q < 0: matter heating
 *
 * A nonzero signed result is never clamped.  A result whose sign is not
 * larger than its supplied input-uncertainty bound is returned as
 * LINE_NET_UNRESOLVED_CANCELLATION and must not be published. */

#define LINE_NET_FOUR_PI 12.56637061435917295385057353311801153679
#define LINE_NET_CMFGEN_RE_INTERNAL_TO_CGS \
    1.256637061435917295385057353311801153679e-9

typedef enum {
    LINE_NET_OK_COOLING = 1,
    LINE_NET_OK_HEATING,
    LINE_NET_EXACT_ZERO,
    LINE_NET_UNRESOLVED_CANCELLATION,
    LINE_NET_INVALID_INPUT
} LineNetStatus;

typedef struct {
    /* Integrated line components in mutually consistent per-steradian units. */
    double emission_per_sr;
    double integrated_opacity;
    double jbar;

    /* Absolute input bounds, not tunable sign tolerances. */
    double jbar_absolute_uncertainty;
    double other_net_absolute_uncertainty_per_sr;

    /* CMFGEN deck line scale.  Must be finite and strictly positive. */
    double deck_scale;

    /* True only when upstream typed provenance proves the entire cell zero. */
    int exact_zero_provenance;
} LineNetComponentInput;

typedef struct {
    double emission_per_sr;
    double absorption_per_sr;
    double net_per_sr;
    double signed_rate;
    double cooling;
    double heating;
    double absolute_uncertainty;
    double cancellation_condition;
    LineNetStatus status;
} LineNetResult;

typedef enum {
    LINE_NET_NEGATIVE_OPACITY_NONE = 0,
    LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK = 1
} LineNetNegativeOpacityPolicy;

typedef struct {
    double raw_integrated_opacity;
    double emission_per_sr;
    double effective_integrated_opacity;
    /* Optical depth consumed by the declared benchmark operator.  It is the
     * raw signed tau unless the typed CMFGEN tau<-0.5 policy is active. */
    double effective_tau;
    int srce_chk_applied;
    int exact_zero_provenance;
} LineNetSobolevMaterial;

typedef struct {
    double beta;
    double one_minus_beta_over_tau;
    double continuum_term;
    double local_emission_term;
    double jbar;
    /* Propagated continuum-input bound only.  It intentionally does not claim
     * a full rounding enclosure for beta/companion/Jbar arithmetic. */
    double jbar_absolute_uncertainty;
} LineNetSobolevRadiation;

#ifdef __cplusplus
extern "C" {
#endif

LineNetStatus line_net_rate_evaluate(const LineNetComponentInput *input,
                                     LineNetResult *result);

/* CMFGEN direct-bracket Sobolev material.  The emission is the raw integrated
 * n_u A_ul h nu/(4 pi), not the beta-weighted A2-09 escape diagnostic. */
int line_net_sobolev_material(double n_upper, double A_ul, double nu,
                              double tau, double time_explosion,
                              LineNetNegativeOpacityPolicy policy,
                              unsigned num_simultaneous_lines,
                              LineNetSobolevMaterial *material);

/* CMFGEN EXPONX and the cancellation-safe companion (1-beta)/tau.  The
 * branch points and small-tau polynomial are those in subs/exponx.f. */
int line_net_cmfgen_exponx(double tau, double *beta,
                           double *one_minus_beta_over_tau);

/* Non-overlap, homologous (sigma=0) CMFGEN Sobolev line operator:
 *
 *   Jbar = beta J_cont + (1-beta) S_line
 *        = beta J_cont + eta (c t/nu) (1-beta)/tau_eff .
 *
 * The second form remains finite at tau_eff=0 and never constructs the
 * possibly singular source function eta/chi.  The material and its signed
 * raw tau are not modified. */
int line_net_sobolev_radiation(const LineNetSobolevMaterial *material,
                               double continuum_j,
                               double continuum_j_absolute_uncertainty,
                               double nu, double time_explosion,
                               LineNetSobolevRadiation *radiation);

/* Ratio converting an integrated opacity reconstructed from the declared
 * Sobolev oscillator-strength coefficient to the opacity implied by the
 * Einstein B_lu coefficient consumed by the SE matrix.  This is an identity
 * diagnostic; it must not be used as a value repair. */
int line_net_einstein_opacity_ratio(double sobolev_coefficient,
                                    double f_lu, double wavelength_cm,
                                    double B_lu, double nu,
                                    double *ratio);

/* pi e^2/(m_e c) from the same SI-2019/CODATA constants used by the I20
 * atomic-data contract.  Diagnostic source of truth; it does not mutate tau. */
double line_net_exact_sobolev_coefficient(void);

/* CMFGEN SCL_LN super-level energy correction. */
int line_net_cmfgen_scl_ln(double lower_super_energy_eV,
                           double upper_super_energy_eV, double line_nu,
                           double atom_number_density,
                           double scl_ln_fac, double density_limit,
                           double *deck_scale);

/* Convert the CMFGEN direct-bracket witness
 *   deck_scale * ETAL_MAT * ZNET
 * from CMFGEN RE internal units to erg cm^-3 s^-1.  This conversion does not
 * reconstruct ZNET by subtracting two large terms. */
int line_net_cmfgen_internal_to_cgs(double etal_mat, double znet,
                                    double deck_scale, double *q_cgs);

const char *line_net_status_name(LineNetStatus status);

#ifdef __cplusplus
}
#endif

#endif
