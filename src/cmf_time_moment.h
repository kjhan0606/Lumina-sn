#ifndef LUMINA_CMF_TIME_MOMENT_H
#define LUMINA_CMF_TIME_MOMENT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Internal transport component, not yet connected to the production driver.
 * First-order v/c homologous moment equations (Hillier & Dessart 2012, 5-6).
 * J,K=fJ: shell volume averages; H: radial faces, with F=4*pi*H.
 * Every spectral value is a bin average, ascending in frequency. Same velocity
 * and comoving-frequency grids at both epochs. No implicit remap or LTE seed.
 * Coherent isotropic scattering cancels in the zeroth moment only.
 * Signed absorption is retained. An inadmissible implicit time step is
 * reported separately; the caller must retry a smaller physical dt, not clamp.
 */
typedef struct {
    int n_shells, n_bins;
    double time_old, time_new;             /* seconds, 0<old<new */
    const double *radius_edges;             /* ns+1, current epoch; inner>=0 */
    const double *frequency_edges;          /* nb+1, Hz */
    const double *chi_total;                /* ns*nb, cm^-1 */
    const double *chi_scattering;           /* ns*nb, cm^-1 */
    const double *eta_fixed;                /* ns*nb, intensity/cm */
    const double *eddington;                /* ns*nb, actual K/J, 0<f<=1 */
    const double *old_J;                    /* ns*nb, previous physical epoch */
    const double *old_H;                    /* (ns+1)*nb, signed */
    const double *inner_H;                  /* nb, imposed physical flux */
    const double *outer_h, *outer_f;        /* nb, H/J and K/J at outer face */
} CMFTimeMomentInput;

typedef enum {
    CMF_TIME_OK=0, CMF_TIME_INVALID=1, CMF_TIME_ALLOCATION=2,
    CMF_TIME_NONFINITE=3, CMF_TIME_NEGATIVE=4, CMF_TIME_STEP_TOO_LARGE=5,
    CMF_TIME_NOT_CONVERGED=6
} CMFTimeMomentStatus;

/* Backward Euler in r^3 J/r^3 H, finite-volume radial divergence and an upwind
 * frequency derivative. No incoming radiation above the blue grid edge.
 * outer_h=0 imposes a reflecting wall (tests); otherwise an actual measured
 * outer closure is required. The production integration must qualify both
 * spectral boundaries and derive closures from the formal solution.
 * Outputs are copied only on success; old state may alias its own new array.
 */
CMFTimeMomentStatus cmf_time_moment_step(const CMFTimeMomentInput *p,
                                        double *new_J, double *new_H);

#ifdef __cplusplus
}
#endif
#endif
