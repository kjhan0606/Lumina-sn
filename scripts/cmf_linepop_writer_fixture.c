/* [CMF-LINEPOP T2] offline writer fixture — no model, no GPU, no plasma solve.
 *
 * Builds the smallest state that exercises every branch of
 * cmfgen_dump_line_populations: two shells, four coarse bins, three lines of
 * which one sits OUTSIDE the selection window (so it must contribute to the
 * chi_line round trip but must NOT produce a row), one line with fully defined
 * NLTE populations (tau round trip live) and one without.
 *
 * Seeds (negative controls):
 *   LP_SEED_CHI_DRIFT=1  perturb cs.chi_line by one ulp  -> round trip must
 *                        report chi_line_roundtrip_bitwise=false
 *   LP_SEED_MAXROWS=N    force the row cap  -> writer must fail closed
 */
#include "lumina_cmfgen.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* This standalone fixture links only lumina_cmfgen.c (see the Makefile
 * target); the production per-line destruction table lives in lumina_plasma.c.
 * eps_phys is OFF in the fixture, so the value is never consumed — the stub
 * exists to satisfy the linker and returns the same "table not built" sentinel
 * the production routine returns before its table exists. */
double radeq_line_eps_phys(int line, double n_e, double T_e, double tau) {
    (void)line; (void)n_e; (void)T_e; (void)tau;
    return -1.0;
}

#define NS 2
#define NB 4
#define NL 3

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s OUTPUT\n", argv[0]);
        return 2;
    }
    double r_inner[NS] = {1.0e14, 2.0e14};
    double r_outer[NS] = {2.0e14, 4.0e14};
    double nu_min = 1.0e14, nu_max = 1.0e16;
    double d_log_nu = log(nu_max / nu_min) / (double)NB;
    double nu[NB], dnu[NB];
    for (int b = 0; b < NB; b++) {
        nu[b] = nu_min * exp(d_log_nu * ((double)b + 0.5));
        dnu[b] = nu[b] * d_log_nu;
    }
    /* two lines inside 600-3000 A, one at ~5000 A outside it */
    double line_nu[NL] = {3.0e15, 2.0e15, 6.0e14};
    double lam_cm[NL];
    for (int l = 0; l < NL; l++) lam_cm[l] = 2.99792458e10 / line_nu[l];
    double tau[NL * NS] = {2.0, 0.5, 1.0e-3, 3.0, 4.0, 0.25};
    double src_S[NL * NS] = {0.0, 0.0, 1.5e-6, 0.0, 0.0, 0.0};
    double ne[NS] = {1.0e9, 5.0e8};
    double Te[NS] = {12000.0, 9000.0};
    double Tr[NS] = {10470.09324, 10470.09324};

    double chi_line[NS * NB], chi_line_th[NS * NB];
    double chi_abs[NS * NB], chi_tot[NS * NB];
    memset(chi_line, 0, sizeof(chi_line));
    memset(chi_line_th, 0, sizeof(chi_line_th));
    for (int q = 0; q < NS * NB; q++) { chi_abs[q] = 1.0e-12; chi_tot[q] = 2.0e-12; }

    /* Reproduce the assemble accumulation EXACTLY (same order, same
     * expressions) so the writer's round-trip check has a true reference. */
    double inv_ct = 1.0 / (2.99792458e10 * 1683072.0);
    for (int s = 0; s < NS; s++)
        for (int l = 0; l < NL; l++) {
            double t = tau[(size_t)l * NS + s];
            if (t <= 1e-12) continue;
            if (line_nu[l] <= nu_min || line_nu[l] >= nu_max) continue;
            int b = (int)floor(log(line_nu[l] / nu_min) / d_log_nu);
            if (b < 0 || b >= NB) continue;
            double frac = (t > 1e-6) ? -expm1(-t) : t;
            chi_line[(size_t)s * NB + b] += frac * line_nu[l] * inv_ct / dnu[b];
        }
    if (getenv("LP_SEED_CHI_DRIFT"))
        chi_line[0 * NB + 2] = nextafter(chi_line[0 * NB + 2], 1.0e300);

    /* one ion population (Fe III), three levels */
    int level_num[3] = {0, 1, 2};
    int level_g[3] = {9, 7, 5};
    double level_E[3] = {0.0, 3.0, 8.3};
    int level_offset[2] = {0, 3};
    int ion_pop_Z[1] = {26}, ion_pop_stage[1] = {2};
    int line_lower[NL] = {0, 0, 1};
    int line_upper[NL] = {2, 1, 2};
    int line_Z[NL] = {26, 26, 26};
    int line_ion[NL] = {2, 2, 2};
    double line_f[NL] = {0.4, 0.2, 0.05};
    double line_A[NL] = {5.0e8, 2.0e8, 1.0e7};
    int g2n[3] = {0, 1, 2};
    int line_map[NL] = {0, 0, -1};   /* line 2 is not NLTE-mapped */
    double pops[3 * NS] = {1.0e6, 5.0e5, 2.0e4, 9.0e3, 3.0e2, 1.0e2};

    Geometry geo; CMFGENState cs; OpacityState opac; PlasmaState plasma;
    NLTEConfig nlte; AtomicData atom;
    memset(&geo,0,sizeof(geo));       memset(&cs,0,sizeof(cs));
    memset(&opac,0,sizeof(opac));     memset(&plasma,0,sizeof(plasma));
    memset(&nlte,0,sizeof(nlte));     memset(&atom,0,sizeof(atom));

    geo.n_shells=NS; geo.r_inner=r_inner; geo.r_outer=r_outer;
    geo.time_explosion=1683072.0;
    cs.n_shells=NS; cs.n_bins=NB; cs.nu=nu; cs.dnu=dnu;
    cs.nu_min=nu_min; cs.nu_max=nu_max; cs.d_log_nu=d_log_nu;
    cs.chi_line=chi_line; cs.chi_line_th=chi_line_th;
    cs.chi_abs=chi_abs; cs.chi_tot=chi_tot;
    cs.frozen_morph_eps=-1.0; cs.cont_only=0;
    opac.n_lines=NL; opac.n_shells=NS; opac.line_list_nu=line_nu;
    opac.tau_sobolev=tau; opac.line_source_S=src_S; opac.electron_density=ne;
    plasma.T_e=Te; plasma.n_electron=ne;
    nlte.global_to_nlte_level=g2n; nlte.nlte_level_populations=pops;
    nlte.nlte_line_map=line_map; nlte.n_nlte_levels_total=3;
    atom.n_lines=NL; atom.n_levels=3; atom.n_ion_pops=1;
    atom.line_atomic_number=line_Z; atom.line_ion_number=line_ion;
    atom.line_level_lower=line_lower; atom.line_level_upper=line_upper;
    atom.line_f_lu=line_f; atom.line_A_ul=line_A;
    atom.line_wavelength_cm=lam_cm; atom.line_nu=line_nu;
    atom.level_num=level_num; atom.level_g=level_g;
    atom.level_energy_eV=level_E; atom.level_offset=level_offset;
    atom.ion_pop_Z=ion_pop_Z; atom.ion_pop_stage=ion_pop_stage;

    const char *it_env = getenv("LP_FIXTURE_ITER");
    int iter = it_env ? atoi(it_env) : 10;
    return cmfgen_dump_line_populations(&cs,&geo,&opac,&plasma,&nlte,&atom,
                                        iter,iter,argv[1]) == 0 ? 0 : 1;
}
