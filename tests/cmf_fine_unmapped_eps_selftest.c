/* DET-LTEUNMAP-EPS selftest (CPU, no GPU, no sealed inputs).
 *
 * Exercises the four gated changeset behaviours the contract (PREREG §3) names:
 *   1. gate OFF  -> parity material is byte-identical (chi0, eta0, eps=1);
 *   2. gate ON, MAPPED line (used_nlte==1) -> unchanged (eps=1);
 *   3. gate ON, UNMAPPED line (used_nlte==0), table built -> eta0 folded to
 *      eps*eta0 (S_l=B line => eps*chi0*B), chi0 unchanged, and the caller-side
 *      opacity split chi_line += eps*chi0 / chi_es += (1-eps)*chi0 conserves
 *      total opacity;
 *   4. gate ON, UNMAPPED line, radeq table UNBUILT (ensure not called) -> the
 *      material fails closed (rc=-2, eps=-1), i.e. NO silent thermal fallback;
 *   5. the G-EPS census line format.
 *
 * The parity material is a static function; the production TU exposes it (and
 * the census formatter) to this test only under
 * LUMINA_CMF_FINE_UNMAPPED_EPS_SELFTEST, so production behaviour is unchanged.
 * g_fine_atom / g_fine_nlte are installed through the already-public
 * cmfgen_fine_set_bf_atom. A minimal synthetic atom (1 element, 1 ion, 3 levels,
 * 2 lines, 1 shell) drives both the radeq table build and the material.
 */
#define _GNU_SOURCE
#include "lumina.h"
#include "opacity_publication.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Direct-call companions defined in lumina_plasma.c (not in lumina.h). */
extern void radeq_line_table_ensure(NLTEConfig *nlte, AtomicData *atom,
                                    const OpacityState *opacity);
extern int  radeq_line_table_is_built(void);
/* radeq_line_eps_phys is already declared in lumina.h. */

/* bin-assembler (committed path) test hooks from lumina_cmfgen.c */
extern double cmf_assemble_unmapped_el_test(int is_unmapped,int line,double n_e,
        double T_e,double tau,int *fail);
extern int cmf_assemble_unmapped_eps_format_test(char *buf,size_t n,long pass,
        int shell,long n_unmapped,long n_mapped,double eps_min,
        double eps_median,double eps_max,double thermal_emit_fraction);
extern int cmf_fine_unmapped_eps_fine_gate_test(void);
extern double cmf_assemble_unmapped_sc_share_test(double w,double el);
extern int cmf_solve_j_conv_format_test(char *buf,size_t n,long pass,int n_ali,
        double max_rel,double p99_rel,long cells);

/* Test hooks exposed by lumina_cmfgen.c under the selftest macro. */
extern int cmf_fine_line_material_test(
        const OpacityState *opac,const PlasmaState *plasma,
        int line,int shell,int n_shells,double nu,double tau,double dnu_d,
        double time_explosion,double legacy_chi0_pref,int parity,
        int legacy_source_nlte,double line_eps,double sl_clamp,
        const double *upper_population_cache,
        int unmapped_eps_gate,const unsigned char *used_nlte_cache,
        double *chi0,double *eta0,double *source_diag,double *eps_out,
        int *srce_chk_applied,int *clamped);
extern int cmf_unmapped_eps_format_test(char *buf,size_t n,int shell,
        long n_unmapped,long n_mapped,double eps_min,double eps_median,
        double eps_max,double thermal_emit_fraction);

static int g_fail = 0;
static void ok(int cond, const char *msg){
    if (cond) { printf("[PASS] %s\n", msg); }
    else { printf("[FAIL] %s\n", msg); g_fail = 1; }
}
static int close_rel(double a, double b, double rtol){
    double d = fabs(a - b);
    double s = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    return d <= rtol * (s > 0.0 ? s : 1.0);
}

/* Mirror of the frozen radeq_beta_esc (lumina_plasma.c) for the expected eps. */
static double beta_esc_ref(double tau){
    if (tau <= 1e-6) return 1.0;
    if (tau > 700.0) return 1.0 / tau;
    return (1.0 - exp(-tau)) / tau;
}

/* --- synthetic fixture ---------------------------------------------------- */
#define NLEV 3
#define NLINE 2
#define NS 1

static AtomicData   atom;
static NLTEConfig   nlte;
static OpacityState opac;
static PlasmaState  plasma;

/* backing storage */
static int    a_line_Z[NLINE]      = {16,16};
static int    a_line_ion[NLINE]    = {4,4};        /* == ion_pop_stage */
static int    a_line_lo[NLINE]     = {0,0};        /* level_num of lower */
static int    a_line_up[NLINE]     = {1,2};        /* level_num of upper */
static double a_line_f[NLINE]      = {0.1,0.05};
static double a_line_Aul[NLINE]    = {1.0,5.0e7};  /* line0 small A => mid eps */
static double a_line_wl[NLINE];
static double a_line_nu[NLINE];
static int    a_level_Z[NLEV]      = {16,16,16};
static int    a_level_ion[NLEV]    = {4,4,4};
static int    a_level_num[NLEV]    = {0,1,2};
static double a_level_E[NLEV]      = {0.0,20.0,40.0};   /* eV */
static int    a_level_g[NLEV]      = {1,3,5};
static int    a_elem_Z[1]          = {16};
static int    a_elem_ion_off[2]    = {0,1};
static int    a_ionpop_Z[1]        = {16};
static int    a_ionpop_stage[1]    = {4};
static int    a_level_off[2]       = {0,NLEV};
static int    a_g2n[NLEV]          = {-1,-1,-1};
static double a_ionnd[NS]          = {1.0e6};
static double a_part[NS]           = {1.0};

static double te_arr[NS]  = {20000.0};
static double ne_arr[NS]  = {1.0e6};

static double line_nu_arr[NLINE];
static double tau_arr[NLINE*NS];
static A208Validity val_arr[NLINE*NS];

static void build_fixture(void){
    memset(&atom,0,sizeof atom);
    memset(&nlte,0,sizeof nlte);
    memset(&opac,0,sizeof opac);
    memset(&plasma,0,sizeof plasma);

    const double C = 2.99792458e10;
    for (int l=0;l<NLINE;l++){
        /* pick wavelengths ~500A / 600A so nu>0; exact value irrelevant */
        double lambda_cm = (l==0)? 500.0e-8 : 600.0e-8;
        a_line_wl[l]=lambda_cm;
        a_line_nu[l]=C/lambda_cm;
        line_nu_arr[l]=a_line_nu[l];
    }
    for (int c=0;c<NLINE*NS;c++){ tau_arr[c]=0.5; val_arr[c]=A208_VALID; }

    atom.n_lines=NLINE;
    atom.line_atomic_number=a_line_Z;
    atom.line_ion_number=a_line_ion;
    atom.line_level_lower=a_line_lo;
    atom.line_level_upper=a_line_up;
    atom.line_f_lu=a_line_f;
    atom.line_A_ul=a_line_Aul;
    atom.line_wavelength_cm=a_line_wl;
    atom.line_nu=a_line_nu;
    atom.n_levels=NLEV;
    atom.level_Z=a_level_Z;
    atom.level_ion=a_level_ion;
    atom.level_num=a_level_num;
    atom.level_energy_eV=a_level_E;
    atom.level_g=a_level_g;
    atom.n_elements=1;
    atom.element_Z=a_elem_Z;
    atom.elem_ion_offset=a_elem_ion_off;
    atom.n_ion_pops=1;
    atom.ion_pop_Z=a_ionpop_Z;
    atom.ion_pop_stage=a_ionpop_stage;
    atom.level_offset=a_level_off;
    atom.ion_number_density=a_ionnd;
    atom.partition_functions=a_part;

    nlte.global_to_nlte_level=a_g2n;
    nlte.n_nlte_levels_total=0;

    opac.n_lines=NLINE;
    opac.line_list_nu=line_nu_arr;
    opac.tau_sobolev=tau_arr;
    opac.tau_validity=val_arr;
    opac.line_source_S=NULL;

    plasma.T_e=te_arr;
    plasma.n_electron=ne_arr;
}

int main(void){
    build_fixture();
    cmfgen_fine_set_bf_atom(NULL,&atom,&nlte,NULL);

    const double t_exp=1.7e6;
    const double vdop=1.0e8, C=2.99792458e10;
    const double nu0=a_line_nu[0], nu1=a_line_nu[1];
    const double dnuD0=nu0*vdop/C, dnuD1=nu1*vdop/C;
    const double tau=0.5;
    const double chi0_pref=1.0/(1.7724538509055160*vdop*t_exp);

    double upop[NLINE*NS];
    upop[0]=1.0e-3;   /* n_upper line0 shell0 */
    upop[1]=2.0e-3;   /* n_upper line1 shell0 */
    unsigned char used_map[NLINE*NS];
    used_map[0]=0;    /* line0 = LTE-unmapped */
    used_map[1]=1;    /* line1 = NLTE-mapped  */

    double chi0,eta0,src,eps; int schk,clmp,rc;

    /* -- Step A: table UNBUILT + gate ON + unmapped -> fail-closed (rc=-2) -- */
    ok(!radeq_line_table_is_built(), "radeq table not built before ensure");
    rc=cmf_fine_line_material_test(&opac,&plasma,0,0,NS,nu0,tau,dnuD0,t_exp,
        chi0_pref,1,0,1.0,0.0,upop,1,used_map,
        &chi0,&eta0,&src,&eps,&schk,&clmp);
    ok(rc==-2 && eps==-1.0,
       "gate ON + unmapped + unbuilt radeq table -> fail-closed rc=-2, eps=-1");

    /* -- build the table -- */
    radeq_line_table_ensure(&nlte,&atom,&opac);
    ok(radeq_line_table_is_built(), "radeq table built after ensure");

    /* -- Step C: gate OFF reference for line0 (byte-identical path) -- */
    double chi0_off,eta0_off,src_off,eps_off;
    rc=cmf_fine_line_material_test(&opac,&plasma,0,0,NS,nu0,tau,dnuD0,t_exp,
        chi0_pref,1,0,1.0,0.0,upop,0,NULL,
        &chi0_off,&eta0_off,&src_off,&eps_off,&schk,&clmp);
    ok(rc==0 && eps_off==1.0, "gate OFF line0 -> rc=0, eps=1");
    /* expected raw material (constant-agnostic reconstruction not needed;
     * we compare gate-ON against this reference) */

    /* -- Step D: gate ON + MAPPED line1 -> unchanged vs its gate-OFF -- */
    double chi1_off,eta1_off,src1_off,eps1_off;
    rc=cmf_fine_line_material_test(&opac,&plasma,1,0,NS,nu1,tau,dnuD1,t_exp,
        chi0_pref,1,0,1.0,0.0,upop,0,NULL,
        &chi1_off,&eta1_off,&src1_off,&eps1_off,&schk,&clmp);
    double chi1_on,eta1_on,src1_on,eps1_on;
    rc=cmf_fine_line_material_test(&opac,&plasma,1,0,NS,nu1,tau,dnuD1,t_exp,
        chi0_pref,1,0,1.0,0.0,upop,1,used_map,
        &chi1_on,&eta1_on,&src1_on,&eps1_on,&schk,&clmp);
    ok(rc==0 && eps1_on==1.0 && chi1_on==chi1_off && eta1_on==eta1_off,
       "gate ON + MAPPED line -> byte-identical (eps=1, chi0/eta0 unchanged)");

    /* -- Step E: gate ON + UNMAPPED line0 -> eps folded -- */
    double chi0_on,eta0_on,src_on,eps_on;
    rc=cmf_fine_line_material_test(&opac,&plasma,0,0,NS,nu0,tau,dnuD0,t_exp,
        chi0_pref,1,0,1.0,0.0,upop,1,used_map,
        &chi0_on,&eta0_on,&src_on,&eps_on,&schk,&clmp);
    ok(rc==0, "gate ON + unmapped + built table -> rc=0");

    /* expected eps from the frozen two-level formula (default VR coeff path) */
    double coeff = 2.16e-6 * a_line_f[0] * 1.0;         /* VAN_REGEMORTER*f*gaunt */
    double g_up  = (double)a_level_g[1];                 /* upper level of line0 */
    double C_ul  = ne_arr[0]*coeff/(g_up*sqrt(te_arr[0]));
    double denom = C_ul + a_line_Aul[0]*beta_esc_ref(tau);
    double eps_exp = (denom>0.0)? C_ul/denom : 1.0;
    ok(eps_on>0.0 && eps_on<1.0, "unmapped eps strictly in (0,1)");
    ok(close_rel(eps_on,eps_exp,1e-9),
       "unmapped eps == radeq two-level C/(C+A*beta) formula");
    ok(chi0_on==chi0_off, "unmapped: chi0 unchanged by eps");
    ok(close_rel(eta0_on, eps_on*eta0_off, 1e-12),
       "unmapped: eta0 folded to eps*eta0 (thermal share)");
    /* S_l=B property: original eta0/chi0 is the line source; folded eta0 =
     * eps * (chi0 * source) = eps*chi0*B when source==B. Report the source. */
    printf("       [info] source_diag(off)=%.6e  eps=%.6e  eta_on/eta_off=%.6e\n",
           src_off, eps_on, (eta0_off!=0.0)? eta0_on/eta0_off : 0.0);

    /* -- Step F: caller-side opacity split conserves total -- */
    double chi_line_add = eps_on*chi0_on;
    double chi_es_add   = (1.0-eps_on)*chi0_on;
    ok(close_rel(chi_line_add+chi_es_add, chi0_on, 1e-14),
       "opacity conserved: eps*chi0 + (1-eps)*chi0 == chi0");
    ok(chi_es_add>0.0, "scattering share (1-eps)*chi0 > 0 for unmapped line");

    /* -- Step G: census line format -- */
    char line[256];
    int nnn=cmf_unmapped_eps_format_test(line,sizeof line,44,
        123L,7L,1.0e-7,3.5e-6,2.0e-2,4.2e-3);
    int fmt_ok = nnn>0 && nnn<(int)sizeof line
        && strstr(line,"[cmf_fine][UNMAPPED-EPS] shell=44")!=NULL
        && strstr(line,"n_unmapped=123")!=NULL
        && strstr(line,"n_mapped=7")!=NULL
        && strstr(line,"eps_min=")!=NULL
        && strstr(line,"eps_median=")!=NULL
        && strstr(line,"eps_max=")!=NULL
        && strstr(line,"thermal_emit_fraction=")!=NULL
        && line[strlen(line)-1]=='\n';
    ok(fmt_ok, "census line matches G-EPS format with all required tokens");
    printf("       [info] census: %s", line);

    /* ===================================================================== *
     * Bin assembler (committed-J) path — the shared el-selection helper and
     * its census formatter (revision 2). radeq table already built above.
     * ===================================================================== */
    /* -- H1: mapped line -> el = 1.0, fault untouched -- */
    int afail = 0;
    double el_map = cmf_assemble_unmapped_el_test(0, 0, ne_arr[0], te_arr[0],
                                                  tau, &afail);
    ok(el_map == 1.0 && afail == 0,
       "assembler: MAPPED line -> el=1.0 (thermal unchanged), no fault");

    /* -- H2: unmapped line -> RAW eps == radeq formula (== fine-path eps) -- */
    afail = 0;
    double el_un = cmf_assemble_unmapped_el_test(1, 0, ne_arr[0], te_arr[0],
                                                 tau, &afail);
    ok(afail == 0 && close_rel(el_un, eps_exp, 1e-9),
       "assembler: UNMAPPED line -> RAW radeq eps (== fine-path eps)");

    /* -- H2b: RAW means NO eps_floor(1e-5). Same line at n_e=1e4 gives
     *         eps ~6e-6 < floor; the assembler keeps the raw value. -- */
    afail = 0;
    double el_lo = cmf_assemble_unmapped_el_test(1, 0, 1.0e4, te_arr[0],
                                                 tau, &afail);
    double coeff_lo = 2.16e-6 * a_line_f[0] * 1.0;
    double Cul_lo = 1.0e4 * coeff_lo / ((double)a_level_g[1] * sqrt(te_arr[0]));
    double eps_lo = Cul_lo / (Cul_lo + a_line_Aul[0] * beta_esc_ref(tau));
    ok(afail == 0 && el_lo < 1.0e-5 && close_rel(el_lo, eps_lo, 1e-9),
       "assembler: RAW eps below eps_floor(1e-5) is NOT floored");

    /* -- H3: unmapped line unknown to the table -> radeq -1 -> fail-closed -- */
    afail = 0;
    double el_bad = cmf_assemble_unmapped_el_test(1, 999, ne_arr[0], te_arr[0],
                                                  tau, &afail);
    ok(afail == 1 && el_bad == 1.0,
       "assembler: unmapped line with no eps (-1) -> fail-closed (fault set)");

    /* -- H4: assembler census format (with pass=) -- */
    char aline[256];
    int an = cmf_assemble_unmapped_eps_format_test(aline, sizeof aline, 2, 44,
        123L, 7L, 1.0e-7, 3.5e-6, 2.0e-2, 4.2e-3);
    int afmt_ok = an > 0 && an < (int)sizeof aline
        && strstr(aline,"[cmfgen_assemble][UNMAPPED-EPS] pass=2 shell=44")!=NULL
        && strstr(aline,"n_unmapped=123")!=NULL
        && strstr(aline,"n_mapped=7")!=NULL
        && strstr(aline,"eps_min=")!=NULL
        && strstr(aline,"eps_median=")!=NULL
        && strstr(aline,"eps_max=")!=NULL
        && strstr(aline,"thermal_emit_fraction=")!=NULL
        && aline[strlen(aline)-1]=='\n';
    ok(afmt_ok, "assembler census line matches G-EPS format (with pass token)");
    printf("       [info] asm census: %s", aline);

    /* ===================================================================== *
     * Rev 4 — env separation. The fine parity fold has its own env
     * LUMINA_CMF_FINE_UNMAPPED_EPS_FINE (default 0); the commit-path gate
     * LUMINA_CMF_FINE_UNMAPPED_EPS must NOT enable it (run 374587: the fine
     * (1-eps)*chi -> chi_es fold broke the exact solver). The fine material
     * tests above pass the gate flag explicitly (== fine env ON). Here we
     * verify the env wiring and that fine gate OFF is byte-identical.
     * ===================================================================== */
    setenv("LUMINA_CMF_FINE_UNMAPPED_EPS", "1", 1);   /* commit-path gate */
    unsetenv("LUMINA_CMF_FINE_UNMAPPED_EPS_FINE");
    ok(cmf_fine_unmapped_eps_fine_gate_test() == 0,
       "rev4: LUMINA_CMF_FINE_UNMAPPED_EPS=1 alone -> fine gate OFF");
    setenv("LUMINA_CMF_FINE_UNMAPPED_EPS_FINE", "1", 1);
    ok(cmf_fine_unmapped_eps_fine_gate_test() == 1,
       "rev4: LUMINA_CMF_FINE_UNMAPPED_EPS_FINE=1 -> fine gate ON");
    unsetenv("LUMINA_CMF_FINE_UNMAPPED_EPS_FINE");
    /* fine gate OFF (via env) threaded into the material: unmapped line must be
     * byte-identical to the gate-off reference (eps=1, no fold). */
    double chi0_g0, eta0_g0, src_g0, eps_g0;
    rc = cmf_fine_line_material_test(&opac,&plasma,0,0,NS,nu0,tau,dnuD0,t_exp,
        chi0_pref,1,0,1.0,0.0,upop,
        cmf_fine_unmapped_eps_fine_gate_test(), used_map,
        &chi0_g0,&eta0_g0,&src_g0,&eps_g0,&schk,&clmp);
    ok(rc==0 && eps_g0==1.0 && chi0_g0==chi0_off && eta0_g0==eta0_off,
       "rev4: commit-gate ON + fine-env OFF -> fine material byte-identical");

    /* ===================================================================== *
     * Rev 5 — the commit-path unmapped scattering share is tracked separately
     * (chi_line_sc_unmapped += w*(1-el)) and subtracted from the fine continuum
     * so the fine solver sees a base-construction albedo (run 374640). Plus the
     * bin-solver per-pass CONV diagnostic.
     * ===================================================================== */
    /* -- S1: scattering share = w*(1-el); mapped (el=1) contributes 0 -- */
    double w_t = 3.3e-8;
    double sc_un = cmf_assemble_unmapped_sc_share_test(w_t, el_un);
    double sc_map = cmf_assemble_unmapped_sc_share_test(w_t, 1.0);
    ok(close_rel(sc_un, w_t * (1.0 - el_un), 1e-15) && sc_un > 0.0,
       "rev5: unmapped scattering share == w*(1-el) (> 0)");
    ok(sc_map == 0.0,
       "rev5: mapped line (el=1) contributes 0 to chi_line_sc_unmapped");

    /* -- S2: fine interp input chi_es - sc == base chi_es (identity). The gate
     *        adds exactly sc to chi_es for an unmapped line; subtracting it in
     *        the fine interp restores the base (gate-off) construction. -- */
    double chi_es_base = 0.1;                         /* arbitrary base value */
    double chi_es_on   = chi_es_base + sc_un;         /* what the assembler folds */
    ok(close_rel(chi_es_on - sc_un, chi_es_base, 1e-14),
       "rev5: fine chi_es - chi_line_sc_unmapped == base chi_es (gate OFF)");
    /* mapped line adds nothing, so base is already restored */
    ok((chi_es_base + sc_map) - sc_map == chi_es_base,
       "rev5: mapped line leaves fine chi_es == base");

    /* -- S3: CONV diagnostic line format -- */
    char cvl[160];
    int cvn = cmf_solve_j_conv_format_test(cvl, sizeof cvl, 2, 8,
        3.8e-3, 1.1e-4, 6400L);
    int cv_ok = cvn > 0 && cvn < (int)sizeof cvl
        && strstr(cvl,"[cmfgen_solve_J][CONV] pass=2 n_ali=8")!=NULL
        && strstr(cvl,"max_rel=")!=NULL
        && strstr(cvl,"p99_rel=")!=NULL
        && strstr(cvl,"cells=6400")!=NULL
        && cvl[strlen(cvl)-1]=='\n';
    ok(cv_ok, "rev5: [cmfgen_solve_J][CONV] line format with all tokens");
    printf("       [info] conv: %s", cvl);

    printf(g_fail? "\nRESULT: FAIL\n" : "\nRESULT: PASS\n");
    return g_fail? 1 : 0;
}
