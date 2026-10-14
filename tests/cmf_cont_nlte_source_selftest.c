/* DET-CONT-NLTE-BALANCE selftest (CPU, no GPU, no sealed inputs).
 *
 * Verifies the contract (PREREG §3) at the level of the shared, single-source
 * expressions the bin assembler uses:
 *   - gate OFF: the S_fixed continuum numerator is byte-identical to the legacy
 *     Kirchhoff form (chi_bf+chi_ff)*B + eta_ln;
 *   - gate ON: the numerator is (eta_bf + chi_ff*B + eta_ln) — the bf term
 *     chi_bf*B is replaced by the real Milne eta_bf, ff stays thermal;
 *   - LTE limit (eta_bf == chi_bf*B): gate ON == gate OFF exactly;
 *   - fail-closed: bf / eta_bf absent, LUMINA_CMF_BF_MILNE<1, or a bf-opaque
 *     shell with all-zero eta_bf -> fault (no silent B fallback);
 *   - the G-SRC census line format.
 *
 * The assembler's inline S_fixed numerator, the fail-closed predicates, and the
 * census formatter are exposed as test hooks so production and this test can
 * never diverge. Pure functions => no CMFGENState / BFOpacity fixture needed.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* test hooks from lumina_cmfgen.c (under the selftest macro) */
extern double cmf_cont_src_numerator_test(int gate,double chi_bf,double chi_ff,
        double B,double eta_bf,double eta_ln);
extern int cmf_assemble_cont_src_format_test(char *buf,size_t n,long pass,
        int shell,long n_bf_bins,double s_over_b_mean,double s_over_b_min,
        double s_over_b_max);
extern int cmf_cont_src_precond_fault_test(int bf_present,int eta_bf_present,
        int bf_milne);
/* rev 1 (R1-1): the second arg is the per-shell Milne-part flag eta_milne_ok
 * (recorded BEFORE ff is summed), not an ff-inclusive "any eta positive". */
extern int cmf_cont_src_shell_fault_test(long n_bf_bins,int milne_ok);
/* rev 1 (R1-3): channel-dump CSV row/header hooks from lumina_plasma.c. */
extern int ion_channel_dump_format_test(char *buf,size_t n,long gen,int shell,
        int Z,int stage,double T_e,double n_e,double x_e,
        double gamma_phot,double gamma_nt,double c_ion,
        double ne_alpha_rr,double ne_alpha_dr,double c_rec,double r);
extern const char *ion_channel_dump_header_test(void);

static int g_fail = 0;
static void ok(int cond,const char *msg){
    if (cond) printf("[PASS] %s\n",msg);
    else { printf("[FAIL] %s\n",msg); g_fail = 1; }
}
static int close_rel(double a,double b,double rtol){
    double d=fabs(a-b), s=fabs(a)>fabs(b)?fabs(a):fabs(b);
    return d<=rtol*(s>0.0?s:1.0);
}
/* count comma-separated fields on the first line (ignores a trailing newline) */
static int count_fields(const char *s){
    int f = 1;
    for (; *s && *s != '\n'; ++s) if (*s == ',') ++f;
    return f;
}

int main(void){
    double chi_bf = 2.5e-3, chi_ff = 4.0e-4, B = 1.3e-5;
    double eta_bf = 7.1e-9, eta_ln = 3.3e-8;

    /* -- gate OFF: legacy Kirchhoff numerator, byte-identical -- */
    double num_off = cmf_cont_src_numerator_test(0, chi_bf, chi_ff, B, eta_bf,
                                                 eta_ln);
    double legacy  = (chi_bf + chi_ff) * B + eta_ln;
    ok(num_off == legacy,
       "gate OFF numerator == (chi_bf+chi_ff)*B + eta_ln (byte-identical)");

    /* -- gate ON: bf term replaced by Milne eta_bf, ff thermal -- */
    double num_on = cmf_cont_src_numerator_test(1, chi_bf, chi_ff, B, eta_bf,
                                                eta_ln);
    ok(close_rel(num_on, eta_bf + chi_ff * B + eta_ln, 1e-12),
       "gate ON numerator == eta_bf + chi_ff*B + eta_ln");
    ok(num_on != num_off,
       "gate ON numerator differs from OFF for eta_bf != chi_bf*B");

    /* -- LTE limit: eta_bf == chi_bf*B => ON == OFF exactly -- */
    double eta_lte = chi_bf * B;
    double num_on_lte  = cmf_cont_src_numerator_test(1, chi_bf, chi_ff, B,
                                                     eta_lte, eta_ln);
    double num_off_lte = cmf_cont_src_numerator_test(0, chi_bf, chi_ff, B,
                                                     eta_lte, eta_ln);
    ok(num_on_lte == num_off_lte,
       "LTE limit (eta_bf=chi_bf*B): gate ON == gate OFF exactly");

    /* -- fail-closed preconditions -- */
    ok(cmf_cont_src_precond_fault_test(0, 0, 2) == 1,
       "fail-closed: bf absent -> fault");
    ok(cmf_cont_src_precond_fault_test(1, 0, 2) == 1,
       "fail-closed: eta_bf absent -> fault");
    ok(cmf_cont_src_precond_fault_test(1, 1, 0) == 1,
       "fail-closed: LUMINA_CMF_BF_MILNE<1 -> fault");
    ok(cmf_cont_src_precond_fault_test(1, 1, 2) == 0,
       "preconditions satisfied (bf+eta_bf+MILNE>=1) -> no fault");

    /* -- fail-closed (rev 1, R1-1): Milne-part flag, not ff-inclusive eta -- */
    ok(cmf_cont_src_shell_fault_test(5, 0) == 1,
       "fail-closed: bf-opaque bins but Milne part all-zero (eta_milne_ok=0) -> fault");
    /* an ff-only shell presents bf-opaque bins (ff makes bf_get_eta>0) yet has
     * eta_milne_ok=0 -> must still fault (pure-ff thermal fallback detected). */
    ok(cmf_cont_src_shell_fault_test(5, 0) == 1,
       "fail-closed: ff-only shell (Milne part 0) -> fault (no pure-ff fallback)");
    ok(cmf_cont_src_shell_fault_test(5, 1) == 0,
       "shell with Milne part present (eta_milne_ok=1) -> no fault");
    ok(cmf_cont_src_shell_fault_test(0, 0) == 0,
       "shell with no bf-opaque bins -> no fault (no census contribution)");

    /* -- G-SRC census line format -- */
    char line[192];
    int n = cmf_assemble_cont_src_format_test(line, sizeof line, 9, 16,
        847L, 3.4e-1, 8.0e-2, 1.7e0);
    int fmt_ok = n > 0 && n < (int)sizeof line
        && strstr(line,"[cmfgen_assemble][CONT-SRC] pass=9 shell=16")!=NULL
        && strstr(line,"n_bf_bins=847")!=NULL
        && strstr(line,"S_bf_over_B_mean=")!=NULL
        && strstr(line,"S_bf_over_B_min=")!=NULL
        && strstr(line,"S_bf_over_B_max=")!=NULL
        && line[strlen(line)-1]=='\n';
    ok(fmt_ok, "G-SRC census line format with all required tokens");
    printf("       [info] census: %s", line);

    /* -- channel dump (rev 1, R1-3): header column count, row format, and the
     *    by-construction identity r*(ne_alpha_rr+ne_alpha_dr+c_rec) ==
     *    gamma_phot+gamma_nt+c_ion, with a column-swap negative control. -- */
    const char *hdr = ion_channel_dump_header_test();
    ok(count_fields(hdr) == 14,
       "channel dump header has exactly 14 columns");
    ok(strncmp(hdr, "gen,shell,Z,stage,T_e,n_e,x_e,gamma_phot,gamma_nt,c_ion,"
                    "ne_alpha_rr,ne_alpha_dr,c_rec,r", 60) == 0,
       "channel dump header column order matches contract");

    double gph=1.2e-6, gnt=3.4e-7, cio=5.6e-8;
    double nerr=2.1e-13, nedr=7.7e-14, crec=1.1e-13;
    double num = gph + gnt + cio, den = nerr + nedr + crec;
    double rr  = num / den;
    char crow[256];
    int cn = ion_channel_dump_format_test(crow, sizeof crow, 7L, 16, 16, 1,
        8.0e3, 2.0e8, 1.5, gph, gnt, cio, nerr, nedr, crec, rr);
    int crow_ok = cn > 0 && cn < (int)sizeof crow
        && count_fields(crow) == 14
        && strncmp(crow, "7,16,16,1,", 10) == 0
        && crow[strlen(crow)-1] == '\n';
    ok(crow_ok, "channel dump row: 14 fields, gen/shell/Z/stage prefix, newline");
    printf("       [info] channel: %s", crow);

    ok(close_rel(rr * den, num, 1e-9),
       "channel dump identity r*(ne_rr+ne_dr+c_rec) == gamma_phot+gamma_nt+c_ion");
    /* swap gamma_phot <-> ne_alpha_rr columns => identity must break */
    double num_sw = nerr + gnt + cio;   /* gamma_phot slot now holds ne_alpha_rr */
    double den_sw = gph  + nedr + crec; /* ne_alpha_rr slot now holds gamma_phot */
    ok(!close_rel(rr * den_sw, num_sw, 1e-9),
       "column swap (gamma_phot<->ne_alpha_rr) breaks the identity (neg control)");

    printf(g_fail? "\nRESULT: FAIL\n" : "\nRESULT: PASS\n");
    return g_fail? 1 : 0;
}
