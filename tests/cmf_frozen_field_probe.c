/* 동일 O 포획 계수의 고정 방출 형식해 1회. 자기일관 해/스펙트럼 아님. */
#include "lumina_cmfgen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_array(const char *root, const char *name, double *a, size_t n) {
    char path[2048]; snprintf(path, sizeof path, "%s/%s", root, name);
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    size_t got = fread(a, sizeof(double), n, f); fclose(f);
    return got != n;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) return 2; /* capture, cells, output J, optional scatter */
    int refined = argc == 5 && !strcmp(argv[4], "scatter-refined");
    int close_scattering = argc == 5 && (!strcmp(argv[4], "scatter") || refined);
    if (argc == 5 && !close_scattering) return 2;
    enum { NS = 50, NB = 1234, MAXR = 50+32+512 };
    size_t cells = (size_t)NS * NB;
    double ri[NS], ro[NS], p[MAXR], nu[NB], dnu[NB];
    double *chi = calloc(cells, sizeof(double)), *zero = calloc(cells, sizeof(double));
    double *src = calloc(cells, sizeof(double)), *J = calloc(cells, sizeof(double));
    double *old = calloc(cells, sizeof(double)), *expected = calloc(cells, sizeof(double));
    double *lambda = calloc(cells, sizeof(double));
    double *scatter = calloc(cells, sizeof(double)), *solution = calloc(cells, sizeof(double));
    double *lo = calloc(cells, sizeof(double)), *up = calloc(cells, sizeof(double)), *r = calloc(cells, sizeof(double));
    if (!chi || !zero || !src || !J || !old || !expected || !lambda || !scatter || !solution || !lo || !up || !r) return 3;
    if (read_array(argv[1], "r_inner.f64", ri, NS) || read_array(argv[1], "r_outer.f64", ro, NS) ||
        read_array(argv[1], "nu.f64", nu, NB) || read_array(argv[1], "dnu.f64", dnu, NB)) return 4;
    char path[2048], line[4096];
    snprintf(path, sizeof path, "%s/meta.txt", argv[1]);
    FILE *f = fopen(path, "r"); if (!f) return 5;
    double T = NAN, scale = NAN;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "T_inner=", 8)) T = strtod(line + 8, NULL);
        if (!strncmp(line, "inner_bb_scale=", 15)) scale = strtod(line + 15, NULL);
    }
    fclose(f);
    if (!(T > 0) || scale != 1.0) return 6;
    f = fopen(argv[2], "r"); if (!f) return 7;
    if (!fgets(line, sizeof line, f)) return 8;
    size_t count = 0;
    while (fgets(line, sizeof line, f)) {
        int s, b; double j0, jf, ct, cs, ef, et, q;
        if (sscanf(line, "%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &s, &b, &j0, &jf, &ct, &cs, &ef, &et, &q) != 9) return 9;
        if (s != (int)(count / NB) || b != (int)(count % NB) || count >= cells || !(ct > 0)) return 10;
        old[count] = J[count] = j0; expected[count] = jf;
        chi[count] = ct; scatter[count] = cs;
        src[count] = (close_scattering ? ef : et) / ct; count++;
    }
    fclose(f); if (count != cells) return 11;
    int nc = refined ? 32 : 8, nr = nc+NS;
    for (int k = 0; k < nc; k++) p[k] = ri[0] * (k + .5) / nc;
    for (int s = 0; s < NS; s++) p[nc+s] = ro[s];
    if (refined) for (int k = 0; k < 512; k++)
        p[nr++] = ri[0]+(ro[NS-1]-ri[0])*(k+.5)/512;
    Geometry geo = {0}; geo.n_shells = NS; geo.r_inner = ri; geo.r_outer = ro;
    CMFGENState cs = {0};
    cs.n_shells = NS; cs.n_bins = NB; cs.n_rays = nr; cs.p_ray = p;
    cs.nu = nu; cs.chi_tot = chi; cs.chi_es = close_scattering ? scatter : zero;
    cs.S_fixed = src; cs.J = J; cs.lambda_star = lambda;
    cs.tri_lo = lo; cs.tri_up = up; cs.tri_r = r;
    if (close_scattering) {
        setenv("LUMINA_CMFGEN_LAMBDA_TRI", "1", 1);
        setenv("LUMINA_CMFGEN_ALI_TOL", "1e-10", 1);
        setenv("LUMINA_CMFGEN_ALI_MINIT", "2", 1);
    }
    cmfgen_solve_J(&cs, &geo, T, close_scattering ? 256 : 1);
    double closure_residual = 0;
    if (close_scattering) {
        memcpy(solution, J, cells*sizeof(double));
        for (size_t i = 0; i < cells; i++) src[i] += scatter[i]/chi[i]*J[i];
        cs.chi_es = zero;
        cmfgen_solve_J(&cs, &geo, T, 1); /* independent unaccelerated fixed-point residual */
        for (size_t i = 0; i < cells; i++) {
            double e = J[i]+solution[i] > 0 ? fabs(J[i]-solution[i])/(J[i]+solution[i]) : 0;
            if (!isfinite(e)) return 15;
            if (e > closure_residual) closure_residual = e;
        }
        memcpy(J, solution, cells*sizeof(double));
    }
    double referr = 0, integrated_delta = 0;
    printf("{\"scope\":\"%s\",\"input_rays\":%d,\"scattering_closure_residual\":%.17g,\"shells\":[",
           close_scattering ? "fixed material, scattering closed; not population/thermal closure or emergent spectrum" :
           "one formal application, frozen total emissivity, not population/thermal/scattering closure", nr, closure_residual);
    for (int s = 0; s < NS; s++) {
        double jm0 = 0, jm1 = 0, delta = 0;
        for (int b = 0; b < NB; b++) {
            size_t i = (size_t)s*NB+b;
            if (!(J[i] >= 0) || !isfinite(J[i])) return 12;
            double e = (J[i] + expected[i]) > 0 ? fabs(J[i] - expected[i]) / (J[i] + expected[i]) : 0;
            if (e > referr) referr = e;
            jm0 += old[i] * dnu[b]; jm1 += J[i] * dnu[b];
            delta += chi[i] * (J[i] - old[i]) * dnu[b];
        }
        double volume = 4*M_PI/3*(ro[s]*ro[s]*ro[s] - ri[s]*ri[s]*ri[s]);
        delta *= 4*M_PI*volume; integrated_delta += delta;
        printf("%s{\"shell\":%d,\"integrated_J_ratio\":%.17g,\"chi_weighted_delta_J_power\":%.17g}",
               s ? "," : "", s, jm1/jm0, delta);
    }
    printf("],\"max_symmetric_difference_from_old_formal\":%.17g,\"chi_weighted_delta_J_power\":%.17g}\n", referr, integrated_delta);
    f = fopen(argv[3], "wbx"); if (!f) return 13;
    if (fwrite(J, sizeof(double), cells, f) != cells) return 14;
    fclose(f);
    free(chi); free(zero); free(src); free(J); free(old); free(expected); free(lambda);
    free(scatter); free(solution); free(lo); free(up); free(r);
    return close_scattering && closure_residual > 1e-6 ? 16 : 0;
}
