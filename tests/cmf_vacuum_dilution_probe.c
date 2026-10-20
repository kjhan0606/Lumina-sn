/* 실제 coarse 수송의 구형 해석해·소스/흡수 미분 비교.
 * 봉인 생산 소스 또는 별도 수정 소스의 실제 함수를 직접 링크한다.
 * 시험은 생산 입력을 변경하지 않는다. */
#include "lumina_cmfgen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern void formal_solve_bin_test(CMFGENState *, const Geometry *, int,
                                 const double *, double, double *, double *);
extern void formal_djdchi_fs_S_test(CMFGENState *, const Geometry *, double, double *);
extern double formal_djdchi_bnu_test(double, double);

/* 독립 해석해: 균질 외피에서 경계까지의 실제 광로로 계산한다.
 * 코어 limb 양쪽을 따로 적분하므로 불연속을 수치 평활화하지 않는다. */
static double homogeneous_J(double r, double rin, double rout, double chi) {
    const int nm = 32768;
    double limb = sqrt(1 - (rin/r)*(rin/r)), result = 0;
    for (int core = 0; core < 2; core++) {
        double lo = core ? limb : 0, hi = core ? 1 : limb;
        double dm = (hi-lo)/nm;
        for (int m = 0; m < nm; m++) {
            double mu = lo + (m+.5)*dm, p2 = r*r*(1-mu*mu), z = r*mu;
            double zo = sqrt(rout*rout-p2);
            double inward = .3 * -expm1(-chi*(zo-z));
            double outward;
            if (core) {
                double zc = sqrt(fmax(0, rin*rin-p2));
                double ex = exp(-chi*(z-zc));
                outward = ex + .3*(1-ex);
            } else outward = .3 * -expm1(-chi*(zo+z));
            result += .5*dm*(inward+outward);
        }
    }
    return result;
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) return 2;
    int require_physical = argc == 4, failures = 0;
    enum { NS = 50 };
    double ri[NS], ro[NS], chi[NS] = {0}, src[NS] = {0}, J[NS], lambda[NS];
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(ri, sizeof(double), NS, f) != NS) return 3;
    fclose(f);
    f = fopen(argv[2], "rb");
    if (!f || fread(ro, sizeof(double), NS, f) != NS) return 4;
    fclose(f);
    Geometry geo = {0};
    geo.n_shells = NS; geo.r_inner = ri; geo.r_outer = ro;
    printf("{\"scope\":\"production coarse formal solver, zero opacity/emissivity, unit inner intensity\",\"cases\":[\n");
    const int cores[] = {8, 16, 64};
    for (int c = 0; c < 3; c++) {
        for (int s = 0; s < NS; s++) chi[s] = src[s] = 0;
        int nc = cores[c];
        double *p = malloc((nc + NS) * sizeof(double));
        if (!p) return 5;
        for (int k = 0; k < nc; k++) p[k] = ri[0] * (k + 0.5) / nc;
        for (int s = 0; s < NS; s++) p[nc + s] = ro[s];
        CMFGENState cs = {0};
        cs.n_shells = NS; cs.n_bins = 1; cs.n_rays = nc + NS;
        cs.p_ray = p; cs.chi_tot = chi;
        formal_solve_bin_test(&cs, &geo, 0, src, 1.0, J, lambda);
        double maxrel = 0;
        printf("%s{\"core_rays\":%d,\"shells\":[", c ? ",\n" : "", nc);
        for (int s = 0; s < NS; s++) {
            double r = 0.5 * (ri[s] + ro[s]);
            double q = ri[0] / r;
            double exact = 0.5 * q*q / (1 + sqrt(1 - q*q));
            double rel = fabs(J[s] / exact - 1);
            if (!isfinite(J[s]) || !isfinite(rel)) return 6;
            if (rel > maxrel) maxrel = rel;
            printf("%s{\"shell\":%d,\"J_over_B\":%.17g,\"analytic_W\":%.17g,\"ratio\":%.17g}",
                   s ? "," : "", s, J[s], exact, J[s] / exact);
        }
        for (int s = 0; s < NS; s++) { chi[s] = 1e-9; src[s] = 1; }
        formal_solve_bin_test(&cs, &geo, 0, src, 1.0, J, lambda);
        double opaque_error = 0;
        for (int s = 0; s < NS; s++)
            if (fabs(J[s] - 1) > opaque_error) opaque_error = fabs(J[s] - 1);
        double freq = 1e15, deriv[NS], plus[NS], minus[NS];
        cs.nu = &freq; cs.S_fixed = src;
        for (int s = 0; s < NS; s++) { chi[s] = 1e-15; src[s] = .2 + .01*s; }
        formal_djdchi_fs_S_test(&cs, &geo, 1e4, deriv);
        double bin = formal_djdchi_bnu_test(freq, 1e4), derivative_error = 0;
        for (int s = 0; s < NS; s++) {
            double step = 1e-20;
            chi[s] += step;
            formal_solve_bin_test(&cs, &geo, 0, src, bin, plus, lambda);
            chi[s] -= 2*step;
            formal_solve_bin_test(&cs, &geo, 0, src, bin, minus, lambda);
            chi[s] += step;
            double finite_difference = (plus[s] - minus[s]) / (2*step);
            double scale = fmax(fabs(finite_difference), fabs(deriv[s]));
            double err = scale > 0 ? fabs(finite_difference - deriv[s]) / scale : 0;
            if (!isfinite(err)) return 7;
            if (err > derivative_error) derivative_error = err;
        }
        double original_lambda[NS], lambda_error = 0;
        formal_solve_bin_test(&cs, &geo, 0, src, bin, J, original_lambda);
        for (int s = 0; s < NS; s++) {
            double step = 1e-5;
            src[s] += step;
            formal_solve_bin_test(&cs, &geo, 0, src, bin, plus, lambda);
            src[s] -= 2*step;
            formal_solve_bin_test(&cs, &geo, 0, src, bin, minus, lambda);
            src[s] += step;
            double finite_difference = (plus[s]-minus[s])/(2*step);
            double err = fabs(finite_difference-original_lambda[s]);
            if (!isfinite(err) || original_lambda[s] < 0 || original_lambda[s] > 1) return 10;
            if (err > lambda_error) lambda_error = err;
        }
        if (maxrel > 1e-10 || opaque_error > 1e-12 || derivative_error > 1e-4 || lambda_error > 1e-8) failures++;
        printf("],\"max_relative_error\":%.17g,\"opaque_error\":%.17g,\"derivative_relative_error\":%.17g,\"lambda_absolute_error\":%.17g}",
               maxrel, opaque_error, derivative_error, lambda_error);
        free(p);
    }
    printf("\n],\"homogeneous_shell\":[");
    enum { NC = 1024, NH = 8192 };
    double *densep = malloc((NC+NH)*sizeof(double));
    if (!densep) return 8;
    for (int k = 0; k < NC; k++) densep[k] = ri[0]*(k+.5)/NC;
    for (int k = 0; k < NH; k++) densep[NC+k] = ri[0]+(ro[NS-1]-ri[0])*(k+.5)/NH;
    CMFGENState dense = {0};
    dense.n_shells = NS; dense.n_bins = 1; dense.n_rays = NC+NH;
    dense.p_ray = densep; dense.chi_tot = chi;
    const double optical[] = {.001, 1, 100};
    for (int c = 0; c < 3; c++) {
        for (int s = 0; s < NS; s++) { chi[s] = optical[c]/ro[NS-1]; src[s] = .3; }
        formal_solve_bin_test(&dense, &geo, 0, src, 1, J, lambda);
        double maxerr = 0;
        for (int s = 0; s < NS; s++) {
            double expected = homogeneous_J(.5*(ri[s]+ro[s]), ri[0], ro[NS-1], chi[s]);
            double err = fabs(J[s]/expected-1);
            if (!isfinite(err)) return 9;
            if (err > maxerr) maxerr = err;
        }
        printf("%s{\"chi_Router\":%.17g,\"max_relative_error\":%.17g}", c ? "," : "", optical[c], maxerr);
        if (maxerr > 5e-4) failures++;
    }
    free(densep);
    puts("]}");
    return require_physical && failures ? 1 : 0;
}
