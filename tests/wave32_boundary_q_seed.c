#include <math.h>
#include <float.h>
#include <stdio.h>

int nlte_ew_test_q_projection(const double *, int, double *);
double nlte_ew_test_assembly_residual(const double *, const double *, int);
int nlte_ew_test_negative_audit(const double *, int, double, double *, double *);
double nlte_ew_test_boundary_row_audit(
    const double *, const double *, const double *, int, double);
double nlte_ew_test_boundary_flux_audit(
    const double *, const double *, int, int, int, int,
    const double *, const double *, const double *, const double *);
double nlte_ew_test_boundary_population_fraction(int, double, double);
int nlte_ew_test_boundary_tau_add(double, int, double *, double *);

#define A(A_,n_,i_,j_) ((A_)[(j_)*(n_)+(i_)])

int main(void) {
    double good[4] = {0.1, 0.2, 0.3, 0.4};
    double negative[4] = {0.1, -0.2, 0.3, 0.8};
    double nonfinite[4] = {0.1, 0.2, NAN, 0.7};
    double bad_sum[4] = {0.1, 0.2, 0.3, 0.3};
    double sum = 0.0;
    int pass_good = nlte_ew_test_q_projection(good, 4, &sum);
    int pass_negative = nlte_ew_test_q_projection(negative, 4, NULL);
    int pass_nonfinite = nlte_ew_test_q_projection(nonfinite, 4, NULL);
    int pass_bad_sum = nlte_ew_test_q_projection(bad_sum, 4, NULL);
    double plane[4] = {-4.0, 4.0, 0.0, 0.0};
    double ledger[2] = {4.0, 0.0};
    double matrix_good = nlte_ew_test_assembly_residual(plane, ledger, 2);
    plane[0] = -3.0;
    double bad_debit = nlte_ew_test_assembly_residual(plane, ledger, 2);
    plane[0] = -4.0;
    plane[1] = 3.0;
    double bad_target = nlte_ew_test_assembly_residual(plane, ledger, 2);

    double x[3] = {2.0, 3.0, 5.0};
    double Anorm[9] = {0.0}, b[3] = {10.0, 0.0, 0.0};
    for (int j = 0; j < 3; j++) A(Anorm,3,0,j) = 1.0;
    double row_good = nlte_ew_test_boundary_row_audit(Anorm,b,x,3,10.0);
    A(Anorm,3,0,1) = 0.5;
    double row_seeded = nlte_ew_test_boundary_row_audit(Anorm,b,x,3,10.0);
    A(Anorm,3,0,1) = 1.0;
    b[0] = NAN;
    double row_b0_nan = nlte_ew_test_boundary_row_audit(Anorm,b,x,3,10.0);
    int row_b0_nan_gate_pass = row_b0_nan <= 1e-12;
    b[0] = 10.0;

    double Araw[9] = {0.0};
    double forward0[3] = {4.0,6.0,0.0}, forward1[3] = {0.0};
    double reverse0[3] = {2.0,3.0,0.0}, reverse1[3] = {0.0};
    A(Araw,3,2,0)=4.0; A(Araw,3,2,1)=6.0;
    A(Araw,3,0,2)=2.0; A(Araw,3,1,2)=3.0;
    double flux_good = nlte_ew_test_boundary_flux_audit(
        Araw,x,3,2,0,2,forward0,forward1,reverse0,reverse1);
    A(Araw,3,2,1)=0.0;
    double route_seeded = nlte_ew_test_boundary_flux_audit(
        Araw,x,3,2,0,2,forward0,forward1,reverse0,reverse1);
    A(Araw,3,2,1)=6.0;
    A(Araw,3,0,2)=1.5; A(Araw,3,1,2)=3.5; /* total unchanged */
    double q_coupling_seeded = nlte_ew_test_boundary_flux_audit(
        Araw,x,3,2,0,2,forward0,forward1,reverse0,reverse1);
    A(Araw,3,0,2)=2.0; A(Araw,3,1,2)=3.0;
    forward0[0] = NAN;
    double flux_nan = nlte_ew_test_boundary_flux_audit(
        Araw,x,3,2,0,2,forward0,forward1,reverse0,reverse1);
    forward0[0] = 4.0;

    plane[1] = NAN;
    double matrix_nan = nlte_ew_test_assembly_residual(plane,ledger,2);

    double n_elem_finite_fraction =
        nlte_ew_test_boundary_population_fraction(1, 4.0, 10.0);
    double n_elem_nan_fraction =
        nlte_ew_test_boundary_population_fraction(1, 4.0, NAN);
    int n_elem_nan_gate_pass = n_elem_nan_fraction <= 1e-8;
    double tau_all = 0.0, tau_boundary = 0.0;
    int tau_first_rc = nlte_ew_test_boundary_tau_add(
        DBL_MAX, 0, &tau_all, &tau_boundary);
    int tau_overflow_rc = nlte_ew_test_boundary_tau_add(
        DBL_MAX, 0, &tau_all, &tau_boundary);
    double tau_overflow_fraction =
        tau_all > 0.0 ? tau_boundary / tau_all : 0.0;
    if (!isfinite(tau_overflow_fraction)) tau_overflow_fraction = INFINITY;
    int tau_overflow_gate_pass = tau_overflow_fraction <= 1e-4;
    double tau_normal_all = 0.0, tau_normal_boundary = 0.0;
    int tau_normal_rc0 = nlte_ew_test_boundary_tau_add(
        1.0, 0, &tau_normal_all, &tau_normal_boundary);
    int tau_normal_rc1 = nlte_ew_test_boundary_tau_add(
        2.0, 1, &tau_normal_all, &tau_normal_boundary);

    double roundoff_x[3] = {1.0,-DBL_EPSILON,2.0};
    double negative_x[3] = {1.0,-1e-12,2.0};
    double raw_min=0.0,error_bound=0.0;
    int roundoff_negative = nlte_ew_test_negative_audit(
        roundoff_x,3,1.0,&raw_min,&error_bound);
    int seeded_negative = nlte_ew_test_negative_audit(
        negative_x,3,1.0,NULL,NULL);
    printf("good=%d sum=%.17g negative=%d nonfinite=%d bad_sum=%d\n",
           pass_good, sum, pass_negative, pass_nonfinite, pass_bad_sum);
    printf("matrix_good=%.17g bad_debit=%.17g bad_target=%.17g\n",
           matrix_good, bad_debit, bad_target);
    printf("row_good=%.17g conservation_row_seeded=%.17g\n",
           row_good,row_seeded);
    printf("b0_nan_residual=%g b0_nan_gate_pass=%d\n",
           row_b0_nan,row_b0_nan_gate_pass);
    printf("flux_good=%.17g boundary_route_seeded=%.17g "
           "q_coupling_seeded=%.17g\n",
           flux_good,route_seeded,q_coupling_seeded);
    printf("matrix_nan=%g flux_nan=%g\n",matrix_nan,flux_nan);
    printf("n_elem_finite_fraction=%.17g n_elem_nan_fraction=%g "
           "gate_pass=%d\n",n_elem_finite_fraction,
           n_elem_nan_fraction,n_elem_nan_gate_pass);
    printf("tau_first_rc=%d tau_overflow_rc=%d tau_all=%g "
           "tau_boundary=%g opacity_fraction=%g gate_pass=%d\n",
           tau_first_rc,tau_overflow_rc,tau_all,tau_boundary,
           tau_overflow_fraction,tau_overflow_gate_pass);
    printf("tau_normal_rc=%d/%d tau_all=%.17g tau_boundary=%.17g\n",
           tau_normal_rc0,tau_normal_rc1,tau_normal_all,
           tau_normal_boundary);
    printf("roundoff_negative=%d raw_min=%.17g error_bound=%.17g "
           "seeded_negative=%d\n",
           roundoff_negative,raw_min,error_bound,seeded_negative);
    return pass_good && fabs(sum - 1.0) <= 1e-12 && !pass_negative &&
           !pass_nonfinite && !pass_bad_sum && matrix_good == 0.0 &&
           bad_debit > 1e-12 && bad_target > 1e-12 && row_good == 0.0 &&
           row_seeded > 1e-12 && isinf(row_b0_nan) &&
           !row_b0_nan_gate_pass && flux_good == 0.0 &&
           route_seeded > 1e-12 && q_coupling_seeded > 1e-12 &&
           isinf(matrix_nan) && isinf(flux_nan) &&
           n_elem_finite_fraction == 0.4 &&
           isinf(n_elem_nan_fraction) && !n_elem_nan_gate_pass &&
           tau_first_rc == 0 && tau_overflow_rc == -1 &&
           isinf(tau_all) && isinf(tau_boundary) &&
           isinf(tau_overflow_fraction) && !tau_overflow_gate_pass &&
           tau_normal_rc0 == 0 && tau_normal_rc1 == 0 &&
           tau_normal_all == 3.0 && tau_normal_boundary == 2.0 &&
           roundoff_negative == 0 && raw_min == -DBL_EPSILON &&
           error_bound >= DBL_EPSILON && seeded_negative == 1 ? 0 : 1;
}
