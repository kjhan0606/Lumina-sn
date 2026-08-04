#include <math.h>
#include <stdio.h>

double nlte_ew_test_assembly_residual(const double *, const double *, int);

int main(void) {
    /* Column-major two-state rate plane: 0 -> 1 at rate 4. */
    double A[4] = {-4.0, 4.0, 0.0, 0.0};
    double ledger[2] = {4.0, 0.0};
    double baseline = nlte_ew_test_assembly_residual(A, ledger, 2);
    A[0] = -3.0; /* seeded diagonal-debit corruption; ledger stays independent */
    double seeded = nlte_ew_test_assembly_residual(A, ledger, 2);
    int gate_pass = seeded <= 1e-12;
    printf("baseline_residual=%.17g seeded_residual=%.17g gate_pass=%d\n",
           baseline, seeded, gate_pass);
    return (baseline == 0.0 && seeded > 1e-12 && !gate_pass) ? 0 : 1;
}
