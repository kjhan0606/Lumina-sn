#include "cmfgen_adiabatic.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define KB 1.380649e-16
#define PI 3.141592653589793238462643383279502884

static int close_rel(double actual, double expected, double tolerance)
{
    double scale = fmax(fabs(expected), 1.0e-300);
    return fabs(actual - expected) <= tolerance * scale;
}

static int fail(const char *message)
{
    fprintf(stderr, "[CMFGEN-ADIABATIC][FAIL] %s\n", message);
    return 1;
}

static int output_preserved(const CmfgenAdiabaticCell *actual,
                            const CmfgenAdiabaticCell *sentinel,
                            size_t n)
{
    return memcmp(actual, sentinel, n * sizeof(*actual)) == 0;
}

int main(void)
{
    enum { N = 3 };
    const double epoch = 1.0e6;
    double r[N] = {1.0e14, 2.0e14, 4.0e14};
    double v[N] = {1.0e8, 2.0e8, 4.0e8};
    double T[N] = {1.0e4, 1.2e4, 1.6e4};
    double n_atom[N] = {2.0e9, 2.0e9, 2.0e9};
    double n_e[N] = {1.0e9, 1.2e9, 1.6e9};
    double u[N] = {1.0e-11, 1.2e-11, 1.6e-11};
    CmfgenAdiabaticInput input = {
        N, epoch, r, v, T, n_atom, n_e, u
    };
    CmfgenAdiabaticCell out[N];
    memset(out, 0, sizeof(out));

    if (cmfgen_adiabatic_v3_homologous_evaluate(&input, out) !=
        CMFGEN_ADIABATIC_OK)
        return fail("four-component known answer rejected");

    const double dT_dr = 2.0e-11;
    const double dgamma_dr = 1.0e-15;
    const double du_dr = 2.0e-26;
    for (size_t s = 0; s < N; ++s) {
        double qT = 1.5 * (n_atom[s] + n_e[s]) * KB * v[s] * dT_dr;
        double qV = 3.0 * (n_atom[s] + n_e[s]) * KB * T[s] / epoch;
        double qG = 1.5 * n_atom[s] * KB * T[s] * v[s] * dgamma_dr;
        double qU = n_atom[s] * v[s] * du_dr;
        double q = ((qT + qV) + qG) + qU;
        if (!close_rel(out[s].temperature_gradient, qT, 3.0e-15) ||
            !close_rel(out[s].velocity_divergence, qV, 3.0e-15) ||
            !close_rel(out[s].electron_fraction_gradient, qG, 3.0e-15) ||
            !close_rel(out[s].internal_energy_gradient, qU, 3.0e-15) ||
            !close_rel(out[s].signed_total, q, 3.0e-15) ||
            out[s].heating != 0.0 || out[s].cooling != out[s].signed_total)
            return fail("four-component cgs value mismatch");
    }

    /* Round trip the exact 10^4 K / km s^-1 / 10^10 cm CMFGEN scaling at s=2. */
    {
        const size_t s = 2, nb = 1;
        double scale = 1.0e9 * KB / (4.0 * PI);
        double diagnostic_to_cgs = 4.0e-10 * PI;
        double dR = (r[s] - r[nb]) / 1.0e10;
        double Vkms = v[s] / 1.0e5;
        double T4 = T[s] / 1.0e4;
        double T4nb = T[nb] / 1.0e4;
        double gamma = n_e[s] / n_atom[s];
        double gamma_nb = n_e[nb] / n_atom[nb];
        double int_en4 = u[s] / (KB * 1.0e4);
        double int_en4_nb = u[nb] / (KB * 1.0e4);
        double A = 1.5 * scale * (n_atom[s] + n_e[s]) * Vkms / dR;
        double B = scale * (n_atom[s] + n_e[s]) * Vkms * 3.0 /
                   (r[s] / 1.0e10);
        double C = 1.5 * scale * n_atom[s] * Vkms / dR;
        double D = scale * n_atom[s] * Vkms / dR;
        if (!close_rel(A * (T4 - T4nb) * diagnostic_to_cgs,
                       out[s].temperature_gradient, 5.0e-15) ||
            !close_rel(B * T4 * diagnostic_to_cgs,
                       out[s].velocity_divergence, 5.0e-15) ||
            !close_rel(C * T4 * (gamma - gamma_nb) * diagnostic_to_cgs,
                       out[s].electron_fraction_gradient, 5.0e-15) ||
            !close_rel(D * (int_en4 - int_en4_nb) * diagnostic_to_cgs,
                       out[s].internal_energy_gradient, 5.0e-15))
            return fail("CMFGEN scaled-to-cgs round trip mismatch");
    }

    /* Constant T, electron fraction and internal energy leaves only 3P/t. */
    double T_const[N] = {1.0e4, 1.0e4, 1.0e4};
    double ne_const[N] = {1.0e9, 1.0e9, 1.0e9};
    double u_const[N] = {2.0e-11, 2.0e-11, 2.0e-11};
    input.temperature_K = T_const;
    input.n_electron_cm3 = ne_const;
    input.internal_energy_atom_erg = u_const;
    if (cmfgen_adiabatic_v3_homologous_evaluate(&input, out) !=
        CMFGEN_ADIABATIC_OK)
        return fail("constant-state known answer rejected");
    for (size_t s = 0; s < N; ++s) {
        double expected = 3.0 * (n_atom[s] + ne_const[s]) * KB *
                          T_const[s] / epoch;
        if (out[s].temperature_gradient != 0.0 ||
            out[s].electron_fraction_gradient != 0.0 ||
            out[s].internal_energy_gradient != 0.0 ||
            !close_rel(out[s].velocity_divergence, expected, 2.0e-15) ||
            out[s].signed_total != out[s].velocity_divergence)
            return fail("constant-state divergence-only closure failed");
    }

    /* A decreasing outward internal energy is signed heating, never zeroed. */
    double u_heating[N] = {3.0e-9, 2.0e-9, 0.0};
    input.internal_energy_atom_erg = u_heating;
    if (cmfgen_adiabatic_v3_homologous_evaluate(&input, out) !=
        CMFGEN_ADIABATIC_OK)
        return fail("signed heating case rejected");
    for (size_t s = 0; s < N; ++s)
        if (!(out[s].signed_total < 0.0) || out[s].cooling != 0.0 ||
            out[s].heating != -out[s].signed_total)
            return fail("negative WORK was not preserved as heating");

    /* Every failure must preserve the caller's candidate byte-for-byte. */
    CmfgenAdiabaticCell sentinel[N];
    memset(sentinel, 0x5a, sizeof(sentinel));
    memcpy(out, sentinel, sizeof(out));
    double v_bad[N] = {1.0e8, 2.1e8, 4.0e8};
    input.velocity_cm_s = v_bad;
    if (cmfgen_adiabatic_v3_homologous_evaluate(&input, out) !=
            CMFGEN_ADIABATIC_NON_HOMOLOGOUS ||
        !output_preserved(out, sentinel, N))
        return fail("non-homologous geometry did not fail atomically");

    input.velocity_cm_s = v;
    double r_bad[N] = {1.0e14, 1.0e14, 4.0e14};
    input.radius_cm = r_bad;
    if (cmfgen_adiabatic_v3_homologous_evaluate(&input, out) !=
            CMFGEN_ADIABATIC_INVALID_RADIUS_GRID ||
        !output_preserved(out, sentinel, N))
        return fail("bad radius grid did not fail atomically");

    input.radius_cm = r;
    double na_bad[N] = {2.0e9, 0.0, 2.0e9};
    input.n_atom_cm3 = na_bad;
    if (cmfgen_adiabatic_v3_homologous_evaluate(&input, out) !=
            CMFGEN_ADIABATIC_INVALID_DENSITY ||
        !output_preserved(out, sentinel, N))
        return fail("zero atom density did not fail atomically");

    input.n_atom_cm3 = n_atom;
    double T_bad[N] = {1.0e4, NAN, 1.0e4};
    input.temperature_K = T_bad;
    if (cmfgen_adiabatic_v3_homologous_evaluate(&input, out) !=
            CMFGEN_ADIABATIC_NONFINITE_INPUT ||
        !output_preserved(out, sentinel, N))
        return fail("nonfinite temperature did not fail atomically");

    input.temperature_K = T_const;
    double u_bad[N] = {2.0e-11, -1.0, 2.0e-11};
    input.internal_energy_atom_erg = u_bad;
    if (cmfgen_adiabatic_v3_homologous_evaluate(&input, out) !=
            CMFGEN_ADIABATIC_INVALID_INTERNAL_ENERGY ||
        !output_preserved(out, sentinel, N))
        return fail("negative internal energy did not fail atomically");

    input.internal_energy_atom_erg = u_const;
    input.n_shells = 1;
    if (cmfgen_adiabatic_v3_homologous_evaluate(&input, out) !=
            CMFGEN_ADIABATIC_NEED_TWO_SHELLS ||
        !output_preserved(out, sentinel, N))
        return fail("single-shell input did not fail atomically");

    input.n_shells = N;
    double u_overflow[N] = {DBL_MAX, 0.0, DBL_MAX};
    input.internal_energy_atom_erg = u_overflow;
    if (cmfgen_adiabatic_v3_homologous_evaluate(&input, out) !=
            CMFGEN_ADIABATIC_NONFINITE_RESULT ||
        !output_preserved(out, sentinel, N))
        return fail("overflowing result did not fail atomically");

    printf("[CMFGEN-ADIABATIC][SELFTEST] status=PASS shells=3 "
           "components=4 signed_heating=PASS cmfgen_unit_roundtrip=PASS "
           "boundary_stencil=PASS negative_controls=7 atomic_rollback=PASS "
           "production_publication=BLOCKED_ALL_SHELL_TRANSACTION\n");
    return 0;
}
