#include "lumina.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int find_ion(const AtomicData *atom, int Z, int stage) {
    for (int ip = 0; ip < atom->n_ion_pops; ip++)
        if (atom->ion_pop_Z[ip] == Z && atom->ion_pop_stage[ip] == stage)
            return ip;
    return -1;
}

static double canonical_active_tau(const AtomicData *atom, int line,
                                   double T_e, double time_explosion) {
    int ip = find_ion(atom, atom->line_atomic_number[line],
                      atom->line_ion_number[line]);
    if (ip < 0) return 0.0;

    int lower = -1, upper = -1;
    for (int l = atom->level_offset[ip]; l < atom->level_offset[ip + 1]; l++) {
        if (atom->level_num[l] == atom->line_level_lower[line]) lower = l;
        if (atom->level_num[l] == atom->line_level_upper[line]) upper = l;
        if (lower >= 0 && upper >= 0) break;
    }
    if (lower < 0 || upper < 0) return 0.0;

    double n_ion = atom->ion_number_density[ip];
    double Z_part = atom->partition_functions[ip];
    double ground_energy_eV = atom->level_energy_eV[atom->level_offset[ip]];
    for (int l = atom->level_offset[ip] + 1;
         l < atom->level_offset[ip + 1]; l++) {
        if (atom->level_energy_eV[l] < ground_energy_eV)
            ground_energy_eV = atom->level_energy_eV[l];
    }
    double lower_boltz = (atom->level_energy_eV[lower] - ground_energy_eV) *
                         EV_TO_ERG / (K_BOLTZMANN * T_e);
    double upper_boltz = (atom->level_energy_eV[upper] - ground_energy_eV) *
                         EV_TO_ERG / (K_BOLTZMANN * T_e);
    double n_lower = 0.0, n_upper = 0.0;
    if (lower_boltz < 745.0) {
        double f_lower = (double)atom->level_g[lower] *
                         exp(-lower_boltz) / Z_part;
        n_lower = n_ion * f_lower;
    }
    if (upper_boltz < 745.0) {
        double f_upper = (double)atom->level_g[upper] *
                         exp(-upper_boltz) / Z_part;
        n_upper = n_ion * f_upper;
    }
    double tau = SOBOLEV_COEFF * atom->line_f_lu[line] *
                 atom->line_wavelength_cm[line] * time_explosion *
                 (n_lower - ((double)atom->level_g[lower] /
                             (double)atom->level_g[upper]) * n_upper);
    if (tau == 0.0) tau = 0.0;
    return tau;
}

/* canonical-tau 는 **덱에 고정된 바이트-parity 트립와이어**다.  tau 는 선 파장과
 * f_lu 에 의존하므로(위 canonical_active_tau 참조) 덱이 다르면 값도 다르다.
 * 단일 기대값은 한 덱에서만 유효하므로 덱별로 등록한다.
 * 등록되지 않은 덱은 **조용히 통과시키지 않고 실패**한다 — 기준선 없는 덱을
 * PASS 로 세탁하지 않는다.  scripts/run_zinert_selftest.py 의 표와 짝이다. */
typedef struct {
    const char *deck;      /* 덱 디렉터리 basename */
    long        active_lines;
    uint64_t    fnv64;
} TauBaseline;

static const TauBaseline TAU_BASELINES[] = {
    /* 종래 생산 덱(2,565,342 선). A2-07 의 LTE@T_rad,W -> LTE@T_e 변경 후 재수립. */
    {"tardis_reference_toy06_19p48d",      2211572, UINT64_C(0x4a80c65d9c37fad9)},
    /* 덱 전환 T1: jnu4 종속 + I20 수리 + CMFGEN_EXACT_HYD=1 (2,220,953 선).
     * 2026-08-06 T2 배터리 실측으로 수립. 같은 실행에서 구조 불변량
     * (bit_differences=0 · inactive_nonzero=0 · audit_rc=0)이 전부 성립함을 확인했다. */
    {"tardis_reference_toy06_19p48d_jnu4", 1867183, UINT64_C(0xf32d10df3421058b)},
    /* 19apr23 active-only quarantine deck after SH-GRID and exact-Hyd
     * promotion.  2026-08-08 fail-closed MEASURE_ONLY run: inactive_nonzero=0,
     * bit_differences=0, audit_rc=0.  Sigma changes cannot enter this tau-only
     * fixture; the value pins the sealed line/level topology. */
    {"tardis_reference_toy06_19p48d_sivcaiv_active",
                                              2588798, UINT64_C(0x6c53c2f89ad53e47)},
};

static const char *deck_basename(const char *path) {
    size_t n = strlen(path);
    while (n > 0 && path[n - 1] == '/') n--;          /* 후행 슬래시 제거 */
    size_t start = 0;
    for (size_t i = 0; i < n; i++)
        if (path[i] == '/') start = i + 1;
    static char buf[512];
    size_t len = n - start;
    if (len >= sizeof buf) len = sizeof buf - 1;
    memcpy(buf, path + start, len);
    buf[len] = '\0';
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s REFERENCE_DECK\n", argv[0]);
        return 64;
    }
    const char *deck_name = deck_basename(argv[1]);
    const TauBaseline *bl = NULL;
    for (size_t i = 0; i < sizeof TAU_BASELINES / sizeof TAU_BASELINES[0]; i++)
        if (strcmp(TAU_BASELINES[i].deck, deck_name) == 0) bl = &TAU_BASELINES[i];
    const int baseline_registered = bl != NULL;
    if (!baseline_registered)
        printf("[Z-INERT-CANONICAL-TAU] no baseline registered for deck '%s'; "
               "measuring invariants but refusing PASS\n", deck_name);
    const long expected_active_lines = bl ? bl->active_lines : -1;
    const uint64_t expected_active_tau_fnv64 = bl ? bl->fnv64 : UINT64_C(0);

    enum { DECK_SHELLS = 50 };
    AtomicData atom;
    if (load_atomic_data(&atom, argv[1], DECK_SHELLS) != 0) return 2;

    unsigned char inactive_Z[100] = {0};
    for (int e = 0; e < atom.n_elements; e++) {
        int Z = atom.element_Z[e];
        if (Z > 0 && Z < 100)
            inactive_Z[Z] = (unsigned char)
                lumina_zinert_element_inactive(&atom, e, DECK_SHELLS);
    }

    double *original_abundances = atom.abundances;
    double *original_ions = atom.ion_number_density;
    double *original_partition = atom.partition_functions;
    atom.abundances = calloc((size_t)atom.n_elements, sizeof(double));
    atom.ion_number_density = calloc((size_t)atom.n_ion_pops, sizeof(double));
    atom.partition_functions = calloc((size_t)atom.n_ion_pops, sizeof(double));
    double *tau = calloc((size_t)atom.n_lines, sizeof(double));
    double *source = calloc((size_t)atom.n_lines, sizeof(double));
    if (!atom.abundances || !atom.ion_number_density ||
        !atom.partition_functions ||
        !tau || !source) {
        fprintf(stderr, "[Z-INERT-CANONICAL-TAU][FATAL] allocation failed\n");
        return 3;
    }

    /* This is a one-shell execution view of the whole-deck classification.
     * An element which is active in any canonical shell must remain active in
     * the view even if its first physical shell happens to be exactly zero. */
    for (int e = 0; e < atom.n_elements; e++) {
        int Z = atom.element_Z[e];
        atom.abundances[e] =
            (Z > 0 && Z < 100 && inactive_Z[Z]) ? 0.0 : 1.0;
    }
    for (int ip = 0; ip < atom.n_ion_pops; ip++) {
        int Z = atom.ion_pop_Z[ip];
        atom.ion_number_density[ip] =
            (Z > 0 && Z < 100 && inactive_Z[Z]) ? 0.0 : (double)(17 * (ip + 1));
        atom.partition_functions[ip] = 1.0 + 0.125 * (double)(ip % 7);
    }

    double T_e[1] = {10000.0};
    PlasmaState plasma;
    OpacityState opacity;
    memset(&plasma, 0, sizeof plasma);
    memset(&opacity, 0, sizeof opacity);
    plasma.n_shells = 1;
    plasma.T_e = T_e;
    plasma.T_e_generation = 1;
    opacity.n_lines = atom.n_lines;
    opacity.n_shells = 1;
    opacity.tau_sobolev = tau;
    opacity.line_source_S = source;

    const double time_explosion = 86400.0;
    lumina_oracle_compute_tau_sobolev(&atom, &plasma, &opacity,
                                       time_explosion);

    long inactive_lines = 0, inactive_nonzero = 0;
    long active_lines = 0, active_bit_differences = 0;
    uint64_t active_hash = UINT64_C(14695981039346656037);
    for (int line = 0; line < atom.n_lines; line++) {
        int Z = atom.line_atomic_number[line];
        if (Z > 0 && Z < 100 && inactive_Z[Z]) {
            inactive_lines++;
            if (tau[line] != 0.0) inactive_nonzero++;
            continue;
        }
        double expected = canonical_active_tau(&atom, line, T_e[0],
                                               time_explosion);
        active_lines++;
        if (memcmp(&tau[line], &expected, sizeof expected) != 0)
            active_bit_differences++;
        const unsigned char *bytes = (const unsigned char *)&tau[line];
        for (size_t b = 0; b < sizeof tau[line]; b++) {
            active_hash ^= bytes[b];
            active_hash *= UINT64_C(1099511628211);
        }
    }

    int audit_rc = lumina_zinert_validate(&atom, NULL, &opacity, 1,
                                          "canonical-tau-transport");
    int passed = baseline_registered && !audit_rc && inactive_nonzero == 0 &&
                 active_lines == expected_active_lines &&
                 active_bit_differences == 0 &&
                 active_hash == expected_active_tau_fnv64;
    printf("[Z-INERT-CANONICAL-TAU] inactive_lines=%ld inactive_nonzero=%ld "
           "active_lines=%ld active_tau_bit_differences=%ld "
           "active_tau_fnv64=%016llx audit_rc=%d verdict=%s\n",
           inactive_lines, inactive_nonzero, active_lines,
           active_bit_differences, (unsigned long long)active_hash, audit_rc,
           passed ? "PASS" : (baseline_registered ? "FAIL" : "MEASURE_ONLY"));

    free(atom.abundances);
    free(atom.ion_number_density);
    free(atom.partition_functions);
    free(tau);
    free(source);
    atom.abundances = original_abundances;
    atom.ion_number_density = original_ions;
    atom.partition_functions = original_partition;
    free_atomic_data(&atom);

    if (!baseline_registered) return 65;
    return (audit_rc || inactive_nonzero != 0 ||
            active_lines != expected_active_lines ||
            active_bit_differences != 0 ||
            active_hash != expected_active_tau_fnv64) ? 1 : 0;
}
