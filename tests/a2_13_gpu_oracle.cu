extern "C" {
#include "bf_rate_jnu.h"
}
#include "gpu_physics_kernels.h"
#include "gpu_opacity_kernels.h"
#include "gpu_emissivity_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char Q_HASH[] =
    "1111111111111111111111111111111111111111111111111111111111111111";
static const char PROFILE_HASH[] =
    "2222222222222222222222222222222222222222222222222222222222222222";

static int fixture_commit(RadiationFieldOwner *owner)
{
    const size_t ns = 2, nb = LUMINA_RADFIELD_N_BINS;
    const double v_inner[2] = {1e8, 2e8}, v_outer[2] = {2e8, 3e8};
    const uint64_t line_id[2] = {17, 29};
    const double line_jbar[4] = {2e-12, 3e-12, 5e-12, 7e-12};
    const int32_t line_validity[4] = {
        LINE_JBAR_VALID, LINE_JBAR_VALID, LINE_JBAR_VALID, LINE_JBAR_VALID};
    double *j = (double *)malloc(ns * nb * sizeof(double));
    RadiationFieldValidityState *validity =
        (RadiationFieldValidityState *)malloc(ns * nb * sizeof(*validity));
    if (!j || !validity) { free(j); free(validity); return -1; }
    for (size_t i = 0; i < ns * nb; ++i) {
        j[i] = (1.0 + (double)(i % 13)) * 1e-12;
        validity[i] = RADIATION_FIELD_VALID;
    }
    RadiationFieldCommitRequest request;
    memset(&request, 0, sizeof(request));
    request.provenance_kind = RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
    request.producer = "A2_13_GPU_ORACLE";
    request.generation = 1; request.epoch = 86400.0; request.n_shells = ns;
    request.v_inner = v_inner; request.v_outer = v_outer;
    request.source_n_bins = nb;
    request.source_frequency_bin_edges = owner->field.frequency_bin_edges.values;
    request.source_J_nu = j; request.source_validity = validity;
    request.statistic_kind = RADIATION_FIELD_DETERMINISTIC;
    request.line_n = 2; request.line_id = line_id;
    request.line_q_set_hash = Q_HASH; request.line_profile_id = 7;
    request.line_profile_hash = PROFILE_HASH;
    request.line_jbar = line_jbar; request.line_validity = line_validity;
    int rc = radiation_field_commit(owner, &request);
    free(j); free(validity);
    return rc;
}

static int close_rel(double a, double b, double tol)
{
    double scale = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    return fabs(a - b) <= tol * (scale > 1e-300 ? scale : 1.0);
}

int main(void)
{
    RadiationFieldOwner owner;
    GpuRadiationFieldMirror *mirror = NULL;
    GpuRadiationFieldReport report;
    GpuRadiationFieldDeviceView device_view;
    RadiationFieldView cpu_view;
    memset(&owner, 0, sizeof(owner)); memset(&report, 0, sizeof(report));
    if (radiation_field_owner_init(&owner, 2) || fixture_commit(&owner)) {
        fprintf(stderr, "A2_13_GPU_ORACLE_FIXTURE_FAIL\n"); return 70;
    }
    mirror = gpu_radiation_field_create();
    if (!mirror || gpu_radiation_field_sync(&owner, 86400.0, 2, 1, Q_HASH, 7,
            PROFILE_HASH, mirror, &report, NULL, GPU_RF_POISON_NONE) != GPU_RF_OK ||
        gpu_radiation_field_device_view(&owner, 1, Q_HASH, 7, PROFILE_HASH,
            mirror, &device_view, &report) != GPU_RF_OK ||
        radiation_field_read_view(&owner, 86400.0, 2, 1, &cpu_view) !=
            RADIATION_FIELD_VIEW_OK) {
        fprintf(stderr, "A2_13_GPU_ORACLE_MIRROR_FAIL status=%s\n",
                gpu_rf_status_name(report.status)); return 71;
    }

    const size_t b0 = 117, b1 = 133;
    const double sigma_nu[3] = {
        cpu_view.frequency_bin_edges[b0] * 1.0001,
        cpu_view.frequency_bin_edges[b1], cpu_view.frequency_bin_edges[b1 + 20]};
    const double sigma_value[3] = {3e-18, 2e-18, 0.0};
    BfCrossSection sigma = {3, sigma_nu, sigma_value, sigma_nu[0]};
    BfRateResult cpu_bf;
    GpuBfRateCell gpu_bf;
    GpuPhysicsCounters counters;
    gpu_physics_counters_init(&counters, 1);
    if (bf_rate_gamma_from_view(&cpu_view, 1, &sigma, &cpu_bf) ||
        gpu_physics_bf_rate(&device_view, 1, sigma_nu, sigma_value, 3,
                            sigma.nu_threshold, &gpu_bf, &counters, NULL) ||
        cpu_bf.state != BF_RATE_VALID || gpu_bf.validity != GPU_PHYSICS_VALID ||
        !close_rel(cpu_bf.gamma, gpu_bf.gamma, 2e-13)) {
        fprintf(stderr, "A2_13_BF_ORACLE_FAIL cpu=%.17g gpu=%.17g\n",
                cpu_bf.gamma, gpu_bf.gamma); return 72;
    }
    counters.cpu_gpu_bf_compared++;

    const double B_lu = 2.5, B_ul = 1.25, A_ul = 9.0;
    GpuBbRateCell gpu_bb;
    if (gpu_physics_bb_rate(&device_view, 1, 29, B_lu, B_ul, A_ul,
                            &gpu_bb, &counters, NULL) ||
        gpu_bb.validity != GPU_PHYSICS_VALID ||
        gpu_bb.jbar != 7e-12 || gpu_bb.upward != B_lu * 7e-12 ||
        gpu_bb.stimulated_downward != B_ul * 7e-12 ||
        gpu_bb.spontaneous_downward != A_ul ||
        counters.coarse_reintegration_attempts || counters.fine_grid_attempts ||
        counters.legacy_scalar_reads) {
        fprintf(stderr, "A2_13_BB_ORACLE_FAIL validity=%d\n", gpu_bb.validity);
        return 73;
    }
    counters.cpu_gpu_bb_compared++;

    const double es[3] = {2.0, 1.0, 0.5};
    const double bb[3] = {-0.25, -2.0, 0.0};
    const double bf_net[3] = {-0.5, 0.25, -0.75};
    const double ff[3] = {0.1, 0.5, 0.0};
    const double event[3] = {0.5, 0.25, 0.75};
    GpuOpacityCell opacity[3];
    if (gpu_physics_signed_opacity(&device_view, 1, es, bb, bf_net, ff,
            event, 3, opacity, &counters, NULL)) return 75;
    for (size_t i = 0; i < 3; ++i) {
        double cpu_total = ((es[i] + bb[i]) + bf_net[i]) + ff[i];
        if (opacity[i].total != cpu_total ||
            opacity[i].bf_event_measure != event[i] ||
            opacity[i].bf != bf_net[i]) {
            fprintf(stderr, "A2_14_OPACITY_ORACLE_FAIL cell=%zu\n", i);
            return 76;
        }
    }

    const size_t cells = device_view.n_shells * device_view.n_bins;
    double *component = (double *)calloc(5 * cells, sizeof(double));
    double *eta_total = (double *)malloc(cells * sizeof(double));
    double *cdf = (double *)malloc(cells * sizeof(double));
    double sample_u[2] = {0.25, 0.75}, sample_nu[2];
    if (!component || !eta_total || !cdf) return 77;
    for (size_t k = 0; k < 5; ++k)
        for (size_t i = 0; i < cells; ++i)
            component[k * cells + i] = (double)(k + 1) * (1.0 + i % 7) * 1e-30;
    if (gpu_physics_emissivity_cdf(&device_view, 1, component, sample_u,
            eta_total, cdf, sample_nu, &counters, NULL)) return 78;
    for (size_t i = 0; i < cells; ++i) {
        double expected = 0.0;
        for (size_t k = 0; k < 5; ++k) expected += component[k * cells + i];
        if (eta_total[i] != expected) return 79;
    }
    for (size_t s = 0; s < device_view.n_shells; ++s) {
        if (cdf[(s + 1) * device_view.n_bins - 1] != 1.0 ||
            !(sample_nu[s] >= cpu_view.frequency_bin_edges[0]) ||
            !(sample_nu[s] <= cpu_view.frequency_bin_edges[cpu_view.n_bins]))
            return 80;
    }
    counters.rng_draws_cpu += device_view.n_shells;
    free(component); free(eta_total); free(cdf);
    if (counters.cpu_gpu_bf_compared != 1 ||
        counters.cpu_gpu_bb_compared != 1 ||
        !gpu_physics_forbidden_attempts_zero(&counters)) return 74;
    printf("A2_13_15_GPU_ORACLE PASS bf=PASS bb=PASS conjunction=PASS "
           "opacity=PASS emissivity=PASS rng_cpu=%llu rng_gpu=%llu "
           "physical_launches=%llu coarse=0 fine=0 scalar=0\n",
           (unsigned long long)counters.rng_draws_cpu,
           (unsigned long long)counters.rng_draws_gpu,
           (unsigned long long)counters.physical_launches);
    gpu_radiation_field_destroy(mirror);
    radiation_field_owner_free(&owner);
    return 0;
}
