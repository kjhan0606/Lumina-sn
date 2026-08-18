#include "gpu_radiation_field.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char Q_HASH[] =
    "1111111111111111111111111111111111111111111111111111111111111111";
static const char E_HASH[] =
    "3333333333333333333333333333333333333333333333333333333333333333";
static const char PROFILE_HASH[] =
    "2222222222222222222222222222222222222222222222222222222222222222";

typedef enum {
    FIXTURE_OK = 0,
    FIXTURE_BAD_ARGUMENT = 1,
    FIXTURE_SIZE_OVERFLOW = 2,
    FIXTURE_HOST_ALLOCATION = 3,
    FIXTURE_COMMIT_REJECTED = 4
} FixtureStatus;

static const char *fixture_status_name(FixtureStatus status)
{
    switch (status) {
    case FIXTURE_OK: return "OK";
    case FIXTURE_BAD_ARGUMENT: return "BAD_ARGUMENT";
    case FIXTURE_SIZE_OVERFLOW: return "SIZE_OVERFLOW";
    case FIXTURE_HOST_ALLOCATION: return "HOST_ALLOCATION";
    case FIXTURE_COMMIT_REJECTED: return "COMMIT_REJECTED";
    default: return "UNKNOWN";
    }
}

static FixtureStatus commit_fixture(RadiationFieldOwner *owner,
                                    uint64_t generation, size_t n_shells)
{
    if (!owner || !owner->enabled || generation == 0 || n_shells == 0 ||
        n_shells > 4 || owner->field.J_nu.n_shells != n_shells)
        return FIXTURE_BAD_ARGUMENT;
    if (n_shells > SIZE_MAX / (size_t)LUMINA_RADFIELD_N_BINS ||
        n_shells * (size_t)LUMINA_RADFIELD_N_BINS > SIZE_MAX / sizeof(double))
        return FIXTURE_SIZE_OVERFLOW;
    size_t cells = n_shells * LUMINA_RADFIELD_N_BINS;
    double *j = (double *)malloc(cells * sizeof(double));
    RadiationFieldValidityState *validity =
        (RadiationFieldValidityState *)malloc(cells * sizeof(*validity));
    uint64_t line_id[4] = {11, 17, 23, 29};
    uint64_t rate_line_id[2] = {17, 29};
    double line_jbar[16];
    int32_t line_validity[16];
    if (!j || !validity) {
        free(j);
        free(validity);
        return FIXTURE_HOST_ALLOCATION;
    }
    for (size_t i = 0; i < cells; ++i) {
        j[i] = (double)(i + 1) * 1e-20;
        validity[i] = RADIATION_FIELD_VALID;
    }
    for (size_t i = 0; i < 4 * n_shells; ++i) {
        line_jbar[i] = (double)(i + 1) * 1e-12;
        line_validity[i] = LINE_JBAR_VALID;
    }
    /* radiation_field_commit requires a finite, contiguous shell geometry.
     * The old fixture omitted both pointers, so every GPU-node invocation
     * stopped in the CPU commit validator before any CUDA operation. */
    const double v_inner[4] = {1.0e8, 2.0e8, 3.0e8, 4.0e8};
    const double v_outer[4] = {2.0e8, 3.0e8, 4.0e8, 5.0e8};
    RadiationFieldCommitRequest request;
    memset(&request, 0, sizeof(request));
    request.provenance_kind = RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
    request.producer = "A2_12_GPU_LIFECYCLE_FIXTURE";
    request.generation = generation;
    request.epoch = 86400.0;
    request.n_shells = n_shells;
    request.v_inner = v_inner;
    request.v_outer = v_outer;
    request.source_n_bins = LUMINA_RADFIELD_N_BINS;
    request.source_frequency_bin_edges = owner->field.frequency_bin_edges.values;
    request.source_J_nu = j;
    request.source_validity = validity;
    request.statistic_kind = RADIATION_FIELD_DETERMINISTIC;
    request.line_n = 4;
    request.line_id = line_id;
    request.line_q_set_hash = E_HASH;
    request.line_set_kind = LINE_JBAR_SET_ENERGY_DOMAIN;
    request.line_rate_graph_n = 2;
    request.line_rate_graph_id = rate_line_id;
    request.line_rate_graph_hash = Q_HASH;
    request.line_profile_id = 7;
    request.line_profile_hash = PROFILE_HASH;
    request.line_provenance_kind =
        RADIATION_FIELD_PROVENANCE_CMFGEN_LINE_PROFILE_INTEGRAL;
    request.line_producer = LUMINA_LINE_JBAR_DETERMINISTIC_PRODUCER;
    request.line_jbar = line_jbar;
    request.line_validity = line_validity;
    int rc = radiation_field_commit(owner, &request);
    free(j);
    free(validity);
    return rc == 0 ? FIXTURE_OK : FIXTURE_COMMIT_REJECTED;
}

static int valid_mode(const char *mode)
{
    if (!strcmp(mode, "positive") || !strcmp(mode, "fixture")) return 1;
    return mode[0] == 'N' && mode[1] >= '1' && mode[1] <= '9' &&
           mode[2] == '\0';
}

static int expected_failure(const char *mode, const char *marker, int rc)
{
    printf("%s status=EXPECTED_NONZERO child_rc=%d physical_launches=0\n",
           marker, rc);
    (void)mode;
    return rc;
}

int main(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && !valid_mode(argv[1]))) {
        fprintf(stderr, "A2_12_FIXTURE_BAD_ARGUMENT usage=%s "
                "[positive|fixture|N1..N9]\n", argv[0]);
        return 2;
    }
    const char *mode = argc > 1 ? argv[1] : "positive";
    RadiationFieldOwner owner;
    GpuRadiationFieldMirror *mirror = NULL;
    GpuRadiationFieldReport report;
    GpuRadiationFieldStatus status;
    memset(&owner, 0, sizeof(owner));
    memset(&report, 0, sizeof(report));
    if (radiation_field_owner_init(&owner, 2) != 0) {
        fprintf(stderr, "A2_12_FIXTURE_OWNER_INIT_FAIL n_shells=2 "
                "host_bytes_min=%zu\n",
                2 * (size_t)LUMINA_RADFIELD_N_BINS *
                    (2 * sizeof(double) + sizeof(RadiationFieldValidityState) +
                     2 * sizeof(uint64_t)));
        return 70;
    }
    FixtureStatus fixture_status = commit_fixture(&owner, 1, 2);
    if (fixture_status != FIXTURE_OK) {
        fprintf(stderr, "A2_12_FIXTURE_COMMIT_FAIL reason=%s generation=1 "
                "n_shells=2 n_bins=%d geometry=present\n",
                fixture_status_name(fixture_status), LUMINA_RADFIELD_N_BINS);
        radiation_field_owner_free(&owner);
        return 70;
    }
    if (!strcmp(mode, "fixture")) {
        printf("A2_12_FIXTURE_CPU PASS generation=1 n_shells=2 n_bins=%d\n",
               LUMINA_RADFIELD_N_BINS);
        radiation_field_owner_free(&owner);
        return 0;
    }
    mirror = gpu_radiation_field_create();
    if (!mirror) {
        fprintf(stderr, "A2_12_FIXTURE_MIRROR_CREATE_FAIL reason=HOST_ALLOCATION\n");
        radiation_field_owner_free(&owner);
        return 70;
    }

    if (!strcmp(mode, "N1")) {
        owner.field.generation.required_generation = 2;
        status = gpu_radiation_field_sync(&owner, 86400.0, 2, 1, Q_HASH, 7,
            PROFILE_HASH, mirror, &report, NULL, GPU_RF_POISON_NONE);
        if (status == GPU_RF_STALE_CPU)
            return expected_failure(mode, "A2_12_NEG_CPU_STALE_FAIL", 41);
    } else if (!strcmp(mode, "N2")) {
        owner.line_jbar_cache.generation.required_generation = 0;
        owner.line_jbar_cache.generation.computed_generation = 0;
        status = gpu_radiation_field_sync(&owner, 86400.0, 2, 1, Q_HASH, 7,
            PROFILE_HASH, mirror, &report, NULL, GPU_RF_POISON_NONE);
        if (status == GPU_RF_STALE_LINE)
            return expected_failure(mode, "A2_12_NEG_CACHE_GENERATION_FAIL", 42);
    } else if (!strcmp(mode, "N3")) {
        status = gpu_radiation_field_sync(&owner, 86400.0, 2, 1, Q_HASH, 7,
            PROFILE_HASH, mirror, &report, NULL, GPU_RF_POISON_LINE_ID_SHUFFLE);
        if (status == GPU_RF_LINE_ID_MISMATCH)
            return expected_failure(mode, "A2_12_NEG_LINE_ID_MAPPING_FAIL", 43);
    } else if (!strcmp(mode, "N4")) {
        status = gpu_radiation_field_sync(&owner, 86400.0, 2, 1, Q_HASH, 7,
            PROFILE_HASH, mirror, &report, NULL, GPU_RF_POISON_NONE);
        owner.field.generation.required_generation = 2;
        owner.field.generation.computed_generation = 2;
        owner.line_jbar_cache.generation.required_generation = 2;
        owner.line_jbar_cache.generation.computed_generation = 2;
        status = gpu_radiation_field_require_ready(&owner, 2, mirror, &report);
        if (status == GPU_RF_CPU_GPU_GENERATION_MISMATCH)
            return expected_failure(mode, "A2_12_NEG_CPU_GPU_GENERATION_FAIL", 44);
    } else if (!strcmp(mode, "N5")) {
        status = gpu_radiation_field_sync(&owner, 86400.0, 2, 1, Q_HASH, 7,
            PROFILE_HASH, mirror, &report, NULL, GPU_RF_POISON_PARTIAL_COPY);
        const GpuRadiationFieldCounters *c = gpu_radiation_field_counters(mirror);
        if (status == GPU_RF_PARTIAL_UPLOAD && c->partial_upload_failures == 1 &&
            c->copy_failures == 0 && gpu_rf_counters_conserve(c))
            return expected_failure(mode, "A2_12_NEG_PARTIAL_UPLOAD_FAIL", 45);
    } else if (!strcmp(mode, "N6")) {
        owner.field.validity.values[0] = RADIATION_FIELD_UNSAMPLED;
        status = gpu_radiation_field_sync(&owner, 86400.0, 2, 1, Q_HASH, 7,
            PROFILE_HASH, mirror, &report, NULL, GPU_RF_POISON_NONE);
        if (status == GPU_RF_INVALID_CELL)
            return expected_failure(mode, "A2_12_NEG_INVALID_VALIDITY_FAIL", 46);
    } else if (!strcmp(mode, "N7")) {
        GpuRadiationFieldCounters c;
        gpu_rf_counters_init(&c);
        c.fallback_attempts++;
        gpu_rf_record_blocked_launch(&c);
        if (c.fallback_attempts == 1 && c.physical_launches == 0 &&
            gpu_rf_counters_conserve(&c))
            return expected_failure(mode, "A2_12_NEG_FALLBACK_FAIL", 47);
    } else if (!strcmp(mode, "N8")) {
        status = gpu_radiation_field_sync(&owner, 86400.0, 2, 1, Q_HASH, 7,
            PROFILE_HASH, mirror, &report, NULL, GPU_RF_POISON_REPORTED_BYTES);
        if (status == GPU_RF_UPLOAD_BYTES_MISMATCH)
            return expected_failure(mode, "A2_12_NEG_UPLOAD_BYTES_FAIL", 48);
    } else if (!strcmp(mode, "N9")) {
        status = gpu_radiation_field_reset(&owner, 2, mirror, &report, NULL);
        if (status == GPU_RESET_GENERATION_MISMATCH)
            return expected_failure(mode, "A2_12_NEG_RESET_GENERATION_FAIL", 49);
    } else {
        status = gpu_radiation_field_sync(&owner, 86400.0, 2, 1, Q_HASH, 7,
            PROFILE_HASH, mirror, &report, NULL, GPU_RF_POISON_NONE);
        if (status != GPU_RF_OK ||
            gpu_radiation_field_require_ready(&owner, 1, mirror, &report) != GPU_RF_OK ||
            report.total_upload_bytes != report.committed_bytes ||
            report.cache_upload_bytes == 0)
            return 71;
        GpuRadiationFieldDeviceView device_view;
        uint64_t gathered_id[2] = {0, 0};
        double gathered_jbar[4] = {0.0, 0.0, 0.0, 0.0};
        if (gpu_radiation_field_device_view(
                &owner, 1, Q_HASH, 7, PROFILE_HASH, mirror,
                &device_view, &report) != GPU_RF_OK ||
            device_view.n_lines != 2 || device_view.n_shells != 2 ||
            cudaMemcpy(gathered_id, device_view.line_id,
                       sizeof(gathered_id), cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            cudaMemcpy(gathered_jbar, device_view.line_jbar,
                       sizeof(gathered_jbar), cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            gathered_id[0] != 17 || gathered_id[1] != 29 ||
            gathered_jbar[0] != 3.0e-12 ||
            gathered_jbar[1] != 4.0e-12 ||
            gathered_jbar[2] != 7.0e-12 ||
            gathered_jbar[3] != 8.0e-12)
            return 75;
        if (gpu_radiation_field_reset(&owner, 1, mirror, &report, NULL) != GPU_RF_OK ||
            gpu_radiation_field_state(mirror) != GPU_RF_DIRTY)
            return 72;
        status = gpu_radiation_field_sync(&owner, 86400.0, 2, 1, Q_HASH, 7,
            PROFILE_HASH, mirror, &report, NULL, GPU_RF_POISON_NONE);
        if (status != GPU_RF_OK) return 73;
        gpu_radiation_field_free(mirror, &report);
        gpu_radiation_field_free(mirror, &report);
        if (gpu_radiation_field_state(mirror) != GPU_RF_EMPTY ||
            !gpu_rf_counters_conserve(gpu_radiation_field_counters(mirror)))
            return 74;
        printf("A2_12_GPU_LIFECYCLE PASS sparse_qg_from_qe=PASS "
               "n_shells=%zu n_bins=%zu n_lines=%zu "
               "cache_upload_bytes=%llu total_upload_bytes=%llu\n",
               report.n_shells, report.n_bins, report.n_lines,
               (unsigned long long)report.cache_upload_bytes,
               (unsigned long long)report.total_upload_bytes);
        gpu_radiation_field_destroy(mirror);
        radiation_field_owner_free(&owner);
        return 0;
    }
    fprintf(stderr, "A2_12_NEGATIVE_CONTROL_INTERNAL_FAIL mode=%s status=%s\n",
            mode, gpu_rf_status_name(status));
    return 79;
}
