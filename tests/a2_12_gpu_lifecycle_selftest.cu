#include "gpu_radiation_field.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char Q_HASH[] =
    "1111111111111111111111111111111111111111111111111111111111111111";
static const char PROFILE_HASH[] =
    "2222222222222222222222222222222222222222222222222222222222222222";

static int commit_fixture(RadiationFieldOwner *owner, uint64_t generation,
                          size_t n_shells)
{
    size_t cells = n_shells * LUMINA_RADFIELD_N_BINS;
    double *j = (double *)malloc(cells * sizeof(double));
    RadiationFieldValidityState *validity =
        (RadiationFieldValidityState *)malloc(cells * sizeof(*validity));
    uint64_t line_id[2] = {17, 29};
    double line_jbar[8], line_se[8];
    int32_t line_validity[8];
    if (!j || !validity || n_shells > 4) return -1;
    for (size_t i = 0; i < cells; ++i) {
        j[i] = (double)(i + 1) * 1e-20;
        validity[i] = RADIATION_FIELD_VALID;
    }
    for (size_t i = 0; i < 2 * n_shells; ++i) {
        line_jbar[i] = (double)(i + 1) * 1e-12;
        line_se[i] = 0.0;
        line_validity[i] = LINE_JBAR_VALID;
    }
    RadiationFieldCommitRequest request;
    memset(&request, 0, sizeof(request));
    request.provenance_kind = RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
    request.producer = "A2_12_GPU_LIFECYCLE_FIXTURE";
    request.generation = generation;
    request.epoch = 86400.0;
    request.n_shells = n_shells;
    request.source_n_bins = LUMINA_RADFIELD_N_BINS;
    request.source_frequency_bin_edges = owner->field.frequency_bin_edges.values;
    request.source_J_nu = j;
    request.source_validity = validity;
    request.statistic_kind = RADIATION_FIELD_DETERMINISTIC;
    request.line_n = 2;
    request.line_id = line_id;
    request.line_q_set_hash = Q_HASH;
    request.line_profile_id = 7;
    request.line_profile_hash = PROFILE_HASH;
    request.line_jbar = line_jbar;
    request.line_validity = line_validity;
    int rc = radiation_field_commit(owner, &request);
    free(j);
    free(validity);
    return rc;
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
    const char *mode = argc > 1 ? argv[1] : "positive";
    RadiationFieldOwner owner;
    GpuRadiationFieldMirror *mirror = NULL;
    GpuRadiationFieldReport report;
    GpuRadiationFieldStatus status;
    memset(&owner, 0, sizeof(owner));
    memset(&report, 0, sizeof(report));
    if (radiation_field_owner_init(&owner, 2) != 0 ||
        commit_fixture(&owner, 1, 2) != 0 ||
        !(mirror = gpu_radiation_field_create())) {
        fprintf(stderr, "A2_12_FIXTURE_SETUP_FAIL\n");
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
        printf("A2_12_GPU_LIFECYCLE PASS n_shells=%zu n_bins=%zu n_lines=%zu "
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
