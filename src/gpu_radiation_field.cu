#include "gpu_radiation_field.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    void *component[GPU_RF_COMPONENT_COUNT];
    uint64_t bytes[GPU_RF_COMPONENT_COUNT];
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
} GpuRfBuffers;

typedef struct {
    uint64_t required_generation;
    uint64_t computed_generation;
    uint64_t line_required_generation;
    uint64_t line_computed_generation;
    uint64_t upload_serial;
    uint64_t n_shells;
    uint64_t n_bins;
    uint64_t n_lines;
    uint32_t units;
    uint32_t frame;
    double epoch;
} GpuRfFixedMetadata;

struct GpuRadiationFieldMirror {
    GpuRadiationFieldState state;
    GpuRfBuffers live;
    uint64_t cpu_required_generation;
    uint64_t cpu_computed_generation;
    uint64_t line_required_generation;
    uint64_t line_computed_generation;
    uint64_t gpu_committed_generation;
    uint64_t upload_serial;
    uint64_t committed_bytes;
    size_t n_shells;
    size_t n_bins;
    size_t n_lines;
    RadiationFieldUnits units;
    RadiationFieldFrame frame;
    double epoch;
    char edge_sha256[65];
    char q_set_sha256[65];
    uint64_t profile_id;
    char profile_sha256[65];
    cudaEvent_t ready_event;
    int event_valid;
    GpuRadiationFieldCounters counters;
};

static GpuRadiationFieldMirror *production_mirror;
static const RadiationFieldOwner *production_owner;

enum {
    C_FIELD_EDGE = 0, C_FIELD_VALUE = 1, C_FIELD_VALIDITY = 2,
    C_LINE_ID = 3, C_LINE_VALUE = 4, C_LINE_VALIDITY = 5,
    C_LINE_COUNT = 6, C_LINE_SE = 7, C_METADATA = 8
};

static int checked_mul(size_t a, size_t b, uint64_t *out)
{
    if (!out || (a != 0 && b > SIZE_MAX / a)) return -1;
    size_t value = a * b;
    if ((uint64_t)value > UINT64_MAX) return -1;
    *out = (uint64_t)value;
    return 0;
}

static int checked_add(uint64_t a, uint64_t b, uint64_t *out)
{
    if (!out || b > UINT64_MAX - a) return -1;
    *out = a + b;
    return 0;
}

static void free_buffers(GpuRfBuffers *buffers)
{
    int i;
    if (!buffers) return;
    for (i = 0; i < GPU_RF_COMPONENT_COUNT; ++i) {
        if (buffers->component[i]) cudaFree(buffers->component[i]);
        buffers->component[i] = NULL;
        buffers->bytes[i] = 0;
    }
    if (buffers->start_event) cudaEventDestroy(buffers->start_event);
    if (buffers->stop_event) cudaEventDestroy(buffers->stop_event);
    buffers->start_event = NULL;
    buffers->stop_event = NULL;
}

static uint64_t buffer_total(const GpuRfBuffers *buffers)
{
    uint64_t total = 0;
    int i;
    if (!buffers) return 0;
    for (i = 0; i < GPU_RF_COMPONENT_COUNT; ++i) total += buffers->bytes[i];
    return total;
}

static void invalidate_public(GpuRadiationFieldMirror *mirror,
                              GpuRadiationFieldState state)
{
    if (!mirror) return;
    mirror->gpu_committed_generation = 0;
    mirror->state = state;
}

static GpuRadiationFieldStatus fail_sync(GpuRadiationFieldMirror *mirror,
                                         GpuRadiationFieldReport *report,
                                         GpuRadiationFieldStatus status,
                                         GpuRfBuffers *candidate,
                                         uint64_t invalid_field,
                                         uint64_t invalid_line)
{
    uint64_t discarded = buffer_total(candidate);
    free_buffers(candidate);
    invalidate_public(mirror, GPU_RF_FAILED);
    if (report) {
        report->status = status;
        report->discarded_candidate_bytes = discarded;
    }
    gpu_rf_record_sync_failure(&mirror->counters, status,
                               invalid_field, invalid_line);
    return status;
}

static int compute_bytes(size_t n_shells, size_t n_bins, size_t n_lines,
                         GpuRfBuffers *candidate,
                         GpuRadiationFieldReport *report)
{
    uint64_t cells, line_cells, total = 0, cache = 0;
    if (checked_mul(n_shells, n_bins, &cells) != 0 ||
        checked_mul(n_lines, n_shells, &line_cells) != 0 ||
        checked_mul(n_bins + 1, sizeof(double), &candidate->bytes[C_FIELD_EDGE]) != 0 ||
        checked_mul(cells, sizeof(double), &candidate->bytes[C_FIELD_VALUE]) != 0 ||
        checked_mul(cells, sizeof(RadiationFieldValidityState),
                    &candidate->bytes[C_FIELD_VALIDITY]) != 0 ||
        checked_mul(n_lines, sizeof(uint64_t), &candidate->bytes[C_LINE_ID]) != 0 ||
        checked_mul(line_cells, sizeof(double), &candidate->bytes[C_LINE_VALUE]) != 0 ||
        checked_mul(line_cells, sizeof(LineJbarValidityState),
                    &candidate->bytes[C_LINE_VALIDITY]) != 0 ||
        checked_mul(line_cells, sizeof(uint64_t), &candidate->bytes[C_LINE_COUNT]) != 0 ||
        checked_mul(line_cells, sizeof(double), &candidate->bytes[C_LINE_SE]) != 0)
        return -1;
    candidate->bytes[C_METADATA] = sizeof(GpuRfFixedMetadata);
    for (int i = 0; i < GPU_RF_COMPONENT_COUNT; ++i)
        if (checked_add(total, candidate->bytes[i], &total) != 0) return -1;
    for (int i = C_LINE_ID; i <= C_LINE_SE; ++i)
        if (checked_add(cache, candidate->bytes[i], &cache) != 0) return -1;
    /* Fixed line-cache metadata is carried in the one fixed metadata transfer. */
    if (checked_add(cache, sizeof(GpuRfFixedMetadata), &cache) != 0) return -1;
    if (report) {
        memcpy(report->requested, candidate->bytes, sizeof(report->requested));
        report->cache_upload_bytes = cache;
        report->total_upload_bytes = total;
        report->field_validity_size = sizeof(RadiationFieldValidityState);
        report->line_validity_size = sizeof(LineJbarValidityState);
    }
    return 0;
}

GpuRadiationFieldMirror *gpu_radiation_field_create(void)
{
    GpuRadiationFieldMirror *mirror =
        (GpuRadiationFieldMirror *)calloc(1, sizeof(*mirror));
    if (mirror) {
        mirror->state = GPU_RF_EMPTY;
        gpu_rf_counters_init(&mirror->counters);
    }
    return mirror;
}

GpuRadiationFieldStatus gpu_radiation_field_sync(
    const RadiationFieldOwner *owner, double expected_epoch,
    size_t expected_n_shells, uint64_t expected_generation,
    const char *expected_q_set_hash, uint64_t expected_profile_id,
    const char *expected_profile_hash, GpuRadiationFieldMirror *mirror,
    GpuRadiationFieldReport *report, void *cuda_stream,
    GpuRadiationFieldPoison poison)
{
    RadiationFieldView field_view;
    LineJbarView line_view;
    GpuRfBuffers candidate;
    GpuRfFixedMetadata metadata;
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    uint64_t invalid_field = 0, invalid_line = 0, copied = 0;
    size_t field_cells, line_cells;
    int i;
    memset(&candidate, 0, sizeof(candidate));
    memset(&metadata, 0, sizeof(metadata));
    if (report) memset(report, 0, sizeof(*report));
    if (!mirror) return GPU_RF_NOT_READY;
    invalidate_public(mirror, GPU_RF_UPLOADING);

    if (poison == GPU_RF_POISON_CPU_CHANGED)
        return fail_sync(mirror, report, GPU_CPU_CHANGED_DURING_UPLOAD,
                         &candidate, 0, 0);
    if (!owner || !owner->enabled)
        return fail_sync(mirror, report, GPU_RF_DISABLED, &candidate, 0, 0);
    if (owner->field.generation.required_generation != expected_generation ||
        owner->field.generation.computed_generation != expected_generation)
        return fail_sync(mirror, report, GPU_RF_STALE_CPU, &candidate, 0, 0);
    if (owner->line_jbar_cache.generation.required_generation != expected_generation ||
        owner->line_jbar_cache.generation.computed_generation != expected_generation)
        return fail_sync(mirror, report, GPU_RF_STALE_LINE, &candidate, 0, 0);
    if (radiation_field_read_view(owner, expected_epoch, expected_n_shells,
                                  expected_generation, &field_view) !=
        RADIATION_FIELD_VIEW_OK)
        return fail_sync(mirror, report, GPU_RF_SHAPE_OR_HASH_MISMATCH,
                         &candidate, 0, 0);
    /* ★2026-08-08: DET-R6(3ca077d)가 이 view API 에 expected_profile_hash 를 추가하고
     * GPU 호출부를 안 고쳐 **GPU 빌드가 깨진 채 커밋**됐다(CPU 타깃만 빌드했다).
     * 인자는 이 함수의 매개변수로 이미 들어와 있다 — 넘기기만 하면 된다.
     * NULL 을 넘기는 것은 검사 무력화이므로 금지. */
    if (radiation_field_line_jbar_view(owner, expected_epoch, expected_n_shells,
            expected_generation, expected_q_set_hash, expected_profile_id,
            expected_profile_hash,
            &line_view) != LINE_JBAR_VIEW_OK)
        return fail_sync(mirror, report, GPU_RF_PROFILE_OR_QSET_MISMATCH,
                         &candidate, 0, 0);
    if (owner->field.generation.required_generation != expected_generation ||
        owner->field.generation.computed_generation != expected_generation)
        return fail_sync(mirror, report, GPU_CPU_CHANGED_DURING_UPLOAD,
                         &candidate, 0, 0);
    if (field_view.n_shells != expected_n_shells ||
        field_view.n_bins != LUMINA_RADFIELD_N_BINS ||
        owner->field.units != RADIATION_FIELD_UNITS_ERG_S_NEG1_CM_NEG2_HZ_NEG1_SR_NEG1 ||
        owner->field.frame != RADIATION_FIELD_FRAME_SHELL_COMOVING ||
        strcmp(LUMINA_RADFIELD_EDGE_SHA256,
               owner->field.provenance.frequency_edge_sha256) != 0)
        return fail_sync(mirror, report, GPU_RF_SHAPE_OR_HASH_MISMATCH,
                         &candidate, 0, 0);
    for (i = 1; i < (int)line_view.n_lines; ++i)
        if (line_view.line_id[i - 1] >= line_view.line_id[i])
            return fail_sync(mirror, report, GPU_RF_LINE_ID_MISMATCH,
                             &candidate, 0, 0);
    if (!expected_q_set_hash || !expected_profile_hash ||
        !owner->line_jbar_cache.q_set_hash ||
        !owner->line_profile_hash_storage ||
        strcmp(expected_q_set_hash, owner->line_jbar_cache.q_set_hash) != 0 ||
        expected_profile_id != owner->line_profile_id ||
        strcmp(expected_profile_hash, owner->line_profile_hash_storage) != 0)
        return fail_sync(mirror, report, GPU_RF_PROFILE_OR_QSET_MISMATCH,
                         &candidate, 0, 0);
    field_cells = field_view.n_shells * field_view.n_bins;
    line_cells = line_view.n_lines * line_view.n_shells;
    for (size_t k = 0; k < field_cells; ++k)
        if (field_view.validity[k] != RADIATION_FIELD_VALID &&
            field_view.validity[k] != RADIATION_FIELD_EXACT_ZERO) invalid_field++;
    for (size_t k = 0; k < line_cells; ++k)
        if (line_view.validity[k] != LINE_JBAR_VALID &&
            line_view.validity[k] != LINE_JBAR_EXACT_ZERO) invalid_line++;
    if (invalid_field || invalid_line)
        return fail_sync(mirror, report, GPU_RF_INVALID_CELL, &candidate,
                         invalid_field, invalid_line);
    if (poison == GPU_RF_POISON_LINE_ID_SHUFFLE)
        return fail_sync(mirror, report, GPU_RF_LINE_ID_MISMATCH,
                         &candidate, 0, 0);
    if (compute_bytes(field_view.n_shells, field_view.n_bins,
                      line_view.n_lines, &candidate, report) != 0)
        return fail_sync(mirror, report, GPU_RF_SHAPE_OR_HASH_MISMATCH,
                         &candidate, 0, 0);
    if (poison == GPU_RF_POISON_REPORTED_BYTES && report)
        report->cache_upload_bytes -= candidate.bytes[C_LINE_SE];

    for (i = 0; i < GPU_RF_COMPONENT_COUNT; ++i) {
        if (cudaMalloc(&candidate.component[i], (size_t)candidate.bytes[i]) !=
            cudaSuccess)
            return fail_sync(mirror, report, GPU_RF_ALLOCATION_FAILURE,
                             &candidate, 0, 0);
    }
    if (report)
        report->peak_live_device_bytes = mirror->committed_bytes +
                                         buffer_total(&candidate);
    if (cudaEventCreate(&candidate.start_event) != cudaSuccess ||
        cudaEventCreate(&candidate.stop_event) != cudaSuccess ||
        cudaEventRecord(candidate.start_event, stream) != cudaSuccess)
        return fail_sync(mirror, report, GPU_RF_EVENT_FAILURE,
                         &candidate, 0, 0);

    const void *sources[GPU_RF_COMPONENT_COUNT] = {
        field_view.frequency_bin_edges, field_view.J_nu, field_view.validity,
        line_view.line_id, line_view.jbar, line_view.validity,
        line_view.count, line_view.se, &metadata
    };
    metadata.required_generation = owner->field.generation.required_generation;
    metadata.computed_generation = owner->field.generation.computed_generation;
    metadata.line_required_generation =
        owner->line_jbar_cache.generation.required_generation;
    metadata.line_computed_generation =
        owner->line_jbar_cache.generation.computed_generation;
    metadata.upload_serial = mirror->upload_serial + 1;
    metadata.n_shells = field_view.n_shells;
    metadata.n_bins = field_view.n_bins;
    metadata.n_lines = line_view.n_lines;
    metadata.units = (uint32_t)owner->field.units;
    metadata.frame = (uint32_t)owner->field.frame;
    metadata.epoch = owner->field.epoch;

    for (i = 0; i < GPU_RF_COMPONENT_COUNT; ++i) {
        if (poison == GPU_RF_POISON_PARTIAL_COPY && i == C_LINE_VALIDITY)
            return fail_sync(mirror, report, GPU_RF_PARTIAL_UPLOAD,
                             &candidate, 0, 0);
        cudaError_t rc = cudaMemcpyAsync(candidate.component[i], sources[i],
            (size_t)candidate.bytes[i], cudaMemcpyHostToDevice, stream);
        if (rc != cudaSuccess) {
            GpuRadiationFieldStatus status = copied ? GPU_RF_PARTIAL_UPLOAD
                                                    : GPU_RF_COPY_FAILURE;
            return fail_sync(mirror, report, status, &candidate, 0, 0);
        }
        copied++;
        if (report) report->succeeded[i] = candidate.bytes[i];
    }
    if (cudaEventRecord(candidate.stop_event, stream) != cudaSuccess ||
        cudaEventSynchronize(candidate.stop_event) != cudaSuccess)
        return fail_sync(mirror, report, GPU_RF_EVENT_FAILURE,
                         &candidate, 0, 0);
    float elapsed_ms = 0.0f;
    if (cudaEventElapsedTime(&elapsed_ms, candidate.start_event,
                             candidate.stop_event) != cudaSuccess)
        return fail_sync(mirror, report, GPU_RF_EVENT_FAILURE,
                         &candidate, 0, 0);
    cudaEventDestroy(candidate.start_event);
    candidate.start_event = NULL;

    /* D2H attestation is deliberately exhaustive: every uploaded element and
     * fixed-width metadata byte is compared before READY can be published. */
    for (i = 0; i < GPU_RF_COMPONENT_COUNT; ++i) {
        void *attested = malloc((size_t)candidate.bytes[i]);
        if (!attested || cudaMemcpy(attested, candidate.component[i],
                (size_t)candidate.bytes[i], cudaMemcpyDeviceToHost) != cudaSuccess ||
            memcmp(attested, sources[i], (size_t)candidate.bytes[i]) != 0) {
            free(attested);
            return fail_sync(mirror, report, GPU_RF_PARTIAL_UPLOAD,
                             &candidate, 0, 0);
        }
        free(attested);
    }

    uint64_t expected_cache = sizeof(GpuRfFixedMetadata);
    for (i = C_LINE_ID; i <= C_LINE_SE; ++i)
        expected_cache += candidate.bytes[i];
    if (!report || report->cache_upload_bytes != expected_cache ||
        report->total_upload_bytes != buffer_total(&candidate))
        return fail_sync(mirror, report, GPU_RF_UPLOAD_BYTES_MISMATCH,
                         &candidate, 0, 0);

    GpuRfBuffers old = mirror->live;
    cudaEvent_t old_ready_event = mirror->ready_event;
    mirror->live = candidate;
    mirror->ready_event = candidate.stop_event;
    mirror->event_valid = 1;
    mirror->live.stop_event = NULL;
    memset(&candidate, 0, sizeof(candidate));
    mirror->cpu_required_generation = expected_generation;
    mirror->cpu_computed_generation = expected_generation;
    mirror->line_required_generation = expected_generation;
    mirror->line_computed_generation = expected_generation;
    mirror->gpu_committed_generation = expected_generation;
    mirror->upload_serial++;
    mirror->n_shells = field_view.n_shells;
    mirror->n_bins = field_view.n_bins;
    mirror->n_lines = line_view.n_lines;
    mirror->units = owner->field.units;
    mirror->frame = owner->field.frame;
    mirror->epoch = owner->field.epoch;
    snprintf(mirror->edge_sha256, sizeof(mirror->edge_sha256), "%s",
             LUMINA_RADFIELD_EDGE_SHA256);
    snprintf(mirror->q_set_sha256, sizeof(mirror->q_set_sha256), "%s",
             expected_q_set_hash);
    mirror->profile_id = expected_profile_id;
    snprintf(mirror->profile_sha256, sizeof(mirror->profile_sha256), "%s",
             expected_profile_hash);
    mirror->committed_bytes = buffer_total(&mirror->live);
    mirror->state = GPU_RF_READY;
    free_buffers(&old);
    if (old_ready_event) cudaEventDestroy(old_ready_event);
    gpu_rf_record_sync_commit(&mirror->counters);
    if (report) {
        report->committed_bytes = mirror->committed_bytes;
        report->event_elapsed_seconds = elapsed_ms / 1000.0;
        report->upload_serial = mirror->upload_serial;
        report->generation = expected_generation;
        report->n_shells = mirror->n_shells;
        report->n_bins = mirror->n_bins;
        report->n_lines = mirror->n_lines;
        report->status = GPU_RF_OK;
    }
    return GPU_RF_OK;
}

GpuRadiationFieldStatus gpu_radiation_field_require_ready(
    const RadiationFieldOwner *owner, uint64_t expected_generation,
    const GpuRadiationFieldMirror *mirror, GpuRadiationFieldReport *report)
{
    GpuRadiationFieldStatus status = GPU_RF_OK;
    if (!mirror || mirror->state != GPU_RF_READY) status = GPU_RF_NOT_READY;
    else if (!owner || !owner->enabled) status = GPU_RF_DISABLED;
    else if (owner->field.generation.required_generation != expected_generation ||
             owner->field.generation.computed_generation != expected_generation)
        status = GPU_RF_STALE_CPU;
    else if (owner->line_jbar_cache.generation.required_generation != expected_generation ||
             owner->line_jbar_cache.generation.computed_generation != expected_generation)
        status = GPU_RF_STALE_LINE;
    else if (mirror->gpu_committed_generation != expected_generation)
        status = GPU_RF_CPU_GPU_GENERATION_MISMATCH;
    else if (mirror->n_shells != owner->field.J_nu.n_shells ||
             mirror->n_bins != owner->field.J_nu.n_bins ||
             mirror->n_lines != owner->line_n_compact ||
             mirror->units != owner->field.units || mirror->frame != owner->field.frame ||
             mirror->epoch != owner->field.epoch)
        status = GPU_RF_SHAPE_OR_HASH_MISMATCH;
    gpu_rf_record_ready((GpuRadiationFieldCounters *)&mirror->counters,
                        status == GPU_RF_OK);
    if (report) report->status = status;
    return status;
}

GpuRadiationFieldStatus gpu_radiation_field_device_view(
    const RadiationFieldOwner *owner, uint64_t expected_generation,
    const char *expected_q_set_hash, uint64_t expected_profile_id,
    const char *expected_profile_hash,
    const GpuRadiationFieldMirror *mirror,
    GpuRadiationFieldDeviceView *out, GpuRadiationFieldReport *report)
{
    if (!out) {
        if (report) report->status = GPU_RF_NOT_READY;
        return GPU_RF_NOT_READY;
    }
    memset(out, 0, sizeof(*out));
    GpuRadiationFieldStatus status = gpu_radiation_field_require_ready(
        owner, expected_generation, mirror, report);
    if (status != GPU_RF_OK) return status;
    if (!expected_q_set_hash || !expected_profile_hash ||
        strcmp(expected_q_set_hash, mirror->q_set_sha256) != 0 ||
        expected_profile_id != mirror->profile_id ||
        strcmp(expected_profile_hash, mirror->profile_sha256) != 0) {
        if (report) report->status = GPU_RF_PROFILE_OR_QSET_MISMATCH;
        return GPU_RF_PROFILE_OR_QSET_MISMATCH;
    }
    out->frequency_bin_edges =
        (const double *)mirror->live.component[C_FIELD_EDGE];
    out->J_nu = (const double *)mirror->live.component[C_FIELD_VALUE];
    out->field_validity = (const RadiationFieldValidityState *)
        mirror->live.component[C_FIELD_VALIDITY];
    out->line_id = (const uint64_t *)mirror->live.component[C_LINE_ID];
    out->line_jbar = (const double *)mirror->live.component[C_LINE_VALUE];
    out->line_validity = (const LineJbarValidityState *)
        mirror->live.component[C_LINE_VALIDITY];
    out->line_count = (const uint64_t *)mirror->live.component[C_LINE_COUNT];
    out->line_se = (const double *)mirror->live.component[C_LINE_SE];
    out->generation = mirror->gpu_committed_generation;
    out->n_shells = mirror->n_shells;
    out->n_bins = mirror->n_bins;
    out->n_lines = mirror->n_lines;
    return GPU_RF_OK;
}

GpuRadiationFieldStatus gpu_radiation_field_reset(
    const RadiationFieldOwner *owner, uint64_t required_generation,
    GpuRadiationFieldMirror *mirror, GpuRadiationFieldReport *report,
    void *cuda_stream)
{
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    if (!mirror || !owner ||
        owner->field.generation.required_generation != required_generation ||
        owner->line_jbar_cache.generation.required_generation != required_generation) {
        if (report) report->status = GPU_RESET_GENERATION_MISMATCH;
        return GPU_RESET_GENERATION_MISMATCH;
    }
    invalidate_public(mirror, GPU_RF_DIRTY);
    mirror->counters.reset_count++;
    for (int i = 0; i < GPU_RF_COMPONENT_COUNT; ++i)
        if (mirror->live.component[i] &&
            cudaMemsetAsync(mirror->live.component[i], 0,
                            (size_t)mirror->live.bytes[i], stream) != cudaSuccess) {
            mirror->state = GPU_RF_FAILED;
            if (report) report->status = GPU_RF_CUDA_FAILURE;
            return GPU_RF_CUDA_FAILURE;
        }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        mirror->state = GPU_RF_FAILED;
        if (report) report->status = GPU_RF_CUDA_FAILURE;
        return GPU_RF_CUDA_FAILURE;
    }
    if (report) report->status = GPU_RF_OK;
    return GPU_RF_OK;
}

void gpu_radiation_field_free(GpuRadiationFieldMirror *mirror,
                              GpuRadiationFieldReport *report)
{
    if (!mirror) return;
    invalidate_public(mirror, GPU_RF_EMPTY);
    free_buffers(&mirror->live);
    if (mirror->ready_event) cudaEventDestroy(mirror->ready_event);
    mirror->ready_event = NULL;
    mirror->event_valid = 0;
    mirror->committed_bytes = 0;
    mirror->counters.free_count++;
    if (report) report->status = GPU_RF_OK;
}

void gpu_radiation_field_destroy(GpuRadiationFieldMirror *mirror)
{
    if (!mirror) return;
    gpu_radiation_field_free(mirror, NULL);
    free(mirror);
}

const GpuRadiationFieldCounters *gpu_radiation_field_counters(
    const GpuRadiationFieldMirror *mirror)
{
    return mirror ? &mirror->counters : NULL;
}

GpuRadiationFieldState gpu_radiation_field_state(
    const GpuRadiationFieldMirror *mirror)
{
    return mirror ? mirror->state : GPU_RF_EMPTY;
}

GpuRadiationFieldStatus gpu_radiation_field_production_bind(
    const RadiationFieldOwner *owner, GpuRadiationFieldReport *report,
    void *cuda_stream)
{
    if (!owner || !owner->enabled ||
        owner->field.generation.required_generation == 0 ||
        owner->field.generation.required_generation !=
            owner->field.generation.computed_generation ||
        !owner->line_jbar_cache.q_set_hash ||
        !owner->line_profile_hash_storage || owner->line_profile_id == 0) {
        if (report) { memset(report, 0, sizeof(*report));
                      report->status = GPU_RF_NOT_READY; }
        return GPU_RF_NOT_READY;
    }
    if (!production_mirror) production_mirror = gpu_radiation_field_create();
    if (!production_mirror) return GPU_RF_ALLOCATION_FAILURE;
    uint64_t generation = owner->field.generation.required_generation;
    if (production_owner == owner &&
        gpu_radiation_field_state(production_mirror) == GPU_RF_READY &&
        production_mirror->gpu_committed_generation == generation)
        return gpu_radiation_field_require_ready(owner, generation,
                                                  production_mirror, report);
    GpuRadiationFieldStatus status = gpu_radiation_field_sync(
        owner, owner->field.epoch, owner->field.J_nu.n_shells, generation,
        owner->line_jbar_cache.q_set_hash, owner->line_profile_id,
        owner->line_profile_hash_storage, production_mirror, report,
        cuda_stream, GPU_RF_POISON_NONE);
    if (status == GPU_RF_OK) production_owner = owner;
    return status;
}

GpuRadiationFieldStatus gpu_radiation_field_production_view(
    const RadiationFieldOwner *owner, GpuRadiationFieldDeviceView *out,
    GpuRadiationFieldReport *report)
{
    if (!owner || owner != production_owner || !production_mirror)
        return GPU_RF_NOT_READY;
    return gpu_radiation_field_device_view(
        owner, owner->field.generation.required_generation,
        owner->line_jbar_cache.q_set_hash, owner->line_profile_id,
        owner->line_profile_hash_storage, production_mirror, out, report);
}

void gpu_radiation_field_production_release(void)
{
    if (production_mirror) gpu_radiation_field_destroy(production_mirror);
    production_mirror = NULL;
    production_owner = NULL;
}
