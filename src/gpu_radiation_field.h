#ifndef LUMINA_GPU_RADIATION_FIELD_H
#define LUMINA_GPU_RADIATION_FIELD_H

#include "gpu_radiation_field_contract.h"
#include "radiation_field.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GPU_RF_COMPONENT_COUNT 9

typedef enum {
    GPU_RF_POISON_NONE = 0,
    GPU_RF_POISON_CPU_CHANGED = 1,
    GPU_RF_POISON_LINE_ID_SHUFFLE = 2,
    GPU_RF_POISON_PARTIAL_COPY = 3,
    GPU_RF_POISON_REPORTED_BYTES = 4
} GpuRadiationFieldPoison;

typedef struct {
    uint64_t requested[GPU_RF_COMPONENT_COUNT];
    uint64_t succeeded[GPU_RF_COMPONENT_COUNT];
    uint64_t cache_upload_bytes;
    uint64_t total_upload_bytes;
    uint64_t committed_bytes;
    uint64_t discarded_candidate_bytes;
    uint64_t peak_live_device_bytes;
    uint64_t upload_serial;
    uint64_t generation;
    size_t n_shells;
    size_t n_bins;
    size_t n_lines;
    uint32_t field_validity_size;
    uint32_t line_validity_size;
    double event_elapsed_seconds;
    GpuRadiationFieldStatus status;
    char secondary_diagnostics[256];
} GpuRadiationFieldReport;

typedef struct GpuRadiationFieldMirror GpuRadiationFieldMirror;

/* Read-only device descriptor published only from a READY mirror.  Consumers
 * must obtain it for every generation; no raw mirror buffer accessor exists. */
typedef struct {
    const double *frequency_bin_edges;
    const double *J_nu;
    const RadiationFieldValidityState *field_validity;
    const uint64_t *line_id;
    const double *line_jbar;
    const LineJbarValidityState *line_validity;
    const uint64_t *line_count;
    const double *line_se;
    uint64_t generation;
    size_t n_shells;
    size_t n_bins;
    size_t n_lines;
} GpuRadiationFieldDeviceView;

GpuRadiationFieldMirror *gpu_radiation_field_create(void);
GpuRadiationFieldStatus gpu_radiation_field_sync(
    const RadiationFieldOwner *owner, double expected_epoch,
    size_t expected_n_shells, uint64_t expected_generation,
    const char *expected_q_set_hash, uint64_t expected_profile_id,
    const char *expected_profile_hash, GpuRadiationFieldMirror *mirror,
    GpuRadiationFieldReport *report, void *cuda_stream,
    GpuRadiationFieldPoison poison);
GpuRadiationFieldStatus gpu_radiation_field_require_ready(
    const RadiationFieldOwner *owner, uint64_t expected_generation,
    const GpuRadiationFieldMirror *mirror, GpuRadiationFieldReport *report);
GpuRadiationFieldStatus gpu_radiation_field_device_view(
    const RadiationFieldOwner *owner, uint64_t expected_generation,
    const char *expected_q_set_hash, uint64_t expected_profile_id,
    const char *expected_profile_hash,
    const GpuRadiationFieldMirror *mirror,
    GpuRadiationFieldDeviceView *out, GpuRadiationFieldReport *report);
GpuRadiationFieldStatus gpu_radiation_field_reset(
    const RadiationFieldOwner *owner, uint64_t required_generation,
    GpuRadiationFieldMirror *mirror, GpuRadiationFieldReport *report,
    void *cuda_stream);
void gpu_radiation_field_free(GpuRadiationFieldMirror *mirror,
                              GpuRadiationFieldReport *report);
void gpu_radiation_field_destroy(GpuRadiationFieldMirror *mirror);
const GpuRadiationFieldCounters *gpu_radiation_field_counters(
    const GpuRadiationFieldMirror *mirror);
GpuRadiationFieldState gpu_radiation_field_state(
    const GpuRadiationFieldMirror *mirror);

#ifdef __cplusplus
}
#endif

#endif
