#ifndef LUMINA_GPU_RADIATION_FIELD_CONTRACT_H
#define LUMINA_GPU_RADIATION_FIELD_CONTRACT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GPU_RF_OK = 0,
    GPU_RF_DISABLED = 1,
    GPU_RF_STALE_CPU = 2,
    GPU_RF_STALE_LINE = 3,
    GPU_RF_CPU_GPU_GENERATION_MISMATCH = 4,
    GPU_RF_SHAPE_OR_HASH_MISMATCH = 5,
    GPU_RF_LINE_ID_MISMATCH = 6,
    GPU_RF_PROFILE_OR_QSET_MISMATCH = 7,
    GPU_RF_INVALID_CELL = 8,
    GPU_RF_ALLOCATION_FAILURE = 9,
    GPU_RF_PARTIAL_UPLOAD = 10,
    GPU_RF_COPY_FAILURE = 11,
    GPU_RF_EVENT_FAILURE = 12,
    GPU_RF_NOT_READY = 13,
    GPU_CPU_CHANGED_DURING_UPLOAD = 14,
    GPU_RF_CUDA_FAILURE = 15,
    GPU_RF_UPLOAD_BYTES_MISMATCH = 16,
    GPU_RESET_GENERATION_MISMATCH = 17,
    GPU_RATE_NOT_MIGRATED = 18,
    GPU_OPACITY_NOT_MIGRATED = 19,
    GPU_EMISSIVITY_NOT_MIGRATED = 20,
    BLOCKED_GPU_FALLBACK_FORBIDDEN = 21,
    GPU_RF_STATUS_COUNT = 22
} GpuRadiationFieldStatus;

typedef enum {
    GPU_RF_EMPTY = 0,
    GPU_RF_DIRTY = 1,
    GPU_RF_UPLOADING = 2,
    GPU_RF_READY = 3,
    GPU_RF_FAILED = 4
} GpuRadiationFieldState;

typedef struct {
    uint64_t sync_attempts;
    uint64_t sync_commits;
    uint64_t sync_failed_attempts;
    uint64_t reset_count;
    uint64_t free_count;
    uint64_t ready_checks;
    uint64_t ready_passes;
    uint64_t ready_failures;
    uint64_t launch_attempts;
    uint64_t blocked_launches;
    uint64_t physical_launches;
    uint64_t stale_cpu_failures;
    uint64_t stale_line_failures;
    uint64_t cpu_gpu_generation_failures;
    uint64_t shape_hash_failures;
    uint64_t line_id_failures;
    uint64_t profile_qset_failures;
    uint64_t invalid_field_cells;
    uint64_t invalid_line_cells;
    uint64_t allocation_failures;
    uint64_t copy_failures;
    uint64_t event_failures;
    uint64_t partial_upload_failures;
    uint64_t fallback_attempts;
    uint64_t zero_substitution_attempts;
    uint64_t sync_root_cause[GPU_RF_STATUS_COUNT];
} GpuRadiationFieldCounters;

const char *gpu_rf_status_name(GpuRadiationFieldStatus status);
void gpu_rf_counters_init(GpuRadiationFieldCounters *counters);
void gpu_rf_record_sync_commit(GpuRadiationFieldCounters *counters);
void gpu_rf_record_sync_failure(GpuRadiationFieldCounters *counters,
                                GpuRadiationFieldStatus root_cause,
                                uint64_t invalid_field_cells,
                                uint64_t invalid_line_cells);
void gpu_rf_record_ready(GpuRadiationFieldCounters *counters, int passed);
void gpu_rf_record_blocked_launch(GpuRadiationFieldCounters *counters);
void gpu_rf_record_physical_launch(GpuRadiationFieldCounters *counters);
int gpu_rf_counters_conserve(const GpuRadiationFieldCounters *counters);
int gpu_rf_block_unmigrated(GpuRadiationFieldCounters *counters,
                            GpuRadiationFieldStatus status,
                            const char *site);

#ifdef __cplusplus
}
#endif

#endif
