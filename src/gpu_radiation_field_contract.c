#include "gpu_radiation_field_contract.h"

#include <stdio.h>
#include <string.h>

static const char *const gpu_rf_status_names[GPU_RF_STATUS_COUNT] = {
    "GPU_RF_OK", "GPU_RF_DISABLED", "GPU_RF_STALE_CPU",
    "GPU_RF_STALE_LINE", "GPU_RF_CPU_GPU_GENERATION_MISMATCH",
    "GPU_RF_SHAPE_OR_HASH_MISMATCH", "GPU_RF_LINE_ID_MISMATCH",
    "GPU_RF_PROFILE_OR_QSET_MISMATCH", "GPU_RF_INVALID_CELL",
    "GPU_RF_ALLOCATION_FAILURE", "GPU_RF_PARTIAL_UPLOAD",
    "GPU_RF_COPY_FAILURE", "GPU_RF_EVENT_FAILURE", "GPU_RF_NOT_READY",
    "GPU_CPU_CHANGED_DURING_UPLOAD", "GPU_RF_CUDA_FAILURE",
    "GPU_RF_UPLOAD_BYTES_MISMATCH", "GPU_RESET_GENERATION_MISMATCH",
    "GPU_RATE_NOT_MIGRATED", "GPU_OPACITY_NOT_MIGRATED",
    "GPU_EMISSIVITY_NOT_MIGRATED", "BLOCKED_GPU_FALLBACK_FORBIDDEN"
};

const char *gpu_rf_status_name(GpuRadiationFieldStatus status)
{
    if (status < GPU_RF_OK || status >= GPU_RF_STATUS_COUNT)
        return "GPU_RF_UNKNOWN_STATUS";
    return gpu_rf_status_names[status];
}

void gpu_rf_counters_init(GpuRadiationFieldCounters *counters)
{
    if (counters) memset(counters, 0, sizeof(*counters));
}

void gpu_rf_record_sync_commit(GpuRadiationFieldCounters *counters)
{
    if (!counters) return;
    counters->sync_attempts++;
    counters->sync_commits++;
}

void gpu_rf_record_sync_failure(GpuRadiationFieldCounters *counters,
                                GpuRadiationFieldStatus root_cause,
                                uint64_t invalid_field_cells,
                                uint64_t invalid_line_cells)
{
    if (!counters) return;
    counters->sync_attempts++;
    counters->sync_failed_attempts++;
    if (root_cause > GPU_RF_OK && root_cause < GPU_RF_STATUS_COUNT)
        counters->sync_root_cause[root_cause]++;
    switch (root_cause) {
    case GPU_RF_STALE_CPU: counters->stale_cpu_failures++; break;
    case GPU_RF_STALE_LINE: counters->stale_line_failures++; break;
    case GPU_RF_CPU_GPU_GENERATION_MISMATCH:
        counters->cpu_gpu_generation_failures++; break;
    case GPU_RF_SHAPE_OR_HASH_MISMATCH: counters->shape_hash_failures++; break;
    case GPU_RF_LINE_ID_MISMATCH: counters->line_id_failures++; break;
    case GPU_RF_PROFILE_OR_QSET_MISMATCH:
        counters->profile_qset_failures++; break;
    case GPU_RF_INVALID_CELL:
        counters->invalid_field_cells += invalid_field_cells;
        counters->invalid_line_cells += invalid_line_cells;
        break;
    case GPU_RF_ALLOCATION_FAILURE: counters->allocation_failures++; break;
    case GPU_RF_PARTIAL_UPLOAD: counters->partial_upload_failures++; break;
    case GPU_RF_COPY_FAILURE: counters->copy_failures++; break;
    case GPU_RF_EVENT_FAILURE: counters->event_failures++; break;
    default: break;
    }
}

void gpu_rf_record_ready(GpuRadiationFieldCounters *counters, int passed)
{
    if (!counters) return;
    counters->ready_checks++;
    if (passed) counters->ready_passes++;
    else counters->ready_failures++;
}

void gpu_rf_record_blocked_launch(GpuRadiationFieldCounters *counters)
{
    if (!counters) return;
    counters->launch_attempts++;
    counters->blocked_launches++;
}

void gpu_rf_record_physical_launch(GpuRadiationFieldCounters *counters)
{
    if (!counters) return;
    counters->launch_attempts++;
    counters->physical_launches++;
}

int gpu_rf_counters_conserve(const GpuRadiationFieldCounters *counters)
{
    uint64_t roots = 0;
    size_t i;
    if (!counters) return 0;
    for (i = 1; i < GPU_RF_STATUS_COUNT; ++i)
        roots += counters->sync_root_cause[i];
    return counters->sync_attempts ==
               counters->sync_commits + counters->sync_failed_attempts &&
           counters->sync_failed_attempts == roots &&
           counters->ready_checks ==
               counters->ready_passes + counters->ready_failures &&
           counters->launch_attempts ==
               counters->physical_launches + counters->blocked_launches;
}

int gpu_rf_block_unmigrated(GpuRadiationFieldCounters *counters,
                            GpuRadiationFieldStatus status,
                            const char *site)
{
    if (status != GPU_RATE_NOT_MIGRATED &&
        status != GPU_OPACITY_NOT_MIGRATED &&
        status != GPU_EMISSIVITY_NOT_MIGRATED)
        status = GPU_RF_NOT_READY;
    gpu_rf_record_blocked_launch(counters);
    fprintf(stderr,
            "[A2-12][BLOCKED] status=%s site=%s physical_launches=0\n",
            gpu_rf_status_name(status), site ? site : "unknown");
    return -(int)status;
}
