#include "gpu_radiation_field_contract.h"

#include <stdio.h>

int main(void)
{
    GpuRadiationFieldCounters c;
    gpu_rf_counters_init(&c);
    gpu_rf_record_sync_commit(&c);
    gpu_rf_record_sync_failure(&c, GPU_RF_PARTIAL_UPLOAD, 0, 0);
    gpu_rf_record_ready(&c, 1);
    gpu_rf_record_ready(&c, 0);
    gpu_rf_record_blocked_launch(&c);
    gpu_rf_record_physical_launch(&c);
    if (c.sync_attempts != 2 || c.sync_commits != 1 ||
        c.sync_failed_attempts != 1 || c.partial_upload_failures != 1 ||
        c.copy_failures != 0 || c.sync_root_cause[GPU_RF_PARTIAL_UPLOAD] != 1 ||
        !gpu_rf_counters_conserve(&c)) {
        fprintf(stderr, "FAIL_COUNTER_NONCONSERVATION\n");
        return 1;
    }
    puts("A2_12_CONTRACT_SELFTEST PASS conservation=PASS root_cause_priority=PASS");
    return 0;
}
