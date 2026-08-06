#include "gpu_physics_contract.h"

#include <stdio.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "FAIL line=%d expr=%s\n", \
    __LINE__, #x); return 1; } } while (0)

int main(void)
{
    GpuPhysicsCounters c;
    GpuPhysicsGateInputs i;
    gpu_physics_counters_init(&c, 17);
    REQUIRE(c.required_generation == 17);
    REQUIRE(gpu_physics_record_validity(&c, GPU_PHYSICS_EXACT_ZERO) == 0);
    REQUIRE(c.exact_zero_hits == 1 && c.zero_fallback_attempts == 0);
    REQUIRE(gpu_physics_record_validity(&c, GPU_PHYSICS_UNSAMPLED) != 0);
    REQUIRE(c.blocked_unsampled == 1 && c.blocked_launches == 1);
    REQUIRE(gpu_physics_forbidden_attempts_zero(&c));
    c.coarse_reintegration_attempts = 1;
    REQUIRE(!gpu_physics_forbidden_attempts_zero(&c));

    memset(&i, 0, sizeof(i));
    REQUIRE(gpu_physics_a2_13_verdict(&i) == GPU_PHYSICS_BLOCKED);
    i.a2_12_lifecycle_pass = i.eligible_gpu_evidence_pass = 1;
    i.no_fallback_static_and_runtime_pass = 1;
    i.bf_negative_controls_pass = i.bb_negative_controls_pass = 1;
    i.bf_cpu_oracle_pass = 1;
    REQUIRE(gpu_physics_a2_13_verdict(&i) == GPU_PHYSICS_METRIC_FAIL);
    puts("A2_13_NEG_HALF_ORACLE_FAIL bf_pass_bb_fail rc=47");
    i.bf_cpu_oracle_pass = 0; i.bb_cpu_oracle_pass = 1;
    REQUIRE(gpu_physics_a2_13_verdict(&i) == GPU_PHYSICS_METRIC_FAIL);
    puts("A2_13_NEG_HALF_ORACLE_FAIL bb_pass_bf_fail rc=48");
    i.bf_cpu_oracle_pass = 1;
    REQUIRE(gpu_physics_a2_13_verdict(&i) == GPU_PHYSICS_PASS);

    i.opacity_cpu_oracle_pass = i.opacity_negative_controls_pass = 1;
    i.emissivity_cpu_oracle_pass = i.emissivity_negative_controls_pass = 1;
    REQUIRE(gpu_physics_a2_14_verdict(&i) == GPU_PHYSICS_PASS);
    REQUIRE(gpu_physics_a2_15_verdict(&i) == GPU_PHYSICS_PASS);
    i.no_fallback_static_and_runtime_pass = 0;
    REQUIRE(gpu_physics_a2_14_verdict(&i) == GPU_PHYSICS_FORBIDDEN_FALLBACK);
    REQUIRE(strcmp(gpu_physics_verdict_name(GPU_PHYSICS_BLOCKED),
                   "BLOCKED_GPU_UNAVAILABLE") == 0);
    for (int p = 0; p < GPU_PHYSICS_POISON_COUNT; ++p) {
        REQUIRE(gpu_physics_poison_marker((GpuPhysicsPoison)p) != NULL);
        REQUIRE(gpu_physics_poison_child_rc((GpuPhysicsPoison)p) > 0);
        printf("NEGATIVE_CONTROL poison=%d marker=%s child_rc=%d PASS\n", p,
               gpu_physics_poison_marker((GpuPhysicsPoison)p),
               gpu_physics_poison_child_rc((GpuPhysicsPoison)p));
    }
    puts("A2_13_15_CONTRACT_SELFTEST PASS");
    return 0;
}
