#include "gpu_physics_contract.h"

#include <string.h>

void gpu_physics_counters_init(GpuPhysicsCounters *counters,
                               uint64_t required_generation)
{
    if (!counters) return;
    memset(counters, 0, sizeof(*counters));
    counters->required_generation = required_generation;
}

int gpu_physics_record_validity(GpuPhysicsCounters *counters,
                                GpuPhysicsValidity validity)
{
    if (!counters) return -1;
    switch (validity) {
    case GPU_PHYSICS_VALID: return 0;
    case GPU_PHYSICS_EXACT_ZERO: counters->exact_zero_hits++; return 0;
    case GPU_PHYSICS_UNSAMPLED: counters->blocked_unsampled++; break;
    case GPU_PHYSICS_STALE: counters->blocked_stale++; break;
    case GPU_PHYSICS_OOG: counters->blocked_oog++; break;
    case GPU_PHYSICS_MISS: counters->blocked_miss++; break;
    case GPU_PHYSICS_PROFILE_MISMATCH: counters->blocked_profile++; break;
    case GPU_PHYSICS_QSET_MISMATCH: counters->blocked_qset++; break;
    case GPU_PHYSICS_NONFINITE: counters->blocked_nonfinite++; break;
    default: counters->blocked_nonfinite++; break;
    }
    counters->blocked_launches++;
    return -1;
}

int gpu_physics_forbidden_attempts_zero(const GpuPhysicsCounters *c)
{
    if (!c) return 0;
    return c->coarse_reintegration_attempts == 0 &&
           c->fine_grid_attempts == 0 && c->legacy_scalar_reads == 0 &&
           c->cpu_fallback_attempts == 0 && c->zero_fallback_attempts == 0 &&
           c->floor_attempts == 0 && c->partial_publish_attempts == 0;
}

static GpuPhysicsVerdict common_precondition(const GpuPhysicsGateInputs *i)
{
    if (!i) return GPU_PHYSICS_USAGE_OR_SCHEMA;
    if (!i->a2_12_lifecycle_pass || !i->eligible_gpu_evidence_pass)
        return GPU_PHYSICS_BLOCKED;
    if (!i->no_fallback_static_and_runtime_pass)
        return GPU_PHYSICS_FORBIDDEN_FALLBACK;
    return GPU_PHYSICS_PASS;
}

GpuPhysicsVerdict gpu_physics_a2_13_verdict(const GpuPhysicsGateInputs *i)
{
    GpuPhysicsVerdict pre = common_precondition(i);
    if (pre != GPU_PHYSICS_PASS) return pre;
    /* A2-13 is deliberately a conjunction.  A half-oracle PASS is impossible. */
    return i->bf_cpu_oracle_pass && i->bb_cpu_oracle_pass &&
           i->bf_negative_controls_pass && i->bb_negative_controls_pass
               ? GPU_PHYSICS_PASS : GPU_PHYSICS_METRIC_FAIL;
}

GpuPhysicsVerdict gpu_physics_a2_14_verdict(const GpuPhysicsGateInputs *i)
{
    GpuPhysicsVerdict pre = common_precondition(i);
    if (pre != GPU_PHYSICS_PASS) return pre;
    return i->opacity_cpu_oracle_pass && i->opacity_negative_controls_pass
               ? GPU_PHYSICS_PASS : GPU_PHYSICS_METRIC_FAIL;
}

GpuPhysicsVerdict gpu_physics_a2_15_verdict(const GpuPhysicsGateInputs *i)
{
    GpuPhysicsVerdict pre = common_precondition(i);
    if (pre != GPU_PHYSICS_PASS) return pre;
    return i->emissivity_cpu_oracle_pass && i->emissivity_negative_controls_pass
               ? GPU_PHYSICS_PASS : GPU_PHYSICS_METRIC_FAIL;
}

const char *gpu_physics_verdict_name(GpuPhysicsVerdict verdict)
{
    switch (verdict) {
    case GPU_PHYSICS_PASS: return "PASS";
    case GPU_PHYSICS_USAGE_OR_SCHEMA: return "USAGE_OR_SCHEMA";
    case GPU_PHYSICS_BLOCKED: return "BLOCKED_GPU_UNAVAILABLE";
    case GPU_PHYSICS_METRIC_FAIL: return "FAIL";
    case GPU_PHYSICS_FORBIDDEN_FALLBACK: return "FORBIDDEN_FALLBACK";
    default: return "UNKNOWN";
    }
}

static const char *const poison_markers[GPU_PHYSICS_POISON_COUNT] = {
    "A2_13_NEG_BF_EDGE_FAIL", "A2_13_NEG_BF_STIM_FAIL",
    "A2_13_NEG_BB_LINE_ID_FAIL", "A2_13_NEG_BB_STALE_FAIL",
    "A2_13_NEG_COARSE_FALLBACK_FAIL", "A2_13_NEG_FINE_GRID_FAIL",
    "A2_13_NEG_HALF_ORACLE_FAIL", "A2_13_NEG_HALF_ORACLE_FAIL",
    "A2_14_NEG_STIM_OFF_FAIL", "A2_14_NEG_SIGN_CLAMP_FAIL",
    "A2_14_NEG_CHANNEL_DROP_FAIL", "A2_14_NEG_MEASURE_ALIAS_FAIL",
    "A2_15_NEG_DEST_PERMUTE_FAIL", "A2_15_NEG_CHANNEL_DROP_FAIL",
    "A2_15_NEG_PLANCK_FALLBACK_FAIL", "A2_15_NEG_RNG_COUNT_FAIL"
};
static const int poison_rc[GPU_PHYSICS_POISON_COUNT] = {
    41, 42, 43, 44, 45, 46, 47, 48, 51, 52, 53, 54, 61, 62, 63, 64
};

const char *gpu_physics_poison_marker(GpuPhysicsPoison poison)
{
    if (poison < 0 || poison >= GPU_PHYSICS_POISON_COUNT) return NULL;
    return poison_markers[poison];
}

int gpu_physics_poison_child_rc(GpuPhysicsPoison poison)
{
    if (poison < 0 || poison >= GPU_PHYSICS_POISON_COUNT) return 2;
    return poison_rc[poison];
}
