#ifndef LUMINA_GPU_PHYSICS_CONTRACT_H
#define LUMINA_GPU_PHYSICS_CONTRACT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GPU_PHYSICS_VALID = 0,
    GPU_PHYSICS_EXACT_ZERO = 1,
    GPU_PHYSICS_UNSAMPLED = 2,
    GPU_PHYSICS_STALE = 3,
    GPU_PHYSICS_OOG = 4,
    GPU_PHYSICS_MISS = 5,
    GPU_PHYSICS_PROFILE_MISMATCH = 6,
    GPU_PHYSICS_QSET_MISMATCH = 7,
    GPU_PHYSICS_NONFINITE = 8
} GpuPhysicsValidity;

typedef enum {
    GPU_PHYSICS_PASS = 0,
    GPU_PHYSICS_USAGE_OR_SCHEMA = 2,
    GPU_PHYSICS_BLOCKED = 3,
    GPU_PHYSICS_METRIC_FAIL = 4,
    GPU_PHYSICS_FORBIDDEN_FALLBACK = 5
} GpuPhysicsVerdict;

typedef struct {
    uint64_t required_generation, gpu_generation, line_generation;
    uint64_t bf_cells_attempted, bf_cells_published;
    uint64_t bb_cells_attempted, bb_cells_published;
    uint64_t opacity_cells_attempted, opacity_cells_published;
    uint64_t emissivity_cells_attempted, emissivity_cells_published;
    uint64_t line_cache_lookups, line_cache_hits, exact_zero_hits;
    uint64_t blocked_unsampled, blocked_stale, blocked_oog, blocked_miss;
    uint64_t blocked_profile, blocked_qset, blocked_nonfinite;
    uint64_t blocked_generation;
    uint64_t coarse_reintegration_attempts, fine_grid_attempts;
    uint64_t legacy_scalar_reads, cpu_fallback_attempts;
    uint64_t zero_fallback_attempts, floor_attempts;
    uint64_t partial_publish_attempts, physical_launches, blocked_launches;
    uint64_t cpu_gpu_bf_compared, cpu_gpu_bb_compared;
    uint64_t cpu_gpu_opacity_compared, cpu_gpu_emissivity_compared;
    uint64_t rng_draws_cpu, rng_draws_gpu;
} GpuPhysicsCounters;

typedef struct {
    int bf_cpu_oracle_pass, bb_cpu_oracle_pass;
    int bf_negative_controls_pass, bb_negative_controls_pass;
    int opacity_cpu_oracle_pass, opacity_negative_controls_pass;
    int emissivity_cpu_oracle_pass, emissivity_negative_controls_pass;
    int no_fallback_static_and_runtime_pass;
    int a2_12_lifecycle_pass, eligible_gpu_evidence_pass;
} GpuPhysicsGateInputs;

typedef enum {
    GPU_PHYSICS_N13_BF_EDGE = 0, GPU_PHYSICS_N13_BF_STIM,
    GPU_PHYSICS_N13_BB_LINE_ID, GPU_PHYSICS_N13_BB_STALE,
    GPU_PHYSICS_N13_COARSE_FALLBACK, GPU_PHYSICS_N13_FINE_GRID,
    GPU_PHYSICS_N13_HALF_BF, GPU_PHYSICS_N13_HALF_BB,
    GPU_PHYSICS_N14_STIM_OFF, GPU_PHYSICS_N14_SIGN_CLAMP,
    GPU_PHYSICS_N14_CHANNEL_DROP, GPU_PHYSICS_N14_MEASURE_ALIAS,
    GPU_PHYSICS_N15_DEST_PERMUTE, GPU_PHYSICS_N15_CHANNEL_DROP,
    GPU_PHYSICS_N15_PLANCK_FALLBACK, GPU_PHYSICS_N15_RNG_COUNT,
    GPU_PHYSICS_POISON_COUNT
} GpuPhysicsPoison;

void gpu_physics_counters_init(GpuPhysicsCounters *counters,
                               uint64_t required_generation);
int gpu_physics_record_validity(GpuPhysicsCounters *counters,
                                GpuPhysicsValidity validity);
int gpu_physics_forbidden_attempts_zero(const GpuPhysicsCounters *counters);
GpuPhysicsVerdict gpu_physics_a2_13_verdict(const GpuPhysicsGateInputs *inputs);
GpuPhysicsVerdict gpu_physics_a2_14_verdict(const GpuPhysicsGateInputs *inputs);
GpuPhysicsVerdict gpu_physics_a2_15_verdict(const GpuPhysicsGateInputs *inputs);
const char *gpu_physics_verdict_name(GpuPhysicsVerdict verdict);
const char *gpu_physics_poison_marker(GpuPhysicsPoison poison);
int gpu_physics_poison_child_rc(GpuPhysicsPoison poison);

#ifdef __cplusplus
}
#endif
#endif
