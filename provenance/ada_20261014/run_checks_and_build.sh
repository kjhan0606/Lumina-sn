#!/usr/bin/env bash
set -euo pipefail
[[ $(hostname) == LagEunha ]] || exit 2
export MODULEPATH=/opt/ohpc/pub/modulefiles
set +u
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
set -u
A=/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/ada_20261014
SRC=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-ada-20261014
B=$A/build1
MODEL=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/model
[[ ! -e "$B" ]] || exit 3
mkdir "$B"
exec 9>/gpfs/kjhan/lumina/.lageunha-gpu0.lock
flock -n 9 || exit 4
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 5
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$SRC"
xargs -d '\n' -a provenance/sprint_20261012/accel_manifest_paths.txt sha256sum > "$B/source.sha256"
{ hostname; date -Is; nvcc --version; gcc --version; } > "$B/environment.txt"
for mode in default managed; do
    extra=()
    [[ $mode == managed ]] && extra=(-DLUMINA_CMF_EXACT_MANAGED_MEMORY=1)
    timeout 300s nvcc -O2 -arch=sm_89 -std=c++14 -Xcompiler -fopenmp -Isrc \
        "${extra[@]}" -o "$B/bench_$mode" tests/cmf_exact_multigpu_reduced_bench.cu \
        src/cmf_exact_multigpu.cu src/cmf_exact_sliding.c src/cmf_error_envelope.c \
        -lm > "$B/bench_${mode}_build.log" 2>&1
    CMF_MGPU_REDUCED_MODE=one CMF_MGPU_REDUCED_EPOCH=1 \
    CMF_MGPU_REDUCED_OUTPUT="$B/result_$mode.bin" \
        timeout 300s "$B/bench_$mode" "$MODEL/geometry.csv" > "$B/bench_$mode.log" 2>&1
    grep 'SPLIT_RESULT PASS' "$B/bench_$mode.log"
done
cmp "$B/result_default.bin" "$B/result_managed.bin"
printf 'ALLOCATION_VARIANT_BYTE_PARITY_PASS\n' | tee "$B/parity.txt"
units=(
 src/lumina_cuda.cu src/lumina_bf_gemm.cu src/lumina_nlte_gemm.cu
 src/lumina_nlte_assemble.cu src/lumina_cmf_solve.cu src/cmf_exact_multigpu.cu
 src/gpu_radiation_field.cu src/gpu_radiation_field_contract.c
 src/gpu_physics_contract.c src/gpu_physics_kernels.cu src/gpu_opacity_kernels.cu
 src/gpu_emissivity_kernels.cu src/lumina_atomic.c src/lumina_plasma.c
 src/population_contract.c src/atomic_internal_energy.c src/opacity_publication.c
 src/emissivity_publication.c src/radeq_publication.c src/line_net_rate.c
 src/cmfgen_adiabatic.c src/nlte_population_candidate.c src/physics_comparison.c
 src/bf_rate_jnu.c src/radiation_field.c src/line_jbar.c src/cmf_exact_sliding.c
 src/cmf_error_envelope.c src/seed_capability.c src/lumina_element_wide.c
 src/lumina_cmfgen.c src/cmfgen_final_writer.c src/a210_coupled_newton.c
)
set +e
timeout 1200s nvcc -O2 -arch=sm_89 -std=c++14 -Xcompiler -fopenmp \
    -DLUMINA_HAS_CUDA_BF_GEMM -DLUMINA_CMF_EXACT_MANAGED_MEMORY=1 \
    -o "$B/lumina_cuda" "${units[@]}" -lm -lcublas -Xcompiler -fopenmp \
    > "$B/build.log" 2> "$B/build.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$B/build.rc"
(( rc == 0 )) || exit "$rc"
sha256sum -c "$B/source.sha256" > "$B/source_verify_after.log"
cd "$B"
sha256sum lumina_cuda > binary.sha256
printf 'ADA_BUILD_PASS\n'
