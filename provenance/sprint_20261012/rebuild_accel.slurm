#!/usr/bin/env bash
# 기존 cuda_build3 컴파일 명령으로 보존된 Git 소스를 재빌드한다.
# 사용: sbatch rebuild_accel.slurm SOURCE_ROOT NEW_OUTPUT_DIR
#SBATCH --job-name=LUMINA_ACCEL_REPRO
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --exclude=syn102,syn03
#SBATCH --no-requeue
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/accel/logs/rebuild_%j.log
set -euo pipefail
SRC=${1:?SOURCE_ROOT required}
B=${2:?NEW_OUTPUT_DIR required}
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Slurm job required'; exit 1; }
case "$SRC" in /*) ;; *) echo 'SOURCE_ROOT must be absolute'; exit 2;; esac
case "$B" in /gpfs/kjhan/*|/scratch/kjhan/*) ;; *) echo 'Output must be under user scratch'; exit 2;; esac
[[ ! -e "$B" ]] || { echo "BUILD_DIR_EXISTS $B"; exit 3; }
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
mkdir -- "$B"
cd -- "$SRC"
cp provenance/sprint_20261012/accel_source_sha_at_build.txt "$B/source_expected.sha256"
sha256sum -c "$B/source_expected.sha256" > "$B/source_verify_before.log"
{ nvcc --version; gcc --version; } > "$B/compiler.txt"
set +e
nice -n 10 nvcc -O2 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -std=c++14 -Xcompiler -fopenmp -DLUMINA_HAS_CUDA_BF_GEMM -o "$B/lumina_cuda" src/lumina_cuda.cu src/lumina_bf_gemm.cu src/lumina_nlte_gemm.cu src/lumina_nlte_assemble.cu src/lumina_cmf_solve.cu src/cmf_exact_multigpu.cu src/gpu_radiation_field.cu src/gpu_radiation_field_contract.c src/gpu_physics_contract.c src/gpu_physics_kernels.cu src/gpu_opacity_kernels.cu src/gpu_emissivity_kernels.cu src/lumina_atomic.c src/lumina_plasma.c src/population_contract.c src/atomic_internal_energy.c src/opacity_publication.c src/emissivity_publication.c src/radeq_publication.c src/line_net_rate.c src/cmfgen_adiabatic.c src/nlte_population_candidate.c src/physics_comparison.c src/bf_rate_jnu.c src/radiation_field.c src/line_jbar.c src/cmf_exact_sliding.c src/cmf_error_envelope.c src/seed_capability.c src/lumina_element_wide.c src/lumina_cmfgen.c src/cmfgen_final_writer.c src/a210_coupled_newton.c -lm -lcublas -Xcompiler -fopenmp > "$B/build.log" 2> "$B/build.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$B/build.rc"
if ! sha256sum -c "$B/source_expected.sha256" > "$B/source_verify_after.log"; then
    printf '99\n' > "$B/build.rc"
    exit 99
fi
(( rc == 0 )) || exit "$rc"
cd -- "$B"
sha256sum lumina_cuda > binary.sha256
