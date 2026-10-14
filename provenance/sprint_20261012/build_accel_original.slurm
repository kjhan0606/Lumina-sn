#!/bin/bash
#SBATCH --job-name=LUMINA_SPRINT_ACCEL_BUILD
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --exclude=syn102,syn03
#SBATCH --no-requeue
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/accel/logs/%x_%j.log
# Sprint FREE_T accel build: O (DET-OBS-CAPTURE B2) nvcc recipe + src/a210_coupled_newton.c.  usage: sbatch build_accel.slurm BUILD_DIR_NAME
set -uo pipefail
A=/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/accel; SRC=$A/source; B=$A/${1:?build dir name}
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
source /opt/ohpc/admin/lmod/lmod/init/bash && module purge && module load gnu13/13.2.0 cuda/13.0.2 || { echo "[STOP] module"; exit 1; }
[[ ! -e $B ]] || { echo "BUILD_DIR_EXISTS $B"; exit 3; }
mkdir -p $B && cd $SRC || exit 9
cp $A/manifest_paths_accel.txt $B/manifest_paths.txt
xargs -a $B/manifest_paths.txt sha256sum > $B/source_sha_at_build.txt || exit 5
{ nvcc --version | tail -2; gcc --version | head -1; } > $B/compiler.txt
nice -n 10 nvcc -O2 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -std=c++14 -Xcompiler -fopenmp -DLUMINA_HAS_CUDA_BF_GEMM -o $B/lumina_cuda src/lumina_cuda.cu src/lumina_bf_gemm.cu src/lumina_nlte_gemm.cu src/lumina_nlte_assemble.cu src/lumina_cmf_solve.cu src/cmf_exact_multigpu.cu src/gpu_radiation_field.cu src/gpu_radiation_field_contract.c src/gpu_physics_contract.c src/gpu_physics_kernels.cu src/gpu_opacity_kernels.cu src/gpu_emissivity_kernels.cu src/lumina_atomic.c src/lumina_plasma.c src/population_contract.c src/atomic_internal_energy.c src/opacity_publication.c src/emissivity_publication.c src/radeq_publication.c src/line_net_rate.c src/cmfgen_adiabatic.c src/nlte_population_candidate.c src/physics_comparison.c src/bf_rate_jnu.c src/radiation_field.c src/line_jbar.c src/cmf_exact_sliding.c src/cmf_error_envelope.c src/seed_capability.c src/lumina_element_wide.c src/lumina_cmfgen.c src/cmfgen_final_writer.c src/a210_coupled_newton.c -lm -lcublas -Xcompiler -fopenmp > $B/build.log 2> $B/build.stderr
rc=$?; echo $rc > $B/build.rc
xargs -a $B/manifest_paths.txt sha256sum | diff -q - $B/source_sha_at_build.txt >/dev/null || { echo "TREE_CHANGED_DURING_BUILD"; echo 99 > $B/build.rc; exit 6; }
[[ $rc -eq 0 ]] && ( cd $B && sha256sum lumina_cuda ) > $B/binary.sha256
grep -iE "error|warning" $B/build.stderr | grep -E "a210_coupled_newton|lumina_plasma.c|radeq_publication.c" | head -40
echo "rc=$rc end=$(date -Is)"
exit $rc
