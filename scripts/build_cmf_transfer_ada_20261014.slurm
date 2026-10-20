#!/bin/bash
#SBATCH --job-name=CMF_TRANSFER_BUILD
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/transfer-build-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/transfer-build-%j.err
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
build_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-angular-20261014
build_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/transfer-build-${SLURM_JOB_ID}
mkdir "$build_out"
cd "$build_src"
git rev-parse HEAD > "$build_out/commit.txt"
[[ $(git rev-parse HEAD) == 80b54acd048f14b314aacf032ba5c6dd3bbaa3f9 ]] || exit 4
git diff --exit-code -- src
git diff --cached --exit-code -- src
git ls-files -z -- src | xargs -0 sha256sum > "$build_out/source.sha256"
{ hostname; date -Is; nvcc --version; gcc --version; } > "$build_out/environment.txt"
build_units=(
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
timeout 1100s nvcc -O2 -arch=sm_89 -std=c++14 -Xcompiler -fopenmp \
    -DLUMINA_HAS_CUDA_BF_GEMM -DLUMINA_CMF_EXACT_MANAGED_MEMORY=1 \
    -o "$build_out/lumina_cuda" "${build_units[@]}" -lm -lcublas -Xcompiler -fopenmp \
    > "$build_out/build.log" 2> "$build_out/build.stderr"
sha256sum -c "$build_out/source.sha256" > "$build_out/source_verify_after.log"
sha256sum "$build_out/lumina_cuda" > "$build_out/binary.sha256"
echo "build=$build_out; compilation only, no GPU execution"
