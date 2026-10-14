#!/bin/bash
#SBATCH --job-name=CMF_COMMON_BUILD
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-build-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-build-%j.err
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
build_src=${2:-/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015}
build_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-build-${SLURM_JOB_ID}
mkdir "$build_out"
cd "$build_src"
build_variant=${1:-ada}
case "$build_variant" in
 ada) build_arch=(-arch=sm_89); build_managed=1 ;;
 cluster) build_arch=(-gencode arch=compute_80,code=sm_80
     -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
     -gencode arch=compute_90,code=sm_90); build_managed=0 ;;
 *) exit 2 ;;
esac
printf 'variant=%s managed=%s\n' "$build_variant" "$build_managed" > "$build_out/variant.txt"
git rev-parse HEAD > "$build_out/commit.txt"
git diff --exit-code -- src
git diff --cached --exit-code -- src
git ls-files -z -- src | xargs -0 sha256sum > "$build_out/source.sha256"
mkdir "$build_out/source"
git ls-files -z -- src | xargs -0 cp --parents -t "$build_out/source/"
cp scripts/build_cmf_common_field.slurm "$build_out/runner.slurm"
cd "$build_out/source"
sha256sum -c "$build_out/source.sha256" > "$build_out/snapshot_verify.log"
build_units=(
 src/lumina_cuda.cu src/lumina_bf_gemm.cu src/lumina_nlte_gemm.cu
 src/lumina_nlte_assemble.cu src/lumina_cmf_solve.cu src/cmf_exact_multigpu.cu
 src/cmf_observer.c src/cmf_observer.cu
 src/gpu_radiation_field.cu src/gpu_radiation_field_contract.c
 src/gpu_physics_contract.c src/gpu_physics_kernels.cu src/gpu_opacity_kernels.cu
 src/gpu_emissivity_kernels.cu src/lumina_atomic.c src/lumina_plasma.c
 src/population_contract.c src/atomic_internal_energy.c src/opacity_publication.c
 src/emissivity_publication.c src/radeq_publication.c src/line_net_rate.c
 src/cmfgen_adiabatic.c src/nlte_population_candidate.c src/physics_comparison.c
 src/bf_rate_jnu.c src/radiation_field.c src/line_jbar.c src/cmf_exact_sliding.c
 src/cmf_error_envelope.c src/cmf_time_moment.c src/cmf_time_field.c src/cmf_time_epoch.c src/seed_capability.c src/lumina_element_wide.c
 src/lumina_cmfgen.c src/cmfgen_final_writer.c src/a210_coupled_newton.c
)
nvcc -O2 "${build_arch[@]}" -std=c++14 -Xcompiler -fopenmp \
 -DLUMINA_HAS_CUDA_BF_GEMM -DLUMINA_CMF_EXACT_MANAGED_MEMORY="$build_managed" \
 -o "$build_out/lumina_cuda" "${build_units[@]}" -lm -lcublas -Xcompiler -fopenmp \
 > "$build_out/build.log" 2> "$build_out/build.stderr"
sha256sum -c "$build_out/source.sha256" > "$build_out/verify.log"
sha256sum "$build_out/lumina_cuda" > "$build_out/binary.sha256"
echo "build=$build_out; compilation only, GPU execution=0"
