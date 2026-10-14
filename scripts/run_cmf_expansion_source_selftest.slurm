#!/bin/bash
#SBATCH --job-name=CMF_SOURCE_PHYSICS
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/expansion-source-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/expansion-source-%j.err
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
probe_repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015
probe_src=${1:-$probe_repo/src}
probe_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/expansion-source-${SLURM_JOB_ID}
mkdir "$probe_out"
sha256sum "$probe_repo/tests/cmf_expansion_source_selftest.c" \
 "$probe_src/cmf_exact_sliding.c" "$probe_src/cmf_exact_sliding.h" \
 "$probe_src/cmf_error_envelope.c" "$probe_src/cmf_cell_geometry.h" \
 "$probe_repo/tests/cmf_exact_sliding_selftest.c" \
 "$probe_repo/tests/cmf_error_envelope_selftest.c" \
 "$probe_repo/scripts/run_cmf_expansion_source_selftest.slurm" > "$probe_out/source.sha256"
gcc -O2 -std=gnu11 -fopenmp -I"$probe_src" \
 "$probe_repo/tests/cmf_expansion_source_selftest.c" \
 "$probe_src/cmf_exact_sliding.c" "$probe_src/cmf_error_envelope.c" -lm -o "$probe_out/test"
"$probe_out/test" "${2:-report}" > "$probe_out/test.log"
if [[ ${2:-report} == require-physical ]]; then
 gcc -O2 -std=gnu11 -fopenmp -I"$probe_src" \
  "$probe_repo/tests/cmf_exact_sliding_selftest.c" \
  "$probe_src/cmf_exact_sliding.c" "$probe_src/cmf_error_envelope.c" -lm -o "$probe_out/regression"
 "$probe_out/regression" > "$probe_out/regression.log" 2> "$probe_out/regression.err"
 gcc -O2 -std=gnu11 -I"$probe_src" "$probe_repo/tests/cmf_error_envelope_selftest.c" \
  "$probe_src/cmf_error_envelope.c" -lm -o "$probe_out/envelope"
 "$probe_out/envelope" > "$probe_out/envelope.log" 2> "$probe_out/envelope.err"
fi
if [[ ${3:-cpu} == gpu ]]; then
 module load cuda/13.0.2
 sha256sum "$probe_src/cmf_exact_multigpu.cu" "$probe_src/cmf_exact_multigpu.h" >> "$probe_out/source.sha256"
 nvcc -O2 -arch=sm_86 -std=c++14 -DCMF_TEST_CUDA -Xcompiler -fopenmp \
  -I"$probe_src" "$probe_repo/tests/cmf_expansion_source_selftest.c" \
  "$probe_src/cmf_exact_sliding.c" "$probe_src/cmf_error_envelope.c" \
  "$probe_src/cmf_exact_multigpu.cu" -lm -o "$probe_out/gpu-test" \
  > "$probe_out/gpu-build.log" 2>&1
 "$probe_out/gpu-test" require-physical > "$probe_out/gpu-test.log" 2> "$probe_out/gpu-test.err"
fi
sha256sum -c "$probe_out/source.sha256" > "$probe_out/verify.log"
echo "result=$probe_out"
