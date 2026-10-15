#!/bin/bash
#SBATCH --job-name=CMF_GPU_SCHEDULE
#SBATCH --partition=a10,a40,a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --exclude=syn102,syn03
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/schedule-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/schedule-%j.err
set -euo pipefail
umask 027
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while read -r schedule_key; do unset "$schedule_key"; done < <(compgen -v LUMINA_ || true)
schedule_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
schedule_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/schedule-${SLURM_JOB_ID}
mkdir "$schedule_out" "$schedule_out/source"
cd "$schedule_src"
git rev-parse HEAD > "$schedule_out/commit.txt"
schedule_files=(src/cmf_anderson.h src/cmf_cell_geometry.h src/cmf_exact_sliding.h
 src/cmf_exact_sliding.c src/cmf_error_envelope.h src/cmf_error_envelope.c
 src/cmf_exact_multigpu.h src/cmf_exact_multigpu.cu tests/cmf_schedule_selftest.cu
 tests/cmf_exact_multigpu_selftest.cu scripts/run_cmf_schedule_selftest.slurm)
cp --parents "${schedule_files[@]}" "$schedule_out/source/"
cp /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-initial-423540/work/initial_predictor.csv "$schedule_out/geometry.csv"
sha256sum "$schedule_out/geometry.csv" > "$schedule_out/input.sha256"
cd "$schedule_out/source"
sha256sum "${schedule_files[@]}" > "$schedule_out/source.sha256"
schedule_units=(src/cmf_exact_sliding.c src/cmf_error_envelope.c src/cmf_exact_multigpu.cu)
nvcc -O2 -arch=native -std=c++14 -Xcompiler -fopenmp -Isrc \
 tests/cmf_schedule_selftest.cu "${schedule_units[@]}" -lm -o "$schedule_out/schedule-test" > "$schedule_out/build.log" 2>&1
"$schedule_out/schedule-test" "$schedule_out/geometry.csv" > "$schedule_out/test.log" 2> "$schedule_out/test.err"
nvcc -O2 -arch=native -std=c++14 -Xcompiler -fopenmp -Isrc \
 tests/cmf_exact_multigpu_selftest.cu "${schedule_units[@]}" -lm -o "$schedule_out/exact-test" >> "$schedule_out/build.log" 2>&1
CMF_MGPU_TEST_DEVICES=1 "$schedule_out/exact-test" > "$schedule_out/exact.log" 2> "$schedule_out/exact.err"
sha256sum -c "$schedule_out/source.sha256" > "$schedule_out/source_verify.log"
sha256sum -c "$schedule_out/input.sha256" > "$schedule_out/input_verify.log"
