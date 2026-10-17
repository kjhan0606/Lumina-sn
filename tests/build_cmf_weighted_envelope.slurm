#!/bin/bash
#SBATCH --job-name=CMF_WEIGHTED_BUILD
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/weighted-build-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/weighted-build-%j.err
set -euo pipefail
umask 027
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
we_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
we_root=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/weighted-build-${SLURM_JOB_ID:?}
mkdir "$we_root" "$we_root/source"
cd "$we_src"
we_files=(src/cmf_exact_multigpu.cu src/cmf_exact_multigpu.h src/cmf_exact_sliding.c
 src/cmf_exact_sliding.h src/cmf_error_envelope.c src/cmf_error_envelope.h
 src/cmf_cell_geometry.h src/cmf_anderson.h src/cmf_anderson_multi.h tests/cmf_operator_norm_probe.cu
 tests/cmf_error_envelope_selftest.c tests/cmf_exact_multigpu_selftest.cu
 tests/cmf_emissivity_selftest.cu)
sha256sum "${we_files[@]}" > "$we_root/source.sha256"
git rev-parse HEAD > "$we_root/commit.txt"
git diff --binary -- src tests > "$we_root/change.patch"
cp --parents "${we_files[@]}" "$we_root/source/"
cp tests/build_cmf_weighted_envelope.slurm "$we_root/runner.slurm"
cd "$we_root/source"
sha256sum -c "$we_root/source.sha256" > "$we_root/source_verify.log"
gcc -O2 -std=c11 -Wall -Wextra -Werror -Isrc tests/cmf_error_envelope_selftest.c \
 src/cmf_error_envelope.c -lm -o "$we_root/cpu_selftest"
"$we_root/cpu_selftest" > "$we_root/cpu_selftest.log" 2> "$we_root/cpu_selftest.err"
we_flags=(-O2 -arch=sm_89 -std=c++14 -Xcompiler -fopenmp -DLUMINA_CMF_EXACT_MANAGED_MEMORY=1 -Isrc)
nvcc "${we_flags[@]}" -DCMF_OPERATOR_WEIGHTED_SEED_TEST tests/cmf_operator_norm_probe.cu \
 src/cmf_exact_sliding.c src/cmf_error_envelope.c -lm -o "$we_root/probe" > "$we_root/probe_build.log" 2> "$we_root/probe_build.err"
for we_test in cmf_exact_multigpu_selftest cmf_emissivity_selftest;do
 nvcc "${we_flags[@]}" "tests/$we_test.cu" src/cmf_exact_multigpu.cu \
  src/cmf_exact_sliding.c src/cmf_error_envelope.c -lm -o "$we_root/$we_test" > "$we_root/$we_test.build.log" 2> "$we_root/$we_test.build.err"
done
sha256sum -c "$we_root/source.sha256" > "$we_root/source_after_verify.log"
cd "$we_root"
sha256sum probe cpu_selftest cmf_exact_multigpu_selftest cmf_emissivity_selftest > binary.sha256
date -Is > finished.txt
cat cpu_selftest.log
