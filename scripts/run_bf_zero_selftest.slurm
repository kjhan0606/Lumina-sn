#!/bin/bash
#SBATCH --job-name=CMF_BF_ZERO
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/bf-zero-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/bf-zero-%j.err
set -euo pipefail
umask 027
bf_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015
bf_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/bf-zero-${SLURM_JOB_ID}
bf_asset=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/atomic-stage-422722/cmfgen_sigma_bf.bin
bf_levels=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/model/levels.csv
mkdir "$bf_out"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$bf_src"
git rev-parse HEAD > "$bf_out/commit.txt"
sha256sum src/bf_rate_jnu.c src/bf_rate_jnu.h src/radiation_field.c \
 src/radiation_field.h src/line_jbar.c src/line_jbar.h src/lumina_frequency_grid.h \
 src/seed_capability.c src/seed_capability.h \
 tests/a2_05_bf_rate_selftest.c scripts/run_bf_zero_selftest.slurm > "$bf_out/source.sha256"
cp src/bf_rate_jnu.c tests/a2_05_bf_rate_selftest.c "$bf_out/"
sha256sum "$bf_asset" "$bf_levels" > "$bf_out/input.sha256"
gcc -O2 -std=c11 -Wall -Wextra -Isrc tests/a2_05_bf_rate_selftest.c \
 src/bf_rate_jnu.c src/radiation_field.c src/line_jbar.c src/seed_capability.c \
 -lm -o "$bf_out/selftest" \
 > "$bf_out/build.log" 2>&1
bf_rc=0
"$bf_out/selftest" "$bf_asset" "$bf_levels" > "$bf_out/test.log" 2> "$bf_out/test.err" || bf_rc=$?
sha256sum -c "$bf_out/source.sha256" > "$bf_out/source_verify.log"
printf 'selftest_rc=%s\n' "$bf_rc"
exit "$bf_rc"
