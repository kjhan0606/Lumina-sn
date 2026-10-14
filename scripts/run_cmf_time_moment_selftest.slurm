#!/bin/bash
#SBATCH --job-name=CMF_TIME_MOMENT_TEST
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/time-moment-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/time-moment-%j.err
set -euo pipefail
umask 027
moment_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
moment_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/time-moment-${SLURM_JOB_ID}
mkdir "$moment_out"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$moment_src"
git rev-parse HEAD > "$moment_out/commit.txt"
cp src/cmf_time_moment.c src/cmf_time_moment.h tests/cmf_time_moment_selftest.c \
 scripts/run_cmf_time_moment_selftest.slurm "$moment_out/"
sha256sum src/cmf_time_moment.c src/cmf_time_moment.h tests/cmf_time_moment_selftest.c \
 scripts/run_cmf_time_moment_selftest.slurm > "$moment_out/source.sha256"
gcc -O2 -std=c11 -Wall -Wextra -Werror -Isrc src/cmf_time_moment.c \
 tests/cmf_time_moment_selftest.c -lm -o "$moment_out/selftest" > "$moment_out/build.log" 2>&1
"$moment_out/selftest" > "$moment_out/test.log" 2> "$moment_out/test.err"
gcc -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc \
 src/cmf_time_moment.c tests/cmf_time_moment_selftest.c -lm -o "$moment_out/selftest_san" \
 > "$moment_out/build_san.log" 2>&1
ASAN_OPTIONS=detect_leaks=1 "$moment_out/selftest_san" > "$moment_out/san.log" 2> "$moment_out/san.err"
sha256sum -c "$moment_out/source.sha256" > "$moment_out/source_verify.log"
