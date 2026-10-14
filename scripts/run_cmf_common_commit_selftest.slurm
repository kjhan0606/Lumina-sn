#!/bin/bash
#SBATCH --job-name=CMF_PROFILE_COMMIT
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-profile-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-profile-%j.err
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while read -r profile_key; do unset "$profile_key"; done < <(compgen -v LUMINA_ || true)
profile_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015
profile_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-profile-${SLURM_JOB_ID}
mkdir "$profile_out"
cd "$profile_src"
profile_files=(src/lumina_cmfgen.c src/line_jbar.c src/radiation_field.c src/seed_capability.c)
sha256sum "${profile_files[@]}" src/line_jbar.h src/lumina_cmfgen.h \
 tests/cmf_common_commit_selftest.c scripts/run_cmf_common_commit_selftest.slurm > "$profile_out/source.sha256"
gcc -O1 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
 tests/cmf_common_commit_selftest.c "${profile_files[@]}" -Wl,--gc-sections -lm \
 -o "$profile_out/test" > "$profile_out/build.log" 2>&1
sha256sum -c "$profile_out/source.sha256" > "$profile_out/verify.log"
cd "$profile_out"
./test > test.log 2> test.err
cd "$profile_src"
gcc -O2 -std=gnu11 -Isrc tests/a2_06_line_jbar_selftest.c src/line_jbar.c -lm \
 -o "$profile_out/line-test" > "$profile_out/line-build.log" 2>&1
"$profile_out/line-test" > "$profile_out/line.log" 2> "$profile_out/line.err"
sha256sum -c "$profile_out/source.sha256" >> "$profile_out/verify.log"
printf 'result=%s\n' "$profile_out"
