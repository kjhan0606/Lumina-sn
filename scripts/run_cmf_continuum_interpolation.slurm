#!/bin/bash
#SBATCH --job-name=CMF_CONT_INTERPOLATION
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/continuum-interpolation-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/continuum-interpolation-%j.err
set -euo pipefail
umask 027
ci_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
ci_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
ci_ref=$ci_base/toy-material-424656
ci_out=$ci_base/continuum-interpolation-${SLURM_JOB_ID:?}
mkdir "$ci_out" "$ci_out/source"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ci_src"
[[ "$(git rev-parse --show-toplevel)" == "$ci_src" ]]
git rev-parse HEAD > "$ci_out/commit.txt"
(cd "$ci_ref/source";sha256sum -c "$ci_ref/source.sha256") > "$ci_out/reference-source-verify.log"
while read -r ci_hash ci_path;do
 [[ "$ci_path" == src/* ]] || continue
 cmp "$ci_path" "$ci_ref/source/$ci_path"
done < "$ci_ref/source.sha256"
(cd "$ci_ref/input";sha256sum -c "$ci_ref/input.sha256") > "$ci_out/input-verify.log"
sha256sum -c "$ci_ref/iterate.sha256" > "$ci_out/iterate-verify.log"
cp tests/cmf_toy_continuum_interpolation.c "$ci_out/source/"
scontrol write batch_script "$SLURM_JOB_ID" "$ci_out/source/runner.slurm"
cd "$ci_out"
sha256sum source/* "$ci_ref/source.sha256" "$ci_ref/objects/"*.o > source.sha256
gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections \
 -I"$ci_ref/source/src" source/cmf_toy_continuum_interpolation.c "$ci_ref/objects/"*.o \
 -Wl,--gc-sections -lm -o continuum-interpolation > build.log 2>&1
while read -r ci_key;do unset "$ci_key";done < <(compgen -v LUMINA_ || true)
while IFS= read -r ci_entry;do export "$ci_entry";done < "$ci_ref/env_lumina_effective.txt"
env | LC_ALL=C sort | grep '^LUMINA_' > env_lumina_effective.txt
ci_id=$(sha256sum "$ci_ref/iterate_identity.txt");ci_id=${ci_id%% *}
./continuum-interpolation "$ci_ref/input/snia_toy06_2d.dat" "$ci_ref/input" 100 \
 "$ci_ref/iterate_input.bin" "$ci_id" > test.log 2> test.err
grep '^CONTINUUM_INTERPOLATION_DONE ' test.log
sha256sum -c source.sha256 > source-verify.log
sha256sum -c "$ci_ref/iterate.sha256" > iterate-final-verify.log
sha256sum continuum-interpolation continuum_interpolation.csv test.log test.err env_lumina_effective.txt > output.sha256
