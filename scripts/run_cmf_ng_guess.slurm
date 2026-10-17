#!/bin/bash
#SBATCH --job-name=CMF_NG_INITIAL_GUESS
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/ng-guess-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/ng-guess-%j.err
set -euo pipefail
umask 027
ng_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
ng_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
ng_prior=$ng_base/ng-damped-424693
ng_ref=$ng_base/toy-material-424656
ng_out=$ng_base/ng-guess-${SLURM_JOB_ID:?}
mkdir "$ng_out" "$ng_out/source" "$ng_out/native"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 python/3.11.14
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$ng_src"
[[ "$(git rev-parse --show-toplevel)" == "$ng_src" ]]
(cd "$ng_prior";sha256sum -c source.sha256;sha256sum -c output.sha256) > "$ng_out/prior-verify.log"
(cd "$ng_ref/source";sha256sum -c "$ng_ref/source.sha256") > "$ng_out/reader-source-verify.log"
(cd "$ng_ref/input";sha256sum -c "$ng_ref/input.sha256") > "$ng_out/input-verify.log"
cp "$ng_prior/source/assess_damped.py" "$ng_prior/source/original.py" "$ng_out/source/"
cp "$ng_prior/ngaccel.so" "$ng_out/"
cp "$ng_prior/damped_direction.json" "$ng_out/previous_direction.json"
cp scripts/write_cmf_ng_guess.py tests/cmf_toy_iterate_replay.c "$ng_out/source/"
scontrol write batch_script "$SLURM_JOB_ID" "$ng_out/source/runner.slurm"
cd "$ng_out"
sha256sum source/* ngaccel.so previous_direction.json "$ng_ref/source.sha256" "$ng_ref/objects/"*.o > source.sha256
gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections \
 -I"$ng_ref/source/src" source/cmf_toy_iterate_replay.c "$ng_ref/objects/"*.o \
 -Wl,--gc-sections -lm -o state-replay > build.log 2>&1
python3 source/write_cmf_ng_guess.py > preparation.log
ng_id=$(sha256sum guess_identity.txt);ng_id=${ng_id%% *}
while read -r ng_key;do unset "$ng_key";done < <(compgen -v LUMINA_ || true)
while IFS= read -r ng_entry;do export "$ng_entry";done < "$ng_ref/env_lumina_effective.txt"
cd native
../state-replay "$ng_ref/input/snia_toy06_2d.dat" "$ng_ref/input" \
 ../guess_input.bin "$ng_id" > test.log 2> test.err
read -r ng_old_generation < <(od -An -tu8 -j112 -N8 ../guess_input.bin)
read -r ng_new_generation < <(od -An -tu8 -j112 -N8 state_roundtrip.bin)
(( ng_new_generation == ng_old_generation + 1 ))
cmp -n 112 ../guess_input.bin state_roundtrip.bin
cmp -i 120 ../guess_input.bin state_roundtrip.bin
grep '^ITERATE_REPLAY_READ_DONE ' test.log
cd ..
sha256sum -c source.sha256 > source-verify.log
sha256sum guess_input.bin guess_identity.txt damped_direction.json preparation.log \
 state-replay native/state_roundtrip.bin native/material_roundtrip.csv native/test.log native/test.err > output.sha256
printf 'NG_GUESS_NATIVE_PASS RTE_solved=0 SE_solved=0 convergence_acceleration=UNMEASURED physical_goal=NOT_JUDGED\n'
