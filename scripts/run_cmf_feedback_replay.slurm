#!/bin/bash
#SBATCH --job-name=CMF_FEEDBACK_STATE_REPLAY
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/feedback-replay-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/feedback-replay-%j.err
set -euo pipefail
umask 027
fr_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
fr_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
fr_run=${1:?completed feedback run required}
fr_ref=$fr_base/toy-material-424656
fr_out=$fr_base/feedback-replay-${SLURM_JOB_ID:?}
[[ "$fr_run" == "$fr_base"/feedback-field-* ]]
mkdir "$fr_out" "$fr_out/source"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$fr_src"
[[ "$(git rev-parse --show-toplevel)" == "$fr_src" ]]
(cd "$fr_run";sha256sum -c output.sha256;sha256sum -c executable-output.sha256) > "$fr_out/producer-verify.log"
(cd "$fr_run/source";sha256sum -c ../source.sha256) > "$fr_out/producer-source-verify.log"
(cd "$fr_ref/source";sha256sum -c "$fr_ref/source.sha256") > "$fr_out/reader-source-verify.log"
(cd "$fr_ref/input";sha256sum -c "$fr_ref/input.sha256") > "$fr_out/input-verify.log"
while read -r fr_hash fr_path;do
 [[ "$fr_path" == src/*.c || "$fr_path" == src/*.h ]] || continue
 cmp "$fr_ref/source/$fr_path" "$fr_run/source/$fr_path"
done < "$fr_ref/source.sha256"
cp tests/cmf_toy_iterate_replay.c "$fr_out/source/"
scontrol write batch_script "$SLURM_JOB_ID" "$fr_out/source/runner.slurm"
cd "$fr_out"
sha256sum source/* "$fr_ref/source.sha256" "$fr_ref/objects/"*.o > source.sha256
gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections \
 -I"$fr_ref/source/src" source/cmf_toy_iterate_replay.c "$fr_ref/objects/"*.o \
 -Wl,--gc-sections -lm -o state-replay > build.log 2>&1
while read -r fr_key;do unset "$fr_key";done < <(compgen -v LUMINA_ || true)
while IFS= read -r fr_entry;do export "$fr_entry";done < "$fr_ref/env_lumina_effective.txt"
for fr_mode in zero plus minus;do
 mkdir "$fr_mode"
 (
  cd "$fr_mode"
  (cd "$fr_run/$fr_mode";sha256sum -c output.sha256) > producer-branch-verify.log
  fr_id=$(sha256sum "$fr_run/$fr_mode/diagnostic_identity.txt");fr_id=${fr_id%% *}
  ../state-replay "$fr_ref/input/snia_toy06_2d.dat" "$fr_ref/input" \
   "$fr_run/$fr_mode/diagnostic_state.bin" "$fr_id" > test.log 2> test.err
  # 유일한 변경은 native reader가 올리는 인구 세대(112..119)다.
  read -r fr_old_generation < <(od -An -tu8 -j112 -N8 "$fr_run/$fr_mode/diagnostic_state.bin")
  read -r fr_new_generation < <(od -An -tu8 -j112 -N8 state_roundtrip.bin)
  (( fr_new_generation == fr_old_generation + 1 ))
  cmp -n 112 "$fr_run/$fr_mode/diagnostic_state.bin" state_roundtrip.bin
  cmp -i 120 "$fr_run/$fr_mode/diagnostic_state.bin" state_roundtrip.bin
  cmp <(tail -n +2 "$fr_run/$fr_mode/diagnostic_material.csv") <(tail -n +2 material_roundtrip.csv)
  sha256sum state_roundtrip.bin material_roundtrip.csv test.log test.err > output.sha256
  grep '^ITERATE_REPLAY_READ_DONE ' test.log
 )
done
sha256sum -c source.sha256 > source-verify.log
sha256sum state-replay zero/output.sha256 plus/output.sha256 minus/output.sha256 > output.sha256
printf 'FEEDBACK_REPLAY_DONE full_state_byte_equal_except_population_generation=1 physics_verdict=NOT_JUDGED\n'
