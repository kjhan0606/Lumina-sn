#!/bin/bash
#SBATCH --job-name=CMF_COLLISION_COMPLEMENT
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/collision-complement-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/collision-complement-%j.err
set -euo pipefail
umask 027
collision_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
collision_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
collision_prior=$collision_base/toy-material-424100
collision_state=$collision_base/toy-material-424175
collision_out=$collision_base/collision-complement-$SLURM_JOB_ID
mkdir "$collision_out"
mkdir "$collision_out/source"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 python/3.11.14
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
scontrol write batch_script "$SLURM_JOB_ID" "$collision_out/runner.slurm"
cd "$collision_src"
git rev-parse HEAD > "$collision_out/commit.txt"
collision_files=(src/*.h src/lumina_plasma.c tests/cmf_collision_complement_selftest.c
 tests/cmf_toy_collision_complement.c tests/cmf_collision_prescription_check.py
 scripts/run_cmf_collision_complement.slurm)
cp --parents "${collision_files[@]}" "$collision_out/source/"
(cd "$collision_prior/source";sha256sum -c "$collision_prior/source.sha256") > "$collision_out/prior_source_verify.log"
(cd "$collision_state/input";sha256sum -c "$collision_state/input.sha256") > "$collision_out/input_verify.log"
sha256sum -c "$collision_state/iterate.sha256" > "$collision_out/iterate_verify.log"
collision_objects=()
for collision_object in "$collision_prior/objects/"*.o;do
 collision_unit=${collision_object##*/};collision_unit=${collision_unit%.o}
 if [[ "$collision_unit" != lumina_plasma ]];then
  cmp "src/$collision_unit.c" "$collision_prior/source/src/$collision_unit.c"
  collision_objects+=("$collision_object")
 fi
done
for collision_header in "$collision_prior/source/src/"*.h;do
 cmp "src/${collision_header##*/}" "$collision_header"
done
sha256sum "${collision_objects[@]}" > "$collision_out/prior_objects.sha256"
cd "$collision_out/source"
sha256sum "${collision_files[@]}" > "$collision_out/source.sha256"
gcc -O2 -std=c11 -Wall -Wextra -Werror -Isrc tests/cmf_collision_complement_selftest.c -lm -o "$collision_out/unit"
"$collision_out/unit" > "$collision_out/unit.log"
gcc -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc tests/cmf_collision_complement_selftest.c -lm -o "$collision_out/unit_san"
ASAN_OPTIONS=detect_leaks=1 "$collision_out/unit_san" > "$collision_out/unit_san.log" 2> "$collision_out/unit_san.err"
gcc -O2 -std=gnu11 -D_GNU_SOURCE -DLUMINA_TIME_PAIR_SELFTEST -fopenmp -ffunction-sections -fdata-sections -Isrc -c src/lumina_plasma.c -o "$collision_out/plasma.o" > "$collision_out/build.log" 2>&1
gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -Isrc tests/cmf_toy_collision_complement.c "$collision_out/plasma.o" "${collision_objects[@]}" -Wl,--gc-sections -lm -o "$collision_out/actual" >> "$collision_out/build.log" 2>&1
while read -r collision_key;do unset "$collision_key";done < <(compgen -v LUMINA_ || true)
while IFS= read -r collision_entry;do export "$collision_entry";done < "$collision_prior/env_lumina_effective.txt"
env | LC_ALL=C sort | grep '^LUMINA_' > "$collision_out/env_lumina_effective.txt"
cd "$collision_out"
python3 source/tests/cmf_collision_prescription_check.py "$collision_state/input" > prescriptions.log
collision_id=$(sha256sum "$collision_state/iterate_identity.txt");collision_id=${collision_id%% *}
./actual "$collision_state/input/snia_toy06_2d.dat" "$collision_state/input" "$collision_state/iterate_input.bin" "$collision_id" > actual.log 2> actual.err
(cd source;sha256sum -c ../source.sha256) > source_verify.log
sha256sum -c prior_objects.sha256 > prior_objects_verify.log
sha256sum unit unit_san actual actual.log collision_prescriptions.json > output.sha256
