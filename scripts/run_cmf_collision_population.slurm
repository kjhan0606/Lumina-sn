#!/bin/bash
#SBATCH --job-name=CMF_COLLISION_POPULATION
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/collision-population-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/collision-population-%j.err
set -euo pipefail
umask 027
pop_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
pop_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
pop_prior=$pop_base/toy-material-424100
pop_changed=$pop_base/collision-complement-424211
pop_mode=${1:-comparison}
[[ "$pop_mode" == comparison || "$pop_mode" == sanitized ]] || exit 2
pop_out=$pop_base/collision-population-$SLURM_JOB_ID
mkdir "$pop_out"
mkdir "$pop_out/source"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 python/3.11.14
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
scontrol write batch_script "$SLURM_JOB_ID" "$pop_out/runner.slurm"
cd "$pop_src"
git rev-parse HEAD > "$pop_out/commit.txt"
[[ $(git rev-parse HEAD:src/lumina_plasma.c) == $(git hash-object "$pop_prior/source/src/lumina_plasma.c") ]]
pop_files=(tests/cmf_toy_collision_population.c tests/cmf_collision_population_compare.py scripts/run_cmf_collision_population.slurm)
cp --parents "${pop_files[@]}" "$pop_out/source/"
cmp src/lumina_plasma.c "$pop_changed/source/src/lumina_plasma.c"
cmp src/cmf_collision_complement.h "$pop_changed/source/src/cmf_collision_complement.h"
(cd "$pop_changed/source";sha256sum -c "$pop_changed/source.sha256") > "$pop_out/changed_source_verify.log"
(cd "$pop_prior/source";sha256sum -c "$pop_prior/source.sha256") > "$pop_out/prior_source_verify.log"
(cd "$pop_prior/input";sha256sum -c "$pop_prior/input.sha256") > "$pop_out/input_verify.log"
sha256sum -c "$pop_prior/nonthermal.sha256" > "$pop_out/nonthermal_verify.log"
sha256sum -c "$pop_changed/prior_objects.sha256" > "$pop_out/prior_objects_verify.log"
for pop_state in toy-material-424083 toy-material-424086 toy-material-424175;do
 sha256sum -c "$pop_base/$pop_state/iterate.sha256" > "$pop_out/$pop_state-verify.log"
 cmp "$pop_base/$pop_state/iterate_identity.txt" "$pop_base/toy-material-424086/iterate_identity.txt"
done
pop_rest=()
for pop_object in "$pop_prior/objects/"*.o;do
 pop_unit=${pop_object##*/};pop_unit=${pop_unit%.o}
 [[ "$pop_unit" == lumina_plasma ]] || pop_rest+=("$pop_object")
done
sha256sum "$pop_changed/plasma.o" "$pop_prior/objects/"*.o > "$pop_out/objects.sha256"
cd "$pop_out/source"
sha256sum "${pop_files[@]}" > "$pop_out/source.sha256"
pop_variants=(old new)
pop_iterations=(1 2)
if [[ "$pop_mode" == sanitized ]];then
 pop_variants=(sanitized);pop_iterations=(1)
 gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections \
  -fsanitize=address,undefined -fno-omit-frame-pointer -I"$pop_changed/source/src" \
  -c "$pop_changed/source/src/lumina_plasma.c" -o "$pop_out/plasma-san.o" > "$pop_out/build-san-plasma.log" 2>&1
fi
for pop_variant in "${pop_variants[@]}";do
 if [[ "$pop_variant" == old ]];then pop_plasma=$pop_prior/objects/lumina_plasma.o;else pop_plasma=$pop_changed/plasma.o;fi
 pop_flags=(-O2)
 if [[ "$pop_variant" == sanitized ]];then
  pop_plasma=$pop_out/plasma-san.o;pop_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
 fi
 gcc "${pop_flags[@]}" -std=gnu11 -D_GNU_SOURCE -fopenmp -I"$pop_changed/source/src" tests/cmf_toy_collision_population.c \
  "$pop_plasma" "${pop_rest[@]}" -Wl,--gc-sections -lm -o "$pop_out/$pop_variant" > "$pop_out/build-$pop_variant.log" 2>&1
done
while read -r pop_key;do unset "$pop_key";done < <(compgen -v LUMINA_ || true)
while IFS= read -r pop_entry;do export "$pop_entry";done < "$pop_prior/env_lumina_effective.txt"
env | LC_ALL=C sort | grep '^LUMINA_' > "$pop_out/env_lumina_effective.txt"
pop_id=$(sha256sum "$pop_base/toy-material-424086/iterate_identity.txt");pop_id=${pop_id%% *}
export ASAN_OPTIONS=detect_leaks=1
for pop_iteration in "${pop_iterations[@]}";do
 if [[ "$pop_iteration" == 1 ]];then pop_state=toy-material-424086;pop_field=toy-material-424083;else pop_state=toy-material-424175;pop_field=toy-material-424086;fi
 for pop_variant in "${pop_variants[@]}";do
  mkdir "$pop_out/$pop_variant-$pop_iteration"
  (cd "$pop_out/$pop_variant-$pop_iteration";"../$pop_variant" "$pop_prior/input/snia_toy06_2d.dat" \
   "$pop_prior/input" "$pop_base/$pop_state/iterate_input.bin" "$pop_id" "$pop_base/$pop_field/iterate_input.bin" \
   "$pop_prior/nonthermal_atomic.dat" "$pop_prior/nonthermal_resolved_bindings.csv" "$pop_prior/initial_gamma.csv" > run.log 2> run.err)
 done
done
cd "$pop_out"
(cd source;sha256sum -c ../source.sha256) > source_verify.log
sha256sum -c objects.sha256 > objects_verify.log
if [[ "$pop_mode" == comparison ]];then
 python3 source/tests/cmf_collision_population_compare.py "$pop_out" > comparison.log
 sha256sum old new old-1/ions.csv new-1/ions.csv old-2/ions.csv new-2/ions.csv \
  old-1/levels.csv new-1/levels.csv old-2/levels.csv new-2/levels.csv comparison.json > output.sha256
else
 sha256sum sanitized sanitized-1/ions.csv sanitized-1/levels.csv > output.sha256
fi
