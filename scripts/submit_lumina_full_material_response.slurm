#!/bin/bash -l
# 완료한 BF90/BB90 재사용 + 전 FL쌍 C 소비. 공동 charge/RE/수송 갱신 없음.
#SBATCH --job-name=lumina-full-model-response
#SBATCH --partition=normal
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
set -euo pipefail
umask 027
ulimit -c 0
module purge
module load gnu13/13.2.0
repository=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
parent=$(realpath "${1:?새 parent 필요}")
source_commit=${2:?봉인 commit 필요}
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
test -d "$parent"
test ! -e "$parent/source"
mkdir "$parent/source" "$parent/build"
git -C "$repository" archive "$source_commit" src scripts tests data | tar -x -C "$parent/source"
project="$parent/source"
test "$(sha256sum "$repository/data/atomic/ionization_reference.csv" | cut -d ' ' -f 1)" = \
  660141ef55e5b9028f6a8f374a28a149bcf81e67f1f888fc0e157397aea4fa9a
test ! -e "$project/data/atomic/ionization_reference.csv"
cp -- "$repository/data/atomic/ionization_reference.csv" "$project/data/atomic/ionization_reference.csv"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$project/scripts"
export LUMINA_ARTIS_PARITY=0 LUMINA_NLTE_BF_COLLISION=1
export LUMINA_TOPION_LEVELS_FILE="$project/data/atomic/topion_levels.csv"
export LUMINA_IONIZATION_REFERENCE_FILE="$project/data/atomic/ionization_reference.csv"
core=(lumina_plasma lumina_element_wide nlte_model_se nlte_model_bb nlte_model_coll_bb
      population_contract opacity_publication emissivity_publication radeq_publication
      line_net_rate cmfgen_adiabatic bf_rate_jnu radiation_field seed_capability
      atomic_internal_energy nlte_population_candidate line_jbar lumina_atomic
      nlte_model_material_bridge)
cd "$project"
files=()
for name in "${core[@]}"; do files+=("src/$name.c"); done
sha256sum src/*.h "${files[@]}" scripts/lumina_model_material_bridge.py \
  scripts/run_lumina_full_material_response.py scripts/lumina_model_collision_source.py \
  data/atomic/ionization_reference.csv data/atomic/topion_levels.csv > "$parent/build/source-before.sha256"
objects=()
for name in "${core[@]}"; do
    gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -fPIC -fvisibility=hidden \
      -ffunction-sections -fdata-sections -I"$project/src" -c "src/$name.c" -o "$parent/build/$name.o"
    objects+=("$parent/build/$name.o")
done
gcc -shared -fopenmp -Wl,--gc-sections -Wl,-z,defs -o "$parent/build/bridge.so" "${objects[@]}" -lm
sha256sum "$parent/build/bridge.so" > "$parent/build/binary.sha256"
timeout -k 30 6900s python3 scripts/run_lumina_full_material_response.py \
  --library "$parent/build/bridge.so" --source-commit "$source_commit" --out "$parent/run"
sha256sum -c "$parent/build/source-before.sha256" > "$parent/build/source-after-check.txt"
