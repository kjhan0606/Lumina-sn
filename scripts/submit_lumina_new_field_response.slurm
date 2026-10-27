#!/bin/bash -l
# 사용자 승인: 새 J 물질응답만 CPU16/32GiB/4h, 단회, 재시도0.
#SBATCH --job-name=lumina-new-J-response
#SBATCH --partition=normal
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
set -euo pipefail
umask 027
ulimit -c 0
module purge
module load gnu13/13.2.0
repository=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
parent=$(realpath "${1:?새 parent 필요}")
source_commit=${2:?봉인 commit 필요}
shift 2
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
test -d "$parent"
test ! -e "$parent/source"
test ! -e "$parent/run"
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
objects=()
for name in "${core[@]}"; do
    visibility=hidden
    if [[ "$name" = cmfgen_adiabatic ]]; then visibility=default; fi
    gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -fPIC -fvisibility="$visibility" \
      -ffunction-sections -fdata-sections -I"$project/src" -c "src/$name.c" -o "$parent/build/$name.o"
    objects+=("$parent/build/$name.o")
done
gcc -shared -fopenmp -Wl,--gc-sections -Wl,-z,defs -o "$parent/build/bridge.so" "${objects[@]}" -lm
gcc -O2 -std=c11 -Wall -Wextra -Werror -shared -fPIC src/model_line_linear_jbar.c -lm -o "$parent/build/lines.so"
sha256sum "$parent/build/bridge.so" "$parent/build/lines.so" > "$parent/build/binary.sha256"
timeout -k 30 14000s python3 scripts/lumina_new_field_response.py --seconds 13800 \
    --library "$parent/build/bridge.so" --line-library "$parent/build/lines.so" \
    --source-commit "$source_commit" --out "$parent/run" "$@"
