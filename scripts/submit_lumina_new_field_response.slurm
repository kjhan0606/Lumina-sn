#!/bin/bash -l
# 기본 새 J 고정 T 물질응답은 CPU16/32GiB/4h.
# 장시간 열보정은 별도 승인된 단회에만 sbatch --time=1-06:00:00 및
# LUMINA_RESPONSE_LONG_THERMAL=1, LUMINA_RESPONSE_SECONDS=106200을 사용한다.
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
response_seconds=${LUMINA_RESPONSE_SECONDS:-13800}
[[ "$response_seconds" =~ ^[0-9]+$ ]]
max_seconds=13800
if [[ "${LUMINA_RESPONSE_EXECUTOR:-slurm}" = manual-lageunha ]]; then max_seconds=42600; fi
if [[ "${LUMINA_RESPONSE_LONG_THERMAL:-0}" = 1 ]]; then
    [[ "${LUMINA_RESPONSE_EXECUTOR:-slurm}" = slurm ]]
    max_seconds=106200
    [[ " $* " = *" --temperature-evaluations "* ]]
    [[ "${SLURM_TIMELIMIT:-}" =~ ^[0-9]+$ ]]
    (( SLURM_TIMELIMIT*60 >= response_seconds+300 ))
fi
(( response_seconds >= 600 && response_seconds <= max_seconds ))
if [[ "${LUMINA_RESPONSE_EXECUTOR:-slurm}" = slurm ]]; then
    module purge
    module load gnu13/13.2.0
elif [[ "${LUMINA_RESPONSE_EXECUTOR}" != manual-lageunha ]]; then
    echo '알 수 없는 물질응답 실행 방식' >&2
    exit 2
fi
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
gcc -O2 -std=c11 -Wall -Wextra -Werror -shared -fPIC -fopenmp src/model_line_linear_jbar.c -lm -o "$parent/build/lines.so"
sha256sum "$parent/build/bridge.so" "$parent/build/lines.so" > "$parent/build/binary.sha256"
timeout -k 30 "$((response_seconds+200))s" python3 scripts/lumina_new_field_response.py --seconds "$response_seconds" \
    --library "$parent/build/bridge.so" --line-library "$parent/build/lines.so" \
    --source-commit "$source_commit" --out "$parent/run" "$@"
