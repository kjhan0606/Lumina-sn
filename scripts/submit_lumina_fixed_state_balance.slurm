#!/bin/bash -l
# 같은 상태의 잔차 후처리만 수행한다. FF/SE/BF/BB/RT/native 재실행 없음.
#SBATCH --job-name=lumina-fixed-state-balance
#SBATCH --partition=normal
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$project/scripts"
cd "$project"
sha256sum scripts/lumina_native_heat_sources.py scripts/lumina_model_balance.py \
  scripts/run_lumina_fixed_state_balance.py src/cmfgen_adiabatic.c src/cmfgen_adiabatic.h \
  scripts/lumina_model_energy.py > "$parent/build/source-before.sha256"
python3 -m pytest -q -p no:cacheprovider tests/test_lumina_native_heat_sources.py tests/test_lumina_model_balance.py
if test -n "${3:-}"; then
    test -f "$3"
    gcc -O2 -std=c11 -Wall -Wextra -Werror -shared -fPIC src/cmfgen_adiabatic.c -lm -o "$parent/build/adiabatic.so"
    timeout -k 15 480s python3 scripts/run_lumina_fixed_state_balance.py \
      --ff-result "$3" --adiabatic-library "$parent/build/adiabatic.so" \
      --source-commit "$source_commit" --out "$parent/run"
else
    printf 'UNIT_ONLY: no FF result supplied; actual energy balance NOT evaluated\n'
fi
sha256sum -c "$parent/build/source-before.sha256" > "$parent/build/source-after-check.txt"
