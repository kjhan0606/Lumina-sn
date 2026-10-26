#!/bin/bash -l
# 동일 원70/J70/node49의 Si/S 물질 응답 단회 비교. 새 RT/native 실행 없음.
#SBATCH --job-name=lumina-sameJ-response
#SBATCH --partition=normal
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
set -euo pipefail
umask 027
ulimit -c 0
module purge
module load gnu13/13.2.0
test "${SLURM_CPUS_PER_TASK:-}" = 1
test "${SLURM_MEM_PER_NODE:-}" = 4096
parent=$(realpath "${1:?새 parent 필요}")
test -d "$parent"
test ! -e "$parent/build"
test ! -e "$parent/run"
project=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$project/scripts"
mkdir "$parent/build"
timeout -k 15 300s bash "$project/tests/run_bf_point_tests.sh" "$project" "$parent/build" \
  > "$parent/build.stdout" 2> "$parent/build.stderr"
cd "$project"
gcc -O2 -std=c11 -fPIC -shared -Wl,-z,defs -Isrc -o "$parent/build/point.so" \
  src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c -lm
gcc -O2 -std=c11 -fPIC -shared -Wl,-z,defs -Isrc -o "$parent/build/lines.so" \
  src/model_source_kernels.c -lm
timeout -k 15 120s python3 -m pytest -q -p no:cacheprovider \
  tests/test_model_point_reference.py tests/test_model_point_dissolution.py \
  tests/test_lumina_reference_response.py > "$parent/reference-tests.stdout" 2> "$parent/reference-tests.stderr"
timeout -k 15 3100s python3 scripts/run_lumina_reference_response.py \
  --base /gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json \
  --baseline /gpfs/kjhan/cmfgen_runs/lumina_direct_slprofile_20261026-UUiwsU/native-occupation.json \
  --prediction /gpfs/kjhan/cmfgen_runs/lumina_prrr_4ion_20261026-LpCj1K/run/comparison.json \
  --rate-contract /gpfs/kjhan/cmfgen_runs/lumina_rate_contract_20261026-bCgTkl/rates.csv \
  --native-root /gpfs/kjhan/cmfgen_runs/lumina_fixedj_exp70_20261026-phmMnx \
  --collision /gpfs/kjhan/cmfgen_runs/lumina_fullpair8_20261026-qXSzbE/fullpair.bin \
  --point-library "$parent/build/point.so" --lines-library "$parent/build/lines.so" \
  --projection "$parent/build/model_projection" --out "$parent/run"
