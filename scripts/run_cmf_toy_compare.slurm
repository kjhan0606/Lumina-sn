#!/bin/bash
#SBATCH --job-name=CMF_TOY_COMPARE
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-compare-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-compare-%j.err
set -euo pipefail
umask 027
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load python/3.11.14
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
compare_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
compare_public=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06
compare_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-compare-${SLURM_JOB_ID}
mkdir "$compare_out"
mkdir "$compare_out/source"
cd "$compare_src"
compare_files=(scripts/compare_cmf_toy_epoch.py scripts/read_cmf_common_material.py
 scripts/run_cmf_toy_compare.slurm tests/cmf_toy_comparison_selftest.py
 scripts/compare_cmf_initial_iterate.py scripts/diagnose_cmf_saha_reference.py)
cp --parents "${compare_files[@]}" "$compare_out/source/"
cd "$compare_out/source"
sha256sum "${compare_files[@]}" > "$compare_out/source.sha256"
python3 tests/cmf_toy_comparison_selftest.py "$compare_public/spectra_toy06_cmfgen.txt" > "$compare_out/selftest.log"
if [[ "${1:-selftest}" == initial-saha ]]; then
 (cd "$2/source"; sha256sum -c "$2/source.sha256") > "$compare_out/original_source_verify.log"
 (cd "$2/input"; sha256sum -c "$2/input.sha256") > "$compare_out/original_input_verify.log"
 sha256sum "$2/toy_initial_material.csv" "$2/input/levels.csv" "$2/input/ionization_energies.csv" \
  "$compare_public/phys_toy06_cmfgen.txt" "$compare_public/ionfrac_si_toy06_cmfgen.txt" \
  "$compare_public/ionfrac_s_toy06_cmfgen.txt" > "$compare_out/saha_input.sha256"
 python3 scripts/diagnose_cmf_saha_reference.py --run "$2" --public "$compare_public" \
  --out "$compare_out/saha" > "$compare_out/saha.json"
 sha256sum -c "$compare_out/saha_input.sha256" > "$compare_out/saha_input_verify.log"
elif [[ "${1:-selftest}" == initial-grid ]]; then
 for compare_root in "$2" "$3"; do
  (cd "$compare_root/source"; sha256sum -c "$compare_root/source.sha256") >> "$compare_out/original_source_verify.log"
  (cd "$compare_root/input"; sha256sum -c "$compare_root/input.sha256") >> "$compare_out/original_input_verify.log"
 done
 # 해상도만의 대조이므로 원자자료/충돌표/연속채널까지 같은 입력이어야 한다.
 cmp "$2/input.sha256" "$3/input.sha256"
 sha256sum -c "$2/iterate.sha256" > "$compare_out/iterate_verify.log"
 cmp "$2/input/snia_toy06_2d.dat" "$3/input/snia_toy06_2d.dat"
 sha256sum "$2/toy_initial_material.csv" "$2/iterate_material.csv" "$3/toy_initial_material.csv" \
  "$2/input/snia_toy06_2d.dat" \
  "$compare_public/phys_toy06_cmfgen.txt" "$compare_public/ionfrac_si_toy06_cmfgen.txt" \
  "$compare_public/ionfrac_s_toy06_cmfgen.txt" > "$compare_out/grid_input.sha256"
 python3 scripts/compare_cmf_initial_iterate.py --run "$2" --finer "$3" --public "$compare_public" \
  > "$compare_out/grid.json"
 sha256sum -c "$compare_out/grid_input.sha256" > "$compare_out/grid_input_verify.log"
elif [[ "${1:-selftest}" != selftest ]]; then
 (cd "$1/work"; sha256sum -c "$1/output.sha256") > "$compare_out/output_verify.log"
 python3 scripts/compare_cmf_toy_epoch.py --run "$1" --public "$compare_public" \
  --out "$compare_out/comparison" > "$compare_out/comparison.log"
fi
sha256sum -c "$compare_out/source.sha256" > "$compare_out/source_verify.log"
