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
 scripts/compare_cmf_initial_iterate.py scripts/diagnose_cmf_saha_reference.py scripts/compare_cmf_initial_guess.py)
cp --parents "${compare_files[@]}" "$compare_out/source/"
cd "$compare_out/source"
sha256sum "${compare_files[@]}" > "$compare_out/source.sha256"
if [[ "${1:-selftest}" != initial-guess ]]; then
 python3 tests/cmf_toy_comparison_selftest.py "$compare_public/spectra_toy06_cmfgen.txt" > "$compare_out/selftest.log"
fi
if [[ "${1:-selftest}" == initial-guess ]]; then
 for compare_root in "$2" "$3"; do
  (cd "$compare_root/source"; sha256sum -c "$compare_root/source.sha256") >> "$compare_out/original_source_verify.log"
  (cd "$compare_root/input"; sha256sum -c "$compare_root/input.sha256") >> "$compare_out/original_input_verify.log"
  sha256sum -c "$compare_root/iterate.sha256" >> "$compare_out/iterate_verify.log"
 done
 unset PYTHONHOME PYTHONPATH
 /home/kjhan/miniconda3/bin/python3 scripts/compare_cmf_initial_guess.py "$2" "$3" "$4" \
  "$compare_out/field-change" > "$compare_out/comparison.log"
 sha256sum "$compare_out/field-change/comparison.json" > "$compare_out/output.sha256"
elif [[ "${1:-selftest}" == initial-saha || "${1:-selftest}" == initial-saha-model ]]; then
 compare_saha_args=()
 compare_python=python3
 if [[ "$1" == initial-saha-model ]]; then
  # 기존 원자료 parser는 numpy를 사용한다. 설치하지 않고 검증된 기존 환경을 지정한다.
  compare_python=/home/kjhan/miniconda3/bin/python3
  # module Python의 표준 라이브러리를 다른 버전의 해석기에 섞지 않는다.
  unset PYTHONHOME PYTHONPATH
  "$compare_python" -c 'import sys,numpy; print(sys.executable); print(sys.version); print(numpy.__version__); print(numpy.__file__)' > "$compare_out/python_identity.txt"
  cp /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_parser.py scripts/
  sha256sum scripts/cmfgen_parser.py >> "$compare_out/source.sha256"
  mkdir "$compare_out/raw"
  compare_raw=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen
  compare_raw_files=(SIL/II/19apr23/osc_data SIL/III/19apr23/osc_data SUL/II/19apr23/osc_data SUL/III/19apr23/osc_data)
  (cd "$compare_raw"; cp --parents "${compare_raw_files[@]}" "$compare_out/raw/")
  (cd "$compare_out/raw"; sha256sum "${compare_raw_files[@]}") > "$compare_out/raw.sha256"
  compare_saha_args=(--raw "$compare_out/raw")
 fi
 (cd "$2/source"; sha256sum -c "$2/source.sha256") > "$compare_out/original_source_verify.log"
 (cd "$2/input"; sha256sum -c "$2/input.sha256") > "$compare_out/original_input_verify.log"
 sha256sum "$2/toy_initial_material.csv" "$2/input/levels.csv" "$2/input/ionization_energies.csv" \
  "$compare_public/phys_toy06_cmfgen.txt" "$compare_public/ionfrac_si_toy06_cmfgen.txt" \
  "$compare_public/ionfrac_s_toy06_cmfgen.txt" > "$compare_out/saha_input.sha256"
 "$compare_python" scripts/diagnose_cmf_saha_reference.py --run "$2" --public "$compare_public" \
  --out "$compare_out/saha" "${compare_saha_args[@]}" > "$compare_out/saha.json"
 if [[ "$1" == initial-saha-model ]]; then
  (cd "$compare_out/raw"; sha256sum -c "$compare_out/raw.sha256") > "$compare_out/raw_verify.log"
 fi
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
 if [[ -f "$3/iterate_material.csv" ]]; then
  sha256sum -c "$3/iterate.sha256" >> "$compare_out/iterate_verify.log"
  sha256sum "$3/iterate_material.csv" >> "$compare_out/grid_input.sha256"
 fi
 python3 scripts/compare_cmf_initial_iterate.py --run "$2" --finer "$3" --public "$compare_public" \
  > "$compare_out/grid.json"
 sha256sum -c "$compare_out/grid_input.sha256" > "$compare_out/grid_input_verify.log"
elif [[ "${1:-selftest}" != selftest ]]; then
 (cd "$1/work"; sha256sum -c "$1/output.sha256") > "$compare_out/output_verify.log"
 python3 scripts/compare_cmf_toy_epoch.py --run "$1" --public "$compare_public" \
  --out "$compare_out/comparison" > "$compare_out/comparison.log"
fi
sha256sum -c "$compare_out/source.sha256" > "$compare_out/source_verify.log"
