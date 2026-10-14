#!/bin/bash
#SBATCH --job-name=CMF_COMMON_READ
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-readout-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-readout-%j.err
# 생산 작업 성공 뒤 한 번만 실행한다. 상주 감시나 자동 재실행은 하지 않는다.
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
read_run=${1:?common physics run root required}
read_label=${2:-COMMON_GAUSSIAN}
read_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015
read_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-readout-${SLURM_JOB_ID}
read_tool=/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/tools/sprint_readout.py
read_public=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06
read_te=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/fixed_te_profile.txt
mkdir "$read_out"
sha256sum "$read_src/scripts/read_cmf_common_material.py" \
 "$read_src/tests/test_cmf_common_material.py" > "$read_out/reader.sha256"
/home/kjhan/miniconda3/bin/python3 "$read_src/tests/test_cmf_common_material.py" \
 > "$read_out/selftest.log" 2>&1
if [[ $read_run == --selftest ]]; then
 sha256sum -c "$read_out/reader.sha256" > "$read_out/verify.log"
 exit 0
fi
test -s "$read_run/work/lumina_spectrum_common_obs.csv"
test -s "$read_run/work/lumina_spectrum_common_obs.json"
test -s "$read_run/work/cmfgen_final_publication.json"
sha256sum "$read_tool" "$read_public/spectra_toy06_cmfgen.txt" \
 "$read_public/lbol_edep_toy06_cmfgen.txt" "$read_te" \
 "$read_run/work/lumina_spectrum_common_obs.csv" \
 "$read_run/work/lumina_spectrum_common_obs.json" \
 "$read_run/work/cmfgen_final_publication.json" > "$read_out/input.sha256"
/home/kjhan/miniconda3/bin/python3 "$read_tool" --run "$read_run" \
 --label "$read_label" --out "$read_out/spectrum" --spectrum lumina_spectrum_common_obs.csv
/home/kjhan/miniconda3/bin/python3 "$read_src/scripts/read_cmf_common_material.py" \
 --run "$read_run" --model /gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/model \
 --public "$read_public" --out "$read_out/material"
sha256sum -c "$read_out/input.sha256" > "$read_out/verify.log"
sha256sum -c "$read_out/reader.sha256" >> "$read_out/verify.log"
printf 'readout=%s scope=spectrum_temperature_and_Si_S final_goal=NOT_JUDGED\n' "$read_out"
