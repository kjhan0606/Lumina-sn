#!/bin/bash
#SBATCH --job-name=CMF_T_RESPONSE_COMPARE
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/temperature-compare-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/temperature-compare-%j.err
set -euo pipefail
umask 027
tc_source=${1:?sealed comparison source directory required}
tc_constant=${2:?constant response root required}
tc_reconstructed=${3:?reconstructed response root required}
tc_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
tc_public=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06
tc_out=$tc_base/temperature-compare-${SLURM_JOB_ID}
[[ "$tc_source" == "$tc_base"/temperature-compare-source.* ]]
[[ "$tc_constant" == "$tc_base"/temperature-response-* && "$tc_reconstructed" == "$tc_base"/temperature-response-* ]]
mkdir "$tc_out" "$tc_out/source"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load python/3.11.14
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
(cd "$tc_source";sha256sum -c source.sha256) > "$tc_out/source-input-verify.log"
for tc_file in compare_cmf_temperature_response.py compare_cmf_initial_iterate.py read_cmf_common_material.py;do
 cp "$tc_source/$tc_file" "$tc_out/source/"
done
cp "$tc_source/source.sha256" "$tc_out/"
scontrol write batch_script "$SLURM_JOB_ID" "$tc_out/runner.slurm"
for tc_mode in constant reconstructed;do
 if [[ "$tc_mode" == constant ]];then tc_run=$tc_constant;else tc_run=$tc_reconstructed;fi
 (cd "$tc_run/source";sha256sum -c "$tc_run/source.sha256") > "$tc_out/$tc_mode-source-verify.log"
 (cd "$tc_run/work";sha256sum -c "$tc_run/output.sha256") > "$tc_out/$tc_mode-output-verify.log"
 sha256sum -c "$tc_run/executable-output.sha256" > "$tc_out/$tc_mode-executable-verify.log"
done
python3 "$tc_out/source/compare_cmf_temperature_response.py" "$tc_constant" "$tc_reconstructed" "$tc_public" "$tc_out/comparison" > "$tc_out/comparison.log"
cd "$tc_out"
sha256sum source/*.py runner.slurm comparison/ionization.csv comparison/parent-ionization.csv comparison/summary.json comparison.log > output.sha256
sha256sum -c output.sha256 > output-verify.log
