#!/bin/bash
#SBATCH --job-name=CMF_THERMAL_TIME
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:0
#SBATCH --time=00:02:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/thermal-time-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/thermal-time-%j.err
set -euo pipefail
umask 027
thermal_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015
thermal_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/thermal-time-${SLURM_JOB_ID}
mkdir "$thermal_out"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$thermal_src"
git rev-parse HEAD > "$thermal_out/commit.txt"
sha256sum src/cmfgen_adiabatic.c src/cmfgen_adiabatic.h \
 scripts/diagnose_cmf_reference_inputs.py scripts/diagnose_cmf_thermal_time.py \
 scripts/run_cmf_thermal_time_diagnostic.slurm > "$thermal_out/source.sha256"
gcc -O2 -Wall -Wextra -std=c11 -fPIC -shared -Isrc src/cmfgen_adiabatic.c \
 -lm -o "$thermal_out/libadiabatic.so" > "$thermal_out/build.log" 2>&1
/home/kjhan/miniconda3/bin/python3 scripts/diagnose_cmf_thermal_time.py \
 "$thermal_out/libadiabatic.so" "$thermal_out"
sha256sum -c "$thermal_out/source.sha256" > "$thermal_out/source_verify.log"
