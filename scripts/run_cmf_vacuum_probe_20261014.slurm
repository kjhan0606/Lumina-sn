#!/bin/bash
#SBATCH --job-name=CMF_VACUUM
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/vacuum-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/vacuum-%j.err
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
probe_repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
probe_src=${1:-/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/accel/source/src}
probe_capture=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/det_obs_capture
probe_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/vacuum-${SLURM_JOB_ID}
mkdir "$probe_out"
sha256sum "$probe_repo/tests/cmf_vacuum_dilution_probe.c" "$probe_src/lumina_cmfgen.c" "$probe_src/lumina_cmfgen.h" "$probe_src/lumina.h" "$probe_capture/r_inner.f64" "$probe_capture/r_outer.f64" > "$probe_out/inputs.sha256"
gcc -O2 -std=gnu11 -fopenmp -ffunction-sections -fdata-sections -DLUMINA_POP_DJDN_SELFTEST -I"$probe_src" "$probe_repo/tests/cmf_vacuum_dilution_probe.c" "$probe_src/lumina_cmfgen.c" -Wl,--gc-sections -lm -o "$probe_out/probe"
if [[ $# -gt 0 ]]; then
    "$probe_out/probe" "$probe_capture/r_inner.f64" "$probe_capture/r_outer.f64" require-physical > "$probe_out/result.json"
else
    "$probe_out/probe" "$probe_capture/r_inner.f64" "$probe_capture/r_outer.f64" > "$probe_out/result.json"
fi
sha256sum -c "$probe_out/inputs.sha256"
echo "result=$probe_out/result.json"
