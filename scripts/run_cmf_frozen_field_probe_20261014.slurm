#!/bin/bash
#SBATCH --job-name=CMF_ANGLE_FROZEN
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/angular-frozen-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/angular-frozen-%j.err
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LUMINA_INNER_BB_SCALE=1
unset LUMINA_CMFGEN_LAMBDA_TRI LUMINA_CMFGEN_CELLDIAG
probe_repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
probe_base=/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/accel/source/src
probe_new=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-angular-20261014/src
probe_capture=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/det_obs_capture
probe_cells=/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012/out/r1_budget_419905/operator_cells.csv
probe_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/angular-frozen-${SLURM_JOB_ID}
probe_args=()
probe_variants=(base corrected)
if [[ ${1:-} == scatter ]]; then
    probe_args=(scatter)
elif [[ ${1:-} == scatter-refined ]]; then
    probe_args=(scatter-refined)
    probe_variants=(corrected)
elif [[ $# -gt 0 ]]; then exit 2; fi
mkdir "$probe_out"
sha256sum "$probe_repo/tests/cmf_frozen_field_probe.c" "$probe_base/lumina_cmfgen.c" "$probe_new/lumina_cmfgen.c" "$probe_cells" "$probe_capture/meta.txt" "$probe_capture/r_inner.f64" "$probe_capture/r_outer.f64" "$probe_capture/nu.f64" "$probe_capture/dnu.f64" > "$probe_out/inputs.sha256"
cp --no-clobber "$probe_repo/tests/cmf_frozen_field_probe.c" "$probe_out/fixture.c"
for probe_variant in "${probe_variants[@]}"; do
    if [[ "$probe_variant" == base ]]; then probe_src=$probe_base; else probe_src=$probe_new; fi
    cp --no-clobber "$probe_src/lumina_cmfgen.c" "$probe_out/$probe_variant.source.c"
    gcc -O2 -std=gnu11 -fopenmp -ffunction-sections -fdata-sections -I"$probe_src" "$probe_out/fixture.c" "$probe_out/$probe_variant.source.c" -Wl,--gc-sections -lm -o "$probe_out/$probe_variant"
    "$probe_out/$probe_variant" "$probe_capture" "$probe_cells" "$probe_out/$probe_variant.J.f64" "${probe_args[@]}" > "$probe_out/$probe_variant.json"
done
sha256sum -c "$probe_out/inputs.sha256"
echo "result=$probe_out"
