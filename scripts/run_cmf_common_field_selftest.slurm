#!/bin/bash
#SBATCH --job-name=CMF_COMMON_TEST
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-field-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-field-%j.err
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while read -r common_key; do unset "$common_key"; done < <(compgen -v LUMINA_ || true)
common_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015
common_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-field-${SLURM_JOB_ID}
mkdir "$common_out"
cd "$common_src"
common_units=(lumina_transport a2_02c_segment_capture radiation_field bf_rate_jnu
 line_jbar seed_capability jnu_seed population_contract atomic_internal_energy
 opacity_publication emissivity_publication radeq_publication line_net_rate
 cmfgen_adiabatic cmf_exact_sliding cmf_error_envelope nlte_population_candidate
 physics_comparison gpu_physics_contract lumina_plasma lumina_element_wide
 lumina_atomic lumina_cmfgen cmfgen_final_writer a210_coupled_newton)
common_files=()
for common_unit in "${common_units[@]}"; do common_files+=("src/${common_unit}.c"); done
sha256sum "${common_files[@]}" src/lumina.h src/lumina_cmfgen.h src/radiation_field.h \
 src/env_universe.h src/cmf_cell_geometry.h tests/cmf_common_field_selftest.c tests/a2_06_dual_commit_selftest.c \
 scripts/run_cmf_common_field_selftest.slurm > "$common_out/source.sha256"
gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections \
 -Isrc tests/cmf_common_field_selftest.c "${common_files[@]}" \
 -Wl,--gc-sections -lm -o "$common_out/test" > "$common_out/build.log" 2>&1
gcc -O2 -std=gnu11 -Isrc tests/a2_06_dual_commit_selftest.c \
 src/radiation_field.c src/seed_capability.c -lm -o "$common_out/dual-test"
sha256sum -c "$common_out/source.sha256" > "$common_out/verify.log"
cd "$common_out"
./test > test.log 2> test.err
./dual-test > dual.log 2> dual.err
echo "result=$common_out"
