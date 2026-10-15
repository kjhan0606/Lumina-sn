#!/bin/bash
#SBATCH --job-name=TOY_ION_CEILING
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-ceiling-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-ceiling-%j.err
set -euo pipefail
umask 027
ceiling_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
ceiling_root=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
ceiling_out=$ceiling_root/toy-ceiling-${SLURM_JOB_ID}
ceiling_original=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/model
mkdir "$ceiling_out"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cp "$ceiling_src/scripts/build_toy_ion_ceiling.py" "$ceiling_src/scripts/run_toy_ion_ceiling.slurm" "$ceiling_out/"
(cd "$ceiling_src"; git rev-parse HEAD) > "$ceiling_out/commit.txt"
(cd "$ceiling_out"; sha256sum build_toy_ion_ceiling.py run_toy_ion_ceiling.slurm > source.sha256)
/home/kjhan/miniconda3/bin/python3 "$ceiling_out/build_toy_ion_ceiling.py" \
 "$ceiling_original" "$ceiling_out/input" > "$ceiling_out/subset.json"
bash "$ceiling_src/scripts/run_bf_channels_input.slurm" "$ceiling_out/input"
bash "$ceiling_src/scripts/run_cmf_toy_bootstrap.slurm" 100 initial-prepare \
 "$ceiling_out/input" "$ceiling_root/bf-input-${SLURM_JOB_ID}"
(cd "$ceiling_out"; sha256sum -c source.sha256 > source_verify.log)
echo 'ION_CEILING_PREPARED: not full published atomic model; initial radiation not solved'
