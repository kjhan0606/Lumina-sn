#!/bin/bash
#SBATCH --job-name=CMF_OPERATOR_BUILD
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/operator-build-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/operator-build-%j.err
set -euo pipefail
umask 027
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
op_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
op_prod=$op_base/common-build-425846
op_root=$op_base/operator-build-${SLURM_JOB_ID:?}
mkdir "$op_root"
cp /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015/tests/cmf_operator_norm_probe.cu "$op_root/"
cp "$0" "$op_root/build.slurm"
cd "$op_root"
(cd "$op_prod/source";sha256sum -c "$op_prod/source.sha256") > production_source_verify.log
sha256sum cmf_operator_norm_probe.cu build.slurm > source.sha256
date -Is > started.txt
nvcc -O2 -arch=sm_89 -std=c++14 -Xcompiler -fopenmp \
 -DLUMINA_CMF_EXACT_MANAGED_MEMORY=1 -I"$op_prod/source/src" \
 cmf_operator_norm_probe.cu "$op_prod/source/src/cmf_exact_sliding.c" \
 "$op_prod/source/src/cmf_error_envelope.c" -lm -o probe > build.log 2> build.err
sha256sum -c source.sha256 > source_verify.log
sha256sum probe > binary.sha256
date -Is > finished.txt
printf 'OPERATOR_BUILD_COMPLETE root=%s GPU_execution=0\n' "$op_root"
