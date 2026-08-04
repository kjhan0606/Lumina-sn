#!/bin/bash
#SBATCH --job-name=toy
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/toy/%x_%j.out
ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
cd $ROOT
T=${TOYDIR}
mkdir -p logs/toy/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
cd logs/toy/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
env LUMINA_PURE_CMFGEN=1 LUMINA_PURE_CMFGEN_ITER=6 LUMINA_BF_OPACITY=1 \
    LUMINA_CMFGEN_SIGMA_BF=$ROOT/$T/cmfgen_sigma_bf.bin \
    LUMINA_RADEQ_TE=1 LUMINA_NLTE_COLL_FIX=1 LUMINA_NLTE_ION_LOCK=1 \
    ${EXTRA_ENV} \
    $ROOT/lumina_cuda $ROOT/$T 2000 6 virtual nlte > stdout.log 2> stderr.log
echo "done: $(pwd)"
