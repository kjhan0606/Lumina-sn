#!/bin/bash
#SBATCH --job-name=a10_kx_jblbid
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# [IUP-JBLUE] OFF-path bit-identity seal: run the EXACT a10_kx consume config
# (gate LUMINA_IUP_JBLUE unset) on the new binary and on ./lumina_cuda.preJblue,
# then diff plasma_state + formal spectrum + MC emergent. Must be BYTE-IDENTICAL:
# the gate-off path adds no estimator tallies, no allocations, no RNG-order change.
# Small config (PKTS=100000 NITER=4) — identity, not physics, is the question.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=100000 NITER=4
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
unset LUMINA_IUP_JBLUE

for BIN in ./lumina_cuda ./lumina_cuda.preJblue; do
  TAG=$(basename "$BIN")
  rm -f lumina_spectrum_formal.csv lumina_plasma_state.csv lumina_spectrum_coevolve_mc.csv
  ( export P0TAG="jblbid_$TAG"
    export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
    export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
    export LUMINA_RUN_BIN="$BIN"
    sed "s|./lumina_cuda |\"\$LUMINA_RUN_BIN\" |" scripts/run_coevolve_s01.sh > /tmp/run_coevolve_byteid_$$.sh
    bash /tmp/run_coevolve_byteid_$$.sh consume
    rm -f /tmp/run_coevolve_byteid_$$.sh )
  OUT=logs/coevolve_consume_jblbid_$TAG
  for f in lumina_spectrum_coevolve_mc.csv; do
    [ -f "$f" ] && cp -f "$f" "$OUT/$f"
  done
done

echo "=== [JBLUE-BYTEID] diff plasma_state ==="
diff logs/coevolve_consume_jblbid_lumina_cuda/lumina_plasma_state.csv \
     logs/coevolve_consume_jblbid_lumina_cuda.preJblue/lumina_plasma_state.csv \
     && echo "BYTE-IDENTICAL plasma_state OK" || echo "DIFFERS: plasma_state FAIL"
echo "=== [JBLUE-BYTEID] diff spectrum_formal ==="
diff logs/coevolve_consume_jblbid_lumina_cuda/lumina_spectrum_formal.csv \
     logs/coevolve_consume_jblbid_lumina_cuda.preJblue/lumina_spectrum_formal.csv \
     && echo "BYTE-IDENTICAL spectrum OK" || echo "DIFFERS: spectrum FAIL"
echo "=== [JBLUE-BYTEID] diff MC emergent ==="
diff logs/coevolve_consume_jblbid_lumina_cuda/lumina_spectrum_coevolve_mc.csv \
     logs/coevolve_consume_jblbid_lumina_cuda.preJblue/lumina_spectrum_coevolve_mc.csv \
     && echo "BYTE-IDENTICAL MC emergent OK" || echo "DIFFERS: MC emergent FAIL"
echo "a10_kx_jblbid DONE"
