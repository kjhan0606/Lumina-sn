#!/bin/bash
#SBATCH --job-name=parity28_binfield
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
# parity27-tepin: single physics delta vs parity26 = LUMINA_C1_SUPERBIN_TEPIN
# (ARTIS radfield.cc superbin semantics: coarse bins entirely <1085A take
# T_R:=T_e, W:=J/intB — kills the C1 Wien-extrapolated EUV hallucination pump
# behind Si III b4=3.08, dig_F13/F14). Observation adds: C1_BIN_DUMP (per-bin
# W/T_R/mode CSV, closes dig_F13 open item) + bb-route kpkt tag 0x16 (in-binary,
# dig_F14 router direct measurement). FINAL_RESOLVE + JBAR_DUMP kept for the
# re-solve b4 battery. Binary withParityR (d32ceba9), P physics otherwise.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_BIN=lumina_cuda.withParityR
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export PKTS=100000 NITER=12 P0TAG=parity28
export LUMINA_MA_REAL_UPSILON=1 LUMINA_MA_LINE_DESTRUCT=1 LUMINA_ALPHA_SPINGATE=1
export LUMINA_SIMUL_CAP_TOPION=1 LUMINA_FB_COOL_KT=1 LUMINA_RADEQ_OMEGA_FLOOR=1
export LUMINA_MA_RADRECOMB=1 LUMINA_C1_DEGEN_FALLBACK=1 LUMINA_SUPER_LEVELS=1
export LUMINA_GPH_SIGMA_CMFGEN=1 LUMINA_GPH_ALLLEVEL=1 LUMINA_GPH_ALLLEVEL_NLTE=1
export LUMINA_EVENT_LOG_CAP=400 LUMINA_JNU_FINE_DUMP=1
export LUMINA_NLTE_FINAL_RESOLVE=1 LUMINA_JBAR_DUMP=1
export LUMINA_C1_SUPERBIN_TEPIN=1 LUMINA_C1_BIN_DUMP=1
export LUMINA_IUP_BINFIELD=1
exec bash scripts/run_coevolve_s01.sh consume
