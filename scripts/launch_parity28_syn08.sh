#!/bin/bash
# parity28-binfield on syn08 (manual, GPU set by caller via CUDA_VISIBLE_DEVICES).
# Mirror of scripts/sbatch_parity27_tepin.sh — single physics delta vs parity26 =
# LUMINA_C1_SUPERBIN_TEPIN; observation adds C1_BIN_DUMP + 0x16 tag (in-binary).
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 1
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -6 | tr '\n' '; ')"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_BIN=lumina_cuda.withParityR
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
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
