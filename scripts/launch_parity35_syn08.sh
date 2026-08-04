#!/bin/bash
# parity35-slwrite: repair (A) judgment run — silicon gets an NLTE line source.
# Single effective variable vs parity33 = LUMINA_SL_WRITE_SKIPZ=1 on the
# withParityT binary (T_off was verified equivalent to parity33's withParityS to
# within 4 printf-midpoint 1-ULP rows; see judge_smoke_slwrite.py R1).
# NO LINERES_CONSUME here — mode-2 remains rejected (catastrophic cancellation).
#
# What changes physically: Si II (790 lines) + Si III (669 lines) in-window stop
# emitting as LTE B(T_e) in the binned + fine solves and use the NLTE source
# S_l = 2hv^3/c^2 / (g_u n_l / g_l n_u - 1) instead. Nebular tau is preserved
# (smoke R4: 1639 Si III betas bit-identical with the gate on).
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 1
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -8 | tr '\n' '; ')"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_SL_WRITE_SKIPZ=1
export LUMINA_BIN=lumina_cuda.withParityT
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export PKTS=100000 NITER=12 P0TAG=parity35
export LUMINA_MA_REAL_UPSILON=1 LUMINA_MA_LINE_DESTRUCT=1 LUMINA_ALPHA_SPINGATE=1
export LUMINA_SIMUL_CAP_TOPION=1 LUMINA_FB_COOL_KT=1 LUMINA_RADEQ_OMEGA_FLOOR=1
export LUMINA_MA_RADRECOMB=1 LUMINA_C1_DEGEN_FALLBACK=1 LUMINA_SUPER_LEVELS=1
export LUMINA_GPH_SIGMA_CMFGEN=1 LUMINA_GPH_ALLLEVEL=1 LUMINA_GPH_ALLLEVEL_NLTE=1
export LUMINA_EVENT_LOG_CAP=400 LUMINA_JNU_FINE_DUMP=1
export LUMINA_NLTE_FINAL_RESOLVE=1 LUMINA_JBAR_DUMP=1
export LUMINA_C1_SUPERBIN_TEPIN=1 LUMINA_C1_BIN_DUMP=1
export LUMINA_RADEQ_DB_FB=1
export LUMINA_CMF_ADV_SPLIT=1 LUMINA_CMF_FINE_ALI=20000
export LUMINA_LINE_THERM=1 LUMINA_LINE_THERM_SMAX=49
export LUMINA_CMF_FINE_LINEDUMP=1 LUMINA_CMF_FINE_LINEDUMP_SHELL=8

exec bash scripts/run_coevolve_s01.sh consume
