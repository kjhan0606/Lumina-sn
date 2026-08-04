#!/bin/bash
# parity34-consume: un-HOLD the deterministic line field at the POPULATION solve.
# Single variable vs parity33 = LUMINA_CMF_LINERES_CONSUME=1 (plasma.c:13004).
#
# What was already on (do not confuse): LUMINA_CMF_LINERES_JBAR=2 (set by
# run_coevolve_s01.sh) is the PRODUCER (cuda.cu:7036, fills jbar_line_det on the
# FINAL pure iteration only) plus the up-rate consumer (plasma.c:3379). The
# population-solve consumer is a separate gate and has been OFF for the whole
# parity campaign -- that is the HOLD being lifted here.
#
# Offline verification on parity33 artifacts (s8, in-window 1000-4000A):
#   785,887 of 786,556 in-window lines (99.915%) have <3 MC crossings, so today
#   they pump off the BINNED continuum J. Switching them to the repaired det
#   field lowers the pump by median 13.0x (10.6/16.3/18.0/2.4x by band).
#   The 669 MC-sampled lines change by only +-25%. So this is a large, mostly
#   binned->det substitution, in the direction of CMFGEN truth (J_binned/truth
#   was 5-22x too hot; J_fine/truth is now 0.31-1.90).
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 1
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -8 | tr '\n' '; ')"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_BIN=lumina_cuda.withParityS
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export PKTS=100000 NITER=12 P0TAG=parity34
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
export LUMINA_CMF_LINERES_CONSUME=1
exec bash scripts/run_coevolve_s01.sh consume
