#!/bin/bash
# ARTIS-parity comprehensive run #1 (A-E+DIAG, binary lumina_cuda.withParity c0b06239)
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 1
export OMP_NUM_THREADS=32
export LUMINA_ARTIS_PARITY=1
export LUMINA_BIN=${LUMINA_BIN:-lumina_cuda.withParity}
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=0
export PKTS=${PK0:-100000} NITER=${NI0:-12} P0TAG=${PTAG:-parity1}
exec bash scripts/run_coevolve_s01.sh consume
