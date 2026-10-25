#!/bin/bash
# 별도 승인·예약한 syntax Slurm GPU1/CPU1/16GiB의 단회 전체장 실행만 한다.
source /etc/profile
module purge
module load gnu13 cuda/13.0.2
set -euo pipefail
lumina_resident_root="$1"
lumina_resident_parent="$2"
shift 2
test -n "${SLURM_JOB_ID:-}"
test "$lumina_resident_parent" -ge 240
test "$lumina_resident_parent" -le 86400
cd "$lumina_resident_root"
# 검증한 바이너리를 실행 전에 이 새 root에 고정한다. 여기서 다시 빌드/재시도하지 않는다.
test -f chain_cuda.so
test -f geometry.so
test ! -e resident-field.json
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_MODULE_LOADING=EAGER
timeout -k 5 "$((lumina_resident_parent-30))" python3 code/scripts/lumina_resident_run.py "$PWD" \
    --seconds "$((lumina_resident_parent-180))" "$@"
