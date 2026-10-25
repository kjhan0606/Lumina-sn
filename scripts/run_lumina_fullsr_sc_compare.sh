#!/bin/bash
# grammar CPU 단일 작업에서 완성 자체장만 같은 물질의 기존 BF 소비자로 비교한다.
source /etc/profile
module purge
module load gnu13
set -euo pipefail
lumina_compare_root="$1"
lumina_first_result="$2"
cd "$lumina_compare_root"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
timeout -k 5 1240 python3 code/scripts/compare_lumina_fullsr_sc.py \
    "$lumina_first_result" "$lumina_compare_root/rates" --seconds 1200
