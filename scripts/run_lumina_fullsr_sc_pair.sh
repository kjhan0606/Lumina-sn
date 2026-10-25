#!/bin/bash
# 세 번째 CPU 할당: 동일 coarse 설정 J2와 J1/J2/CMFGEN 율을 한 부모 한도 안에서 비교한다.
source /etc/profile
module purge
module load gnu13
set -euo pipefail
lumina_pair_root="$1"
lumina_first_root="$2"
lumina_first_sha="$3"
lumina_pair_start=$SECONDS
cd "$lumina_pair_root"
test ! -e fullsr_sc.so
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=16
export OMP_PROC_BIND=close OMP_PLACES=cores
timeout -k 5 90 c++ -O3 -std=c++17 -ffp-contract=off -fopenmp -Wall -Wextra -Werror \
    -fPIC -shared code/src/cmf_fullsr_sc.cpp -o fullsr_sc.so >compile.log 2>&1
lumina_field_seconds=$((2650-SECONDS+lumina_pair_start-360))
if ((lumina_field_seconds>2280)); then lumina_field_seconds=2280; fi
if ((lumina_field_seconds<60)); then exit 3; fi
timeout -k 5 "$((lumina_field_seconds+20))" python3 code/scripts/lumina_fullsr_sc_actual.py \
    "$lumina_pair_root" --seconds "$lumina_field_seconds" --threads 16 \
    --input-j "$lumina_first_root/J.npy" --input-j-sha256 "$lumina_first_sha"
lumina_rate_seconds=$((2650-SECONDS+lumina_pair_start-10))
if ((lumina_rate_seconds>1200)); then lumina_rate_seconds=1200; fi
if ((lumina_rate_seconds<300)); then
    printf '%s\n' 'J2는 보존했지만 같은 부모 한도의 율 비교 시간이 부족하다.'
    exit 3
fi
timeout -k 5 "$((lumina_rate_seconds+5))" python3 code/scripts/compare_lumina_fullsr_sc.py \
    "$lumina_first_root/sc-result.json" "$lumina_pair_root/rates" \
    --second "$lumina_pair_root/sc-result.json" --seconds "$lumina_rate_seconds"
