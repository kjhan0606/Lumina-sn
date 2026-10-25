#!/bin/bash
# grammar CPU Slurm에서만 실행. GPU 요청/사용과 자동 재시도는 없다.
source /etc/profile
module purge
module load gnu13
set -euo pipefail
lumina_sc_root="$1"
shift
cd "$lumina_sc_root"
test ! -e fullsr_sc.so
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=16
export OMP_PROC_BIND=close OMP_PLACES=cores
timeout -k 5 90 c++ -O3 -std=c++17 -ffp-contract=off -fopenmp -Wall -Wextra -Werror \
    -fPIC -shared code/src/cmf_fullsr_sc.cpp -o fullsr_sc.so >compile.log 2>&1
sha256sum fullsr_sc.so
timeout -k 5 2600 python3 code/scripts/lumina_fullsr_sc_actual.py "$lumina_sc_root" "$@"
