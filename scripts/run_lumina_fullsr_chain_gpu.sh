#!/bin/bash
# syntax GPU Slurm의 단일 할당. CPU1·host/device16GiB·전체10분 안에서만 실행한다.
source /etc/profile
module purge
module load gnu13 cuda/13.0.2
set -euo pipefail
cd "$1"
lumina_chain_parent="${2:-600}"
test "$lumina_chain_parent" -ge 60
test "$lumina_chain_parent" -le 600
lumina_chain_mode="${3:-samples}"
lumina_chain_start=$SECONDS
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_MODULE_LOADING=EAGER
if test "$lumina_chain_mode" = full; then
    test "$lumina_chain_parent" -le 480
    test -f chain_cuda.so
    test -f frozen.so
    lumina_chain_args=(--full-interval)
else
test "$lumina_chain_mode" = samples
test ! -e chain_cuda.so
test ! -e frozen.so
lumina_chain_args=()
timeout -k 5 120 nvcc -O2 -std=c++17 -arch=sm_80 --threads=1 --fmad=false --ftz=false \
    --prec-div=true --prec-sqrt=true -Xcompiler=-fPIC,-ffp-contract=off,-Wall,-Wextra,-Werror \
    -DCMF_GPU_FULLSR_CHAIN -DCMF_GPU_TOTAL_CAP_RESERVATION -Xptxas=-v -Icode/src -shared \
    code/src/cmf_fullsr_chain.cu code/src/cmf_gpu_runtime.cu -o chain_cuda.so >compile-gpu.log 2>&1
timeout -k 5 60 c++ -O3 -std=c++17 -ffp-contract=off -Wall -Wextra -Werror -fopenmp \
    -fPIC -shared code/src/cmf_fullsr_sc.cpp code/src/cmf_fullsr_frozen.cpp -o frozen.so >compile-cpu.log 2>&1
fi
lumina_chain_remaining=$((lumina_chain_parent-20-(SECONDS-lumina_chain_start)))
test "$lumina_chain_remaining" -gt 0
timeout -k 5 "$lumina_chain_remaining" python3 code/scripts/run_lumina_fullsr_chain_gpu.py "$PWD" --parent-seconds "$lumina_chain_parent" "${lumina_chain_args[@]}"
