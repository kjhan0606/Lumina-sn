#!/bin/bash
# CPU 전용 strict 빌드. 기본 경로와 협력 커널을 모두 확인한다.
source /etc/profile
module purge
module load gnu13 cuda/13.0.2
set -euo pipefail
cd "$1"
test ! -e diagnostic.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -v 16777216
timeout -k 5 120 taskset -c 25 nvcc -O2 -std=c++17 -arch=sm_80 --threads=1 \
    --fmad=false --ftz=false -DCMF_GPU_LINEAR -DCMF_GPU_SOURCE_MEASURE -DCMF_GPU_COMPACT_MEASURE \
    -Xcompiler=-fPIC,-ffp-contract=off,-Wall,-Wextra,-Werror -Isrc -c src/cmf_linear_cuda.cu \
    -o original-linear.o >original-compile.log 2>&1
timeout -k 5 240 taskset -c 25 nvcc -O2 -std=c++17 -arch=sm_80 --threads=1 \
    --fmad=false --ftz=false --prec-div=true --prec-sqrt=true -DCMF_GPU_LINEAR \
    -DCMF_GPU_SOURCE_MEASURE -DCMF_GPU_COMPACT_MEASURE -DCMF_GPU_DIAGNOSTIC_RANGE \
    -DCMF_GPU_ENVELOPE_UPPER -DCMF_GPU_ADAPTIVE_LINEAR_EXP -DCMF_GPU_TOTAL_CAP_RESERVATION \
    -DCMF_GPU_ENDPOINT_CACHE -DCMF_GPU_CHI_ONLY_TAU -DCMF_GPU_WARP_MEASURE -Xptxas=-v \
    -Xcompiler=-fPIC,-ffp-contract=off,-Wall,-Wextra,-Werror -Isrc -shared \
    src/cmf_linear_cuda.cu src/cmf_linear_cuda_aux.cu src/cmf_gpu_runtime.cu tests/warp_measure_probe.cu \
    -o diagnostic.so >compile.log 2>&1
