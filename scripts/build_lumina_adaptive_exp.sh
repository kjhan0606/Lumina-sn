#!/bin/bash
# GPU 문맥 없이 선택형 급수의 strict CUDA 바이너리만 만든다.
source /etc/profile
module purge
module load gnu13 cuda/13.0.2
set -euo pipefail
run_root=$1
cd "$run_root"
test ! -e diagnostic.so
test ! -e compile.log
ulimit -v 16777216
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout -k 5 240 taskset -c 0 nvcc -O2 -std=c++17 -arch=sm_89 --threads=1 \
    --fmad=false --ftz=false --prec-div=true --prec-sqrt=true -DCMF_GPU_LINEAR \
    -DCMF_GPU_SOURCE_MEASURE -DCMF_GPU_COMPACT_MEASURE -DCMF_GPU_DIAGNOSTIC_RANGE \
    -DCMF_GPU_ENVELOPE_UPPER -DCMF_GPU_ADAPTIVE_LINEAR_EXP -Xptxas=-v \
    -Xcompiler=-fPIC,-ffp-contract=off,-Wall,-Wextra,-Werror -Isrc -shared \
    src/cmf_linear_cuda.cu src/cmf_linear_cuda_aux.cu src/cmf_gpu_runtime.cu -o diagnostic.so >compile.log 2>&1
