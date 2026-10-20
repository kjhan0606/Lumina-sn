#!/bin/bash
# 새 격리 디렉터리에서만 실행한다. 계산 노드에서 호출할 것.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo '사용법: build_cmfgen_source_kernels.sh CMFGEN_SOURCE REPOSITORY NEW_OUTPUT' >&2
    exit 2
fi
source_root=$(realpath "$1")
repo_root=$(realpath "$2")
build_root=$3
mkdir "$build_root"
cp "$source_root/tools/set_kind_module.f" "$source_root/subs/gff.f" \
   "$repo_root/scripts/cmfgen_source_native.f90" \
   "$repo_root/src/model_source_kernels.c" "$build_root/"
cd "$build_root"
sha256sum set_kind_module.f gff.f cmfgen_source_native.f90 model_source_kernels.c > source.sha256
gfortran -O2 -fPIC -ffixed-line-length-none -c set_kind_module.f gff.f
gfortran -O2 -fPIC -c cmfgen_source_native.f90
gcc -O2 -std=c11 -Wall -Wextra -fopenmp -fPIC -c model_source_kernels.c
gfortran -shared -fopenmp -o model_sources.so set_kind_module.o gff.o \
    cmfgen_source_native.o model_source_kernels.o
sha256sum model_sources.so
