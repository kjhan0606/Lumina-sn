#!/usr/bin/env bash
set -euo pipefail
umask 027
test "$(hostname -s)" = LagEunha
aa_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
aa_build=${1:?frozen build required}
[[ "$aa_build" =~ ^$aa_base/acceleration-gpu-build-20261017\.[A-Za-z0-9]+$ ]]
exec 9> "$aa_base/.cmf-toy-ada.lock"
flock -n 9
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
(cd "$aa_build"; sha256sum -c source.sha256; sha256sum -c binary.sha256)
aa_root=$(mktemp -d "$aa_base/acceleration-gpu-run-20261017.XXXXXX")
printf 'GPU_ACCELERATION_RUN_ROOT=%s\n' "$aa_root"
cd "$aa_root"
cp "$aa_build/source.sha256" "$aa_build/binary.sha256" .
cp /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015/tests/run_cmf_acceleration_gpu_ada.sh .
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0 CMF_MGPU_TEST_DEVICES=1
ulimit -v 134217728
while read -r aa_key; do unset "$aa_key"; done < <(compgen -v LUMINA_ || true)
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv > gpu.csv
date -Is > started.txt
aa_start=$SECONDS
set +e
timeout -k 5s 180s bash -c '
 set -e
 "$1/cmf_acceleration_gpu" > acceleration.log 2> acceleration.err
 env LUMINA_CMF_ANDERSON1=1 LUMINA_CMF_ANDERSON_DEPTH=1 "$1/cmf_exact_multigpu_selftest" > depth1.log 2> depth1.err
 env LUMINA_CMF_ANDERSON1=1 LUMINA_CMF_ANDERSON_DEPTH=4 "$1/cmf_exact_multigpu_selftest" > depth4.log 2> depth4.err
' bash "$aa_build"
aa_rc=$?
set -e
printf 'exit=%s elapsed_seconds=%s gpu_count=1\n' "$aa_rc" "$((SECONDS-aa_start))" > RUN_FOOTER.txt
date -Is > finished.txt
(cd "$aa_build"; sha256sum -c source.sha256; sha256sum -c binary.sha256) > frozen_verify.log
sha256sum ./*.log ./*.err ./*.txt ./*.sh gpu.csv > output.sha256
cat RUN_FOOTER.txt
if [[ "$aa_rc" == 0 ]]; then
 grep -E 'GPU_ACCELERATION|SELFTEST PASS' acceleration.log depth1.log depth4.log
else
 tail -50 ./*.err
fi
exit "$aa_rc"
