#!/usr/bin/env bash
set -euo pipefail
umask 027
test "$(hostname -s)" = LagEunha
ulimit -v 2097152
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ap_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
ap_root=$(mktemp -d /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/acceleration-probe-20261017.XXXXXX)
printf 'ACCELERATION_PROBE_ROOT=%s\n' "$ap_root"
cd "$ap_root"
cp "$ap_src/tests/cmf_acceleration_probe.c" "$ap_src/tests/cmf_acceleration_probe.py" \
 "$ap_src/tests/cmf_anderson_multi_selftest.c" \
 "$ap_src/tests/run_cmf_acceleration_probe.sh" .
for ap_file in cmf_exact_sliding.c cmf_exact_sliding.h cmf_cell_geometry.h cmf_anderson.h cmf_anderson_multi.h cmf_error_envelope.c cmf_error_envelope.h; do
 cp "$ap_src/src/$ap_file" .
done
sha256sum ./* > source.sha256
date -Is > started.txt
gcc -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -fsanitize=undefined -fno-sanitize-recover=all \
 -I. cmf_anderson_multi_selftest.c -lm -o helper > helper-build.log 2>&1
./helper > helper.log 2>&1
gcc -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -fopenmp -fPIC -shared -I. \
 cmf_acceleration_probe.c cmf_exact_sliding.c cmf_error_envelope.c -lm -o probe.so > build.log 2>&1
python3 -B cmf_acceleration_probe.py > probe.log 2> probe.err
sha256sum -c source.sha256 > source_verify.log
sha256sum probe.so probe.json probe.log probe.err > output.sha256
date -Is > finished.txt
cat probe.log
