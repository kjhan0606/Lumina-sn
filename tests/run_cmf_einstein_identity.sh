#!/usr/bin/env bash
set -euo pipefail
umask 027
test "$(hostname -s)" = LagEunha
ulimit -v 8388608
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ei_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
ei_root=$(mktemp -d /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/einstein-identity-20261017.XXXXXX)
printf 'EINSTEIN_IDENTITY_ROOT=%s\n' "$ei_root"
cd "$ei_src"
cp --parents src/opacity_publication.c src/opacity_publication.h src/line_net_rate.c src/line_net_rate.h \
 tests/a2_08_signed_opacity_selftest.c tests/line_net_rate_selftest.c \
 tests/cmf_einstein_material_identity.py tests/run_cmf_einstein_identity.sh "$ei_root"
cd "$ei_root"
sha256sum src/* tests/* > source.sha256
date -Is > started.txt
gcc -O2 -std=gnu11 -Wall -Wextra -Isrc tests/a2_08_signed_opacity_selftest.c src/opacity_publication.c -lm -o opacity_test > build.log 2>&1
gcc -O2 -std=gnu11 -Wall -Wextra -Isrc tests/line_net_rate_selftest.c src/line_net_rate.c -lm -o line_test >> build.log 2>&1
gcc -O2 -std=gnu11 -Wall -Wextra -shared -fPIC -Isrc src/opacity_publication.c src/line_net_rate.c -lm -o identity.so >> build.log 2>&1
./opacity_test > opacity.log 2>&1
./line_test > line.log 2>&1
python3 -B tests/cmf_einstein_material_identity.py "$ei_root/identity.so" > identity.log 2> identity.err
sha256sum -c source.sha256 > source_verify.log
sha256sum opacity_test line_test identity.so opacity.log line.log identity.log identity.err > output.sha256
date -Is > finished.txt
cat identity.log
