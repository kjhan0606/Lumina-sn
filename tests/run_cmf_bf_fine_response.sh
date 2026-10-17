#!/usr/bin/env bash
set -euo pipefail
umask 027
# Q-GOAL=PASS: 같은 저장장에서 BF 평균화가 Si/S 비에 미치는 영향을 분리한다.
# Q-LEAN=APPROPRIATE: CPU2/2GiB/10분, 기존 한 셸, GPU0, 새 복사장 없음.
test "$(hostname -s)" = LagEunha
ulimit -v 2097152
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
bf_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
bf_root=$(mktemp -d "$bf_base/bf-fine-response-20261017.XXXXXX")
printf 'BF_FINE_RESPONSE_ROOT=%s\n' "$bf_root"
cd "$bf_root"
cp /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015/tests/cmf_bf_fine_response.py .
cp /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015/tests/run_cmf_bf_fine_response.sh .
sha256sum ./*.py ./*.sh > source.sha256
sha256sum "$bf_base/toy-material-425144/iterate_input.bin" "$bf_base/toy-material-425144/input/bf_channels.bin" \
 "$bf_base/toy-material-425144/input/levels.csv" "$bf_base/nt-line-response-20261016.3ZVCFq/"matrix_Z*.bin \
 "$bf_base/nt-line-response-20261016.3ZVCFq/"levels_Z*.csv > inputs.sha256
date -Is > started.txt
python3 -B cmf_bf_fine_response.py > analysis.log 2> analysis.err
sha256sum -c source.sha256 > source_verify.log
sha256sum -c inputs.sha256 > input_verify.log
sha256sum summary.json analysis.log analysis.err > output.sha256
date -Is > finished.txt
cat summary.json
