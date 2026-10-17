#!/usr/bin/env bash
set -euo pipefail
umask 027
# Q-GOAL=PASS: 실제 Si/S의 국소 결합을 수렴시켜 큰 차이의 원인 후보를 좁힌다.
# Q-LEAN=APPROPRIATE: 기존 율행렬/계수/미분을 재사용하며 생산 계측을 추가하지 않는다.
# 고정T/ne/비열/배경/비국소 잔여항, 전체 공간 공동해 및 최종 판정은 제외한다.
test "$(hostname -s)" = LagEunha
ulimit -v 8388608
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nonlinear_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
nonlinear_repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
case "${1:-nonlinear}" in
 nonlinear) nonlinear_prefix=nonlinear-local; nonlinear_option=--nonlinear ;;
 bands) nonlinear_prefix=frozen-band; nonlinear_option=--band-response ;;
 *) exit 2 ;;
esac
nonlinear_root=$(mktemp -d "$nonlinear_base/$nonlinear_prefix-20261017.XXXXXX")
printf 'NONLINEAR_LOCAL_ROOT=%s\n' "$nonlinear_root"
cd "$nonlinear_root"
cp "$nonlinear_repo/tests/cmf_bf_transport_duality.py" \
   "$nonlinear_repo/tests/cmf_nonlinear_local_feedback.py" \
   "$nonlinear_repo/tests/cmf_frozen_band_response.py" \
   "$nonlinear_repo/tests/run_cmf_nonlinear_local_feedback.sh" .
sha256sum ./*.py ./*.sh > source.sha256
# 선행 시험의 고정 입력 전체를 다시 대조한다. 경로는 해당 명시적 JSON만 사용한다.
python3 -B - "$nonlinear_base/bf-duality-verified-20261017.vuJ2bX/summary.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
for name, expected in json.loads(Path(sys.argv[1]).read_text())['sha256'].items():
    with Path(name).open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    assert actual == expected, name
print('SEALED_INPUTS_VERIFIED', flush=True)
PY
date -Is > started.txt
set +e
timeout 8m python3 -B cmf_bf_transport_duality.py summary.json --feedback "$nonlinear_option" > analysis.log 2> analysis.err
nonlinear_rc=$?
set -e
date -Is > finished.txt
printf 'NONLINEAR_LOCAL_EXIT=%d\n' "$nonlinear_rc"
sha256sum -c source.sha256
if test "$nonlinear_rc" -eq 0; then
    python3 -B - "$nonlinear_base/bf-duality-verified-20261017.vuJ2bX/summary.json" <<'PY'
import json, sys
from pathlib import Path
before = json.loads(Path(sys.argv[1]).read_text())
after = json.loads(Path('summary.json').read_text())
for key in ('ions', 'local_feedback'):
    assert before[key] == after[key], key
print('PRIOR_RESULT_UNCHANGED')
PY
    sha256sum summary.json analysis.log analysis.err > output.sha256
fi
exit "$nonlinear_rc"
