#!/usr/bin/env bash
set -euo pipefail
umask 027
# Q-GOAL=PASS: 소광/전이율 불일치가 Si/S 반응에 미치는 조건부 영향을 분리.
# Q-LEAN=APPROPRIATE: 기존 저장장·행렬·국소 연산자 재사용, 새 GPU/RTE 없음.
# CPU2/8GiB, 외부 timeout 5m. 원자자료/생산 소스/기존 산출 변경 없음.
test "$(hostname -s)" = LagEunha
ulimit -v 8388608
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ef_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
ef_prior=$ef_base/nt-line-response-20261016.3ZVCFq
ef_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
ef_root=$(mktemp -d "$ef_base/einstein-feedback-20261017.XXXXXX")
printf 'EINSTEIN_FEEDBACK_ROOT=%s\n' "$ef_root"
cd "$ef_root"
cp "$ef_src/tests/cmf_local_feedback.py" analysis.py
cp "$ef_src/tests/run_cmf_einstein_feedback.sh" run.sh
for ef_name in matrix_Z14.bin matrix_Z16.bin levels_Z14.csv levels_Z16.csv lines_Z14.csv lines_Z16.csv nonthermal_rates.csv; do
    ln -s "$ef_prior/$ef_name" "$ef_name"
done
sha256sum ./*.bin ./*.csv analysis.py run.sh \
 "$ef_base/toy-material-425144/iterate_input.bin" \
 "$ef_base/toy-material-425144/input/line_list.csv" \
 "$ef_base/toy-material-425144/input/levels.csv" \
 "$ef_base/toy-material-425144/input/bf_channels.bin" \
 "$ef_base/fine-material-20261016.z5VQYD/fine_shell39.bin" \
 "$ef_base/fine-material-20261016.z5VQYD/lines.csv" > inputs.sha256
date -Is > started.txt
python3 -B analysis.py "$ef_root" --bf --nt --einstein > analysis.log 2> analysis.err
sha256sum -c inputs.sha256 > input_verify.log
sha256sum summary_bf.json analysis.log analysis.err > output.sha256
date -Is > finished.txt
cat summary_bf.json
