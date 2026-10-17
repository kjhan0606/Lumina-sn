#!/usr/bin/env bash
set -euo pipefail
umask 027
# Q-GOAL=PASS: 누락 정상 이류항의 국소 물질-장 증폭이 큰 S 차이를 설명하는지 분리.
# Q-LEAN=APPROPRIATE: 기존 행렬/Jacobian/저장장 재사용. CPU2/8GiB/10분/GPU0.
# 새 RTE/물리해/생산 입력 변경 없음. 원소 보존과 기존 미분 검사를 유지한다.
test "$(hostname -s)" = LagEunha
ulimit -v 8388608
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ar_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
ar_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
ar_nt=$ar_base/nt-line-response-20261016.3ZVCFq
ar_root=$(mktemp -d "$ar_base/advection-response-20261017.XXXXXX")
printf 'ADVECTION_RESPONSE_ROOT=%s\n' "$ar_root"
cd "$ar_root"
cp "$ar_src/tests/cmf_local_feedback.py" "$ar_src/tests/run_cmf_advection_response.sh" .
cp "$ar_nt/"matrix_Z*.bin "$ar_nt/"levels_Z*.csv "$ar_nt/"lines_Z*.csv "$ar_nt/nonthermal_rates.csv" .
sha256sum ./*.py ./*.sh > source.sha256
sha256sum ./*.bin ./*.csv > copied_inputs.sha256
(cd "$ar_nt";sha256sum -c output.sha256) > original_inputs_verify.log
sha256sum "$ar_base/toy-material-425144/iterate_input.bin" \
 "$ar_base/toy-material-425144/input/levels.csv" "$ar_base/toy-material-425144/input/bf_channels.bin" \
 "$ar_base/fine-material-20261016.z5VQYD/fine_shell39.bin" > inputs.sha256
date -Is > started.txt
python3 -B cmf_local_feedback.py "$ar_root" --bf --nt --advection > analysis.log 2> analysis.err
sha256sum -c source.sha256 > source_verify.log
sha256sum -c copied_inputs.sha256 > copied_inputs_verify.log
sha256sum -c inputs.sha256 > inputs_verify.log
sha256sum summary_bf.json analysis.log analysis.err > output.sha256
date -Is > finished.txt
cat summary_bf.json
