#!/bin/bash
# 사용자 승인한 단발 Ada 계산. systemd의 6시간 미만 상한 안에서 실행한다.
set -euo pipefail
umask 027
[[ "$(hostname -s)" == LagEunha && -z "${SLURM_JOB_ID:-}" ]]
ada_out=${1:?new absolute output directory required}
ada_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
ada_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
[[ "$ada_out" =~ ^$ada_base/ada-evolution-[0-9]{8}-[0-9]{4}$ ]]
cd "$ada_src"
[[ "$(git rev-parse --show-toplevel)" == "$ada_src" ]]
# 동일 프로젝트의 Ada 중복 투척을 막는 단일 실행 잠금이다. 재시도 루프는 없다.
exec 9>"$ada_base/.cmf-toy-ada.lock"
flock -n 9
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
mkdir "$ada_out" "$ada_out/runners"
exec >"$ada_out/launch.log" 2>"$ada_out/launch.err"
ada_started=$SECONDS
trap 'ada_rc=$?; printf "RUN_FOOTER exit=%s elapsed_seconds=%s GPUs=1\n" "$ada_rc" "$((SECONDS-ada_started))" > "$ada_out/RUN_FOOTER.txt"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
hostname > "$ada_out/host.txt"
git rev-parse HEAD > "$ada_out/worktree_commit.txt"
date --iso-8601=seconds > "$ada_out/start.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv > "$ada_out/gpu.txt"
cp scripts/run_cmf_toy_initial.slurm scripts/run_cmf_toy_step.slurm \
 scripts/run_cmf_toy_bootstrap.slurm scripts/run_cmf_toy_compare.slurm \
 scripts/run_cmf_toy_ada.sh "$ada_out/runners/"
(cd "$ada_out/runners"; sha256sum -- *.slurm *.sh) > "$ada_out/runners.sha256"
ada_initial_rc=0
env CMF_TOY_RUN_OUT="$ada_out/initial" timeout --signal=TERM --kill-after=15s 4h \
 bash "$ada_out/runners/run_cmf_toy_initial.slurm" \
 "$ada_base/common-build-424525" "$ada_base/toy-material-423395" anderson gpu \
 "$ada_base/nonthermal-423416" "$ada_base/toy-initial-ng-424731" \
 anderson t10 mixture resume > "$ada_out/initial-runner.log" 2> "$ada_out/initial-runner.err" \
 || ada_initial_rc=$?
printf 'exit=%s\nelapsed_seconds=%s\n' "$ada_initial_rc" "$((SECONDS-ada_started))" > "$ada_out/initial_status.txt"
# 계산 종료 후 기존 CPU 판독만 제출한다. 로그인 노드에서는 계산하지 않는다.
if [[ -f "$ada_out/initial/work/initial_iterate.bin" ]]; then
 ssh -o BatchMode=yes -o ConnectTimeout=10 syntax \
 "cd '$ada_src' && sbatch --parsable --job-name=CMF_ADA_EXTRACT '$ada_out/runners/run_cmf_toy_bootstrap.slurm' 100 material-iterate '$ada_out/initial/input' '$ada_out/initial/input' '$ada_out/initial'" \
 > "$ada_out/extract_job.txt" 2> "$ada_out/extract_submit.err" \
 || printf 'CPU 판독 제출 실패: extract_submit.err 확인\n' >&2
fi
if (( ada_initial_rc != 0 )); then
 printf '초기 공동해 미완료: 시간 진화를 시작하지 않는다. exit=%s\n' "$ada_initial_rc"
 exit "$ada_initial_rc"
fi
ada_step_started=$SECONDS
ada_step_rc=0
env CMF_TOY_RUN_OUT="$ada_out/step" timeout --signal=TERM --kill-after=15s 2h \
 bash "$ada_out/runners/run_cmf_toy_step.slurm" \
 "$ada_base/common-build-424525" "$ada_out/initial" 174528 anderson gpu \
 > "$ada_out/step-runner.log" 2> "$ada_out/step-runner.err" || ada_step_rc=$?
printf 'exit=%s\nelapsed_seconds=%s\n' "$ada_step_rc" "$((SECONDS-ada_step_started))" > "$ada_out/step_status.txt"
if (( ada_step_rc == 0 )); then
 ssh -o BatchMode=yes -o ConnectTimeout=10 syntax \
 "cd '$ada_src' && sbatch --parsable --job-name=CMF_ADA_STEP_COMPARE '$ada_out/runners/run_cmf_toy_compare.slurm' '$ada_out/step'" \
 > "$ada_out/compare_job.txt" 2> "$ada_out/compare_submit.err" \
 || printf 'CPU 비교 제출 실패: compare_submit.err 확인\n' >&2
fi
exit "$ada_step_rc"
