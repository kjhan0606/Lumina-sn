#!/usr/bin/env bash
set -euo pipefail
umask 027
# 전체 BB 순·역과정 수정본의 새 자기일관 초기해 시도. CPU8/GPU1/128GiB/최대6시간.
# 일곱 번째 인자는 실행 초 상한(기본7200, 종료 유예15초는 별도 예약).
# fresh는 LTE/Saha 추정, guess는 이전 미수렴 물질 추정이며 모두 새 복사장을 푼다.
# resume는 네 번째 인자의 동일 정체성 저장점을 재개하며 완료 반복 수를 보존한다.
# resume-material은 동일 신원의 중간 물질 추정을 명시적으로 선택한다. J는 다시 푼다.
# material-guess는 binary만 바뀐 중간 물질을 원래 신원으로 읽어 새 J의 추정에 쓴다.
# 시간 진화·최종 CMFGEN 일치의 완료를 이 실행 시작이나 중간 반복으로 대체하지 않는다.
fd_limit=${7:-7200}
fd_aa_depth=${8:-1}
[[ $# -le 8 && "$fd_limit" =~ ^[1-9][0-9]{0,4}$ ]] || exit 2
case "$fd_aa_depth" in 1|2|4) ;; *) exit 2 ;; esac
(( fd_limit >= 60 && fd_limit <= 21600 )) || exit 2
# 읽기 전용 검사: 호스트/락/디렉터리/GPU에 접근하지 않는다.
if [[ "${CMF_FULL_DOMAIN_CHECK_LIMIT_ONLY:-0}" == 1 ]];then
 printf 'runtime_seconds=%s kill_grace_seconds=15 max_timeout_seconds=%s\n' "$fd_limit" "$((fd_limit+15))"
 exit 0
fi
[[ "$(hostname -s)" == LagEunha && -z "${SLURM_JOB_ID:-}" ]]
fd_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
fd_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
fd_out=${1:?new absolute output directory required}
fd_start=${2:-guess}
fd_refinements=${3:-42}
fd_previous=${4:-}
fd_backend=${5:-ada}
fd_build=${6:-$fd_base/common-build-425547}
case "$fd_backend" in ada|gpu) ;; *) exit 2 ;; esac
[[ "$fd_build" =~ ^$fd_base/common-build-[0-9]+$ ]]
test -f "$fd_build/binary.sha256"
case "$fd_refinements" in 1|42) ;; *) exit 2 ;; esac
case "$fd_start" in
 guess|material-guess|refinement-guess|binary-refinement-guess|accelerated-guess)
  if [[ -z "$fd_previous" ]];then
   [[ "$fd_start" == guess ]] || exit 2
   fd_resume="$fd_base/ada-evolution-20261016-1036/initial"
  else
   [[ "$fd_previous" =~ ^$fd_base/((first-field|baseline)-[0-9]+-[0-9]{8}\.[A-Za-z0-9]+|ada-evolution-[0-9]{8}-[0-9]{4}/initial)$ ]] || exit 2
   if [[ "$fd_start" == material-guess ]];then
    test -f "$fd_previous/work/initial_material_guess.bin"
   else
    test -f "$fd_previous/work/initial_iterate.bin"
   fi
   test -f "$fd_previous/iterate_identity.txt"
   fd_resume=$fd_previous
  fi
  ;;
 fresh) [[ -z "$fd_previous" ]] || exit 2; fd_resume= ;;
 resume|resume-material)
  [[ "$fd_previous" =~ ^$fd_base/ada-evolution-[0-9]{8}-[0-9]{4}/initial$ ]] || exit 2
  fd_resume=$fd_previous
  # 아래 공통 runner가 binary/input/effective env의 전체 정체성을 대조한다.
  # 검사 우회나 저장점의 신원 변경은 하지 않는다.
  ;;
 *) exit 2 ;;
esac
[[ "$fd_out" =~ ^$fd_base/ada-evolution-[0-9]{8}-[0-9]{4}$ ]]
exec 9>"$fd_base/.cmf-toy-ada.lock"
flock -n 9
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
ulimit -v 134217728
mkdir "$fd_out" "$fd_out/runners"
printf 'FULL_DOMAIN_ADA_ROOT=%s\n' "$fd_out"
fd_started=$SECONDS
trap 'fd_rc=$?; printf "RUN_FOOTER exit=%s elapsed_seconds=%s GPUs=1\n" "$fd_rc" "$((SECONDS-fd_started))" > "$fd_out/RUN_FOOTER.txt"' EXIT
hostname > "$fd_out/host.txt"
date -Is > "$fd_out/start.txt"
printf 'start=%s\nsource=%s\nLTE_is_initial_guess_only=1\n' "$fd_start" "$fd_resume" > "$fd_out/initialization.txt"
printf 'runtime_seconds=%s\nkill_grace_seconds=15\n' "$fd_limit" > "$fd_out/runtime_limit.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv > "$fd_out/gpu.txt"
cp "$fd_src/scripts/run_cmf_toy_initial.slurm" "$fd_src/scripts/run_cmf_full_domain_ada.sh" "$fd_out/runners/"
(cd "$fd_out/runners";sha256sum ./*) > "$fd_out/runners.sha256"
env CMF_TOY_CPUS=8 CMF_TOY_RUN_OUT="$fd_out/initial" \
 timeout --signal=TERM --kill-after=15s "${fd_limit}s" \
 bash "$fd_out/runners/run_cmf_toy_initial.slurm" \
 "$fd_build" "$fd_base/toy-material-423395" anderson "$fd_backend" \
 "$fd_base/nonthermal-423416" "$fd_resume" \
 anderson t10 mixture "$fd_start" "$fd_refinements" "$fd_aa_depth" > "$fd_out/initial-runner.log" 2> "$fd_out/initial-runner.err"
