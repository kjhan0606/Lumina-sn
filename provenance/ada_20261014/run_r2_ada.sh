#!/usr/bin/env bash
# One authorized Ada attempt, fixed physics/deck; no automatic restart.
set -euo pipefail
umask 027
A=/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/ada_20261014
R=$A/run1
B=$A/build1
O=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1
MODEL=$O/input/model
SRC=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-ada-20261014
BIN=$B/lumina_cuda
FIRST_ITER_DEADLINE_S=21600
fail() { printf 'ADA_RUN_FAIL reason=%s\n' "$*" >&2; exit 70; }
[[ $(hostname) == LagEunha ]] || fail WRONG_HOST
[[ -n ${INVOCATION_ID:-} ]] || fail REQUIRE_SYSTEMD_SUPERVISION
export MODULEPATH=/opt/ohpc/pub/modulefiles
set +u
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
set -u
exec 9>/gpfs/kjhan/lumina/.lageunha-gpu0.lock
flock -n 9 || fail GPU_LOCK_BUSY
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || fail GPU_ALREADY_BUSY
[[ $(nvidia-smi --query-gpu=uuid --format=csv,noheader) == GPU-42b3367f-1df7-2b2d-7961-7781c87b022d ]] || fail GPU_IDENTITY
[[ -d $R/work && -z $(ls -A "$R/work") ]] || fail WORK_NOT_EMPTY
[[ $(cat "$B/build.rc") == 0 ]] || fail BUILD_FAILED
( cd "$B"; sha256sum -c binary.sha256 ) > "$R/binary_verify.log" 2>&1
( cd "$SRC"; sha256sum -c "$B/source.sha256" ) > "$R/source_verify.log" 2>&1
( cd "$MODEL"; sha256sum -c "$O/input/deck.sha256" ) > "$R/deck_verify.log" 2>&1
while IFS='=' read -r name _; do
  if [[ $name == LUMINA_* ]]; then unset "$name"; fi
done < <(env)
source "$R/exports.env"
[[ -z ${LUMINA_FIXED_TE_PROFILE+x} && ${LUMINA_CMF_DEP_SOURCE:-} == 0 && ${LUMINA_PURE_CMFGEN_ITER:-} == 6 && ${LUMINA_NLTE_NE_CLOSURE:-} == 1 && ${LUMINA_CMF_FINE_MGPU_DEVICES:-} == 1 ]] || fail CONFIG_CONTRACT
[[ ${LUMINA_PHYSICS_COMPARISON_DIR:-} == "$R/work/physics_comparison" ]] || fail OUTPUT_CONTRACT
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$R/gpu_identity.csv"
env | LC_ALL=C sort | grep '^LUMINA_' > "$R/env_lumina_effective.txt"
( set -o noclobber; printf 'host=%s invocation=%s\n' "$(hostname)" "$INVOCATION_ID" > "$R/RUN_ATTEMPTED" ) || fail ALREADY_ATTEMPTED
start=$(date +%s)
pid=
watchdog=0
finish() {
  local rc=$?
  trap - EXIT TERM INT
  if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || :
    for ((i=0; i<12; i++)); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 5
    done
    kill -KILL "$pid" 2>/dev/null || :
    wait "$pid" 2>/dev/null || :
  fi
  {
    printf 'schema=ADA_R2_RUN_V1\nhost=%s\ninvocation=%s\n' "$(hostname)" "$INVOCATION_ID"
    printf 'model_rc=%s\nelapsed_seconds=%s\nwatchdog_stop=%s\nbinary=%s\nmodel_dir=%s\n' "$rc" "$(( $(date +%s)-start ))" "$watchdog" "$BIN" "$MODEL"
    printf 'det_complete=%s\n' "$(grep -c '\[DET-TRANSACTIONAL\] status=COMPLETE' "$R/stdout.log" || :)"
  } > "$R/RUN_FOOTER.txt"
  grep -E '\[FATAL\]|\[BLOCKED\]' "$R/stdout.log" "$R/stderr.log" > "$R/fatal_scan.log" || :
  printf 'ADA_RUN_DONE rc=%s watchdog=%s\n' "$rc" "$watchdog"
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
( ulimit -f 1048576; cd "$R/work"; exec "$BIN" "$MODEL" 1 2 spectrum nlte ) > "$R/stdout.log" 2> "$R/stderr.log" &
pid=$!
printf 'model_pid=%s start_unix=%s start_iso=%s\n' "$pid" "$start" "$(date -Is)" > "$R/MODEL_PID.txt"
printf 'ADA_RUN_STARTED model_pid=%s start=%s\n' "$pid" "$(date -Is)"
first=$R/work/physics_comparison/physics_DET_iter0000.manifest.json
check_deadline() {
  local now=$1
  if [[ -e $first ]]; then
    (( $(stat -c %Y "$first") <= start+FIRST_ITER_DEADLINE_S ))
  else
    (( now < start+FIRST_ITER_DEADLINE_S ))
  fi
}
while kill -0 "$pid" 2>/dev/null; do
  if ! check_deadline "$(date +%s)"; then
    watchdog=1
    printf 'FIRST_ITERATION_DEADLINE_FAILED deadline_s=%s at=%s\n' "$FIRST_ITER_DEADLINE_S" "$(date -Is)" > "$R/WATCHDOG_STOP.txt"
    exit 124
  fi
  sleep 60
done
set +e
wait "$pid"
rc=$?
set -e
# Catch completion between the final two watcher checks.
if ! check_deadline "$(date +%s)"; then
  watchdog=1
  printf 'FIRST_ITERATION_DEADLINE_FAILED at_final_check=%s\n' "$(date -Is)" > "$R/WATCHDOG_STOP.txt"
  rc=124
fi
# A zero process exit without all six committed transactions is not success.
if ((rc==0)); then
  count=$(grep -c '\[DET-TRANSACTIONAL\] status=COMPLETE' "$R/stdout.log" || :)
  commits=$(grep -c 'event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=' "$R/stderr.log" || :)
  [[ $count == 1 && $commits == 6 && -f $R/work/physics_comparison/physics_DET_iter0005.manifest.json ]] || rc=71
fi
exit "$rc"
