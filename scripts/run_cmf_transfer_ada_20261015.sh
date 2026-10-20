#!/bin/bash
# 사용자 전권 위임 후 수송/원자자료 수정 확인 런 RT1. 원본 R2와 분리한다.
# 자동 재시작 없음. 첫 반복 6h, 서비스 전체 22h(기존 1.18h 포함 24h 이내).
set -euo pipefail
umask 027
transfer_root=/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/transfer_20261015/run1
transfer_build=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/transfer-build-422721
transfer_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-angular-20261014
transfer_original=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1
transfer_model=$transfer_original/input/model
transfer_sigma=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/atomic-stage-422722/cmfgen_sigma_bf.bin
transfer_bin=$transfer_build/lumina_cuda
transfer_deadline=21600
fail() { printf 'TRANSFER_RUN_FAIL reason=%s\n' "$*" >&2; exit 70; }
[[ $(hostname) == LagEunha && -n ${INVOCATION_ID:-} ]] || fail HOST_OR_SUPERVISION
exec >> "$transfer_root/runner.log" 2>&1
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
[[ -d $transfer_root/work && -z $(ls -A "$transfer_root/work") ]] || fail WORK_NOT_EMPTY
[[ $(git -C "$transfer_src" rev-parse HEAD) == 80b54acd048f14b314aacf032ba5c6dd3bbaa3f9 ]] || fail SOURCE_COMMIT
sha256sum -c "$transfer_build/binary.sha256" > "$transfer_root/binary_verify.log"
( cd "$transfer_src"; sha256sum -c "$transfer_build/source.sha256" ) > "$transfer_root/source_verify.log"
( cd "$transfer_model"; sha256sum -c "$transfer_original/input/deck.sha256" ) > "$transfer_root/original_deck_verify.log"
[[ $(sha256sum "$transfer_sigma") == "ee7d689e14f5426f6d1dc28fa31a0d5727b9eac41d0157bdcb2b8c0d6e9a0635  $transfer_sigma" ]] || fail ATOMIC_REPAIR_IDENTITY
transfer_base_env=/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/ada_20261014/run1/exports.env
[[ $(sha256sum "$transfer_base_env") == "923c3ad35f0c5df373f74ac0189bf8010e4485b419df189c902d821698d9c71a  $transfer_base_env" ]] || fail BASE_ENV_IDENTITY
while IFS='=' read -r name _; do
    if [[ $name == LUMINA_* ]]; then unset "$name"; fi
done < <(env)
source "$transfer_root/exports.env"
[[ -z ${LUMINA_FIXED_TE_PROFILE+x} && ${LUMINA_CMF_DEP_SOURCE:-} == 0 && ${LUMINA_PURE_CMFGEN_ITER:-} == 6 && ${LUMINA_NLTE_NE_CLOSURE:-} == 1 && ${LUMINA_CMF_FINE_MGPU_DEVICES:-} == 1 ]] || fail CONFIG_CONTRACT
[[ $LUMINA_CMFGEN_SIGMA_BF == "$transfer_sigma" && $LUMINA_PHYSICS_COMPARISON_DIR == "$transfer_root/work/physics_comparison" ]] || fail INPUT_OUTPUT_CONTRACT
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$transfer_root/gpu_identity.csv"
env | LC_ALL=C sort | grep '^LUMINA_' > "$transfer_root/env_lumina_effective.txt"
( set -o noclobber; printf 'host=%s invocation=%s\n' "$(hostname)" "$INVOCATION_ID" > "$transfer_root/RUN_ATTEMPTED" ) || fail ALREADY_ATTEMPTED
transfer_start=$(date +%s)
transfer_pid=
transfer_watchdog=0
finish() {
    local rc=$?
    trap - EXIT TERM INT
    if [[ -n $transfer_pid ]] && kill -0 "$transfer_pid" 2>/dev/null; then
        kill -TERM "$transfer_pid" 2>/dev/null || :
        for ((i=0; i<12; i++)); do
            kill -0 "$transfer_pid" 2>/dev/null || break
            sleep 5
        done
        kill -KILL "$transfer_pid" 2>/dev/null || :
        wait "$transfer_pid" 2>/dev/null || :
    fi
    {
        printf 'schema=TRANSFER_RT1_V1\nhost=%s\ninvocation=%s\n' "$(hostname)" "$INVOCATION_ID"
        printf 'model_rc=%s\nelapsed_seconds=%s\nwatchdog_stop=%s\nbinary=%s\nsigma=%s\n' "$rc" "$(( $(date +%s)-transfer_start ))" "$transfer_watchdog" "$transfer_bin" "$transfer_sigma"
    } > "$transfer_root/RUN_FOOTER.txt"
    grep -E '\[FATAL\]|\[BLOCKED\]' "$transfer_root/stdout.log" "$transfer_root/stderr.log" > "$transfer_root/fatal_scan.log" || :
    printf 'TRANSFER_RUN_DONE rc=%s watchdog=%s\n' "$rc" "$transfer_watchdog"
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
( ulimit -f 1048576; cd "$transfer_root/work"; exec "$transfer_bin" "$transfer_model" 1 2 spectrum nlte ) > "$transfer_root/stdout.log" 2> "$transfer_root/stderr.log" &
transfer_pid=$!
printf 'model_pid=%s start_unix=%s start_iso=%s\n' "$transfer_pid" "$transfer_start" "$(date -Is)" > "$transfer_root/MODEL_PID.txt"
printf 'TRANSFER_RUN_STARTED model_pid=%s start=%s\n' "$transfer_pid" "$(date -Is)"
transfer_first=$transfer_root/work/physics_comparison/physics_DET_iter0000.manifest.json
check_deadline() {
    if [[ -e $transfer_first ]]; then
        (( $(stat -c %Y "$transfer_first") <= transfer_start+transfer_deadline ))
    else
        (( $(date +%s) < transfer_start+transfer_deadline ))
    fi
}
while kill -0 "$transfer_pid" 2>/dev/null; do
    if ! check_deadline; then
        transfer_watchdog=1
        printf 'FIRST_ITERATION_DEADLINE_FAILED at=%s\n' "$(date -Is)" > "$transfer_root/WATCHDOG_STOP.txt"
        exit 124
    fi
    sleep 60
done
set +e
wait "$transfer_pid"
transfer_rc=$?
set -e
if ! check_deadline; then transfer_watchdog=1; transfer_rc=124; fi
if ((transfer_rc==0)); then
    count=$(grep -c '\[DET-TRANSACTIONAL\] status=COMPLETE' "$transfer_root/stdout.log" || :)
    commits=$(grep -c 'event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=' "$transfer_root/stderr.log" || :)
    [[ $count == 1 && $commits == 6 && -f $transfer_root/work/physics_comparison/physics_DET_iter0005.manifest.json ]] || transfer_rc=71
fi
exit "$transfer_rc"
