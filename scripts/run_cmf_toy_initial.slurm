#!/bin/bash
#SBATCH --job-name=CMF_TOY_INITIAL
#SBATCH --partition=a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-initial-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-initial-%j.err
set -euo pipefail
umask 027
initial_build=${1:?cluster build directory required}
initial_input=${2:?verified initial-prepare input directory required}
initial_iteration=${3:-plain}
initial_backend=${4:-gpu}
initial_nt=${5:-}
initial_resume=${6:-}
case "$initial_backend" in
 gpu) initial_variant=cluster;initial_binary=$initial_build/lumina_cuda;initial_devices=1 ;;
 cpu) initial_variant=cpu;initial_binary=$initial_build/lumina_cpu;initial_devices=0 ;;
 *) exit 2 ;;
esac
case "$initial_iteration" in plain|anderson) ;; *) exit 2 ;; esac
initial_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
initial_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-initial-${SLURM_JOB_ID}
mkdir "$initial_out"
mkdir "$initial_out/input" "$initial_out/work"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
grep -Fxq "variant=$initial_variant managed=0" "$initial_build/variant.txt"
(cd "$initial_build/source"; sha256sum -c "$initial_build/source.sha256") > "$initial_out/build_source_verify.log"
sha256sum -c "$initial_build/binary.sha256" > "$initial_out/binary_verify.log"
# 실행되는 것은 위에서 검증한 봉인 binary/source다. 이후 작업 트리 편집이
# 이전 실행의 재현을 막지 않도록 현재 작업 트리와의 동일성은 요구하지 않는다.
(cd "$initial_input/input"; sha256sum -c "$initial_input/input.sha256") > "$initial_out/original_input_verify.log"
# 복사 대상은 이전 sealed 입력 manifest의 단일 디렉터리 파일만이다.
while read -r initial_hash initial_file; do
 [[ "$initial_file" != */* && "$initial_file" != .* ]]
 cp "$initial_input/input/$initial_file" "$initial_out/input/$initial_file"
done < "$initial_input/input.sha256"
cp "$initial_input/input.sha256" "$initial_build/commit.txt" "$initial_build/binary.sha256" "$initial_out/"
initial_nt_arguments=()
initial_nt_mode=ALL_THERMAL_CONTROL
if [[ -n "$initial_nt" ]]; then
 # 연결표의 global ID가 이번 원자 모형과 같은지 확인한다.
 (cd "$initial_nt/input"; sha256sum -c "$initial_nt/input.sha256") > "$initial_out/nonthermal_input_verify.log"
 for initial_table in levels.csv ionization_energies.csv bf_channels.terminal.csv; do
  cmp "$initial_nt/input/$initial_table" "$initial_out/input/$initial_table"
 done
 test ! -e "$initial_out/input/arnaud_rothenflug_23jun15"
 test ! -e "$initial_out/input/nonthermal_resolved_bindings.csv"
 cp "$initial_nt/input/arnaud_rothenflug_23jun15" "$initial_nt/nonthermal_resolved_bindings.csv" "$initial_out/input/"
 (cd "$initial_out/input"; sha256sum arnaud_rothenflug_23jun15 nonthermal_resolved_bindings.csv) >> "$initial_out/input.sha256"
 initial_nt_arguments=("$initial_out/input/arnaud_rothenflug_23jun15" "$initial_out/input/nonthermal_resolved_bindings.csv")
 initial_nt_mode=RESOLVED_NONTHERMAL_SUBSET
fi
# 대기 중 저장소가 바뀌어도 실제 제출된 Slurm 스크립트를 보존한다.
cp "$0" "$initial_out/runner.slurm"
sha256sum "$initial_out/runner.slurm" > "$initial_out/runner.sha256"
(cd "$initial_out/input"; sha256sum -c "$initial_out/input.sha256") > "$initial_out/input_verify.log"
while read -r initial_key; do unset "$initial_key"; done < <(compgen -v LUMINA_ || true)
export LUMINA_PURE_CMFGEN=1 LUMINA_CMF_COMMON_FIELD=1 LUMINA_CMF_BF_MILNE=2
export LUMINA_SUPER_LEVELS=1 LUMINA_NLTE_COLL_ION=1 LUMINA_OMEGA_CMFGEN=1
export LUMINA_NLTE_NE_CLOSURE=1 LUMINA_CMF_FINE_VDOP=5000000 LUMINA_CMF_FINE_PPD=3
export LUMINA_CMF_FINE_ALI=768 LUMINA_CMF_FINE_TOL=1e-8 LUMINA_CMF_FINE_PHOTOION=1
export LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS=42 LUMINA_CMF_FINE_MGPU_DEVICES=$initial_devices
if [[ "$initial_iteration" == anderson ]]; then export LUMINA_CMF_ANDERSON1=1; fi
export LUMINA_BF_CHANNELS_FILE=$initial_out/input/bf_channels.bin
export LUMINA_TOPION_LEVELS_FILE=$initial_out/input/bf_channels.terminal.csv
export LUMINA_IONIZATION_REFERENCE_FILE=$initial_out/input/ionization_reference.csv
env | LC_ALL=C sort | grep '^LUMINA_' > "$initial_out/env_lumina_effective.txt"
# 경로 대신 봉인 내용과 수치 설정을 식별한다. CPU/GPU binary 변경도 거부한다.
{
 cut -d' ' -f1 "$initial_build/binary.sha256"
 LC_ALL=C sort "$initial_out/input.sha256"
 printf 'iteration=%s\nmode=%s\nshells=100\npackets=200000\nouter_cap=128\nouter_tol=1e-5\n' "$initial_iteration" "$initial_nt_mode"
 sed -E '/^LUMINA_(BF_CHANNELS_FILE|TOPION_LEVELS_FILE|IONIZATION_REFERENCE_FILE)=/d' "$initial_out/env_lumina_effective.txt"
} > "$initial_out/iterate_identity.txt"
initial_id=$(sha256sum "$initial_out/iterate_identity.txt");initial_id=${initial_id%% *}
export LUMINA_INITIAL_ITERATE_ID=$initial_id
if [[ -n "$initial_resume" ]]; then
 test -f "$initial_resume/work/initial_iterate.bin"
 cmp "$initial_resume/iterate_identity.txt" "$initial_out/iterate_identity.txt"
 cp "$initial_resume/work/initial_iterate.bin" "$initial_out/restart_input.bin"
 sha256sum "$initial_resume/work/initial_iterate.bin" "$initial_out/restart_input.bin" > "$initial_out/restart_input.sha256"
 cmp "$initial_resume/work/initial_iterate.bin" "$initial_out/restart_input.bin"
 export LUMINA_INITIAL_ITERATE_INPUT=$initial_out/restart_input.bin
fi
printf 'mode=%s\nbackend=%s\nsource=%s\nbuild=%s\ninput=%s\n' \
 "$initial_nt_mode" "$initial_backend" "$initial_build/source" "$initial_build" "$initial_out/input" > "$initial_out/identity.txt"
cd "$initial_out/work"
# 2GiB/file 상한. 최신 수치 반복 하나와 수렴 성공시의 물리 이력을 구분한다.
(ulimit -f 2097152; exec "$initial_binary" --cmf-toy-initial \
 "$initial_out/input/snia_toy06_2d.dat" "$initial_out/input" \
 "$initial_out/input/NUC_DECAY_DATA_ni56_co56_fe56" 100 200000 128 "${initial_nt_arguments[@]}") \
 > "$initial_out/initial.log" 2> "$initial_out/initial.err"
sha256sum initial_state.bin initial_converged_material.csv initial_spectrum.csv initial_spectrum.json \
 > "$initial_out/output.sha256"
echo "INITIAL_COMPLETE: 2-day $initial_nt_mode; final spectrum goal not achieved"
