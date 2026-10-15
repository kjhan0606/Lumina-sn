#!/bin/bash
#SBATCH --job-name=CMF_TOY_STEP
#SBATCH --partition=a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-step-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-step-%j.err
set -euo pipefail
umask 027
step_build=${1:?sealed build required}
step_previous=${2:?successful physical epoch run required}
step_epoch=${3:?new epoch in seconds required}
step_iteration=${4:-plain}
step_backend=${5:-gpu}
case "$step_backend" in
 gpu) step_variant=cluster;step_binary=$step_build/lumina_cuda;step_devices=1 ;;
 cpu) step_variant=cpu;step_binary=$step_build/lumina_cpu;step_devices=0 ;;
 *) exit 2 ;;
esac
case "$step_iteration" in plain|anderson) ;; *) exit 2 ;; esac
[[ "$step_epoch" =~ ^[0-9]+([.][0-9]+)?$ ]]
step_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
step_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-step-${SLURM_JOB_ID}
mkdir "$step_out"
mkdir "$step_out/input" "$step_out/work"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
grep -Fxq "variant=$step_variant managed=0" "$step_build/variant.txt"
(cd "$step_build/source"; sha256sum -c "$step_build/source.sha256") > "$step_out/build_source_verify.log"
sha256sum -c "$step_build/binary.sha256" > "$step_out/binary_verify.log"
(cd "$step_previous/input"; sha256sum -c "$step_previous/input.sha256") > "$step_out/previous_input_verify.log"
(cd "$step_previous/work"; sha256sum -c "$step_previous/output.sha256") > "$step_out/previous_output_verify.log"
if grep -Eq '^TOY_INITIAL_CONVERGED ' "$step_previous/initial.log" 2>/dev/null; then
 step_state=initial_state.bin
elif grep -Eq '^TOY_STEP_CONVERGED ' "$step_previous/step.log" 2>/dev/null; then
 step_state=step_state.bin
else
 echo '실제 수렴한 과거가 없다. predictor/합성 fixture는 생산 과거로 사용하지 않는다.' >&2
 exit 1
fi
grep -Eq "^[0-9a-f]{64}  $step_state$" "$step_previous/output.sha256"
while read -r step_hash step_file; do
 [[ "$step_file" != */* && "$step_file" != .* ]]
 cp "$step_previous/input/$step_file" "$step_out/input/$step_file"
done < "$step_previous/input.sha256"
cp "$step_previous/work/$step_state" "$step_out/previous_state.bin"
cp "$step_previous/input.sha256" "$step_build/commit.txt" "$step_build/binary.sha256" "$step_out/"
step_nt_arguments=()
step_nt_mode=ALL_THERMAL_CONTROL
if [[ -e "$step_out/input/nonthermal_resolved_bindings.csv" || -e "$step_out/input/arnaud_rothenflug_23jun15" ]]; then
 for step_nt_file in nonthermal_resolved_bindings.csv arnaud_rothenflug_23jun15; do
  grep -Eq "^[0-9a-f]{64}  $step_nt_file$" "$step_out/input.sha256"
 done
 step_nt_arguments=("$step_out/input/arnaud_rothenflug_23jun15" "$step_out/input/nonthermal_resolved_bindings.csv")
 step_nt_mode=RESOLVED_NONTHERMAL_SUBSET
fi
step_previous_log=$step_previous/step.log
if [[ "$step_state" == initial_state.bin ]]; then step_previous_log=$step_previous/initial.log; fi
grep -Eq "^TOY_(INITIAL|STEP)_CONVERGED .*nonthermal=$step_nt_mode " "$step_previous_log"
cp "$step_src/scripts/run_cmf_toy_step.slurm" "$step_out/runner.slurm"
sha256sum "$step_out/runner.slurm" "$step_out/previous_state.bin" > "$step_out/runner_history.sha256"
(cd "$step_out/input"; sha256sum -c "$step_out/input.sha256") > "$step_out/input_verify.log"
while read -r step_key; do unset "$step_key"; done < <(compgen -v LUMINA_ || true)
export LUMINA_PURE_CMFGEN=1 LUMINA_CMF_COMMON_FIELD=1 LUMINA_CMF_BF_MILNE=2
export LUMINA_SUPER_LEVELS=1 LUMINA_NLTE_COLL_ION=1 LUMINA_OMEGA_CMFGEN=1
export LUMINA_NLTE_NE_CLOSURE=1 LUMINA_CMF_FINE_VDOP=5000000 LUMINA_CMF_FINE_PPD=3
export LUMINA_CMF_FINE_ALI=768 LUMINA_CMF_FINE_TOL=1e-8 LUMINA_CMF_FINE_PHOTOION=1
export LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS=42 LUMINA_CMF_FINE_MGPU_DEVICES=$step_devices
if [[ "$step_iteration" == anderson ]]; then export LUMINA_CMF_ANDERSON1=1; fi
export LUMINA_BF_CHANNELS_FILE=$step_out/input/bf_channels.bin
export LUMINA_TOPION_LEVELS_FILE=$step_out/input/bf_channels.terminal.csv
export LUMINA_IONIZATION_REFERENCE_FILE=$step_out/input/ionization_reference.csv
env | LC_ALL=C sort | grep '^LUMINA_' > "$step_out/env_lumina_effective.txt"
printf 'mode=%s\nbackend=%s\nsource=%s\nbuild=%s\ninput=%s\nprevious=%s\nepoch_s=%s\n' \
 "$step_nt_mode" "$step_backend" "$step_build/source" "$step_build" "$step_out/input" "$step_previous" "$step_epoch" > "$step_out/identity.txt"
cd "$step_out/work"
(ulimit -f 2097152; exec "$step_binary" --cmf-toy-step \
 "$step_out/input/snia_toy06_2d.dat" "$step_out/input" \
 "$step_out/input/NUC_DECAY_DATA_ni56_co56_fe56" "$step_out/previous_state.bin" \
 "$step_epoch" 100 200000 128 "${step_nt_arguments[@]}") > "$step_out/step.log" 2> "$step_out/step.err"
sha256sum step_state.bin step_material.csv step_spectrum.csv step_spectrum.json > "$step_out/output.sha256"
sha256sum -c "$step_out/runner_history.sha256" > "$step_out/history_preserved.log"
echo "STEP_COMPLETE: 자유 T 시간해 $step_nt_mode. 최종 CMFGEN 동등성 판정 아님."
