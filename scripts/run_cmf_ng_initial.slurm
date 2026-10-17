#!/bin/bash
#SBATCH --job-name=CMF_NG_INITIAL
#SBATCH --partition=a100,a100_pcie,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-initial-ng-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-initial-ng-%j.err
set -euo pipefail
umask 027
ng_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
ng_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
ng_guess=${1:?validated numerical guess directory required}
ng_parent=$ng_base/toy-initial-424528
ng_build=$ng_base/common-build-424525
ng_out=$ng_base/toy-initial-ng-${SLURM_JOB_ID:?}
[[ "$ng_guess" == "$ng_base"/ng-guess-* ]]
cd "$ng_src"
[[ "$(git rev-parse --show-toplevel)" == "$ng_src" ]]
mkdir "$ng_out" "$ng_out/input" "$ng_out/work"
ng_start=$SECONDS
trap 'ng_rc=$?; printf "RUN_FOOTER exit=%s elapsed_seconds=%s GPUs=1\n" "$ng_rc" "$((SECONDS-ng_start))"' EXIT
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2 python/3.11.14
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
grep -Fxq 'variant=cluster managed=0' "$ng_build/variant.txt"
(cd "$ng_build/source";sha256sum -c "$ng_build/source.sha256") > "$ng_out/build-source-verify.log"
sha256sum -c "$ng_build/binary.sha256" > "$ng_out/binary-verify.log"
(cd "$ng_guess";sha256sum -c source.sha256;sha256sum -c output.sha256) > "$ng_out/guess-verify.log"
grep '^ITERATE_REPLAY_READ_DONE ' "$ng_guess/native/test.log" > "$ng_out/native-proof.log"
(cd "$ng_parent/input";sha256sum -c "$ng_parent/input.sha256") > "$ng_out/parent-input-verify.log"
while read -r ng_hash ng_file;do
 [[ "$ng_file" != */* && "$ng_file" != .* ]]
 cp "$ng_parent/input/$ng_file" "$ng_out/input/$ng_file"
done < "$ng_parent/input.sha256"
cp "$ng_parent/input.sha256" "$ng_parent/iterate_identity.txt" "$ng_out/"
cp "$ng_build/binary.sha256" "$ng_build/commit.txt" "$ng_out/"
cp "$ng_guess/guess_identity.txt" "$ng_out/guess_source_identity.txt"
cp "$ng_guess/guess_input.bin" "$ng_out/restart_input.bin"
scontrol write batch_script "$SLURM_JOB_ID" "$ng_out/runner.slurm"
sha256sum "$ng_out/runner.slurm" > "$ng_out/runner.sha256"
while read -r ng_key;do unset "$ng_key";done < <(compgen -v LUMINA_ || true)
while IFS= read -r ng_entry;do export "$ng_entry";done < "$ng_parent/env_lumina_effective.txt"
export LUMINA_BF_CHANNELS_FILE=$ng_out/input/bf_channels.bin
export LUMINA_TOPION_LEVELS_FILE=$ng_out/input/bf_channels.terminal.csv
export LUMINA_IONIZATION_REFERENCE_FILE=$ng_out/input/ionization_reference.csv
env | LC_ALL=C sort | grep '^LUMINA_' > "$ng_out/env_lumina_effective.txt"
{
 cut -d' ' -f1 "$ng_build/binary.sha256"
 LC_ALL=C sort "$ng_out/input.sha256"
 printf 'iteration=anderson\nmode=RESOLVED_NONTHERMAL_SUBSET\nshells=100\npackets=200000\nouter_cap=128\nouter_tol=1e-5\n'
 sed -E '/^LUMINA_(BF_CHANNELS_FILE|TOPION_LEVELS_FILE|IONIZATION_REFERENCE_FILE)=/d' "$ng_out/env_lumina_effective.txt"
} > "$ng_out/reconstructed_identity.txt"
cmp "$ng_out/iterate_identity.txt" "$ng_out/reconstructed_identity.txt"
ng_run_id=$(sha256sum "$ng_out/iterate_identity.txt");ng_run_id=${ng_run_id%% *}
ng_guess_id=$(sha256sum "$ng_out/guess_source_identity.txt");ng_guess_id=${ng_guess_id%% *}
[[ "$ng_run_id" != "$ng_guess_id" ]]
python3 - "$ng_out/guess_source_identity.txt" "$ng_run_id" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    identity = json.load(stream)
assert identity['parent_identity'] == sys.argv[2]
assert identity['purpose'] == 'DAMPED_NG_NUMERICAL_GUESS_NOT_SOLUTION'
assert identity['physical_history'] is False and identity['converged'] is False
assert identity['parent_iteration'] == 4 and identity['alpha'] == .25
PY
export LUMINA_INITIAL_ITERATE_ID=$ng_run_id LUMINA_INITIAL_GUESS_ID=$ng_guess_id
export LUMINA_INITIAL_ITERATE_INPUT=$ng_out/restart_input.bin
sha256sum "$ng_guess/guess_input.bin" "$ng_out/restart_input.bin" > "$ng_out/restart_input.sha256"
cmp "$ng_guess/guess_input.bin" "$ng_out/restart_input.bin"
(cd "$ng_out/input";sha256sum -c "$ng_out/input.sha256") > "$ng_out/input-verify.log"
printf 'start=damped_ng_numerical_guess\nsource=%s\nsource_identity=%s\nrun_identity=%s\nphysical_history=0\n' \
 "$ng_guess" "$ng_guess_id" "$ng_run_id" > "$ng_out/start_provenance.txt"
printf 'mode=RESOLVED_NONTHERMAL_SUBSET\nbackend=gpu\nbuild=%s\ninput=%s\n' \
 "$ng_build" "$ng_out/input" > "$ng_out/identity.txt"
cd "$ng_out/work"
(ulimit -f 2097152; exec "$ng_build/lumina_cuda" --cmf-toy-initial \
 "$ng_out/input/snia_toy06_2d.dat" "$ng_out/input" \
 "$ng_out/input/NUC_DECAY_DATA_ni56_co56_fe56" 100 200000 128 \
 "$ng_out/input/arnaud_rothenflug_23jun15" "$ng_out/input/nonthermal_resolved_bindings.csv") \
 > "$ng_out/initial.log" 2> "$ng_out/initial.err"
sha256sum initial_state.bin initial_converged_material.csv initial_spectrum.csv initial_spectrum.json \
 > "$ng_out/output.sha256"
echo 'INITIAL_COMPLETE: numerical guess used; final 19.48-day goal NOT_JUDGED'
