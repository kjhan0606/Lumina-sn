#!/usr/bin/env bash
#SBATCH --job-name=lumina-independent-final-checks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --export=ALL

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 PAIR_ROOT RESPONSE_ROOT SOURCES_ROOT" >&2
    exit 2
fi

pair_root=$1
response_root=$2
sources_root=$3
analysis_root="$pair_root/run/analysis"
binding_report="$analysis_root/current_j_h025_same_state_binding_20261107.json"
replay_report="$analysis_root/current_j_h025_line_grid_replay_90depth_20261107.json"
native_root="$pair_root/run/h025/native"
linearization="$response_root/thermal-direction-2/linearization.npz"
run_result="$response_root/result.json"
line_output="$analysis_root/current_j_h025_line_quadrature_grammar_repro_20261107.json"
schur_output="$response_root/thermal-direction-2/independent-schur-direction-grammar-repro-20261107.json"

for required in "$binding_report" "$replay_report" "$native_root/EDDFACTOR" \
                "$linearization" "$run_result" "$sources_root/sources.json"; do
    [[ -f "$required" ]] || { echo "missing input: $required" >&2; exit 3; }
done
[[ ! -e "$line_output" && ! -e "$schur_output" ]] || {
    echo "refusing to overwrite prior analysis output" >&2
    exit 4
}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun --ntasks=1 --cpus-per-task=1 python3 \
    scripts/diagnose_cmfgen_line_quadrature_operators.py \
    --sources "$sources_root" \
    --response-root "$response_root" \
    --native-root "$native_root" \
    --binding-report "$binding_report" \
    --replay-report "$replay_report" \
    --output "$line_output"

srun --ntasks=1 --cpus-per-task=1 python3 \
    scripts/diagnose_lumina_thermal_direction_conditioning.py \
    --linearization "$linearization" \
    --compare-linearization "$response_root/thermal-direction-1/linearization.npz" \
    --run-result "$run_result" \
    --output "$schur_output"

printf 'independent-checks=complete\n'
