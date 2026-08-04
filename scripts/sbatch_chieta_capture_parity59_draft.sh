#!/usr/bin/env bash
# DRAFT ONLY -- DO NOT SUBMIT FROM THIS FILE WITHOUT DRIVER REVIEW.
# Single job-per-run parity59 chi/eta capture on an H200, falling back to an
# 80-GB H100. All mutable output stays in a job-private GPFS directory.
#SBATCH --job-name=chieta_p59_capture
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.out
#SBATCH --error=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.err

set -euo pipefail

DEPLOY_ROOT=/gpfs/kjhan/lumina_runner2
CERT_ENV=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_CHIETA_CAPTURE.env
EXPECTED_RUN_SHA=f42d52bf904a575c43cbad12c00260e06bf2f45fbd9727232cac7827f1ee6c9d
EXPECTED_BENCH_SHA=bad088f5cb866adc94aae303a8adc5dd8c9c828dc7edf1529797b392607ae2fa
RUN_DIR="$DEPLOY_ROOT/scratch/chieta_capture_parity59_${SLURM_JOB_ID:?}"
export RUN_DIR

module load cuda/13.0.2 2>/dev/null || true
mkdir -p "$DEPLOY_ROOT/slurm" "$RUN_DIR"

# Prevent inherited experimental gates from widening the RESOLVED CONFIG.
while IFS='=' read -r key _; do
    case "$key" in
        LUMINA_*|SUPER_*|OMP_NUM*) unset "$key" ;;
    esac
done < <(env)

# shellcheck source=/dev/null
source "$CERT_ENV"

RUN_BIN="$DEPLOY_ROOT/$LUMINA_BIN"
ORACLE_BENCH="$DEPLOY_ROOT/bench_frozen_oracle"
MODEL="$DEPLOY_ROOT/$LUMINA_MODEL_DIR"
test -x "$RUN_BIN"
test -x "$ORACLE_BENCH"
test -d "$MODEL"
test "$(sha256sum "$RUN_BIN" | awk '{print $1}')" = "$EXPECTED_RUN_SHA"
test "$(sha256sum "$ORACLE_BENCH" | awk '{print $1}')" = "$EXPECTED_BENCH_SHA"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
case "$gpu_name" in
    *H200*|*H100*) ;;
    *) echo "FATAL: unsupported GPU: $gpu_name" >&2; exit 2 ;;
esac
test "$gpu_mem" -ge 80000

# Preserve parity59 argv exactly while isolating all root-level products.
ln -s "$RUN_BIN" "$RUN_DIR/$LUMINA_BIN"
ln -s "$DEPLOY_ROOT/data" "$RUN_DIR/data"
cp "$CERT_ENV" "$RUN_DIR/"
cd "$RUN_DIR"

echo "CAPTURE START host=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION"
echo "GPU=$gpu_name memory_MiB=$gpu_mem"
echo "RUN_DIR=$RUN_DIR"
echo "RUN_BINARY_SHA256=$EXPECTED_RUN_SHA"

set +e
./"$LUMINA_BIN" \
    data/tardis_reference_toy06_19p48d_sivcaiv \
    100000 12 spectrum nlte > stdout.log 2> stderr.log
run_rc=$?
set -e
test "$run_rc" -eq 0
grep -q '^=== END RUN FOOTER (122 vars) ===$' stdout.log
grep -qxF "  LUMINA_CMF_FROZEN_CHIETA_DUMP=$LUMINA_CMF_FROZEN_CHIETA_DUMP" stdout.log
grep -qxF '  LUMINA_CMF_FROZEN_CHIETA_ITER=10' stdout.log
grep -qxF '  LUMINA_EVENT_LOG=1' stdout.log

python3 "$DEPLOY_ROOT/scripts/cmf_chieta_check.py" \
    --expected-iteration "$CHIETA_EXPECTED_ITER" \
    "$LUMINA_CMF_FROZEN_CHIETA_DUMP" | tee chieta_check.log

# Turn the just-captured production state into the requested frozen-cell oracle
# CSVs. Each replay is CPU-only and fail-closed; s20 is mandatory.
IFS=, read -r -a oracle_cells <<< "$CHIETA_ORACLE_CELLS"
for shell in "${oracle_cells[@]}"; do
    out="$RUN_DIR/oracle_s${shell}"
    env -i PATH="$PATH" \
        LUMINA_FROZEN_ORACLE_ONLY_SHELL="$shell" \
        "$ORACLE_BENCH" "$RUN_DIR" "$MODEL" "$out" \
        > "$RUN_DIR/oracle_s${shell}.stdout.log" \
        2> "$RUN_DIR/oracle_s${shell}.stderr.log"
    test -s "$out/lumina_oracle_cell_s${shell}.csv"
done

sha256sum \
    "$LUMINA_CMF_FROZEN_CHIETA_DUMP" \
    oracle_s*/lumina_oracle_cell_s*.csv \
    > capture_sha256.txt

echo "=== CAPTURE JOB FOOTER ==="
echo "run_rc=$run_rc"
echo "run_footer=present"
echo "chieta_contract=PASS expected_iter=$CHIETA_EXPECTED_ITER"
echo "oracle_cells=$CHIETA_ORACLE_CELLS"
echo "result_dir=$RUN_DIR"
echo "submission=performed_by_driver_only"
echo "=== END CAPTURE JOB FOOTER ==="
