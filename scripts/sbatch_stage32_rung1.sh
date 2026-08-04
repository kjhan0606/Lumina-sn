#!/usr/bin/env bash
# DRAFT -- submitted by the driver only, after the C review clears the patch.
# Stage 3.2 Rung 1: read-only Sobolev local-response side-band capture at iter 10.
#SBATCH --job-name=s32_rung1
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=06:00:00
# No --exclude here on purpose: grammar072/078/080 are nodes of the grammar CPU
# cluster.  This GPU cluster is syn* (h200=syn104, h100=syn[08-09]), where those
# names do not resolve and slurm rejects the submission outright.
#SBATCH --output=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.out
#SBATCH --error=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.err

set -euo pipefail

DEPLOY_ROOT=/gpfs/kjhan/lumina_runner2
CERT_ENV=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_INSTR.env
: "${STAGE32_BIN_SHA:?STAGE32_BIN_SHA must be exported by the driver at submit time}"
RUN_DIR="$DEPLOY_ROOT/scratch/s32_rung1_${SLURM_JOB_ID:?}"
export RUN_DIR

module load cuda/13.0.2 2>/dev/null || true
mkdir -p "$DEPLOY_ROOT/slurm" "$RUN_DIR"

while IFS='=' read -r key _; do
    case "$key" in LUMINA_*|SUPER_*|OMP_NUM*) unset "$key" ;; esac
done < <(env)

# shellcheck source=/dev/null
source "$CERT_ENV"

# The cert env pins the binary of the epoch it certifies.  This rung needs a
# newly built one, so the driver names it explicitly; the sha check below is the
# real gate.  Never overwrite a lettered binary that an existing capture links.
LUMINA_BIN="${STAGE32_BIN_NAME:-$LUMINA_BIN}"
export LUMINA_BIN

RUN_BIN="$DEPLOY_ROOT/$LUMINA_BIN"
MODEL="$DEPLOY_ROOT/$LUMINA_MODEL_DIR"
test -x "$RUN_BIN"
test -d "$MODEL"
test "$(sha256sum "$RUN_BIN" | awk '{print $1}')" = "$STAGE32_BIN_SHA"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
case "$gpu_name" in *H200*|*H100*) ;; *) echo "FATAL: unsupported GPU: $gpu_name" >&2; exit 2 ;; esac
test "$gpu_mem" -ge 80000

ln -s "$RUN_BIN" "$RUN_DIR/$LUMINA_BIN"
ln -s "$DEPLOY_ROOT/data" "$RUN_DIR/data"
cp "$CERT_ENV" "$RUN_DIR/"
mkdir -p "$RUN_DIR/validation/gate_b_dual_oracle"
cd "$RUN_DIR"

export LUMINA_STAGE32_RUNG1_DUMP="$RUN_DIR/stage32_rung1_iter10"
export LUMINA_STAGE32_RUNG1_ITER=10

echo "RUNG1 START host=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION"
echo "GPU=$gpu_name memory_MiB=$gpu_mem"
echo "RUN_BINARY_SHA256=$STAGE32_BIN_SHA"

set +e
./"$LUMINA_BIN" data/tardis_reference_toy06_19p48d_sivcaiv \
    100000 12 spectrum nlte > stdout.log 2> stderr.log
run_rc=$?
set -e
test "$run_rc" -eq 0

# The footer variable count grows whenever an instrument adds a gate. Record it
# rather than pinning it, and check the lines that actually matter one by one.
footer_vars=$(grep -ao 'END RUN FOOTER ([0-9]* vars)' stdout.log | tail -n 1)
test -n "$footer_vars"
echo "footer=$footer_vars"
grep -qxF "  LUMINA_STAGE32_RUNG1_DUMP=$LUMINA_STAGE32_RUNG1_DUMP" stdout.log
grep -qxF '  LUMINA_STAGE32_RUNG1_ITER=10' stdout.log
grep -qxF '  LUMINA_EVENT_LOG=1' stdout.log

# The v4 writer emits "<base>.iter%03d" (R-N4 generation discipline); the base
# path itself is never created.  Derive the real names from the same rule, and
# fail loudly if the generation-stamped payload is absent.
R1_PAYLOAD=$(printf '%s.iter%03d' "$LUMINA_STAGE32_RUNG1_DUMP" "$LUMINA_STAGE32_RUNG1_ITER")
R1_MANIFEST="$R1_PAYLOAD.manifest.json"
test -s "$R1_PAYLOAD"
test -s "$R1_MANIFEST"
test ! -e "$R1_PAYLOAD.quarantine"
sha256sum "$R1_PAYLOAD" "$R1_MANIFEST" > rung1_sha256.txt

# Carry both lineages into the footer so the driver can verify offline without
# reading them back out of the same manifest it is meant to check.
r1_field_gen=$(grep -o '"field_generation"[[:space:]]*:[[:space:]]*[0-9]*' "$R1_MANIFEST" | grep -o '[0-9]*$')
r1_lambda_gen=$(grep -o '"lambda_generation"[[:space:]]*:[[:space:]]*[0-9]*' "$R1_MANIFEST" | grep -o '[0-9]*$')
test -n "$r1_field_gen"
test -n "$r1_lambda_gen"
test "$r1_field_gen" = "$r1_lambda_gen"

echo "=== RUNG1 JOB FOOTER ==="
echo "run_rc=$run_rc"
echo "$footer_vars"
echo "run_binary=$LUMINA_BIN"
echo "r1_payload=$R1_PAYLOAD"
echo "r1_field_generation=$r1_field_gen"
echo "r1_lambda_generation=$r1_lambda_gen"
echo "result_dir=$RUN_DIR"
echo "submission=performed_by_driver_only"
echo "=== END RUNG1 JOB FOOTER ==="
