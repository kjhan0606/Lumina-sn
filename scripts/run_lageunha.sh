#!/usr/bin/env bash
# ============================================================================
# run_lageunha.sh — launch a pure-CMFGEN run on the lageunha server.
#
# WHY: the syntax slurm cluster is congested (Priority queue, multi-day waits).
# lageunha has a dedicated RTX 5000 Ada (32 GB, CC 8.9), NO slurm, and a SHARED
# filesystem (same /home/kjhan/... paths). Our binary runs on Ada (sm_86 cubin
# is minor-forward-compatible to 8.9). With super-levels the NLTE matrix fits
# 32 GB. This wrapper bakes the champion + super-level config and handles the
# cd / env-var-name / OMP / nohup details I kept getting wrong by hand.
#
# RUN THIS FROM syntax (it ssh-es into lageunha). Shared fs => no copy.
#
# USAGE:
#   scripts/run_lageunha.sh TAG [KEY=VAL ...]
#   scripts/run_lageunha.sh jeqb1 JEQB=1
#   scripts/run_lageunha.sh superA SUPER_LEVELS=0          # cutoff/super off
#   scripts/run_lageunha.sh dump13 LUMINA_NLTE_MATDUMP=1 LUMINA_POP_SHELL=13
# Overrides (KEY=VAL) WIN over the baked defaults (later assignment wins).
# Work dir: logs/ddc15_pc_phase3_<tag-suffix>_TAG ; driver log: logs/TAG.driver.log
# ============================================================================
set -euo pipefail
REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
HOST=lageunha

TAG="${1:-}"; [ -z "$TAG" ] && { echo "usage: run_lageunha.sh TAG [KEY=VAL ...]" >&2; exit 2; }
shift || true

# --- baked champion + super-level + emergent defaults (harness var names) ---
DEF="OMP_NUM_THREADS=32 SUPER_LEVELS=1 LUMINA_SUPER_CUTOFF=100 \
LSTAR=1 LINE_RE=1 TE_RATIO=1.0 FROZENIN=1 JNU_PHOTOION=1 LINERES_JBAR=1 \
FINE_EMERGENT=1 LAMLO=3000 LAMHI=12000 FINE_SL_CLAMP=1.0 N_ITER=3 FINE_DIAG=1"

# overrides appended after defaults -> win (A=1 A=2 cmd: last wins)
OVR=""
for kv in "$@"; do
  case "$kv" in *=*) OVR="$OVR $kv";; *) echo "bad arg '$kv' (KEY=VAL)" >&2; exit 2;; esac
done

echo "=== run_lageunha: tag=$TAG ==="
echo "  defaults : $DEF"
echo "  overrides: ${OVR:-(none)}"

ssh -o ConnectTimeout=10 "$HOST" \
  "cd $REPO && SLURM_JOB_ID=$TAG $DEF $OVR nohup bash scripts/slurm_ddc15_pure_cmfgen_phase3.sh > logs/$TAG.driver.log 2>&1 </dev/null & echo '  launched PID '\$!"

echo "  driver log: logs/$TAG.driver.log"
echo "  work dir  : logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_$TAG  (shared fs, read from syntax)"
echo "  monitor   : grep -E '\\[CMFGEN\\] iter|Super-levels:' logs/ddc15_pc_phase3_*_$TAG/stdout.log"
