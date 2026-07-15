#!/bin/bash
# seq_driver.sh -- hands-free driver for the CMFGEN toy06 SN Ia time sequence.
#
# For each epoch in seq_ladder.txt it:
#   * monitors the run to convergence (cmfgen_convergence.py),
#   * on CONVERGENCE: runs the extraction battery, then builds + launches the next
#     epoch (mk_next_epoch.py -> run.sh),
#   * on CRASH / FINISHED-BUT-NOT-CONVERGED: STOPS and reports (never auto-fixes
#     physics; a human decides).
# Epoch 1 (toy06_2d) is LIVE and is ONLY monitored -- never relaunched, never
# touched.  The driver is resumable via seq_state.txt.
#
#   usage:  nohup bash seq_driver.sh >> seq_logs/seq_driver.log 2>&1 &
#
# Env knobs:
#   SEQ_OMP           OMP threads for launched epochs        (default 32)
#   SEQ_POLL          convergence poll interval, seconds     (default 120)
#   CMF_CONV_PCT      convergence threshold, %% max change   (default 1.0)
#   SEQ_NUM_ITS       NUM_ITS for launched epochs            (default 200)
#   SEQ_MAX_EPOCH     stop after this ts_no                  (default 28)
#   SEQ_DRYRUN        1 = plan only: classify epoch 1, show next-epoch build as a
#                     dry-run, do NOT launch anything.
set -u
BASE="$(cd "$(dirname "$0")" && pwd)"
LADDER="$BASE/seq_ladder.txt"
STATE="$BASE/seq_state.txt"
LOG="$BASE/seq_logs/seq_driver.log"
mkdir -p "$BASE/seq_logs"

SEQ_OMP="${SEQ_OMP:-32}"
SEQ_POLL="${SEQ_POLL:-120}"
export CMF_CONV_PCT="${CMF_CONV_PCT:-1.0}"
SEQ_NUM_ITS="${SEQ_NUM_ITS:-200}"
SEQ_MAX_EPOCH="${SEQ_MAX_EPOCH:-28}"
SEQ_DRYRUN="${SEQ_DRYRUN:-0}"

ts()   { date '+%Y-%m-%d %H:%M:%S'; }
say()  { echo "[$(ts)] $*"; }
die()  { echo "[$(ts)] STOP: $*"; echo "STOP $(ts) :: $*" >> "$STATE"; exit 1; }

classify() { # <dir> -> prints STATUS ; sets global CLS_STATUS
  CLS_STATUS="$(python3 "$BASE/cmfgen_convergence.py" "$1" 2>/dev/null | head -1)"
  echo "$CLS_STATUS"
}

# ------- read ladder into arrays -------------------------------------------
declare -a TSNO AGE DIR
while read -r a b c _; do
  [ -z "${a:-}" ] && continue
  case "$a" in \#*) continue;; esac
  TSNO+=("$a"); AGE+=("$b"); DIR+=("$BASE/$c")
done < "$LADDER"
N=${#TSNO[@]}
say "loaded $N ladder epochs (2.0 -> ${AGE[$((N-1))]} d); CONV_PCT=$CMF_CONV_PCT OMP=$SEQ_OMP"

# ------- determine resume point --------------------------------------------
# start_i = first ladder index (0-based) not yet CONVERGED+extracted.
start_i=0
if [ -f "$STATE" ]; then
  last_done="$(awk -F'=' '/^converged_index=/{v=$2} END{print v}' "$STATE" 2>/dev/null)"
  if [ -n "${last_done:-}" ]; then start_i=$(( last_done + 1 )); fi
fi
say "resume at ladder index $start_i (ts_no ${TSNO[$start_i]:-?}, age ${AGE[$start_i]:-?} d)"

# ------- main loop ----------------------------------------------------------
i=$start_i
while [ "$i" -lt "$N" ]; do
  tsno="${TSNO[$i]}"; age="${AGE[$i]}"; dir="${DIR[$i]}"
  [ "$tsno" -gt "$SEQ_MAX_EPOCH" ] && { say "reached SEQ_MAX_EPOCH=$SEQ_MAX_EPOCH; stopping."; break; }
  say "=== epoch ts_no=$tsno  age=$age d  dir=$dir ==="

  # -- (A) ensure the epoch is running or finished -------------------------
  if [ "$i" -eq 0 ]; then
    # epoch 1 = live toy06_2d: monitor only, never launch/touch
    if [ ! -d "$dir" ]; then die "epoch-1 dir $dir missing"; fi
    say "epoch 1 is the LIVE model; monitoring only (not launching)."
  else
    st="$(classify "$dir")"
    if [ "$st" = "NO_RUN" ] || [ ! -d "$dir" ]; then
      # build from the previous (converged) epoch and launch
      prev="${DIR[$((i-1))]}"
      if [ ! -f "$prev/SN_HYDRO_FOR_NEXT_MODEL" ]; then
        die "previous epoch $prev has no SN_HYDRO_FOR_NEXT_MODEL (not converged?)"
      fi
      if [ "$SEQ_DRYRUN" = 1 ]; then
        say "[DRYRUN] would build epoch $tsno; running mk_next_epoch --dry-run:"
        python3 "$BASE/mk_next_epoch.py" "$prev" "${dir}.DRYRUN" \
            --age "$age" --ts-no "$tsno" --num-its "$SEQ_NUM_ITS" --dry-run
        rm -rf "${dir}.DRYRUN"
        say "[DRYRUN] would then: bash $dir/run.sh ; extract_all.sh $dir"
        break
      fi
      say "building epoch $tsno from $prev ..."
      python3 "$BASE/mk_next_epoch.py" "$prev" "$dir" \
          --age "$age" --ts-no "$tsno" --num-its "$SEQ_NUM_ITS" \
          || die "mk_next_epoch failed for ts_no=$tsno"
      say "launching epoch $tsno (OMP=$SEQ_OMP) ..."
      ( export OMP_NUM_THREADS="$SEQ_OMP"; bash "$dir/run.sh" ) \
          >> "$BASE/seq_logs/epoch_${tsno}.log" 2>&1
      say "epoch $tsno run.sh returned."
    else
      say "epoch $tsno already present (status=$st); not rebuilding."
    fi
  fi

  # -- (B) poll to a terminal state ---------------------------------------
  while :; do
    st="$(classify "$dir")"
    case "$st" in
      RUNNING)
        [ "$SEQ_DRYRUN" = 1 ] && { say "[DRYRUN] epoch $tsno status RUNNING; stopping plan."; exit 0; }
        sleep "$SEQ_POLL";;
      CONVERGED)            break;;
      CRASHED)              die "epoch ts_no=$tsno ($dir) CRASHED/diverged. See $dir/OUTGEN, $dir/batch.log. Not auto-fixing.";;
      FINISHED_NOT_CONVERGED) die "epoch ts_no=$tsno ($dir) FINISHED but did NOT converge (ran out of NUM_ITS or corrections too large). Inspect $dir; a human must decide (e.g. raise NUM_ITS). Not auto-fixing.";;
      NO_RUN)               die "epoch ts_no=$tsno ($dir) shows NO_RUN after launch; unexpected.";;
      *)                    die "epoch ts_no=$tsno ($dir) unknown status '$st'.";;
    esac
  done
  say "epoch ts_no=$tsno CONVERGED."

  # -- (C) extraction battery ---------------------------------------------
  say "extracting epoch $tsno -> $BASE/extract/cmfgen_toy06_${age}d ..."
  bash "$BASE/extract/extract_all.sh" "$dir" "$BASE/extract" "$age" \
      >> "$BASE/seq_logs/extract_${tsno}.log" 2>&1 \
      && say "extraction ok." || say "WARN: extraction returned non-zero (see seq_logs/extract_${tsno}.log)"

  # -- (D) record progress, advance ---------------------------------------
  {
    echo "converged_index=$i"
    echo "converged_ts_no=$tsno"
    echo "converged_age=$age"
    echo "converged_dir=$dir"
    echo "converged_at=$(ts)"
  } >> "$STATE"
  say "state updated (converged_index=$i)."
  i=$(( i + 1 ))
done

if [ "$i" -ge "$N" ]; then
  say "=== SEQUENCE COMPLETE: reached ${AGE[$((N-1))]} d. ==="
  say "Next: validate 19.48d against StaNdaRT phys/ionfrac/spectra, then 3-code cross-check."
fi
