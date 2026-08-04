#!/bin/bash
# Persistent MANUAL runner holding syn104 physical GPU 7 (outside slurm).
# Serves scripts dropped into manual_spool/queue/ one at a time, preserves each
# run's repo-root artifacts, and survives logout (launch with setsid+nohup).
#
# Standing rule (feedback_syn08_manual_yield_rule): our manual run squats a GPU
# outside the slurm ledger, so ANY foreign compute process on the same physical
# GPU means we get out IMMEDIATELY — no thresholds. Here that means: kill our
# binary at once, requeue the interrupted script, and wait for the GPU to go
# idle again before resuming. Yielding is unconditional; the runner itself just
# keeps standing by so work resumes automatically when the slot frees.
#
# Serialization is mandatory (not merely tidy): every run writes the SAME
# repo-root artifact filenames, so two concurrent runs would corrupt each other.
set -u
R=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
GPU_UUID=7b43c3c6                  # syn104 physical GPU 7
export CUDA_VISIBLE_DEVICES=7
SPOOL=$R/manual_spool
LOG=$R/logs/manual_gpu_runner.log
BIN_PAT='lumina_cuda[.]withParity'
IDLE_CONFIRM=10                    # consecutive idle polls required before (re)starting
POLL=30
ART="lumina_levelpop_resolve_ema.csv lumina_levelpop_resolve_raw.csv lumina_jbar_dump.csv
     lumina_c1_bins.csv cmf_fine_linedump_s8.csv lumina_events.bin lumina_events_lines.bin"

mkdir -p "$SPOOL/queue" "$SPOOL/running" "$SPOOL/done" "$R/logs"
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

foreign() {   # echo foreign pids on our GPU; ours = $1 (0 = none of ours)
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | grep "$GPU_UUID" | awk -F', ' -v p="$1" '$2 != p {print $2}'
}

wait_idle() {  # block until GPU has no foreign process for IDLE_CONFIRM polls
  local n=0
  while :; do
    [ -e "$SPOOL/STOP" ] && return 1
    if [ -z "$(foreign 0)" ]; then
      n=$((n+1)); [ "$n" -ge "$IDLE_CONFIRM" ] && return 0
    else
      [ "$n" -gt 0 ] && say "GPU busy again — idle counter reset"
      n=0
    fi
    sleep "$POLL"
  done
}

preserve() {  # $1 = run tag  -- FAIL-CLOSED: only files THIS run actually wrote.
  # Repo root accumulates artifacts across runs. An unconditional copy hands the
  # judge a mixture (e.g. 2026-07-27: a killed parity34 left its own c1_bins next
  # to parity33's levelpop/linedump) and the fossil reads as a real number.
  # $STAMP is touched at run start; anything not newer than it is a fossil and is
  # SKIPPED, so a missing file makes the battery fail loudly instead of lying.
  local d=$R/logs/coevolve_consume_$1 n=0 skip=""
  [ -d "$d" ] || return 0
  for f in $ART; do
    [ -f "$R/$f" ] || continue
    if [ -z "$STAMP" ] || [ "$R/$f" -nt "$STAMP" ]; then
      cp -p "$R/$f" "$d/" 2>/dev/null && n=$((n+1))
    else
      skip="$skip $f"
    fi
  done
  say "preserved $n fresh artifact(s) -> $d"
  [ -n "$skip" ] && say "  SKIPPED as pre-run fossils:$skip"
  return 0
}

envcheck() {  # $1 = launch script (intent)  $2 = pid (authority)  $3 = tag
  local scr=$1 pid=$2 tag=$3 bad=0 k v got
  [ -r "/proc/$pid/environ" ] || { say "envcheck: /proc/$pid/environ unreadable — ABORT"; return 1; }
  # every literal `export LUMINA_X=v` in the launcher must survive to the process
  while IFS= read -r line; do
    for kv in $line; do
      case "$kv" in LUMINA_*=*) ;; *) continue ;; esac
      k=${kv%%=*}; v=${kv#*=}
      case "$v" in \$*|*\$*|\"*|\'*) continue ;; esac   # skip ${VAR:-...} forms
      got=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n "s/^$k=//p" | head -1)
      if [ "$got" != "$v" ]; then
        say "  ENV MISMATCH $k: launcher wants '$v', process has '${got:-<unset>}'"
        bad=$((bad+1))
      fi
    done
  done < <(sed -n 's/^[[:space:]]*export[[:space:]]\+//p' "$scr")
  if [ "$bad" -gt 0 ]; then say "envcheck: $bad gate(s) did not reach $tag"; return 1; fi
  say "envcheck: all launcher LUMINA_* gates confirmed in /proc/$pid/environ"
  return 0
}

say "manual runner up on $(hostname) pid=$$ GPU=$GPU_UUID"
while :; do
  if [ -e "$SPOOL/STOP" ]; then say "STOP file — runner exiting"; rm -f "$SPOOL/STOP"; break; fi
  next=$(ls -1 "$SPOOL/queue" 2>/dev/null | sort | head -1)
  if [ -z "$next" ]; then sleep "$POLL"; continue; fi

  wait_idle || { rm -f "$SPOOL/STOP"; say "STOP during idle wait — exiting"; break; }
  tag=$(grep -o 'P0TAG=[A-Za-z0-9_]*' "$SPOOL/queue/$next" | head -1 | cut -d= -f2)
  tag=${tag:-${next%.sh}}
  mv "$SPOOL/queue/$next" "$SPOOL/running/$next"
  say "START $next (tag=$tag)"
  STAMP=$SPOOL/.runstamp; : > "$STAMP"   # preserve() copies only what is newer
  nohup bash "$SPOOL/running/$next" > "$R/logs/${tag}_manual_launch.log" 2>&1 &
  wrap=$!
  sleep 25
  pid=$(pgrep -f "$BIN_PAT" | head -1)
  if [ -z "$pid" ]; then
    say "$tag FAILED TO START (see logs/${tag}_manual_launch.log)"
    mv "$SPOOL/running/$next" "$SPOOL/done/$next.nostart"; continue
  fi
  say "$tag pid=$pid"

  # ---- fail-closed env-chain preflight -----------------------------------
  # The launch script is only the INTENT. Between it and the binary sits
  # run_coevolve_s01.sh, which may override or unset a gate (2026-07-27: an
  # unconditional `unset LUMINA_CMF_LINERES_CONSUME` silently turned parity34
  # into a byte-clone of parity33 and cost 83 GPU-minutes). Authority is the
  # process environment, so compare intent against /proc/PID/environ and abort
  # in the first 25 s rather than judge a run that never tested anything.
  if ! envcheck "$SPOOL/running/$next" "$pid" "$tag"; then
    pkill -f "$BIN_PAT"; kill "$wrap" 2>/dev/null
    mv "$SPOOL/running/$next" "$SPOOL/done/$next.envmismatch"
    say "$tag ABORTED on env-chain mismatch — NOT requeued (fix the chain first)"
    continue
  fi

  yielded=0
  while kill -0 "$pid" 2>/dev/null; do
    f=$(foreign "$pid")
    if [ -n "$f" ]; then
      pkill -f "$BIN_PAT"; kill "$wrap" 2>/dev/null
      say "AUTO-YIELD: foreign pid(s) $f on GPU — killed $tag, requeued, standing by"
      yielded=1; break
    fi
    if [ -e "$SPOOL/STOP" ]; then
      pkill -f "$BIN_PAT"; kill "$wrap" 2>/dev/null
      say "STOP during $tag — killed and requeued"; yielded=1; break
    fi
    sleep "$POLL"
  done

  if [ "$yielded" = 1 ]; then
    # The retry truncates logs/coevolve_consume_<tag>/{stdout,stderr}.log, so the
    # yielded attempt's evidence vanishes (2026-07-27: a 78-min attempt's log was
    # clobbered by a 26-s retry while it was still being read). Park it first.
    d=$R/logs/coevolve_consume_$tag
    if [ -f "$d/stdout.log" ]; then
      k=1; while [ -e "$d/stdout.log.yield$k" ]; do k=$((k+1)); done
      for L in stdout stderr; do [ -f "$d/$L.log" ] && mv "$d/$L.log" "$d/$L.log.yield$k"; done
      say "  parked yielded attempt's logs as *.log.yield$k"
    fi
    mv "$SPOOL/running/$next" "$SPOOL/queue/$next"     # retry when the slot frees
    sleep "$POLL"; continue
  fi
  if grep -q "END RUN FOOTER" "$R/logs/coevolve_consume_$tag/stdout.log" 2>/dev/null; then
    preserve "$tag"
    mv "$SPOOL/running/$next" "$SPOOL/done/$next.ok"
    say "$tag COMPLETED (END RUN FOOTER present)"
  else
    preserve "$tag"
    mv "$SPOOL/running/$next" "$SPOOL/done/$next.nofooter"
    say "$tag ENDED WITHOUT FOOTER — artifacts preserved, NOT requeued (needs a human look)"
  fi
done
say "manual runner down"
