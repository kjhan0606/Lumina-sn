#!/bin/bash
# Unattended chain for the logged-out window (2026-07-27).
#   wait parity31 (running) -> preserve its repo-root artifacts -> launch parity32 -> preserve
# Standing rule honored: if ANY foreign compute process appears on our physical GPU,
# kill our run immediately and stop the chain (feedback_syn08_manual_yield_rule).
# Chaining (not parallel) is mandatory: both runs write the same repo-root artifact
# names, so parity32 must not start before parity31's are preserved.
R=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
GPU_UUID=7b43c3c6                 # syn104 physical GPU 7
BIN_PAT='lumina_cuda[.]withParityS'
LOG=$R/logs/chain_parity31_32.log
ART="lumina_levelpop_resolve_ema.csv lumina_levelpop_resolve_raw.csv lumina_jbar_dump.csv
     lumina_c1_bins.csv cmf_fine_linedump_s8.csv lumina_events.bin lumina_events_lines.bin"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

foreign() {   # echoes foreign pids on our GPU (ours = $1)
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | grep "$GPU_UUID" | awk -F', ' -v p="$1" '$2 != p {print $2}'
}

preserve() {  # $1 = run tag
  local d=$R/logs/coevolve_consume_$1
  for f in $ART; do [ -f "$R/$f" ] && cp -p "$R/$f" "$d/" 2>/dev/null; done
  say "preserved repo-root artifacts -> $d"
}

watch_until_done() {  # $1 = pid, $2 = tag ; returns 0 = completed, 1 = yielded/died
  local pid=$1 tag=$2
  while kill -0 "$pid" 2>/dev/null; do
    local f; f=$(foreign "$pid")
    if [ -n "$f" ]; then
      pkill -f "$BIN_PAT"
      say "AUTO-YIELD: foreign pid(s) $f on GPU — killed $tag, chain stops"
      return 1
    fi
    sleep 30
  done
  if grep -q "END RUN FOOTER" "$R/logs/coevolve_consume_$tag/stdout.log" 2>/dev/null; then
    say "$tag COMPLETED (END RUN FOOTER present)"; return 0
  fi
  say "$tag ENDED WITHOUT FOOTER — chain stops"; return 1
}

say "chain start on $(hostname)"
P31=$(pgrep -f "$BIN_PAT" | head -1)
if [ -z "$P31" ]; then say "parity31 not running at chain start — abort"; exit 1; fi
say "watching parity31 pid=$P31"
watch_until_done "$P31" parity31 || exit 1
preserve parity31

if [ -n "$(foreign 0)" ]; then say "GPU occupied by others — not launching parity32"; exit 1; fi
say "launching parity32 (DB_FB single-variable)"
CUDA_VISIBLE_DEVICES=7 nohup bash $R/scripts/launch_parity32_syn08.sh \
    > $R/logs/parity32_syn104_launch.log 2>&1 &
sleep 25
P32=$(pgrep -f "$BIN_PAT" | head -1)
if [ -z "$P32" ]; then say "parity32 failed to start"; exit 1; fi
say "parity32 pid=$P32"
watch_until_done "$P32" parity32 || exit 1
preserve parity32
say "chain done"
