#!/bin/bash
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the repository used to build the patched binary}"
: "${LUMINA_BIN:?set LUMINA_BIN to a binary built with patches/standart_abundance_path_audit.patch}"
DECK="${1:-$REPO_ROOT/data/tardis_reference_toy06_19p48d_standart}"
LOG="${2:-$REPO_ROOT/abundance_dump_only.log}"

[[ -x "$LUMINA_BIN" ]] || { echo "not executable: $LUMINA_BIN" >&2; exit 2; }
[[ -d "$DECK" ]] || { echo "deck absent: $DECK" >&2; exit 2; }
cd "$REPO_ROOT"
unset CUDA_VISIBLE_DEVICES
LUMINA_ABUNDANCE_DUMP_ONLY=1 "$LUMINA_BIN" "$DECK" 2>&1 | tee "$LOG"

grep -q '^\[ABUNDANCE-SUMMARY\].*excluded_zero=9 .*zero_shells=0/50 .*runtime_renormalization=NONE$' "$LOG"
grep -q '^\[ABUNDANCE-DUMP-ONLY\].*physics and transport not entered$' "$LOG"
echo "ABUNDANCE_DUMP_VERDICT=PASS log=$LOG"
