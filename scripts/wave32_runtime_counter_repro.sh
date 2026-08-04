#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RUNTIME_WRAPPED_BENCH FROZEN MODEL OUT_ROOT" >&2
  exit 2
fi

bench=$1
frozen=$2
model=$3
out_root=$4
mkdir -p "$out_root/positive" "$out_root/negative"

common=(
  LUMINA_FROZEN_ORACLE_ONLY_SHELL=8
  LUMINA_SUPER_LEVELS=0
  LUMINA_NLTE_ELEMENT_WIDE=1
  LUMINA_NLTE_ELEMENT_WIDE_Z=26
  LUMINA_NLTE_ELEMENT_WIDE_SHELL=8
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0
  W32_RUNTIME_COUNTER_ONLY=1
)

env -i PATH="$PATH" "${common[@]}" \
  LUMINA_NLTE_ION_LOCK=1 LUMINA_TOPSTAGE_IV=1 \
  "$bench" "$frozen" "$model" "$out_root/positive"

env -i PATH="$PATH" "${common[@]}" \
  W32_RUNTIME_DISABLE_PIN_TOPSTAGE=1 \
  "$bench" "$frozen" "$model" "$out_root/negative"
