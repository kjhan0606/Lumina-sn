#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
out="$repo/validation/a2_07/lageunha"
: "${A2_07_CHAIN_INPUT:?set A2_07_CHAIN_INPUT to the immutable CHAIN manifest}"
: "${A2_07_ORACLE_INPUT:?set A2_07_ORACLE_INPUT to the immutable ORACLE_INPUT manifest}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$out"
cd "$repo"

env | LC_ALL=C sort > "$out/environment.txt"
git rev-parse HEAD > "$out/source_head.txt"
sha256sum "$A2_07_CHAIN_INPUT" "$A2_07_ORACLE_INPUT" > "$out/input_sha256s.txt"
sha256sum Makefile src/population_contract.c src/population_contract.h \
  src/lumina.h src/lumina_atomic.c src/lumina_cmfgen.c src/lumina_cuda.cu \
  src/lumina_main.c src/lumina_plasma.c tests/a2_07_population_selftest.c \
  tests/zinert_canonical_tau_fixture.c tests/zinert_population_fixture.c \
  scripts/a2_07_population_census.py scripts/a2_07_population_gate.py \
  scripts/a2_07_classic_sweep.py scripts/a2_07_regression_ledger.py \
  scripts/run_a2_07_grammar_debug.sh scripts/run_a2_07_lageunha.sh \
  docs/SPEC_A2_07_V1.md \
  | sha256sum > "$out/source_tree.sha256"

python3 scripts/run_gate_battery.py --log "$out/gate_battery.log"
make l1bf_fixture
python3 scripts/a2_05_l1bf_gate.py --controls --out "$out/upstream_a2_05" \
  2>&1 | tee "$out/a2_05_l1bf.log"
python3 scripts/a2_06_l1bb_gate.py --out "$out/upstream_a2_06" \
  2>&1 | tee "$out/a2_06_l1bb.log"

set +e
python3 scripts/a2_07_population_gate.py --input "$A2_07_CHAIN_INPUT" \
  --output "$out/A2_07_CHAIN_RESULT.json" 2>&1 | tee "$out/a2_07_chain.log"
chain_rc=${PIPESTATUS[0]}
python3 scripts/a2_07_population_gate.py --input "$A2_07_ORACLE_INPUT" \
  --output "$out/A2_07_ORACLE_INPUT_RESULT.json" 2>&1 | tee "$out/a2_07_oracle.log"
oracle_rc=${PIPESTATUS[0]}
set -e

for neg in N1 N2 N3 N4; do
  set +e
  python3 scripts/a2_07_population_gate.py --input "$A2_07_ORACLE_INPUT" \
    --negative "$neg" --output "$out/A2_07_${neg}_RESULT.json" \
    2>&1 | tee "$out/a2_07_${neg}.log"
  child_rc=${PIPESTATUS[0]}
  set -e
  if [[ "$child_rc" -ne 4 ]]; then
    printf 'negative %s returned %d, expected 4\n' "$neg" "$child_rc" >&2
    exit 5
  fi
done

python3 scripts/a2_07_classic_sweep.py --log "$out/a2_07_chain.log" \
  --log "$out/a2_07_oracle.log" --metrics "${A2_07_CLASSIC_METRICS:?set A2_07_CLASSIC_METRICS}" \
  --output "$out/A2_07_CLASSIC_SWEEP.json"

python3 scripts/a2_07_population_gate.py --self-check \
  --output "$out/A2_07_GATE_SELF_CHECK.json" >/dev/null
source_hash=$(awk '{print $1}' "$out/source_tree.sha256")
python3 scripts/a2_07_regression_ledger.py \
  --chain "$out/A2_07_CHAIN_RESULT.json" \
  --oracle "$out/A2_07_ORACLE_INPUT_RESULT.json" \
  --self-check "$out/A2_07_GATE_SELF_CHECK.json" \
  --classic "$out/A2_07_CLASSIC_SWEEP.json" \
  --source-hash "$source_hash" \
  --command "bash $repo/scripts/run_a2_07_lageunha.sh" \
  --output "$repo/validation/a2_07/A2_07_REGRESSION_LEDGER.jsonl"

find "$out" -maxdepth 1 -type f ! -name ARTIFACT_SHA256SUMS.txt -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum \
  > "$out/ARTIFACT_SHA256SUMS.txt"
printf 'A2-07 lageunha driver complete: chain_rc=%d oracle_rc=%d\n' \
  "$chain_rc" "$oracle_rc"
if [[ "$chain_rc" -eq 4 || "$chain_rc" -eq 5 || "$oracle_rc" -ne 0 ]]; then
  exit 4
fi
exit 0
