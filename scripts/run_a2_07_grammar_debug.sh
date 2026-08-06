#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
out="$repo/validation/a2_07/grammar_debug"
mkdir -p "$out"
cd "$repo"

env | LC_ALL=C sort > "$out/environment.txt"
git rev-parse HEAD > "$out/source_head.txt"
git diff --binary -- src tests scripts docs Makefile > "$out/source_worktree.diff"
sha256sum Makefile src/population_contract.c src/population_contract.h \
  src/lumina.h src/lumina_atomic.c src/lumina_cmfgen.c src/lumina_cuda.cu \
  src/lumina_main.c src/lumina_plasma.c tests/a2_07_population_selftest.c \
  tests/zinert_canonical_tau_fixture.c tests/zinert_population_fixture.c \
  scripts/a2_07_population_census.py scripts/a2_07_population_gate.py \
  scripts/a2_07_classic_sweep.py scripts/a2_07_regression_ledger.py \
  scripts/run_a2_07_grammar_debug.sh scripts/run_a2_07_lageunha.sh \
  docs/SPEC_A2_07_V1.md \
  | sha256sum > "$out/source_tree.sha256"

make lumina 2>&1 | tee "$out/make_lumina.log"
targets=(
  selftest_a2_03_radiation_field
  selftest_a2_04_commit
  selftest_a2_05_bf_rate
  selftest_a2_06_line_jbar
  selftest_a2_06_dual_commit
  selftest_a2_07_population
)
for target in "${targets[@]}"; do
  make "$target" 2>&1 | tee "$out/${target}.build.log"
  "./$target" 2>&1 | tee "$out/${target}.run.log"
done

python3 scripts/a2_07_population_census.py \
  --output "$out/A2_07_STATIC_CENSUS.json" \
  2>&1 | tee "$out/a2_07_population_census.log"
python3 scripts/a2_07_population_gate.py --self-check \
  --output "$out/A2_07_GATE_SELF_CHECK.json" \
  2>&1 | tee "$out/a2_07_population_gate_selfcheck.log"
python3 scripts/a2_07_classic_sweep.py --self-check \
  --output "$out/A2_07_CLASSIC_SWEEP_SELF_CHECK.json" \
  2>&1 | tee "$out/a2_07_classic_sweep_selfcheck.log"
python3 -m py_compile scripts/a2_07_population_census.py \
  scripts/a2_07_population_gate.py scripts/a2_07_classic_sweep.py \
  scripts/a2_07_regression_ledger.py

find "$out" -maxdepth 1 -type f ! -name ARTIFACT_SHA256SUMS.txt -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum \
  > "$out/ARTIFACT_SHA256SUMS.txt"
printf '%s\n' 'A2-07 grammar-debug driver: PASS'
