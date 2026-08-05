#!/bin/bash
# A2-02C amended Part 1, stages 1--5.  Run on lageunha, never a login node.
set -euo pipefail

ROOT=${A2_02C_ROOT:-/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn}
OUT=${A2_02C_OUT:-$ROOT/validation/a2_02c}
cd "$ROOT"

PYTHONPYCACHEPREFIX=/tmp/a2_02c_pycache python3 -m py_compile \
  scripts/a2_02c_frequency_union.py \
  scripts/a2_02_prepare_fine_dump.py \
  scripts/a2_02_resolution_ladder.py \
  scripts/a2_02c_segment_replay.py \
  scripts/a2_02c_capture_gate_selftest.py

PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_frequency_union.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_frequency_union.py negative-controls
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_frequency_union.py build \
  --deck data/tardis_reference_toy06_19p48d_sivcaiv_ftos \
  --old-union docs/A2_02_FREQUENCY_UNION.json \
  --old-result validation/a2_02/a2_02_resolution_result.json \
  --output-dir "$OUT"

PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02_prepare_fine_dump.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02_prepare_fine_dump.py \
  --edd /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/EDDFACTOR \
  --rvtj /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ \
  --chieta /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10 \
  --deck data/tardis_reference_toy06_19p48d_sivcaiv_ftos \
  --ledger "$OUT/A2_02C_FREQUENCY_UNION.json" \
  --cohort "$OUT/A2_02C_ESTIMATOR_COHORT.json" \
  --template docs/A2_02C_RESOLUTION_INPUT_TEMPLATE.json \
  --output "$OUT/a2_02c_fine_bin_averages.npz" \
  --manifest "$OUT/A2_02C_RESOLUTION_INPUT.json"

PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02_resolution_ladder.py self-test
set +e
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02_resolution_ladder.py run \
  --manifest "$OUT/A2_02C_RESOLUTION_INPUT.json" \
  --output "$OUT/a2_02c_global_resolution_result.json"
ladder_rc=$?
set -e
if [ "$ladder_rc" -ne 0 ] && [ "$ladder_rc" -ne 3 ]; then
  exit "$ladder_rc"
fi
echo "A2_02C_PART1_DONE ladder_rc=$ladder_rc expected=0_or_3 out=$OUT"
exit "$ladder_rc"
