#!/bin/bash
# Run all golden physics tests. Exit 0 iff every test passes.
# Adds a one-line PASS/FAIL summary per test plus an overall verdict.
set -u
cd "$(dirname "$0")"

declare -a TESTS=(
    "test_beta_sobolev.py"
    "test_planck_bnu.py"
)

fail=0
for t in "${TESTS[@]}"; do
    if python3 "$t" > "/tmp/golden_${t%.py}.out" 2>&1; then
        echo "PASS  $t"
    else
        echo "FAIL  $t  (see /tmp/golden_${t%.py}.out)"
        fail=$((fail + 1))
    fi
done

echo
if [ "$fail" -eq 0 ]; then
    echo "=== all ${#TESTS[@]} golden tests passed ==="
    exit 0
fi
echo "=== $fail / ${#TESTS[@]} golden tests FAILED ==="
exit 1
