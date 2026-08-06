#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="$(mktemp -d /tmp/lumina-zinert.XXXXXX)"
trap 'rm -rf -- "$BUILD_DIR"' EXIT

SERIAL=()
if [[ "${1:-}" == "--serial" ]]; then
    SERIAL=(--serial)
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--serial]" >&2
    exit 2
fi

cd "$REPO_ROOT"
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE \
    -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
    tests/abundance_zero_nlte_fixture.c src/lumina_plasma.c src/bf_rate_jnu.c src/population_contract.c \
    -lm -o "$BUILD_DIR/zinert_validator" >"$BUILD_DIR/validator.build.log" 2>&1 &
P1=$!
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE \
    -DLUMINA_FROZEN_ORACLE -ffunction-sections -fdata-sections \
    -Isrc -Wl,--gc-sections tests/zinert_tau_fixture.c \
    src/lumina_plasma.c src/bf_rate_jnu.c src/population_contract.c -lm -o "$BUILD_DIR/zinert_tau" >"$BUILD_DIR/tau.build.log" 2>&1 &
P2=$!
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE \
    -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
    tests/zinert_population_fixture.c src/lumina_plasma.c src/bf_rate_jnu.c src/population_contract.c \
    -lm -o "$BUILD_DIR/zinert_population" >"$BUILD_DIR/population.build.log" 2>&1 &
P3=$!
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE \
    -DLUMINA_FROZEN_ORACLE -ffunction-sections -fdata-sections \
    -Isrc -Wl,--gc-sections tests/zinert_canonical_tau_fixture.c \
    src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c src/bf_rate_jnu.c src/population_contract.c \
    -lm -o "$BUILD_DIR/zinert_canonical_tau" >"$BUILD_DIR/canonical.build.log" 2>&1 &
P4=$!

for spec in \
    "$P1:$BUILD_DIR/validator.build.log" \
    "$P2:$BUILD_DIR/tau.build.log" \
    "$P3:$BUILD_DIR/population.build.log" \
    "$P4:$BUILD_DIR/canonical.build.log"; do
    pid=${spec%%:*}
    log=${spec#*:}
    if wait "$pid"; then
        :
    else
        rc=$?
        cat "$log" >&2
        exit "$rc"
    fi
done

python3 scripts/run_zinert_selftest.py \
    --validator "$BUILD_DIR/zinert_validator" \
    --tau "$BUILD_DIR/zinert_tau" \
    --population "$BUILD_DIR/zinert_population" \
    --canonical-tau "$BUILD_DIR/zinert_canonical_tau" \
    --deck data/tardis_reference_toy06_19p48d \
    --verify scripts/verify_zinert.py \
    "${SERIAL[@]}"
