#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="$(mktemp -d /tmp/lumina-zinert.XXXXXX)"
trap 'rm -rf -- "$BUILD_DIR"' EXIT

cd "$REPO_ROOT"
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE \
    -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
    tests/abundance_zero_nlte_fixture.c src/lumina_plasma.c \
    -lm -o "$BUILD_DIR/zinert_validator"
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE \
    -DLUMINA_FROZEN_ORACLE -ffunction-sections -fdata-sections \
    -Isrc -Wl,--gc-sections tests/zinert_tau_fixture.c \
    src/lumina_plasma.c -lm -o "$BUILD_DIR/zinert_tau"
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE \
    -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
    tests/zinert_population_fixture.c src/lumina_plasma.c \
    -lm -o "$BUILD_DIR/zinert_population"
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE \
    -DLUMINA_FROZEN_ORACLE -ffunction-sections -fdata-sections \
    -Isrc -Wl,--gc-sections tests/zinert_canonical_tau_fixture.c \
    src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c \
    -lm -o "$BUILD_DIR/zinert_canonical_tau"

"$BUILD_DIR/zinert_validator"
set +e
"$BUILD_DIR/zinert_validator" --inject-phantom
negative_rc=$?
set -e
if [[ "$negative_rc" -eq 0 ]]; then
    echo "[Z-INERT-NEGATIVE][FATAL] phantom population was accepted" >&2
    exit 1
fi
echo "[Z-INERT-NEGATIVE] phantom population rejected rc=$negative_rc PASS"

"$BUILD_DIR/zinert_tau"
"$BUILD_DIR/zinert_population"
"$BUILD_DIR/zinert_canonical_tau" \
    data/tardis_reference_toy06_19p48d
python3 scripts/verify_zinert.py
