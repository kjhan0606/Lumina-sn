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
    tests/abundance_zero_nlte_fixture.c src/lumina_plasma.c src/bf_rate_jnu.c \
    src/population_contract.c src/opacity_publication.c src/emissivity_publication.c \
    src/radeq_publication.c src/gpu_radiation_field_contract.c src/jnu_seed.c \
    src/seed_capability.c src/gpu_physics_contract.c \
    -lm -o "$BUILD_DIR/zinert_validator" >"$BUILD_DIR/validator.build.log" 2>&1 &
P1=$!
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE \
    -DLUMINA_FROZEN_ORACLE -ffunction-sections -fdata-sections \
    -Isrc -Wl,--gc-sections tests/zinert_tau_fixture.c \
    src/lumina_plasma.c src/bf_rate_jnu.c src/population_contract.c \
    src/opacity_publication.c src/emissivity_publication.c src/radeq_publication.c \
    src/gpu_radiation_field_contract.c src/jnu_seed.c src/seed_capability.c \
    src/gpu_physics_contract.c -lm -o "$BUILD_DIR/zinert_tau" >"$BUILD_DIR/tau.build.log" 2>&1 &
P2=$!
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE \
    -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
    tests/zinert_population_fixture.c src/lumina_plasma.c src/bf_rate_jnu.c \
    src/population_contract.c src/opacity_publication.c src/emissivity_publication.c \
    src/radeq_publication.c src/gpu_radiation_field_contract.c src/jnu_seed.c \
    src/seed_capability.c src/gpu_physics_contract.c \
    -lm -o "$BUILD_DIR/zinert_population" >"$BUILD_DIR/population.build.log" 2>&1 &
P3=$!
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE \
    -DLUMINA_FROZEN_ORACLE -ffunction-sections -fdata-sections \
    -Isrc -Wl,--gc-sections tests/zinert_canonical_tau_fixture.c \
    src/lumina_plasma.c src/lumina_element_wide.c src/bf_rate_jnu.c \
    src/population_contract.c src/lumina_atomic.c src/opacity_publication.c \
    src/emissivity_publication.c src/radeq_publication.c \
    src/gpu_radiation_field_contract.c src/jnu_seed.c src/seed_capability.c \
    src/gpu_physics_contract.c \
    -lm -o "$BUILD_DIR/zinert_canonical_tau" >"$BUILD_DIR/canonical.build.log" 2>&1 &
P4=$!
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE -Isrc \
    tests/a2_08_signed_opacity_selftest.c src/opacity_publication.c -lm \
    -o "$BUILD_DIR/a2_08_signed_opacity" >"$BUILD_DIR/a2_08.build.log" 2>&1 &
P5=$!
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE -Isrc \
    tests/a2_09_emissivity_selftest.c src/emissivity_publication.c -lm \
    -o "$BUILD_DIR/a2_09_emissivity" >"$BUILD_DIR/a2_09.build.log" 2>&1 &
P6=$!
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE -Isrc \
    tests/a2_10_radeq_selftest.c src/radeq_publication.c src/population_contract.c -lm \
    -o "$BUILD_DIR/a2_10_radeq" >"$BUILD_DIR/a2_10.build.log" 2>&1 &
P7=$!
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE -Isrc \
    tests/a2_12_contract_selftest.c src/gpu_radiation_field_contract.c \
    src/seed_capability.c -o "$BUILD_DIR/a2_12_contract" >"$BUILD_DIR/a2_12.build.log" 2>&1 &
P8=$!
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE -Isrc \
    tests/a2_13_15_contract_selftest.c src/gpu_physics_contract.c \
    -o "$BUILD_DIR/a2_13_15_contract" >"$BUILD_DIR/a2_13_15.build.log" 2>&1 &
P9=$!
"${CC:-gcc}" -O2 -w -std=gnu11 -D_GNU_SOURCE -Isrc \
    tests/a2_17_jnu_seed_selftest.c src/jnu_seed.c src/radiation_field.c \
    src/seed_capability.c -lm -o "$BUILD_DIR/a2_17_jnu_seed" \
    >"$BUILD_DIR/a2_17.build.log" 2>&1 &
P10=$!

for spec in \
    "$P1:$BUILD_DIR/validator.build.log" \
    "$P2:$BUILD_DIR/tau.build.log" \
    "$P3:$BUILD_DIR/population.build.log" \
    "$P4:$BUILD_DIR/canonical.build.log" \
    "$P5:$BUILD_DIR/a2_08.build.log" \
    "$P6:$BUILD_DIR/a2_09.build.log" \
    "$P7:$BUILD_DIR/a2_10.build.log" \
    "$P8:$BUILD_DIR/a2_12.build.log" \
    "$P9:$BUILD_DIR/a2_13_15.build.log" \
    "$P10:$BUILD_DIR/a2_17.build.log"; do
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
    --a2-08 "$BUILD_DIR/a2_08_signed_opacity" \
    --a2-09 "$BUILD_DIR/a2_09_emissivity" \
    --a2-10 "$BUILD_DIR/a2_10_radeq" \
    --a2-12-contract "$BUILD_DIR/a2_12_contract" \
    --a2-13-15-contract "$BUILD_DIR/a2_13_15_contract" \
    --a2-17-jnu-seed "$BUILD_DIR/a2_17_jnu_seed" \
    --deck data/tardis_reference_toy06_19p48d \
    --verify scripts/verify_zinert.py \
    "${SERIAL[@]}"
