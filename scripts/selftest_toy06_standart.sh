#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FIXTURE_DIR="$(mktemp -d /tmp/lumina_standart_fixture.XXXXXX)"
trap 'rm -rf -- "$FIXTURE_DIR"' EXIT

cd "$REPO_ROOT"
python3 scripts/selftest_toy06_standart.py
"${CC:-cc}" -std=c11 -Wall -Wextra -Werror \
    tests/abundance_loader_short_row_fixture.c \
    -o "$FIXTURE_DIR/abundance_loader_short_row_fixture"
"$FIXTURE_DIR/abundance_loader_short_row_fixture"

# Apply the proposed src patch only to a disposable copy.  This exercises the
# real shared exclusion helper without modifying the repository src/ tree.
cp -a src "$FIXTURE_DIR/src"
(
    cd "$FIXTURE_DIR"
    git apply --unsafe-paths "$REPO_ROOT/patches/standart_abundance_path_audit.patch"
)
"${CC:-cc}" -O0 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections \
    -I"$FIXTURE_DIR/src" -Wl,--gc-sections \
    tests/abundance_zero_nlte_fixture.c \
    "$FIXTURE_DIR/src/lumina_plasma.c" \
    "$FIXTURE_DIR/src/lumina_atomic.c" \
    "$FIXTURE_DIR/src/lumina_element_wide.c" -lm \
    -o "$FIXTURE_DIR/abundance_zero_nlte_fixture"
"$FIXTURE_DIR/abundance_zero_nlte_fixture"
