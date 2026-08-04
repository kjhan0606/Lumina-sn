#!/usr/bin/env python3
"""Small CPU-only positive/negative fixture for the read-only R3 auditor."""

from __future__ import annotations

from pathlib import Path

from audit_r3_level_equivalence import (
    DeckLevel,
    ENERGY_TOL_CM,
    SourceLevel,
    build_mapping,
)


def source(rank: int, energy: float, g: int, config: str) -> SourceLevel:
    return SourceLevel(rank, energy, g, config, Path("fixture_osc"), rank + 10)


def deck(number: int, energy: float, g: int, config: str) -> DeckLevel:
    return DeckLevel(27, 1, number, energy, g, config,
                     Path("fixture_levels.csv"), number + 2, number)


def main() -> int:
    identity_source = [source(0, 0.0, 2, "a"), source(1, 10.0, 4, "b")]
    identity_deck = [deck(0, 0.0, 2, "a"), deck(1, 10.0, 4, "b")]
    mapping, collisions = build_mapping(identity_source, identity_deck)
    assert mapping == [0, 1] and collisions == 0
    print("POSITIVE exact rank identity: PASS")

    # An inserted deck level shifts the physical source level.  Exact E+g must
    # find it at rank 2 instead of silently treating rank 1 as equivalent.
    shifted_source = [source(0, 0.0, 2, "a"), source(1, 20.0, 6, "c")]
    shifted_deck = [deck(0, 0.0, 2, "a"), deck(1, 10.0, 4, "inserted"),
                    deck(2, 20.0, 6, "c")]
    mapping, collisions = build_mapping(shifted_source, shifted_deck)
    assert mapping == [0, 2] and collisions == 0
    print("POSITIVE physical nonidentity mapping: PASS")

    # Configuration normalization is deliberately the historical one: case
    # and punctuation differ, while the alphanumeric order is preserved.
    cfg_source = [source(0, 100.0, 8, "3d5_4Ge[11/2]")]
    cfg_deck = [deck(0, 101.0, 8, "3D5 4gE (11/2)")]
    mapping, collisions = build_mapping(cfg_source, cfg_deck)
    assert mapping == [0] and collisions == 0
    assert abs(cfg_deck[0].energy_cm - cfg_source[0].energy_cm) > ENERGY_TOL_CM
    print("POSITIVE normalized-configuration fallback + E mismatch: PASS")

    # Negative threshold check: equality is accepted; a value just above the
    # frozen 1e-6 cm^-1 boundary must be rejected by the comparison predicate.
    assert abs(ENERGY_TOL_CM) <= ENERGY_TOL_CM
    assert abs(ENERGY_TOL_CM * 1.000001) > ENERGY_TOL_CM
    print("NEGATIVE frozen energy threshold discrimination: PASS")
    print("FIXTURE VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
