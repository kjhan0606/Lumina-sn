#!/usr/bin/env python3
"""Fail-closed translation of CMFGEN ion identifiers.

CMFGEN does not use ordinary Roman spelling for every stage: ``2`` denotes
stage II, ``SIX`` denotes stage VI, and ``SEV`` denotes stage VII.  It also
uses ``Sk`` for silicon and ``Nk`` for nickel.  Parse only exact, documented
element-prefix/stage-suffix pairs so labels such as ``FeSIX`` cannot be split
into the fictitious element ``FeS`` and stage IX.
"""

from __future__ import annotations


# This is the convention independently recorded by the CMFGEN POP writer
# parser (scripts/cmfgen_extract/parse_pops.py) and the active VADAT deck.
# Values are spectroscopic stage labels: I=1, II=2, ... .
CMFGEN_STAGE = {
    "I": 1, "2": 2, "III": 3, "IV": 4, "V": 5, "SIX": 6,
    "SEV": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
}

# Values are (chemical symbol, atomic number).  Sk and Nk are CMFGEN tokens,
# not chemical element symbols.  Longest valid prefix matching is essential.
CMFGEN_ELEMENT = {
    "H": ("H", 1), "He": ("He", 2), "C": ("C", 6),
    "N": ("N", 7), "O": ("O", 8), "Ne": ("Ne", 10),
    "Na": ("Na", 11), "Mg": ("Mg", 12), "Al": ("Al", 13),
    "Sk": ("Si", 14), "Si": ("Si", 14), "P": ("P", 15),
    "S": ("S", 16), "Cl": ("Cl", 17), "Ar": ("Ar", 18),
    "K": ("K", 19), "Ca": ("Ca", 20), "Sc": ("Sc", 21),
    "Ti": ("Ti", 22), "V": ("V", 23), "Cr": ("Cr", 24),
    "Mn": ("Mn", 25), "Fe": ("Fe", 26), "Co": ("Co", 27),
    "Ni": ("Ni", 28), "Nk": ("Ni", 28),
}


def roman_stage(value: int) -> str:
    if value <= 0 or value > 20:
        raise ValueError(f"unsupported spectroscopic ion stage: {value}")
    table = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    rendered = ""
    for number, token in table:
        while value >= number:
            rendered += token
            value -= number
    return rendered


def parse_cmfgen_ion_id(ion_id: str) -> tuple[int, int, str]:
    """Return ``(Z, spectroscopic_stage, normalized_species)``.

    Unknown or ambiguous identifiers are rejected rather than omitted or
    guessed.  The raw CMFGEN identifier remains available to the caller for
    provenance.
    """
    if not ion_id or not ion_id.isascii():
        raise ValueError(f"invalid CMFGEN ion identifier: {ion_id!r}")
    candidates: list[tuple[int, int, str]] = []
    for prefix in sorted(CMFGEN_ELEMENT, key=lambda item: (-len(item), item)):
        if not ion_id.startswith(prefix):
            continue
        suffix = ion_id[len(prefix):]
        if suffix not in CMFGEN_STAGE:
            continue
        symbol, atomic_number = CMFGEN_ELEMENT[prefix]
        stage = CMFGEN_STAGE[suffix]
        candidates.append(
            (atomic_number, stage, f"{symbol} {roman_stage(stage)}"))
    if len(candidates) != 1:
        reason = "ambiguous" if candidates else "unrecognized"
        raise ValueError(f"{reason} CMFGEN ion identifier: {ion_id!r}")
    return candidates[0]
