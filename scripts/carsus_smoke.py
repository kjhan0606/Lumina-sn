#!/usr/bin/env python3
"""Smoke test: build a tiny TARDIS-format atomic HDF5 with Fe II + Si II via carsus.

Validates the toolchain (NIST + GFALL + CMFGEN priority + zeta + macro-atom prep)
end-to-end before running the full 14-element rebuild.
"""
import logging
import os
logging.basicConfig(level=logging.INFO)

CMFGEN_PATH = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen/"
OUT_HDF5 = "/tmp/carsus_smoke_FeII_SiII.h5"

print("[1/6] NIST atomic weights ...")
from carsus.io.nist import NISTWeightsComp, NISTIonizationEnergies
atomic_weights = NISTWeightsComp()

print("[2/6] NIST ionization energies (H-Zn) ...")
ionization_energies = NISTIonizationEnergies("H-Zn")

print("[3/6] GFALL reader (H-Zn) ...")
from carsus.io.kurucz import GFALLReader
gfall_reader = GFALLReader("H-Zn")

print("[4/6] CMFGEN reader (Si 0-1, Fe 1) ...")
from carsus.io.cmfgen import CMFGENReader
cmfgen_reader = CMFGENReader.from_config(
    "Si 0-1; Fe 1",
    CMFGEN_PATH,
    priority=30,
    ionization_energies=False,
    cross_sections=False,
    collisions=True,
    drop_mismatched_labels=True,
)

print("[5/6] Knox-Long zeta ...")
from carsus.io.zeta import KnoxLongZeta
zeta_data = KnoxLongZeta()

print("[6/6] TARDISAtomData assembly ...")
from carsus.io.output import TARDISAtomData
atom_data = TARDISAtomData(
    atomic_weights,
    ionization_energies,
    gfall_reader,
    zeta_data,
    cmfgen_reader=cmfgen_reader,
)

print(f"  levels:           {atom_data.levels.shape}")
print(f"  lines:            {atom_data.lines.shape}")
print(f"  macro_atom:       {atom_data.macro_atom.shape}")
print(f"  macro_atom_refs:  {atom_data.macro_atom_references.shape}")

print(f"\n[done] Writing to {OUT_HDF5}")
atom_data.to_hdf(OUT_HDF5)
print("OK", os.path.getsize(OUT_HDF5)/1e6, "MB")
