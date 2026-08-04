#!/usr/bin/env python3
"""Build a TARDIS-format atomic HDF5 with CMFGEN priority for SN UV blanketing.

Backbone: NIST weights + ionization energies + GFALL line list (H-Zn).
Override:  CMFGEN-19apr23 levels/lines for iron-peak ions and IME ions
           that LUMINA-SN uses for SN 2011fe.

Output schema is identical to data/atomic/kurucz_cd23_chianti_H_He.h5,
so scripts/expand_atomic_data.py can read it without modification.
"""
import logging, os, sys, time
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

CMFGEN_PATH = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen/"
CONFIG_YML  = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_config_lumina.yml"
OUT_HDF5    = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/kurucz_cd23_cmfgen_lumina.h5"

# Ions to ingest from CMFGEN.  Format follows carsus.parse_selected_species:
# space-separated symbol+ion_charge, semicolons separate items.
CMFGEN_IONS = (
    "C 0-3; "
    "O 0-1; "
    "Mg 1-2; "
    "Al 1-2; "
    "Si 0-2; "
    "S 0-2; "
    "Ca 1-2; "
    "Sc 1; "
    "Ti 1-2; "
    "Cr 1-2; "
    "Mn 1-2; "
    "Fe 1-2; "
    "Co 1-2; "
    "Ni 1-2"
)

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

log(f"output  : {OUT_HDF5}")
log(f"config  : {CONFIG_YML}")
log(f"cmfgen  : {CMFGEN_PATH}")
log(f"ions    : {CMFGEN_IONS}")
log("")

log("[1/6] NIST atomic weights ...")
from carsus.io.nist import NISTWeightsComp, NISTIonizationEnergies
atomic_weights = NISTWeightsComp()

log("[2/6] NIST ionization energies (H-Zn) ...")
ionization_energies = NISTIonizationEnergies("H-Zn")

log("[3/6] GFALL reader (H-Zn backbone) ...")
from carsus.io.kurucz import GFALLReader
gfall_reader = GFALLReader("H-Zn")

log("[4/6] CMFGEN reader with extended LUMINA config ...")
from carsus.io.cmfgen import CMFGENReader
cmfgen_reader = CMFGENReader.from_config(
    CMFGEN_IONS,
    CMFGEN_PATH,
    config_yaml=CONFIG_YML,
    priority=30,
    ionization_energies=False,
    cross_sections=False,
    collisions=True,
    drop_mismatched_labels=True,
)

log("[5/6] Knox-Long zeta ...")
from carsus.io.zeta import KnoxLongZeta
zeta_data = KnoxLongZeta()

log("[6/6] TARDISAtomData assembly + macro-atom prep ...")
from carsus.io.output import TARDISAtomData
atom_data = TARDISAtomData(
    atomic_weights,
    ionization_energies,
    gfall_reader,
    zeta_data,
    cmfgen_reader=cmfgen_reader,
)

log(f"  levels:           {atom_data.levels.shape}")
log(f"  lines:            {atom_data.lines.shape}")
log(f"  macro_atom:       {atom_data.macro_atom.shape}")
log(f"  macro_atom_refs:  {atom_data.macro_atom_references.shape}")

log(f"writing HDF5 -> {OUT_HDF5}")
atom_data.to_hdf(OUT_HDF5)
log(f"DONE  size = {os.path.getsize(OUT_HDF5)/1e6:.1f} MB  total time = {time.time()-t0:.1f}s")
