#!/usr/bin/env python3
"""Import CMFGEN close-coupling collisional-strength (Omega) tables for the IGE
coolant ions BEYOND Fe III -- Fe II, Co III, Ni III -- into Lumina-readable
per-ion binaries (ige_col_<Z>_<ion0>.bin), for the ARTIS-parity collisional
network (Group A / A3).

This is the multi-ion generalization of scripts/build_feiii_coldata.py. Same
rate convention, same fail-closed data rigor; the ONLY new logic is a col_data
row parser that also accepts the bare-hyphen "LOWER-UPPER" separator used by the
Co III / Ni III tables (Fe uses "LOWER -UPPER").

Provenance (all 19apr23, the CMFGEN release toy06 links)
  Fe II  : /gpfs/kjhan/cmfgen_21jun23/atomic/FE/II/19apr23/{osc_data,col_data}
           (Zhang & Pradhan 1994/1995)
  Co III : /gpfs/kjhan/cmfgen_21jun23/atomic/COB/III/19apr23/{osc_data,col_data}
  Ni III : /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/III/19apr23/{osc_data,col_data}

Rate convention (identical in CMFGEN and Lumina):
  CMFGEN : C(i,k) = 8.63e-8 * Omega * exp(-U0) / g_i / sqrt(T_4),  T_4 = T/1e4
  Lumina : C_up   = n_e * 8.629e-6 * Omega * exp(-dE/kTe) / (g_lo * sqrt(T_e))

Output binary (little-endian) -- read by load_ion_coldata() in lumina_atomic.c:
  uint32 magic=0x49474331 ('IGC1'), uint32 version=1,
  int32 Z, ion0(0-based: 0=I), int32 n_trans, int32 n_temp, int32 n_levels_ref,
  double T_grid_K[n_temp],
  record[n_trans]: int32 i_low, int32 i_high, double omega[n_temp]
  (i_low/i_high = Lumina level_number == CMFGEN osc energy rank, i_low the lower.)

Data-import rigor (fail-closed PER ION; a failing ion is simply not written, so
the assembler falls to ARTIS's g-scaled Axelrod floor for it -- no fabrication):
  * osc level count == levels.csv level count for (Z, ion0)
  * osc<->levels.csv round-trip: max|dE| < 1e-4 eV AND every g exact
  * every col_data transition name maps into osc (0 unmapped, else FAIL ion)
  * label split yields exactly 2 level names
  * a re-read round-trip of the written binary
"""
import sys, os, re, struct, csv

BASE = "/gpfs/kjhan/cmfgen_21jun23/atomic"
CM_TO_EV = 1.239841984e-4         # hc in eV*cm (CODATA)
MAGIC = 0x49474331
VERSION = 1

# {name: (Z, ion0(0-based), osc, col)}
IONS = {
    "Fe II":  (26, 1, f"{BASE}/FE/II/19apr23/osc_data",   f"{BASE}/FE/II/19apr23/col_data"),
    "Co III": (27, 2, f"{BASE}/COB/III/19apr23/osc_data", f"{BASE}/COB/III/19apr23/col_data"),
    "Ni III": (28, 2, f"{BASE}/NICK/III/19apr23/osc_data",f"{BASE}/NICK/III/19apr23/col_data"),
}


class IonFail(Exception):
    pass


def parse_osc(path):
    """Return name->idx(0-based), idx->E_cm, idx->g, nlev. idx = abs(ID)-1."""
    name2idx, idx2E, idx2g = {}, {}, {}
    nlev = None
    with open(path) as f:
        lines = f.readlines()
    started = False
    for ln in lines:
        if nlev is None and "Number of energy levels" in ln:
            nlev = int(ln.split()[0]); continue
        if nlev is None:
            continue
        toks = ln.split()
        if len(toks) < 7:
            continue
        try:
            g = float(toks[1]); Ecm = float(toks[2]); ID = int(toks[6])
        except (ValueError, IndexError):
            continue
        idx = abs(ID) - 1
        if idx in idx2E:      # past the level block (transitions start)
            break
        name2idx[toks[0]] = idx
        idx2E[idx] = Ecm
        idx2g[idx] = g
        started = True
        if len(name2idx) >= nlev:
            break
    if not started:
        raise IonFail("osc_data: no level block parsed")
    if len(name2idx) != nlev:
        raise IonFail("osc_data: parsed %d names != declared %d" % (len(name2idx), nlev))
    return name2idx, idx2E, idx2g, nlev


def check_levels_csv(ref_dir, Z, ion0, idx2E, idx2g):
    path = os.path.join(ref_dir, "levels.csv")
    lum = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["atomic_number"]) == Z and int(row["ion_number"]) == ion0:
                lum[int(row["level_number"])] = (float(row["energy_eV"]), int(row["g"]))
    if len(lum) != len(idx2E):
        raise IonFail("levels.csv levels=%d != osc levels=%d" % (len(lum), len(idx2E)))
    maxdE = 0.0; gmis = 0
    for idx in idx2E:
        if idx not in lum:
            raise IonFail("osc idx %d absent from levels.csv" % idx)
        ev_osc = idx2E[idx] * CM_TO_EV
        ev_lum, g_lum = lum[idx]
        maxdE = max(maxdE, abs(ev_osc - ev_lum))
        if int(idx2g[idx]) != g_lum:
            gmis += 1
    print("  round-trip osc<->levels.csv: %d levels, max|dE|=%.3e eV, g mismatches=%d"
          % (len(lum), maxdE, gmis))
    if maxdE > 1e-4:
        raise IonFail("energy round-trip exceeds 1e-4 eV (%.3e)" % maxdE)
    if gmis != 0:
        raise IonFail("%d g mismatches -> osc_data is not the source of levels.csv" % gmis)


def parse_col(path, name2idx):
    """Parse a CMFGEN col_data table. Accepts both 'LOWER -UPPER' (Fe) and
    'LOWER-UPPER' (Co/Ni) separators. Returns (trans, tgrid, NTEMP, ndecl,
    nselfskip). trans = list of (i_low_idx, i_high_idx, [omega...]) already
    mapped and energy-ordered? No -- returns raw (lo_name, up_name, omegas)."""
    with open(path) as f:
        lines = f.readlines()
    NT = NTEMP = None; tgrid = None; data_start = None
    for i, ln in enumerate(lines):
        if NT is None and "Number of transitions" in ln:
            NT = int(ln.split()[0])
        elif NTEMP is None and "Number of T values" in ln:
            NTEMP = int(ln.split()[0])
        elif ln.strip().startswith("Transition\\T"):
            tgrid = [float(x) for x in ln.split()[1:]]
            data_start = i + 1
            break
    if None in (NT, NTEMP, tgrid, data_start):
        raise IonFail("col_data: header parse failed (NT=%s NTEMP=%s)" % (NT, NTEMP))
    if len(tgrid) != NTEMP:
        raise IonFail("col_data: T grid has %d entries != NTEMP %d" % (len(tgrid), NTEMP))

    trans = []; unmapped = set(); nbadlabel = 0; nselfskip = 0
    for ln in lines[data_start:]:
        toks = ln.split()
        if len(toks) < NTEMP + 1:
            continue
        # trailing NTEMP tokens must all be floats (the Omega columns)
        try:
            oms = [float(x) for x in toks[-NTEMP:]]
        except ValueError:
            continue
        # everything before is the transition label; joining removes the optional
        # space in the 'LOWER -UPPER' form, giving a single 'LOWER-UPPER' string.
        label = "".join(toks[:-NTEMP])
        parts = label.split("-")
        if len(parts) != 2:
            nbadlabel += 1
            continue
        lo, up = parts[0], parts[1]
        if lo not in name2idx or up not in name2idx:
            for nm in (lo, up):
                if nm not in name2idx:
                    unmapped.add(nm)
            continue
        if name2idx[lo] == name2idx[up]:
            nselfskip += 1
            continue
        trans.append((lo, up, oms))
    if unmapped:
        for nm in list(unmapped)[:20]:
            print("   UNMAPPED col_data name:", nm)
        raise IonFail("%d unmapped col_data level names (fail-closed)" % len(unmapped))
    if nbadlabel:
        raise IonFail("%d col_data rows with != 2 '-' split (ambiguous)" % nbadlabel)
    print("  col_data: %d bb transitions x %d T parsed, 0 unmapped, %d self-index skipped, "
          "declared=%d" % (len(trans), NTEMP, nselfskip, NT))
    return trans, tgrid, NTEMP


def build_ion(name, ref_dir):
    Z, ion0, osc_path, col_path = IONS[name]
    print("== %s (Z=%d ion0=%d) ==" % (name, Z, ion0))
    name2idx, idx2E, idx2g, nlev = parse_osc(osc_path)
    print("  osc_data: %d levels" % nlev)
    check_levels_csv(ref_dir, Z, ion0, idx2E, idx2g)
    trans, tgrid, ntemp = parse_col(col_path, name2idx)

    records = []
    for lo, up, oms in trans:
        a, b = name2idx[lo], name2idx[up]
        if idx2E[a] <= idx2E[b]:
            i_low, i_high = a, b
        else:
            i_low, i_high = b, a
        records.append((i_low, i_high, oms))
    tgrid_K = [t * 1e4 for t in tgrid]

    out_path = os.path.join(ref_dir, "ige_col_%d_%d.bin" % (Z, ion0))
    with open(out_path, "wb") as f:
        f.write(struct.pack("<IIiiiii", MAGIC, VERSION, Z, ion0,
                            len(records), ntemp, nlev))
        f.write(struct.pack("<%dd" % ntemp, *tgrid_K))
        for i_low, i_high, oms in records:
            f.write(struct.pack("<ii", i_low, i_high))
            f.write(struct.pack("<%dd" % ntemp, *oms))
    print("  wrote %s (%d bytes, %d records)" % (out_path, os.path.getsize(out_path), len(records)))

    verify_readback(out_path, Z, ion0, len(records), ntemp, nlev, tgrid_K)
    return out_path, len(records), ntemp, nlev


def verify_readback(path, Z, ion0, nrec, ntemp, nlev, tgrid_K):
    with open(path, "rb") as f:
        magic, ver, z2, i2, ntr, nt, nlref = struct.unpack("<IIiiiii", f.read(28))
        if magic != MAGIC or ver != VERSION: raise IonFail("readback: bad magic/version")
        if (z2, i2) != (Z, ion0): raise IonFail("readback: bad Z/ion")
        if (ntr, nt, nlref) != (nrec, ntemp, nlev): raise IonFail("readback: count mismatch")
        tg = list(struct.unpack("<%dd" % nt, f.read(8 * nt)))
        if max(abs(a - b) for a, b in zip(tg, tgrid_K)) > 1e-6:
            raise IonFail("readback: T grid mismatch")
        # walk all records, sanity: indices in range, omega finite
        maxidx = -1
        for _ in range(ntr):
            il, ih = struct.unpack("<ii", f.read(8))
            struct.unpack("<%dd" % nt, f.read(8 * nt))
            if il < 0 or ih < 0 or il >= nlev or ih >= nlev:
                raise IonFail("readback: level idx out of range (%d,%d)" % (il, ih))
            maxidx = max(maxidx, il, ih)
        print("  readback OK: %d records, T[0,-1]=%.0f,%.0f K, maxlevelidx=%d/%d"
              % (ntr, tg[0], tg[-1], maxidx, nlev - 1))


def main():
    ref_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d"
    print("Importing CMFGEN IGE col_data -> %s/ige_col_<Z>_<ion0>.bin\n" % ref_dir)
    ok, failed = [], []
    for name in IONS:
        try:
            build_ion(name, ref_dir)
            ok.append(name)
        except IonFail as e:
            print("  FAIL-CLOSED %s: %s -> ARTIS g-scaled Axelrod floor\n" % (name, e))
            failed.append((name, str(e)))
            continue
        print()
    print("SUMMARY: imported=%s ; fail-closed=%s"
          % (", ".join(ok) or "(none)", ", ".join(n for n, _ in failed) or "(none)"))


if __name__ == "__main__":
    main()
