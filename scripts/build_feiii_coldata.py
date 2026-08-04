#!/usr/bin/env python3
"""Import CMFGEN Fe III collisional data (Zhang 1996) into a Lumina-readable
binary table (feiii_col_zhang.bin).

Provenance
----------
  Collision strengths : /gpfs/kjhan/cmfgen_21jun23/atomic/FE/III/19apr23/col_data
                        (Zhang H.L., A&A Sup. Ser. 119, 523; 22139 transitions x
                        20 T; toy06 links FeIII_COL_DATA -> this file)
  Level names/energies: .../19apr23/osc_data (1500 levels; toy06 FeIII_F_OSCDAT)
  Cross-check          : Lumina levels.csv (must be the SAME osc_data, so the
                         level_number == osc energy rank identity holds)

Rate convention (identical in CMFGEN and Lumina):
  CMFGEN : C(i,k) = 8.63e-8 * Omega * exp(-U0) / g_i / sqrt(T_4),  T_4 = T/1e4
  Lumina : C_up   = n_e * 8.629e-6 * Omega * exp(-dE/kTe) / (g_lo * sqrt(T_e))
  (8.63e-8/sqrt(T_4) == 8.63e-6/sqrt(T); Omega is symmetric.)

Output binary (little-endian) — see load_feiii_coldata() in lumina_atomic.c:
  uint32 magic=0x46454333('FEC3'), uint32 version=1,
  int32 Z=26, ion=2, int32 n_trans, int32 n_temp, int32 n_levels_ref,
  double T_grid_K[n_temp],
  record[n_trans]: int32 i_low, int32 i_high, double omega[n_temp]
  (i_low/i_high = Lumina level_number, i_low the lower-energy level.)

Data-import rigor: osc<->levels.csv round-trip (energy + g), zero unmapped
col_data names, transition-count match, and a re-read round-trip of the written
binary — any violation aborts (fail-closed, no silent correction).
"""
import sys, os, re, struct, csv

OSC = "/gpfs/kjhan/cmfgen_21jun23/atomic/FE/III/19apr23/osc_data"
COL = "/gpfs/kjhan/cmfgen_21jun23/atomic/FE/III/19apr23/col_data"
CM_TO_EV = 1.239841984e-4         # hc in eV*cm (CODATA)
MAGIC = 0x46454333
VERSION = 1

def die(msg):
    print("FATAL:", msg); sys.exit(1)

def parse_osc(path):
    """Return dict name->idx(0-based), and lists idx->(E_cm, g). idx = abs(ID)-1."""
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
        # data row: name g E(cm) 10^15Hz eV Lam ID arad c4 c6
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
        die("osc_data: no level block parsed")
    if len(name2idx) != nlev:
        die("osc_data: parsed %d names != declared %d" % (len(name2idx), nlev))
    return name2idx, idx2E, idx2g, nlev

def check_levels_csv(ref_dir, idx2E, idx2g):
    path = os.path.join(ref_dir, "levels.csv")
    lum = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["atomic_number"]) == 26 and int(row["ion_number"]) == 2:
                lum[int(row["level_number"])] = (float(row["energy_eV"]), int(row["g"]))
    if len(lum) != len(idx2E):
        die("levels.csv FeIII levels=%d != osc levels=%d" % (len(lum), len(idx2E)))
    maxdE = 0.0; gmis = 0
    for idx in idx2E:
        if idx not in lum:
            die("osc idx %d absent from levels.csv" % idx)
        ev_osc = idx2E[idx] * CM_TO_EV
        ev_lum, g_lum = lum[idx]
        maxdE = max(maxdE, abs(ev_osc - ev_lum))
        if int(idx2g[idx]) != g_lum:
            gmis += 1
    print("  round-trip osc<->levels.csv: %d levels, max|dE|=%.3e eV, g mismatches=%d"
          % (len(lum), maxdE, gmis))
    if maxdE > 1e-4:
        die("energy round-trip exceeds 1e-4 eV (%.3e) -> level order mismatch" % maxdE)
    if gmis != 0:
        die("%d g mismatches -> osc_data is not the source of levels.csv" % gmis)

def parse_col(path, name2idx):
    with open(path) as f:
        lines = f.readlines()
    NT = NTEMP = None; tgrid = None; data_start = None
    for i, ln in enumerate(lines):
        if "Number of transitions" in ln: NT = int(ln.split()[0])
        elif "Number of T values" in ln: NTEMP = int(ln.split()[0])
        elif ln.strip().startswith("Transition\\T"):
            tgrid = [float(x) for x in ln.split()[1:]]
            data_start = i + 1
            break
    if None in (NT, NTEMP, tgrid, data_start):
        die("col_data: header parse failed (NT=%s NTEMP=%s)" % (NT, NTEMP))
    if len(tgrid) != NTEMP:
        die("col_data: T grid has %d entries != NTEMP %d" % (len(tgrid), NTEMP))
    pat = re.compile(r'^(\S+)\s+-(\S+)\s+(.*)$')
    trans = []; unmapped = set()
    for ln in lines[data_start:]:
        m = pat.match(ln)
        if not m:
            continue
        lo, up, rest = m.group(1), m.group(2), m.group(3)
        oms = rest.split()
        if len(oms) != NTEMP:
            continue
        for nm in (lo, up):
            if nm not in name2idx:
                unmapped.add(nm)
        trans.append((lo, up, [float(x) for x in oms]))
    if unmapped:
        for nm in list(unmapped)[:20]:
            print("   UNMAPPED col_data name:", nm)
        die("%d unmapped col_data level names (fail-closed)" % len(unmapped))
    if len(trans) != NT:
        die("col_data: parsed %d transitions != declared %d" % (len(trans), NT))
    print("  col_data: %d transitions x %d T parsed, 0 unmapped names" % (len(trans), NTEMP))
    return trans, tgrid, NTEMP

def main():
    ref_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d"
    out_path = os.path.join(ref_dir, "feiii_col_zhang.bin")
    print("Importing CMFGEN Fe III col_data (Zhang 1996) -> %s" % out_path)

    name2idx, idx2E, idx2g, nlev = parse_osc(OSC)
    print("  osc_data: %d Fe III levels" % nlev)
    check_levels_csv(ref_dir, idx2E, idx2g)
    trans, tgrid, ntemp = parse_col(COL, name2idx)

    # Build records: order (i_low, i_high) by energy; drop self/degenerate-index.
    records = []
    nskip_self = 0
    for lo, up, oms in trans:
        a, b = name2idx[lo], name2idx[up]
        if a == b:
            nskip_self += 1; continue
        # order by energy so i_low is the lower-energy level
        if idx2E[a] <= idx2E[b]:
            i_low, i_high = a, b
        else:
            i_low, i_high = b, a
        records.append((i_low, i_high, oms))
    print("  records: %d (%d self-index skipped)" % (len(records), nskip_self))

    tgrid_K = [t * 1e4 for t in tgrid]

    with open(out_path, "wb") as f:
        f.write(struct.pack("<IIiiiii", MAGIC, VERSION, 26, 2,
                            len(records), ntemp, nlev))
        f.write(struct.pack("<%dd" % ntemp, *tgrid_K))
        for i_low, i_high, oms in records:
            f.write(struct.pack("<ii", i_low, i_high))
            f.write(struct.pack("<%dd" % ntemp, *oms))
    print("  wrote %d bytes" % os.path.getsize(out_path))

    # ---- round-trip: re-read and verify header + a spot transition ----
    verify_readback(out_path, name2idx, idx2E, tgrid_K, ntemp, nlev)
    print("DONE. feiii_col_zhang.bin ready (n_trans=%d, n_temp=%d)." % (len(records), ntemp))

def verify_readback(path, name2idx, idx2E, tgrid_K, ntemp, nlev):
    with open(path, "rb") as f:
        magic, ver, Z, ion, ntr, nt, nlref = struct.unpack("<IIiiiii", f.read(28))
        if magic != MAGIC or ver != VERSION: die("readback: bad magic/version")
        if (Z, ion) != (26, 2): die("readback: bad Z/ion")
        if nt != ntemp or nlref != nlev: die("readback: n_temp/n_levels mismatch")
        tg = list(struct.unpack("<%dd" % nt, f.read(8 * nt)))
        if max(abs(a - b) for a, b in zip(tg, tgrid_K)) > 1e-6:
            die("readback: T grid mismatch")
        # find the level 25 <-> level 17 transition (5.083 <-> 3.731 eV);
        # expected Omega(T=1e4 K) = 8.76 (from the Zhang table, verified).
        i25 = name2idx["3d5(6S)4s_5Se[2]"]      # == 25
        i17 = name2idx["3d5(6S)4s_7Se[3]"]      # == 17
        if (i25, i17) != (25, 17):
            die("readback: sanity level indices off (%d,%d)" % (i25, i17))
        t_idx = tg.index(1.0e4) if 1.0e4 in tg else 4
        found = None
        for _ in range(ntr):
            il, ih = struct.unpack("<ii", f.read(8))
            oms = struct.unpack("<%dd" % nt, f.read(8 * nt))
            if {il, ih} == {i25, i17}:
                found = oms[t_idx]; break
        if found is None:
            die("readback: 25<->17 transition not found")
        if abs(found - 8.76) > 1e-6:
            die("readback: 25<->17 Omega(1e4K)=%.4f != 8.76" % found)
        print("  readback OK: 25<->17 Omega(1e4K)=%.3f (expected 8.76); T grid[0,-1]=%.0f,%.0f K"
              % (found, tg[0], tg[-1]))

if __name__ == "__main__":
    main()
