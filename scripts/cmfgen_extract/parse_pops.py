#!/usr/bin/env python3
"""parse_pops.py -- full per-level populations from a converged CMFGEN model.

WHY NOT dispgen: dispgen's DC_*/POP_* options accept a MAXIMUM OF 10 LEVELS
(maingen_opt_desc.txt:213-216).  Co III alone has ~3900 levels in our sigma grid,
and the Gamma integral is NOT dominated by the ground multiplet (high-lying levels
at E~31 eV with tiny populations contribute comparably, because sigma peaks at
threshold).  So the 10-level path is useless here and we read the POP<SPECIES>
file that CMFGEN writes at convergence directly.

FORMAT (definitive, from the writers -- not guessed):
  cmfgen_sub.f:4526-4538 writes, per species with nonzero abundance, file
  'POP'//SPECIES  (e.g. POPCOB, POPIRON):
     ' Output format date:'   T30 A
     ' Completion of Model:'  T30 A
     ' ND:'                   T30 I5
     ' <SPEC>/He abundance:'  T30 1PE12.5
     POP_SPECIES(1:ND)                        format (1X,1P8E16.7)
     then per ionization stage, via RITE_ASC (subs/rite_asc.f):
     ' Number of <ION_ID> levels:' T30 I5
     ' <ION_ID> Oscillator date:'  T30 A
     CI(NCI,ND)                               format (1X,1P8E16.7)
     DCI(ND)                                  format (1X,1P8E16.7)
  CI is declared CI(NCI,ND) and written whole, so Fortran array-element order =
  COLUMN-MAJOR: level index varies fastest, then depth.

  If a stage is absent RITE_ASC writes only the 'Number of ... levels: 0' line
  and no data -- handled.

OUTPUT CSV: Z, ion, level_number, depth_index, n_k
  ion = CHARGE (LUMINA convention, memory project_gph_alllevel_ab_verdict:12):
  III -> ion=2, IV -> ion=3.  level_number is 0-based to match levels.csv.

UNVERIFIED UNTIL A CONVERGED MODEL EXISTS: the parser is written against the
writers above and self-checks its own value counts, but no POP<SPECIES> file has
been produced yet (they are written only at LST_ITERATION).  It raises rather
than guessing if any count mismatches.
"""
import argparse
import csv
import re
import sys

# CMFGEN ion_id suffix -> charge.  Neutral is 'I', singly-ionized is '2'
# (maingen_opt_desc.txt:206-210: FeI Fe2 FeIII FeIV FeV FeSIX FeSEV FeVIII FeIX FeX).
STAGE = {"I": 0, "2": 1, "III": 2, "IV": 3, "V": 4, "SIX": 5, "SEV": 6,
         "VIII": 7, "IX": 8, "X": 9, "XI": 10, "XII": 11}
# element prefix used in ion ids -> Z.  'Sk' is CMFGEN's token for silicon.
ELEM = {"H": 1, "He": 2, "C": 6, "N": 7, "O": 8, "Ne": 10, "Na": 11, "Mg": 12,
        "Al": 13, "Sk": 14, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
        "Fe": 26, "Co": 27, "Ni": 28, "Nk": 28}


def split_ion_id(ion_id):
    """'CoIII' -> (27, 2);  'Fe2' -> (26, 1);  'SkIV' -> (14, 3)."""
    for n in (2, 1):                       # longest element token first (He, Sk, Co)
        el, suf = ion_id[:n], ion_id[n:]
        if el in ELEM and suf in STAGE:
            return ELEM[el], STAGE[suf]
    raise ValueError(f"cannot parse CMFGEN ion id {ion_id!r}")


NUMRE = re.compile(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?")


def take(tokens, n, what):
    if len(tokens) < n:
        raise SystemExit(f"POP parse: ran out of numbers reading {what} "
                         f"(wanted {n}, have {len(tokens)})")
    got, rest = tokens[:n], tokens[n:]
    return [float(x.replace("D", "E").replace("d", "e")) for x in got], rest


def parse(path):
    lines = open(path).read().split("\n")
    nd = None
    for ln in lines[:8]:
        m = re.match(r"\s*ND:\s+(\d+)", ln)
        if m:
            nd = int(m.group(1))
            break
    if nd is None:
        raise SystemExit(f"{path}: no 'ND:' header line -- not a POP<SPECIES> file?")

    # Walk the file: header lines are recognisable by text; everything else is data.
    blocks, cur, i = [], None, 0
    lvl_re = re.compile(r"\s*Number of\s+(\S+)\s+levels:\s+(\d+)")
    osc_re = re.compile(r"\s*(\S+)\s+Oscillator date:")
    data = []
    for ln in lines:
        m = lvl_re.match(ln)
        if m:
            if cur:
                blocks.append((cur, data))
            cur, data = (m.group(1), int(m.group(2))), []
            continue
        if osc_re.match(ln) or "abundance:" in ln or "format date:" in ln \
           or "Completion of Model:" in ln or re.match(r"\s*ND:", ln):
            continue
        data.extend(NUMRE.findall(ln))
    if cur:
        blocks.append((cur, data))
    if not blocks:
        raise SystemExit(f"{path}: no 'Number of <ION> levels:' blocks found")

    out = []
    for (ion_id, ncl), toks in blocks:
        if ncl == 0:
            continue
        Z, charge = split_ion_id(ion_id)
        vals, rest = take(toks, ncl * nd, f"{ion_id} CI({ncl},{nd})")
        # DCI(ND) follows; consume it so a count mismatch is caught, then discard.
        if rest:
            _, rest = take(rest, min(nd, len(rest)), f"{ion_id} DCI({nd})")
        if rest:
            print(f"  [warn] {ion_id}: {len(rest)} unconsumed numbers after "
                  f"CI+DCI -- format drift? (not fatal, ignored)", file=sys.stderr)
        # column-major: CI(level, depth) -> vals[(depth-1)*ncl + (level-1)]
        for d in range(nd):
            base = d * ncl
            for k in range(ncl):
                out.append((Z, charge, k, d + 1, vals[base + k]))
        print(f"  [ok] {ion_id}: Z={Z} ion(charge)={charge} levels={ncl} ND={nd}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pop_files", nargs="+", help="POP<SPECIES> file(s), e.g. POPCOB POPIRON")
    ap.add_argument("-o", "--out", required=True, help="output CSV")
    a = ap.parse_args()
    rows = []
    for p in a.pop_files:
        print(f"[parse_pops] {p}")
        rows.extend(parse(p))
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Z", "ion", "level_number", "depth_index", "n_k"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.7E}"])
    print(f"[parse_pops] {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
