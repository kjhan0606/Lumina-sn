#!/usr/bin/env python3
"""Full-coverage import of CMFGEN col_data collision strengths (Upsilon/Omega)
for EVERY ion present in a Lumina reference directory's levels.csv.

This is the coverage-complete generalization of scripts/build_ige_coldata.py
(which hard-codes 3 ions and hard-fails whenever the CMFGEN osc level count
differs from levels.csv).  Purpose: replace the invented LUMINA_RADEQ_OMEGA_FLOOR
(Omega >= 1) with real data wherever CMFGEN actually has it, and to MEASURE
exactly where it does not (that measurement is the input to the floor decision).

Nothing is fabricated.  Every relaxation vs build_ige_coldata.py is a strictly
fail-closed one:

  * TRUNCATED reference level sets (levels.csv keeps the first N of the CMFGEN
    osc_data levels, e.g. Ca III 200 of 232, Fe IV 200 of 1000).  We verify the
    PREFIX IDENTITY -- for every level_number k < N, energy and g must match
    osc_data rank k -- and then keep only the transitions whose BOTH endpoints
    are < N.  Transitions reaching above the cap are DROPPED, never remapped.
  * Fortran 'D' exponents and 'LOWER-UPPER' / 'LOWER -UPPER' label separators.
  * col_data rows without a '-' (the Omega(i,i) collisional-IONIZATION block)
    are counted and skipped rather than aborting the ion.
  * Any ion whose T grid, level round-trip or name mapping fails is written as
    a manifest row with the reason and NO binary -- it keeps whatever fallback
    the code already had.

Rate convention (identical in CMFGEN and Lumina):
  CMFGEN : C(i,k) = 8.63e-8 * Omega * exp(-U0) / g_i / sqrt(T_4),  T_4 = T/1e4
  Lumina : C_up   = n_e * 8.629e-6 * Omega * exp(-dE/kTe) / (g_lo * sqrt(T_e))
CMFGEN col_data tabulates the T axis in units of 1e4 K; we store Kelvin.

Output binary (little-endian) -- read by load_ion_coldata() in lumina_atomic.c:
  uint32 magic=0x49474331 ('IGC1'), uint32 version=1,
  int32 Z, ion0(0-based: 0=I), int32 n_trans, int32 n_temp, int32 n_levels_ref,
  double T_grid_K[n_temp],
  record[n_trans]: int32 i_low, int32 i_high, double omega[n_temp]

Usage:
  build_cmfgen_coldata_all.py --ref-dir DIR --audit      # table only, writes nothing
  build_cmfgen_coldata_all.py --ref-dir DIR --write      # bake bins + manifest
"""
import sys, os, re, struct, csv, argparse, math

BASE = "/gpfs/kjhan/cmfgen_21jun23/atomic"
VINTAGE = "19apr23"
CM_TO_EV = 1.239841984e-4         # hc in eV*cm (CODATA)
MAGIC = 0x49474331
VERSION = 1
E_TOL_EV = 1e-4

ELEM_DIR = {6: "CARB", 7: "NIT", 8: "OXY", 9: "FLU", 10: "NEON", 11: "NA",
            12: "MG", 13: "AL", 14: "SIL", 15: "PHOS", 16: "SUL", 17: "CHL",
            18: "ARG", 19: "POT", 20: "CA", 21: "SCAN", 22: "TIT", 23: "VAN",
            24: "CHRO", 25: "MAN", 26: "FE", 27: "COB", 28: "NICK"}
ELEM_SYM = {6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg",
            13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K",
            20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
            26: "Fe", 27: "Co", 28: "Ni"}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


class IonFail(Exception):
    pass


def fortran_float(tok):
    """CMFGEN tables mix 1.0E+00, 1.0D+00 and bare 0.3D0."""
    return float(tok.replace("D", "E").replace("d", "e"))


# ---------------------------------------------------------------- osc_data ---
def parse_osc(path):
    """Return name->idx(0-based), idx->E_cm, idx->g, nlev_declared.

    Robust to the varying column layout between CMFGEN species files.  A level
    row is  NAME  G  E(cm-1)  <floats...>  ID  [<floats...>] : the ID is the one
    BARE INTEGER token (every other numeric column is written with a '.' or an
    exponent), and it runs 1,2,3,...  Its column index is 6 for the IGE files
    but 5 for e.g. Ca III, so we locate it by type rather than by position.
    Some files carry a negative ID for levels with no bound-free route -> abs().
    """
    INT_RE = re.compile(r"^[+-]?\d+$")
    with open(path, errors="replace") as f:
        lines = f.readlines()
    nlev = None
    for ln in lines:
        if "Number of energy levels" in ln:
            nlev = int(ln.split()[0])
            break
    if nlev is None:
        raise IonFail("osc_data: 'Number of energy levels' not found")

    name2idx, idx2E, idx2g = {}, {}, {}
    dup_names = 0
    want = 1
    for ln in lines:
        toks = ln.split()
        if len(toks) < 4:
            continue
        ID = None
        for t in toks[3:]:
            if INT_RE.match(t):
                ID = abs(int(t))
                break
        if ID is None or ID != want:
            continue
        try:
            g = fortran_float(toks[1]); Ecm = fortran_float(toks[2])
        except ValueError:
            continue
        idx = ID - 1
        if toks[0] in name2idx:
            dup_names += 1
        name2idx[toks[0]] = idx
        idx2E[idx] = Ecm
        idx2g[idx] = g
        want += 1
        if want > nlev:
            break
    if len(idx2E) != nlev:
        raise IonFail("osc_data: parsed %d level rows != declared %d"
                      % (len(idx2E), nlev))
    if dup_names:
        raise IonFail("osc_data: %d duplicate level names (name map ambiguous)" % dup_names)
    return name2idx, idx2E, idx2g, nlev


# ---------------------------------------------------------------- col_data ---
# Everything below reproduces CMFGEN's own reader,
#   /gpfs/kjhan/cmfgen_src/cur_cmf/subs/gen_omega_rd_v2.f  (GEN_OMEGA_RD_V2),
# line for line, so that what we bake is what CMFGEN would have used:
#   L236-246  T grid = the FIRST NUM_TVALS values after 'ion\\T'  (a longer grid
#             line is truncated, not an error -- this is what unlocks Ca IV)
#   L274      exactly NUM_TRANS non-blank records are consumed (L524-527)
#   L284-285  LOW_LEV = text before the FIRST '-'
#   L346-353  UP_LEV  = text up to the first DOUBLE space; the remaining numbers
#             are read list-directed, so only the first NUM_TVALS are used
#   L292-339  lower-level match: exact name, or (col name J-resolved) the model
#             level named by the term part; a col TERM name matches ALL model
#             J-members within +/-MAX_LEV_SEP (L135-145) of the first match
#   L361-366  UP_LEV == 'I' is a collisional-IONIZATION record, not bound-bound
#   L413/446/484-486  Omega(i,j) += COL_VEC * g_i * g_j / (GL_SUM*GU_SUM)
#             -- note '+=': REPEATED level pairs are SUMMED, not overwritten
#             (Ni IV ships 95 such repeats)
#   L466      NL == NUP records are dropped
#   L437      any tabulated value <= 0 is a fatal error in CMFGEN


def parse_col_header(path):
    """Return (n_trans_decl, n_temp_decl, tgrid, data_start, lines)."""
    with open(path, errors="replace") as f:
        lines = f.readlines()
    NT = NTEMP = None; tgrid = None; data_start = None
    for i, ln in enumerate(lines):
        if NT is None and "Number of transitions" in ln:
            try: NT = int(ln.split()[0])
            except ValueError: pass
        elif NTEMP is None and "Number of T values" in ln:
            try: NTEMP = int(ln.split()[0])
            except ValueError: pass
        elif "Transition\\T" in ln:
            rest = ln.split("ion\\T", 1)[1]
            vals = []
            for t in rest.split():
                try: vals.append(fortran_float(t))
                except ValueError: break      # Fortran list-read stops at junk
            tgrid = vals
            data_start = i + 1
            break
    return NT, NTEMP, tgrid, data_start, lines


# Deliberately does NOT match the 'Set to zero for the time being / Changed
# OMEGA to 0.1 to be consistant with previous models' boilerplate: that comment
# describes the OMEGA_SET default for NON-tabulated transitions and appears even
# in files with excellent tables (Fe II/III/IV).  Only flag statements about the
# tabulated values themselves.
SUSPECT_RE = re.compile(r"wild guess|are guesses|value are guesses|"
                        r"arbitrar|dummy|placeholder|made up|invented", re.I)


def col_quality_note(lines, data_start):
    """CMFGEN annotates several col_data files as estimates rather than
    calculations ('Wild guesses based on CI data', 'Changed OMEGA to 0.1 to be
    consistant with previous models', ...).  Surface those verbatim so a reader
    of the manifest never mistakes a placeholder for a close-coupling table."""
    hits = []
    for ln in lines[:data_start]:
        t = ln.strip()
        if t and SUSPECT_RE.search(t) and not t.startswith("*"):
            hits.append(" ".join(t.split())[:90])
    seen = set(); out = []
    for h in hits:
        if h not in seen:
            seen.add(h); out.append(h)
    return " | ".join(out[:3])


def term_of(name):
    k = name.find("[")
    return name if k < 0 else name[:k]


def build_level_maps(name2idx, nlev):
    """name -> idx, term -> [idx...], and CMFGEN's MAX_LEV_SEP (L135-145)."""
    idx2name = {v: k for k, v in name2idx.items()}
    term2idx = {}
    for nm, i in name2idx.items():
        term2idx.setdefault(term_of(nm), []).append(i)
    for t in term2idx:
        term2idx[t].sort()
    max_lev_sep = 1
    for t, mem in term2idx.items():
        if len(mem) > 1 and "[" in idx2name[mem[0]]:
            max_lev_sep = max(max_lev_sep, mem[-1] - mem[0])
    return idx2name, term2idx, max_lev_sep


def match_level(nm, name2idx, term2idx, idx2name, max_lev_sep):
    """CMFGEN GEN_OMEGA_RD_V2 level matching.  Returns (list_of_idx, mode) with
    mode in {'exact', 'unsplit', 'term'}, or (None, reason)."""
    if "[" in nm:
        # L295-309: only a single unique match is possible
        if nm in name2idx:
            return [name2idx[nm]], "exact"
        t = term_of(nm)
        if t in name2idx:                       # model level is NOT J-split
            return [name2idx[t]], "unsplit"
        return None, "no_match"
    # L310-338: col name is a TERM -> every J-member of that term
    if nm in name2idx and "[" not in nm:
        return [name2idx[nm]], "exact"
    mem = term2idx.get(nm)
    if not mem:
        return None, "no_match"
    if mem[-1] - mem[0] > max_lev_sep:
        # outside CMFGEN's +/-MAX_LEV_SEP window -> membership is search-order
        # dependent; refuse rather than guess.
        return None, "term_span_%d_gt_max_lev_sep_%d" % (mem[-1] - mem[0], max_lev_sep)
    return list(mem), "term"


def parse_col(path, name2idx, idx2g, nlev):
    """CMFGEN-exact read.  Returns (pairs, tgrid_1e4K, ntemp, stats) where
    pairs = {(idx_a, idx_b): [omega...]} in osc-index space, already
    g-redistributed and accumulated exactly as OMEGA_TABLE would be."""
    NT, NTEMP, tgrid, data_start, lines = parse_col_header(path)
    if NTEMP is None:
        raise IonFail("col_data: 'Number of T values' not found")
    if NT is None:
        raise IonFail("col_data: 'Number of transitions' not found")
    if NT == 0:
        raise IonFail("col_data declares 0 transitions -> CMFGEN itself has NO "
                      "tabulated Omega for this ion (it uses the approximate "
                      "formulae only; cf omega_gen_v2.f L151-191)")
    if not tgrid or data_start is None:
        raise IonFail("col_data: 'Transition\\T' grid line not found/parsable")
    if len(tgrid) < NTEMP:
        raise IonFail("col_data: T-grid line has %d values < declared NTEMP %d"
                      % (len(tgrid), NTEMP))
    n_tgrid_extra = len(tgrid) - NTEMP
    tgrid = tgrid[:NTEMP]                                  # gen_omega_rd L246
    for i in range(NTEMP - 1):
        if tgrid[0] <= 0.0 or tgrid[i] >= tgrid[i + 1]:
            raise IonFail("col_data: T values not positive/monotonic at i=%d" % i)

    idx2name, term2idx, max_lev_sep = build_level_maps(name2idx, nlev)
    pairs = {}
    st = dict(n_rows=0, n_ioniz=0, n_nomatch=0, n_self=0, n_short=0, n_nonpos=0,
              max_sumrule_err=0.0,
              n_term_rows=0, n_term_reject=0, n_accum=0, n_tgrid_extra=n_tgrid_extra,
              max_lev_sep=max_lev_sep)
    nomatch_names = set(); term_reject = set()
    order = []
    nread = 0
    for ln in lines[data_start:]:
        if nread >= NT:
            break
        if not ln.strip():
            continue                                       # L524-527
        if "-" not in ln:
            continue
        L = ln.index("-")
        lo_nm = ln[:L].strip()
        rest = ln[L + 1:].lstrip()
        m = re.search(r"  ", rest)
        if not m:
            continue
        up_nm = rest[:m.start()].strip()
        if not lo_nm or not up_nm:
            continue
        vals = []
        for t in rest[m.start():].split():
            try: vals.append(fortran_float(t))
            except ValueError: break
        if len(vals) < NTEMP:
            st["n_short"] += 1
            nread += 1
            continue
        oms = vals[:NTEMP]                                 # list-directed read
        nread += 1
        st["n_rows"] += 1
        if min(oms) <= 0.0:
            st["n_nonpos"] += 1                            # CMFGEN L437: fatal
            continue
        if up_nm == "I":
            st["n_ioniz"] += 1
            continue
        lo_idx, lo_mode = match_level(lo_nm, name2idx, term2idx, idx2name, max_lev_sep)
        up_idx, up_mode = match_level(up_nm, name2idx, term2idx, idx2name, max_lev_sep)
        if lo_idx is None or up_idx is None:
            for nm, r in ((lo_nm, lo_mode), (up_nm, up_mode)):
                if isinstance(r, str) and r.startswith("term_span"):
                    st["n_term_reject"] += 1
                    term_reject.add(nm)
            if (lo_idx is None and str(lo_mode) == "no_match") or \
               (up_idx is None and str(up_mode) == "no_match"):
                st["n_nomatch"] += 1
                if lo_idx is None: nomatch_names.add(lo_nm)
                if up_idx is None: nomatch_names.add(up_nm)
            continue
        if lo_mode == "term" or up_mode == "term":
            st["n_term_rows"] += 1
        gl_sum = sum(idx2g[i] for i in lo_idx)
        gu_sum = sum(idx2g[j] for j in up_idx)
        norm = 1.0 / gl_sum / gu_sum
        # One-to-one match: g_i*g_j/(GL_SUM*GU_SUM) is 1 by construction, so skip
        # the round-trip (it only injects ~4e-16 of rounding).
        unit_w = (len(lo_idx) == 1 and len(up_idx) == 1)
        if set(lo_idx).isdisjoint(up_idx):
            # gen_omega_rd_v2 L423-425: SUM_i SUM_j Omega(i,j) must recover the
            # tabulated multiplet OMEGA exactly.  Verify it, don't assume it.
            wsum = sum(norm * idx2g[i] * idx2g[j] for i in lo_idx for j in up_idx)
            st["max_sumrule_err"] = max(st["max_sumrule_err"], abs(wsum - 1.0))
        for i in lo_idx:
            for j in up_idx:
                if i == j:
                    st["n_self"] += 1
                    continue                               # L466
                # CMFGEN keys LOC_INDX by the ORDERED (NL,NUP): a repeat of the
                # SAME ordered pair accumulates (+=), while the reverse order is
                # a separate slot that omega_gen_v2 later ASSIGNS (last wins).
                key = (i, j)
                w = 1.0 if unit_w else norm * idx2g[i] * idx2g[j]
                if key in pairs:
                    st["n_accum"] += 1
                    pairs[key] = [a + w * b for a, b in zip(pairs[key], oms)]
                else:
                    pairs[key] = [w * b for b in oms]
                order.append(key)

    # Collapse the ordered LOC_INDX slots to physical (unordered) level pairs.
    # If both (i,j) and (j,i) were filled they are two distinct CMFGEN slots and
    # omega_gen_v2's OMEGA(I,J)= assignment keeps the LAST one written.
    last_pos = {}
    for k, key in enumerate(order):
        last_pos[key] = k
    out = {}; n_bidir = 0; max_bidir_rel = 0.0
    for (i, j), v in pairs.items():
        a, b = (i, j) if i < j else (j, i)
        rev = (j, i)
        if rev in pairs:
            n_bidir += 1
            u = pairs[rev]
            den = max(max(abs(x) for x in v), max(abs(x) for x in u))
            if den > 0:
                max_bidir_rel = max(max_bidir_rel,
                                    max(abs(x - y) for x, y in zip(v, u)) / den)
            if last_pos[rev] > last_pos[(i, j)]:
                continue                                   # the reverse slot wins
        out[(a, b)] = v
    st["n_decl"] = NT
    st["n_read"] = nread
    st["n_bidir"] = n_bidir // 2 if n_bidir else 0
    st["max_bidir_rel"] = max_bidir_rel
    st["nomatch_names"] = sorted(nomatch_names)
    st["term_reject_names"] = sorted(term_reject)
    st["quality_note"] = col_quality_note(lines, data_start)
    return out, tgrid, NTEMP, st


# ------------------------------------------------------------- levels.csv ----
def load_ref_levels(ref_dir):
    """(Z,ion0) -> {level_number: (E_eV, g)}"""
    out = {}
    with open(os.path.join(ref_dir, "levels.csv")) as f:
        for r in csv.DictReader(f):
            key = (int(r["atomic_number"]), int(r["ion_number"]))
            out.setdefault(key, {})[int(r["level_number"])] = (
                float(r["energy_eV"]), int(r["g"]))
    return out


def check_prefix(lum, idx2E, idx2g):
    """levels.csv must be the energy-ordered PREFIX of osc_data.
    Returns (N, max_dE_eV, n_g_mismatch)."""
    N = len(lum)
    if sorted(lum) != list(range(N)):
        raise IonFail("levels.csv level_number set is not 0..%d contiguous" % (N - 1))
    if N > len(idx2E):
        raise IonFail("levels.csv has %d levels > osc_data %d (not a prefix)"
                      % (N, len(idx2E)))
    maxdE = 0.0; gmis = 0
    for k in range(N):
        ev_lum, g_lum = lum[k]
        maxdE = max(maxdE, abs(idx2E[k] * CM_TO_EV - ev_lum))
        if int(idx2g[k]) != g_lum:
            gmis += 1
    if maxdE > E_TOL_EV:
        raise IonFail("prefix energy round-trip max|dE|=%.3e eV > %.0e "
                      "(levels.csv is not this osc_data's prefix)" % (maxdE, E_TOL_EV))
    if gmis:
        raise IonFail("%d g mismatches in prefix (levels.csv is not this osc_data)" % gmis)
    return N, maxdE, gmis


# ------------------------------------------------------------------ build ----
def build_ion(Z, ion0, lum, ref_dir, write, verbose=True, sources=None):
    """Returns a manifest dict."""
    edir = ELEM_DIR.get(Z)
    ion_name = "%s %s" % (ELEM_SYM.get(Z, "Z%d" % Z), ROMAN[ion0])
    m = dict(ion=ion_name, Z=Z, ion0=ion0, osc="", col="",
             n_levels_osc="", n_levels_ref=len(lum),
             n_trans_source="", n_temp="", n_mapped=0, n_dropped=0,
             drop_reasons="", omega_min="", omega_max="", omega_median="",
             max_sumrule_err="", cmfgen_quality_note="", out_bin="", status="")
    if edir is None:
        m["status"] = "SKIP: no CMFGEN element directory for Z=%d" % Z
        return m
    selected = sources.get((Z, ion0)) if sources is not None else None
    if selected is not None:
        osc_path, col_path = selected
        d = os.path.dirname(osc_path)
    else:
        d = "%s/%s/%s/%s" % (BASE, edir, ROMAN[ion0], VINTAGE)
        osc_path, col_path = d + "/osc_data", d + "/col_data"
    m["osc"], m["col"] = osc_path, col_path
    if not os.path.exists(osc_path) or not os.path.exists(col_path):
        m["status"] = "SKIP: no %s/{osc_data,col_data}" % d
        return m

    try:
        name2idx, idx2E, idx2g, nlev_osc = parse_osc(osc_path)
        m["n_levels_osc"] = nlev_osc
        N, maxdE, gmis = check_prefix(lum, idx2E, idx2g)
        pairs, tgrid, ntemp, st = parse_col(col_path, name2idx, idx2g, nlev_osc)
        m["n_trans_source"] = st["n_decl"]
        m["n_temp"] = ntemp
    except IonFail as e:
        m["status"] = "SKIP: %s" % e
        return m

    if st["n_term_reject"]:
        m["status"] = ("SKIP: %d rows whose col_data TERM spans a non-contiguous "
                       "J block (> MAX_LEV_SEP=%d) e.g. %s -- membership would be "
                       "search-order dependent"
                       % (st["n_term_reject"], st["max_lev_sep"],
                          st["term_reject_names"][:3]))
        return m
    if st["n_nonpos"]:
        m["status"] = ("SKIP: %d rows with a tabulated Omega <= 0 (CMFGEN "
                       "gen_omega_rd_v2 L437 treats this as fatal)" % st["n_nonpos"])
        return m
    if st["n_short"]:
        m["status"] = "SKIP: %d rows with fewer than NTEMP=%d values" % (st["n_short"], ntemp)
        return m
    if st["n_read"] != st["n_decl"]:
        m["status"] = ("SKIP: consumed %d records but header declares %d"
                       % (st["n_read"], st["n_decl"]))
        return m
    if not pairs:
        m["status"] = ("SKIP: 0 bound-bound pairs after CMFGEN name matching "
                       "(%d rows, %d no-match, %d ionization)"
                       % (st["n_rows"], st["n_nomatch"], st["n_ioniz"]))
        return m

    # ---- apply the levels.csv truncation (drop, never remap) ----------------
    recs = {}
    n_trunc = 0; n_inv = 0
    for (a, b), oms in pairs.items():
        if a >= N or b >= N:
            n_trunc += 1
            continue
        # osc index order is energy order; assert it and store energy-ordered.
        if idx2E[a] > idx2E[b]:
            n_inv += 1
            i_low, i_high = b, a
        else:
            i_low, i_high = a, b
        recs[(i_low, i_high)] = oms
    if not recs:
        m["status"] = ("SKIP: all %d bb pairs lie above the levels.csv cap N=%d "
                       "(osc has %d)" % (len(pairs), N, nlev_osc))
        m["n_dropped"] = len(pairs)
        return m

    vals = [v for oms in recs.values() for v in oms]
    n_neg = sum(1 for v in vals if v < 0.0)
    n_nan = sum(1 for v in vals if not math.isfinite(v))
    if n_neg or n_nan:
        m["status"] = "SKIP: Omega sanity failed (negatives=%d nonfinite=%d)" % (n_neg, n_nan)
        return m

    sv = sorted(vals)
    m["omega_min"] = "%.6g" % sv[0]
    m["omega_max"] = "%.6g" % sv[-1]
    m["omega_median"] = "%.6g" % sv[len(sv) // 2]
    m["n_mapped"] = len(recs)
    m["n_dropped"] = st["n_ioniz"] + st["n_nomatch"] + n_trunc
    m["drop_reasons"] = (
        "nomatch_rows=%d(names=%d);ioniz_rows=%d;self_pairs=%d;above_levels_cap=%d;"
        "accum_repeats=%d;term_rows=%d;bidir_slots=%d;tgrid_extra_cols=%d"
        % (st["n_nomatch"], len(st["nomatch_names"]), st["n_ioniz"], st["n_self"],
           n_trunc, st["n_accum"], st["n_term_rows"], st["n_bidir"],
           st["n_tgrid_extra"]))
    m["max_sumrule_err"] = "%.3e" % st["max_sumrule_err"]
    m["cmfgen_quality_note"] = st["quality_note"]
    if st["max_sumrule_err"] > 1e-10:
        m["status"] = ("SKIP: term->J redistribution sum rule violated "
                       "(max rel err %.3e)" % st["max_sumrule_err"])
        return m

    tgrid_K = [t * 1e4 for t in tgrid]
    out_name = "ige_col_%d_%d_cmfgen.bin" % (Z, ion0)
    out_path = os.path.join(ref_dir, out_name)
    m["out_bin"] = out_name

    records = sorted(recs.items())
    if write:
        with open(out_path, "wb") as f:
            f.write(struct.pack("<IIiiiii", MAGIC, VERSION, Z, ion0,
                                len(records), ntemp, N))
            f.write(struct.pack("<%dd" % ntemp, *tgrid_K))
            for (i_low, i_high), oms in records:
                f.write(struct.pack("<ii", i_low, i_high))
                f.write(struct.pack("<%dd" % ntemp, *oms))
        try:
            verify_readback(out_path, Z, ion0, records, ntemp, N, tgrid_K)
        except IonFail as e:
            os.remove(out_path)
            m["out_bin"] = ""
            m["status"] = "SKIP: readback verification failed: %s" % e
            return m
    m["status"] = "OK"
    if verbose:
        print("  %-8s Z=%2d ion0=%d osc=%5d ref=%5d src=%6d -> pairs=%6d "
              "(cap=%d nomatch=%d ioniz=%d self=%d acc=%d term=%d inv=%d) nT=%2d "
              "Om[%s..%s] med=%s"
              % (ion_name, Z, ion0, nlev_osc, N, st["n_decl"], len(records),
                 n_trunc, st["n_nomatch"], st["n_ioniz"], st["n_self"],
                 st["n_accum"], st["n_term_rows"], n_inv, ntemp,
                 m["omega_min"], m["omega_max"], m["omega_median"]))
    return m


def load_source_manifest(path):
    """Read exact osc/col choices emitted by expand_atomic_data_cmfgen.py."""
    required = {"atomic_number", "ion_number", "osc_path", "col_path"}
    result = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise IonFail("source manifest missing columns: %s" % sorted(missing))
        for row in reader:
            key = (int(row["atomic_number"]), int(row["ion_number"]))
            if key in result:
                raise IonFail("duplicate source-manifest ion %s" % (key,))
            if not row["osc_path"] or not row["col_path"]:
                raise IonFail("source manifest has blank osc/col for %s" % (key,))
            result[key] = (row["osc_path"], row["col_path"])
    return result


def verify_readback(path, Z, ion0, records, ntemp, nlev_ref, tgrid_K):
    """Byte-level round-trip: header, T grid, every record index+value."""
    with open(path, "rb") as f:
        magic, ver, z2, i2, ntr, nt, nlref = struct.unpack("<IIiiiii", f.read(28))
        if magic != MAGIC or ver != VERSION: raise IonFail("bad magic/version")
        if (z2, i2) != (Z, ion0): raise IonFail("bad Z/ion")
        if (ntr, nt, nlref) != (len(records), ntemp, nlev_ref):
            raise IonFail("count mismatch %s vs %s" % ((ntr, nt, nlref),
                                                       (len(records), ntemp, nlev_ref)))
        tg = list(struct.unpack("<%dd" % nt, f.read(8 * nt)))
        if max(abs(a - b) for a, b in zip(tg, tgrid_K)) > 1e-9:
            raise IonFail("T grid mismatch")
        for k, ((i_low, i_high), oms) in enumerate(records):
            il, ih = struct.unpack("<ii", f.read(8))
            back = list(struct.unpack("<%dd" % nt, f.read(8 * nt)))
            if (il, ih) != (i_low, i_high):
                raise IonFail("record %d index mismatch" % k)
            if il < 0 or ih < 0 or il >= nlev_ref or ih >= nlev_ref:
                raise IonFail("record %d index out of levels.csv range" % k)
            if back != oms:
                raise IonFail("record %d omega bits differ" % k)
        if f.read(1):
            raise IonFail("trailing bytes after last record")


# --------------------------------------------------------------- coverage ----
# The runtime loader (src/lumina_cuda.cu L6079-6084) reads the PLAIN-named
# ige_col_<Z>_<ion0>.bin for a HARD-CODED list of 7 ions, plus feiii_col_zhang.bin
# for Fe III; LUMINA_MAX_COL_IONS (src/lumina.h L471) caps the generic tables at 8.
BASELINE_IONS = [(26, 1), (27, 2), (28, 2), (16, 2), (14, 1), (16, 1), (14, 2)]


def read_bin_pairs(path):
    with open(path, "rb") as f:
        magic, ver, Z, ion0, ntr, nt, nlev = struct.unpack("<IIiiiii", f.read(28))
        f.read(8 * nt)
        pairs = set()
        for _ in range(ntr):
            a, b = struct.unpack("<ii", f.read(8))
            f.read(8 * nt)
            pairs.add((a, b) if a < b else (b, a))
    return (Z, ion0), pairs, nlev


def coverage(ref_dir):
    """Count line_list.csv bb transitions that get a REAL tabulated Upsilon,
    before (today's 8 loaded tables) and after (everything baked here)."""
    base, new = {}, {}
    fz = os.path.join(ref_dir, "feiii_col_zhang.bin")
    if os.path.exists(fz):
        k, pr, _ = read_bin_pairs(fz); base[k] = pr
    for (Z, ion0) in BASELINE_IONS:
        p = os.path.join(ref_dir, "ige_col_%d_%d.bin" % (Z, ion0))
        if os.path.exists(p):
            k, pr, _ = read_bin_pairs(p); base[k] = pr
    for fn in sorted(os.listdir(ref_dir)):
        if fn.startswith("ige_col_") and fn.endswith("_cmfgen.bin"):
            k, pr, _ = read_bin_pairs(os.path.join(ref_dir, fn))
            new[k] = pr

    n_lines = 0
    per_ion = {}
    with open(os.path.join(ref_dir, "line_list.csv")) as f:
        hdr = f.readline().rstrip("\n").split(",")
        iZ = hdr.index("atomic_number"); ii = hdr.index("ion_number")
        il = hdr.index("level_number_lower"); iu = hdr.index("level_number_upper")
        for ln in f:
            t = ln.split(",")
            Z = int(t[iZ]); ion0 = int(t[ii])
            a = int(t[il]); b = int(t[iu])
            if a > b: a, b = b, a
            n_lines += 1
            key = (Z, ion0)
            d = per_ion.get(key)
            if d is None:
                d = per_ion[key] = [0, 0, 0]      # n_lines, base_hits, new_hits
            d[0] += 1
            if key in base and (a, b) in base[key]: d[1] += 1
            if key in new and (a, b) in new[key]:   d[2] += 1
    return n_lines, per_ion, set(base), set(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="/home/kjhan/BACKUP/Eunha.A1/Claude/"
                    "Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv")
    ap.add_argument("--write", action="store_true",
                    help="write the .bin files and the manifest (default: audit only)")
    ap.add_argument("--manifest", default="coldata_cmfgen_manifest.csv")
    ap.add_argument("--source-manifest", default="",
                    help="atomic_vintage_manifest.csv with exact osc/col paths")
    ap.add_argument("--only", default="", help="comma list of Z:ion0 to restrict to")
    ap.add_argument("--coverage", action="store_true",
                    help="report the real-Upsilon line coverage before/after")
    args = ap.parse_args()

    if args.coverage:
        n_lines, per_ion, base_ions, new_ions = coverage(args.ref_dir)
        tb = sum(v[1] for v in per_ion.values())
        tn = sum(v[2] for v in per_ion.values())
        cb = sum(v[0] for k, v in per_ion.items() if k in base_ions)
        cn = sum(v[0] for k, v in per_ion.items() if k in new_ions)
        print("line_list.csv bb transitions      : %d" % n_lines)
        print("tables loaded today               : %d ions" % len(base_ions))
        print("tables baked here                 : %d ions" % len(new_ions))
        print("real-Upsilon lines  BEFORE        : %d  (%.3f%% of all, %.2f%% of covered-ion lines=%d)"
              % (tb, 100.0 * tb / n_lines, 100.0 * tb / max(cb, 1), cb))
        print("real-Upsilon lines  AFTER         : %d  (%.3f%% of all, %.2f%% of covered-ion lines=%d)"
              % (tn, 100.0 * tn / n_lines, 100.0 * tn / max(cn, 1), cn))
        print("\n%-8s %10s %10s %10s %8s" % ("ion", "lines", "before", "after", "after%"))
        for k in sorted(per_ion, key=lambda x: -per_ion[x][2]):
            nl, b, nn = per_ion[k]
            if nn or b:
                print("%-8s %10d %10d %10d %7.2f%%"
                      % ("%s %s" % (ELEM_SYM.get(k[0], k[0]), ROMAN[k[1]]), nl, b, nn,
                         100.0 * nn / nl))
        return

    ref = load_ref_levels(args.ref_dir)
    sources = load_source_manifest(args.source_manifest) if args.source_manifest else None
    keys = sorted(ref)
    if args.only:
        want = set()
        for tok in args.only.split(","):
            z, i = tok.split(":"); want.add((int(z), int(i)))
        keys = [k for k in keys if k in want]

    print("CMFGEN col_data -> %s  (%s)\n" % (args.ref_dir,
                                             "WRITE" if args.write else "AUDIT ONLY"))
    mans = []
    for (Z, ion0) in keys:
        mans.append(build_ion(Z, ion0, ref[(Z, ion0)], args.ref_dir, args.write,
                              sources=sources))

    ok = [m for m in mans if m["status"] == "OK"]
    print("\n%d/%d ions imported, %d transitions total"
          % (len(ok), len(mans), sum(m["n_mapped"] for m in ok)))
    for m in mans:
        if m["status"] != "OK":
            print("  -- %-8s %s" % (m["ion"], m["status"]))

    if args.write:
        cols = ["ion", "Z", "ion0", "osc", "col", "n_levels_osc", "n_levels_ref",
                "n_trans_source", "n_temp", "n_mapped", "n_dropped", "drop_reasons",
                "omega_min", "omega_max", "omega_median", "max_sumrule_err",
                "cmfgen_quality_note", "out_bin", "status"]
        path = os.path.join(args.ref_dir, args.manifest)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for m in mans:
                w.writerow({c: m.get(c, "") for c in cols})
        print("\nmanifest -> %s" % path)


if __name__ == "__main__":
    main()
