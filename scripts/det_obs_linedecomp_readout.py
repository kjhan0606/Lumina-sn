#!/usr/bin/env python3
"""DET-OBS-LINEDECOMP readout (prereg docs/PREREG_DET_OBS_LINEDECOMP_2026-10-07.md).

Subcommands
  selftest  --c-selftest BIN --exe BIN --replay-bin BIN --fixture DIR --out DIR
  decompose --capture DIR --line-list CSV --deck SHA --prior DIR --exe BIN --threads N --out DIR
  paired    --e DIR --r DIR --prior-paired DIR --out DIR

Exact reference: exact arithmetic over the computed shared values (attenuation
factors, Doppler factors, ray coordinates, the band functional with the shared
floating-point wavelength differences, the L_nu -> L_lambda factor).  Every
reported quantity is an interval [lo, hi] enclosing that reference (outward
rounding of every operation; sums are math.fsum, correctly rounded, then one
step outward) plus a round-to-nearest middle value.  Every command writes only
into a new --out directory.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

CM_C = 2.99792458e10
C8 = 8.0 * math.pi * math.pi                 # = 8.0 * M_PI_VAL * M_PI_VAL in the replay
U = 2.0 ** -53
BANDS = (("FUVb", 1500.0, 2000.0), ("UV2500", 2500.0, 3000.0))
LANE_MAPPED = "mapped_line"
REC_OUT = np.dtype([("l", "<u4"), ("s", "<u4"), ("m", "u1"), ("S", "<f8"), ("om", "<f8"),
                    ("Hlo", "<f8", (2,)), ("Hmid", "<f8", (2,)), ("Hhi", "<f8", (2,))])
REC_IN = np.dtype([("l", "<u4"), ("s", "<u4"), ("tau", "<f8"), ("S", "<f8"), ("eps", "<f8"), ("m", "u1")])
CAPTURE_FILES = ("lines.bin", "chi_a.f64", "chi_e.f64", "eta_c.f64", "eta_dep.f64", "J.f64", "Te.f64",
                 "ne.f64", "nu.f64", "dnu.f64", "line_nu.f64", "r_inner.f64", "r_outer.f64", "meta.txt")
FACTORS = ("F_S", "F_tau", "F_H", "F_plus", "F_minus")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------ interval helpers
def up(x):
    return math.nextafter(x, math.inf)


def dn(x):
    return math.nextafter(x, -math.inf)


def vup(x):
    return np.nextafter(x, np.inf)


def vdn(x):
    return np.nextafter(x, -np.inf)


class Iv:
    """vector interval: lo, mid, hi arrays (or floats)."""
    __slots__ = ("lo", "mid", "hi")

    def __init__(self, lo, mid, hi):
        self.lo, self.mid, self.hi = lo, mid, hi

    @staticmethod
    def exact(x):
        x = np.asarray(x, dtype=np.float64)
        return Iv(x.copy(), x.copy(), x.copy())

    @staticmethod
    def of(x):
        """enclosure of an exact value whose round-to-nearest result is x."""
        x = np.asarray(x, dtype=np.float64)
        return Iv(vdn(x), x, vup(x))

    def __add__(self, o):
        return Iv(vdn(self.lo + o.lo), self.mid + o.mid, vup(self.hi + o.hi))

    def __sub__(self, o):
        return Iv(vdn(self.lo - o.hi), self.mid - o.mid, vup(self.hi - o.lo))

    def __neg__(self):
        return Iv(-self.hi, -self.mid, -self.lo)

    def __mul__(self, o):
        c = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        lo = np.minimum.reduce(c)
        hi = np.maximum.reduce(c)
        return Iv(vdn(lo), self.mid * o.mid, vup(hi))

    def half(self):
        return Iv(vdn(self.lo * 0.5), self.mid * 0.5, vup(self.hi * 0.5))

    def pos(self):
        return Iv(np.maximum(self.lo, 0.0), np.maximum(self.mid, 0.0), np.maximum(self.hi, 0.0))

    def take(self, idx):
        return Iv(self.lo[idx], self.mid[idx], self.hi[idx])


def isum(iv):
    """enclosure of the exact sum of interval elements (scalar Iv of floats)."""
    lo = np.asarray(iv.lo, dtype=np.float64).ravel()
    hi = np.asarray(iv.hi, dtype=np.float64).ravel()
    mid = np.asarray(iv.mid, dtype=np.float64).ravel()
    if lo.size == 0:
        return Iv(0.0, 0.0, 0.0)
    return Iv(dn(math.fsum(lo)), math.fsum(mid), up(math.fsum(hi)))


def group_sums(keys, iv):
    """per unique key (sorted): enclosure of the sum.  keys: 1-d int64 array."""
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    uniq, start = np.unique(k, return_index=True)
    stop = np.append(start[1:], k.size)
    lo, mid, hi = np.asarray(iv.lo)[order], np.asarray(iv.mid)[order], np.asarray(iv.hi)[order]
    out = {}
    for u_, a, b in zip(uniq.tolist(), start.tolist(), stop.tolist()):
        out[u_] = Iv(dn(math.fsum(lo[a:b])), math.fsum(mid[a:b]), up(math.fsum(hi[a:b])))
    return out


def overlap(a, b):
    return bool(a.lo <= b.hi and b.lo <= a.hi)


def iv_json(iv):
    return {"lo": float(iv.lo), "mid": float(iv.mid), "hi": float(iv.hi)}


# ------------------------------------------------------------ band functional weights
def band_weights_exact(lam, lo, hi):
    """band_integral (nt_eta_readout) as a linear functional of the values at the
    ascending wavelength points lam, with the shared fp differences: {i: Fraction}."""
    if lam[0] > lo or lam[-1] < hi:
        return None
    w = {}
    for i in range(len(lam) - 1):
        l0, l1 = lam[i], lam[i + 1]
        if l1 <= l0:
            return None
        a, b = max(l0, lo), min(l1, hi)
        if b <= a:
            continue
        beta = Fraction(l1 - l0)
        ta, tb = Fraction(a - l0) / beta, Fraction(b - l0) / beta
        g = Fraction(b - a) / 2
        w[i] = w.get(i, Fraction(0)) + g * ((1 - ta) + (1 - tb))
        w[i + 1] = w.get(i + 1, Fraction(0)) + g * (ta + tb)
    return w


def bin_weights(nu, lam_points):
    """per band: {k: Fraction} total weight on L_nu(k) (band functional x L_lambda
    factor x 8 pi^2); lam_points = the replay components wavelength column (ascending)."""
    NB = len(nu)
    lam_calc = [(CM_C / float(nu[k])) * 1.0e8 for k in range(NB - 1, -1, -1)]
    if list(lam_points) != lam_calc:
        raise ValueError("WAVELENGTH_GRID_MISMATCH")
    out = []
    for name, lo, hi in BANDS:
        wi = band_weights_exact(lam_points, lo, hi)
        if wi is None:
            raise ValueError(f"BAND_UNSUPPORTED:{name}")
        wk = {}
        for i, w in wi.items():
            k = NB - 1 - i
            lam_cm = CM_C / float(nu[k])
            lam2 = lam_cm * lam_cm
            wk[k] = w * Fraction(C8) * Fraction(CM_C) / Fraction(lam2) * Fraction(1.0e-8)
        out.append(wk)
    return out


def write_weights(path, wb):
    with open(path, "x") as f:
        for b, wk in enumerate(wb):
            for k in sorted(wk):
                x = wk[k]
                if x == 0:
                    continue
                m = float(x)                       # correctly rounded: x within half an ulp of m
                lo, hi = dn(m), up(m)
                f.write(f"{b} {k} {lo.hex()} {m.hex()} {hi.hex()}\n")


# ------------------------------------------------------------ G-JOIN
def join_reasons(line_list, deck_sha, line_nu, key="row"):
    """line id = 0-based data row of line_list.csv; the row nu must equal the
    capture line_nu bit for bit.  key="csv_id" (negative control) joins through
    the CSV line_id column instead."""
    why = []
    want = None
    if deck_sha is not None:
        for ln in open(deck_sha):
            parts = ln.split()
            if len(parts) == 2 and parts[1].lstrip("*") == "line_list.csv":
                want = parts[0]
        if want is None or sha(line_list) != want:
            why.append("JOIN_LINE_LIST_NOT_SEALED")
    df = pd.read_csv(line_list, usecols=["atomic_number", "ion_number", "level_number_lower",
                                         "level_number_upper", "line_id", "nu", "wavelength"],
                     dtype={"atomic_number": np.int64, "ion_number": np.int64}, float_precision="round_trip")
    if len(df) != len(line_nu):
        return why + [f"JOIN_ROW_COUNT:{len(df)}!={len(line_nu)}"], None
    nu = df["nu"].to_numpy(np.float64)
    if key == "csv_id":
        idx = df["line_id"].to_numpy(np.int64)
        if idx.min() < 0 or idx.max() >= len(df):
            return why + ["JOIN_CSV_ID_RANGE"], None
        inv = np.empty_like(idx)
        inv[idx] = np.arange(len(df))
        nu = nu[inv]
        df = df.iloc[inv].reset_index(drop=True)
    if not np.array_equal(nu.view(np.uint64), np.asarray(line_nu, np.float64).view(np.uint64)):
        why.append(f"JOIN_NU_MISMATCH:n={int((nu != line_nu).sum())}")
    return why, df


# ------------------------------------------------------------ state decomposition
def read_meta(capdir):
    m = {}
    for ln in open(Path(capdir) / "meta.txt"):
        if "=" in ln:
            k, v = ln.rstrip("\n").split("=", 1)
            m[k] = v
    return m


def pair_L(rec):
    """per band: enclosure of j*H with j = S*om (S, om computed values)."""
    S, om = rec["S"], rec["om"]
    jm = S * om
    j = Iv(vdn(jm), jm, vup(jm))
    out = []
    for b in range(len(BANDS)):
        H = Iv(rec["Hlo"][:, b], rec["Hmid"][:, b], rec["Hhi"][:, b])
        out.append(j * H)
    return j, out


def classify_levels(groups):
    """groups: {key: Iv} of non-negative luminosity.  §6.1 rules -> (code, branch, lead, shares)."""
    if not groups:
        return 9, "X", None, {}
    tot = Iv(dn(math.fsum(g.lo for g in groups.values())), math.fsum(g.mid for g in groups.values()),
             up(math.fsum(g.hi for g in groups.values())))
    if not tot.lo > 0.0:
        return 2, "UNRESOLVED", None, {}
    sh = {k: {"lo": dn(g.lo / tot.hi), "mid": g.mid / tot.mid, "hi": min(1.0, up(g.hi / tot.lo))}
          for k, g in groups.items()}
    for k, s in sh.items():
        if s["lo"] > 0.5:
            return 0, "MAJORITY", k, sh
    best = max(groups, key=lambda k: groups[k].mid)
    if all(groups[best].lo > groups[j].hi for j in groups if j != best):
        return 1, "MIXED_LARGEST", best, sh
    return 2, "UNRESOLVED", best, sh


def keyname(level, k, NS=None):
    if level == "ion_shell":
        z, ion, s = k // 1000000, (k // 1000) % 1000, k % 1000
        return f"Z{z}_ion{ion}_sh{s}"
    if level == "ion":
        return f"Z{k // 1000}_ion{k % 1000}"
    return f"sh{k}"


def level_keys(z, ion, s):
    z, ion, s = z.astype(np.int64), ion.astype(np.int64), s.astype(np.int64)
    return {"ion_shell": z * 1000000 + ion * 1000 + s, "ion": z * 1000 + ion, "shell": s}


def cmd_decompose(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cap, prior = Path(a.capture), Path(a.prior)
    why = []
    meta = read_meta(cap)
    NS, NB, NL = int(meta["NS"]), int(meta["NB"]), int(meta["NL"])
    nu = np.fromfile(cap / "nu.f64", "<f8")
    line_nu = np.fromfile(cap / "line_nu.f64", "<f8")
    jw, df = join_reasons(a.line_list, a.deck, line_nu)
    why += [f"G-JOIN:{x}" for x in jw]
    # prior identity link: the prior replay read this capture's committed manifest
    prior_res = json.load(open(prior / "result.json"))
    if prior_res.get("validity") != "VALID":
        why.append("PRIOR_NOT_VALID")
    lam_pts = [float(r[0]) for r in list(csv.reader(open(prior / "replay/components.csv")))[1:]]
    lam_pts.sort()
    wb = bin_weights(nu, lam_pts)
    write_weights(out / "weights.txt", wb)
    if why:
        (out / "result.json").write_text(json.dumps({"validity": "X", "reasons": why}, indent=1))
        print(json.dumps({"validity": "X", "reasons": why}))
        return 0
    cmd = [str(a.exe), "backward", str(cap), str(out / "weights.txt"), str(out / "pairs.bin"), str(a.threads)]
    rr = subprocess.run(cmd, capture_output=True, text=True)
    (out / "exe.stdout").write_text(rr.stdout)
    (out / "exe.stderr").write_text(rr.stderr)
    if rr.returncode != 0:
        why.append(f"G-DECOMP:EXE_RC:{rr.returncode}")
        (out / "result.json").write_text(json.dumps({"validity": "X", "reasons": why}, indent=1))
        print(json.dumps({"validity": "X", "reasons": why}))
        return 0
    rec = np.fromfile(out / "pairs.bin", dtype=REC_OUT)
    j, L = pair_L(rec)
    z = df["atomic_number"].to_numpy()[rec["l"]]
    ion = df["ion_number"].to_numpy()[rec["l"]]
    keys = level_keys(z, ion, rec["s"])
    r_in = np.fromfile(cap / "r_inner.f64", "<f8")
    r_out = np.fromfile(cap / "r_outer.f64", "<f8")
    res = {"schema": "DET_OBS_LINEDECOMP_STATE_V1", "readout_sha256": sha(Path(__file__).resolve()),
           "capture": str(cap), "pairs_written": int(rec.size), "bands": {}}
    closure = {}
    for b, (bname, lo, hi) in enumerate(BANDS):
        tot = isum(L[b])
        pm = prior_res["bands"][bname]["lanes"][LANE_MAPPED]
        pr = Iv(dn(pm["L"] - pm["err"]), pm["L"], up(pm["L"] + pm["err"]))
        # the prior injects fl(D*D*D*S): |fl - D^3 S| <= ((1+u)^3 - 1) D^3 S < 4u D^3 S
        tw = Iv(dn(tot.lo * (1.0 - 4.0 * U)), tot.mid, up(tot.hi * (1.0 + 4.0 * U)))
        ok1 = overlap(tw, pr)
        lev = {}
        ok2 = True
        for level in ("ion_shell", "ion", "shell"):
            g = group_sums(keys[level], L[b])
            s_lo = dn(math.fsum(v.lo for v in g.values()))
            s_hi = up(math.fsum(v.hi for v in g.values()))
            ok2 = ok2 and s_lo <= tot.hi and tot.lo <= s_hi
            code, br, lead, sh = classify_levels(g)
            lev[level] = {"code": code, "branch": br, "lead": None if lead is None else keyname(level, lead),
                          "groups": {keyname(level, k): {"L": iv_json(v), "share": sh.get(k)} for k, v in g.items()}}
        closure[bname] = {"sum_pairs": iv_json(tot), "prior_mapped": iv_json(pr), "closure_prior_ok": ok1,
                          "levels_consistent": ok2,
                          "rel_mid_vs_prior": (tot.mid - pm["L"]) / pm["L"] if pm["L"] else None}
        if not ok1:
            why.append(f"G-DECOMP:CLOSURE_PRIOR:{bname}")
        if not ok2:
            why.append(f"G-DECOMP:CLOSURE_LEVELS:{bname}")
        res["bands"][bname] = {"L_mapped": iv_json(tot), "levels": lev}
    # tables
    for level, fname in (("ion_shell", "ion_shell.csv"), ("ion", "ions.csv"), ("shell", "shells.csv")):
        with open(out / fname, "x", newline="") as f:
            w = csv.writer(f)
            head = ["group"] + (["r_inner_cm", "r_outer_cm"] if level == "shell" else [])
            for bname, _, _ in BANDS:
                head += [f"{bname}_L_lo", f"{bname}_L_mid", f"{bname}_L_hi", f"{bname}_share_lo", f"{bname}_share_hi"]
            w.writerow(head)
            names = sorted({g for bname, _, _ in BANDS for g in res["bands"][bname]["levels"][level]["groups"]})
            for gname in names:
                row = [gname]
                if level == "shell":
                    s = int(gname[2:]); row += [repr(float(r_in[s])), repr(float(r_out[s]))]
                for bname, _, _ in BANDS:
                    g = res["bands"][bname]["levels"][level]["groups"].get(gname)
                    if g is None:
                        row += ["0", "0", "0", "0", "0"]
                    else:
                        row += [repr(g["L"]["lo"]), repr(g["L"]["mid"]), repr(g["L"]["hi"]),
                                repr(g["share"]["lo"]) if g["share"] else "", repr(g["share"]["hi"]) if g["share"] else ""]
                w.writerow(row)
    res["validity"] = "VALID" if not why else "X"
    res["reasons"] = why
    (out / "closure.json").write_text(json.dumps(closure, indent=1))
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    man = {"schema": "DET_OBS_LINEDECOMP_MANIFEST_V1", "command": sys.argv,
           "inputs": {"capture_manifest_sha256": sha(cap / "manifest.json") if (cap / "manifest.json").exists() else None,
                      "line_list_sha256": sha(a.line_list), "prior_result_sha256": sha(prior / "result.json"),
                      "prior_components_sha256": sha(prior / "replay/components.csv")},
           "exe": {"path": str(a.exe), "sha256": sha(a.exe), "command": cmd, "stdout": rr.stdout.strip()},
           "readout_sha256": sha(Path(__file__).resolve()),
           "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"},
           "pairs_schema": "u32 line, u32 shell, u8 mapped, f64 S_used, f64 om, f64 H_lo[2], H_mid[2], H_hi[2] (bands FUVb, UV2500)"}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"validity": res["validity"], "reasons": why,
                      "closure": {k: {"ok": v["closure_prior_ok"], "rel": v["rel_mid_vs_prior"]} for k, v in closure.items()}}))
    return 0


# ------------------------------------------------------------ paired factor decomposition
def state_arrays(sdir):
    """pairs.bin of one state + the mapped consumed key set of its capture."""
    res = json.load(open(Path(sdir) / "result.json"))
    cap = Path(res["capture"])
    meta = read_meta(cap)
    NS = int(meta["NS"])
    rec = np.fromfile(Path(sdir) / "pairs.bin", dtype=REC_OUT)
    lines = np.fromfile(cap / "lines.bin", dtype=REC_IN)
    return res, NS, rec, lines


def factor_terms(SE, fE, HE, SR, fR, HR):
    """common pairs: exact symmetric split.  S, f are exact computed values;
    H are intervals.  Returns dict of Iv."""
    SEi, SRi, fEi, fRi = Iv.exact(SE), Iv.exact(SR), Iv.exact(fE), Iv.exact(fR)
    Hbar = (HE + HR).half()
    dH = HE - HR
    jE, jR = SEi * fEi, SRi * fRi
    jbar = (jE + jR).half()
    FS = (SEi - SRi) * (fEi + fRi).half() * Hbar
    Ft = (SEi + SRi).half() * (fEi - fRi) * Hbar
    FH = jbar * dH
    return {"F_S": FS, "F_tau": Ft, "F_H": FH, "LE": jE * HE, "LR": jR * HR}


def build_paired(E, R, band):
    """per pair in the union of reached mapped pairs: L_E, L_R and the five factors
    (Iv arrays), the pair keys (line, shell) and the class codes."""
    resE, NS, recE, linesE = E
    resR, NSR, recR, linesR = R
    if NS != NSR:
        raise ValueError("INPUT_ALIGNMENT_FAILED:NS")
    def keys(r):
        return r["l"].astype(np.int64) * NS + r["s"].astype(np.int64)
    kE, kR = keys(recE), keys(recR)
    kLE, kLR = keys(linesE), keys(linesR)
    U_ = np.union1d(kE, kR)
    n = U_.size
    def lookup(kset, kq):
        pos = np.searchsorted(kset, kq)
        pos = np.minimum(pos, max(kset.size - 1, 0))
        return pos, (kset.size > 0) & (kset[pos] == kq)
    sE = np.argsort(kE); kEs = kE[sE]
    sR = np.argsort(kR); kRs = kR[sR]
    pE, inE = lookup(kEs, U_)
    pR, inR = lookup(kRs, U_)
    # consumed-set status from the captures (lines.bin is sorted by (line, shell))
    pLE, cE = lookup(kLE, U_)
    pLR, cR = lookup(kLR, U_)
    mE = cE & (linesE["m"][pLE] == 1)
    mR = cR & (linesR["m"][pLR] == 1)
    def take(rec, srt, pos, present, lines, plines, field):
        v = np.zeros(n)
        v[present] = rec[field][srt[pos[present]]]
        return v
    SE = np.where(inE, take(recE, sE, pE, inE, None, None, "S"), linesE["S"][pLE] if linesE.size else 0.0)
    fE = np.where(inE, take(recE, sE, pE, inE, None, None, "om"), -np.expm1(-linesE["tau"][pLE]) if linesE.size else 0.0)
    SR = np.where(inR, take(recR, sR, pR, inR, None, None, "S"), linesR["S"][pLR] if linesR.size else 0.0)
    fR = np.where(inR, take(recR, sR, pR, inR, None, None, "om"), -np.expm1(-linesR["tau"][pLR]) if linesR.size else 0.0)
    def Hiv(rec, srt, pos, present):
        lo, mid, hi = np.zeros(n), np.zeros(n), np.zeros(n)
        ix = srt[pos[present]]
        lo[present], mid[present], hi[present] = rec["Hlo"][ix, band], rec["Hmid"][ix, band], rec["Hhi"][ix, band]
        return Iv(lo, mid, hi)
    HE, HR = Hiv(recE, sE, pE, inE), Hiv(recR, sR, pR, inR)
    if np.any(inE & ~mE) or np.any(inR & ~mR):
        raise ValueError("INPUT_ALIGNMENT_FAILED:REACHED_PAIR_NOT_MAPPED")
    common, plus, minus = mE & mR, mE & ~mR, mR & ~mE
    ft = factor_terms(SE, fE, HE, SR, fR, HR)
    zero = Iv(np.zeros(n), np.zeros(n), np.zeros(n))
    def mask(iv, m):
        return Iv(np.where(m, iv.lo, 0.0), np.where(m, iv.mid, 0.0), np.where(m, iv.hi, 0.0))
    LE = mask(ft["LE"], mE)
    LR = mask(ft["LR"], mR)
    F = {"F_S": mask(ft["F_S"], common), "F_tau": mask(ft["F_tau"], common), "F_H": mask(ft["F_H"], common),
         "F_plus": mask(ft["LE"], plus), "F_minus": mask(-ft["LR"], minus)}
    cls = {"common": common, "plus": plus, "minus": minus,
           "plus_reclassified": plus & cR & ~mR, "minus_reclassified": minus & cE & ~mE}
    return U_ // NS, U_ % NS, LE, LR, F, cls, zero


def diff_branch(dL):
    """§6.2 on group differences: sign of the total, P/M, lead of the dominant side."""
    tot = Iv(dn(math.fsum(v.lo for v in dL.values())), math.fsum(v.mid for v in dL.values()),
             up(math.fsum(v.hi for v in dL.values())))
    P = Iv(dn(math.fsum(max(v.lo, 0.0) for v in dL.values())), math.fsum(max(v.mid, 0.0) for v in dL.values()),
           up(math.fsum(max(v.hi, 0.0) for v in dL.values())))
    M = Iv(dn(math.fsum(max(-v.hi, 0.0) for v in dL.values())), math.fsum(max(-v.mid, 0.0) for v in dL.values()),
           up(math.fsum(max(-v.lo, 0.0) for v in dL.values())))
    if tot.lo > 0.0:
        side = {k: Iv(max(v.lo, 0.0), max(v.mid, 0.0), max(v.hi, 0.0)) for k, v in dL.items()}
        sign = "POSITIVE"
    elif tot.hi < 0.0:
        side = {k: Iv(max(-v.hi, 0.0), max(-v.mid, 0.0), max(-v.lo, 0.0)) for k, v in dL.items()}
        sign = "NEGATIVE"
    else:
        return {"total": iv_json(tot), "P": iv_json(P), "M": iv_json(M), "sign": "UNDETERMINED",
                "code": 2, "branch": "UNRESOLVED", "lead": None}
    code, br, lead, sh = classify_levels({k: v for k, v in side.items() if v.hi > 0.0})
    return {"total": iv_json(tot), "P": iv_json(P), "M": iv_json(M), "sign": sign, "code": code,
            "branch": br, "lead": lead, "shares": sh}


def factor_branch(fac):
    """§6.3 on the five factor sums of one group (or band): positive / negative parts and
    the dominant factor of the side matching the sign of their total."""
    tot = Iv(dn(math.fsum(v.lo for v in fac.values())), math.fsum(v.mid for v in fac.values()),
             up(math.fsum(v.hi for v in fac.values())))
    pos = {k: Iv(max(v.lo, 0.0), max(v.mid, 0.0), max(v.hi, 0.0)) for k, v in fac.items()}
    neg = {k: Iv(max(-v.hi, 0.0), max(-v.mid, 0.0), max(-v.lo, 0.0)) for k, v in fac.items()}
    P = math.fsum(v.mid for v in pos.values())
    M = math.fsum(v.mid for v in neg.values())
    side = pos if tot.lo > 0.0 else neg if tot.hi < 0.0 else None
    if side is None:
        return {"code": 2, "branch": "UNRESOLVED", "lead": None, "P_mid": P, "M_mid": M}
    code, br, lead, sh = classify_levels({k: v for k, v in side.items() if v.hi > 0.0})
    return {"code": code, "branch": br, "lead": lead, "P_mid": P, "M_mid": M,
            "opposite_over_same": (M / P if tot.lo > 0.0 else P / M) if (P and M) else 0.0, "shares": sh}


def cmd_paired(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    E, R = state_arrays(a.e), state_arrays(a.r)
    why = []
    if E[0].get("validity") != "VALID" or R[0].get("validity") != "VALID":
        why.append("INPUT_NOT_VALID")
    df = None
    for d in (a.e, a.r):
        man = json.load(open(Path(d) / "manifest.json"))
        cmdv = man["command"]
        ll = cmdv[cmdv.index("--line-list") + 1]
        if df is None:
            _, df = join_reasons(ll, None, np.fromfile(Path(E[0]["capture"]) / "line_nu.f64", "<f8"))
            ll0 = sha(ll)
        elif sha(ll) != ll0:
            why.append("INPUT_ALIGNMENT_FAILED:LINE_LIST_DIFFERS")
    for f in ("nu.f64", "line_nu.f64", "r_inner.f64", "r_outer.f64"):
        if sha(Path(E[0]["capture"]) / f) != sha(Path(R[0]["capture"]) / f):
            why.append(f"INPUT_ALIGNMENT_FAILED:{f}")
    me, mr = read_meta(E[0]["capture"]), read_meta(R[0]["capture"])
    for k in ("NS", "NB", "NL", "nu_min", "nu_max", "d_log_nu", "inv_ct", "time_explosion"):
        if me.get(k) != mr.get(k):
            why.append(f"INPUT_ALIGNMENT_FAILED:meta:{k}")
    prior = json.load(open(Path(a.prior_paired) / "result.json"))
    res = {"schema": "DET_OBS_LINEDECOMP_PAIRED_V1", "readout_sha256": sha(Path(__file__).resolve()),
           "inputs": {"E": str(a.e), "R": str(a.r), "prior_paired": str(a.prior_paired)}, "bands": {}}
    tables = {}
    if not why:
        for b, (bname, _, _) in enumerate(BANDS):
            line, shell, LE, LR, F, cls, _ = build_paired(E, R, b)
            z = df["atomic_number"].to_numpy()[line]
            ion = df["ion_number"].to_numpy()[line]
            keys = level_keys(z, ion, shell)
            band = {"pair_counts": {k: int(v.sum()) for k, v in cls.items()}, "levels": {}}
            # closure 4: total difference vs the prior mapped E'-R' difference
            sLE, sLR = isum(LE), isum(LR)
            dtot = sLE - sLR
            pd_ = prior["bands"][bname]["difference"]["D"][LANE_MAPPED]
            prv = Iv(dn(pd_["D"] - pd_["err"]), pd_["D"], up(pd_["D"] + pd_["err"]))
            wid = up(4.0 * U * up(abs(sLE.hi) + abs(sLR.hi)))
            dtw = Iv(dn(dtot.lo - wid), dtot.mid, up(dtot.hi + wid))
            c4 = overlap(dtw, prv)
            band["closure_prior_difference"] = {"sum_groups": iv_json(dtot), "prior": iv_json(prv), "ok": c4}
            if not c4:
                why.append(f"G-DECOMP:CLOSURE_PRIOR_DIFF:{bname}")
            for level in ("ion_shell", "ion", "shell"):
                gE, gR = group_sums(keys[level], LE), group_sums(keys[level], LR)
                gF = {f: group_sums(keys[level], F[f]) for f in FACTORS}
                dL, rows, c3 = {}, {}, True
                for k in gE:
                    d = gE[k] - gR[k]
                    fs = Iv(0.0, 0.0, 0.0)
                    for f in FACTORS:
                        fs = Iv(dn(fs.lo + gF[f][k].lo), fs.mid + gF[f][k].mid, up(fs.hi + gF[f][k].hi))
                    ok = overlap(d, fs)
                    c3 = c3 and ok
                    dL[k] = d
                    rows[k] = {"L_E": gE[k], "L_R": gR[k], "dL": d, "factors": {f: gF[f][k] for f in FACTORS},
                               "closure_ok": ok}
                if not c3:
                    why.append(f"G-DECOMP:CLOSURE_FACTORS:{bname}:{level}")
                br = diff_branch(dL)
                lead = br.get("lead")
                band_fac = {f: isum(F[f]) for f in FACTORS}
                lv = {"difference": {**{k: v for k, v in br.items() if k != "shares"},
                                     "lead": None if lead is None else keyname(level, lead)},
                      "factors_band": {f: iv_json(v) for f, v in band_fac.items()},
                      "factor_branch_band": factor_branch(band_fac)}
                eb = classify_levels(gE)
                lv["E_output"] = {"code": eb[0], "branch": eb[1], "lead": None if eb[2] is None else keyname(level, eb[2])}
                if lead is not None:
                    lv["factor_branch_lead"] = factor_branch(rows[lead]["factors"])
                band["levels"][level] = lv
                tables[(bname, level)] = rows
            res["bands"][bname] = band
    res["validity"] = "VALID" if not why else "X"
    res["reasons"] = why
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    for level, fname in (("ion_shell", "ion_shell.csv"), ("ion", "ions.csv"), ("shell", "shells.csv")):
        if not tables:
            break
        with open(out / fname, "x", newline="") as f:
            w = csv.writer(f)
            head = ["group"]
            for bname, _, _ in BANDS:
                for q in ("L_E", "L_R", "dL", *FACTORS):
                    head += [f"{bname}_{q}_lo", f"{bname}_{q}_mid", f"{bname}_{q}_hi"]
                head.append(f"{bname}_closure_ok")
            w.writerow(head)
            names = sorted({k for bname, _, _ in BANDS for k in tables[(bname, level)]})
            for k in names:
                row = [keyname(level, k)]
                for bname, _, _ in BANDS:
                    r = tables[(bname, level)].get(k)
                    for q in ("L_E", "L_R", "dL", *FACTORS):
                        v = None if r is None else (r["factors"][q] if q in FACTORS else r[q])
                        row += ["0", "0", "0"] if v is None else [repr(float(v.lo)), repr(float(v.mid)), repr(float(v.hi))]
                    row.append("" if r is None else int(r["closure_ok"]))
                w.writerow(row)
    man = {"schema": "DET_OBS_LINEDECOMP_MANIFEST_V1", "command": sys.argv,
           "inputs": {"E_manifest": sha(Path(a.e) / "manifest.json"), "R_manifest": sha(Path(a.r) / "manifest.json"),
                      "prior_paired_result": sha(Path(a.prior_paired) / "result.json")},
           "readout_sha256": sha(Path(__file__).resolve()),
           "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    summ = {bn: {lv: {"diff": v["levels"][lv]["difference"]["branch"], "lead": v["levels"][lv]["difference"]["lead"],
                      "factor_lead": v["levels"][lv].get("factor_branch_lead", {}).get("lead")}
                 for lv in v["levels"]} for bn, v in res["bands"].items()}
    print(json.dumps({"validity": res["validity"], "reasons": why, "summary": summ}))
    return 0


# ------------------------------------------------------------ selftest
REQUIRED_CASES = (
    "C direct tests (forward single-source reference, faults 1-2) pass",
    "G-JOIN: row order join passes on the real line list layout",
    "G-JOIN: joining through the CSV line_id column is rejected",
    "G-JOIN: swapped rows are rejected",
    "G-JOIN: unsealed line list is rejected",
    "fixture: sum of groups encloses the replay mapped lane (FUV-like band of the fixture)",
    "same-frequency lines of different ions stay separate groups and close",
    "factors: symmetric split closes per pair",
    "factors: dropping the cross term breaks the closure",
    "factors: flipping the exit sign breaks the closure",
    "factors: same state gives zero differences",
    "factors: swapping E and R flips every difference",
    "branch: interval widths decide (a narrow-mid majority with wide intervals is unresolved)",
    "branch: majority / mixed largest",
)


def seal_capture(d):
    files = {f: {"bytes": (d / f).stat().st_size, "sha256": sha(d / f)} for f in CAPTURE_FILES}
    (d / "manifest.json").write_text(json.dumps({"schema": "DET_OBS_CAPTURE_V1", "transaction": "COMMITTED",
                                                 "fixture": 1, "files": files}, separators=(",", ":")))


def derive_fixture(src, dst, nu1_scale):
    """small capture derived from the C-battery fixture: line 0 becomes mapped in even
    shells (eps 0), line 1 moves to line 0's frequency x nu1_scale (1.0 = equal).
    Sealed with a COMMITTED manifest so the T2 replay accepts it."""
    dst.mkdir(parents=True)
    for f in CAPTURE_FILES:
        if f not in ("lines.bin", "line_nu.f64", "meta.txt"):
            shutil.copy(src / f, dst / f)
    ln = np.fromfile(src / "line_nu.f64", "<f8").copy()
    ln[1] = ln[0] * nu1_scale if nu1_scale != 1.0 else ln[0]
    ln.tofile(dst / "line_nu.f64")
    r = np.fromfile(src / "lines.bin", dtype=REC_IN).copy()
    sel = (r["l"] == 0) & (r["s"] % 2 == 0)
    r["m"][sel] = 1
    r["eps"][sel] = 0.0
    r.tofile(dst / "lines.bin")
    nm = int(r["m"].sum())
    with open(dst / "meta.txt", "w") as f:
        for line in open(src / "meta.txt"):
            if line.startswith("n_mapped="):
                line = f"n_mapped={nm}\n"
            elif line.startswith("n_unmapped="):
                line = f"n_unmapped={r.size - nm}\n"
            f.write(line)
    seal_capture(dst)
    return ln


def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = []

    def case(name, got, want):
        ok = bool(got == want)
        res.append({"case": name, "pass": ok, "got": str(got)[:300], "want": str(want)[:300]})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {str(got)[:200]} want {str(want)[:200]}")

    fx = Path(a.fixture)
    tmp = out / "fx"; tmp.mkdir()
    # C direct tests on a fixture where an unmapped line resonates just downstream of a
    # mapped injection (line 1 at 0.998 x line 0's frequency; line 0 mapped in even shells)
    near = tmp / "near/det_obs_capture"
    derive_fixture(fx, near, 0.998)
    rc = subprocess.run([str(a.c_selftest), str(near), str(out / "c_selftest")], capture_output=True, text=True)
    (out / "c_selftest.log").write_text(rc.stdout + rc.stderr)
    case("C direct tests (forward single-source reference, faults 1-2) pass", rc.returncode, 0)
    # G-JOIN on synthetic line lists with the real column layout
    line_nu = np.fromfile(fx / "line_nu.f64", "<f8")
    NLf = line_nu.size
    head = "atomic_number,ion_number,level_number_lower,level_number_upper,line_id,wavelength,f_ul,f_lu,nu,B_lu,B_ul,A_ul,wavelength_cm\n"
    def write_ll(path, rows):
        with open(path, "w") as f:
            f.write(head)
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
        p2 = path.with_suffix(".sha256")
        p2.write_text(f"{sha(path)}  line_list.csv\n")
        return path, p2
    rows = [[16 - 2 * i, 4 - i, 0, i + 1, NLf - 1 - i, 0, 0, 0, repr(float(line_nu[i])), 0, 0, 0, 0] for i in range(NLf)]
    ll, dk = write_ll(tmp / "ll_ok.csv", rows)
    w0, df0 = join_reasons(ll, dk, line_nu)
    case("G-JOIN: row order join passes on the real line list layout", w0, [])
    w1, _ = join_reasons(ll, dk, line_nu, key="csv_id")
    case("G-JOIN: joining through the CSV line_id column is rejected", any(x.startswith("JOIN_NU_MISMATCH") for x in w1), True)
    sw = [rows[1], rows[0]] + rows[2:]
    ll2, dk2 = write_ll(tmp / "ll_swap.csv", sw)
    w2, _ = join_reasons(ll2, dk2, line_nu)
    case("G-JOIN: swapped rows are rejected", any(x.startswith("JOIN_NU_MISMATCH") for x in w2), True)
    (tmp / "bad.sha256").write_text("0" * 64 + "  line_list.csv\n")
    w3, _ = join_reasons(ll, tmp / "bad.sha256", line_nu)
    case("G-JOIN: unsealed line list is rejected", "JOIN_LINE_LIST_NOT_SEALED" in w3, True)
    # fixture closure against the T2 replay mapped lane (fixture band = the lines' region)
    global BANDS
    saved = BANDS
    BANDS = (("FX_A", 450.0, 560.0), ("FX_B", 560.0, 700.0))
    try:
        def run_state(capdir, comps, ll_path, dk_path, tag):
            pr = tmp / f"prior_{tag}"
            (pr / "replay").mkdir(parents=True)
            shutil.copy(comps, pr / "replay/components.csv")
            data = sorted([[float(x) for x in r[:-1]] for r in list(csv.reader(open(comps)))[1:]])
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from det_obs_capture_readout import component_bands
            lanes = component_bands([r + [1] for r in data], {n: (lo, hi) for n, lo, hi in BANDS})
            (pr / "result.json").write_text(json.dumps({"validity": "VALID", "bands": lanes}, default=str))
            ns = argparse.Namespace(capture=str(capdir), line_list=str(ll_path), deck=str(dk_path), prior=str(pr),
                                    exe=a.exe, threads=4, out=str(tmp / f"dec_{tag}"))
            cmd_decompose(ns)
            return json.load(open(tmp / f"dec_{tag}/result.json"))
        r1 = run_state(fx, a.fixture_components, ll, dk, "fx")
        case("fixture: sum of groups encloses the replay mapped lane (FUV-like band of the fixture)",
             (r1.get("validity"), r1.get("reasons")), ("VALID", []))
        # same-frequency lines of different ions: line 1 moved onto line 0's frequency
        eq = tmp / "eqnu/det_obs_capture"
        ln2 = derive_fixture(fx, eq, 1.0)
        rr = subprocess.run([str(a.replay_bin), str(eq), str(tmp / "eqnu_replay"), "4"], capture_output=True, text=True)
        rows_eq = [r[:] for r in rows]; rows_eq[1][8] = repr(float(ln2[1]))
        ll3, dk3 = write_ll(tmp / "ll_eq.csv", rows_eq)
        r2 = run_state(eq, tmp / "eqnu_replay/components.csv", ll3, dk3, "eq") if rr.returncode == 0 else {"validity": f"REPLAY_RC_{rr.returncode}"}
        ions = sorted(r2.get("bands", {}).get("FX_A", {}).get("levels", {}).get("ion", {}).get("groups", {}).keys()) \
            if r2.get("validity") == "VALID" else []
        case("same-frequency lines of different ions stay separate groups and close",
             (r2.get("validity"), len(ions) >= 2), ("VALID", True))
    finally:
        BANDS = saved
    # factor algebra on synthetic pairs
    rng = np.random.default_rng(7)
    n = 200
    SE, SR = rng.uniform(1, 3, n), rng.uniform(1, 3, n)
    fE, fR = rng.uniform(0.1, 1, n), rng.uniform(0.1, 1, n)
    HEm, HRm = rng.uniform(0.5, 2, n), rng.uniform(0.5, 2, n)
    HE, HR = Iv.of(HEm), Iv.of(HRm)
    ft = factor_terms(SE, fE, HE, SR, fR, HR)
    d = isum(ft["LE"]) - isum(ft["LR"])
    fs = isum(ft["F_S"]) + isum(ft["F_tau"]) + isum(ft["F_H"])
    case("factors: symmetric split closes per pair", overlap(d, fs), True)
    jE, jR = Iv.exact(SE) * Iv.exact(fE), Iv.exact(SR) * Iv.exact(fR)
    bad = isum((jE - jR) * HR) + isum(jR * (HE - HR))          # cross term (dj dH) dropped
    case("factors: dropping the cross term breaks the closure", overlap(d, bad), False)
    # exit sign flip on a set term
    plus_L = isum(ft["LE"].take(slice(0, 10)))
    d2 = isum(ft["LE"]) - isum(ft["LR"].take(slice(10, n)))   # pairs 0..9 only in E
    good2 = isum(ft["F_S"].take(slice(10, n))) + isum(ft["F_tau"].take(slice(10, n))) + isum(ft["F_H"].take(slice(10, n))) + plus_L
    flip = isum(ft["F_S"].take(slice(10, n))) + isum(ft["F_tau"].take(slice(10, n))) + isum(ft["F_H"].take(slice(10, n))) - plus_L
    case("factors: flipping the exit sign breaks the closure", (bool(overlap(d2, good2)), bool(overlap(d2, flip))), (True, False))
    same = factor_terms(SE, fE, HE, SE, fE, HE)
    z = [isum(same[k]) for k in ("F_S", "F_tau", "F_H")]
    case("factors: same state gives zero differences",
         all(v.mid == 0.0 and v.lo <= 0.0 <= v.hi for v in z), True)
    sw_ = factor_terms(SR, fR, HR, SE, fE, HE)
    case("factors: swapping E and R flips every difference",
         all(np.allclose(sw_[k].mid, -ft[k].mid, rtol=0, atol=1e-12) for k in ("F_S", "F_tau", "F_H")), True)
    wide = {1: Iv(0.52, 0.6, 0.8), 2: Iv(0.2, 0.4, 0.6)}
    case("branch: interval widths decide (a narrow-mid majority with wide intervals is unresolved)",
         classify_levels(wide)[1], "UNRESOLVED")
    case("branch: majority / mixed largest",
         (classify_levels({1: Iv(0.7, 0.7, 0.7), 2: Iv(0.3, 0.3, 0.3)})[1],
          classify_levels({1: Iv(0.4, 0.4, 0.4), 2: Iv(0.3, 0.3, 0.3), 3: Iv(0.3, 0.3, 0.3)})[1]),
         ("MAJORITY", "MIXED_LARGEST"))
    names = {r["case"] for r in res}
    case("battery covers every required case", sorted(set(REQUIRED_CASES) - names), [])
    nfail = sum(1 for r in res if not r["pass"])
    (out / "readout_selftest.json").write_text(json.dumps(
        {"readout_sha256": sha(Path(__file__).resolve()), "n_cases": len(res), "failures": nfail, "cases": res}, indent=1))
    print(f"DET-OBS-LINEDECOMP readout selftest: {len(res) - nfail}/{len(res)} -> {'PASS' if nfail == 0 else 'FAIL'}")
    return 0 if nfail == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("selftest")
    for k in ("--c-selftest", "--exe", "--replay-bin", "--fixture", "--fixture-components", "--out"):
        p.add_argument(k, required=True)
    p = sp.add_parser("decompose")
    for k in ("--capture", "--line-list", "--deck", "--prior", "--exe", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--threads", default="16")
    p = sp.add_parser("paired")
    for k in ("--e", "--r", "--prior-paired", "--out"):
        p.add_argument(k, required=True)
    a = ap.parse_args()
    return {"selftest": cmd_selftest, "decompose": cmd_decompose, "paired": cmd_paired}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
