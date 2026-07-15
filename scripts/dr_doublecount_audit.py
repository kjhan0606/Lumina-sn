#!/usr/bin/env python3
"""DR double-counting audit for LUMINA-SN frozen-in / tdep recombination.

QUESTION
--------
frozenin_alpha_rr (src/lumina_plasma.c:2708) computes the recombination
coefficient alpha_RR(X+1 -> X) as a Milne integral of per-level
photoionization cross sections sigma_bf (from cmfgen_sigma_bf.bin) against
the Planck function B(T_e). Some of those sigma_bf were built from
RESONANCE-CARRYING CMFGEN phot files (Opacity-Project / Nahar data smoothed
at 3000 km/s). A Milne integral over a resonance-carrying sigma ALREADY
includes the resonant (dielectronic-like) recombination. If the analytic
Badnell/NORAD DR table is then ADDED (gate LUMINA_FROZENIN_DR=1), the
resonant channel is DOUBLE-COUNTED.

This script quantifies, for the key pairs, what fraction of the tabulated
alpha_DR is already inside our Milne alpha:

    alpha_full   = Milne over the ACTUAL bin sigma (resonances included)
    alpha_smooth = Milne over a resonance-suppressed background sigma
    d_alpha_res  = alpha_full - alpha_smooth   (resonant recomb in the bin)
    f_double     = d_alpha_res / alpha_DR(table)

Plus a builder-fidelity cross-check on the NATIVE phot files.

ANALYSIS ONLY. Reads data + reproduces the C prescription in Python.
"""
from __future__ import annotations
import struct, csv, math, sys, os
from pathlib import Path
import numpy as np
from scipy.ndimage import grey_opening

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
sys.path.insert(0, str(ROOT / "scripts"))
from cmfgen_parser import parse_osc, parse_phot  # noqa: E402

# ---- physical constants (CGS), identical to src/lumina_plasma.c ----
H   = 6.62607015e-27
KB  = 1.380649e-16
C   = 2.99792458e10
ME  = 9.1093837015e-28
EV  = 1.602176634e-12
PI  = math.pi
KB_EV = 8.617333262e-5

REF = ROOT / "data" / "tardis_reference_toy06_19p48d"
BIN = REF / "cmfgen_sigma_bf.bin"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

T_LIST = [10470, 11811, 13120, 16014]

# ---------------------------------------------------------------------------
# Load the sigma_bf binary (same reader as scripts/db_photoion_calc.py)
# ---------------------------------------------------------------------------
with open(BIN, "rb") as f:
    magic, ver, NLEV, NFREQ = struct.unpack("<IIii", f.read(16))
    numin, numax = struct.unpack("<dd", f.read(16))
    flags = np.frombuffer(f.read(NLEV), dtype=np.int8)
pad = (8 - (NLEV % 8)) % 8
SIG = np.memmap(BIN, dtype="<f8", mode="r", offset=32 + NLEV + pad,
                shape=(NLEV, NFREQ))
log_numin = math.log(numin)
dlog = (math.log(numax) - log_numin) / NFREQ
nu_c = np.exp(log_numin + (np.arange(NFREQ) + 0.5) * dlog)
nu_lo = np.exp(log_numin + np.arange(NFREQ) * dlog)
nu_hi = np.exp(log_numin + (np.arange(NFREQ) + 1) * dlog)
dnu = nu_hi - nu_lo

# ---- levels + ionization energies ----
levZ = []; levI = []; levN = []; levE = []; levG = []
for r in csv.DictReader(open(REF / "levels.csv")):
    levZ.append(int(r["atomic_number"])); levI.append(int(r["ion_number"]))
    levN.append(int(r["level_number"]));  levE.append(float(r["energy_eV"]))
    levG.append(int(r["g"]))
levZ = np.array(levZ); levI = np.array(levI); levN = np.array(levN)
levE = np.array(levE); levG = np.array(levG)
assert len(levZ) == NLEV

CHI = {}
for r in csv.DictReader(open(REF / "ionization_energies.csv")):
    CHI[(int(r["atomic_number"]), int(r["ion_number"]))] = \
        float(r["ionization_energy_eV"])

# ---------------------------------------------------------------------------
# DR table entries (transcribed from src/lumina_plasma.c DR_TABLE, with the
# dr_lookup first-match convention). ion_recomb = recombining (upper) stage.
# ---------------------------------------------------------------------------
# (16,2) BADNELL "S III -> S II"     (only (16,2) entry)               line 3890
# (14,2) BADNELL "Si III -> Si II"   (FIRST match; a NORAD (14,2) also
#         exists later at line 3973 but dr_lookup returns the Badnell one) line 3859
# (26,3) NORAD  "Fe IV -> Fe III"    (TOTAL RR+DR, not DR-only)         line 3943
DR_TABLE = {
    (16, 2): dict(source="BADNELL",
        c=[3.040e-07,4.393e-07,1.609e-06,4.980e-06,3.457e-05,8.617e-03,9.284e-04],
        E=[5.016e+01,3.266e+02,3.102e+03,1.210e+04,4.969e+04,2.010e+05,2.575e+05]),
    (14, 2): dict(source="BADNELL(first-match)",
        c=[2.930e-06,2.803e-06,9.023e-05,6.909e-03,2.582e-05],
        E=[1.162e+02,5.721e+03,3.477e+04,1.176e+05,3.505e+06]),
    (26, 3): dict(source="NORAD(total RR+DR)",
        c=[2.762952e-07,6.854108e-07,1.112431e-05,3.071916e-05,2.100067e-06,9.690480e-04],
        E=[9.502922e+01,7.045133e+02,1.554180e+04,4.609481e+04,3.392230e+03,2.483458e+05]),
}
# also carry the NORAD (14,2) "Si III->Si II" for reference (NOT selected by lookup)
DR_TABLE_NORAD_SI = dict(source="NORAD(total, NOT selected)",
    c=[9.651958e-08,3.283584e-07,1.361371e-06,6.897597e-03,2.329208e-05],
    E=[1.094415e+02,8.894031e+02,4.719150e+03,1.160976e+05,2.613932e+04])

def alpha_dr(entry, T):
    """dr_alpha_eval: T^-1.5 * sum c_i exp(-E_i/T)  (boost=1)."""
    s = 0.0
    for c, E in zip(entry["c"], entry["E"]):
        a = -E / T
        if a < -700.0:
            continue
        s += c * math.exp(a)
    return s * T ** -1.5

# ---------------------------------------------------------------------------
# Milne alpha, replicating frozenin_alpha_rr EXACTLY (lumina_plasma.c:2708-2769)
# sigma_provider(gl) returns the NFREQ sigma row to use for level gl (full or
# resonance-suppressed). Returns (alpha_total, per-level contributions).
# ---------------------------------------------------------------------------
def milne_alpha(Z, ion, T, sigma_provider):
    chi0 = CHI[(Z, ion)]
    kT = KB * T
    # upper-ion partition U_ion (code 2721-2736)
    idxu = np.where((levZ == Z) & (levI == ion + 1))[0]
    xu = levE[idxu] * EV / kT
    u = float(np.sum(np.where(xu < 50.0, levG[idxu] * np.exp(-np.minimum(xu, 50)), 0.0)))
    if u >= 1.0:
        U_ion = u
    else:
        U_ion = float(levG[idxu[0]]) if len(idxu) and levG[idxu[0]] >= 1 else 1.0
    lam3 = (H * H / (2.0 * PI * ME * KB * T)) ** 1.5
    idx = np.where((levZ == Z) & (levI == ion))[0]
    a_tot = 0.0
    contribs = []
    for gl in idx:
        if not flags[gl]:
            continue
        E_l_erg = levE[gl] * EV
        chi_l = chi0 * EV - E_l_erg
        if chi_l <= 0.0:
            continue
        nu_th = chi_l / H
        sig = sigma_provider(gl)
        x = H * nu_c / kT
        m = (nu_c >= nu_th) & (sig > 0.0) & (x < 700.0)
        if not m.any():
            continue
        B = 2.0 * H * nu_c[m] ** 3 / C ** 2 / np.expm1(x[m])
        Rbf = float(np.sum(4.0 * PI * B * sig[m] / (H * nu_c[m]) * dnu[m]))
        a_l = Rbf * lam3 * levG[gl] / (2.0 * U_ion) * math.exp(chi_l / kT)
        a_tot += a_l
        contribs.append((gl, levN[gl], levE[gl], a_l))
    return a_tot, U_ion, contribs

# ---------------------------------------------------------------------------
# Resonance-suppressed background: multiplicative morphological opening
# (grey opening of log sigma) over the contiguous positive region above
# threshold. Opening removes upward spikes narrower than the structuring
# element while preserving the smooth declining background (identity on
# monotone segments). window W in bins.
# ---------------------------------------------------------------------------
def envelope_row(sig, W):
    out = np.zeros_like(sig)
    pos = np.where(sig > 0.0)[0]
    if pos.size == 0:
        return out
    i0, i1 = pos[0], pos[-1] + 1
    seg = sig[i0:i1].astype(float)
    # fill any interior zeros by nearest positive (keeps opening well-defined)
    if (seg <= 0).any():
        good = seg > 0
        if good.sum() < 2:
            out[i0:i1] = seg
            return out
        xi = np.arange(seg.size)
        seg = np.interp(xi, xi[good], seg[good])
    L = np.log(seg)
    env = np.exp(grey_opening(L, size=W, mode="nearest"))
    env = np.minimum(env, seg)          # opening <= original (guard fp)
    out[i0:i1] = env
    return out

# cache enveloped rows per (gl, W)
_env_cache = {}
def make_provider(W):
    def prov(gl):
        key = (gl, W)
        e = _env_cache.get(key)
        if e is None:
            e = envelope_row(np.asarray(SIG[gl]), W)
            _env_cache[key] = e
        return e
    return prov

def full_provider(gl):
    return np.asarray(SIG[gl])

PAIRS = [
    ("S III -> II",  16, 1, (16, 2)),   # lower ion S II, DR table (16,2)
    ("Si III -> II", 14, 1, (14, 2)),   # lower ion Si II, DR table (14,2)
    ("Fe IV -> III", 26, 2, (26, 3)),   # lower ion Fe III, DR table (26,3)
]
W_PRIMARY = 15          # ~8% frequency window (removes 3000 km/s smoothed resonances)
W_SENS = [7, 15, 31]    # 3.7% / 8% / 16% sensitivity bracket

# ===========================================================================
# SELF-TEST: alpha_full at J=B must reproduce Saha (r/saha=1). Reuse ground
# pair Fe III as in db_photoion_calc.
# ===========================================================================
def saha_selftest():
    Z, ion, T = 26, 2, 11811
    a_full, U_ion, _ = milne_alpha(Z, ion, T, full_provider)
    # Saha: n_{k+1} n_e / n_k = 2 U_up/U_low / lam3 * exp(-chi/kT)
    # alpha (Milne) * (n_{k+1} n_e) should balance photoion G at J=B -> r=saha.
    # Simpler algebraic identity check handled in db_photoion_calc (r/saha=1.000);
    # here just report alpha is finite/positive.
    print(f"[selftest] Fe III alpha_full(T={T}) = {a_full:.4e} cm3/s  U_up={U_ion:.2f}")

# ===========================================================================
# PART 1 -- primary bin-based double-count table
# ===========================================================================
def part1():
    rows = []
    print("\n" + "=" * 96)
    print("PART 1 -- Milne double-count on the ACTUAL bin sigma (code-relevant)")
    print("=" * 96)
    for name, Z, ion, dkey in PAIRS:
        dr = DR_TABLE[dkey]
        print(f"\n### {name}   (lower ion Z={Z} ion={ion}; DR table {dkey} {dr['source']})")
        hdr = (f"{'T[K]':>7} {'alpha_full':>12} {'alpha_smooth':>12} "
               f"{'d_alpha_res':>12} {'alpha_DR':>12} {'f_double':>9} "
               f"{'res_frac':>9}")
        print(hdr)
        for T in T_LIST:
            a_full, _, _ = milne_alpha(Z, ion, T, full_provider)
            a_sm, _, _ = milne_alpha(Z, ion, T, make_provider(W_PRIMARY))
            d_res = a_full - a_sm
            a_dr = alpha_dr(dr, T)
            f_double = d_res / a_dr if a_dr > 0 else float("nan")
            res_frac = d_res / a_full if a_full > 0 else float("nan")
            print(f"{T:7d} {a_full:12.4e} {a_sm:12.4e} {d_res:12.4e} "
                  f"{a_dr:12.4e} {f_double:9.3f} {res_frac:9.3f}")
            rows.append((name, T, a_full, a_sm, d_res, a_dr, f_double, res_frac))
        # window sensitivity at T=11811
        T = 11811
        a_full, _, _ = milne_alpha(Z, ion, T, full_provider)
        sens = []
        for W in W_SENS:
            a_sm, _, _ = milne_alpha(Z, ion, T, make_provider(W))
            sens.append((W, a_full - a_sm))
        a_dr = alpha_dr(dr, T)
        sstr = ", ".join(f"W={W}:d_res={d:.3e}(f={d/a_dr:.2f})" for W, d in sens)
        print(f"   [window sensitivity T={T}]  {sstr}")
    return rows

# ===========================================================================
# PART 2 -- native builder-fidelity cross-check
#   Reproduce the baker (np.interp point-sampling) from the native phot file
#   and compare Milne-weighted resonance area native vs bin-reproduced.
# ===========================================================================
NATIVE = {
    (16, 1): ("SUL/II/30oct12", "s2_osc",     "phot_sm_3000"),
    (14, 1): ("SIL/II/19apr23", "osc_data",   "phot_data_A"),
    (26, 2): ("FE/III/30oct12", "FeIII_OSC",  "phot_sm_3000.dat"),
}
CMFGEN_ROOT = ROOT / "data" / "atomic" / "cmfgen"
CM2EV = 1.239841984e-4
TABULATED = {20, 21, 22}

import re as _re
_TERM_RE = _re.compile(r"\[[^\[\]]*\]\s*$")
def _norm(s):
    return s.strip().lower()
def _term(s):
    return _TERM_RE.sub("", _norm(s))

def native_resonance_area(Z, ion, T, w_frac=0.02):
    """Return (area_native, area_bin, npairs, ntab, config_examples).
    area = sum_l prefactor_l * Milne-weighted resonance area, where
    prefactor_l = g_l * exp(chi_l/kT). Ratio area_bin/area_native = fraction
    of native resonance area retained by the 1000-pt point-sampling.
    Matching mirrors the baker: exact J-resolved config, else term-stripped
    (phot files are term-grouped; osc is J-resolved -> aggregate the term:
    E = lowest J-level energy, g = sum of J-level g)."""
    sub, osc_name, phot_name = NATIVE[(Z, ion)]
    d = CMFGEN_ROOT / sub
    osc = parse_osc(d / osc_name)
    phot = parse_phot(d / phot_name)
    E_ion = osc.ionization_eV
    # exact J-resolved: config -> (E_eV, g)
    cfg2lev = {}
    term_E = {}     # term -> min energy
    term_g = {}     # term -> summed g
    for lv in osc.levels:
        E = float(lv["E_eV"]) if lv["E_eV"] else float(lv["E_cm"]) * CM2EV
        g = float(lv["g"])
        c = _norm(str(lv["config"]))
        if c not in cfg2lev:
            cfg2lev[c] = (E, g)
        t = _term(str(lv["config"]))
        term_E[t] = min(term_E.get(t, 1e30), E)
        term_g[t] = term_g.get(t, 0.0) + g
    # add term-level keys (only if not already an exact key)
    for t in term_E:
        if t not in cfg2lev:
            cfg2lev[t] = (term_E[t], term_g[t])
    kT = KB * T
    W_bin = max(3, int(round(w_frac / dlog)))     # bin-grid opening window
    area_nat = 0.0
    area_bin = 0.0
    ntab = 0
    examples = []
    for e in phot.entries:
        if e.cs_type not in TABULATED or e.energy.size < 4:
            continue
        c = _norm(e.config)
        if c not in cfg2lev:
            c = _term(e.config)
            if c not in cfg2lev:
                continue
        E_l, g_l = cfg2lev[c]
        chi_l = (E_ion - E_l) * EV
        if chi_l <= 0:
            continue
        nu_th = chi_l / H
        pref = g_l * math.exp(min(chi_l / kT, 700.0))
        # ---- native curve ----
        nu_nat = e.energy * nu_th
        sig_nat = e.sigma_Mb * 1.0e-18
        ok = (nu_nat > 0) & (sig_nat > 0)
        if ok.sum() < 4:
            continue
        nu_nat = nu_nat[ok]; sig_nat = sig_nat[ok]
        # sort by frequency
        so = np.argsort(nu_nat)
        nu_nat = nu_nat[so]; sig_nat = sig_nat[so]
        x = H * nu_nat / kT
        kok = x < 700.0
        if kok.sum() < 4:
            continue
        nu_n = nu_nat[kok]; sg_n = sig_nat[kok]
        K_n = (2.0 * H * nu_n ** 3 / C ** 2 / np.expm1(H * nu_n / kT)) / (H * nu_n)
        env_n = _env_arbitrary(nu_n, sg_n, w_frac)
        _tz = getattr(np, "trapezoid", np.trapz)
        R_full_n = _tz(4.0 * PI * sg_n * K_n, nu_n)
        R_sm_n = _tz(4.0 * PI * env_n * K_n, nu_n)
        res_n = max(R_full_n - R_sm_n, 0.0)
        # ---- bin reproduction: point-sample native onto 1000-pt log grid ----
        idx = np.searchsorted(nu_c, nu_th)
        sig_bin = np.zeros(NFREQ)
        if idx < NFREQ:
            sig_bin[idx:] = np.interp(nu_c[idx:], nu_nat, sig_nat,
                                      left=0.0, right=sig_nat[-1])
        xb = H * nu_c / kT
        mb = (nu_c >= nu_th) & (sig_bin > 0) & (xb < 700.0)
        if mb.sum() >= 4:
            nu_b = nu_c[mb]; sg_b = sig_bin[mb]
            K_b = (2.0 * H * nu_b ** 3 / C ** 2 / np.expm1(H * nu_b / kT)) / (H * nu_b)
            env_b = np.exp(grey_opening(np.log(sg_b), size=W_bin, mode="nearest"))
            env_b = np.minimum(env_b, sg_b)
            R_full_b = float(np.sum(4.0 * PI * sg_b * K_b * dnu[mb]))
            R_sm_b = float(np.sum(4.0 * PI * env_b * K_b * dnu[mb]))
            res_b = max(R_full_b - R_sm_b, 0.0)
        else:
            res_b = 0.0
        area_nat += pref * res_n
        area_bin += pref * res_b
        ntab += 1
        if len(examples) < 3:
            examples.append((e.config, e.energy.size, res_n, res_b))
    return area_nat, area_bin, ntab, examples

def _env_arbitrary(nu, sig, w_frac):
    """Multiplicative morphological opening on an arbitrary (sorted) nu grid:
    erosion then dilation with a frequency window |dln nu| <= w_frac/2."""
    ln = np.log(nu)
    half = w_frac / 2.0
    n = ln.size
    lo = np.searchsorted(ln, ln - half, side="left")
    hi = np.searchsorted(ln, ln + half, side="right")
    ero = np.empty(n)
    for i in range(n):
        ero[i] = sig[lo[i]:hi[i]].min()
    dil = np.empty(n)
    for i in range(n):
        dil[i] = ero[lo[i]:hi[i]].max()
    return np.minimum(dil, sig)

def part2():
    print("\n" + "=" * 96)
    print("PART 2 -- native builder-fidelity cross-check "
          "(resonance-area retention of the 1000-pt point-sampling)")
    print("=" * 96)
    out = []
    for name, Z, ion, dkey in PAIRS:
        sub, osc_name, phot_name = NATIVE[(Z, ion)]
        print(f"\n### {name}  native={sub}/{phot_name}")
        for T in [11811]:
            a_nat, a_bin, ntab, ex = native_resonance_area(Z, ion, T)
            ret = a_bin / a_nat if a_nat > 0 else float("nan")
            print(f"  T={T}: tabulated-levels={ntab}  "
                  f"native_res_area={a_nat:.3e}  bin_res_area={a_bin:.3e}  "
                  f"retention(bin/native)={ret:.3f}")
            for cfg, npts, rn, rb in ex:
                print(f"      e.g. {cfg:28s} native_npts={npts:5d} "
                      f"res_n={rn:.2e} res_bin={rb:.2e}")
            out.append((name, T, a_nat, a_bin, ret, ntab))
    return out

# ===========================================================================
# PART 3 -- envelope sanity PNGs (2-3 example levels per ion)
# ===========================================================================
def part3_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [("S_II", 16, 1), ("Fe_III", 26, 2), ("Si_II", 14, 1)]
    for tag, Z, ion in picks:
        idx = np.where((levZ == Z) & (levI == ion) & (flags != 0))[0]
        # pick the ground level + two with the most resonance structure
        chi0 = CHI[(Z, ion)]
        cand = []
        for gl in idx:
            s = np.asarray(SIG[gl])
            if (s > 0).sum() < 20:
                continue
            env = envelope_row(s, W_PRIMARY)
            res = np.sum(np.maximum(s - env, 0))
            cand.append((res, gl))
        cand.sort(reverse=True)
        chosen = [idx[0]] + [gl for _, gl in cand[:2] if gl != idx[0]]
        chosen = chosen[:3]
        fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 4))
        if len(chosen) == 1:
            axes = [axes]
        for ax, gl in zip(axes, chosen):
            s = np.asarray(SIG[gl])
            env = envelope_row(s, W_PRIMARY)
            m = s > 0
            ax.loglog(nu_c[m], s[m], "-", lw=0.8, color="C0", label="bin sigma")
            ax.loglog(nu_c[m], env[m], "-", lw=1.4, color="C3",
                      label=f"envelope W={W_PRIMARY}")
            E_l = levE[gl]; chi_l = chi0 - E_l
            ax.set_title(f"{tag} lvl {levN[gl]} chi_l={chi_l:.2f}eV")
            ax.set_xlabel("nu [Hz]"); ax.set_ylabel("sigma [cm^2]")
            ax.legend(fontsize=7)
        fig.tight_layout()
        out = FIGDIR / f"dr_envelope_{tag}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  wrote {out}")

def part1b():
    """Context: compare Milne alpha_full to an INDEPENDENT total-recomb (RR+DR)
    benchmark from the literature, to show whether Milne already saturates the
    physical recombination before any DR table is added."""
    print("\n" + "=" * 96)
    print("PART 1b -- Milne alpha_full vs independent total-recomb benchmark")
    print("=" * 96)
    # Si III->II: NORAD (14,2) unified RR+DR total (Nahar Pradhan Zhang 2000).
    print("\n### Si III -> II : Milne alpha_full vs NORAD (14,2) independent TOTAL RR+DR")
    print(f"{'T[K]':>7} {'Milne_full':>12} {'NORAD_total':>12} {'full/total':>11} "
          f"{'Badnell_DR':>12} {'(full+DR)/total':>16}")
    for T in T_LIST:
        a_full, _, _ = milne_alpha(14, 1, T, full_provider)
        a_norad = alpha_dr(DR_TABLE_NORAD_SI, T)
        a_bad = alpha_dr(DR_TABLE[(14, 2)], T)
        print(f"{T:7d} {a_full:12.4e} {a_norad:12.4e} {a_full/a_norad:11.2f} "
              f"{a_bad:12.4e} {(a_full+a_bad)/a_norad:16.2f}")
    # Fe IV->III: NORAD (26,3) is itself a TOTAL RR+DR -> the whole table
    # overlaps Milne.
    print("\n### Fe IV -> III : Milne alpha_full vs NORAD (26,3) [the table is TOTAL RR+DR]")
    print(f"{'T[K]':>7} {'Milne_full':>12} {'NORAD_total':>12} {'full/total':>11} "
          f"{'(full+table)/total':>18}")
    for T in T_LIST:
        a_full, _, _ = milne_alpha(26, 2, T, full_provider)
        a_norad = alpha_dr(DR_TABLE[(26, 3)], T)
        print(f"{T:7d} {a_full:12.4e} {a_norad:12.4e} {a_full/a_norad:11.2f} "
              f"{(a_full+a_norad)/a_norad:18.2f}")

if __name__ == "__main__":
    print(f"bin: NLEV={NLEV} NFREQ={NFREQ} grid[{numin:.2e},{numax:.2e}] dlog={dlog:.5f}")
    saha_selftest()
    r1 = part1()
    part1b()
    r2 = part2()
    print("\n" + "=" * 96)
    print("PART 3 -- envelope sanity plots")
    print("=" * 96)
    part3_plots()
    print("\nDONE.")
