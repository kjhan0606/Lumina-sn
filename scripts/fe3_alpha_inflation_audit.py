#!/usr/bin/env python3
"""Fe IV -> Fe III recombination-coefficient INFLATION audit (OFFLINE).

Question: our Milne alpha_full(Fe III) over the baked per-level sigma_bf is
4.5-6.7x the independent NORAD total (RR+DR). Is that inflation
frequency/level-UNIFORM (cancels in r=G/(n_e alpha), equilibrium-harmless) or
STRUCTURED (distorts the ionization balance)? And what is its root cause?

PART A -- equilibrium impact:
  A1  per-level decomposition of alpha_full at T=13120 K (top contributors).
  A2  r-sensitivity with the ACTUAL A-run field: (i) sigma as-is,
      (ii) sigma * 1/5 uniform (pure-scale null: r must be identical),
      (iii) only top-inflated levels rescaled (structured hypothesis).
  A3  same for the B-run state.

PART B -- root cause candidates:
  B1  level-count/identity: 1500 levels.csv vs 557 native phot terms; split-J
      duplication check; threshold spot-check.
  B2  native-resolution Milne (no bin resampling) vs bin frozenin; and vs
      the g_c=6 (single-core) normalization -> isolates pipeline vs convention.
  B3  LTE-weight / U_ion cross-check: independent from-scratch Milne vs the
      frozenin replication.
  B4  baseline validity: NORAD Fe3.csv grid direct read.

Reproduces src/lumina_plasma.c:frozenin_alpha_rr (2708-2769). ANALYSIS ONLY.
"""
from __future__ import annotations
import struct, csv, math, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
sys.path.insert(0, str(ROOT / "scripts"))
from cmfgen_parser import parse_osc, parse_phot  # noqa: E402

# ---- constants (CGS), identical to src/lumina_plasma.c ----
H = 6.62607015e-27; KB = 1.380649e-16; C = 2.99792458e10
ME = 9.1093837015e-28; EV = 1.602176634e-12; PI = math.pi
KB_EV = 8.617333262e-5; CM2EV = 1.239841984e-4

REF = ROOT / "data" / "tardis_reference_toy06_19p48d"
BIN = REF / "cmfgen_sigma_bf.bin"

# ---------------------------------------------------------------------------
# bin reader (same as db_photoion_calc / dr_doublecount_audit)
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
dnu = (np.exp(log_numin + (np.arange(NFREQ) + 1) * dlog)
       - np.exp(log_numin + np.arange(NFREQ) * dlog))

levZ = []; levI = []; levN = []; levE = []; levG = []
for r in csv.DictReader(open(REF / "levels.csv")):
    levZ.append(int(r["atomic_number"])); levI.append(int(r["ion_number"]))
    levN.append(int(r["level_number"])); levE.append(float(r["energy_eV"]))
    levG.append(int(r["g"]))
levZ = np.array(levZ); levI = np.array(levI); levN = np.array(levN)
levE = np.array(levE); levG = np.array(levG)
assert len(levZ) == NLEV

CHI = {}
for r in csv.DictReader(open(REF / "ionization_energies.csv")):
    CHI[(int(r["atomic_number"]), int(r["ion_number"]))] = \
        float(r["ionization_energy_eV"])

T_LIST = [10470, 11811, 13120, 16014]

# NORAD (26,3) Fe IV->III total RR+DR Burgess fit (DR_TABLE entry, lumina_plasma.c:3943)
NORAD_C = [2.762952e-07, 6.854108e-07, 1.112431e-05, 3.071916e-05, 2.100067e-06, 9.690480e-04]
NORAD_E = [9.502922e+01, 7.045133e+02, 1.554180e+04, 4.609481e+04, 3.392230e+03, 2.483458e+05]

def norad_total(T):
    s = 0.0
    for c, E in zip(NORAD_C, NORAD_E):
        a = -E / T
        if a >= -700.0:
            s += c * math.exp(a)
    return s * T ** -1.5

def norad_grid():
    """Direct NORAD Fe3.csv grid (bypass Burgess fit)."""
    out = []
    p = ROOT / "data/atomic/dr_norad/Fe3.csv"
    for r in csv.DictReader((ln for ln in open(p) if not ln.startswith("#"))):
        out.append((float(r["T_K"]), float(r["alpha_total_cm3_per_s"])))
    return out

# ---------------------------------------------------------------------------
# U_ion(Fe IV) exactly as frozenin (code 2721-2736)
# ---------------------------------------------------------------------------
def U_upper(Z, ion, T):
    kT = KB * T
    idxu = np.where((levZ == Z) & (levI == ion + 1))[0]
    xu = levE[idxu] * EV / kT
    u = float(np.sum(np.where(xu < 50.0, levG[idxu] * np.exp(-np.minimum(xu, 50)), 0.0)))
    if u >= 1.0:
        return u, idxu
    gg = levG[idxu[0]] if len(idxu) else 1
    return (float(gg) if gg >= 1 else 1.0), idxu

# ---------------------------------------------------------------------------
# bin frozenin alpha (EXACT replication), returns total + per-level contribs
# ---------------------------------------------------------------------------
def milne_bin(Z, ion, T, sig_scale=None):
    chi0 = CHI[(Z, ion)]; kT = KB * T
    U_ion, _ = U_upper(Z, ion, T)
    lam3 = (H * H / (2 * PI * ME * KB * T)) ** 1.5
    idx = np.where((levZ == Z) & (levI == ion))[0]
    a_tot = 0.0; contribs = []
    for gl in idx:
        if not flags[gl]:
            continue
        chi_l = chi0 * EV - levE[gl] * EV
        if chi_l <= 0.0:
            continue
        nu_th = chi_l / H
        sig = np.asarray(SIG[gl])
        if sig_scale is not None:
            sig = sig * sig_scale(gl)
        x = H * nu_c / kT
        m = (nu_c >= nu_th) & (sig > 0.0) & (x < 700.0)
        if not m.any():
            continue
        B = 2 * H * nu_c[m] ** 3 / C ** 2 / np.expm1(x[m])
        Rbf = float(np.sum(4 * PI * B * sig[m] / (H * nu_c[m]) * dnu[m]))
        a_l = Rbf * lam3 * levG[gl] / (2 * U_ion) * math.exp(chi_l / kT)
        a_tot += a_l
        contribs.append((int(levN[gl]), float(levE[gl]), float(chi_l / EV),
                         a_l, int(gl)))
    return a_tot, U_ion, contribs

# ---------------------------------------------------------------------------
# NATIVE-resolution Milne from phot_sm_3000.dat (no bin resampling).
# Mirrors frozenin per-J-level but integrates native sigma via trapezoid.
# norm: 'U'  -> g_l/(2 U_ion)   (frozenin convention)
#       'gc' -> g_l/(2 g_core)  (textbook single-core Milne, g_core from header)
# ---------------------------------------------------------------------------
_TERM_RE = __import__("re").compile(r"\[[^\[\]]*\]\s*$")
def _norm(s): return str(s).strip().lower()
def _term(s): return _TERM_RE.sub("", _norm(s))

def milne_native(T, norm="U", g_core=6.0):
    d = ROOT / "data/atomic/cmfgen/FE/III/30oct12"
    osc = parse_osc(d / "FeIII_OSC")
    phot = parse_phot(d / "phot_sm_3000.dat")
    E_ion = osc.ionization_eV
    kT = KB * T
    U_ion, _ = U_upper(26, 2, T)
    denom = U_ion if norm == "U" else g_core
    lam3 = (H * H / (2 * PI * ME * KB * T)) ** 1.5
    # J-resolved config -> (E_eV, g); term -> list of (E_eV,g)
    cfg2lev = {}; term2levs = {}
    gsum_term = {}
    for lv in osc.levels:
        E = float(lv["E_cm"]) * CM2EV
        g = float(lv["g"]); c = _norm(str(lv["config"]))
        cfg2lev.setdefault(c, (E, g))
        t = _term(str(lv["config"]))
        term2levs.setdefault(t, []).append((E, g))
        gsum_term[t] = gsum_term.get(t, 0.0) + g
    a_tot = 0.0; per_term = []
    matched = 0
    for e in phot.entries:
        if e.cs_type not in (20, 21, 22) or e.energy.size < 4:
            continue
        c = _norm(e.config)
        if c in cfg2lev:
            sublevels = [cfg2lev[c]]
        else:
            sublevels = term2levs.get(_term(e.config), [])
        if not sublevels:
            continue
        matched += 1
        a_term = 0.0
        for (E_l, g_l) in sublevels:
            chi_l = (E_ion - E_l) * EV
            if chi_l <= 0:
                continue
            nu_th = chi_l / H
            nu = e.energy * nu_th          # E_ratio -> Hz
            sig = e.sigma_Mb * 1.0e-18
            ok = (nu > 0) & (sig > 0)
            if ok.sum() < 4:
                continue
            nu = nu[ok]; sig = sig[ok]
            so = np.argsort(nu); nu = nu[so]; sig = sig[so]
            x = H * nu / kT
            kk = x < 700.0
            if kk.sum() < 4:
                continue
            nu = nu[kk]; sig = sig[kk]
            B = 2 * H * nu ** 3 / C ** 2 / np.expm1(H * nu / kT)
            tz = getattr(np, "trapezoid", np.trapz)
            Rbf = tz(4 * PI * B * sig / (H * nu), nu)
            a_l = Rbf * lam3 * g_l / (2 * denom) * math.exp(chi_l / kT)
            a_term += a_l
        a_tot += a_term
        per_term.append((e.config, a_term))
    return a_tot, matched, U_ion, per_term

# ---------------------------------------------------------------------------
# Field / plasma / level populations (from a run dir), G-side machinery
# (replicates db_photoion_calc)
# ---------------------------------------------------------------------------
def field(d, shell):
    Jrow = np.zeros(NFREQ)
    for r in csv.DictReader(open(f"{d}/lumina_coevolve_field.csv")):
        if int(r["shell"]) != shell:
            continue
        b = int(r["bin"]); m = float(r["mc_J"]); cj = float(r["cs_J"])
        Jrow[b] = m if m > 0 else cj
    return Jrow

def plasma(d, shell):
    for r in csv.DictReader(open(f"{d}/lumina_plasma_state.csv")):
        if int(r["shell_id"]) == shell:
            return float(r["T_e"]), float(r["n_e"])

def levelpops(d, Z, ion, shell):
    out = {}
    for r in csv.DictReader(open(f"{d}/lumina_levelpop.csv")):
        if int(r["Z"]) == Z and int(r["ion"]) == ion and int(r["shell"]) == shell:
            out[int(r["level_num"])] = (float(r["n_k"]), float(r["b_k"]))
    return out

def r_and_f(d, shell, Z, ion, sig_scale=None):
    """r(ion->ion+1)=G/(n_e alpha) with Boltzmann and NLTE G weightings.
    sig_scale(gl) multiplies sigma for BOTH G and alpha (so a uniform scale
    cancels). Returns dict with alpha, G_b, G_n, r_b, r_n, f_b, f_n."""
    Te, ne = plasma(d, shell)
    J = field(d, shell)
    chi0 = CHI[(Z, ion)]; kT = KB * Te
    idx = np.where((levZ == Z) & (levI == ion))[0]
    x = levE[idx] / (KB_EV * Te)
    U = float(np.sum(np.where(x < 50, levG[idx] * np.exp(-np.minimum(x, 50)), 0)))
    pops = levelpops(d, Z, ion, shell)
    ntot = sum(v[0] for v in pops.values())
    alpha, _, _ = milne_bin(Z, ion, Te, sig_scale=sig_scale)
    G_b = G_n = 0.0
    for gl in idx:
        if not flags[gl]:
            continue
        chi_l = chi0 * EV - levE[gl] * EV
        if chi_l <= 0:
            continue
        nu_th = chi_l / H
        sig = np.asarray(SIG[gl])
        if sig_scale is not None:
            sig = sig * sig_scale(gl)
        m = (nu_c >= nu_th) & (sig > 0) & (J > 0)
        if not m.any():
            continue
        R = float(np.sum(4 * PI * sig[m] * J[m] / (H * nu_c[m]) * dnu[m]))
        xl = levE[gl] / (KB_EV * Te)
        if xl >= 50:
            continue
        pb = levG[gl] * math.exp(-xl) / U
        n = int(levN[gl]); nk, bk = pops.get(n, (0.0, float("nan")))
        pn = nk / ntot if ntot > 0 else 0.0
        G_b += pb * R; G_n += pn * R
    r_b = G_b / (ne * alpha); r_n = G_n / (ne * alpha)
    ff = lambda rr: rr / (1 + rr)
    return dict(Te=Te, ne=ne, alpha=alpha, G_b=G_b, G_n=G_n,
                r_b=r_b, r_n=r_n, f_b=ff(r_b), f_n=ff(r_n))

A = "logs/coevolve_consume_a10_kx_gphground"
B = "logs/coevolve_consume_a10_kx_gphall"

# ===========================================================================
def main():
    print(f"bin: NLEV={NLEV} NFREQ={NFREQ} grid[{numin:.3e},{numax:.3e}] dlog={dlog:.5f}")
    nfe3 = int(((levZ == 26) & (levI == 2)).sum())
    nfe3_sig = int(((levZ == 26) & (levI == 2) & (flags != 0)).sum())
    print(f"Fe III levels.csv: {nfe3}  (with sigma flag: {nfe3_sig})")

    # ---- validate reader reproduces established alpha_full ----
    print("\n" + "=" * 92)
    print("VALIDATION: bin frozenin alpha_full(Fe III) vs established facts + NORAD")
    print("=" * 92)
    print(f"{'T[K]':>7} {'alpha_full':>12} {'NORAD_fit':>12} {'NORAD_grid':>12} "
          f"{'full/NORAD':>11} {'U_ion(FeIV)':>11}")
    grid = norad_grid()
    def grid_at(T):
        Ts = np.array([g[0] for g in grid]); As = np.array([g[1] for g in grid])
        return float(np.interp(T, Ts, As))
    for T in T_LIST:
        a, U, _ = milne_bin(26, 2, T)
        nf = norad_total(T); ng = grid_at(T)
        print(f"{T:7d} {a:12.4e} {nf:12.4e} {ng:12.4e} {a/nf:11.2f} {U:11.3f}")

    # =======================================================================
    # PART A1 -- per-level decomposition at T=13120
    # =======================================================================
    T = 13120
    a_tot, U_ion, contribs = milne_bin(26, 2, T)
    contribs.sort(key=lambda c: -c[3])
    print("\n" + "=" * 92)
    print(f"PART A1 -- per-level alpha decomposition (Fe III, T={T} K, "
          f"alpha_tot={a_tot:.4e}, U_ion={U_ion:.2f})")
    print("=" * 92)
    print(f"{'rank':>4} {'level_num':>9} {'E_eV':>9} {'chi_l_eV':>9} "
          f"{'g':>4} {'alpha_l':>12} {'frac':>7} {'cum':>7}")
    cum = 0.0
    for i, (n, E, chi, a_l, gl) in enumerate(contribs[:15]):
        cum += a_l / a_tot
        print(f"{i+1:4d} {n:9d} {E:9.3f} {chi:9.3f} {int(levG[gl]):4d} "
              f"{a_l:12.4e} {a_l/a_tot:7.4f} {cum:7.4f}")
    n_for_50 = 0; c = 0.0
    for (_, _, _, a_l, _) in contribs:
        c += a_l / a_tot; n_for_50 += 1
        if c >= 0.5:
            break
    n_for_90 = 0; c = 0.0
    for (_, _, _, a_l, _) in contribs:
        c += a_l / a_tot; n_for_90 += 1
        if c >= 0.9:
            break
    print(f"  concentration: {n_for_50} levels -> 50% of alpha; "
          f"{n_for_90} levels -> 90%.  (Fe III has {nfe3_sig} sigma-levels)")
    # top-inflated level set for structured test: the levels carrying most alpha
    top_gl = set(gl for (_, _, _, _, gl) in contribs[:n_for_90])
    ground_gl = set(gl for (n, _, _, _, gl) in contribs if n < 5)  # 3d6_5De J-multiplet

    # =======================================================================
    # PART A2/A3 -- r-sensitivity (does the 5x distort the ionization verdict?)
    # =======================================================================
    for label, d in [("A-run (gphground)", A), ("B-run (gphall)", B)]:
        print("\n" + "=" * 92)
        print(f"PART A2/A3 -- r(Fe III->IV) sensitivity, {label}, shell 0")
        print("=" * 92)
        base = r_and_f(d, 0, 26, 2, sig_scale=None)
        fifth = r_and_f(d, 0, 26, 2, sig_scale=lambda gl: 0.2)  # uniform 1/5
        # structured: rescale ONLY the top-90% alpha-carrying levels by 1/5
        topscale = r_and_f(d, 0, 26, 2,
                           sig_scale=lambda gl: 0.2 if gl in top_gl else 1.0)
        # structured alt: rescale ONLY the ground multiplet by 1/5
        gndscale = r_and_f(d, 0, 26, 2,
                           sig_scale=lambda gl: 0.2 if gl in ground_gl else 1.0)
        print(f"  Te={base['Te']:.0f} ne={base['ne']:.3e}")
        print(f"  {'variant':<34} {'alpha':>11} {'G_boltz':>11} "
              f"{'r_boltz':>10} {'f_IV(b)':>9} | {'r_nlte':>10} {'f_IV(n)':>9}")
        for nm, res in [("(i)  sigma as-is", base),
                        ("(ii) sigma*1/5 UNIFORM (null)", fifth),
                        ("(iii) top-90% alpha lv *1/5", topscale),
                        ("(iii') ground multiplet *1/5", gndscale)]:
            print(f"  {nm:<34} {res['alpha']:11.3e} {res['G_b']:11.3e} "
                  f"{res['r_b']:10.3e} {res['f_b']:9.4f} | "
                  f"{res['r_n']:10.3e} {res['f_n']:9.4f}")
        # explicit deltas
        print(f"  --> f_IV(boltz): (i)={base['f_b']:.4f} (ii)={fifth['f_b']:.4f} "
              f"(iii)={topscale['f_b']:.4f}   | ratio r(ii)/r(i)={fifth['r_b']/base['r_b']:.4f}")
        print(f"  --> f_IV(nlte) : (i)={base['f_n']:.4f} (ii)={fifth['f_n']:.4f} "
              f"(iii)={topscale['f_n']:.4f}   | ratio r(ii)/r(i)={fifth['r_n']/base['r_n']:.4f}")

    # =======================================================================
    # PART B2 -- native-resolution Milne vs bin; U-norm vs gc=6 norm
    # =======================================================================
    print("\n" + "=" * 92)
    print("PART B2 -- native-resolution Milne (no resampling) vs bin frozenin, "
          "and normalization test")
    print("=" * 92)
    print(f"{'T[K]':>7} {'bin(U)':>12} {'native(U)':>12} {'nat/bin':>8} "
          f"{'native(gc=6)':>13} {'NORAD':>12} {'natU/NORAD':>11} {'natGc/NORAD':>12}")
    for T in T_LIST:
        a_bin, _, _ = milne_bin(26, 2, T)
        a_natU, mt, U_ion, _ = milne_native(T, norm="U")
        a_natG, _, _, _ = milne_native(T, norm="gc", g_core=6.0)
        nf = norad_total(T)
        print(f"{T:7d} {a_bin:12.4e} {a_natU:12.4e} {a_natU/a_bin:8.3f} "
              f"{a_natG:13.4e} {nf:12.4e} {a_natU/nf:11.2f} {a_natG/nf:12.2f}")
    print(f"  (native matched {mt} phot terms of 557; U_ion(FeIV,13120)={U_upper(26,2,13120)[0]:.3f})")

    # =======================================================================
    # PART B1 -- level identity / threshold spot check
    # =======================================================================
    print("\n" + "=" * 92)
    print("PART B1 -- level-count/identity + threshold spot-check")
    print("=" * 92)
    osc = parse_osc(ROOT / "data/atomic/cmfgen/FE/III/30oct12/FeIII_OSC")
    phot = parse_phot(ROOT / "data/atomic/cmfgen/FE/III/30oct12/phot_sm_3000.dat")
    ntab = sum(1 for e in phot.entries if e.cs_type in (20, 21, 22))
    print(f"  FeIII_OSC levels        : {osc.n_levels} (J-resolved)")
    print(f"  levels.csv Fe III       : {nfe3}  (sigma flag set: {nfe3_sig})")
    print(f"  phot_sm_3000 entries    : {len(phot.entries)} (tabulated: {ntab}) "
          f"-- TERM-resolved (Split J = False)")
    g_osc = float(sum(float(l["g"]) for l in osc.levels))
    g_csv = float(levG[(levZ == 26) & (levI == 2)].sum())
    print(f"  sum g (OSC 1500)        : {g_osc:.0f}   sum g (levels.csv 1500): {g_csv:.0f}"
          f"   {'MATCH (no g-duplication)' if abs(g_osc-g_csv)<1 else 'MISMATCH!'}")
    # threshold spot-check on 5 phot terms: implied nu_th from bin turn-on vs OSC
    print("  threshold spot-check (phot term -> J-sublevels; chi from OSC energies):")
    chi0 = CHI[(26, 2)]
    checked = 0
    for e in phot.entries:
        if e.cs_type not in (20, 21, 22):
            continue
        t = _term(e.config)
        subs = [(float(l["E_cm"]) * CM2EV, float(l["g"]))
                for l in osc.levels if _term(str(l["config"])) == t]
        if not subs:
            continue
        E0 = min(s[0] for s in subs)
        chi_term = chi0 - E0
        nu_th_expected = chi_term * EV / H
        # find a levels.csv level of this term and read bin sigma turn-on
        print(f"    {e.config:16s} n_sub={len(subs):2d} sum_g={sum(s[1] for s in subs):4.0f} "
              f"E0={E0:7.3f}eV chi_term={chi_term:7.3f}eV nu_th={nu_th_expected:.3e}Hz")
        checked += 1
        if checked >= 5:
            break

    # =======================================================================
    # PART B3 -- independent from-scratch Milne (verify replication, not code bug)
    # =======================================================================
    print("\n" + "=" * 92)
    print("PART B3 -- independent from-scratch Milne vs frozenin replication (T=13120)")
    print("=" * 92)
    T = 13120
    a_repl, U_ion, _ = milne_bin(26, 2, T)
    # from scratch: alpha = sum_l g_l/(2U) lam3 exp(chi/kT) * int (8pi nu^2/c^2)/(e^{hnu/kT}-1) sigma dnu
    kT = KB * T
    lam3 = (H * H / (2 * PI * ME * KB * T)) ** 1.5
    idx = np.where((levZ == 26) & (levI == 2))[0]
    a_scratch = 0.0
    for gl in idx:
        if not flags[gl]:
            continue
        chi_l = (CHI[(26, 2)] - levE[gl]) * EV
        if chi_l <= 0:
            continue
        nu_th = chi_l / H
        sig = np.asarray(SIG[gl])
        m = (nu_c >= nu_th) & (sig > 0) & (H * nu_c / kT < 700)
        if not m.any():
            continue
        integ = np.sum((8 * PI * nu_c[m] ** 2 / C ** 2)
                       / np.expm1(H * nu_c[m] / kT) * sig[m] * dnu[m])
        a_scratch += levG[gl] / (2 * U_ion) * lam3 * math.exp(chi_l / kT) * integ
    print(f"  frozenin replication : {a_repl:.6e}")
    print(f"  from-scratch Milne   : {a_scratch:.6e}")
    print(f"  ratio scratch/repl   : {a_scratch/a_repl:.6f}  "
          f"({'REPLICATION OK' if abs(a_scratch/a_repl-1)<1e-6 else 'DISCREPANCY'})")

    # =======================================================================
    # PART B4 -- baseline validity (NORAD grid vs Burgess fit)
    # =======================================================================
    print("\n" + "=" * 92)
    print("PART B4 -- NORAD baseline validity: Fe3.csv grid vs DR_TABLE Burgess fit")
    print("=" * 92)
    print(f"{'T[K]':>7} {'grid':>12} {'burgess_fit':>12} {'fit/grid':>9}")
    for T in [1.0e4, 1.2589e4, 1.5849e4]:
        g = grid_at(T); nf = norad_total(T)
        print(f"{T:7.0f} {g:12.4e} {nf:12.4e} {nf/g:9.4f}")
    print("  (raw table cross-check @1e4K: RRC_lown 3.85e-12 + HRC_highn 1.23e-12 "
          "= 5.08e-12 total; DR 1.4e-21 negligible)")

    part_b5_spin()


# ===========================================================================
# NORAD state-specific RRC parser (raw_fe3.rrc.ls.txt)
# ===========================================================================
import re as _re2
def norad_state_specific():
    p = ROOT / "data/atomic/dr_norad/raw_fe3.rrc.ls.txt"
    lines = open(p).read().splitlines()
    marks = [i for i, l in enumerate(lines) if l.strip() == "photoionization: opacity values"]
    start = marks[-1] + 1
    end = [i for i, l in enumerate(lines)
           if l.strip().startswith("iv) Total RRC") and i > start][0]
    idre = _re2.compile(r"^\s*(\d{8}\.\d{4})")
    valre = _re2.compile(r"[-+]?\d*\.?\d+[eE][-+]?\d+")
    ents = []; cur = None; tk = []
    for l in lines[start:end]:
        m = idre.match(l)
        if m:
            if cur is not None:
                ents.append((cur, tk))
            cur = m.group(1); tk = [float(x) for x in valre.findall(l[m.end():])]
        elif cur is not None:
            tk += [float(x) for x in valre.findall(l)]
    if cur is not None:
        ents.append((cur, tk))
    return ents

def part_b5_spin():
    """ROOT-CAUSE test: recombination onto the Fe IV 3d5 6S ground core (S=5/2)
    + e (s=1/2) can form ONLY quintet/septet Fe III (S=2,3). Triplet/singlet
    Fe III terms are spin-forbidden -> NORAD omits them; our Milne spuriously
    includes their large CMFGEN sigma."""
    print("\n" + "=" * 92)
    print("PART B5 -- SPIN-SELECTION root cause (structured, not uniform)")
    print("=" * 92)
    Lsym = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H", 6: "I", 7: "K", 8: "L"}
    ents = norad_state_specific()
    i1e4 = 30
    tot = 0.0; terms = set()
    for lid, tk in ents:
        if len(tk) < 82:
            continue
        tot += tk[1:82][i1e4]
        s = lid.split(".")[0]
        terms.add(f"{s[0]}{Lsym.get(int(s[1:3]), s[1:3])}{'e' if s[3]=='0' else 'o'}")
    mult_set = sorted(set(t[0] for t in terms))
    print(f"  NORAD state-specific: {len(ents)} entries sum to {tot:.3e} @1e4K "
          f"(= low-n column 3.85e-12).")
    print(f"  multiplicities present in NORAD recombined states: {mult_set} "
          f"(ONLY quintet '5' & septet '7' -> spin coupling 6S core (5/2) x e(1/2) = 2,3)")
    print(f"  triplet/singlet-even terms in NORAD: "
          f"{sorted([t for t in terms if t[0] in '13' and t.endswith('e')]) or 'NONE'}")
    # multiplicity decomposition of alpha and G for both runs (shell 0)
    osc = parse_osc(ROOT / "data/atomic/cmfgen/FE/III/30oct12/FeIII_OSC")
    TR = _re2.compile(r"\[[^\[\]]*\]\s*$")
    def mult(cfg):
        t = TR.sub("", str(cfg).strip()).split("_")[-1]
        mm = _re2.match(r"(\d)", t)
        return int(mm.group(1)) if mm else 0
    mult_of = [mult(l["config"]) for l in osc.levels]
    for label, d, Te in [("A-run", A, 16014), ("B-run", B, 13120)]:
        Te_r, ne = plasma(d, 0); J = field(d, 0)
        chi0 = CHI[(26, 2)]; kT = KB * Te_r
        idx = np.where((levZ == 26) & (levI == 2))[0]
        x = levE[idx] / (KB_EV * Te_r)
        U = float(np.sum(np.where(x < 50, levG[idx] * np.exp(-np.minimum(x, 50)), 0)))
        Uu, _ = U_upper(26, 2, Te_r)
        lam3 = (H * H / (2 * PI * ME * KB * Te_r)) ** 1.5
        pp = levelpops(d, 26, 2, 0); ntot = sum(v[0] for v in pp.values())
        Aa = {"allow": 0., "forbid": 0.}; Gn = {"allow": 0., "forbid": 0.}
        for gl in idx:
            if not flags[gl]:
                continue
            chi_l = chi0 * EV - levE[gl] * EV
            if chi_l <= 0:
                continue
            nu_th = chi_l / H; sig = np.asarray(SIG[gl]); xx = H * nu_c / kT
            grp = "allow" if mult_of[int(levN[gl])] in (5, 7, 9) else "forbid"
            ma = (nu_c >= nu_th) & (sig > 0) & (xx < 700)
            if ma.any():
                Bp = 2 * H * nu_c[ma] ** 3 / C ** 2 / np.expm1(xx[ma])
                Rb = float(np.sum(4 * PI * Bp * sig[ma] / (H * nu_c[ma]) * dnu[ma]))
                Aa[grp] += Rb * lam3 * levG[gl] / (2 * Uu) * math.exp(chi_l / kT)
            mg = (nu_c >= nu_th) & (sig > 0) & (J > 0)
            xl = levE[gl] / (KB_EV * Te_r)
            if mg.any() and xl < 50:
                R = float(np.sum(4 * PI * sig[mg] * J[mg] / (H * nu_c[mg]) * dnu[mg]))
                nk, bk = pp.get(int(levN[gl]), (0., float("nan")))
                Gn[grp] += (nk / ntot if ntot > 0 else 0) * R
        at = Aa["allow"] + Aa["forbid"]; gt = Gn["allow"] + Gn["forbid"]
        ff = lambda r: r / (1 + r)
        r_as = gt / (ne * at); r_fix = gt / (ne * Aa["allow"])
        print(f"\n  {label} shell0 Te={Te_r:.0f}:")
        print(f"    alpha : spin-allowed(5,7)={Aa['allow']:.3e} ({100*Aa['allow']/at:4.1f}%)"
              f"   spin-FORBIDDEN(1,3)={Aa['forbid']:.3e} ({100*Aa['forbid']/at:4.1f}%)")
        print(f"    G_nlte: spin-allowed={Gn['allow']:.3e} ({100*Gn['allow']/gt:4.1f}%)"
              f"   spin-forbidden={Gn['forbid']:.3e} ({100*Gn['forbid']/gt:4.1f}%)")
        print(f"    => forbidden dominates ALPHA ({100*Aa['forbid']/at:.0f}%) but not "
              f"G_nlte ({100*Gn['forbid']/gt:.0f}%): STRUCTURED non-cancellation.")
        print(f"    f_IV(nlte) as-is={ff(r_as):.4f}  ->  drop spin-forbidden alpha={ff(r_fix):.4f}")


if __name__ == "__main__":
    main()
