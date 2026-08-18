"""Y2 step 10 (NEW in the formal cert; no prelim counterpart).

Mechanical re-derivation of the Y3 pre-registration items P1-P8 on the clean
iter-11 field.  Two things the prelim could not do:

 (i) P1/P2 are registered over the SHELL BAND s20-35, but the prelim only ever
     evaluated s20 and s30.  Here the Boltzmann-weighted and ground-level
     C2/GEMM ratios are computed for EVERY shell 20..35 so the registered
     bracket can be tested against its own domain.
 (ii) the OPTIONAL population-weighted ratio.  The clean run dumped
     lumina_levelpop_resolve_{raw,ema}.csv (writer: src/lumina_cuda.cu:855-895,
     schema shell,Z,ion,level_num,E_eV,g,n_k,n_ground,b_k,has_sigma,n_sig_pos;
     a SINGLE post-iter-11 final-resolve snapshot, NO iter column -- gate
     LUMINA_NLTE_FINAL_RESOLVE, src/lumina_cuda.cu:8592-8686).  n_k is the NLTE
     level population the matrix actually weights with (lumina_plasma.c:14686),
     so  Gamma_pop = sum_l n_k[l] R_bf[l] / sum_l n_k[l]  replaces the
     Boltzmann bracket by the realized weighting.

Everything else (sigma, thresholds, both integrals) is byte-identical to step5.
"""
import os
import numpy as np
import pandas as pd
import y2_common as Y

IT = int(os.environ.get("Y2_IT", "-1"))     # -1 = iter 11, -2 = iter 10
SUF = "" if IT == -1 else f"_it{IT}"
J_C1 = np.load(os.path.join(Y.OUT, "_cache_JC1.npy"))
bfr = np.load(os.path.join(Y.OUT, "_cache_bfr.npy"))
TR = np.load(os.path.join(Y.OUT, "_cache_TR.npy"))
Te = TR[IT, :, 14].copy()
assert (Te > 0).all(), "this iter must have all 50 shells T_e-pinned"

lv, chi = Y.load_levels(), Y.load_ioniz()
sg = Y.SigmaBF(Y.sigma_path())
SIG = np.memmap(Y.sigma_path(), dtype="<f8", mode="r", offset=sg.data_off,
                shape=(sg.n_lev, sg.n_freq))
PB = 4.0 * np.pi / (Y.H_PLANCK * Y.NU_MID) * Y.DNU

CONSUMED = [(14, 1, "Si II"), (20, 1, "Ca II"), (26, 1, "Fe II"), (16, 1, "S II"),
            (27, 1, "Co II"), (28, 1, "Ni II"), (6, 1, "C II"), (12, 1, "Mg II"),
            (22, 1, "Ti II"), (24, 1, "Cr II"), (13, 1, "Al II"), (21, 1, "Sc II"),
            (25, 1, "Mn II"), (8, 0, "O I"), (8, 1, "O II")]
REFONLY = [(14, 2, "Si III"), (16, 2, "S III"), (26, 2, "Fe III"),
           (26, 3, "Fe IV"), (27, 2, "Co III")]

# ---------------------------------------------------- optional: level pops
POPS = {}
for tag in ("raw", "ema"):
    p = os.path.join(Y.RUN, f"lumina_levelpop_resolve_{tag}.csv")
    if not os.path.exists(p):
        print(f"[pop] {tag}: file absent -> skipped")
        continue
    lp = pd.read_csv(p, usecols=["shell", "Z", "ion", "level_num", "n_k"])
    POPS[tag] = lp
    print(f"[pop] {tag}: {len(lp)} rows, shells {lp.shell.min()}..{lp.shell.max()}, "
          f"{lp.groupby(['Z','ion']).ngroups} ions")


def rates(Z, ion, shells):
    """-> (E, gw, ok, c2[nlev,nsh], gm[nlev,nsh]) for one ion, iter 11."""
    sub = lv[(lv.atomic_number == Z) & (lv.ion_number == ion)]
    chi_eV = chi.get((Z, ion))
    if len(sub) == 0 or chi_eV is None:
        return None
    g = sub.gidx.values
    E = sub.energy_eV.values
    gw = sub.g.values.astype(float)
    lnum = sub.level_number.values
    nu_th = (chi_eV - E) * Y.EV_TO_ERG / Y.H_PLANCK
    ok = nu_th > 0
    sigma = np.array(SIG[g, :])
    has = sg.has[g].astype(bool)
    s0 = 7.91e-18 / float(ion + 1) ** 2
    kr = np.where(Y.NU_MID[None, :] >= nu_th[:, None],
                  s0 * (nu_th[:, None] / Y.NU_MID[None, :]) ** 3, 0.0)
    sigma = np.where(has[:, None], sigma, kr)
    above = Y.NU_MID[None, :] >= nu_th[:, None]
    m_c2 = (sigma > 0) & above
    pref = sigma * PB[None, :]
    C2 = np.zeros((len(sub), len(shells)))
    GM = np.zeros((len(sub), len(shells)))
    for k, s in enumerate(shells):
        Jc, bf = J_C1[IT, s], bfr[IT, s]
        use = bf > 0
        c2 = np.where(m_c2, np.where(use[None, :], sigma * bf[None, :],
                                     pref * Jc[None, :]), 0.0).sum(1)
        gm = np.where(sigma > 0, pref * Jc[None, :], 0.0).sum(1)
        c2[~ok] = 0.0
        gm[~ok] = 0.0
        C2[:, k], GM[:, k] = c2, gm
    return dict(E=E, gw=gw, lnum=lnum, ok=ok, C2=C2, GM=GM, n=len(sub))


# ---- model composition: an ion with zero elemental abundance in a shell has NO
# ---- atoms there, so its Gamma ratio is a pure rate-coefficient diagnostic and
# ---- cannot move b_k / n_e / the spectrum.  data/.../abundances.csv
_ab = pd.read_csv(os.path.join(Y.MODEL, "abundances.csv")).set_index("atomic_number")
_ab.columns = _ab.columns.astype(int)
XEL = {int(Z): _ab.loc[Z].values.astype(float) for Z in _ab.index}
print("\n[composition] mass fraction != 0 shell ranges (data/.../abundances.csv):")
for Z, v in XEL.items():
    nz = np.nonzero(v)[0]
    print(f"  Z={Z:3d}: " + (f"shells {nz.min()}..{nz.max()} (n={len(nz)})"
                             if len(nz) else "ZERO IN ALL 50 SHELLS"))
print("  elements NOT in abundances.csv at all (=> zero everywhere): "
      "Mg(12) Al(13) Sc(21) Ti(22) V(23) Cr(24) Mn(25)")

ALL_SH = list(range(50))
rows = []
for Z, ion, nm in CONSUMED + REFONLY:
    r = rates(Z, ion, ALL_SH)
    if r is None:
        continue
    for k, s in enumerate(ALL_SH):
        c2, gm = r["C2"][:, k], r["GM"][:, k]
        x = r["E"] * Y.EV_TO_ERG / (Y.K_B * Te[s])
        w = np.where(x < 500.0, r["gw"] * np.exp(np.clip(-x, -500, 0)), 0.0)
        f = w / w.sum() if w.sum() > 0 else w
        G2, GG = float((f * c2).sum()), float((f * gm).sum())
        d = dict(ion=nm, Z=Z, ion_number=ion,
                 consumed_in_matrix=(Z, ion, nm) in CONSUMED, shell=s,
                 X_elem=float(XEL[Z][s]) if Z in XEL else 0.0,
                 Gamma_C2=G2, Gamma_GEMM=GG,
                 ratio_boltz=(G2 / GG if GG > 0 else np.nan),
                 ratio_ground=(c2[0] / gm[0] if gm[0] > 0 else np.nan))
        for tag, lp in POPS.items():
            sel = lp[(lp.shell == s) & (lp.Z == Z) & (lp.ion == ion)]
            if len(sel) == 0:
                d[f"ratio_pop_{tag}"] = np.nan
                continue
            nk = sel.set_index("level_num").n_k.reindex(r["lnum"]).values
            nk = np.where(np.isfinite(nk) & (nk > 0), nk, 0.0)
            if nk.sum() <= 0:
                d[f"ratio_pop_{tag}"] = np.nan
                continue
            a, b = float((nk * c2).sum()), float((nk * gm).sum())
            d[f"ratio_pop_{tag}"] = a / b if b > 0 else np.nan
            d[f"n_matched_{tag}"] = int(np.isfinite(
                sel.set_index("level_num").n_k.reindex(r["lnum"]).values).sum())
        rows.append(d)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(Y.OUT, f"y2_prereg_allshells{SUF}.csv"), index=False)

pd.set_option("display.width", 260)
SH6 = [0, 8, 20, 30, 45, 49]
print("\n=== ratio C2/GEMM at iter 11, three weightings, 6 report shells ===")
for col, lab in (("ratio_ground", "GROUND"), ("ratio_pop_raw", "POP(resolve_raw)"),
                 ("ratio_pop_ema", "POP(resolve_ema)"), ("ratio_boltz", "BOLTZMANN")):
    if col not in df:
        continue
    print(f"\n-- {lab} --")
    print(df[df.shell.isin(SH6)].pivot(index="ion", columns="shell", values=col)
          .round(3).to_string())

BAND = list(range(20, 36))
print("\n=== P1/P2 registered band s20-35: Si II / Fe II, every shell ===")
sub = df[df.shell.isin(BAND) & df.ion.isin(["Si II", "Fe II"])]
cols = [c for c in ("ratio_boltz", "ratio_pop_raw", "ratio_pop_ema",
                    "ratio_ground") if c in sub]
print(sub.pivot(index="shell", columns="ion", values=cols).round(3).to_string())

# ------------------------------------------------------------- P1..P8 verdicts
def bracket(vals, lo, hi):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if v.size == 0:
        return "N-A", np.nan, np.nan
    mn, mx = v.min(), v.max()
    if mn >= lo and mx <= hi:
        return "WITHIN-RANGE", mn, mx
    return "OUTSIDE-RANGE", mn, mx


out = []


def bnd(name, sel, col, lo, hi, note):
    st, mn, mx = bracket(sel[col], lo, hi)
    xmax = float(sel.X_elem.max()) if len(sel) else np.nan
    out.append(dict(item=name, registered=f"x{lo}-{hi}", observed_min=mn,
                    observed_max=mx, verdict=st,
                    max_X_elem_in_domain=xmax,
                    has_atoms=bool(xmax > 0), note=note))


b = df[df.shell.isin(BAND)]
bnd("P1 Si II s20-35 pop-wtd(Boltz)", b[b.ion == "Si II"], "ratio_boltz",
    1.6, 3.5, "UP")
bnd("P1 Si II s20-35 ground", b[b.ion == "Si II"], "ratio_ground",
    2.0, 6.5, "UP")
bnd("P2 Fe II s20-35 pop-wtd(Boltz)", b[b.ion == "Fe II"], "ratio_boltz",
    1.4, 3.5, "UP")
bnd("P2 Fe II s20-35 ground", b[b.ion == "Fe II"], "ratio_ground",
    1.2, 6.5, "UP")
s30 = df[df.shell == 30]
bnd("P3 Mg II s30", s30[s30.ion == "Mg II"], "ratio_boltz", 1.4, 1.7, "UP")
bnd("P3 Co II s30", s30[s30.ion == "Co II"], "ratio_boltz", 1.4, 1.7, "UP")
s20 = df[df.shell == 20]
bnd("P4 Ni II s20", s20[s20.ion == "Ni II"], "ratio_boltz", 0.75, 0.98, "DOWN")

# P5: |ratio-1| <= 0.035 for every consumed ion with Gamma > 1 s^-1 at s0/s8
p5 = df[df.shell.isin([0, 8]) & df.consumed_in_matrix & (df.Gamma_C2 > 1.0)].copy()
p5["dev"] = (p5.ratio_boltz - 1.0).abs()
p5 = p5.sort_values("dev", ascending=False)
print("\n=== P5 basis: consumed ions with Gamma_C2 > 1 s^-1 at s0 / s8 ===")
print(p5[["ion", "shell", "X_elem", "Gamma_C2", "Gamma_GEMM", "ratio_boltz", "dev"]]
      .round(4).to_string(index=False))
n_viol = int((p5.dev > 0.035).sum())
out.append(dict(item="P5 s0/s8 |ratio-1| <= 0.035 (Gamma>1, as registered)",
                registered="<=0.035", observed_min=float(p5.dev.min()),
                observed_max=float(p5.dev.max()),
                verdict="WITHIN-RANGE" if n_viol == 0 else "OUTSIDE-RANGE",
                max_X_elem_in_domain=float(p5.X_elem.max()), has_atoms=True,
                note=f"{n_viol} of {len(p5)} ion x shell exceed 0.035"))
p5a = p5[p5.X_elem > 0]
n_viol_a = int((p5a.dev > 0.035).sum())
out.append(dict(item="P5 same, restricted to ions WITH atoms (X_elem>0)",
                registered="<=0.035", observed_min=float(p5a.dev.min()),
                observed_max=float(p5a.dev.max()),
                verdict="WITHIN-RANGE" if n_viol_a == 0 else "OUTSIDE-RANGE",
                max_X_elem_in_domain=float(p5a.X_elem.max()), has_atoms=True,
                note=f"{n_viol_a} of {len(p5a)} ion x shell exceed 0.035"))

# P6: the six NULL-ish ions
P6 = ["Ca II", "Cr II", "Mn II", "Ti II", "Sc II", "O I"]
for scope, sel in (("6 report shells", df[df.shell.isin(SH6)]),
                   ("all 50 shells", df)):
    s6 = sel[sel.ion.isin(P6)]
    st, mn, mx = bracket(s6.ratio_boltz, 0.92, 1.15)
    out.append(dict(item=f"P6 Ca/Cr/Mn/Ti/Sc II + O I ({scope})",
                    registered="x0.92-1.15", observed_min=mn, observed_max=mx,
                    verdict=st, max_X_elem_in_domain=float(s6.X_elem.max()),
                    has_atoms=True, note="NULL-ish"))
s6a = df[df.ion.isin(P6) & df.shell.isin(SH6) & (df.X_elem > 0)]
st, mn, mx = bracket(s6a.ratio_boltz, 0.92, 1.15)
out.append(dict(item="P6 same, restricted to ions WITH atoms (= Ca II only)",
                registered="x0.92-1.15", observed_min=mn, observed_max=mx,
                verdict=st, max_X_elem_in_domain=float(s6a.X_elem.max()),
                has_atoms=True, note=f"ions present: {sorted(s6a.ion.unique())}"))

pv = pd.DataFrame(out)
pv.to_csv(os.path.join(Y.OUT, f"y2_prereg_verdicts{SUF}.csv"), index=False)
print("\n=== P1-P6 mechanical bracket verdicts (iter 11, clean) ===")
print(pv.round(4).to_string(index=False))

# P6 offenders
p6 = df[df.ion.isin(P6) & df.shell.isin(SH6)].copy()
p6["dev"] = (p6.ratio_boltz - 1.0).abs()
print("\n  P6 worst offenders on the 6 report shells:")
print(p6.nlargest(8, "dev")[["ion", "shell", "ratio_boltz", "Gamma_C2"]]
      .round(4).to_string(index=False))

print("\n=== P7 reference-only ions: matrix R_bf is structurally absent ===")
print("  (see step9 for the banner / pair-list proof; here only the reference "
      "C2-vs-GEMM sizes)")
print(df[~df.consumed_in_matrix & df.shell.isin(SH6)]
      .pivot(index="ion", columns="shell", values="ratio_boltz").round(3).to_string())
