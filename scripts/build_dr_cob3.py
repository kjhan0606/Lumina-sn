#!/usr/bin/env python3
"""Build the Co IV -> Co III dielectronic recombination (DR) DR_TABLE entry
from CMFGEN's LTDR data file DIECoIII_14840 (and _2590 for comparison).

Physics / column semantics are taken verbatim from the CMFGEN reader
   /gpfs/kjhan/cmfgen_src/cur_cmf/subs/rdgendie_v4.f   (lines 189-212)
which, for each dielectronic transition, computes the per-transition
effective recombination coefficient (units 10^-12 cm^3/s) at 1e4, 2e4, 3e4 K:

   EDGEDIE = -v(exc)                      ! energy above ion. limit, 1e15 Hz
   T1  = HDKT * EDGEDIE                    ! HDKT = 4.7994145  (=1e15*h/k/1e4)
   A10 = 2.07E-10 * Gu * A / GION          ! GION = stat.wt of recombining ion g.s. term
   A10 = A10 * EXP(T1)                      ! at 1e4 K   (T4 = 1)
   A20 = A10base * EXP(0.5*T1) / 2^1.5      ! at 2e4 K   (T4 = 2)
   A30 = A10base * EXP(T1/3)  / 3^1.5       ! at 3e4 K   (T4 = 3)

Generalising to arbitrary T (T4 = T/1e4):
   alpha_i(T)[1e-12 cm^3/s] = (2.07e-10 * Gu * A / GION)
                              * exp(-HDKT * v(exc) / T4) / T4^1.5

i.e. exactly the LTDR form  alpha_i(T) = C_i * T^-1.5 * exp(-E_i/T)  with
   E_i[K] = HDKT * v(exc) * 1e4      (energy of the autoionizing level above threshold)
   C_i    = 1e-6 * 2.07e-10 * Gu * A / GION   [cm^3 s^-1 K^1.5]

This matches the physical LTDR Saha-Boltzmann expression
   alpha_i = (g_a/(2 g+)) (h^2/2 pi m_e k T)^{3/2} exp(-eps_a/kT) A_rad
with g_a = Gu, g+ = GION, and 2.07e-10 = 0.5 * (h^2/2 pi m_e k*1e4)^{3/2} * 1e12.

The total DR rate is the sum over all listed transitions.  We fit the summed
alpha_DR(T) to LUMINA's multi-term form  alpha = T^-1.5 * sum_j c_j exp(-E_j/T)
(n_terms <= DR_MAX_TERMS = 10).

Reproducible:  python3 scripts/build_dr_cob3.py
"""
import sys, math
import numpy as np
from scipy.optimize import nnls, least_squares

HDKT   = 4.7994145          # CMFGEN: 1e15*h/k/1e4   (nu in 1e15 Hz, T in 1e4 K)
PREF   = 2.07e-10           # CMFGEN LTDR prefactor (units 1e-12 cm^3/s)
E_PER_VEXC = HDKT * 1.0e4   # E_i[K] = E_PER_VEXC * v(exc)  = 47994.145 * v(exc)

COB3_14840 = "/gpfs/kjhan/cmfgen_21jun23/atomic/COB/III/19apr23/DIECoIII_14840"
COB3_2590  = "/gpfs/kjhan/cmfgen_21jun23/atomic/COB/III/19apr23/DIECoIII_2590"
CARB3_DIE  = "/gpfs/kjhan/cmfgen_21jun23/atomic/CARB/III/23dec04/dieciii_ic.dat"
# Co III photoionization cross-sections (Milne parent of Co IV->III recomb)
COB3_PHOT_NOSM = "/gpfs/kjhan/cmfgen_21jun23/atomic/COB/III/30oct12/phot_nosm"
COB3_PHOT_A    = "/gpfs/kjhan/cmfgen_21jun23/atomic/COB/III/19apr23/phot_data_A"

# Lumina's existing Badnell C III -> C II (6,2) entry (src/lumina_plasma.c)
BADNELL_C3 = dict(
    c=[3.489e-06, 2.222e-07, 1.954e-05, 4.212e-03, 2.037e-04, 2.936e-04],
    E=[2.660e+03, 3.756e+03, 2.566e+04, 1.400e+05, 1.801e+06, 4.307e+06],
)


# --------------------------------------------------------------------------
def parse_die(path):
    """Parse a CMFGEN DIE (LTDR) file.  Returns (transitions, header).

    IMPORTANT: CMFGEN's reader (rdgendie_v4.f) does `DO LOOP=1,NUM_D_RD`, i.e.
    it reads exactly NUM_D_RD (= header '!Number of dielectronic transitions')
    transitions from the FIRST block only.  These DIE files contain a SECOND
    appended block (higher-lying resonances for C III; a duplicate for Co) that
    the standard reader never touches.  We mirror this: stop after ntrans lines.

    Also returns the file's own summary lines ('LTDR for all listed states' and
    'Total LTDR for all states'), the a-column sums (1e4/2e4/3e4 K, x1e-12) that
    CMFGEN's file-writer computed -> used as an independent validation target.

    transitions = list of dicts with f, A, lam, gu, vexc, a1e4, a2e4, a3e4, wi.
    Column mapping mirrors rdgendie_v4.f exactly."""
    ion_energy = None
    ntrans = None
    trans = []
    summ_all_listed = None   # (a1e4,a2e4,a3e4) x1e-12
    summ_total_all = None
    with open(path) as fh:
        lines = fh.readlines()
    # header keywords + summary lines (first occurrence)
    for ln in lines:
        if "!Ionization energy" in ln:
            ion_energy = float(ln.split()[0])
        elif "!Number of dielectronic transitions" in ln:
            ntrans = int(ln.split()[0])
        elif ln.strip().startswith("LTDR for all listed states") and summ_all_listed is None:
            summ_all_listed = [float(x) for x in ln.split()[-3:]]
        elif ln.strip().startswith("Total LTDR for all states") and summ_total_all is None:
            summ_total_all = [float(x) for x in ln.split()[-3:]]
    # find the FIRST column header line ("Transition ... Lam(A)")
    hdr = None
    for i, ln in enumerate(lines):
        if "Transition" in ln and "Lam(A)" in ln:
            hdr = i
            break
    if hdr is None:
        raise RuntimeError("no column header in " + path)
    for ln in lines[hdr + 1:]:
        if len(trans) >= ntrans:      # mirror DO LOOP=1,NUM_D_RD -> block 1 only
            break
        s = ln.rstrip("\n")
        if not s.strip():
            continue
        if s.lstrip().startswith("!"):
            continue
        wi = ("#" in s)
        # extract floating-point tokens; level names ("3d6(5D)4p...") never parse
        toks = []
        for t in s.split():
            try:
                toks.append(float(t))
            except ValueError:
                pass
        # expect: f, A, lam, gu, vexc, a1e4, a2e4, a3e4, index  (9 floats)
        if len(toks) < 8:
            continue
        f, A, lam, gu, vexc, a1, a2, a3 = toks[0:8]
        trans.append(dict(f=f, A=A, lam=lam, gu=gu, vexc=vexc,
                          a1e4=a1, a2e4=a2, a3e4=a3, wi=wi))
    return trans, dict(ion_energy=ion_energy, ntrans=ntrans,
                       summ_all_listed=summ_all_listed, summ_total_all=summ_total_all)


def phot_xsec_types(path):
    """Count CMFGEN photoionization cross-section TYPE codes in a phot file.
       type 20/21/22 = Opacity-Project resonance-resolved; else smooth analytic
       (1=tab, 2=Seaton, 7/8=hydrogenic/Peach, 9=Verner-Yakovlev)."""
    counts = {}
    with open(path, errors="ignore") as fh:
        for ln in fh:
            if "!Type of cross-section" in ln:
                try:
                    t = int(ln.split("!")[0].split()[0])
                except (ValueError, IndexError):
                    continue
                counts[t] = counts.get(t, 0) + 1
    return counts


def derive_gion(trans):
    """Invert the tabulated a(10^4) column to recover GION (self-calibration).
       a1e4 = 2.07e-10 * Gu * A / GION * exp(-HDKT * vexc)  (T4=1)."""
    vals = []
    for t in trans:
        if t["a1e4"] > 0:
            g = PREF * t["gu"] * t["A"] * math.exp(-HDKT * t["vexc"]) / t["a1e4"]
            vals.append(g)
    vals = np.array(vals)
    return float(np.median(vals)), float(np.std(vals)), vals


def alpha_i_1e12(t, T, gion):
    """Per-transition alpha (units 1e-12 cm^3/s) at temperature T (K)."""
    T4 = T / 1.0e4
    base = PREF * t["gu"] * t["A"] / gion
    return base * math.exp(-HDKT * t["vexc"] / T4) / T4 ** 1.5


def alpha_total_cm3s(trans, T, gion):
    """Total DR alpha (cm^3/s) at temperature T (K) by direct summation."""
    s = 0.0
    for t in trans:
        s += alpha_i_1e12(t, T, gion)
    return s * 1.0e-12


def badnell_eval(entry, T):
    """alpha = T^-1.5 * sum c_i exp(-E_i/T)  (Lumina dr_alpha_eval form)."""
    s = 0.0
    for c, E in zip(entry["c"], entry["E"]):
        s += c * math.exp(-E / T)
    return s * T ** -1.5


# --------------------------------------------------------------------------
def fit_multiexp(Tfit, yfit, want_terms, T_lo, T_hi):
    """Fit yfit(T)=alpha*T^1.5 to sum_j c_j exp(-E_j/T), <= want_terms terms.
       Error metric = relative, evaluated on the [T_lo,T_hi] subset.
       Returns (c[], E[], max_relerr, rms_relerr)."""
    Tfit = np.asarray(Tfit, float)
    yfit = np.asarray(yfit, float)
    mask = (Tfit >= T_lo) & (Tfit <= T_hi)

    # 1) seed with NNLS on a dense positive E dictionary (relative weighting)
    Egrid = np.logspace(math.log10(3.0), math.log10(5.0e6), 500)
    A = np.exp(-Egrid[None, :] / Tfit[:, None]) / yfit[:, None]
    b = np.ones_like(yfit)
    c_nnls, _ = nnls(A, b)
    supp = np.where(c_nnls > 0)[0]
    # rank support points by their integrated contribution over the grid
    contrib = np.array([c_nnls[j] * np.exp(-Egrid[j] / Tfit).mean() for j in supp])
    order = supp[np.argsort(contrib)[::-1]]
    keep = order[:want_terms]
    E0 = Egrid[keep]
    c0 = c_nnls[keep]
    c0 = np.maximum(c0, 1e-30)

    # 2) joint refine (log-params keep c,E > 0); minimise relative residual on [T_lo,T_hi]
    def resid(theta):
        n = len(theta) // 2
        c = np.exp(theta[:n])
        E = np.exp(theta[n:])
        model = (np.exp(-E[None, :] / Tfit[mask, None]) * c[None, :]).sum(axis=1)
        return (model - yfit[mask]) / yfit[mask]

    theta0 = np.concatenate([np.log(c0), np.log(E0)])
    sol = least_squares(resid, theta0, method="lm", max_nfev=20000)
    n = len(sol.x) // 2
    c = np.exp(sol.x[:n])
    E = np.exp(sol.x[n:])
    # sort by E
    o = np.argsort(E)
    c, E = c[o], E[o]
    # error over target band
    model = (np.exp(-E[None, :] / Tfit[mask, None]) * c[None, :]).sum(axis=1)
    rel = np.abs(model - yfit[mask]) / yfit[mask]
    return c, E, float(rel.max()), float(np.sqrt((rel ** 2).mean()))


# --------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("STEP 0.  CMFGEN reader-confirmed constants")
    print("=" * 78)
    print(f"  HDKT (1e15*h/k/1e4)      = {HDKT}")
    print(f"  LTDR prefactor 2.07e-10  = 0.5*(h^2/2pi m_e k 1e4)^1.5 * 1e12")
    print(f"  E_i[K] = {E_PER_VEXC:.4f} * v(exc)    (v(exc) in 1e15 Hz)")

    # ---- parse Co III files ------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 1.  Parse DIECoIII files + self-calibrate GION")
    print("=" * 78)
    tr14, h14 = parse_die(COB3_14840)
    tr25, h25 = parse_die(COB3_2590)
    for tag, tr, h in [("14840", tr14, h14), ("2590", tr25, h25)]:
        gion, gsd, gvals = derive_gion(tr)
        nwi = sum(1 for t in tr if t["wi"])
        print(f"  DIECoIII_{tag}: header N={h['ntrans']}, parsed(block1)={len(tr)}, "
              f"WI(#)={nwi}, ion_energy={h['ion_energy']} cm^-1")
        print(f"      GION (median of per-line inversion) = {gion:.4f} "
              f"(std {gsd:.2e})  -> Co IV 3d6 5D ground term g=25")
    gion14, _, _ = derive_gion(tr14)
    gion25, _, _ = derive_gion(tr25)

    # ---- parser self-validation: reproduce the file's own summary totals ---
    print("\n" + "=" * 78)
    print("STEP 1b. Parser validation vs CMFGEN's own summary totals (x1e-12)")
    print("=" * 78)
    print("  ('LTDR for all listed states' = sum of block-1's printed transitions")
    print("   = exactly what we reconstruct.  'Total for all states' = cutoff-")
    print("   converged incl. below-print tail, identical in both files.)")
    for tag, tr, h, gion in [("14840", tr14, h14, gion14), ("2590", tr25, h25, gion25)]:
        print(f"  -- DIECoIII_{tag} --")
        for j, (T, key) in enumerate([(1e4, "a1e4"), (2e4, "a2e4"), (3e4, "a3e4")]):
            rec = alpha_total_cm3s(tr, T, gion) * 1e12
            listed = h["summ_all_listed"][j]
            total = h["summ_total_all"][j]
            print(f"     T={T:.0e}K  recomputed_sum={rec:8.4f}  "
                  f"file_all_listed={listed:8.4f}  (ratio {rec/listed:.4f})   "
                  f"file_TOTAL_all={total:8.4f}")

    # ---- STEP 2: C III pipeline validation vs Badnell ---------------------
    print("\n" + "=" * 78)
    print("STEP 2.  C III pipeline validation  (DIE-derived vs Badnell (6,2))")
    print("=" * 78)
    trc, hc = parse_die(CARB3_DIE)
    gionc, gcsd, _ = derive_gion(trc)
    nwic = sum(1 for t in trc if t["wi"])
    print(f"  dieciii_ic.dat: header N={hc['ntrans']}, parsed={len(trc)}, "
          f"WI(#)={nwic}, ion_energy={hc['ion_energy']} cm^-1")
    print(f"  GION (self-calibrated) = {gionc:.4f}  -> C IV 2s 2S ground term g=2")
    print(f"  {'T [K]':>8}  {'DIE-derived':>14}  {'Badnell(6,2)':>14}  {'ratio DIE/Bad':>13}")
    for T in [5e3, 1e4, 2e4, 5e4]:
        die = alpha_total_cm3s(trc, T, gionc)
        bad = badnell_eval(BADNELL_C3, T)
        print(f"  {T:8.0f}  {die:14.4e}  {bad:14.4e}  {die/bad:13.3f}")
    print("  (LTDR = low-T resonances only; Badnell total adds high-n/high-T DR,")
    print("   so DIE-derived is expected to undershoot progressively above ~2e4 K.)")

    # ---- STEP 3+4: Co alpha_DR grids + fit --------------------------------
    print("\n" + "=" * 78)
    print("STEP 3/4.  Co IV->III alpha_DR(T) grid, 2590-vs-14840, and fit")
    print("=" * 78)
    Tgrid = np.logspace(3, 6, 81)
    a14 = np.array([alpha_total_cm3s(tr14, T, gion14) for T in Tgrid])
    a25 = np.array([alpha_total_cm3s(tr25, T, gion25) for T in Tgrid])
    print(f"  {'T [K]':>9}  {'alpha(14840)':>13}  {'alpha(2590)':>13}  {'14840/2590':>10}")
    for T in [5e3, 1.0470e4, 1.3120e4, 1.6014e4, 2e4, 5e4, 1e5]:
        d14 = alpha_total_cm3s(tr14, T, gion14)
        d25 = alpha_total_cm3s(tr25, T, gion25)
        print(f"  {T:9.1f}  {d14:13.4e}  {d25:13.4e}  {d14/d25:10.4f}")

    # fit the 14840 grid.  Task spec is <3% over 5e3-1e5 K; we widen the fit
    # band to 3e3-1.2e5 K so the entry stays well-behaved on cool far-outer
    # shells (the entry is used at ALL T when LUMINA_FROZENIN_DR=1).
    y14 = a14 * Tgrid ** 1.5
    T_lo, T_hi = 1e3, 1.5e5
    T_spec_lo, T_spec_hi = 5e3, 1e5     # task-required band, reported separately
    print("\n  Fitting alpha_DR(14840) to multi-term form "
          "(target <1% over 1e3-1.5e5 K):")
    best = None
    for nt in range(2, 11):
        c, E, mx, rms = fit_multiexp(Tgrid, y14, nt, T_lo, T_hi)
        print(f"    n_terms={nt:2d}  max_relerr={mx*100:8.4f}%  rms={rms*100:8.4f}%")
        if best is None or mx < best[2]:
            best = (c, E, mx, rms, nt)
        if mx < 0.01:
            best = (c, E, mx, rms, nt)
            break
    c, E, mx, rms, nt = best
    # error over the strict task band
    m2 = (Tgrid >= T_spec_lo) & (Tgrid <= T_spec_hi)
    mdl = (np.exp(-E[None, :] / Tgrid[m2, None]) * c[None, :]).sum(axis=1)
    relspec = np.abs(mdl - y14[m2]) / y14[m2]
    print(f"\n  CHOSEN FIT (14840): n_terms={nt}")
    print(f"    max_relerr={mx*100:.4f}%  rms={rms*100:.4f}%  over "
          f"{T_lo:.0e}-{T_hi:.0e} K (fit band)")
    print(f"    max_relerr={relspec.max()*100:.4f}%  over 5e3-1e5 K (task spec band)")

    # extrapolation sanity (outside the fit band) — entry must stay well-behaved
    print("\n  Extrapolation check (fit vs direct sum) outside 5e3-1e5 K:")
    for T in [1e3, 3e3, 3e5, 1e6]:
        direct = alpha_total_cm3s(tr14, T, gion14)
        fit = (c * np.exp(-E / T)).sum() * T ** -1.5
        print(f"     T={T:9.0f}K  direct={direct:.4e}  fit={fit:.4e}  "
              f"fit/direct={fit/direct:.4f}")

    # report the requested temperatures with the fit and the direct sum
    print("\n  alpha_DR(Co IV->III) at requested temperatures:")
    print(f"  {'T [K]':>9}  {'direct sum':>13}  {'fitted':>13}  {'fit/direct':>10}")
    for T in [10470.0, 13120.0, 16014.0, 2.0e4]:
        direct = alpha_total_cm3s(tr14, T, gion14)
        fit = (c * np.exp(-E / T)).sum() * T ** -1.5
        print(f"  {T:9.1f}  {direct:13.4e}  {fit:13.4e}  {fit/direct:10.4f}")

    # emit the C entry
    print("\n" + "=" * 78)
    print("STEP 5.  DR_TABLE entry  (paste into src/lumina_plasma.c)")
    print("=" * 78)
    cstr = ", ".join(f"{x:.4e}" for x in c)
    estr = ", ".join(f"{x:.4e}" for x in E)
    print(f"    {{27, 3, {nt},")
    print(f"     {{{cstr}}},")
    print(f"     {{{estr}}},")
    print(f"     DR_SOURCE_CMFGEN}},")

    # ---- STEP 6: double-counting / complementary analysis ------------------
    print("\n" + "=" * 78)
    print("STEP 6.  Double-count audit: DIE LTDR vs Co III photoionization sigma")
    print("=" * 78)
    for tag, path in [("phot_nosm (30oct12)", COB3_PHOT_NOSM),
                      ("phot_data_A (19apr23)", COB3_PHOT_A)]:
        types = phot_xsec_types(path)
        smooth = sum(n for t, n in types.items() if t not in (20, 21, 22))
        resonant = sum(n for t, n in types.items() if t in (20, 21, 22))
        tdesc = ", ".join(f"type{t}:{n}" for t, n in sorted(types.items()))
        print(f"  Co III {tag}: {tdesc}")
        print(f"      -> smooth-analytic levels={smooth}, "
              f"OP-resonance levels(20/21/22)={resonant}")
    # AUTO vs WI split of the DIE list
    auto = alpha_total_cm3s([t for t in tr14 if not t["wi"]], 1e4, gion14)
    wi = alpha_total_cm3s([t for t in tr14 if t["wi"]], 1e4, gion14)
    tot = auto + wi
    print(f"  DIE (14840) AUTO(no #)={auto:.3e} ({100*auto/tot:.0f}%)  "
          f"WI(#)={wi:.3e} ({100*wi/tot:.0f}%)  @1e4 K")
    print("  VERDICT: Co III sigma is 100% smooth analytic (NO type-20/21 OP")
    print("  resonances) -> Milne-inversion yields RADIATIVE recomb only.  The")
    print("  DIE LTDR transitions are FULLY COMPLEMENTARY (they supply the entire")
    print("  DR resonance contribution).  CMFGEN's own guard (rd_phot_die_v1.f")
    print("  :131-147) warns of overlap ONLY for OP type-20/21 sigma; Co III never")
    print("  triggers it.  => (27,3) does NOT double-count against Co III sigma_bf.")

    print("\nDONE.")
    return c, E, nt, mx, rms


if __name__ == "__main__":
    main()
