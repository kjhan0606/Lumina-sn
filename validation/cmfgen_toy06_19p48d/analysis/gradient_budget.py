#!/usr/bin/env python3
"""gradient_budget.py -- Fable's gradient-budget verdict (OFFLINE).

Decompose the CMFGEN deep->photosphere Fe IV/III ionization gradient into
FIELD / T_e / n_e / residual axes, and measure the SAME axes for Lumina's
B-run (all-level Gph, LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0), so that
  "the missing X dex is carried by AXIS"
is read straight off the table.

Framework (photoionization equilibrium, exactly additive in dex):
    log10[ n(IV)/n(III) ] = log10(Gph) - log10(n_e) - log10(alpha(T_e))
Gradient s0->s8 (deep 4264 -> photosphere 10088 km/s; +dex = deep more ionized):
    D[log ratio] = D[log Gph] - D[log n_e] - D[log alpha]
Field axis   = D[log Gph_gnd]           (ground-only, PURE field, no T_e in weights)
T_e-in-Gph   = D[log Gph_boltz/Gph_gnd] (Boltzmann all-level weights, the run's scheme)
Recomb axis  = -D[log alpha(T_e)]       (Milne recomb coeff)
n_e axis     = -D[log n_e]

Yardstick provenance:
  Gph field source for the B-run (config-verified): src/lumina_plasma.c:5261-5269 /
  :5304-5312 / :5359-5365 blend  J = alpha*g_photoion_mc_J + (1-alpha)*nlte->J_nu,
  alpha=LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0 -> J == g_photoion_mc_J == the MC shadow
  field == 'mc_J' column of lumina_coevolve_field.csv (src/lumina_cuda.cu:5188 memcpy,
  :5324 dump). g_photoion_mc_J is the TRANSPORTED MC field, NOT a dilute-Planck of the
  pinned T_rad (T_rad=10470.093 in all 50 shells, uniq=1) -> its flatness is
  TRANSPORT-REAL, not definitional.  db_photoion_calc.field() reads exactly this
  column (mc_J if >0 else cs_J; floor=1e-30>0 so mc_J is used run-faithfully).
"""
import os, sys, math
import numpy as np

REPO = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
os.environ['LUMINA_REF_DIR']          = f'{REPO}/data/tardis_reference_toy06_19p48d'
os.environ['LUMINA_SIGMA_BIN']        = f'{REPO}/data/tardis_reference_toy06_19p48d/cmfgen_sigma_bf.bin'
os.environ['LUMINA_SIGMA_COIII_PATCH']= f'{REPO}/data/coiii_real_sigma_patch.npz'
sys.path.insert(0, f'{REPO}/scripts')
import db_photoion_calc as dbp

H, KB, C, ME, EV, PI, KB_EV = dbp.H, dbp.KB, dbp.C, dbp.ME, dbp.EV, dbp.PI, dbp.KB_EV
nu_c, dnu, SIG, flags = dbp.nu_c, dbp.dnu, dbp.SIG, dbp.flags
levZ, levI, levN, levE, levG, CHI = dbp.levZ, dbp.levI, dbp.levN, dbp.levE, dbp.levG, dbp.CHI

CLIGHT_A = 2.99792458e18   # A/s  (lam_A = CLIGHT_A / nu_Hz)
STD = f'{REPO}/data/standart_data1/toy06'
JNU4 = '/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4'
LUMDIR = f'{REPO}/logs/coevolve_consume_a10_kx_gphall'

# forming shells: Lumina shell index -> mid velocity (geometry.csv)
SHELLS = [0, 2, 4, 5, 6, 7, 8, 9, 10]
VEL    = [4264, 5720, 7176, 7904, 8632, 9360, 10088, 10816, 11544]
I_S0, I_S8 = 0, 6   # indices in SHELLS/VEL for s0(4264) and s8(10088)

# ---------------------------------------------------------------- CMFGEN parsers
def cmfgen_block(path, t=19.480):
    lines = open(path).read().splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith('#TIME:') and abs(float(ln.split()[1]) - t) < 1e-3:
            start = i; break
    rows = []; j = start + 1
    while j < len(lines):
        s = lines[j].strip()
        if s.startswith('#TIME'): break
        if s and not s.startswith('#'):
            try: rows.append([float(x) for x in s.split()])
            except ValueError: pass
        j += 1
    return np.array(rows)

# ---------------------------------------------------------------- EDDFACTOR reader
def read_info(info):
    v = open(info).read().splitlines()[2].split()
    return dict(ND=int(v[0]), RECL=int(v[1]), WORD=int(v[2]), little=(v[5] == 'T'))
def read_edd(edd):
    info = read_info(edd + '_INFO'); ND = info['ND']; nwr = info['RECL'] // info['WORD']
    dt = '<f8' if info['little'] else '>f8'
    raw = np.fromfile(edd, dtype=dt); raw = raw[:(raw.size // nwr) * nwr].reshape(-1, nwr)
    data = raw[14:]
    good = np.isfinite(data[:, :ND]).all(axis=1) & (data[:, ND] > 0)
    J = data[good, :ND]; FL = data[good, ND]; nu = FL * 1e15
    o = np.argsort(nu); return J[o], nu[o], ND, raw[4, 0]
def rvtj_block(text, label, ND):
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == label:
            vals = []; j = i + 1
            while j < len(lines) and len(vals) < ND:
                try: vals += [float(t) for t in lines[j].split()]
                except ValueError: break
                j += 1
            return np.array(vals[:ND])
    raise KeyError(label)
def J_on_sigma_grid(Jdepth, nu_cmf):
    ln = np.log(nu_cmf); lj = np.log(np.where(Jdepth > 0, Jdepth, 1e-300))
    out = np.exp(np.interp(np.log(nu_c), ln, lj, left=-700, right=-700))
    out[(nu_c < nu_cmf.min()) | (nu_c > nu_cmf.max())] = 0.0
    return out

# ---------------------------------------------------------------- physics kernel
def gph_alpha(J, Te, ne, Z, ion):
    """Return (Gph_gnd, Gph_boltz, alpha, ratio_pred=n(ion+1)/n(ion)) folding field J."""
    chi0 = CHI[(Z, ion)]; kT = KB * Te
    idx  = np.where((levZ == Z) & (levI == ion))[0]
    x    = levE[idx] / (KB_EV * Te)
    U    = float(np.sum(np.where(x < 50, levG[idx] * np.exp(-np.minimum(x, 50)), 0.0)))
    idxu = np.where((levZ == Z) & (levI == ion + 1))[0]
    xu   = levE[idxu] / (KB_EV * Te)
    Uu   = float(np.sum(np.where(xu < 50, levG[idxu] * np.exp(-np.minimum(xu, 50)), 0.0)))
    if Uu < 1: Uu = max(1.0, levG[idxu[0]] if len(idxu) else 1.0)
    lam3 = (H * H / (2 * PI * ME * KB * Te)) ** 1.5
    G_gnd = G_b = alpha = 0.0
    for gl in idx:
        Rb, chi_l = dbp.R_planck(gl, Te, chi0)
        if chi_l > 0 and flags[gl]:
            alpha += Rb * lam3 * levG[gl] / (2 * Uu) * math.exp(min(chi_l / kT, 300))
        R, _ = dbp.R_of_level(gl, J, chi0)
        if R <= 0: continue
        xl = levE[gl] / (KB_EV * Te)
        if xl >= 50: continue
        pb = levG[gl] * math.exp(-xl) / U
        if levN[gl] == 0: G_gnd += R
        G_b += pb * R
    ratio = G_b / (ne * alpha) if alpha > 0 else float('nan')
    return G_gnd, G_b, alpha, ratio

def saha_ratio(Te, ne, Z, ion):
    """LTE Saha n(ion+1)/n(ion): 2 Uu/U / lam3 * exp(-chi0/kT) / ne."""
    chi0 = CHI[(Z, ion)]; kT = KB * Te
    idx  = np.where((levZ == Z) & (levI == ion))[0]
    x    = levE[idx] / (KB_EV * Te)
    U    = float(np.sum(np.where(x < 50, levG[idx] * np.exp(-np.minimum(x, 50)), 0.0)))
    idxu = np.where((levZ == Z) & (levI == ion + 1))[0]
    xu   = levE[idxu] / (KB_EV * Te)
    Uu   = float(np.sum(np.where(xu < 50, levG[idxu] * np.exp(-np.minimum(xu, 50)), 0.0)))
    if Uu < 1: Uu = max(1.0, levG[idxu[0]] if len(idxu) else 1.0)
    lam3 = (H * H / (2 * PI * ME * KB * Te)) ** 1.5
    return 2.0 * Uu / U / lam3 * math.exp(-chi0 * EV / kT) / ne

# ---------------------------------------------------------------- band average
def band_avg(lam, J, lo, hi):
    m = (lam >= lo) & (lam <= hi); v = J[m]
    pos = v[v > 0]
    gm = float(np.exp(np.mean(np.log(pos)))) if pos.size else 0.0
    am = float(np.mean(v)) if v.size else 0.0
    nfloor = int(np.sum(v <= 1e-29))
    return gm, am, v.size, nfloor

# ================================================================= LOAD DATA
print("=== loading CMFGEN converged state (published phys/ionfrac @19.480) ===")
ph = cmfgen_block(f'{STD}/phys_toy06_cmfgen.txt')       # vel,temp,rho,ne,natom
fe = cmfgen_block(f'{STD}/ionfrac_fe_toy06_cmfgen.txt') # vel,fe0..fe5
co = cmfgen_block(f'{STD}/ionfrac_co_toy06_cmfgen.txt') # vel,co0..
vph = ph[:, 0]; Tph = ph[:, 1]; neph = ph[:, 3]
vfe = fe[:, 0]
def cmf_at(v, arr, col):
    d = int(np.argmin(np.abs(arr[:, 0] - v))); return arr[d, col]
def cmf_Tne(v):
    d = int(np.argmin(np.abs(vph - v))); return Tph[d], neph[d]

print("=== loading CMFGEN self-run field (jnu4 EDDFACTOR) ===")
Jc, nuc_cmf, ND, finrec = read_edd(f'{JNU4}/EDDFACTOR')
rt = open(f'{JNU4}/RVTJ').read()
Vc = rvtj_block(rt, 'Velocity (km/s)', ND)
print(f"  EDDFACTOR ND={ND} nfreq={Jc.shape[0]} FINISH_REC={finrec}  V=[{Vc[0]:.0f}..{Vc[-1]:.0f}]")
lam_cmf = CLIGHT_A / nuc_cmf

print("=== loading Lumina B-run (mc_J field, plasma, ion pops) ===")
import csv
# plasma
Te_L = {}; ne_L = {}; Trad_L = {}
for r in csv.DictReader(open(f'{LUMDIR}/lumina_plasma_state.csv')):
    s = int(r['shell_id']); Te_L[s] = float(r['T_e']); ne_L[s] = float(r['n_e']); Trad_L[s] = float(r['T_rad'])
# field (mc_J per shell on the sigma/nlte grid; both are 1000-bin, identical numin/numax)
def lumina_field(shell):
    return dbp.field(LUMDIR, shell)     # mc_J if>0 else cs_J (run-faithful, alpha=1.0)
def lumina_field_lam_J(shell):
    lam = []; Jm = []; Jc_ = []
    for r in csv.DictReader(open(f'{LUMDIR}/lumina_coevolve_field.csv')):
        if int(r['shell']) != shell: continue
        lam.append(float(r['wavelength_A'])); Jm.append(float(r['mc_J'])); Jc_.append(float(r['cs_J']))
    return np.array(lam), np.array(Jm), np.array(Jc_)
# ion pops
pops_L = {}
for r in csv.DictReader(open(f'{LUMDIR}/lumina_ion_pops.csv')):
    pops_L[(int(r['shell_id']), int(r['Z']), int(r['stage']))] = float(r['n_ion'])
def lumina_ratio(shell, Z):
    lo = pops_L.get((shell, Z, 2), 0.0); hi = pops_L.get((shell, Z, 3), 0.0)
    return hi / lo if lo > 0 else float('nan')

# T_rad pin check
uniqTrad = sorted(set(round(v, 3) for v in Trad_L.values()))
print(f"  Lumina T_rad uniq values across 50 shells: {uniqTrad}  (pinned => field flatness NOT from T_rad if driven by mc_J)")

# ================================================================= PER-SHELL TABLE
rows = []
for si, (s, v) in enumerate(zip(SHELLS, VEL)):
    # --- CMFGEN ---
    Tc, nec = cmf_Tne(v)
    d = int(np.argmin(np.abs(Vc - v)))
    Jc_grid = J_on_sigma_grid(Jc[:, d], nuc_cmf)
    fgnd_c, fbz_c, alp_c, rat_c = gph_alpha(Jc_grid, Tc, nec, 26, 2)
    cgnd_c, cbz_c, calp_c, crat_c = gph_alpha(Jc_grid, Tc, nec, 27, 2)
    feIII = cmf_at(v, fe, 3); feIV = cmf_at(v, fe, 4)
    ratio_fe_cmf = feIV / feIII
    coIII = cmf_at(v, co, 3); coIV = cmf_at(v, co, 4)
    ratio_co_cmf = coIV / coIII if coIII > 0 else float('nan')
    saha_c = saha_ratio(Tc, nec, 26, 2)
    # CMFGEN self-run field band avgs
    gm918_c, am918_c, _, _ = band_avg(lam_cmf, Jc[:, d], 918, 1290)
    gm300_c, am300_c, n300_c, fl300_c = band_avg(lam_cmf, Jc[:, d], 300, 450)
    # --- Lumina ---
    TL = Te_L[s]; neL = ne_L[s]
    JL = lumina_field(s)
    fgnd_L, fbz_L, alp_L, rat_L = gph_alpha(JL, TL, neL, 26, 2)
    cgnd_L, cbz_L, calp_L, crat_L = gph_alpha(JL, TL, neL, 27, 2)
    ratio_fe_L = lumina_ratio(s, 26); ratio_co_L = lumina_ratio(s, 27)
    saha_L = saha_ratio(TL, neL, 26, 2)
    lamL, JmL, JcL = lumina_field_lam_J(s)
    gm918_L, am918_L, _, _ = band_avg(lamL, JmL, 918, 1290)
    gm300_L, am300_L, n300_L, fl300_L = band_avg(lamL, JmL, 300, 450)
    gm300_Lcs, am300_Lcs, _, _ = band_avg(lamL, JcL, 300, 450)
    gm918_Lcs, am918_Lcs, _, _ = band_avg(lamL, JcL, 918, 1290)
    rows.append(dict(
        s=s, v=v,
        Tc=Tc, nec=nec, feIII=feIII, feIV=feIV, ratio_fe_cmf=ratio_fe_cmf,
        ratio_co_cmf=ratio_co_cmf, saha_c=saha_c,
        Gg_c=fgnd_c, Gb_c=fbz_c, alp_c=alp_c, rat_c=rat_c,
        cGg_c=cgnd_c, cGb_c=cbz_c, calp_c=calp_c,
        gm918_c=gm918_c, am918_c=am918_c, gm300_c=gm300_c, am300_c=am300_c,
        TL=TL, neL=neL, ratio_fe_L=ratio_fe_L, ratio_co_L=ratio_co_L, saha_L=saha_L,
        Gg_L=fgnd_L, Gb_L=fbz_L, alp_L=alp_L, rat_L=rat_L,
        cGg_L=cgnd_L, cGb_L=cbz_L, calp_L=calp_L,
        gm918_L=gm918_L, am918_L=am918_L, gm300_L=gm300_L, am300_L=am300_L, fl300_L=fl300_L,
        gm918_Lcs=gm918_Lcs, gm300_Lcs=gm300_Lcs,
    ))

def L(x):
    try: return math.log10(x)
    except (ValueError, ZeroDivisionError): return float('nan')
def grad(key, a=I_S0, b=I_S8):
    return L(rows[a][key]) - L(rows[b][key])

# ================================================================= PRINT PER-SHELL
print("\n" + "=" * 118)
print("PER-SHELL (forming shells). CMFGEN=converged phys/ionfrac + self-run J. Lumina=B-run all-level, mc_J field.")
print("=" * 118)
hdr = (f"{'sh':>3}{'v':>7} | {'Te_C':>6}{'Te_L':>6} | {'FeIV/III_C':>11}{'FeIV/III_L':>11} | "
       f"{'Gg_C':>9}{'Gg_L':>9} | {'Gb_C':>9}{'Gb_L':>9} | {'J918_C':>9}{'J918_L':>9}")
print(hdr)
for r in rows:
    print(f"{r['s']:>3}{r['v']:>7} | {r['Tc']:>6.0f}{r['TL']:>6.0f} | "
          f"{r['ratio_fe_cmf']:>11.3e}{r['ratio_fe_L']:>11.3e} | "
          f"{r['Gg_c']:>9.2e}{r['Gg_L']:>9.2e} | {r['Gb_c']:>9.2e}{r['Gb_L']:>9.2e} | "
          f"{r['gm918_c']:>9.2e}{r['gm918_L']:>9.2e}")

# ================================================================= BUDGET (s0->s8)
print("\n" + "=" * 118)
print("GRADIENT BUDGET  s0(4264) -> s8(10088 km/s)   [+dex = deep more ionized]")
print("Identity:  Dlog[FeIV/III] = Dlog(Gph_boltz) - Dlog(n_e) - Dlog(alpha)")
print("=" * 118)

def report(tag, ratio_key, Gg, Gb, alp, ne, saha_key, ratio_actual_grad):
    dGg  = grad(Gg); dGb = grad(Gb); dAlp = grad(alp); dNe = grad(ne)
    dSaha = grad(saha_key)
    pred = dGb - dNe - dAlp
    resid = ratio_actual_grad - pred
    print(f"\n--- {tag} ---")
    print(f"  TOTAL  Dlog[FeIV/III]  (measured, ionfrac/ion_pops) = {ratio_actual_grad:+.2f} dex")
    print(f"  FIELD  Dlog(Gph_gnd)   (pure field, ground-only)    = {dGg:+.2f} dex")
    print(f"  +wts   Dlog(Gph_boltz/Gph_gnd) (T_e Boltzmann wts)  = {dGb - dGg:+.2f} dex")
    print(f"  =Gph   Dlog(Gph_boltz) (field-folded, run scheme)   = {dGb:+.2f} dex")
    print(f"  n_e    -Dlog(n_e)                                   = {-dNe:+.2f} dex")
    print(f"  recomb -Dlog(alpha(T_e))                            = {-dAlp:+.2f} dex")
    print(f"  ----------------------------------------------------------------")
    print(f"  PREDICTED  Gph - n_e - alpha                        = {pred:+.2f} dex")
    print(f"  RESIDUAL   (TOTAL - PREDICTED, closure/NLTE)        = {resid:+.2f} dex")
    print(f"  [cross-check]  Saha-T_e-alone (LTE, own n_e)        = {dSaha:+.2f} dex")
    return dict(total=ratio_actual_grad, field=dGg, wts=dGb - dGg, gph=dGb,
                ne=-dNe, recomb=-dAlp, pred=pred, resid=resid, saha=dSaha)

grad_fe_cmf = L(rows[I_S0]['ratio_fe_cmf']) - L(rows[I_S8]['ratio_fe_cmf'])
grad_fe_L   = L(rows[I_S0]['ratio_fe_L'])   - L(rows[I_S8]['ratio_fe_L'])
grad_co_cmf = L(rows[I_S0]['ratio_co_cmf']) - L(rows[I_S8]['ratio_co_cmf'])
grad_co_L   = L(rows[I_S0]['ratio_co_L'])   - L(rows[I_S8]['ratio_co_L'])

B_cmf = report("CMFGEN  Fe III->IV", 'ratio_fe_cmf', 'Gg_c', 'Gb_c', 'alp_c', 'nec', 'saha_c', grad_fe_cmf)
B_L   = report("LUMINA  Fe III->IV", 'ratio_fe_L',   'Gg_L', 'Gb_L', 'alp_L', 'neL', 'saha_L', grad_fe_L)

# Saha-T_e-alone with n_e held FIXED at s0 (isolate pure T_e leverage)
def saha_Te_only(Terow_key, ne_fixed):
    return L(saha_ratio(rows[I_S0][Terow_key], ne_fixed, 26, 2)) - \
           L(saha_ratio(rows[I_S8][Terow_key], ne_fixed, 26, 2))
dSahaTe_c = saha_Te_only('Tc', rows[I_S0]['nec'])
dSahaTe_L = saha_Te_only('TL', rows[I_S0]['neL'])
# field-band declines
dJ918_c = grad('gm918_c'); dJ918_L = grad('gm918_L')
dJ300_c_gm = grad('gm300_c'); dJ300_c_am = grad('am300_c')
dJ300_L_am = grad('am300_L')

print("\n" + "=" * 118)
print("FIELD-BAND DECLINES (D1/D2)  s0->s8   [+dex = deep brighter]")
print("=" * 118)
print(f"  D1  J(918-1290) geom-mean :  CMFGEN {dJ918_c:+.2f} dex   |   Lumina(mc_J) {dJ918_L:+.2f} dex")
print(f"  D2  J(300-450)  geom-mean :  CMFGEN {dJ300_c_gm:+.2f} dex   |   Lumina(mc_J) floor-dominated (see arith)")
print(f"  D2  J(300-450)  arith-mean:  CMFGEN {dJ300_c_am:+.2f} dex   |   Lumina(mc_J) {dJ300_L_am:+.2f} dex")
print(f"      Lumina 300-450 mc_J floor-bin count per shell (of 77): "
      f"{[r['fl300_L'] for r in rows]}")
print("\n  D4  Saha T_e-alone (n_e fixed@s0):  CMFGEN {:+.2f} dex   |   Lumina {:+.2f} dex".format(dSahaTe_c, dSahaTe_L))
print(f"      T_e(v): CMFGEN {rows[I_S0]['Tc']:.0f}->{rows[I_S8]['Tc']:.0f} K   |   "
      f"Lumina {rows[I_S0]['TL']:.0f}->{rows[I_S8]['TL']:.0f} K")

# ================================================================= WRITE CSV
import csv as _csv
out_csv = f'{REPO}/validation/cmfgen_toy06_19p48d/analysis/gradient_budget_shells.csv'
with open(out_csv, 'w', newline='') as f:
    w = _csv.writer(f)
    w.writerow(['shell', 'v_kms',
                'CMFGEN_Te', 'CMFGEN_ne', 'CMFGEN_FeIV_III', 'CMFGEN_Gph_gnd', 'CMFGEN_Gph_boltz',
                'CMFGEN_alpha', 'CMFGEN_ratio_pred', 'CMFGEN_J918_gm', 'CMFGEN_J300_gm', 'CMFGEN_J300_am',
                'LUM_Te', 'LUM_ne', 'LUM_FeIV_III', 'LUM_Gph_gnd', 'LUM_Gph_boltz',
                'LUM_alpha', 'LUM_ratio_pred', 'LUM_J918_gm', 'LUM_J300_am', 'LUM_J300_floorbins',
                'CMFGEN_CoIV_III', 'LUM_CoIV_III'])
    for r in rows:
        w.writerow([r['s'], r['v'],
                    f"{r['Tc']:.1f}", f"{r['nec']:.4e}", f"{r['ratio_fe_cmf']:.4e}",
                    f"{r['Gg_c']:.4e}", f"{r['Gb_c']:.4e}", f"{r['alp_c']:.4e}", f"{r['rat_c']:.4e}",
                    f"{r['gm918_c']:.4e}", f"{r['gm300_c']:.4e}", f"{r['am300_c']:.4e}",
                    f"{r['TL']:.1f}", f"{r['neL']:.4e}", f"{r['ratio_fe_L']:.4e}",
                    f"{r['Gg_L']:.4e}", f"{r['Gb_L']:.4e}", f"{r['alp_L']:.4e}", f"{r['rat_L']:.4e}",
                    f"{r['gm918_L']:.4e}", f"{r['am300_L']:.4e}", r['fl300_L'],
                    f"{r['ratio_co_cmf']:.4e}", f"{r['ratio_co_L']:.4e}"])
print(f"\n[out] per-shell CSV -> {out_csv}")

# stash summary dict for the markdown writer
np.save(f'{REPO}/validation/cmfgen_toy06_19p48d/analysis/_budget_summary.npy',
        dict(B_cmf=B_cmf, B_L=B_L, grad_fe_cmf=grad_fe_cmf, grad_fe_L=grad_fe_L,
             grad_co_cmf=grad_co_cmf, grad_co_L=grad_co_L,
             dJ918_c=dJ918_c, dJ918_L=dJ918_L, dJ300_c_gm=dJ300_c_gm, dJ300_c_am=dJ300_c_am,
             dJ300_L_am=dJ300_L_am, dSahaTe_c=dSahaTe_c, dSahaTe_L=dSahaTe_L,
             Tc0=rows[I_S0]['Tc'], Tc8=rows[I_S8]['Tc'], TL0=rows[I_S0]['TL'], TL8=rows[I_S8]['TL'],
             floor300=[r['fl300_L'] for r in rows]), allow_pickle=True)
print("[out] summary -> _budget_summary.npy")
