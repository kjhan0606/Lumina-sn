#!/usr/bin/env python3
"""
Phase 1: Joint ML decomposition of HST SN 2011fe B-max spectrum into ion
contributions. SYNAPPS-style inversion.

Model: F_model(λ) / F_cont(λ) = exp(-Σ_lines τ_line(λ; θ))
       τ_line(λ) = τ_max_line × exp(-0.5 × ((v(λ) - v_ion)/σ_v_ion)²)
       τ_max_line = (πe²/m_e c) × f_lu × λ_rest × n_l × t_exp
       n_l = n_ion × g_l × exp(-E_l/kT) / Z(T)

Free params per ion: (v_form_ion, σ_v_ion, log10(n_ion)). 11 ions × 3 = 33 D.
T_exc fixed at 9000 K. ρ(v) and X_ion derived post-hoc from physical ref.

Output:
  - data/sn2011fe/empirical_pcygni_targets.csv
  - figures/empirical_pcygni_ml.png
"""
from __future__ import annotations
import numpy as np, pandas as pd, json
from pathlib import Path
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
REF  = ROOT / "data/tardis_reference_strat6_2011fe_physical"
OUT_CSV = ROOT / "data/sn2011fe/empirical_pcygni_targets.csv"
OUT_PNG = ROOT / "figures/empirical_pcygni_ml.png"

C_KMS   = 299792.458
T_EXP_S = 1486080.0        # 17.2 d
SOBOLEV = 2.6540e-2        # π e² / m_e c   [cm² · Hz]
KB_eV   = 8.617e-5         # eV/K
T_EXC_K = 9000.0
LAM_LO, LAM_HI = 3500, 9000

# Hard-coded atomic data (NIST / Branch+2006 / Hatano1999 effective values).
# (ion, label, lambda_rest_AA, f_lu, E_l_eV, g_l, Z_ion_at_9kK)
LINES = [
    # Si II — 3p² P° lower term, IP_SiII = 16.3 eV → mostly Si II at SN T
    ('SiII', 'Si II 6355', 6359.0, 0.50,  8.121, 6, 5.0),   # blend 6347+6371
    ('SiII', 'Si II 5979', 5972.6, 0.18,  10.07, 4, 5.0),   # blend 5957+5979
    ('SiII', 'Si II 4129', 4129.5, 0.36,  9.840, 4, 5.0),   # blend 4128+4131
    ('SiII', 'Si II 3858', 3858.2, 0.045, 6.860, 2, 5.0),
    # Ca II — ground term
    ('CaII', 'Ca II K',    3933.66, 0.683, 0.0,   2, 3.0),
    ('CaII', 'Ca II H',    3968.47, 0.330, 0.0,   2, 3.0),
    ('CaII', 'Ca II 8498', 8498.02, 0.012, 1.69,  4, 3.0),
    ('CaII', 'Ca II 8542', 8542.09, 0.072, 1.70,  6, 3.0),
    ('CaII', 'Ca II 8662', 8662.14, 0.060, 1.69,  4, 3.0),
    # S II — UV15 multiplet (W feature)
    ('SII',  'S II 5454',  5453.86, 0.022, 13.67, 4, 7.0),
    ('SII',  'S II 5640',  5640.0,  0.015, 13.79, 4, 7.0),
    ('SII',  'S II 5468',  5468.0,  0.014, 13.62, 4, 7.0),
    # Mg II
    ('MgII', 'Mg II 4481', 4481.2,  0.73,  8.864, 4, 3.0),
    # O I — triplet at 7773
    ('OI',   'O I 7773',   7773.83, 0.469, 9.146, 15, 9.0),
    # Fe II — multiplet 42 + UV27/37
    ('FeII', 'Fe II 5169', 5169.03, 0.025, 2.89,  10, 30.0),
    ('FeII', 'Fe II 4924', 4923.92, 0.045, 2.89,  10, 30.0),
    ('FeII', 'Fe II 5018', 5018.43, 0.038, 2.89,   8, 30.0),
    ('FeII', 'Fe II 4549', 4549.47, 0.025, 2.83,   8, 30.0),
    ('FeII', 'Fe II 4233', 4233.17, 0.010, 2.58,  10, 30.0),
    # Fe III
    ('FeIII','Fe III 4404',4404.75, 0.05,  6.31,   9, 25.0),
    ('FeIII','Fe III 5128',5127.39, 0.04,  9.27,   9, 25.0),
    # Co II
    ('CoII', 'Co II 3995', 3995.30, 0.06,  4.92,  12, 30.0),
    ('CoII', 'Co II 4145', 4145.16, 0.04,  4.95,  10, 30.0),
    # Ni II
    ('NiII', 'Ni II 4067', 4067.04, 0.02,  4.00,  10, 25.0),
    ('NiII', 'Ni II 4159', 4159.05, 0.015, 4.00,  10, 25.0),
]
df_lines = pd.DataFrame(LINES, columns=['ion','label','lam','f','E_l','g_l','Z'])
ions = list(df_lines['ion'].unique())
print(f"{len(LINES)} lines across {len(ions)} ions: {ions}")

# ---- Load & preprocess HST ----
def load(p):
    d = pd.read_csv(p); lam = d.iloc[:,0].values
    flu = d['flux_erg_s_cm2_angstrom'].values if 'flux_erg_s_cm2_angstrom' in d.columns else d.iloc[:,1].values
    err = d['error'].values if 'error' in d.columns else 0.05*flu
    return lam, flu, err

hlam, hflu, herr = load(HST)
m = (hlam>=LAM_LO)&(hlam<=LAM_HI)&np.isfinite(hflu)&(hflu>0)
hlam, hflu, herr = hlam[m], hflu[m], herr[m]
print(f"HST [{LAM_LO},{LAM_HI}]Å bins: {len(hlam)}")

def gauss_cont(lam, flu, fwhm_kms=40000.0):
    out = np.zeros_like(flu, dtype=float); beta = fwhm_kms / C_KMS
    for i in range(len(lam)):
        s = beta*lam[i]/2.3548; w=4.0*s
        sel=(lam>=lam[i]-w)&(lam<=lam[i]+w)
        if sel.sum()<2: out[i]=flu[i]; continue
        ww=np.exp(-0.5*((lam[sel]-lam[i])/s)**2)
        out[i]=np.sum(ww*flu[sel])/np.sum(ww)
    return out

# Wider baseline (FWHM=80k) so individual lines aren't absorbed into the kernel.
# Anchor through known clean continuum windows (between strong P-Cygni features).
anchor_windows = [(3700,3800),(4350,4480),(4800,4920),(5500,5780),(6650,6900),
                  (7000,7150),(7600,7700),(8400,8460),(8800,9000)]
mask_anchor = np.zeros_like(hlam, dtype=bool)
for lo, hi in anchor_windows:
    mask_anchor |= (hlam>=lo)&(hlam<=hi)
# Smooth flux at anchors with FWHM=80k, then cubic-spline through smoothed anchors
from scipy.interpolate import UnivariateSpline
hcont_smooth = gauss_cont(hlam, hflu, fwhm_kms=80000.0)
if mask_anchor.sum() >= 8:
    spline = UnivariateSpline(hlam[mask_anchor], hcont_smooth[mask_anchor],
                              s=len(hlam[mask_anchor])*1e-30, k=3)
    hcont = spline(hlam)
    hcont = np.where(hcont>0, hcont, hcont_smooth)
else:
    hcont = hcont_smooth
hnorm = hflu / hcont
hsig  = herr / hcont
hsig = np.clip(hsig, 0.01, None)
print(f"baseline (spline through anchors) median: {np.median(hcont):.3e}")
print(f"  anchor bins: {mask_anchor.sum()} / {len(hlam)}")

# ---- Model ----
LAM_CM = (df_lines['lam'].values) * 1e-8     # rest λ [cm]
F_LU   = df_lines['f'].values
E_L    = df_lines['E_l'].values
G_L    = df_lines['g_l'].values
Z_ION  = df_lines['Z'].values
LAM_AA = df_lines['lam'].values
LINE_ION_IDX = np.array([ions.index(z) for z in df_lines['ion']])

BOLTZ = (G_L / Z_ION) * np.exp(-E_L / (KB_eV * T_EXC_K))
SOB_COEFF = SOBOLEV * F_LU * LAM_CM * T_EXP_S * BOLTZ      # τ_line = SOB_COEFF × n_ion × line_profile

def model_flux(params, lam):
    # params: per ion (v_form_kms, sigma_v_kms, log10_n_ion), in order of `ions`
    v_form  = params[0::3]
    sigma_v = params[1::3]
    log_n   = params[2::3]
    n_ion = 10.0**log_n
    tau_max = SOB_COEFF * n_ion[LINE_ION_IDX]
    v_obs = C_KMS * (1.0 - lam[:,None] / LAM_AA[None,:])   # [nlam, nline]
    v_ion = v_form[LINE_ION_IDX][None,:]
    s_ion = sigma_v[LINE_ION_IDX][None,:]
    prof = np.exp(-0.5 * ((v_obs - v_ion) / s_ion)**2)
    tau_line = tau_max[None,:] * prof
    tau_tot = np.sum(tau_line, axis=1)
    tau_tot = np.clip(tau_tot, 0, 8)   # τ>8 is fully saturated; clip for smooth gradient
    return np.exp(-tau_tot)

def neglogL(params):
    pred = model_flux(params, hlam)
    return 0.5 * np.sum(((hnorm - pred) / hsig)**2)

# ---- Initial guesses ----
init_guess = {
    'SiII':  (10500, 1800,  9.5),
    'CaII':  (10500, 2000,  6.5),
    'SII':   ( 9500, 1600, 10.5),
    'MgII':  (11000, 1800,  9.5),
    'OI':    (13000, 2500,  8.0),
    'FeII':  (12000, 3000, 10.5),
    'FeIII': (12500, 2500, 10.5),
    'CoII':  (12500, 2500, 10.5),
    'NiII':  (12500, 2500, 10.5),
}
x0, bounds = [], []
for ion in ions:
    v0, s0, ln0 = init_guess[ion]
    x0 += [v0, s0, ln0]
    bounds += [(3000, 25000), (1500, 6000), (3, 16)]
x0 = np.array(x0, dtype=float)
print(f"\nx0 χ² = {2*neglogL(x0):.1f}")

# ---- Optimize ----
# Two-stage: global rough search via differential evolution, then local polish.
from scipy.optimize import differential_evolution
print("Stage 1: differential evolution global search...")
res_de = differential_evolution(neglogL, bounds, seed=42, maxiter=80,
                                 popsize=20, tol=1e-6, polish=False,
                                 x0=x0)
print(f"  stage 1 χ² = {2*res_de.fun:.1f}")
print("Stage 2: L-BFGS-B polish...")
res = minimize(neglogL, res_de.x, method='L-BFGS-B', bounds=bounds,
               options={'maxiter': 500, 'ftol': 1e-10})
print(f"final χ² = {2*res.fun:.1f}   converged={res.success}   niter={res.nit}")

# ---- Tabulate ----
v_form  = res.x[0::3]
sigma_v = res.x[1::3]
n_ion   = 10.0**res.x[2::3]

# Density profile (W7-strat6 physical)
geom = pd.read_csv(REF/"geometry.csv"); dens = pd.read_csv(REF/"density.csv")
v_mid_src = (geom['v_inner']+geom['v_outer'])/2/1e5
slope, intercept = np.polyfit(np.log(v_mid_src), np.log(dens['rho']), 1)
def rho_at_v(vkms):
    return np.exp(intercept + slope*np.log(vkms))

M_AMU = 1.6605e-24
ATOMIC_MASS = {'SiII':28, 'CaII':40, 'SII':32, 'MgII':24, 'OI':16,
               'FeII':56, 'FeIII':56, 'CoII':59, 'NiII':58}
rows = []
for i, ion in enumerate(ions):
    v = v_form[i]; sv = sigma_v[i]; ni = n_ion[i]
    rho_v = rho_at_v(v)
    n_total_element = rho_v / (ATOMIC_MASS[ion] * M_AMU)
    X_est = ni / n_total_element
    rows.append(dict(
        ion=ion, n_lines=int(np.sum(LINE_ION_IDX==i)),
        v_form_kms=round(v,0), sigma_v_kms=round(sv,0),
        log10_n_ion=round(np.log10(ni),2),
        n_ion_per_cm3=ni,
        rho_at_v_g_cm3=rho_v,
        X_ion_est=X_est,
    ))
df_tgt = pd.DataFrame(rows).sort_values('v_form_kms').reset_index(drop=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_tgt.to_csv(OUT_CSV, index=False)
print("\n=== Empirical P-Cygni inversion targets ===")
print(df_tgt.to_string(index=False))
print(f"\nWrote {OUT_CSV}")

# ---- Compare to W7-strat6 abundances at v_form ----
ab = pd.read_csv(REF/"abundances.csv")
v_shells = (geom['v_inner']+geom['v_outer'])/2/1e5
Z_OF = {'SiII':14,'CaII':20,'SII':16,'MgII':12,'OI':8,'FeII':26,'FeIII':26,'CoII':27,'NiII':28}
def X_strat_at_v(Z, v_kms):
    if Z not in ab['atomic_number'].values: return np.nan
    row = ab[ab['atomic_number']==Z].iloc[0]
    xs = row.values[1:].astype(float)
    return float(np.interp(v_kms, v_shells, xs))

df_tgt['X_W7strat6'] = [X_strat_at_v(Z_OF[r['ion']], r['v_form_kms']) for _,r in df_tgt.iterrows()]
df_tgt['ratio_emp_over_W7'] = df_tgt['X_ion_est'] / df_tgt['X_W7strat6']
df_tgt.to_csv(OUT_CSV, index=False)
print("\n--- W7-strat6 comparison ---")
print(df_tgt[['ion','v_form_kms','X_ion_est','X_W7strat6','ratio_emp_over_W7']].to_string(index=False))

# ---- Plot ----
pred = model_flux(res.x, hlam)
fig, axes = plt.subplots(3, 1, figsize=(14, 11))   # NO sharex
panels = [(3500, 5000), (5000, 7000), (7000, 9000)]

for ax, (lo, hi) in zip(axes, panels):
    s = (hlam>=lo)&(hlam<=hi)
    ax.plot(hlam[s], hnorm[s], 'k-', lw=1.0, label='HST (baseline-norm)')
    ax.plot(hlam[s], pred[s], 'r-', lw=1.0, alpha=0.8, label='ML model (Σ ions)')
    ax.fill_between(hlam[s], hnorm[s]-hsig[s], hnorm[s]+hsig[s], color='k', alpha=0.1)

    # per-ion contributions (dashed)
    colors = plt.cm.tab10(np.linspace(0,1,len(ions)))
    for i, ion in enumerate(ions):
        line_mask = (LINE_ION_IDX==i)
        v_obs = C_KMS*(1.0 - hlam[s,None]/LAM_AA[None,line_mask])
        prof = np.exp(-0.5*((v_obs - v_form[i])/sigma_v[i])**2)
        tau_ion = np.sum((SOB_COEFF[line_mask]*n_ion[i])[None,:] * prof, axis=1)
        ax.plot(hlam[s], np.exp(-tau_ion), '--', color=colors[i], lw=0.7, alpha=0.7)

    # label troughs
    for _, r in df_lines.iterrows():
        if r['lam']>=lo and r['lam']<=hi:
            ion_v = v_form[ions.index(r['ion'])]
            lam_obs = r['lam']*(1 - ion_v/C_KMS)
            if lo<=lam_obs<=hi:
                ax.axvline(lam_obs, color='gray', alpha=0.2, lw=0.5)
                ax.text(lam_obs, 0.2, r['label'], rotation=90, fontsize=6,
                        ha='right', va='bottom', alpha=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(0, 1.5)
    ax.set_ylabel('F / F_cont'); ax.grid(alpha=0.2)
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Wavelength (Å)')
axes[0].set_title(f'Empirical P-Cygni ML decomposition  —  SN 2011fe B-max  (T_exc={T_EXC_K:.0f}K, t_exp=17.2d)\n'
                  f'χ² = {2*res.fun:.0f}  /  {len(hlam)} bins  ({len(LINES)} lines / {len(ions)} ions)')
plt.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=120)
print(f"\nWrote {OUT_PNG}")
