#!/usr/bin/env python3
"""J_inc continuum falsifier (codex criterion) — decide line-source-fix vs A4.

Reads lumina_frozen_jinc.csv (per shell,bin: chi_es, chi_abs, J_inc, B_Te,
B_Tinner, tau_es_rad, tau_abs_rad, tau_eff_c). At the Ca II / Si II / S II
line-forming shells, reports the continuum thermalization depth tau_eff_c and
the dilution J_inc/B(Tinner), J_inc/B(Te).

Decision (codex):
  tau_eff_c << 1 AND J_inc < B(Tinner) (diluted photospheric) -> trough WILL
    form -> line-source fix suffices (morphology defect = line source).
  tau_eff_c >= 1 with J_inc ~ B(Te)   -> field thermalized -> A4 required.

Usage: analyze_jinc_falsifier.py <run_dir>
"""
import sys
import numpy as np

run = sys.argv[1] if len(sys.argv) > 1 else '.'
path = run.rstrip('/') + '/lumina_frozen_jinc.csv'

with open(path) as fh:
    hdr = fh.readline()
    Tinner = float(hdr.split('T_inner=')[1]) if 'T_inner=' in hdr else float('nan')
cols = np.genfromtxt(path, delimiter=',', names=True, skip_header=1)

shell = cols['shell'].astype(int)
binid = cols['bin'].astype(int)
lamA = cols['lambda_A']
te = cols['Te']
jinc = cols['J_inc']
bte = cols['B_Te']
btin = cols['B_Tinner']
tes = cols['tau_es_rad']
tab = cols['tau_abs_rad']
teff = cols['tau_eff_c']

FEAT = [('Ca II K', 3934), ('S II 5460', 5460), ('Si II 6355', 6355),
        ('Ca II NIR', 8542)]

print(f"T_inner = {Tinner:.0f} K")
print(f"{'feature':<11} {'shell':>5} {'tau_es':>8} {'tau_eff_c':>9} "
      f"{'Jinc/BTin':>9} {'Jinc/BTe':>8} {'verdict':<22}")
print('-' * 78)

overall = []
for name, lam in FEAT:
    # nearest bin to this wavelength
    m = np.abs(lamA - lam) == np.abs(lamA - lam).min()
    b = binid[m][0]
    sel = binid == b
    s_ = shell[sel]; tes_ = tes[sel]; teff_ = teff[sel]
    jbt = np.where(btin[sel] > 0, jinc[sel]/btin[sel], np.nan)
    jbe = np.where(bte[sel] > 0, jinc[sel]/bte[sel], np.nan)
    order = np.argsort(s_)
    s_, tes_, teff_, jbt, jbe = s_[order], tes_[order], teff_[order], jbt[order], jbe[order]
    # line-forming region ~ continuum photosphere: tau_es_rad in [0.1, 2]
    lf = (tes_ >= 0.1) & (tes_ <= 2.0)
    if not lf.any():
        lf = (tes_ >= 0.03)
    idx = np.where(lf)[0]
    # representative shell = first where tau_es ~ 0.3-1
    rep = idx[len(idx)//2] if len(idx) else len(s_)//2
    teff_rep = teff_[rep]; jbt_rep = jbt[rep]; jbe_rep = jbe[rep]
    # falsifier
    if teff_rep < 0.3 and jbt_rep < 0.7:
        v = 'TROUGH (line-src OK)'
    elif teff_rep >= 1.0 and abs(jbe_rep - 1) < 0.3:
        v = 'THERMALIZED (A4)'
    else:
        v = 'marginal'
    overall.append(v)
    print(f"{name:<11} {s_[rep]:>5d} {tes_[rep]:>8.3f} {teff_rep:>9.3f} "
          f"{jbt_rep:>9.3f} {jbe_rep:>8.3f} {v:<22}")

print('-' * 78)
trough = sum('TROUGH' in v for v in overall)
therm = sum('THERMALIZED' in v for v in overall)
print(f"VERDICT: {trough}/{len(FEAT)} features in trough-favorable regime "
      f"(tau_eff_c<<1 & J_inc<B(Tinner)); {therm}/{len(FEAT)} thermalized.")
if trough >= 2:
    print("  => J_inc carries diluted photospheric memory -> line-source fix "
          "should produce P-Cygni (check spectrum). Red continuum = separate A4.")
elif therm >= 2:
    print("  => field thermalized at line shells -> frozen line-source fix DEAD "
          "-> full A4 continuum-transport required.")
else:
    print("  => mixed/marginal -> inspect spectrum + per-shell profile below.")
