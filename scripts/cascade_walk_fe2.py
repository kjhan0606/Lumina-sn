#!/usr/bin/env python3
"""RUNG 0: deterministic Lucy macro-atom cascade walk for Fe II.

Resolves FIELD vs STRUCTURE: does the UV-entry -> UV-exit fraction depend on
the radiation field strength (FIELD root) or is it flat (STRUCTURE root)?

Builds the ARTIS/Lucy-2002 macro-atom transition probabilities from first
principles (line_list A_ul/B_lu/B_ul + level energies), with the radiation
field J_bar(nu) = k * B(nu, T_e). Then solves the absorbing Markov chain:
a packet enters at the upper level of a UV line and random-walks internal
transitions until it EMITS; we bin the emitted photon's band.

Radiative-only (isolates the branching; collisions add thermal pool separately).
Vary k -> if UV-exit flat => STRUCTURE; if UV-exit drops with k => FIELD.

Usage: python3 scripts/cascade_walk_fe2.py [shell=3]
"""
import sys, csv
import numpy as np

H = 6.62607015e-27; KB = 1.380649e-16; C = 2.99792458e10; EV = 1.602176634e-12
Z, ION = 26, 1                      # Fe II
SHELL = int(sys.argv[1]) if len(sys.argv) > 1 else 3

# --- plasma state for T_e at the shell (epay27 frozen champion) ---
ps = {int(r['shell_id']): float(r['T_e'])
      for r in csv.DictReader(open('logs/stage1_toy06_epay27/lumina_plasma_state.csv'))}
T_e = ps[SHELL]

# --- Fe II levels: energy (eV), g ---
E = {}; G = {}
for r in csv.DictReader(open('data/tardis_reference_toy06_19p48d/levels.csv')):
    if int(r['atomic_number']) == Z and int(r['ion_number']) == ION:
        l = int(r['level_number']); E[l] = float(r['energy_eV']); G[l] = float(r['g'])
NL = max(E) + 1
Eev = np.array([E.get(i, 0.0) for i in range(NL)])

# --- Fe II lines: lower, upper, nu, A_ul, B_lu, B_ul ---
low = []; up = []; nu = []; Aul = []; Blu = []; Bul = []
for r in csv.DictReader(open('data/tardis_reference_toy06_19p48d/line_list.csv')):
    if int(r['atomic_number']) == Z and int(r['ion_number']) == ION:
        low.append(int(r['level_number_lower'])); up.append(int(r['level_number_upper']))
        nu.append(float(r['nu'])); Aul.append(float(r['A_ul']))
        Blu.append(float(r['B_lu'])); Bul.append(float(r['B_ul']))
low = np.array(low); up = np.array(up); nu = np.array(nu)
Aul = np.array(Aul); Blu = np.array(Blu); Bul = np.array(Bul)
lam_A = C / nu * 1e8
print(f"Fe II @ shell {SHELL}: T_e={T_e:.0f}K  NL={NL}  nlines={len(nu)}")

BANDS = [('FUV', 0, 1700), ('UVblnk', 1700, 3000), ('CaIIKb', 3000, 3300),
         ('UVtgt', 3300, 3700), ('fluor', 3700, 4400), ('green', 4400, 5500),
         ('red', 5500, 7000), ('NIR1', 7000, 10000), ('NIR2', 10000, 1e9)]
def band_of(lam):
    for i, (nm, lo, hi) in enumerate(BANDS):
        if lo <= lam < hi: return i
    return len(BANDS) - 1
line_band = np.array([band_of(l) for l in lam_A])
UV_BANDS = {0, 1, 2, 3}          # FUV..UVtgt (<3700A) = "UV"
OPT_BANDS = {4, 5, 6}            # fluor+green+red

def Bnu(nu_, T):
    x = H * nu_ / (KB * T)
    return np.where(x < 500, 2 * H * nu_**3 / C**2 / np.expm1(np.clip(x, 1e-30, 500)), 0.0)

def run_walk(k):
    """k = field scaling: J_bar(nu) = k * B(nu, T_e). Returns exit-band vector
    for a UV-entry distribution."""
    Jbar = k * Bnu(nu, T_e)
    # radiative rates per line
    R_down = Aul + Bul * Jbar          # upper->lower (spontaneous + stimulated)
    R_up   = Blu * Jbar                # lower->upper
    hnu    = H * nu                     # emission photon energy
    El = Eev[low] * EV; Eu = Eev[up] * EV
    # Lucy energy-flow weights (ARTIS macroatom.cc):
    #   internal-down (from upper u to lower l): w = R_down * E_l
    #   internal-up   (from lower l to upper u): w = R_up   * E_u ... (target level E)
    #   emission      (from upper u to lower l): w = R_down * (E_u - E_l) = R_down*hnu
    # accumulate per-source-level normalization and transition lists
    # source = upper for down/emit; source = lower for up
    w_idn = R_down * El                 # source=up, dest=low
    w_emit = R_down * hnu               # source=up, dest=absorbing band
    w_iup = R_up * Eu                   # source=low, dest=up
    # total outgoing weight per level
    tot = np.zeros(NL)
    np.add.at(tot, up,  w_idn + w_emit)
    np.add.at(tot, low, w_iup)
    tot[tot == 0] = 1.0
    # Build transition operator by iterating the chain (power method).
    # state vector over levels; each step: distribute to dest levels (internal),
    # accumulate emission into bands.
    # Precompute normalized probabilities per line.
    p_idn = w_idn / tot[up]             # up-level -> low-level
    p_emit = w_emit / tot[up]           # up-level -> emit band(line)
    p_iup = w_iup / tot[low]            # low-level -> up-level

    # entry: UV-line absorption puts macro-atom at the UPPER level.
    # weight entry by UV-line absorption rate ~ B_lu*Jbar (per line), at upper level.
    uvmask = np.array([b in UV_BANDS for b in line_band])
    entry = np.zeros(NL)
    ew = (Blu * Jbar) * uvmask
    np.add.at(entry, up, ew)
    if entry.sum() == 0: return None
    entry /= entry.sum()

    exit_band = np.zeros(len(BANDS))
    s = entry.copy()
    for it in range(2000):
        # emission absorption this step: from each up-level, p_emit into its band
        contrib = s[up] * p_emit
        np.add.at(exit_band, line_band, contrib)
        # internal moves -> next state
        s2 = np.zeros(NL)
        np.add.at(s2, low, s[up] * p_idn)    # internal down: up->low
        np.add.at(s2, up,  s[low] * p_iup)   # internal up:   low->up
        if s2.sum() < 1e-12: break
        s = s2
    return exit_band

print(f"\n{'k=Jbar/B':>9} | " + ' '.join(f"{nm:>6}" for nm, _, _ in BANDS) + " | UVexit% OPTexit%")
for k in [1.0, 2.0, 5.0, 10.0, 30.0]:
    eb = run_walk(k)
    if eb is None: print(f"{k:9.1f} | (no UV entry)"); continue
    tot = eb.sum()
    uv = sum(eb[i] for i in UV_BANDS); opt = sum(eb[i] for i in OPT_BANDS)
    print(f"{k:9.1f} | " + ' '.join(f"{100*eb[i]/tot:6.1f}" for i in range(len(BANDS)))
          + f" | {100*uv/tot:6.1f} {100*opt/tot:6.1f}")

print("\nVERDICT: UVexit% flat vs k => STRUCTURE (branching routes back to UV,")
print("field-independent). UVexit% drops with k => FIELD (super-thermal pump")
print("enables the down-cascade).")
