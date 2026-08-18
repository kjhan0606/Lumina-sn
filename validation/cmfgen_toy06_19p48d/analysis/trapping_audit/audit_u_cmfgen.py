#!/usr/bin/env python3
"""Audit U (CMFGEN side): energy density u(s) = (4pi/c) INT J_nu dnu from EDDFACTOR.
Reuses the validated EDDFACTOR reader from extract_jnu.py (comp_j_blank.f:249/412/808).
J_nu units: erg/cm2/s/Hz/sr (CGS). u units: erg/cm3.
"""
import numpy as np

RUN = "/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4"
EDD = RUN + "/EDDFACTOR"
RVTJ = RUN + "/RVTJ"
CLIGHT_A = 2997.92458       # lam_A = CLIGHT_A / FL(1e15 Hz)
C_CM = 2.99792458e10        # cm/s
FOURPI_OVER_C = 4*np.pi / C_CM
TARGET_V = [4264, 4992, 5720, 6448, 7176, 7904, 8632, 9360, 10088, 10816, 11544]  # task s0..s10 (728 spacing)
SLAB = ['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']
# Lumina overlap nu range (its mc_J grid): nu 1.5e14 .. 3.0e16 Hz
NU_LO, NU_HI = 1.5e14, 3.0e16

def read_info(info):
    L = open(info).read().splitlines()
    v = L[2].split()
    return dict(ND=int(v[0]), RECL=int(v[1]), WORD=int(v[2]), little=(v[5]=='T'))

def read_eddfactor(edd):
    info = read_info(edd+'_INFO')
    ND = info['ND']; nwr = info['RECL']//info['WORD']
    dt = '<f8' if info['little'] else '>f8'
    raw = np.fromfile(edd, dtype=dt)
    n = (raw.size//nwr)*nwr
    raw = raw[:n].reshape(-1, nwr)
    finish = raw[4,0]
    data = raw[14:]
    good = np.isfinite(data[:,:ND]).all(axis=1) & (data[:,ND] > 0)
    J = data[good,:ND]        # [nfreq, ND] erg/cm2/s/Hz/sr
    FL = data[good,ND]        # 1e15 Hz
    return J, FL, ND, finish, int((~good).sum()), data.shape[0]

def parse_rvtj_block(text, label, ND):
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip()==label:
            vals=[]; j=i+1
            while j < len(lines) and len(vals) < ND:
                toks = lines[j].split()
                try: vals += [float(t) for t in toks]
                except ValueError: break
                j += 1
            return np.array(vals[:ND])
    raise KeyError(label)

J, FL, ND, finish, nbad, ntot = read_eddfactor(EDD)
nu = FL*1e15                                   # Hz, descending
lam = CLIGHT_A/FL
print(f"[edd] ND={ND} nfreq_good={J.shape[0]} nbad={nbad}/{ntot} FINISH={finish}")
print(f"[freq] nu range {nu.min():.3e} .. {nu.max():.3e} Hz  (lam {lam.min():.1f} .. {lam.max():.1f} A)")

rt = open(RVTJ).read()
V = parse_rvtj_block(rt, 'Velocity (km/s)', ND)
T = parse_rvtj_block(rt, 'Temperature (10^4K)', ND)*1e4
print(f"[rvtj] V[0]={V[0]:.0f} V[-1]={V[-1]:.0f} km/s  T[0]={T[0]:.0f} T[-1]={T[-1]:.0f} K")

# sort freq ascending for integration
o = np.argsort(nu)
nu_s = nu[o]; J_s = J[o]                        # [nfreq, ND]

# --- ANCHOR CHECK: geometric-mean J over 918-1290A at v=4264 and v=10088 ---
bm = (lam >= 918.0) & (lam <= 1290.0)
lam_b = lam[bm]; Jb = J[bm]                     # [nband, ND]
dv = np.argsort(V); Vasc = V[dv]
def band_geo(vt):
    Jvt = np.empty(lam_b.shape[0])
    for k in range(lam_b.shape[0]):
        yk = Jb[k, dv]; yk = np.where(yk>0, yk, np.nan)
        ly = np.log10(yk); g = np.isfinite(ly)
        Jvt[k] = 10**np.interp(vt, Vasc[g], ly[g])
    return np.exp(np.nanmean(np.log(Jvt)))
print(f"[ANCHOR] band-geo J(v=4264)={band_geo(4264):.4e} (target 2.02e-4)  J(v=10088)={band_geo(10088):.4e} (target 7.73e-7)")

# --- energy density per depth, full range and overlap range ---
# integrate J dnu over full CMFGEN range and over [NU_LO,NU_HI]
u_full = np.empty(ND); u_ovl = np.empty(ND); u_below = np.empty(ND); u_above = np.empty(ND)
mask_ovl = (nu_s >= NU_LO) & (nu_s <= NU_HI)
for d in range(ND):
    y = J_s[:, d]
    u_full[d] = FOURPI_OVER_C*np.trapz(y, nu_s)
    # overlap: trapz over restricted range with interpolated endpoints
    u_ovl[d]  = FOURPI_OVER_C*np.trapz(y[mask_ovl], nu_s[mask_ovl])
    below = nu_s < NU_LO; above = nu_s > NU_HI
    u_below[d]= FOURPI_OVER_C*np.trapz(y[below], nu_s[below]) if below.sum()>1 else 0.0
    u_above[d]= FOURPI_OVER_C*np.trapz(y[above], nu_s[above]) if above.sum()>1 else 0.0

# sanity: innermost u_full vs a T^4 (thermalized)  a=7.5657e-15
a_rad = 7.5657e-15
print(f"[sanity] innermost depth u_full={u_full[-1]:.4e}  a*T^4={a_rad*T[-1]**4:.4e}  ratio={u_full[-1]/(a_rad*T[-1]**4):.3f}")

# interpolate log(u) in velocity to targets
def interp_v(u):
    ly = np.log10(u[dv]); return lambda vt: 10**np.interp(vt, Vasc, ly)
fF = interp_v(u_full); fO = interp_v(u_ovl); fB = interp_v(u_below); fA = interp_v(u_above)

print("\n# CMFGEN energy density (erg/cm3) mapped to Lumina shells")
print(f"{'shell':>5} {'v':>6} {'u_full':>11} {'u_overlap':>11} {'u_below':>11} {'u_above':>11} {'ovl_frac':>8}")
rows = []
for lab, vt in zip(SLAB, TARGET_V):
    uf=fF(vt); uo=fO(vt); ub=fB(vt); ua=fA(vt)
    print(f"{lab:>5} {vt:>6} {uf:>11.4e} {uo:>11.4e} {ub:>11.4e} {ua:>11.4e} {uo/uf:>8.3f}")
    rows.append((lab, vt, uf, uo, ub, ua))

# write CSV
import csv
with open('/tmp/claude-10396/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/ab44c098-ca8e-4ede-a602-b922281a5709/scratchpad/u_cmfgen.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['shell','v_kms','u_full','u_overlap','u_below_1p5e14','u_above_3e16'])
    for r in rows: w.writerow(r)
print("\n[wrote] u_cmfgen.csv")
