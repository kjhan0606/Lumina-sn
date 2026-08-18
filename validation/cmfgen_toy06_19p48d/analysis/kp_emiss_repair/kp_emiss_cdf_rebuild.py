#!/usr/bin/env python3
"""
kp_emiss CDF rebuild — SE pops vs dilute-Boltzmann, offline, s0.

Reproduces plasma.c:2095-2135 (kp_emiss[dst] += n_lower*C_up*dE) exactly for the
k-packet re-excitation weight, built two ways:
  DB : n_lower = dilute-Boltzmann  (the CONVICTED build, plasma.c:2117-2126)
  SE : n_lower = SE level pop n_k  (the stage4 run's NLTE pops, component (i))
Only n_lower differs; C_up (plasma.c:2129-2131), dE, the line set are identical.

Also computes the k-packet continuum split C_ff, C_fb (plasma.c:2243-2268, Kramers
branch = the run's active branch, FB-MULTI print absent) so the continuum share
(C_ff+C_fb)/(C_ff+C_fb+tot) can be evaluated for both builds -> component (ii).

Read-only. No source edits. Every constant cited from src/lumina.h & lumina_plasma.c.
"""
import numpy as np, pandas as pd, sys

DATA = "data/tardis_reference_toy06_19p48d"
RUN  = "logs/coevolve_consume_a10_kx_stage4"
OUT  = "validation/cmfgen_toy06_19p48d/analysis/kp_emiss_repair"
S0   = 0

# ---- constants (src/lumina.h, lumina_plasma.c:1770) ----
H_PLANCK   = 6.62607015e-27      # erg s
K_B        = 1.380649e-16        # erg/K
EV_TO_ERG  = 1.602176634e-12
VAN_REG    = 2.16e-6
AX_OMEGA   = 1.0

# ---- s0 plasma state (lumina_plasma_state.csv) ----
ps = pd.read_csv(f"{RUN}/lumina_plasma_state.csv")
row = ps[ps.shell_id == S0].iloc[0]
W, T_rad, n_e, T_e = row.W, row.T_rad, row.n_e, row.T_e
inv_sqrt_Te = 1.0/np.sqrt(T_e)
kTe = K_B*T_e
print(f"s0: W={W:.4f} T_rad={T_rad:.1f} n_e={n_e:.3e} T_e={T_e:.1f}")

# ---- ion densities (lumina_ion_pops.csv) ----
ip = pd.read_csv(f"{RUN}/lumina_ion_pops.csv")
ip0 = ip[ip.shell_id == S0]
nion = {(int(z),int(st)):float(n) for z,st,n in zip(ip0.Z, ip0.stage, ip0.n_ion)}

# ---- levels.csv: (Z,ion,lvl) -> E_eV,g,meta ; per-ion partition Z_tot(T_rad,W) ----
lv = pd.read_csv(f"{DATA}/levels.csv")
lv_key = list(zip(lv.atomic_number, lv.ion_number, lv.level_number))
E_of   = dict(zip(lv_key, lv.energy_eV.astype(float)))
g_of   = dict(zip(lv_key, lv.g.astype(float)))
meta_of= dict(zip(lv_key, lv.metastable.astype(int)))

# partition functions per ion (compute_partition_functions, plasma.c:494-527)
Ztot = {}
for (z,ion), grp in lv.groupby(["atomic_number","ion_number"]):
    boltz = grp.energy_eV.values*EV_TO_ERG/(K_B*T_rad)
    bf = np.where(boltz<500.0, grp.g.values*np.exp(-np.clip(boltz,0,700)), 0.0)
    ismeta = grp.metastable.values.astype(bool)
    Zt = bf[ismeta].sum() + W*bf[~ismeta].sum()
    Ztot[(int(z),int(ion))] = max(Zt, 1e-300)

# ---- SE level pops (lumina_levelpop.csv, shell 0) : (Z,ion,lvl)->n_k ----
usecols = ["shell","Z","ion","level_num","n_k"]
lp = pd.read_csv(f"{RUN}/lumina_levelpop.csv", usecols=usecols)
lp0 = lp[lp.shell == S0]
nk_SE = {(int(z),int(io),int(l)):float(n)
         for z,io,l,n in zip(lp0.Z, lp0.ion, lp0.level_num, lp0.n_k)}

# ---- line list (up-transitions) ----
ll = pd.read_csv(f"{DATA}/line_list.csv",
                 usecols=["atomic_number","ion_number","level_number_lower",
                          "level_number_upper","f_lu","nu","wavelength"])
Z   = ll.atomic_number.values.astype(int)
ION = ll.ion_number.values.astype(int)
LO  = ll.level_number_lower.values.astype(int)
f_lu= ll.f_lu.values.astype(float)
nu  = ll.nu.values.astype(float)
wl  = ll.wavelength.values.astype(float)     # Angstrom
dE  = H_PLANCK*nu                            # erg
print(f"lines total: {len(ll):,}")

# keep only lines whose lower ion has n_ion>0 (rest weight ~0 anyway) and lower level known
key_lo = list(zip(Z,ION,LO))
g_lo   = np.array([g_of.get(k, np.nan) for k in key_lo])
E_lo   = np.array([E_of.get(k, np.nan) for k in key_lo])
meta_lo= np.array([meta_of.get(k, 0)   for k in key_lo])
nion_lo= np.array([nion.get((z,io),0.0) for z,io in zip(Z,ION)])

valid = np.isfinite(g_lo) & np.isfinite(E_lo) & (g_lo>0) & (nion_lo>0)
print(f"lines with n_ion(lower)>0 & lower level resolved: {valid.sum():,}")

# ---- C_up (plasma.c:2129-2131), identical for DB & SE ----
exp_up = np.exp(-np.clip(dE/kTe, 0, 700))
C_up = np.where(f_lu>1e-10,
                VAN_REG*n_e*f_lu*exp_up*0.2*inv_sqrt_Te/np.where(g_lo>0,g_lo,1),
                8.63e-6*n_e*AX_OMEGA*exp_up*inv_sqrt_Te/np.where(g_lo>0,g_lo,1))

# ---- n_lower DB (plasma.c:2117-2126) ----
boltz_lo = E_lo*EV_TO_ERG/(K_B*T_rad)
wgt = np.where(meta_lo==1, 1.0, W)
Ztot_lo = np.array([Ztot.get((z,io),1e-300) for z,io in zip(Z,ION)])
nlow_DB = np.where(boltz_lo<500.0,
                   nion_lo*wgt*g_lo*np.exp(-np.clip(boltz_lo,0,700))/Ztot_lo, 0.0)

# ---- n_lower SE (levelpop n_k) ----
nlow_SE = np.array([nk_SE.get((z,io,lo),0.0) for z,io,lo in key_lo])

# ---- kp_emiss weight per line ----
w_base = C_up*dE
w_DB = np.where(valid, nlow_DB*w_base, 0.0)
w_SE = np.where(valid, nlow_SE*w_base, 0.0)
tot_DB = w_DB.sum(); tot_SE = w_SE.sum()

# SE coverage: which valid lines have an SE pop present in levelpop
se_present = np.array([ (z,io,lo) in nk_SE for z,io,lo in key_lo])
cov = valid & se_present
print(f"\ntot_DB (all valid) = {tot_DB:.4e} erg/cm3/s   [C_collexc]")
print(f"tot_SE (all valid) = {tot_SE:.4e} erg/cm3/s")
print(f"SE-covered valid lines: {cov.sum():,}/{valid.sum():,}"
      f"  DB weight covered = {w_DB[cov].sum()/tot_DB*100:.2f}%")
print(f"tot_DB (SE-covered set) = {w_DB[cov].sum():.4e}")
print(f"tot_SE (SE-covered set) = {w_SE[cov].sum():.4e}")

# ---- continuum split C_ff, C_fb (plasma.c:2243-2268, Kramers = run's branch) ----
sum_z2n = 0.0; C_fb = 0.0
te4 = (T_e/1e4)**(-0.75)
for (z,st),n in nion.items():
    if st<=0 or n<=0: continue
    sum_z2n += st*st*n
    alpha = 2.6e-13*st*st*te4
    C_fb += alpha*n*n_e*kTe
C_ff = 1.426e-27*np.sqrt(T_e)*n_e*sum_z2n
print(f"\nC_ff = {C_ff:.4e}   C_fb(Kramers) = {C_fb:.4e}   C_cont = {C_ff+C_fb:.4e}")

def cont_share(tot):
    d = C_ff+C_fb+tot
    return (C_ff+C_fb)/d if d>0 else 0.0
print(f"continuum share  DB: {cont_share(tot_DB):.3e}  ({cont_share(tot_DB)*100:.4f}%)")
print(f"continuum share  SE: {cont_share(tot_SE):.3e}  ({cont_share(tot_SE)*100:.4f}%)")
print(f"  (measured event-log ff+fb exit share at s0 corpse ~ 2e-4 = 0.02%)")

# ---- band shares ----
bands = [("EUV_300_450",300,450),("EUV_450_912",450,912),
         ("dFUV_912_1290",912,1290),("CoIV_pile_1290_2000",1290,2000),
         ("pile_core_1490_1650",1490,1650),("valley_1650_2100",1650,2100),
         ("NUV_2100_3200",2100,3200),("opt_3200_7000",3200,7000),
         ("IR_gt7000",7000,1e9)]
def band_table(w, tot):
    rows=[]
    for name,a,b in bands:
        m = (wl>=a)&(wl<b)
        rows.append((name, w[m].sum(), 100*w[m].sum()/tot if tot>0 else 0))
    return rows
print("\nBAND SHARES (fraction of line-emissivity tot):")
print(f"{'band':22s} {'DB %':>9s} {'SE %':>9s}")
bt_DB = band_table(w_DB, tot_DB); bt_SE = band_table(w_SE, tot_SE)
band_rows=[]
for (n,wd,pd_),(_,ws,ps_) in zip(bt_DB,bt_SE):
    print(f"{n:22s} {pd_:9.3f} {ps_:9.3f}")
    band_rows.append((n,wd,pd_,ws,ps_))

# ---- ion shares (which ion dominates the line emissivity = the attractor) ----
def ion_table(w, tot):
    d={}
    for zz,io,ww in zip(Z,ION,w):
        if ww<=0: continue
        d[(zz,io)] = d.get((zz,io),0)+ww
    return sorted(([f"{z}_{io}",v,100*v/tot] for (z,io),v in d.items()),
                  key=lambda r:-r[1])[:8]
print("\nTOP EMITTING IONS (Z_ion, 0-based ion; IV=ion3):")
print(f"{'ion':8s} {'DB %':>9s}    {'ion':8s} {'SE %':>9s}")
it_DB=ion_table(w_DB,tot_DB); it_SE=ion_table(w_SE,tot_SE)
for a,b in zip(it_DB,it_SE):
    print(f"{a[0]:8s} {a[2]:9.3f}    {b[0]:8s} {b[2]:9.3f}")

# ---- write CSVs ----
pd.DataFrame(band_rows, columns=["band","w_DB","pct_DB","w_SE","pct_SE"]).to_csv(
    f"{OUT}/band_shares_s0.csv", index=False)
pd.DataFrame([("tot_DB",tot_DB),("tot_SE",tot_SE),("C_ff",C_ff),("C_fb_kramers",C_fb),
              ("cont_share_DB",cont_share(tot_DB)),("cont_share_SE",cont_share(tot_SE)),
              ("W",W),("T_rad",T_rad),("T_e",T_e),("n_e",n_e),
              ("tot_DB_SEcov",w_DB[cov].sum()),("tot_SE_SEcov",w_SE[cov].sum())],
             columns=["quantity","value"]).to_csv(f"{OUT}/continuum_split_s0.csv", index=False)
allion = sorted(set(zip(Z.tolist(),ION.tolist())))
pd.DataFrame([[f"{z}_{io}",
               np.where((Z==z)&(ION==io),w_DB,0).sum(),
               np.where((Z==z)&(ION==io),w_SE,0).sum()] for z,io in allion
              if np.where((Z==z)&(ION==io),w_DB,0).sum()>0],
             columns=["ion_0based","w_DB","w_SE"]).to_csv(f"{OUT}/ion_shares_s0.csv", index=False)
print(f"\nwrote band_shares_s0.csv, continuum_split_s0.csv, ion_shares_s0.csv to {OUT}")
