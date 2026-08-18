#!/usr/bin/env python3
"""Driver number #1: CMFGEN DEEP-shell (v~4264, s0-equiv) fractional CONTINUUM (ff+bf)
share of emissivity vs lines, integrated over 300-2000 A -- calibrates Lumina's
k-packet continuum branch.

Method: at the CMFGEN depth nearest v=4264 km/s, over 300-2000 A (CMF), decompose the
total emissivity eta_nu into a smooth CONTINUUM FLOOR (rolling low-percentile in nu)
and LINE spikes (eta_total - floor). The floor includes coherent e-scattering
(eta_es ~ (chi_es/chi_floor)*eta_floor); we subtract it via chi_es = n_e sigma_T so the
reported ff+bf is the THERMAL continuum only (e-scattering is NOT a k-packet channel).
Deep tau>>1 => LTE (S~B) => eta split ~ chi split, so the es removal is consistent.
"""
import numpy as np, sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import cmfgen_formation_map as M
D=sys.argv[1] if len(sys.argv)>1 else "/gpfs/kjhan/cmfgen_runs/toy06_19.48d_cmfflux"
SIGMA_T=6.6524587e-25   # cm^2
VTARGET=4264.0
BANDLO,BANDHI=300.0,2000.0

eta_r,ND,_=M.read_edd_records(D+'/ETA_DATA')
chi_r,_,_=M.read_edd_records(D+'/CHI_DATA')
NU=eta_r[:,ND]; lam=M.CLIGHT_A/NU
eta=eta_r[:,:ND]; chi=chi_r[:,:ND]
rt=open(D+'/RVTJ').read()
V=M.parse_rvtj(rt,'Velocity (km/s)',ND)
ED=M.parse_rvtj(rt,'Electron density',ND)   # cm^-3
T=M.parse_rvtj(rt,'Temperature (10^4K)',ND)*1e4

d=int(np.argmin(np.abs(V-VTARGET)))
print(f"deep shell: depth {d}  V={V[d]:.0f} km/s (target {VTARGET})  T={T[d]:.0f} K  n_e={ED[d]:.3e} cm^-3")

# band mask, sort by nu ascending
m=(lam>=BANDLO)&(lam<BANDHI)
nu_b=NU[m]; e_b=eta[m,d]; c_b=chi[m,d]
o=np.argsort(nu_b); nu_b=nu_b[o]; e_b=e_b[o]; c_b=c_b[o]
nuHz=nu_b*1e15
print(f"band {BANDLO}-{BANDHI} A at depth {d}: {nu_b.size} freq points")

# continuum FLOOR: rolling low-percentile envelope over a sliding nu window
def floor_env(x, win=201, pct=15):
    n=x.size; out=np.empty(n)
    h=win//2
    for i in range(n):
        a=max(0,i-h); b=min(n,i+h+1)
        out[i]=np.percentile(x[a:b],pct)
    return out
e_floor=np.minimum(floor_env(e_b), e_b)     # continuum floor of emissivity (incl es)
c_floor=np.minimum(floor_env(c_b), c_b)     # continuum floor of opacity   (incl es)

# electron-scattering opacity (per cm), same units as chi? chi is per 10^10 cm =>
# chi_es[per 1e10cm] = n_e*sigma_T*1e10
chi_es = ED[d]*SIGMA_T*1e10
es_share = np.clip(chi_es/np.maximum(c_floor,1e-99),0,1)   # es fraction of continuum floor per nu

# integrate over nu
def I(y): return np.trapezoid(y,nuHz)
E_tot   = I(e_b)
E_floor = I(e_floor)                 # continuum incl es
E_line  = I(np.clip(e_b-e_floor,0,None))
E_es    = I(e_floor*es_share)        # es part of the floor
E_ffbf  = E_floor - E_es             # thermal continuum ff+bf

print("\n=== emissivity budget over 300-2000 A at the DEEP shell (CMF-frame) ===")
print(f"  total emissivity           INT eta dnu = {E_tot:.4e}")
print(f"  LINE (bound-bound)         frac = {100*E_line/E_tot:5.1f}%")
print(f"  CONTINUUM floor (incl e-s) frac = {100*E_floor/E_tot:5.1f}%")
print(f"    of which e-scattering    frac = {100*E_es/E_tot:5.1f}%  (chi_es={chi_es:.3e}/1e10cm)")
print(f"    of which ff+bf (thermal) frac = {100*E_ffbf/E_tot:5.1f}%")
print("\n>>> DRIVER #1 : k-packet continuum branch calibration")
print(f"    continuum(ff+bf) / [continuum(ff+bf)+line]  = {100*E_ffbf/(E_ffbf+E_line):.1f}%")
print(f"    line            / [continuum(ff+bf)+line]  = {100*E_line/(E_ffbf+E_line):.1f}%")
print(f"    (continuum-incl-es / total = {100*E_floor/E_tot:.1f}%  if es counted as continuum)")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cmfgen_deep_continuum_fraction.csv'),'w') as f:
    f.write("quantity,value\n")
    f.write(f"deep_depth_index,{d}\ndeep_v_kms,{V[d]:.0f}\ndeep_T_K,{T[d]:.0f}\ndeep_ne_cm3,{ED[d]:.4e}\n")
    f.write(f"band_A,{BANDLO}-{BANDHI}\n")
    f.write(f"line_frac_of_total,{E_line/E_tot:.4f}\n")
    f.write(f"continuum_incl_es_frac_of_total,{E_floor/E_tot:.4f}\n")
    f.write(f"es_frac_of_total,{E_es/E_tot:.4f}\n")
    f.write(f"ffbf_frac_of_total,{E_ffbf/E_tot:.4f}\n")
    f.write(f"kpacket_continuum_branch_ffbf_vs_line,{E_ffbf/(E_ffbf+E_line):.4f}\n")
print("[wrote] cmfgen_deep_continuum_fraction.csv")
