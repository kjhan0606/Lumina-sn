#!/usr/bin/env python3
"""Offline radeq ledger at s0 (v=4264) for the B-run coevolve_consume_a10_kx_gphall.
Reproduces simul_r1 term structure with the run's own field + ion pops.
Read-only. No source edits."""
import csv, math, numpy as np, struct, os

H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10; EV=1.602176634e-12
AMU=1.66053906660e-24; A_RAD=7.5657e-15; RY_EV=13.605693
T_EXP=19.48*86400.0
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUN=f"{REPO}/logs/coevolve_consume_a10_kx_gphall"
BASE=f"{REPO}/data/tardis_reference_toy06_19p48d"
S=0
Te=13119.874754; ne=4.426076e9   # run committed s0

# ---- field s0 ----
nu=[]; csJ=[]; mcJ=[]
for r in csv.DictReader(open(f"{RUN}/lumina_coevolve_field.csv")):
    if int(r['shell'])!=S: continue
    lam=float(r['wavelength_A'])
    nu.append(C/(lam*1e-8)); csJ.append(float(r['cs_J'])); mcJ.append(float(r['mc_J']))
nu=np.array(nu); csJ=np.array(csJ); mcJ=np.array(mcJ)
# sort ascending in nu
o=np.argsort(nu); nu=nu[o]; csJ=csJ[o]; mcJ=mcJ[o]

def energy_density(J):
    return 4*math.pi/C*np.trapz(J,nu)
u_mc=energy_density(mcJ); u_cs=energy_density(csJ)
print(f"# u(s0) from mc_J = {u_mc:.2f}   from cs_J = {u_cs:.2f}   erg/cm3")
print(f"# a*Te^4 (Te={Te:.0f}) = {A_RAD*Te**4:.2f}  -> u_mc/(aTe^4) = {u_mc/(A_RAD*Te**4):.3f}")
print(f"# bath-equiv T = (u_mc/a)^0.25 = {(u_mc/A_RAD)**0.25:.0f} K")

# ---- Planck comparison / band decomposition ----
def planck_u(T):
    x=H*nu/(KB*T)
    B=2*H*nu**3/C**2/(np.exp(np.minimum(x,700))-1.0)
    return 4*math.pi/C*np.trapz(B,nu), B
uP13,B13=planck_u(Te)
uP18,B18=planck_u(18760.0)
print(f"# integ over Lumina nu-grid: Planck u(13120)={uP13:.2f}  Planck u(18760)={uP18:.2f}")

# band edges (Angstrom) -> we report energy density fraction per band
# FUV 918-1290, EUV<912, optical 3000-7000, red/IR >7000, NUV 1290-3000
lam=C/nu*1e8
def uband(J,lo,hi):
    m=(lam>=lo)&(lam<hi)
    if m.sum()<2: return 0.0
    return 4*math.pi/C*np.trapz(J[m],nu[m])
bands=[("EUV<912",0,912),("FUV918-1290",912,1290),("NUV1290-3000",1290,3000),
       ("opt3000-7000",3000,7000),("red/IR>7000",7000,1e9)]
print("\n# BAND-RESOLVED energy density u [erg/cm3]  (mc_J bath vs Planck@Te vs Planck@18760)")
print(f"# {'band':16s} {'u_mc':>10s} {'u_Planck13':>11s} {'u_Planck18':>11s} {'mc/P13':>7s}")
for nm,lo,hi in bands:
    a=uband(mcJ,lo,hi); b=uband(B13,lo,hi); c=uband(B18,lo,hi)
    print(f"# {nm:16s} {a:10.2f} {b:11.2f} {c:11.2f} {(a/b if b>0 else 0):7.2f}")
uexcess=u_mc-uP13
print(f"# TOTAL super-thermal excess u_mc - aTe^4 = {u_mc-A_RAD*Te**4:.1f} erg/cm3")
print(f"#          u_mc - Planck13(gridint)      = {uexcess:.1f} erg/cm3")

# ---- ionization energies ----
chi={}
for r in csv.DictReader(open(f"{BASE}/ionization_energies.csv")):
    chi[(int(r['atomic_number']),int(r['ion_number']))]=float(r['ionization_energy_eV'])

# ---- ion pops at s0 ----
npops={}
for r in csv.DictReader(open(f"{RUN}/lumina_ion_pops.csv")):
    if int(r['shell_id'])!=S: continue
    npops[(int(r['Z']),int(r['stage']))]=float(r['n_ion'])

# ---- jtable (CMFGEN J on run grid) for counterfactual ----
def load_jtable():
    p=f"{REPO}/data/cmfgen_jtable_toy06_19p48d.bin"
    with open(p,'rb') as f:
        magic,ver,nsh,nfb=struct.unpack('4i',f.read(16))
        data=np.frombuffer(f.read(),dtype=np.float64)
    data=data.reshape(nsh,nfb)
    return data,nfb
jt,nfb=load_jtable()
# jtable grid: nu_min=1.5e14 nu_max=3e16 d_log_nu, bin center nu_min*exp((bb+0.5)*dln)
NU_MIN=1.5e14; NU_MAX=3e16
dln=math.log(NU_MAX/NU_MIN)/nfb
jt_nu=np.array([NU_MIN*math.exp((bb+0.5)*dln) for bb in range(nfb)])
jt_s0=jt[S]
u_jt=4*math.pi/C*np.trapz(jt_s0[jt_s0>0], jt_nu[jt_s0>0])
print(f"\n# jtable s0: nonzero bins={np.sum(jt_s0>0)}  u(CMFGEN jtable, grid-restricted)={u_jt:.2f}")
# note jtable only covers where CMFGEN had data; compare band FUV
print(f"# jtable FUV band u = {4*math.pi/C*np.trapz(jt_s0[(C/jt_nu*1e8>=918)&(C/jt_nu*1e8<=1290)], jt_nu[(C/jt_nu*1e8>=918)&(C/jt_nu*1e8<=1290)]):.3f}")

# ---- H_photo (bf photoheating) ground-state Kramers, per simul_r1 ELSE-branch formula ----
# G  = sum 4pi sig J/(h nu) dnu ; Hex = sum 4pi sig J/(h nu)(h nu - chi) dnu
# sig = 7.91e-18/zeff^2 (nu0/nu)^3 ; zeff = stage+1
def H_photo_ground(Jgrid, Jnu_grid, use_grid_nu):
    """Jgrid on given nu grid. Returns total H_photo and per-band."""
    Htot=0.0; Gtot=0.0
    perband={nm:0.0 for nm,_,_ in bands}
    # integrate on the field nu grid (ascending). dnu via diff.
    ng=Jnu_grid
    dnu=np.gradient(ng)
    lamg=C/ng*1e8
    for (Z,st),n_ion in npops.items():
        if n_ion<=0: continue
        ch=chi.get((Z,st))
        if ch is None: continue
        nu0=ch*EV/H
        zeff=st+1
        m=ng>=nu0
        if not m.any(): continue
        sig=7.91e-18/zeff**2*(nu0/ng[m])**3
        w=4*math.pi*sig*Jgrid[m]/(H*ng[m])*dnu[m]
        G=np.sum(w); Hx=np.sum(w*(H*ng[m]-ch*EV))
        Gtot+=n_ion*G  # not physically summed like this but for scale
        Htot+=n_ion*Hx
        # band attribution of Hx
        for nm,lo,hi in bands:
            bm=(lamg[m]>=lo)&(lamg[m]<hi)
            perband[nm]+=n_ion*np.sum((w*(H*ng[m]-ch*EV))[bm])
    return Htot,perband

Hph_mc,pb_mc=H_photo_ground(mcJ,nu,True)
Hph_cs,pb_cs=H_photo_ground(csJ,nu,True)
# jtable on its own grid
def H_photo_ground_jt():
    Htot=0.0; perband={nm:0.0 for nm,_,_ in bands}
    ng=jt_nu; dnu=np.gradient(ng); lamg=C/ng*1e8
    J=jt_s0
    for (Z,st),n_ion in npops.items():
        if n_ion<=0: continue
        ch=chi.get((Z,st));
        if ch is None: continue
        nu0=ch*EV/H; zeff=st+1
        m=(ng>=nu0)&(J>0)
        if not m.any(): continue
        sig=7.91e-18/zeff**2*(nu0/ng[m])**3
        w=4*math.pi*sig*J[m]/(H*ng[m])*dnu[m]
        Hx=np.sum(w*(H*ng[m]-ch*EV))
        Htot+=n_ion*Hx
        for nm,lo,hi in bands:
            bm=(lamg[m]>=lo)&(lamg[m]<hi)
            perband[nm]+=n_ion*np.sum((w*(H*ng[m]-ch*EV))[bm])
    return Htot,perband
Hph_jt,pb_jt=H_photo_ground_jt()

print("\n# H_photo (GROUND-state Kramers, sum over all ion pairs at s0) [erg/cm3/s]")
print(f"#   with mc_J  = {Hph_mc:.3e}")
print(f"#   with cs_J  = {Hph_cs:.3e}")
print(f"#   with CMFGEN jtable = {Hph_jt:.3e}")
print(f"#   (all-level enhances rate ~40x for Fe III per log; heating excess less)")
print("\n# H_photo band attribution (mc_J):")
for nm,_,_ in bands: print(f"#   {nm:16s} {pb_mc[nm]:.3e}")
print("# H_photo band attribution (CMFGEN jtable):")
for nm,_,_ in bands: print(f"#   {nm:16s} {pb_jt[nm]:.3e}")

# ---- continuum cooling terms (exact simul_r1) ----
H_dep=1.506865e-03
C_ff=1.426e-27*1.2*ne*ne*math.sqrt(Te)
Gamma_ad=3.0/T_EXP
C_ad=1.5*ne*KB*Te*Gamma_ad
print(f"\n# --- exact continuum terms at Te={Te:.0f}, ne={ne:.3e} ---")
print(f"# H_dep (CMFGEN edep, injected)      = {H_dep:.3e}")
print(f"# C_ff  = 1.7112e-27 ne^2 sqrt(T)    = {C_ff:.3e}")
print(f"# C_ad  = 1.5 ne k T (3/texp)        = {C_ad:.3e}")

# ---- C_fb (approx alpha RR) ----
def alpha_rr(Z,zrec,T):  # hydrogenic RR only (Fe/Co/Ni: no DR table here)
    return 2.6e-13*zrec**1.6*(T/1e4)**-0.8
C_fb=0.0
for (Z,st),n_lo in npops.items():
    nx=npops.get((Z,st+1),0.0)
    if nx<=0: continue
    ch=chi.get((Z,st))
    if ch is None: continue
    al=alpha_rr(Z,st+1,Te)
    C_fb+=ne*nx*al*(ch*EV+KB*Te)
print(f"# C_fb  (approx RR, chi+kT weight)   = {C_fb:.3e}")

print(f"\n# HEATING total (dep + H_photo[mc,ground]) = {H_dep+max(Hph_mc,0):.3e}")
print(f"# non-line COOLING (ff+ad+fb)             = {C_ff+C_ad+C_fb:.3e}")
print(f"# => residual Lambda_line at root (=H-Cnl) = {H_dep+max(Hph_mc,0)-(C_ff+C_ad+C_fb):.3e}")
