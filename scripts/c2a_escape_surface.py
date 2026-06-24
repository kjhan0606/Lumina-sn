#!/usr/bin/env python3
"""C2a: per-frequency tau=2/3 escape surface and its T_e vs wavelength.
too-red test: do blue photons escape from DEEP COLD layers (low escape-T) while
gold's blue escapes near the photosphere? Escape-surface T(lambda) IS the
emergent color temperature per wavelength.

Usage: c2a_escape_surface.py <run_dir>
  run_dir must contain lumina_cmfgen_jnu.csv (LUMINA_CMFGEN_JDUMP=1),
  ref/geometry.csv, lumina_plasma_state.csv.
Gold photosphere T from data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d.
"""
import sys, numpy as np, csv, os
RUN = sys.argv[1] if len(sys.argv) > 1 else '.'
C = 2.99792458e18  # Ang/s

# --- geometry: r_outer per shell (cm), inner shell index 0 ---
gid=[]; rout=[]; rin=[]; vin=[]; vout=[]
for r in csv.DictReader(open(os.path.join(RUN,'ref','geometry.csv'))):
    gid.append(int(r['shell_id'])); rin.append(float(r['r_inner'])); rout.append(float(r['r_outer']))
    vin.append(float(r['v_inner'])); vout.append(float(r['v_outer']))
NS=len(gid); rin=np.array(rin); rout=np.array(rout); dr=rout-rin
vmid=(np.array(vin)+np.array(vout))/2/1e5  # km/s

# --- T_e per shell ---
Te=np.zeros(NS)
for r in csv.DictReader(open(os.path.join(RUN,'lumina_plasma_state.csv'))):
    Te[int(r['shell_id'])]=float(r['T_e'])

# --- opacity dump: shell,bin,nu,J,chi_es,chi_abs,chi_line,chi_tot,... ---
rows=list(csv.DictReader(open(os.path.join(RUN,'lumina_cmfgen_jnu.csv'))))
NB=max(int(x['bin']) for x in rows)+1
nu=np.zeros(NB); chi_es=np.zeros((NS,NB)); chi_abs=np.zeros((NS,NB)); chi_line=np.zeros((NS,NB))
for x in rows:
    s=int(x['shell']); b=int(x['bin']); nu[b]=float(x['nu'])
    chi_es[s,b]=float(x['chi_es']); chi_abs[s,b]=float(x['chi_abs']); chi_line[s,b]=float(x['chi_line'])
chi_cont=chi_es+chi_abs
chi_tot =chi_cont+chi_line
lam=C/nu  # Ang

def escape_shell(chi):
    """For each bin, integrate tau inward from outermost shell; return shell index
    where tau crosses 2/3 (or 0 if never => escapes from core)."""
    out=np.zeros(NB,dtype=int)
    for b in range(NB):
        tau=0.0; hit=-1
        for s in range(NS-1,-1,-1):   # outer -> inner
            tau += chi[s,b]*dr[s]
            if tau>=2.0/3.0: hit=s; break
        out[b]= hit if hit>=0 else 0
    return out

# gold photosphere T (interp to vmid for context)
GOLD='data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d'
def gblock(key,n=115):
    lines=open(GOLD).read().split('\n'); vals=[];grab=False
    for L in lines:
        if key in L: grab=True; continue
        if grab:
            try: row=[float(x) for x in L.split()]
            except:
                if vals: break
                else: continue
            vals+=row
            if len(vals)>=n: break
    return np.array(vals[:n])
gv=gblock('Velocity (km/s)'); gT=gblock('Temperature (10^4 K)')*1e4
o=np.argsort(gv); gv,gT=gv[o],gT[o]

es_cont=escape_shell(chi_cont)
es_tot =escape_shell(chi_tot)

print(f"RUN={RUN}  NS={NS} NB={NB}")
print(f"\n  lambda   |  CONTINUUM escape          |  TOTAL(+line) escape")
print(f"   (A)     | shell  v(km/s)  Te(K)       | shell  v(km/s)  Te(K)  goldT@v")
band_edges=[3500,4000,4500,5000,5500,6000,6630,7000,8000,9000,12000,18000]
for le in band_edges:
    b=int(np.argmin(np.abs(lam-le)))
    sc=es_cont[b]; st=es_tot[b]
    gTc=np.interp(vmid[st],gv,gT)   # gold T at the TOTAL escape velocity
    print(f"  {lam[b]:7.0f}  |  {sc:3d}  {vmid[sc]:7.0f}  {Te[sc]:6.0f}      |  {st:3d}  {vmid[st]:7.0f}  {Te[st]:6.0f}  {gTc:6.0f}")

# summary: flux-weighted escape-T color across optical, cont vs tot
opt=(lam>=4000)&(lam<=9000)
print(f"\nOptical(4000-9000) mean escape Te: continuum={Te[es_cont[opt]].mean():.0f}K  total={Te[es_tot[opt]].mean():.0f}K")
print(f"  blue(4000-5000) escape Te: cont={Te[es_cont[(lam>=4000)&(lam<5000)]].mean():.0f}K  tot={Te[es_tot[(lam>=4000)&(lam<5000)]].mean():.0f}K")
print(f"  NIR (8000-9000) escape Te: cont={Te[es_cont[(lam>=8000)&(lam<9000)]].mean():.0f}K  tot={Te[es_tot[(lam>=8000)&(lam<9000)]].mean():.0f}K")
print("\nINTERPRET: if blue escape-Te << NIR escape-Te (or << gold photosphere T~4400K),")
print("  blue is escaping from deep COLD layers => emergent blue is thermally too-red.")

# ====================================================================
# C1b: color carrier at the escape surface. S_c = (chi_abs*B + chi_es*J)/chi_c.
#   chi_abs dominant => S_c -> B(T_e)  (color OK if T_e matches gold)
#   chi_es  dominant => S_c -> J       (color = scattered field, can be too-red)
# Evaluate at each band's TOTAL escape shell.
# ====================================================================
print("\n" + "="*64)
print("C1b: continuum color carrier at escape surface  (eps_abs = chi_abs/(chi_abs+chi_es))")
print("  lambda    shell   chi_abs    chi_es     eps_abs   carrier")
for le in [3500,4000,4500,5000,5500,6630,8000,12000]:
    b=int(np.argmin(np.abs(lam-le))); s=es_cont[b]
    ca=chi_abs[s,b]; ce=chi_es[s,b]; eps=ca/(ca+ce) if (ca+ce)>0 else 0.0
    car='THERMAL B(Te)' if eps>0.5 else 'SCATTER J'
    print(f"  {lam[b]:7.0f}  {s:4d}   {ca:.3e}  {ce:.3e}   {eps:5.3f}    {car}")
optb=(lam>=4000)&(lam<=9000)
eps_opt=np.array([chi_abs[es_cont[b],b]/max(chi_abs[es_cont[b],b]+chi_es[es_cont[b],b],1e-300) for b in np.where(optb)[0]])
print(f"  optical(4000-9000) mean eps_abs = {eps_opt.mean():.3f}  "
      f"({'thermal-dominated: color set by B(Te)' if eps_opt.mean()>0.5 else 'scatter-dominated: color set by J'})")

# ====================================================================
# C2b-realized: spectral shape of continuum ABSORPTION opacity chi_abs(nu),
#   shell-by-shell near photosphere. Does a blue bf WALL (3500-4500A) dominate?
#   Compare chi_abs(blue) vs chi_abs(red) at the photosphere shell.
# ====================================================================
print("\n" + "="*64)
print("C2b-realized: chi_abs band ratios vs photosphere (blue bf wall test)")
# photosphere shell = continuum escape shell at 6630A (gold peak)
sphot=es_cont[int(np.argmin(np.abs(lam-6630)))]
print(f"  photosphere (cont escape @6630A) = shell {sphot}, v={vmid[sphot]:.0f} km/s, Te={Te[sphot]:.0f}K")
def bandmean(arr,s,lo,hi):
    m=(lam>=lo)&(lam<hi); return arr[s,m].mean() if m.sum() else 0.0
print("  shell  v(km/s)  Te(K)  | chi_abs: UVblue/red  green/red  NIR/red  (red=6630-9000 ref)")
for s in range(NS-1,-1,-max(1,NS//12)):
    red=bandmean(chi_abs,s,6630,9000)
    if red<=0: red=1e-300
    ub=bandmean(chi_abs,s,3500,4500)/red; gr=bandmean(chi_abs,s,5500,6630)/red; ni=bandmean(chi_abs,s,9000,18000)/red
    print(f"  {s:4d}  {vmid[s]:7.0f}  {Te[s]:6.0f}  |   {ub:8.2f}   {gr:8.2f}  {ni:8.2f}")
# total chi_abs spectral shape summed over shells (where is opacity concentrated)
print("\n  chi_abs spectral concentration (sum over shells, by band):")
tot=chi_abs.sum(axis=0)
for lo,hi,lab in [(3500,4500,'UVblue'),(4500,5500,'blue'),(5500,6630,'green'),(6630,9000,'red'),(9000,18000,'NIR')]:
    m=(lam>=lo)&(lam<hi); print(f"    {lab:7s} {lo}-{hi}: mean chi_abs = {tot[m].mean():.3e}")
print("\nINTERPRET C2b: if chi_abs(UVblue)/chi_abs(red) >> 1 at photosphere, a bf wall")
print("  traps blue => thermalized => NIR re-emission (the too-red mechanism).")
