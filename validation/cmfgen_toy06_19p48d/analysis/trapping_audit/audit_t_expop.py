#!/usr/bin/env python3
"""Audit T (Lumina line opacity): expansion-opacity tau in the FUV band and a
Rosseland-weighted mean at T=13120K, per shell s0-s6, using the RUN's own level
populations (lumina_levelpop.csv) and the transport line list, with Lumina's exact
Sobolev formula (lumina_plasma.c:10950, SOBOLEV_COEFF=2.6540281e-2)."""
import numpy as np, csv, time
BASE='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/'
SOB=2.6540281e-2; SIGMA_T=6.6524587e-25
C=2.99792458e10; C_A=2.99792458e18
H=6.62607015e-27; KB=1.380649e-16
T_EXP=19.48*86400.0
NSH=7  # s0..s6
t0=time.time()

# --- geometry + n_e ---
geo={};
with open(BASE+'data/tardis_reference_toy06_19p48d/geometry.csv') as f:
    r=csv.DictReader(f)
    for row in r: geo[int(row['shell_id'])]=(float(row['r_inner']),float(row['r_outer']))
ne={}
with open(BASE+'logs/coevolve_consume_a10_kx_gphall/lumina_plasma_state.csv') as f:
    r=csv.DictReader(f)
    for row in r: ne[int(row['shell_id'])]=float(row['n_e'])
nshell_tot=len(geo)
dr=np.array([geo[s][1]-geo[s][0] for s in range(nshell_tot)])

# --- level pops: build key->(n_k per shell 0..NSH-1), g ---
def key(Z,ion,lev): return (Z*100+ion)*100000+lev
pop={}   # key -> np.array(NSH) number density
gof={}   # key -> g
print(f"[{time.time()-t0:.0f}s] loading levelpop...")
with open(BASE+'logs/coevolve_consume_a10_kx_gphall/lumina_levelpop.csv') as f:
    r=csv.reader(f); hdr=next(r)
    ci={h:i for i,h in enumerate(hdr)}
    for row in r:
        s=int(row[ci['shell']])
        if s>=NSH: continue
        Z=int(row[ci['Z']]); ion=int(row[ci['ion']]); lev=int(row[ci['level_num']])
        k=key(Z,ion,lev)
        if k not in pop:
            pop[k]=np.zeros(NSH); gof[k]=int(row[ci['g']])
        pop[k][s]=float(row[ci['n_k']])
print(f"[{time.time()-t0:.0f}s] levelpop: {len(pop)} unique levels (shells 0..{NSH-1})")

# --- line list (needed cols) via pandas ---
import pandas as pd
print(f"[{time.time()-t0:.0f}s] loading line_list...")
ll=pd.read_csv(BASE+'data/tardis_reference_cmfgen_superlev_ionfix_ddc15strat/line_list.csv',
    usecols=['atomic_number','ion_number','level_number_lower','level_number_upper','f_lu','wavelength','nu'])
Z=ll['atomic_number'].to_numpy(); ion=ll['ion_number'].to_numpy()
lo=ll['level_number_lower'].to_numpy(); up=ll['level_number_upper'].to_numpy()
flu=ll['f_lu'].to_numpy(); lam_A=ll['wavelength'].to_numpy(); nu=ll['nu'].to_numpy()
lam_cm=lam_A*1e-8
klo=(Z*100+ion)*100000+lo; kup=(Z*100+ion)*100000+up
nl=len(Z); print(f"[{time.time()-t0:.0f}s] {nl} lines")

# vectorized lookup: build sorted key arrays
keys=np.fromiter(pop.keys(),dtype=np.int64)
order=np.argsort(keys); keys_s=keys[order]
POP=np.array([pop[k] for k in keys],dtype=np.float64)[order]   # [nk, NSH]
G  =np.array([gof[k] for k in keys],dtype=np.float64)[order]
def lookup(karr):
    idx=np.searchsorted(keys_s,karr)
    idx=np.clip(idx,0,len(keys_s)-1)
    hit=keys_s[idx]==karr
    return idx,hit
ilo,hlo=lookup(klo); iup,hup=lookup(kup)
# n_lower, n_upper per shell; g
nlow=np.where(hlo[:,None],POP[ilo],1e-30)       # [nl,NSH]
nupp=np.where(hup[:,None],POP[iup],1e-30)
glo=np.where(hlo,G[ilo],1.0); gup=np.where(hup,G[iup],1.0)
print(f"[{time.time()-t0:.0f}s] lower-level pop hits: {hlo.mean()*100:.1f}%  upper: {hup.mean()*100:.1f}%")

# --- per-shell tau_sob, expansion opacity binned, Rosseland + FUV ---
# Rosseland bins over Lumina data range
nu_lo,nu_hi=1.5e14,3.0e16
nbin=2000
edges=np.logspace(np.log10(nu_lo),np.log10(nu_hi),nbin+1)
nuc=np.sqrt(edges[:-1]*edges[1:]); dnu=np.diff(edges)
bidx=np.searchsorted(edges,nu)-1
inb=(bidx>=0)&(bidx<nbin)
# FUV band mask (rest wavelength 918-1290)
fuv=(lam_A>=918.0)&(lam_A<=1290.0)
dnu_fuv=C_A/918.0 - C_A/1290.0

T_R=13120.0
def dBdT(nu_,T):
    x=H*nu_/(KB*T); x=np.clip(x,1e-8,700)
    ex=np.exp(x)
    return (2*H*nu_**3/C**2)*(x*ex/((ex-1)**2))/T

res=[]
for s in range(NSH):
    stim=1.0-(glo*nupp[:,s])/(gup*nlow[:,s]); stim=np.clip(stim,0,None)
    tau=SOB*flu*lam_cm*T_EXP*nlow[:,s]*stim
    onemx=1.0-np.exp(-tau)
    # expansion opacity per bin
    w=(nu/dnu[np.clip(bidx,0,nbin-1)])*onemx
    kap_line=np.zeros(nbin)
    np.add.at(kap_line, bidx[inb], (w[inb]))
    kap_line/= (C*T_EXP)   # cm^-1 ; the (nu_j/dnu_bin) already per-bin
    kap_es=ne[s]*SIGMA_T
    kap_tot=kap_line+kap_es
    # Rosseland mean over bins
    wgt=dBdT(nuc,T_R)*dnu
    kap_ross=wgt.sum()/np.sum(wgt/kap_tot)
    kap_ross_es=kap_es  # floor
    # FUV band expansion opacity (single bin over whole band)
    kfuv=(np.sum((nu[fuv]/dnu_fuv)*onemx[fuv]))/(C*T_EXP)+kap_es
    res.append((s,kap_ross,kap_es,kfuv,kap_line.max()))
    print(f"  s{s}: kap_Ross={kap_ross:.3e}  kap_es={kap_es:.3e}  kap_FUV={kfuv:.3e} cm^-1")

# cumulative outward tau (from shell inner edge) using per-shell kappa * dr,
# extend kappa of s6 outward is not computed; report tau over s0..s6 span + es beyond
# For a clean outward tau to surface we need kappa for all shells; we have es for all.
# Line kappa only for s0..s6 -> report tau_line(s0..s6 contribution) + full es outward.
tau_es_out=np.array([ (np.array([ne[k]*SIGMA_T for k in range(nshell_tot)])[i:]*dr[i:]).sum() for i in range(nshell_tot)])
# Ross/FUV: cumulative within s0..s6 (line part) + es floor to surface
kap_ross_arr=np.array([res[s][1] for s in range(NSH)])
kap_fuv_arr =np.array([res[s][3] for s in range(NSH)])
tau_ross_out=np.array([ (kap_ross_arr[i:]*dr[i:NSH]).sum()+tau_es_out[NSH] for i in range(NSH)])
tau_fuv_out =np.array([ (kap_fuv_arr[i:]*dr[i:NSH]).sum()+tau_es_out[NSH]  for i in range(NSH)])

print("\n# Lumina outward optical depth (line part s0..s6 + es floor beyond s6)")
print(f"{'shell':>5} {'tau_Ross':>9} {'tau_FUV':>9} {'tau_es':>8}")
out=[]
for s in range(NSH):
    print(f"s{s:>4} {tau_ross_out[s]:>9.3f} {tau_fuv_out[s]:>9.3f} {tau_es_out[s]:>8.3f}")
    out.append((s,4264+728*s,tau_ross_out[s],tau_fuv_out[s],tau_es_out[s],res[s][1],res[s][3],res[s][2]))
with open('/tmp/claude-10396/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/ab44c098-ca8e-4ede-a602-b922281a5709/scratchpad/tau_lumina_line.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['shell','v','tau_Ross_out','tau_FUV_out','tau_es_out','kap_Ross','kap_FUV','kap_es'])
    for r in out: w.writerow(r)
print(f"\n[{time.time()-t0:.0f}s] [wrote] tau_lumina_line.csv")
