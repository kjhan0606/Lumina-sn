#!/usr/bin/env python3
"""STAGE4 ROUND2 -- Gph b_k-cap sizing.

Reproduces the all-level population-weighted photoionization rate Gph(III->IV)
exactly as src/lumina_plasma.c computes it in the want_nlte_w path (5527-5578,
NLTE-weighted) vs the Boltzmann path (5581-5627), for the DRAINED III combs
(Fe III 26,2 ; Co III 27,2 ; Ni III 28,2), at deep s0 and photospheric s8, as a
function of a b_k CAP applied to the per-level weighting.

Gph_ratio(C) = G_nlte(cap=C) / G_LTE      (G_LTE == cap=1, pure Boltzmann@T_e)

    G(cap) = SUM_{l: has_sigma} popc_l * w_l ,  popc_l = min(b_l,C) n_l^LTE / SUM_k min(b_k,C) n_k^LTE
    w_l    = INT_{nu>=nu_l} 4pi sigma_l(nu) J(nu) / (h nu) dnu

n_l^LTE = n_k / b_k (dumped).  J(nu) = W*B(nu,T_rad) dilute photospheric proxy
(the run's dilute field, plasma.c:9846); a hot B(nu,T_e_deep) field is also run
as a color-sensitivity bracket.  The RATIO is insensitive to the J NORMALIZATION
(cancels top/bottom); only the field COLOR (which thresholds are lit) matters.

Read-only.  Emits gph_cap_sizing.csv + gph_bk_distribution.csv.
"""
import numpy as np, csv, struct
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF=f"{REPO}/data/tardis_reference_toy06_19p48d"
S4 =f"{REPO}/logs/coevolve_consume_a10_kx_stage4"
OUT=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/stage4_round2"
K_B=1.380649e-16; EV=1.602176634e-12; H=6.62607015e-27; C=2.99792458e10

# ---- ionization energies (III->IV thresholds) ----
ioniz={}
with open(f"{REF}/ionization_energies.csv") as f:
    for row in csv.DictReader(f):
        ioniz[(int(row["atomic_number"]),int(row["ion_number"]))]=float(row["ionization_energy_eV"])

# ---- global level order (levels.csv) -> maps (Z,ion,level_num) to sigma row ----
Zc=[];ionc=[];numc=[];Ec=[];gc=[]
with open(f"{REF}/levels.csv") as f:
    for row in csv.DictReader(f):
        Zc.append(int(row["atomic_number"]));ionc.append(int(row["ion_number"]))
        numc.append(int(row["level_number"]));Ec.append(float(row["energy_eV"]))
        gc.append(int(row["g"]))
Zc=np.array(Zc);ionc=np.array(ionc);numc=np.array(numc);Ec=np.array(Ec);gc=np.array(gc)
NLEV=len(Zc)
gidx={(Zc[i],ionc[i],numc[i]):i for i in range(NLEV)}

# ---- sigma_bf (magic CMFD) ----
with open(f"{REF}/cmfgen_sigma_bf.bin","rb") as f:
    magic,ver,nlev,nfreq=struct.unpack("<IIii",f.read(16))
    nu_min,nu_max=struct.unpack("<dd",f.read(16))
    assert magic==0x434D4644 and nlev==NLEV,(hex(magic),nlev,NLEV)
    flag8=np.frombuffer(f.read(nlev),dtype=np.int8)
    pad=(8-(nlev%8))%8
    if pad: f.read(pad)
    sigma=np.frombuffer(f.read(nlev*nfreq*8),dtype=np.float64).reshape(nlev,nfreq)
has_sigma=flag8.astype(bool)
log_numin=np.log(nu_min); d_log_nu=(np.log(nu_max)-log_numin)/nfreq
bb=np.arange(nfreq); log_nu_lo=log_numin+bb*d_log_nu
nu_c=np.exp(log_nu_lo+0.5*d_log_nu)
dnu=np.exp(log_nu_lo+d_log_nu)-np.exp(log_nu_lo)

# ---- plasma state ----
PS={}
with open(f"{S4}/lumina_plasma_state.csv") as f:
    for row in csv.DictReader(f):
        PS[int(row["shell_id"])]=(float(row["W"]),float(row["T_rad"]),float(row["T_e"]))

# ---- level pops for the III combs at s0,s8 ----
def load_comb(Z,ion,shells):
    out={s:[] for s in shells}
    with open(f"{S4}/lumina_levelpop.csv") as f:
        for row in csv.DictReader(f):
            s=int(row["shell"])
            if s not in shells or int(row["Z"])!=Z or int(row["ion"])!=ion: continue
            out[s].append((int(row["level_num"]),float(row["E_eV"]),int(row["g"]),
                           float(row["n_k"]),float(row["b_k"]),int(row["has_sigma"])))
    return out

def planck(nu,T):
    x=H*nu/(K_B*T); x=np.clip(x,1e-30,700.0)
    return (2.0*H*nu**3/(C*C))/np.expm1(x)

CAPS=[np.inf,1e6,1e5,3e4,1e4,5e3,2e3,1e3,5e2,1e2,10.0]
COMBS=[(26,2,"FeIII"),(27,2,"CoIII"),(28,2,"NiIII")]
SHELLS=[0,8]

def gph_ratio(levels,Z,ion,s,field):
    """returns dict cap->ratio(G_nlte(cap)/G_LTE) and the LTE/uncapped absolute."""
    W,T_rad,T_e=PS[s]
    chi_ion=ioniz[(Z,ion)]*EV
    kT=K_B*T_e
    if field=="dilute": Jnu=W*planck(nu_c,T_rad)
    elif field=="hotTe": Jnu=planck(nu_c,T_e)
    else: Jnu=planck(nu_c,T_rad)
    # per-level LTE occupancy n^LTE = g e^{-E/kTe} (unnormalized; U cancels in ratio),
    # rate w_l, b_l ; only sigma-bearing levels contribute to G, ALL to the norm.
    recs=[]  # (nlLTE, b, w, has_sig)
    for (num,E_eV,g,n_k,b_k,hs) in levels:
        E=E_eV*EV; x=E/kT
        if x>=50.0:
            recs.append((0.0,b_k,0.0,False)); continue
        nlLTE=g*np.exp(-x)
        gl=gidx.get((Z,ion,num),-1)
        w=0.0; sig_ok = gl>=0 and has_sigma[gl]
        if sig_ok:
            chi_l=chi_ion-E
            if chi_l>0:
                nu_l=chi_l/H
                sig=sigma[gl]
                sel=(nu_c>=nu_l)&(sig>0)
                if sel.any():
                    w=np.sum(4.0*np.pi*sig[sel]*Jnu[sel]/(H*nu_c[sel])*dnu[sel])
        recs.append((nlLTE,b_k,w,sig_ok and w>0))
    nlLTE=np.array([r[0] for r in recs]); b=np.array([r[1] for r in recs])
    w=np.array([r[2] for r in recs]); sig=np.array([r[3] for r in recs])
    # LTE (cap=1 baseline: b->1)
    normL=np.sum(nlLTE); GL=np.sum((nlLTE/normL)*w)
    out={}
    for C in CAPS:
        bc=np.minimum(b,C)
        norm=np.sum(bc*nlLTE)
        G=np.sum((bc*nlLTE/norm)*w) if norm>0 else 0.0
        out[C]=(G/GL if GL>0 else np.nan)
    return out,GL

rows=[["comb","Z","ion","shell","field","cap","gph_ratio_vs_LTE"]]
dist=[["comb","shell","nlev_sig","bk_med_sig","bk_p90_sig","bk_max_sig","wwt_mean_b(uncapped=ratio)"]]
for (Z,ion,name) in COMBS:
    comb=load_comb(Z,ion,SHELLS)
    for s in SHELLS:
        for field in ("dilute","hotTe"):
            r,GL=gph_ratio(comb[s],Z,ion,s,field)
            for C in CAPS:
                rows.append([name,Z,ion,s,field,("inf" if C==np.inf else f"{C:g}"),f"{r[C]:.4g}"])
        # distribution over sigma-bearing levels (field-independent)
        W,T_rad,T_e=PS[s]; kT=K_B*T_e
        bsig=[]
        for (num,E_eV,g,n_k,b_k,hs) in comb[s]:
            gl=gidx.get((Z,ion,num),-1)
            if gl>=0 and has_sigma[gl] and E_eV*EV/kT<50.0: bsig.append(b_k)
        bsig=np.array(bsig)
        r_dil,_=gph_ratio(comb[s],Z,ion,s,"dilute")
        dist.append([name,s,len(bsig),
                     f"{np.median(bsig):.4g}",f"{np.percentile(bsig,90):.4g}",
                     f"{np.max(bsig):.4g}",f"{r_dil[np.inf]:.4g}"])
        print(f"{name} s{s}: nlev_sig={len(bsig):5d}  bk[med/p90/max]="
              f"{np.median(bsig):.3g}/{np.percentile(bsig,90):.3g}/{np.max(bsig):.3g}"
              f"  ratio(uncap,dilute)={r_dil[np.inf]:.4g}  ratio(cap5e3)={r_dil[5e3]:.4g}"
              f"  ratio(cap1e3)={r_dil[1e3]:.4g}")

with open(f"{OUT}/gph_cap_sizing.csv","w",newline="") as f: csv.writer(f).writerows(rows)
with open(f"{OUT}/gph_bk_distribution.csv","w",newline="") as f: csv.writer(f).writerows(dist)
print(f"\n[out] {OUT}/gph_cap_sizing.csv  gph_bk_distribution.csv")
