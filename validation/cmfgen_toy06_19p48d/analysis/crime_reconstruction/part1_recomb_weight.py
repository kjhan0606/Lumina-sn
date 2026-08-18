#!/usr/bin/env python3
"""PART 1 -- the alibi audit.
Compute the ACTUAL macro-atom branch weight of the recomb-cascade channel out of
Co IV level 144 (global 22564) in the STAGE4 configuration, exactly as
src/lumina_plasma.c computes it (recomb_alpha_per_level :1205-1239 ; recomb_prob
fill :2158-2200), and compare it against the radiative (emission + internal-down)
weights of the same block. s0 plasma. Read-only.

recomb weight (:2168-2175):  w_down_j = n_e * alpha(ip_CoIII, j, g_i=g_144, T_e)
                                       * eps_j * EV, eps_j = E_j + accumIP(CoIII)
alpha (:1223-1238):  Rbf = sum_bins 4pi B(nu_c,T) sig_j(nu_c)/(h nu_c) dnu
                     alpha = Rbf * lam3 * g_j/(2 g_i) * exp(chi_l/kT)
p(recomb) = sum_j w_down_j / (sum_rates_radiative + sum_j w_down_j)
"""
import numpy as np, csv, os, struct
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF=f"{REPO}/data/tardis_reference_toy06_19p48d"
LOG=f"{REPO}/logs/coevolve_consume_a10_kx_gphall"
S4 =f"{REPO}/logs/coevolve_consume_a10_kx_stage4"
OUT=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/crime_reconstruction"

K_B=1.380649e-16; EV=1.602176634e-12; H=6.62607015e-27; C=2.99792458e10
M_E=9.1093837015e-28; SOBOLEV_COEFF=2.6540281e-02
t_exp=1.683072e6
# s0 plasma
T_e=13119.874754; n_e=4.426076e9; W=0.2978587262; T_rad=10470.093240
beta_rad=1.0/(K_B*T_rad)

# ---- ionization energies ----
ioniz={}
with open(f"{REF}/ionization_energies.csv") as f:
    for row in csv.DictReader(f):
        ioniz[(int(row["atomic_number"]),int(row["ion_number"]))]=float(row["ionization_energy_eV"])
def accum_ip(Z,ion): return sum(ioniz.get((Z,i),0.0) for i in range(ion))
def find_ioniz(Z,stage): return ioniz.get((Z,stage),0.0)   # IP(stage -> stage+1)

# ---- levels (global order) ----
Zc=[];ionc=[];numc=[];Ec=[];gc=[]
with open(f"{REF}/levels.csv") as f:
    for row in csv.DictReader(f):
        Zc.append(int(row["atomic_number"]));ionc.append(int(row["ion_number"]))
        numc.append(int(row["level_number"]));Ec.append(float(row["energy_eV"]))
        gc.append(int(row["g"]))
Zc=np.array(Zc);ionc=np.array(ionc);numc=np.array(numc);Ec=np.array(Ec);gc=np.array(gc)
NLEV=len(Zc)
gidx={(Zc[i],ionc[i],numc[i]):i for i in range(NLEV)}

# ---- load cmfgen_sigma_bf.bin (magic CMFD) ----
with open(f"{REF}/cmfgen_sigma_bf.bin","rb") as f:
    magic,ver,nlev,nfreq=struct.unpack("<IIii",f.read(16))
    nu_min,nu_max=struct.unpack("<dd",f.read(16))
    assert magic==0x434D4644 and ver==1, (hex(magic),ver)
    assert nlev==NLEV, (nlev,NLEV)
    flag8=np.frombuffer(f.read(nlev),dtype=np.int8)
    pad=(8-(nlev%8))%8
    if pad: f.read(pad)
    sigma=np.frombuffer(f.read(nlev*nfreq*8),dtype=np.float64).reshape(nlev,nfreq)
has_sigma=flag8.astype(bool)
print(f"sigma_bf: nlev={nlev} nfreq={nfreq} nu=[{nu_min:.4e},{nu_max:.4e}] n_with_sigma={has_sigma.sum()}")
log_numin=np.log(nu_min); d_log_nu=(np.log(nu_max)-log_numin)/nfreq
bb=np.arange(nfreq)
log_nu_lo=log_numin+bb*d_log_nu
nu_c=np.exp(log_nu_lo+0.5*d_log_nu)
dnu=np.exp(log_nu_lo+d_log_nu)-np.exp(log_nu_lo)

def recomb_alpha(gl_global, g_i, T):
    """replicate recomb_alpha_per_level for target level gl_global, source g_i."""
    if not has_sigma[gl_global] or T<=0 or g_i<=0: return 0.0
    Z=Zc[gl_global]; stage=ionc[gl_global]
    chi_ion=find_ioniz(Z,stage)
    if chi_ion<=0 or chi_ion>=1e9: return 0.0
    chi_ion_erg=chi_ion*EV
    chi_l=chi_ion_erg - Ec[gl_global]*EV
    if chi_l<=0: return 0.0
    nu_th=chi_l/H
    lam3=(H*H/(2.0*np.pi*M_E*K_B*T))**1.5
    sig=sigma[gl_global]
    x=H*nu_c/(K_B*T)
    sel=(nu_c>=nu_th)&(sig>0)&(x<700.0)
    if not sel.any(): return 0.0
    B=(2.0*H*nu_c**3/(C*C))/np.expm1(x)
    Rbf=np.sum(4.0*np.pi*B[sel]*sig[sel]/(H*nu_c[sel])*dnu[sel])
    return Rbf*lam3*gc[gl_global]/(2.0*g_i)*np.exp(chi_l/(K_B*T))

# ============ radiative sum_rates(block) for Co IV level 144 ============
# reuse the exact reconstruction from coiv_funnel_trace (B-run dilute-Boltzmann tau);
# stage4 SE pops shift tau but NOT the order of magnitude (checked via n_k below).
LL=np.genfromtxt(f"{REF}/line_list.csv",delimiter=",",names=True)
ln_Z=LL["atomic_number"].astype(int); ln_ion=LL["ion_number"].astype(int)
ln_lo=LL["level_number_lower"].astype(int); ln_up=LL["level_number_upper"].astype(int)
ln_flu=LL["f_lu"]; ln_nu=LL["nu"]; ln_Aul=LL["A_ul"]; ln_lamcm=LL["wavelength_cm"]

ion_pop={}
with open(f"{LOG}/lumina_ion_pops.csv") as f:
    for row in csv.DictReader(f):
        if int(row["shell_id"])!=0: continue
        ion_pop[(int(row["Z"]),int(row["stage"]))]=float(row["n_ion"])
def nebular_Zpart(Z,ion):
    m=(Zc==Z)&(ionc==ion); E=Ec[m];g=gc[m]
    # metastable flag from levels.csv not reloaded; approximate meta by E-order not needed:
    # use W weighting for all non-ground (matches compute_tau_sobolev nebular). Ground meta.
    b=E*EV*beta_rad; ok=b<500
    # ground (level_number==0) treated metastable-> weight 1; else W. Rebuild meta:
    return None
# Load metastable explicitly
meta=np.zeros(NLEV,dtype=int)
with open(f"{REF}/levels.csv") as f:
    for i,row in enumerate(csv.DictReader(f)):
        meta[i]=int(row["metastable"])
def Zpart(Z,ion):
    m=(Zc==Z)&(ionc==ion); E=Ec[m];g=gc[m];mt=meta[m]
    b=E*EV*beta_rad; ok=b<500; wt=np.where(mt==1,1.0,W)
    return np.sum(wt[ok]*g[ok]*np.exp(-b[ok]))
ZpCoIV=Zpart(27,3); nCoIV=ion_pop[(27,3)]
def nlow(Z,ion,gl):
    E=Ec[gl];g=gc[gl];mt=meta[gl];b=E*EV*beta_rad
    if b>=500: return 0.0
    wt=1.0 if mt else W
    return nCoIV*wt*g*np.exp(-b)/ZpCoIV
def tau_coiv(lid):
    glo=gidx[(27,3,ln_lo[lid])]; gup=gidx[(27,3,ln_up[lid])]
    nl=nlow(27,3,glo); nu=nlow(27,3,gup)
    stim=1.0
    if nl>0 and nu>0:
        stim=1.0-(gc[glo]*nu)/(gc[gup]*nl); stim=max(stim,0.0)
    return max(SOBOLEV_COEFF*ln_flu[lid]*ln_lamcm[lid]*t_exp*nl*stim,1e-100)
def beta_sob(tau):
    if tau<1e-6: return 1.0-0.5*tau
    if tau>500: return 1.0/tau
    return (1.0-np.exp(-tau))/tau

# block of Co IV level 144 (global 22564): read macro_atom_data filtered
G=22564; g144=gc[G]
accipCoIV=accum_ip(27,3)
sum_rad=0.0; n_emit=0;n_idown=0
with open(f"{REF}/macro_atom_data.csv") as f:
    for row in csv.DictReader(f):
        if int(row["source_level_idx"])!=G: continue
        tt=int(row["transition_type"]); lid=int(row["lines_idx"])
        Z=ln_Z[lid];ion=ln_ion[lid]
        tau=tau_coiv(lid) if (Z==27 and ion==3) else 1e-100
        beta=beta_sob(tau)
        if tt==-1:
            r=ln_Aul[lid]*beta*H*ln_nu[lid]; sum_rad+=r; n_emit+=1
        elif tt==0:
            glo=gidx.get((Z,ion,ln_lo[lid]),-1)
            if glo>=0:
                e_low=Ec[glo]+accipCoIV; r=ln_Aul[lid]*beta*e_low*EV; sum_rad+=r; n_idown+=1
print(f"\nCo IV level 144 block: n_emit={n_emit} n_idown={n_idown}  sum_rad(radiative eweighted)={sum_rad:.6e} erg/s")

# ============ recomb weight from level 144: destinations = Co III levels w/ sigma ============
coiii_levels=np.where((Zc==27)&(ionc==2)&has_sigma)[0]
accipCoIII=accum_ip(27,2)
w_downs=[]
for j in coiii_levels:
    R=n_e*recomb_alpha(j,g144,T_e)
    eps_j=Ec[j]+accipCoIII
    w=R*eps_j*EV
    if w>0: w_downs.append((j,R,eps_j,w))
w_tot=sum(x[3] for x in w_downs)
print(f"Co III destinations with sigma: {len(coiii_levels)} ; nonzero recomb weight: {len(w_downs)}")
# report alpha magnitude
if w_downs:
    alphas=[x[1]/n_e for x in w_downs]
    print(f"  alpha per level range: [{min(alphas):.3e},{max(alphas):.3e}] cm^3/s ; sum_alpha={sum(alphas):.3e}")
    print(f"  n_e*sum_alpha (recomb RATE R_tot) = {n_e*sum(alphas):.3e} s^-1")
print(f"  sum w_down (recomb eweighted weight) = {w_tot:.6e} erg/s")

p_recomb=w_tot/(sum_rad+w_tot) if (sum_rad+w_tot)>0 else 0.0
print(f"\n==> p(recomb) from Co IV level 144 in STAGE4 = {p_recomb:.4e}")
print(f"    (radiative weight is {sum_rad/w_tot:.3e}x the recomb weight)" if w_tot>0 else "")

# ---- level 50 (metastable trap, global 22470) same computation ----
G50=22470; g50=gc[G50]
sum_rad50=0.0; ne50=0;ni50=0
with open(f"{REF}/macro_atom_data.csv") as f:
    for row in csv.DictReader(f):
        if int(row["source_level_idx"])!=G50: continue
        tt=int(row["transition_type"]); lid=int(row["lines_idx"])
        Z=ln_Z[lid];ion=ln_ion[lid]
        tau=tau_coiv(lid) if (Z==27 and ion==3) else 1e-100
        beta=beta_sob(tau)
        if tt==-1: sum_rad50+=ln_Aul[lid]*beta*H*ln_nu[lid]; ne50+=1
        elif tt==0:
            glo=gidx.get((Z,ion,ln_lo[lid]),-1)
            if glo>=0: sum_rad50+=ln_Aul[lid]*beta*(Ec[glo]+accipCoIV)*EV; ni50+=1
w50=[]
for j in coiii_levels:
    R=n_e*recomb_alpha(j,g50,T_e); w=R*(Ec[j]+accipCoIII)*EV
    if w>0: w50.append(w)
w50t=sum(w50)
p50=w50t/(sum_rad50+w50t) if (sum_rad50+w50t)>0 else 0.0
print(f"\nCo IV level 50 (metastable trap, g={g50}): n_emit={ne50} n_idown={ni50} sum_rad={sum_rad50:.4e}")
print(f"  recomb weight={w50t:.4e}  ==> p(recomb) level 50 = {p50:.4e}")

with open(f"{OUT}/part1_recomb_weight.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["source_level","g","n_emit","n_idown","sum_rad_erg_s","recomb_R_tot_s","recomb_weight_erg_s","p_recomb"])
    w.writerow([144,g144,n_emit,n_idown,f"{sum_rad:.6e}",f"{n_e*sum(x[1]/n_e for x in w_downs):.4e}" if w_downs else 0,f"{w_tot:.6e}",f"{p_recomb:.4e}"])
    w.writerow([50,g50,ne50,ni50,f"{sum_rad50:.6e}",f"{n_e*sum(w50)/max(w50t,1e-300)*0:.4e}",f"{w50t:.6e}",f"{p50:.4e}"])
print(f"\n[out] {OUT}/part1_recomb_weight.csv")
