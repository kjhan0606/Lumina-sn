#!/usr/bin/env python3
"""Full Co IV macro-atom cascade from level 144 (downward-only, J=0 -> conservative
lower bound on UV concentration; internal-up would only recycle UV and worsen the
pile). Confirms the emergent EMISSION-line wavelength distribution.
Replicates the block rates with EWEIGHT=1/NEUTRAL_E=1/IDOWN_BETA=1 at s0."""
import numpy as np, csv, os
from collections import defaultdict

REF="data/tardis_reference_toy06_19p48d"; LOG="logs/coevolve_consume_a10_kx_gphall"
OUT="validation/cmfgen_toy06_19p48d/analysis/coiv_funnel_trace"
SOBOLEV_COEFF=2.6540281e-02; K_B=1.380649e-16; EV=1.602176634e-12
H=6.62607015e-27; C=2.99792458e10; t_exp=1.683072e6
W=0.2978587262; T_rad=10470.093240; beta_rad=1.0/(K_B*T_rad)
CO_G0=22420  # global index of Co IV level 0

# ionization energies -> accum IP for Co IV
ioniz={}
for row in csv.DictReader(open(os.path.join(REF,"ionization_energies.csv"))):
    ioniz[(int(row["atomic_number"]),int(row["ion_number"]))]=float(row["ionization_energy_eV"])
accip_CoIV=sum(ioniz.get((27,i),0.0) for i in range(3))  # 58.449 eV

# Co IV levels (per-ion num -> E,g,meta), indexed 0..199
lE=np.zeros(200); lg=np.zeros(200); lmeta=np.zeros(200)
for row in csv.DictReader(open(os.path.join(REF,"levels.csv"))):
    if int(row["atomic_number"])==27 and int(row["ion_number"])==3:
        n=int(row["level_number"]); lE[n]=float(row["energy_eV"]); lg[n]=int(row["g"]); lmeta[n]=int(row["metastable"])

# line list (needed cols)
LL=np.genfromtxt(os.path.join(REF,"line_list.csv"),delimiter=",",names=True)
ln_Z=LL["atomic_number"].astype(int); ln_ion=LL["ion_number"].astype(int)
ln_lo=LL["level_number_lower"].astype(int); ln_up=LL["level_number_upper"].astype(int)
ln_flu=LL["f_lu"]; ln_nu=LL["nu"]; ln_Aul=LL["A_ul"]; ln_lamcm=LL["wavelength_cm"]

nion_CoIV=None
for row in csv.DictReader(open(os.path.join(LOG,"lumina_ion_pops.csv"))):
    if int(row["shell_id"])==0 and int(row["Z"])==27 and int(row["stage"])==3:
        nion_CoIV=float(row["n_ion"])
# nebular Zpart Co IV
b=lE*EV*beta_rad; wt=np.where(lmeta==1,1.0,W); ok=b<500
Zpart=np.sum(wt[ok]*lg[ok]*np.exp(-b[ok]))

def nlow_coiv(level):  # nebular n of Co IV level (per-ion index)
    bb=lE[level]*EV*beta_rad
    if bb>=500: return 0.0
    return nion_CoIV*(1.0 if lmeta[level] else W)*lg[level]*np.exp(-bb)/Zpart
def tau_line(lid):
    lo=ln_lo[lid]; up=ln_up[lid]
    nl=nlow_coiv(lo);
    bu=lE[up]*EV*beta_rad
    nu=nion_CoIV*(1.0 if lmeta[up] else W)*lg[up]*np.exp(-bu)/Zpart if bu<500 else 0.0
    stim=1.0
    if nl>0 and nu>0:
        stim=1.0-(lg[lo]*nu)/(lg[up]*nl); stim=max(stim,0.0)
    return max(SOBOLEV_COEFF*ln_flu[lid]*ln_lamcm[lid]*t_exp*nl*stim,1e-100)
def beta_sob(tau):
    if tau<1e-6: return 1.0-0.5*tau
    if tau>500: return 1.0/tau
    return (1.0-np.exp(-tau))/tau

# Build per-Co-IV-level downward blocks: list of (kind, prob, dest_localCoIV, lid, lam)
print("Extracting Co IV macro blocks (source in %d..%d)..."%(CO_G0,CO_G0+199))
raw=defaultdict(list)  # src_local -> [(ttype,dst_global,lid)]
for row in csv.DictReader(open(os.path.join(REF,"macro_atom_data.csv"))):
    sg=int(row["source_level_idx"])
    if sg<CO_G0 or sg>CO_G0+199: continue
    raw[sg-CO_G0].append((int(row["transition_type"]),int(row["destination_level_idx"]),int(row["lines_idx"])))

blocks={}
for src,trs in raw.items():
    chans=[]; tot=0.0
    for (tt,dg,lid) in trs:
        if not (ln_Z[lid]==27 and ln_ion[lid]==3):
            continue
        tau=tau_line(lid); beta=beta_sob(tau); lam=C/ln_nu[lid]*1e8
        if tt==-1:
            rate=ln_Aul[lid]*beta*H*ln_nu[lid]; kind="emit"; dloc=None
        elif tt==0:
            e_low=lE[ln_lo[lid]]+accip_CoIV
            rate=ln_Aul[lid]*beta*e_low*EV; kind="idown"; dloc=dg-CO_G0
        else:
            continue  # iup dropped (J=0)
        if rate>0:
            chans.append([kind,rate,dloc,lid,lam]); tot+=rate
    if tot>0:
        for c in chans: c[1]/=tot
    blocks[src]=(chans,tot)

# MC cascade from level 144
rng=np.random.default_rng(12345)
N=400000; MAXIT=500
emit_lam=[]; emit_lid=[]; ended_noemit=0
for _ in range(N):
    lev=144
    for it in range(MAXIT):
        chans,tot=blocks.get(lev,([],0.0))
        if tot<=0 or not chans:
            ended_noemit+=1; break
        r=rng.random(); acc=0.0; done=False
        for (kind,p,dloc,lid,lam) in chans:
            acc+=p
            if r<acc:
                if kind=="emit":
                    emit_lam.append(lam); emit_lid.append(lid); done=True
                else:
                    lev=dloc
                done=True; break
        if kind=="emit" and done: break
        if done and kind=="idown": continue
        # fallthrough safety
    else:
        ended_noemit+=1

emit_lam=np.array(emit_lam)
print("\ncascades=%d  emitted=%d  ended_noemit=%d  accip_CoIV=%.3f eV"%(N,len(emit_lam),ended_noemit,accip_CoIV))
def band(lam):
    if lam<912: return "EUV<912"
    if lam<1290: return "FUV912-1290"
    if lam<1490: return "1290-1490"
    if lam<=1650: return "PILE1490-1650"
    if lam<2000: return "1650-2000"
    if lam<4500: return "NUVblue2000-4500"
    return "opt/IR>4500"
hb=defaultdict(int)
for lam in emit_lam: hb[band(lam)]+=1
print("\nEMERGENT emission-line wavelength distribution (full downward cascade from lev144):")
for bnd in ["EUV<912","FUV912-1290","1290-1490","PILE1490-1650","1650-2000","NUVblue2000-4500","opt/IR>4500"]:
    print("   %-18s %8d  %6.2f%%"%(bnd,hb[bnd],100*hb[bnd]/max(len(emit_lam),1)))
# top emitted lines
cnt=defaultdict(int)
for l in emit_lid: cnt[l]+=1
top=sorted(cnt.items(),key=lambda z:-z[1])[:12]
print("\nTop-12 emergent emission lines:")
for lid,c in top:
    print("   lid=%7d lam=%8.2f A  Z=%d ion=%d  %.2f%%"%(lid, C/ln_nu[lid]*1e8, ln_Z[lid], ln_ion[lid], 100*c/len(emit_lam)))
print("\nMean emergent emission wavelength = %.1f A"%(np.mean(emit_lam)))
print("Fraction in 1490-1650 pile = %.1f%%"%(100*np.mean((emit_lam>=1490)&(emit_lam<=1650))))
