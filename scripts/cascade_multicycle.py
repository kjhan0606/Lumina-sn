#!/usr/bin/env python3
"""RUNG 0c: multi-cycle reprocessing model.

Established: single-cycle Fe II branching re-emits UV ~98% (STRUCTURE, identical
to ARTIS atomic data). Fluorescence must therefore come from MULTI-CYCLE
reprocessing: UV is trapped (tau_UV~100), reabsorbed, re-emitted, and the
~1-2% per-cycle optical leak accumulates until UV converts to optical.

This model iterates the single-cycle exit distribution D (from cascade_walk_fe2)
with per-band ESCAPE probabilities from the actual radial optical depth (jnu
dump). A photon in band b each cycle either escapes (p_esc[b]) or is reabsorbed
and reprocessed (UV -> macro-atom single-cycle D; optical bands assumed to
escape/thermalize since their tau is low). Converges to an emergent band
distribution.

Key knob: p_esc(UV). High escape (UV leaks before reprocessing) -> high emergent
UV (Lumina). Low escape (UV trapped, many cycles) -> fluoresced (ARTIS/CMFGEN).
We compute emergent UV vs an effective UV-trapping depth to locate Lumina.

Usage: python3 scripts/cascade_multicycle.py [shell=3]
"""
import sys, csv
import numpy as np
H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10; EV=1.602176634e-12
SHELL=int(sys.argv[1]) if len(sys.argv)>1 else 3

# --- single-cycle Fe II exit distribution at realistic field (k=1) ---
# reuse the walk machinery inline (radiative, k=1)
ps={int(r['shell_id']):float(r['T_e']) for r in csv.DictReader(open('logs/stage1_toy06_epay27/lumina_plasma_state.csv'))}
T_e=ps[SHELL]
E={};G={}
for r in csv.DictReader(open('data/tardis_reference_toy06_19p48d/levels.csv')):
    if int(r['atomic_number'])==26 and int(r['ion_number'])==1:
        l=int(r['level_number']);E[l]=float(r['energy_eV']);G[l]=float(r['g'])
NL=max(E)+1; Eev=np.array([E.get(i,0.0) for i in range(NL)])
low=[];up=[];nu=[];Aul=[];Blu=[];Bul=[]
for r in csv.DictReader(open('data/tardis_reference_toy06_19p48d/line_list.csv')):
    if int(r['atomic_number'])==26 and int(r['ion_number'])==1:
        low.append(int(r['level_number_lower']));up.append(int(r['level_number_upper']))
        nu.append(float(r['nu']));Aul.append(float(r['A_ul']));Blu.append(float(r['B_lu']));Bul.append(float(r['B_ul']))
low=np.array(low);up=np.array(up);nu=np.array(nu);Aul=np.array(Aul);Blu=np.array(Blu);Bul=np.array(Bul)
lam_A=C/nu*1e8
BANDS=[('FUV',0,1700),('UVblnk',1700,3000),('CaIIKb',3000,3300),('UVtgt',3300,3700),
       ('fluor',3700,4400),('green',4400,5500),('red',5500,7000),('NIR1',7000,10000),('NIR2',10000,1e9)]
def band_of(lam):
    for i,(nm,lo,hi) in enumerate(BANDS):
        if lo<=lam<hi: return i
    return len(BANDS)-1
line_band=np.array([band_of(l) for l in lam_A])
NB=len(BANDS); UV={0,1,2,3}; OPT={4,5,6}
def Bnu(nu_,T):
    x=H*nu_/(KB*T); return np.where(x<500,2*H*nu_**3/C**2/np.expm1(np.clip(x,1e-30,500)),0.0)

def single_cycle_exit_from(entry_bandset, k=1.0):
    """single-cycle exit-band distribution for a photon absorbed in entry_bandset."""
    Jbar=k*Bnu(nu,T_e)
    R_down=Aul+Bul*Jbar; R_up=Blu*Jbar
    El=Eev[low]*EV; Eu=Eev[up]*EV; hnu=H*nu
    w_idn=R_down*El; w_emit=R_down*hnu; w_iup=R_up*Eu
    tot=np.zeros(NL); np.add.at(tot,up,w_idn+w_emit); np.add.at(tot,low,w_iup); tot[tot==0]=1
    p_idn=w_idn/tot[up]; p_emit=w_emit/tot[up]; p_iup=w_iup/tot[low]
    mask=np.array([b in entry_bandset for b in line_band])
    entry=np.zeros(NL); np.add.at(entry,up,(Blu*Jbar)*mask)
    if entry.sum()==0: return None
    entry/=entry.sum()
    exit_band=np.zeros(NB); s=entry.copy()
    for _ in range(2000):
        np.add.at(exit_band,line_band,s[up]*p_emit)
        s2=np.zeros(NL); np.add.at(s2,low,s[up]*p_idn); np.add.at(s2,up,s[low]*p_iup)
        if s2.sum()<1e-12: break
        s=s2
    return exit_band/exit_band.sum()

D=single_cycle_exit_from(UV, k=1.0)   # UV-entry single-cycle exit distribution
print(f"single-cycle UV-entry exit: UV={100*sum(D[i] for i in UV):.1f}%  OPT={100*sum(D[i] for i in OPT):.1f}%")

# --- per-band radial optical depth from jnu dump (Fe II era: use chi_line+chi_abs) ---
d=np.genfromtxt('logs/stage1_toy06_epay27/lumina_cmfgen_jnu.csv',delimiter=',',names=True)
geo={int(r['shell_id']):(float(r['r_inner']),float(r['r_outer'])) for r in csv.DictReader(open('data/tardis_reference_toy06_19p48d/geometry.csv'))}
C_A=2.99792458e18
tau_band=np.zeros(NB)
for s in sorted(set(int(x) for x in d['shell'])):
    if s<SHELL: continue   # outward from the emitting shell
    sub=d[d['shell']==s]; lam=C_A/sub['nu']; dr=geo[s][1]-geo[s][0]
    for b,(nm,lo,hi) in enumerate(BANDS):
        m=(lam>=lo)&(lam<hi)
        if m.sum(): tau_band[b]+=np.mean(sub['chi_line'][m]+sub['chi_abs'][m])*dr
print("radial tau per band (outward from shell {}):".format(SHELL))
print("   "+"  ".join(f"{nm}:{tau_band[b]:.1f}" for b,(nm,_,_) in enumerate(BANDS)))

# escape probability per band (two-stream single-flight ~ 1/(1+tau) is too leaky;
# use exp(-tau) single-flight escape, but photons scatter so effective escape
# per reprocessing cycle ~ 1-exp(-1/(1+tau)) ... keep transparent: p_esc=1/(1+tau))
def emergent(D, tau_band, mode='1/(1+tau)'):
    if mode=='1/(1+tau)': p_esc=1.0/(1.0+tau_band)
    else: p_esc=np.exp(-tau_band)
    # Markov: population per band being reprocessed; UV bands reprocess via D,
    # optical/NIR bands: on reabsorption assume they also reprocess via their own
    # (approx) but their tau is low so they mostly escape. Simplify: only UV
    # reprocesses (macro-atom); non-UV either escapes or is lost to thermal pool.
    pop=np.zeros(NB);
    for b in UV: pop[b]=D[b]      # seed with single-cycle UV output...
    # Actually seed with a pure UV photon entering:
    pop=np.zeros(NB); pop[1]=1.0  # start as UVblnk photon
    emergent=np.zeros(NB)
    for _ in range(100000):
        # escape
        esc=pop*p_esc; emergent+=esc; pop=pop-esc
        if pop.sum()<1e-10: break
        # reabsorbed: UV-band photons -> macro-atom single-cycle D; others -> thermalize
        reUV=sum(pop[b] for b in UV)
        newpop=np.zeros(NB)
        newpop+=reUV*D                      # UV reprocessed via macro-atom
        for b in OPT|{7,8}:                 # non-UV reabsorbed -> re-emit same band (scatter)
            newpop[b]+=pop[b]
        pop=newpop
    return emergent/emergent.sum()

for mode in ['1/(1+tau)','exp(-tau)']:
    em=emergent(D,tau_band,mode)
    uv=100*sum(em[i] for i in UV); opt=100*sum(em[i] for i in OPT)
    print(f"\nMULTI-CYCLE emergent (escape={mode}): UV={uv:.1f}%  OPT={opt:.1f}%")
    print("   "+"  ".join(f"{nm}:{100*em[b]:.1f}" for b,(nm,_,_) in enumerate(BANDS)))
print("\nCMFGEN emergent UV~24%, ARTIS~15%, Lumina epay27~43%.")
print("If multi-cycle w/ real tau gives UV<<43% => reprocessing IS the cure and")
print("Lumina's transport under-reprocesses (escape too easy: injection depth/iters).")
