#!/usr/bin/env python3
"""Path-A: systematic ionization-CEILING / saturation audit, all elements x all shells.

For each element the LUMINA reference includes ion stages up to some maximum
("ceiling"). If the physically-predicted dominant ion stage at a shell's
(T_e, n_e, W, T_rad) lies ABOVE that ceiling, the element is trapped one (or
more) stages too low -> over-populated lower ion -> spurious opacity.

Ionization energies and partition functions U(T) come from CMFGEN's OWN atomic
data (osc_data headers/levels); ceiling comes from the reference levels.csv.
Nebular (dilute) Saha is used: Phi_neb = W*(Te/Trad)^0.5 * Phi_LTE(Trad)
(zeta=1, recombination-to-ground bound; W from plasma_state).
"""
import csv, glob, numpy as np, sys

k_B=1.380649e-16; h=6.62607015e-27; m_e=9.1093837e-28
hc_k=1.4387768775   # cm*K
eV_cm=8065.544

CMFROOT="/gpfs/kjhan/cmfgen_21jun23/atomic"
DIR={6:'CARB',8:'OXY',12:'MG',13:'AL',14:'SIL',16:'SUL',20:'CA',21:'SCAN',
     22:'TIT',23:'VAN',24:'CHRO',25:'MAN',26:'FE',27:'COB',28:'NICK'}
EL={6:'C',8:'O',12:'Mg',13:'Al',14:'Si',16:'S',20:'Ca',21:'Sc',22:'Ti',23:'V',
    24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni'}
ROMAN=['I','II','III','IV','V','VI','VII']
REF="data/tardis_reference_cmfgen_superlev"

def parse_osc(path):
    ionE=None; nlev=None; levels=[]
    try:
        lines=open(path,errors="replace").readlines()
    except FileNotFoundError:
        return None,[]
    def f2f(s): return float(s.replace('D','E').replace('d','E'))
    for ln in lines:
        if "!Number of energy levels" in ln: nlev=int(ln.split()[0])
        if "!Ionization energy" in ln:       ionE=f2f(ln.split()[0])
        if "!Number of transitions" in ln:   break
    started=False;cnt=0
    for ln in lines:
        if "!Number of transitions" in ln: started=True; continue
        if not started or not ln.strip(): continue
        p=ln.split()
        try: g=f2f(p[1]); E=f2f(p[2])
        except (IndexError,ValueError): continue
        levels.append((g,E)); cnt+=1
        if nlev and cnt>=nlev: break
    return ionE, levels

def find_osc(Z, roman):
    base=f"{CMFROOT}/{DIR[Z]}/{roman}"
    hits=sorted(glob.glob(f"{base}/*/osc_data"))
    return hits[-1] if hits else None

def U(levels,T):
    if not levels: return None
    g=np.array([l[0] for l in levels]); E=np.array([l[1] for l in levels])
    return float(np.sum(g*np.exp(-E*hc_k/T)))

# load CMFGEN U + ionization edges per element/stage
print("Loading CMFGEN atomic data...", file=sys.stderr)
ATOM={}  # Z -> {stage_index(0=I): (ionE_cm, levels)}
for Z in DIR:
    ATOM[Z]={}
    for si,roman in enumerate(ROMAN):
        p=find_osc(Z,roman)
        if not p: continue
        ionE,lev=parse_osc(p)
        if ionE is not None and lev:
            ATOM[Z][si]=(ionE,lev)

# ceiling = max ion_number present in reference levels.csv (0=I)
ceil={}
with open(f"{REF}/levels.csv") as fh:
    for r in csv.DictReader(fh):
        z=int(r['atomic_number']); i=int(r['ion_number'])
        ceil[z]=max(ceil.get(z,-1),i)

# shell conditions
shells=[]
with open(f"{REF}/plasma_state.csv") as fh:
    for r in csv.DictReader(fh):
        shells.append([int(r['shell_id']),float(r['W']),float(r['T_rad'])])
ne={}
with open(f"{REF}/electron_densities.csv") as fh:
    for r in csv.DictReader(fh):
        ne[int(r['shell_id'])]=float(r['n_e'])

TE_RATIO=0.9
def saha_lte(Ul,Uh,chi_cm,T,n_e):
    fac=(2*np.pi*m_e*k_B*T/h**2)**1.5
    return (2.0*Uh/Ul/n_e)*fac*np.exp(-chi_cm*hc_k/T)

def dominant_stage(Z,T_rad,W,n_e):
    """nebular fractions across stages; return (dom_idx, frac_at_ceiling_and_above)."""
    stages=sorted(ATOM[Z])
    if not stages: return None,None,None
    T_e=TE_RATIO*T_rad
    ratios={stages[0]:1.0}; cur=1.0
    for si in stages:
        if si+1 not in ATOM[Z]:  # no next stage data -> stop ladder
            # but still can ionize if edge known; use this stage's ionE
            pass
        ionE_cm,lev=ATOM[Z][si]
        nxt=ATOM[Z].get(si+1)
        if nxt is None: break
        Ul=U(lev,T_e); Uh=U(nxt[1],T_e)
        Ulr=U(lev,T_rad); Uhr=U(nxt[1],T_rad)
        phi_lte=saha_lte(Ulr,Uhr,ionE_cm,T_rad,n_e)
        phi_neb=W*(T_e/T_rad)**0.5*phi_lte
        cur*=phi_neb; ratios[si+1]=cur
    tot=sum(ratios.values())
    fr={k:v/tot for k,v in ratios.items()}
    dom=max(fr,key=fr.get)
    c=ceil.get(Z,-1)
    above=sum(v for k,v in fr.items() if k> c)   # fraction predicted ABOVE ceiling
    return dom, fr, above

# audit at representative shells
rep=[0,5,10,15,20,25,29]
print(f"{'El':>3} {'ceil':>4} | predicted dominant stage @ shells "+" ".join(f"s{ s}" for s in rep))
print("-"*90)
trapped={}
for Z in sorted(EL):
    c=ceil.get(Z,-1)
    cells=[]
    for sid in rep:
        if sid>=len(shells): cells.append("  -"); continue
        _,W,Trad=shells[sid]; n_e=ne.get(sid,1e9)
        dom,fr,above=dominant_stage(Z,Trad,W,n_e)
        if dom is None: cells.append("  ?"); continue
        mark="*" if dom>c else " "
        cells.append(f"{ROMAN[dom]:>3}{mark}")
        if dom>c: trapped.setdefault(Z,[]).append((sid,ROMAN[dom],above))
    print(f"{EL[Z]:>3} {ROMAN[c] if c>=0 else '-':>4} | "+" ".join(cells))

print("\n=== TRAPPED (predicted dominant stage ABOVE reference ceiling) ===")
if not trapped:
    print("  none")
for Z,info in sorted(trapped.items()):
    shellrange=f"shells {info[0][0]}-{info[-1][0]}"
    worst=max(a for _,_,a in info)
    print(f"  {EL[Z]:>3} (ceiling {ROMAN[ceil[Z]]}): dominant {info[0][1]} at {shellrange}; "
          f"max fraction predicted above ceiling = {worst:.3f}")
