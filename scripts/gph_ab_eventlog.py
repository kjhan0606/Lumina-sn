#!/usr/bin/env python3
"""GPH A/B event-log comparator: does the transport self-consistently reflect
the ionization change? Primary verdict axis (event log, not CSV).

Angles:
 E1 bf-abs (etype3=photoion) per core shell: count/energy + wavelength histogram
 E2 line palette by (Z, ion): overall + core(shell<=SMAX). Co/Fe III vs IV shift.
 E3 escape (etype6) emergent SED (coarse bins)

Usage: gph_ab_eventlog.py <dirA> <dirB> [SMAX_core]
"""
import sys, numpy as np

EVENT_DTYPE = np.dtype([
    ("pkt_id","<u4"),("line_id","<i4"),("nu_comov","<f4"),("energy","<f4"),
    ("etype","u1"),("shell","u1"),("iter","u1"),("pad","u1")])
LINE_DTYPE = np.dtype([("lam","<f4"),("Z","<u2"),("ion","<u2")])
C=2.99792458e10
ETN={1:"line-abs",2:"line-emit",3:"bf-abs",4:"kpkt-ff",5:"kpkt-fb",6:"escape",7:"e-scat",8:"bf-reemit"}
ROMAN={0:"I",1:"II",2:"III",3:"IV",4:"V",5:"VI"}
ZN={6:"C",8:"O",12:"Mg",13:"Al",14:"Si",16:"S",20:"Ca",21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni"}

def load(d):
    with open(f"{d}/lumina_events.bin","rb") as f:
        h=f.read(32); assert h[:8]==b"LUMEVT01"
        ev=np.frombuffer(f.read(),dtype=EVENT_DTYPE)
    with open(f"{d}/lumina_events_lines.bin","rb") as f:
        assert f.read(8)==b"LUMLIN01"
        ln=np.frombuffer(f.read(),dtype=LINE_DTYPE)
    return ev,ln

SMAX=int(sys.argv[3]) if len(sys.argv)>3 else 3
dA,dB=sys.argv[1],sys.argv[2]
print(f"core = shells 0..{SMAX}")

def analyze(tag,d):
    ev,ln=load(d)
    lam=np.where(ev["nu_comov"]>0, C/ev["nu_comov"]*1e8, 0.0)
    print(f"\n{'='*60}\n{tag}: {len(ev)} events   ({d})")
    # etype dist
    print("  etype:", ", ".join(f"{ETN.get(int(e),e)}={int((ev['etype']==e).sum())}" for e in sorted(np.unique(ev['etype']))))
    out={}
    # E1 bf-abs per core shell
    bf=ev[(ev["etype"]==3)]
    bfc=bf[bf["shell"]<=SMAX]
    lam_bfc=np.where(bfc["nu_comov"]>0,C/bfc["nu_comov"]*1e8,0.0)
    out['bf_core_n']=len(bfc); out['bf_core_e']=float(bfc["energy"].sum())
    print(f"  E1 bf-abs core(s<={SMAX}): n={len(bfc)} E={bfc['energy'].sum():.3e}")
    if len(bfc):
        for lo,hi in [(0,228),(228,504),(504,912),(912,2000),(2000,1e5)]:
            m=(lam_bfc>=lo)&(lam_bfc<hi)
            print(f"      {lo:>5.0f}-{hi:<6.0f}A: n={int(m.sum()):>9} E={bfc['energy'][m].sum():.3e}")
    # E2 line palette by (Z,ion)
    for scope,mask in [("ALL",np.ones(len(ev),bool)),(f"core(s<={SMAX})",ev["shell"]<=SMAX)]:
        le=ev[(np.isin(ev["etype"],[1,2]))&mask&(ev["line_id"]>=0)]
        if not len(le):
            print(f"  E2 {scope}: no line events"); continue
        lid=le["line_id"]; en=le["energy"]
        Z=ln["Z"][lid]; ion=ln["ion"][lid]
        tot=en.sum()
        agg={}
        for z,i,e in zip(Z,ion,en):
            agg[(int(z),int(i))]=agg.get((int(z),int(i)),0.0)+float(e)
        top=sorted(agg.items(),key=lambda kv:-kv[1])[:8]
        print(f"  E2 {scope} line palette (frac of {tot:.3e}):")
        for (z,i),e in top:
            print(f"      {ZN.get(z,z):>3} {ROMAN.get(i,i):<4} {e/tot*100:5.1f}%")
        out[f'palette_{scope}']=agg
    # E3 escape SED
    esc=ev[ev["etype"]==6]
    lam_e=np.where(esc["nu_comov"]>0,C/esc["nu_comov"]*1e8,0.0)
    out['esc']= (lam_e,esc["energy"])
    return out

oA=analyze("A(ground)",dA)
oB=analyze("B(all-level)",dB)

# side-by-side palette shift for IGE
print(f"\n{'='*60}\nPALETTE SHIFT (core) — IGE ion-stage fraction A vs B:")
pA=oA.get(f'palette_core(s<={SMAX})',{}); pB=oB.get(f'palette_core(s<={SMAX})',{})
tA=sum(pA.values()) or 1; tB=sum(pB.values()) or 1
for z in [26,27,28]:
    print(f"  {ZN[z]}: ", end="")
    for i in [2,3,4]:
        fa=pA.get((z,i),0)/tA*100; fb=pB.get((z,i),0)/tB*100
        print(f"{ROMAN[i]} {fa:4.1f}->{fb:4.1f}%  ",end="")
    print()

# E3 escape SED coarse
print(f"\nESCAPE SED (UV<3500 / opt 3500-7000 / NIR>7000, energy frac):")
for tag,o in [("A",oA),("B",oB)]:
    lam,en=o['esc']; t=en.sum() or 1
    uv=en[(lam>0)&(lam<3500)].sum()/t*100
    op=en[(lam>=3500)&(lam<7000)].sum()/t*100
    nir=en[lam>=7000].sum()/t*100
    print(f"  {tag}: UV={uv:.1f}%  opt={op:.1f}%  NIR={nir:.1f}%")
