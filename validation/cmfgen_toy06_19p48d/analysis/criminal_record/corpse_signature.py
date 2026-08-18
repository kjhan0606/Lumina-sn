#!/usr/bin/env python3
"""CRIMINAL RECORD — corpse spot-check: is the kp_emiss attractor/same-ion
signature ARM-INVARIANT (architectural), or an artifact of the gphall config?

Runs the mastermind_test discriminator on physically CONTRASTING corpse arms:
  - gphground : arm A (ground-only Gph) = deep UNDER-ionized (IGE held in III)
  - evlog     : earliest event-log corpse (2026-07-13)
vs the canonical gphall (arm B, over-ionized) already measured in mastermind_test.

If the deep Co IV attractor + low same-ion fraction + ~0.02% ff/fb continuum
persist in an arm with a DIFFERENT ionization balance, the k-packet global
re-emission channel is architectural (predates gphall) — a serial-crime signature.

Read-only. Outputs to criminal_record/ (does NOT touch mastermind_test canon).
Reuses the validated packet-chaining/pairing kernel of same_ion_discriminator.py.
"""
import numpy as np, csv, sys
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OUT=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/criminal_record"
EV=np.dtype([('pkt_id','<u4'),('line_id','<i4'),('nu','<f4'),('energy','<f4'),
             ('etype','u1'),('shell','u1'),('iter','u1'),('pad','u1')])
LINE=np.dtype([('lam','<f4'),('Z','<u2'),('ion','<u2')])
ROMAN=['I','II','III','IV','V','VI','VII','VIII']
def ionlabel(Z,ion0): return f"Z{Z}.{ROMAN[ion0] if 0<=ion0<len(ROMAN) else ion0}"

def run(runname):
    path=f"{REPO}/logs/coevolve_consume_a10_kx_{runname}"
    mm=np.memmap(f"{path}/lumina_events.bin",dtype=EV,mode='r',offset=32)
    N=len(mm)
    pid=np.array(mm['pkt_id']); et=np.array(mm['etype']); sh=np.array(mm['shell'])
    eng=np.array(mm['energy']); lid=np.array(mm['line_id']); del mm
    with open(f"{path}/lumina_events_lines.bin","rb") as f:
        assert f.read(8)==b'LUMLIN01'; lr=np.frombuffer(f.read(),dtype=LINE)
    Lz=lr['Z'].astype(np.int32); Lion=lr['ion'].astype(np.int32); nlines=len(Lz)
    rows=[]; print(f"\n#### {runname}  N={N:,}")

    # exit census (S3 signature: ff/fb continuum share of exits)
    tl=int((et==2).sum()); tff=int((et==4).sum()); tfb=int((et==5).sum())
    contfrac=100*(tff+tfb)/max(tl+tff+tfb,1)
    print(f"  exits: line-emit={tl:,} ff={tff:,} fb={tfb:,}  ff+fb%={contfrac:.3f}")
    rows.append([runname,'exit_census','ff_fb_pct_of_exits',f"{contfrac:.4f}",tl+tff+tfb])

    # pairing (governing most-recent same-packet line-abs for each line-emit)
    order=np.argsort(pid,kind='stable')
    pid_s=pid[order]; et_s=et[order]; sh_s=sh[order]; lid_s=lid[order]
    posn=np.arange(N)
    newg=np.empty(N,bool); newg[0]=True; newg[1:]=pid_s[1:]!=pid_s[:-1]
    gstart=np.where(newg,posn,-1); gstart=np.maximum.accumulate(gstart)
    abspos=np.where(et_s==1,posn,-1); runabs=np.maximum.accumulate(abspos)
    gov=np.where(runabs>=gstart,runabs,-1)
    sel=(et_s==2)&(gov>=0)&(gov!=posn)
    gpos=gov[sel]
    abs_lid=lid_s[gpos]; emit_lid=lid_s[sel]; abs_sh=sh_s[gpos]; emit_sh=sh_s[sel]
    good=(abs_lid>=0)&(abs_lid<nlines)&(emit_lid>=0)&(emit_lid<nlines)
    abs_lid=abs_lid[good]; emit_lid=emit_lid[good]; abs_sh=abs_sh[good]; emit_sh=emit_sh[good]
    aZ=Lz[abs_lid]; aI=Lion[abs_lid]; eZ=Lz[emit_lid]; eI=Lion[emit_lid]
    same=(aZ==eZ)&(aI==eI); npair=len(same)
    def frac(m):
        n=int(m.sum()); return (100.0*np.count_nonzero(same[m])/n if n else float('nan'),n)
    for name,m in [('overall',np.ones(npair,bool)),('s0-2',np.isin(abs_sh,[0,1,2])),
                   ('s7-9',np.isin(abs_sh,[7,8,9]))]:
        f,n=frac(m); print(f"  same-ion {name}: {f:.2f}%  (n={n:,})")
        rows.append([runname,'same_ion',name,f"{f:.3f}",n])

    # deep attractor: Co IV (27,3) share of deep line-emit ENERGY (S1/S6) + who feeds it
    de=(et==2)&np.isin(sh,[0,1,2])&(lid>=0)&(lid<nlines)
    dE=eng[de]; dZ=Lz[lid[de]]; dI=Lion[lid[de]]
    if dE.sum()>0:
        coiv=(dZ==27)&(dI==3); coivE=100*dE[coiv].sum()/dE.sum()
        print(f"  DEEP attractor: Co IV = {coivE:.1f}% of deep line-emit energy")
        rows.append([runname,'deep_attractor','CoIV_emit_energy_pct',f"{coivE:.3f}",int(de.sum())])
    # deep pile absorber concentration (funnel: emissions landing in Co IV, donor mix)
    pile=(eZ==27)&(eI==3)&np.isin(emit_sh,[0,1,2])
    if int(pile.sum()):
        coabs=100*int((pile&(aZ==27)&(aI==3)).sum())/int(pile.sum())
        print(f"  DEEP Co IV pile: absorber was Co IV {coabs:.1f}% (self-recycle)")
        rows.append([runname,'deep_pile','CoIV_selfrecycle_pct',f"{coabs:.3f}",int(pile.sum())])

    # phot attractor: S III (16,2) share of phot line-emit ENERGY
    pe=(et==2)&np.isin(sh,[7,8,9])&(lid>=0)&(lid<nlines)
    pE=eng[pe]; pZ=Lz[lid[pe]]; pI=Lion[lid[pe]]
    if pE.sum()>0:
        siii=(pZ==16)&(pI==2); siiiE=100*pE[siii].sum()/pE.sum()
        print(f"  PHOT attractor: S III = {siiiE:.1f}% of phot line-emit energy")
        rows.append([runname,'phot_attractor','SIII_emit_energy_pct',f"{siiiE:.3f}",int(pe.sum())])
    return rows

allrows=[]
for r in (sys.argv[1:] or ['gphground','evlog']):
    try: allrows+=run(r)
    except Exception as e: print(f"  !! {r}: {e}")
with open(f"{OUT}/corpse_signature.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(['run','kind','key','value','n']); w.writerows(allrows)
print(f"\n[out] {OUT}/corpse_signature.csv")
