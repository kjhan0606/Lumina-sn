#!/usr/bin/env python3
"""MASTERMIND TEST — same-ion fraction of (line-abs -> next line-emit) pairs.

Discriminator:
  manifold-confined macro-atom (ARTIS per-ion internal cascade) => emitter ion
  == absorber ion (same-ion HIGH). global emission-selection (k-packet re-excite
  from a cross-ion thermal CDF) => emitter ion != absorber ion (same-ion LOW).

Reuses the packet-chaining kernel from crime_reconstruction/part3_redist_kernel.py
(validity note there: global monotonic atomicAdd event ordering => within a pkt_id,
file order == causal order; single iteration iter=11). Read-only. gphall = B-run
(no BSRC/LTHERM/EPS gates; etype-2 records the true emitted line_id).
"""
import numpy as np, csv, sys
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OUT=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/mastermind_test"
EV=np.dtype([('pkt_id','<u4'),('line_id','<i4'),('nu','<f4'),('energy','<f4'),
             ('etype','u1'),('shell','u1'),('iter','u1'),('pad','u1')])
LINE=np.dtype([('lam','<f4'),('Z','<u2'),('ion','<u2')])
C_A=2.99792458e18

ROMAN=['I','II','III','IV','V','VI','VII','VIII']
def ionlabel(Z,ion0):  # 0-based ion field -> spectroscopic
    return f"Z{Z}.{ROMAN[ion0] if 0<=ion0<len(ROMAN) else ion0}"

def run(runname):
    path=f"{REPO}/logs/coevolve_consume_a10_kx_{runname}"
    mm=np.memmap(f"{path}/lumina_events.bin",dtype=EV,mode='r',offset=32)
    N=len(mm)
    pid=np.array(mm['pkt_id']); et=np.array(mm['etype']); sh=np.array(mm['shell'])
    nu=np.array(mm['nu']); lid=np.array(mm['line_id']); del mm
    with open(f"{path}/lumina_events_lines.bin","rb") as f:
        assert f.read(8)==b'LUMLIN01'; lr=np.frombuffer(f.read(),dtype=LINE)
    Llam=lr['lam'].astype(np.float64); Lz=lr['Z'].astype(np.int32); Lion=lr['ion'].astype(np.int32)
    nlines=len(Lz)

    # ---- convention sanity: known lines by lambda ----
    def id_near(lam0,tol=0.5):
        c=np.where(np.abs(Llam-lam0)<tol)[0]
        return c
    print(f"\n#### RUN {runname}  (N={N:,} events, {nlines:,} lines)")
    print("  [convention check] lambda -> (Z, ion_field) :")
    for lam0,expect in [(1526.17,"Co IV=(27,3 0-based)"),(2599.4,"Fe II"),(4923.9,"Fe II"),
                        (5018.4,"Fe II"),(1259.5,"S II?")]:
        c=id_near(lam0)
        if len(c):
            z=Lz[c[0]]; io=Lion[c[0]]
            print(f"    {lam0:9.2f}A -> lid {c[0]:>7d}  Z={z} ion_field={io}  => {ionlabel(z,io)}   (expect {expect})")
        else:
            print(f"    {lam0:9.2f}A -> (no line within 0.5A)")

    # ---- pairing: each emission's governing (most-recent same-packet) line-abs ----
    order=np.argsort(pid,kind='stable')
    pid_s=pid[order]; et_s=et[order]; sh_s=sh[order]; lid_s=lid[order]
    posn=np.arange(N)
    newg=np.empty(N,bool); newg[0]=True; newg[1:]=pid_s[1:]!=pid_s[:-1]
    gstart=np.where(newg,posn,-1); gstart=np.maximum.accumulate(gstart)
    is_abs=(et_s==1)
    abspos=np.where(is_abs,posn,-1); runabs=np.maximum.accumulate(abspos)
    gov=np.where(runabs>=gstart,runabs,-1)

    # emission universe. line-emit = etype 2 (has emitted line_id>=0 in B-run).
    # kpkt continuum exits (etype 4 ff, 5 fb) carry line_id<0 = thermal sink.
    is_lineemit=(et_s==2)
    is_kpkt_ff=(et_s==4); is_kpkt_fb=(et_s==5)
    # count thermal-sink share among all "exits"
    tot_lineemit=int(is_lineemit.sum()); tot_ff=int(is_kpkt_ff.sum()); tot_fb=int(is_kpkt_fb.sum())
    print(f"  [exit census] line-emit(etype2)={tot_lineemit:,}  kpkt-ff(4)={tot_ff:,}  "
          f"kpkt-fb(5)={tot_fb:,}  ff+fb share of exits={100*(tot_ff+tot_fb)/max(tot_lineemit+tot_ff+tot_fb,1):.2f}%")

    # valid pairs: line-emit with a governing line-abs, both with resolvable ion
    sel=is_lineemit&(gov>=0)&(gov!=posn)
    gpos=gov[sel]
    abs_lid=lid_s[gpos]; abs_sh=sh_s[gpos]
    emit_lid=lid_s[sel]; emit_sh=sh_s[sel]
    good=(abs_lid>=0)&(abs_lid<nlines)&(emit_lid>=0)&(emit_lid<nlines)
    abs_lid=abs_lid[good]; emit_lid=emit_lid[good]
    abs_sh=abs_sh[good]; emit_sh=emit_sh[good]
    aZ=Lz[abs_lid]; aI=Lion[abs_lid]; eZ=Lz[emit_lid]; eI=Lion[emit_lid]
    same=(aZ==eZ)&(aI==eI)
    npair=len(same)
    print(f"  [pairs] line-emit paired w/ governing line-abs (both ion-resolved) = {npair:,}")

    def frac(mask):
        n=int(mask.sum())
        return (100.0*np.count_nonzero(same[mask])/n if n else float('nan'), n)

    print("  [SAME-ION FRACTION]  (emitter ion == absorber ion)")
    groups=[('overall',np.ones(npair,bool)),
            ('s0-2 (deep)',np.isin(abs_sh,[0,1,2])),
            ('s3-6 (mid)', np.isin(abs_sh,[3,4,5,6])),
            ('s7-9 (phot)',np.isin(abs_sh,[7,8,9]))]
    rows=[]
    for name,m in groups:
        f,n=frac(m); print(f"    {name:14s}: {f:6.2f}%   (n={n:,})")
        rows.append([runname,'shellgroup',name,f"{f:.3f}",n])

    # by donor (absorber) ion, top-10 by pair count, deep + phot
    print("  [by donor ion — deep s0-2]  donor -> same-ion% | top emitter when cross")
    for gname,shells in [('s0-2',[0,1,2]),('s7-9',[7,8,9])]:
        gm=np.isin(abs_sh,shells)
        keys=aZ[gm].astype(np.int64)*100+aI[gm]
        uk,cnt=np.unique(keys,return_counts=True)
        top=uk[np.argsort(-cnt)][:10]
        print(f"    --- {gname} ---")
        for k in top:
            Z=k//100; I=k%100
            dm=gm&(aZ==Z)&(aI==I)
            f,n=frac(dm)
            # dominant emitter for cross-ion of this donor
            cm=dm&(~same)
            if int(cm.sum()):
                ek=eZ[cm].astype(np.int64)*100+eI[cm]
                euk,ecnt=np.unique(ek,return_counts=True)
                eb=euk[np.argmax(ecnt)]; ebl=ionlabel(eb//100,eb%100); esh=100*ecnt.max()/cm.sum()
            else:
                ebl='-'; esh=0.0
            print(f"      {ionlabel(Z,I):10s} same={f:6.2f}%  n={n:>9,}  ->emit {ebl}({esh:.0f}% of cross)")
            rows.append([runname,f'donor_{gname}',ionlabel(Z,I),f"{f:.3f}",n])

    # ---- reconcile funnel "77% Co IV self-recycle": of emissions LANDING in
    #      Co IV 1490-1650 pile at deep shells, what fraction had a Co IV absorber?
    pile=(eZ==27)&(eI==3)&np.isin(emit_sh,[0,1,2])
    if int(pile.sum()):
        coiv_abs=pile&(aZ==27)&(aI==3)
        print(f"  [Co IV pile deep] emit=CoIV(1490-1650 region) n={int(pile.sum()):,}; "
              f"of these, absorber was CoIV: {100*int(coiv_abs.sum())/int(pile.sum()):.2f}%")
        # top absorbers feeding the CoIV pile
        pk=aZ[pile].astype(np.int64)*100+aI[pile]
        puk,pcnt=np.unique(pk,return_counts=True); pt=np.argsort(-pcnt)[:6]
        print("    top donors into CoIV deep pile:", ", ".join(
            f"{ionlabel(puk[i]//100,puk[i]%100)}={100*pcnt[i]/pile.sum():.0f}%" for i in pt))
        rows.append([runname,'coiv_pile_deep','coiv_absorber_frac',
                     f"{100*int(coiv_abs.sum())/int(pile.sum()):.3f}",int(pile.sum())])

    # ---- photosphere: of emissions that are S III (Z16 ion2) at s7-9, absorber ion? ----
    siii=(eZ==16)&(eI==2)&np.isin(emit_sh,[7,8,9])
    if int(siii.sum()):
        pk=aZ[siii].astype(np.int64)*100+aI[siii]
        puk,pcnt=np.unique(pk,return_counts=True); pt=np.argsort(-pcnt)[:6]
        print(f"  [S III phot emit] n={int(siii.sum()):,}; top donors:", ", ".join(
            f"{ionlabel(puk[i]//100,puk[i]%100)}={100*pcnt[i]/siii.sum():.0f}%" for i in pt))
        rows.append([runname,'siii_phot','n_siii_emit_s79','',int(siii.sum())])
    return rows

allrows=[]
for r in (sys.argv[1:] or ['gphall']):
    allrows+=run(r)
with open(f"{OUT}/same_ion_results.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(['run','kind','key','same_ion_pct','n']); w.writerows(allrows)
print(f"\n[out] {OUT}/same_ion_results.csv")
