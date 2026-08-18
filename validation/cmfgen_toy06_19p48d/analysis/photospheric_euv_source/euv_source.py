#!/usr/bin/env python3
"""WHERE does the photospheric EUV (<912A) field come from? -- kpr5 forensics.

Reuses the validated event-log format + packet-chaining kernel from
axis2_valley_forensics/a2_fuv_excess.py and mastermind_test/same_ion_discriminator.py.

Bands: EUV_ALL(<912), EUV 300-450 (Fe III ground edge 404A sits here), 450-912.
Shells (LUMINA 50-grid): phot=[6,7,8,9] (v=8632-10816), deep=[0,1,2], mid=[3,4,5].
CMFGEN benchmark map: gamma CSV 's4'..'s7' == LUMINA shells 6..9.

Coverage (single iter=11, CAP128M): etype 8 (bf-reemit) UNLOGGED -> bf recomb
re-emission invisible; etype 5 (kpkt-fb) IS the logged recomb-continuum channel.
etype 7 (e-scat) logged. Read-only; writes CSV copies only.
"""
import numpy as np, csv, os
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUN="kpr5"
PATH=f"{REPO}/logs/coevolve_consume_a10_kx_{RUN}"
OUT=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/photospheric_euv_source"
EV=np.dtype([('pkt_id','<u4'),('line_id','<i4'),('nu','<f4'),('energy','<f4'),
             ('etype','u1'),('shell','u1'),('iter','u1'),('pad','u1')])
LINE=np.dtype([('lam','<f4'),('Z','<u2'),('ion','<u2')])
C_A=2.99792458e18
ETN={1:"line-abs",2:"line-emit",3:"bf-abs",4:"kpkt-ff/BTe",5:"kpkt-fb",6:"escape",7:"e-scat"}
ROMAN=['I','II','III','IV','V','VI','VII','VIII']
ELEM={1:'H',2:'He',6:'C',7:'N',8:'O',10:'Ne',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',
      16:'S',17:'Cl',18:'Ar',19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',
      26:'Fe',27:'Co',28:'Ni'}
def ionname(z,i): return f"{ELEM.get(int(z),'Z%d'%z)} {ROMAN[i] if 0<=i<len(ROMAN) else i}"
PHOT=[6,7,8,9]; DEEP=[0,1,2]; MID=[3,4,5]

print("Loading kpr5 event log ...")
mm=np.memmap(f"{PATH}/lumina_events.bin",dtype=EV,mode='r',offset=32)
N=len(mm)
pid=np.array(mm['pkt_id']); lid=np.array(mm['line_id']); nu=np.array(mm['nu'])
en=np.array(mm['energy']); et=np.array(mm['etype']); sh=np.array(mm['shell']); del mm
lam=np.where(nu>0,C_A/nu,0.0)
with open(f"{PATH}/lumina_events_lines.bin","rb") as f:
    assert f.read(8)==b'LUMLIN01'; lr=np.frombuffer(f.read(),dtype=LINE)
Lz=lr['Z'].astype(np.int32); Lion=lr['ion'].astype(np.int32); nlines=len(Lz)
print(f"N={N:,}  nlines={nlines:,}")

EUV =(lam>0)&(lam<912.0)
B34 =(lam>=300)&(lam<450)     # 300-450 (Fe III ground edge 404A)
B49 =(lam>=450)&(lam<912)     # 450-912
BANDS=[("EUV<912",EUV),("300-450",B34),("450-912",B49)]
inphot=np.isin(sh,PHOT); indeep=np.isin(sh,DEEP)

rows=[]
# ---------------------------------------------------------------------------
# PART 1: EUV CREATION ledger at photosphere (local emission) by process + ion
# ---------------------------------------------------------------------------
print("\n"+"="*90+"\nPART 1  EUV CREATION (local emission) ledger  [energy-weighted]\n"+"="*90)
def ledger(shells_mask,label):
    print(f"\n--- shells {label} ---")
    for bname,bmask in BANDS:
        m=bmask&shells_mask
        proc={}
        for e in (2,5,4,7):
            me=m&(et==e); proc[e]=(float(en[me].sum()),int(me.sum()))
        create=proc[2][0]+proc[5][0]+proc[4][0]   # line-emit + kpkt-fb + kpkt-ff/BTe
        print(f"  [{bname}] CREATION(2+5+4)={create:.4e}  (e-scat redistrib={proc[7][0]:.3e})")
        for e in (2,5,4,7):
            E,c=proc[e]; frac=100*E/create if (create>0 and e!=7) else float('nan')
            tag=('%.1f%% create'%frac) if e!=7 else '(redistrib)'
            print(f"      etype{e:2d} {ETN[e]:12} E={E:.4e} n={c:>9,}  {tag}")
            rows.append([RUN,label,bname,f"proc_{ETN[e]}",E,c,frac])
        # line-emit by ion (etype 2)
        m2=m&(et==2)&(lid>=0)
        if int(m2.sum()):
            zz=Lz[lid[m2]]; ii=Lion[lid[m2]]; ee=en[m2]
            key=zz*100+ii; ku,inv=np.unique(key,return_inverse=True)
            es=np.zeros(len(ku)); np.add.at(es,inv,ee); o=np.argsort(-es)
            tote=ee.sum()+float(en[m&(et==2)&(lid<0)].sum())
            unt=float(en[m&(et==2)&(lid<0)].sum())
            print(f"      line-emit by ion (top6, %% of {tote:.3e}; untab id<0={100*unt/tote:.1f}%):")
            for j in o[:6]:
                k=ku[j]; print(f"          {ionname(k//100,k%100):9} {100*es[j]/tote:5.1f}%")
                rows.append([RUN,label,bname,f"ion_{ionname(k//100,k%100)}",es[j],0,100*es[j]/tote])
ledger(inphot,"phot_6-9"); ledger(indeep,"deep_0-2")

# ---------------------------------------------------------------------------
# PART 2: SAME-ION vs CROSS-ION for EUV line-emit at photosphere (H-A signature)
#         cross-ion => k-packet global thermal CDF (kp_emiss mastermind)
#         same-ion  => genuine intra-ion macro-atom cascade
# ---------------------------------------------------------------------------
print("\n"+"="*90+"\nPART 2  SAME-ION vs CROSS-ION  (EUV line-emit, governing line-abs)\n"+"="*90)
order=np.argsort(pid,kind='stable')
pid_s=pid[order]; et_s=et[order]; sh_s=sh[order]; lid_s=lid[order]
lam_s=lam[order]; en_s=en[order]
posn=np.arange(N)
newg=np.empty(N,bool); newg[0]=True; newg[1:]=pid_s[1:]!=pid_s[:-1]
gstart=np.where(newg,posn,-1); gstart=np.maximum.accumulate(gstart)
is_abs=(et_s==1)
abspos=np.where(is_abs,posn,-1); runabs=np.maximum.accumulate(abspos)
gov_abs=np.where(runabs>=gstart,runabs,-1)          # most-recent preceding line-abs
# most-recent preceding EMISSION (etype 2,4,5) -- for transport (Part 3)
is_emit=np.isin(et_s,(2,4,5))
emitpos=np.where(is_emit,posn,-1); runemit=np.maximum.accumulate(emitpos)
# gov_emit for an event = most recent emission STRICTLY before it in same pid group
gov_emit_incl=np.where(runemit>=gstart,runemit,-1)

def same_ion_euv(shells,label):
    sel=(et_s==2)&(lam_s>0)&(lam_s<912)&np.isin(sh_s,shells)&(gov_abs>=0)&(gov_abs!=posn)&(lid_s>=0)
    gpos=gov_abs[sel]; aL=lid_s[gpos]; eL=lid_s[sel]; eE=en_s[sel]
    good=(aL>=0)&(aL<nlines)
    aL=aL[good]; eL=eL[good]; eE=eE[good]
    aZ=Lz[aL]; aI=Lion[aL]; eZ=Lz[eL]; eI=Lion[eL]
    same=(aZ==eZ)&(aI==eI)
    n=len(same); Esame=float(eE[same].sum()); Etot=float(eE.sum())
    print(f"\n  [{label}] EUV line-emit paired w/ governing line-abs: n={n:,}")
    print(f"     SAME-ION: {100*np.count_nonzero(same)/n:.2f}% by count, {100*Esame/Etot:.2f}% by energy")
    print(f"     => CROSS-ION (k-packet global CDF share) = {100*(1-Esame/Etot):.2f}% by energy")
    # top emitter ions of CROSS-ion EUV (S III / IGE attractor test)
    cm=~same
    key=eZ[cm]*100+eI[cm]; ku,inv=np.unique(key,return_inverse=True)
    es=np.zeros(len(ku)); np.add.at(es,inv,eE[cm]); o=np.argsort(-es); Ec=eE[cm].sum()
    print(f"     CROSS-ion EUV emitter ions (top8, %% of cross E={Ec:.3e}):")
    for j in o[:8]:
        k=ku[j]; print(f"        {ionname(k//100,k%100):9} {100*es[j]/Ec:5.1f}%")
        rows.append([RUN,label,"crossion_emitter",ionname(k//100,k%100),es[j],0,100*es[j]/Ec])
    rows.append([RUN,label,"same_ion_pct","by_energy",Esame,n,100*Esame/Etot])
    return 100*Esame/Etot
sph=same_ion_euv(PHOT,"phot_6-9"); sdp=same_ion_euv(DEEP,"deep_0-2")

# ---------------------------------------------------------------------------
# PART 3: TRANSPORT -- H-A local vs H-B deep-leaked.
#   For EUV interaction events at the photosphere (line-abs+bf-abs+e-scat = field
#   samples), classify by the SHELL of the packet's most-recent preceding EMISSION.
#   local(6-9)=H-A ; deep(0-5)=H-B leaked-in ; outer(10+)=in-scattered.
# ---------------------------------------------------------------------------
print("\n"+"="*90+"\nPART 3  TRANSPORT: H-A(local) vs H-B(deep-leaked)  at photosphere\n"+"="*90)
sample=np.isin(et_s,(1,3,7))&(lam_s>0)&(lam_s<912)&np.isin(sh_s,PHOT)&(gov_emit_incl>=0)&(gov_emit_incl!=posn)
gpos=gov_emit_incl[sample]; src_sh=sh_s[gpos]; wE=en_s[sample]
Etot=float(wE.sum()); ntot=int(sample.sum())
loc=np.isin(src_sh,PHOT); deep=src_sh<6; outer=src_sh>9
print(f"  EUV field-sample interactions at phot (line-abs+bf-abs+e-scat) w/ known source: n={ntot:,}")
for nm,mk in [("H-A local (src 6-9)",loc),("H-B deep-leaked (src 0-5)",deep),("outer-in (src 10+)",outer)]:
    print(f"     {nm:28}: {100*float(wE[mk].sum())/Etot:5.1f}% by E   ({100*float(mk.sum())/ntot:.1f}% by count)")
    rows.append([RUN,"phot_6-9","transport",nm,float(wE[mk].sum()),int(mk.sum()),100*float(wE[mk].sum())/Etot])
# by band
for bname,(blo,bhi) in [("300-450",(300,450)),("450-912",(450,912))]:
    bm=(lam_s[sample]>=blo)&(lam_s[sample]<bhi)
    if int(bm.sum()):
        E2=float(wE[bm].sum())
        print(f"     -- band {bname}: H-A local={100*float(wE[bm&loc].sum())/E2:.1f}%  "
              f"H-B deep={100*float(wE[bm&deep].sum())/E2:.1f}%  (n={int(bm.sum()):,})")
        rows.append([RUN,"phot_6-9",f"transport_{bname}","H-A_local",float(wE[bm&loc].sum()),int((bm&loc).sum()),100*float(wE[bm&loc].sum())/E2])
        rows.append([RUN,"phot_6-9",f"transport_{bname}","H-B_deep",float(wE[bm&deep].sum()),int((bm&deep).sum()),100*float(wE[bm&deep].sum())/E2])

# ---------------------------------------------------------------------------
# PART 4: k-packet activity proxy at photosphere (p_kpacket not per-shell in stdout)
# ---------------------------------------------------------------------------
print("\n"+"="*90+"\nPART 4  k-packet activity at photosphere (event proxy; stdout has only s0/s49)\n"+"="*90)
for label,shells in [("deep_0-2",DEEP),("mid_3-5",MID),("phot_6-9",PHOT)]:
    m=np.isin(sh,shells)
    nle=int((m&(et==2)).sum()); nff=int((m&(et==4)).sum()); nfb=int((m&(et==5)).sum())
    tot=nle+nff+nfb
    print(f"  {label}: exits line-emit={nle:,} kpkt-ff/BTe={nff:,} kpkt-fb={nfb:,} "
          f"| continuum(ff+fb) share of exits = {100*(nff+nfb)/max(tot,1):.1f}%")
    rows.append([RUN,label,"kpkt_activity","cont_exit_share",nff+nfb,tot,100*(nff+nfb)/max(tot,1)])

with open(f"{OUT}/euv_source_ledger.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(['run','shells','band_or_kind','key','E','n','pct']); w.writerows(rows)
print(f"\n[out] {OUT}/euv_source_ledger.csv")
print(f"\nSUMMARY same-ion EUV line-emit: phot={sph:.1f}%  deep={sdp:.1f}%")
