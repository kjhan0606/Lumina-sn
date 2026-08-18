#!/usr/bin/env python3
"""Case A2 -- NAME the channel that creates the photospheric FUV(918-1290) excess.

Reuses the validated event-log format (scripts/read_events.py) and the part3
packet-chaining method (crime_reconstruction/part3_redist_kernel.py).

Outputs (energy-weighted; packet-energy units, RELATIVE only -- calibrated dex vs
CMFGEN comes from the field J below):
  1. Emission ledger into FUV(918-1290) and feeder(1290-2000) at s7-9 by
     PROCESS (line-emit / kpkt-ff / kpkt-fb / e-scatter) and by emitting (Z,ion).
     Top channels with % share of *creation* (line-emit+ff+fb; e-scatter listed
     separately since it redistributes, does not create).
  2. Redistribution kernel at s7-9: for emissions landing in FUV, the ENTRY band
     of the governing absorption (pkt-id chaining) -> UP-conversion test
     (entry redder than 1290 -> blue out), + entry/exit ion manifolds.
  3. Arithmetic removal: s8 FUV mc_J scaled by (1 - creation share) of top-1 and
     top-3 channels -> compare to CMFGEN 7.7e-7.

COVERAGE (restated per run): single iter=11, CAP128M hit. etype 8 (bf-reemit)
UNLOGGED -> bf recomb continuum invisible. etype 7 (e-scatter) IS logged in these
runs. Read-only; writes CSV copies only.
"""
import numpy as np, csv, os, sys

REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OUT=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/axis2_valley_forensics"
EV=np.dtype([('pkt_id','<u4'),('line_id','<i4'),('nu','<f4'),('energy','<f4'),
             ('etype','u1'),('shell','u1'),('iter','u1'),('pad','u1')])
LINE=np.dtype([('lam','<f4'),('Z','<u2'),('ion','<u2')])
C_A=2.99792458e18
ETN={1:"line-abs",2:"line-emit",3:"bf-abs",4:"kpkt-ff",5:"kpkt-fb",6:"escape",7:"e-scat",8:"bf-reemit"}
# kernel bands (match part3)
EDGES=[100,300,450,918,1290,1490,1650,2100,4500,20000,1e12]
BLAB=['100-300','300-450','450-918','918-1290','1290-1490','1490-1650','1650-2100_VLY',
      '2100-4500','4500-20000','>20000']
NB=len(BLAB)
# line-table ion field is 0-BASED (0=neutral I); spectroscopic stage = ion+1.
# Confirmed: prior validated taskB has "Co IV" at ion field=3; Ca lines carry ion 0,1,2.
IONROM={0:'I',1:'II',2:'III',3:'IV',4:'V',5:'VI',6:'VII'}
def ionname(z,i): return f"{ELEM.get(z,'Z%d'%z)} {IONROM.get(i,str(i))}"
ELEM={1:'H',2:'He',6:'C',7:'N',8:'O',10:'Ne',11:'Na',12:'Mg',13:'Al',14:'Si',
      15:'P',16:'S',17:'Cl',18:'Ar',19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',
      24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni'}

def load(run):
    path=f"{REPO}/logs/coevolve_consume_a10_kx_{run}"
    mm=np.memmap(f"{path}/lumina_events.bin",dtype=EV,mode='r',offset=32)
    d=dict(pid=np.array(mm['pkt_id']),lid=np.array(mm['line_id']),
           nu=np.array(mm['nu']),en=np.array(mm['energy']),
           et=np.array(mm['etype']),sh=np.array(mm['shell']))
    del mm
    with open(f"{path}/lumina_events_lines.bin","rb") as f:
        assert f.read(8)==b'LUMLIN01'; lr=np.frombuffer(f.read(),dtype=LINE)
    d['Lz']=lr['Z'].astype(np.int32); d['Lion']=lr['ion'].astype(np.int32)
    d['lam']=np.where(d['nu']>0,C_A/d['nu'],0.0)
    d['band']=np.digitize(d['lam'],EDGES)-1
    return d

def emission_ledger(d,shells,lo,hi,label,run,rows):
    """process + ion emission ledger into [lo,hi) at the given shells."""
    lam=d['lam']; en=d['en']; et=d['et']; sh=d['sh']; lid=d['lid']
    inband=(lam>=lo)&(lam<hi)&np.isin(sh,shells)
    # process totals
    proc={}
    for e in (2,4,5,7):
        m=inband&(et==e); proc[ETN[e]]=(float(en[m].sum()),int(m.sum()))
    create=proc['line-emit'][0]+proc['kpkt-ff'][0]+proc['kpkt-fb'][0]
    print(f"\n  [{run} {label} {lo:.0f}-{hi:.0f}A]  CREATION(line+ff+fb)={create:.4e}   "
          f"(e-scat redistribute={proc['e-scat'][0]:.4e})")
    for pn in ('line-emit','kpkt-ff','kpkt-fb','e-scat'):
        E,c=proc[pn]; sharec=100*E/create if (create>0 and pn!='e-scat') else float('nan')
        print(f"     {pn:10} E={E:.4e} n={c:>10,}  {'%.1f%% of creation'%sharec if pn!='e-scat' else '(redistrib)'}")
    # line-emit by ion
    m=inband&(et==2)&(lid>=0)
    zz=d['Lz'][lid[m]]; ii=d['Lion'][lid[m]]; ee=en[m]
    key=zz*100+ii
    ku,idx=np.unique(key,return_inverse=True)
    esum=np.zeros(len(ku)); np.add.at(esum,idx,ee)
    order=np.argsort(-esum)
    # combined channel ledger: each ion's line-emit + ff + fb as separate channels
    chans=[(ionname(k//100,k%100)+' line',esum[j]) for j,k in zip(range(len(ku)),ku)]
    chans.append(('kpkt-fb (recomb cont)',proc['kpkt-fb'][0]))
    chans.append(('kpkt-ff (free-free)',proc['kpkt-ff'][0]))
    # line-emit with line_id<0 (macro-atom deactivation, untabulated)
    munk=inband&(et==2)&(lid<0); Eunk=float(en[munk].sum())
    if Eunk>0: chans.append(('line-emit (untabulated id<0)',Eunk))
    chans=sorted(chans,key=lambda x:-x[1])
    tot=sum(c[1] for c in chans)
    print(f"     -- TOP CHANNELS (line-by-ion + ff + fb), % of {tot:.4e} total creation+untab:")
    for nm,E in chans[:10]:
        print(f"        {nm:32} E={E:.4e}  {100*E/tot:5.1f}%")
        rows.append([run,label,f"{lo:.0f}-{hi:.0f}",nm,E,100*E/tot])
    return chans,tot,create,proc

def kernel_fuv(d,shells,run):
    """pkt-id chaining: for emissions landing in FUV(918-1290,band 3) at `shells`,
    tally governing-absorption ENTRY band + entry/exit ion. UP-conversion test."""
    pid=d['pid']; et=d['et']; sh=d['sh']; band=d['band']; lid=d['lid']; en=d['en']
    N=len(pid)
    order=np.argsort(pid,kind='stable')
    et_s=et[order]; sh_s=sh[order]; band_s=band[order]; lid_s=lid[order]; en_s=en[order]
    pid_s=pid[order]
    is_abs=(et_s==1); is_emit=np.isin(et_s,(2,4,5))
    posn=np.arange(N)
    newg=np.empty(N,bool); newg[0]=True; newg[1:]=pid_s[1:]!=pid_s[:-1]
    gstart=np.where(newg,posn,-1); gstart=np.maximum.accumulate(gstart)
    abspos=np.where(is_abs,posn,-1); runabs=np.maximum.accumulate(abspos)
    valid=runabs>=gstart
    gov=np.where(valid,runabs,-1)
    # emissions that land in FUV band(3) at target shells with a valid governing abs
    fuv_exit=is_emit&(band_s==3)&np.isin(sh_s,shells)&(gov>=0)&(gov!=posn)
    gpos=gov[fuv_exit]
    entry_band=band_s[gpos]; entry_lid=lid_s[gpos]
    exit_lid=lid_s[fuv_exit]; exit_en=en_s[fuv_exit]
    # entry-band histogram (energy-weighted)
    print(f"\n  [{run}] FUV-exit(918-1290) redistribution at s{shells}: paired={fuv_exit.sum():,}")
    print(f"     ENTRY band of the absorption that fed each FUV emission (energy-weighted):")
    Etot=exit_en.sum()
    eb_e=np.zeros(NB); np.add.at(eb_e,entry_band[entry_band>=0],exit_en[entry_band>=0])
    # band index 3 == 918-1290 FUV. index<3 = shorter lambda = BLUER (down-conv);
    # index>3 = longer lambda = REDDER (UP-conversion, red in -> blue FUV out).
    for b in range(NB):
        if eb_e[b]<=0: continue
        tag=''
        if b<3: tag='  <-- DOWN-conversion (bluer->FUV)'
        elif b==3: tag='  (FUV->FUV, in-band scatter/cascade)'
        else: tag='  <-- UP-conversion (redder->FUV)'
        print(f"        {BLAB[b]:14} {100*eb_e[b]/Etot:5.1f}%{tag}")
    upmask=(entry_band>3)                    # entry redder than FUV -> up-conversion
    downmask=(entry_band>=0)&(entry_band<3)  # entry bluer than FUV -> down-conversion
    up_share=100*exit_en[upmask].sum()/Etot
    down_share=100*exit_en[downmask].sum()/Etot
    print(f"     => UP-conversion (entry REDDER, bands 1290-inf feeding FUV) = {up_share:.1f}%"
          f"   DOWN-conversion (entry bluer, 100-918) = {down_share:.1f}%   in-band = {100*eb_e[3]/Etot:.1f}%")
    # exit ion manifold (who emits the FUV photon)
    def ion_top(mask,lids,ens,title,n=8):
        m=mask&(lids>=0)
        zz=d['Lz'][lids[m]]; ii=d['Lion'][lids[m]]; ee=ens[m]
        key=zz*100+ii; ku,inv=np.unique(key,return_inverse=True)
        es=np.zeros(len(ku)); np.add.at(es,inv,ee); o=np.argsort(-es)
        T=ee.sum()+float(ens[mask&(lids<0)].sum())
        print(f"     {title} (top ions, % of {T:.3e}; id<0={100*float(ens[mask&(lids<0)].sum())/T:.1f}% untab):")
        for j in o[:n]:
            k=ku[j]; print(f"        {ionname(k//100,k%100):10} {100*es[j]/T:5.1f}%")
    ion_top(np.ones(len(exit_lid),bool),exit_lid,exit_en,"EXIT ion (emits the FUV photon)")
    ion_top(upmask,entry_lid,exit_en,"ENTRY ion of UP-converted (absorbs the REDDER photon that becomes FUV)")
    ion_top(downmask,entry_lid,exit_en,"ENTRY ion of DOWN-converted (absorbs the bluer photon)")
    return eb_e,Etot,up_share

def per_shell_channel_share(d,top_channel_ions,lo,hi):
    """for arithmetic removal: creation-energy share of a given set of channels at each shell.
    top_channel_ions: list of (Z,ion) whose line-emit form the channel, plus flags for ff/fb."""
    pass

# ---- run ----
led_rows=[]
runmap={'bsrc.n12':'Fork B 12-iter','gphall':'B-run(all-lvl Gph)'}
CMFGEN_S8_FUV=7.7e-7
store={}
for run in ['bsrc.n12','gphall']:
    print("\n"+"#"*96); print(f"# RUN {run}  ({runmap[run]})"); print("#"*96)
    d=load(run)
    eh={int(k):int(v) for k,v in zip(*np.unique(d['et'],return_counts=True))}
    print("etype hist:",{ETN.get(k,k):v for k,v in eh.items()})
    # (1) ledgers at s7-9
    chF,totF,createF,procF=emission_ledger(d,[7,8,9],918,1290,'s7-9 FUV',run,led_rows)
    chG,totG,createG,procG=emission_ledger(d,[7,8,9],1290,2000,'s7-9 FEEDER',run,led_rows)
    # (2) kernel
    eb,Etot,up=kernel_fuv(d,[7,8,9],run)
    # (3) arithmetic removal at s8 -- creation-energy channel shares AT s8 only
    print(f"\n  [{run}] ARITHMETIC REMOVAL at s8 (FUV 918-1290):")
    ch8,tot8,create8,proc8=emission_ledger(d,[8],918,1290,'s8 FUV',run,led_rows)
    # field mc_J at s8 FUV
    import csv as _csv
    mcs=[]
    with open(f"{REPO}/logs/coevolve_consume_a10_kx_{run}/lumina_coevolve_field.csv") as f:
        for r in _csv.DictReader(f):
            if int(r['shell'])==8:
                w=float(r['wavelength_A'])
                if 918<=w<1290: mcs.append(float(r['mc_J']))
    mcJ8=float(np.mean(mcs))
    # creation-only total at s8 (exclude untabulated? include, it's real emission). Use create8 for shares.
    top1=ch8[0]; s1=top1[1]/create8 if create8>0 else 0
    top3=ch8[:3]; s3=sum(c[1] for c in top3)/create8 if create8>0 else 0
    print(f"     s8 FUV mc_J(measured)={mcJ8:.3e}   CMFGEN s8 FUV={CMFGEN_S8_FUV:.3e}  "
          f"(excess {np.log10(mcJ8/CMFGEN_S8_FUV):+.2f} dex)")
    print(f"     top-1 channel = {top1[0]} : {100*s1:.1f}% of s8 creation")
    print(f"     top-3 channels: "+"; ".join(f"{c[0]}({100*c[1]/create8:.1f}%)" for c in top3))
    mc_no1=mcJ8*(1-s1); mc_no3=mcJ8*(1-s3)
    print(f"     s8 FUV WITHOUT top-1 = {mc_no1:.3e}  ({np.log10(mc_no1/CMFGEN_S8_FUV):+.2f} dex vs CMFGEN)")
    print(f"     s8 FUV WITHOUT top-3 = {mc_no3:.3e}  ({np.log10(mc_no3/CMFGEN_S8_FUV):+.2f} dex vs CMFGEN)")
    store[run]=dict(mcJ8=mcJ8,s1=s1,s3=s3,top1=top1[0],mc_no1=mc_no1,mc_no3=mc_no3,up=up)
    del d

with open(f"{OUT}/a2_emission_ledger.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(['run','group','band_A','channel','E','pct']); w.writerows(led_rows)
print(f"\n[out] {OUT}/a2_emission_ledger.csv")
