#!/usr/bin/env python3
"""Case V -- the unfilled valley (1650-2100 A) at s0-2 after Fork B.

For runs bsrc.n12 (Fork B 12-iter), gphall (B-run), ltherm (LTHERM):
 1. Valley field: band-avg mc_J, cs_J at s0-2; ratios mc/cs, cs/B(13120), mc/B,
    and vs actual CMFGEN jnu4 J (8.12e-4 s0 / 6.04e-4 s1 / 4.43e-4 s2).
 2. Valley ABSORBER ion shares (etype 1 line-abs, band 1650-2100, s0-2).
 3. Kernel row for valley ENTRIES: pair valley absorption -> next emission; exit
    band distribution + exit ion. "Where does valley energy go" in each run.
 4. NLTE-ion fraction of valley absorption.
Read-only; CSV copies only.
"""
import numpy as np, csv
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OUT=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/axis2_valley_forensics"
EV=np.dtype([('pkt_id','<u4'),('line_id','<i4'),('nu','<f4'),('energy','<f4'),
             ('etype','u1'),('shell','u1'),('iter','u1'),('pad','u1')])
LINE=np.dtype([('lam','<f4'),('Z','<u2'),('ion','<u2')])
C_A=2.99792458e18; H=6.62607015e-27;KB=1.380649e-16;C=2.99792458e10
EDGES=[100,300,450,918,1290,1490,1650,2100,4500,20000,1e12]
BLAB=['100-300','300-450','450-918','918-1290','1290-1490','1490-1650','1650-2100_VLY',
      '2100-4500','4500-20000','>20000']
NB=len(BLAB); VLY=6  # valley band index
ELEM={1:'H',2:'He',6:'C',7:'N',8:'O',10:'Ne',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',
      16:'S',17:'Cl',18:'Ar',19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',
      26:'Fe',27:'Co',28:'Ni'}
# line-table ion field is 0-BASED (0=neutral I); spectroscopic stage = ion+1.
IONROM={0:'I',1:'II',2:'III',3:'IV',4:'V',5:'VI',6:'VII'}
def ionname(z,i): return f"{ELEM.get(z,'Z%d'%z)} {IONROM.get(i,str(i))}"
def Bnu(lamA,Te):
    nu=C_A/lamA; x=H*nu/(KB*Te); return (2*H*nu**3/C**2)/np.expm1(x) if x<700 else 0.0
# ions treated in NLTE-SE by Lumina's plasma solver (Fe/Co/Ni/Si/S/Ca IGE+IME).
# For the valley attribution we flag the IGE NLTE ions vs the rest.
NLTE_IGE={26,27,28}      # Fe, Co, Ni  (statistical-equilibrium IGE)
NLTE_IME={14,16,20}      # Si, S, Ca   (IME, also SE-solved)
CMFGEN_VLY={0:8.1206e-4,1:6.0442e-4,2:4.4279e-4}   # jnu4 band-avg, correct v-grid
B13=Bnu(np.sqrt(1650*2100),13120.)                  # 1.707e-4

def load(run):
    p=f"{REPO}/logs/coevolve_consume_a10_kx_{run}"
    mm=np.memmap(f"{p}/lumina_events.bin",dtype=EV,mode='r',offset=32)
    d=dict(pid=np.array(mm['pid'] if False else mm['pkt_id']),lid=np.array(mm['line_id']),
           nu=np.array(mm['nu']),en=np.array(mm['energy']),
           et=np.array(mm['etype']),sh=np.array(mm['shell']))
    del mm
    with open(f"{p}/lumina_events_lines.bin","rb") as f:
        assert f.read(8)==b'LUMLIN01'; lr=np.frombuffer(f.read(),dtype=LINE)
    d['Lz']=lr['Z'].astype(np.int32); d['Lion']=lr['ion'].astype(np.int32)
    d['lam']=np.where(d['nu']>0,C_A/d['nu'],0.0)
    d['band']=np.digitize(d['lam'],EDGES)-1
    return d

def field_valley(run):
    mc={};cs={}
    with open(f"{REPO}/logs/coevolve_consume_a10_kx_{run}/lumina_coevolve_field.csv") as f:
        for r in csv.DictReader(f):
            s=int(r['shell']); w=float(r['wavelength_A'])
            if 1650<=w<2100:
                mc.setdefault(s,[]).append(float(r['mc_J'])); cs.setdefault(s,[]).append(float(r['cs_J']))
    return {s:(np.mean(mc[s]),np.mean(cs[s])) for s in mc}

def ion_shares(d,mask,lids,ens,n=10):
    m=mask&(lids>=0)
    zz=d['Lz'][lids[m]]; ii=d['Lion'][lids[m]]; ee=ens[m]
    key=zz*100+ii; ku,inv=np.unique(key,return_inverse=True)
    es=np.zeros(len(ku)); np.add.at(es,inv,ee); o=np.argsort(-es)
    T=float(ens[mask].sum())
    out=[]
    for j in o[:n]:
        k=int(ku[j]); out.append((k//100,k%100,es[j],100*es[j]/T))
    return out,T

def kernel_row(d,entry_band_idx,shells):
    """pair absorptions in entry_band_idx at `shells` -> next emission; return
    exit-band energy hist + exit-ion shares + exit lids/ens for reuse."""
    pid=d['pid']; et=d['et']; sh=d['sh']; band=d['band']; lid=d['lid']; en=d['en']
    N=len(pid); order=np.argsort(pid,kind='stable')
    et_s=et[order]; sh_s=sh[order]; band_s=band[order]; lid_s=lid[order]; en_s=en[order]; pid_s=pid[order]
    is_abs=(et_s==1); is_emit=np.isin(et_s,(2,4,5))
    posn=np.arange(N)
    newg=np.empty(N,bool); newg[0]=True; newg[1:]=pid_s[1:]!=pid_s[:-1]
    gstart=np.where(newg,posn,-1); gstart=np.maximum.accumulate(gstart)
    abspos=np.where(is_abs,posn,-1); runabs=np.maximum.accumulate(abspos)
    gov=np.where(runabs>=gstart,runabs,-1)
    # emissions whose governing absorption is in the entry band + target shells
    sel=is_emit&(gov>=0)&(gov!=posn)
    gpos=gov[sel]
    in_ok=(band_s[gpos]==entry_band_idx)&np.isin(sh_s[gpos],shells)
    sel_idx=np.where(sel)[0][in_ok]
    exit_band=band_s[sel_idx]; exit_lid=lid_s[sel_idx]; exit_en=en_s[sel_idx]
    hist=np.zeros(NB); np.add.at(hist,exit_band[exit_band>=0],exit_en[exit_band>=0])
    return hist,exit_lid,exit_en

print(f"B(13120K) at valley center 1861A = {B13:.3e}")
absrows=[];fldrows=[];kernrows=[]
for run in ['bsrc.n12','gphall','ltherm']:
    print("\n"+"#"*92); print(f"# RUN {run}"); print("#"*92)
    d=load(run)
    # (1) field
    fv=field_valley(run)
    print("\n VALLEY 1650-2100 field (band-avg):")
    print(f"  {'sh':>3}{'mc_J':>12}{'cs_J':>12}{'mc/cs':>8}{'cs/B13':>9}{'mc/B13':>9}{'CMFGEN_J':>11}{'cs/CMFGEN':>10}")
    for s in [0,1,2]:
        mcj,csj=fv[s]; cj=CMFGEN_VLY[s]
        print(f"  {s:>3}{mcj:>12.3e}{csj:>12.3e}{mcj/csj:>8.3f}{csj/B13:>9.2f}{mcj/B13:>9.3f}{cj:>11.3e}{csj/cj:>10.2f}")
        fldrows.append([run,s,mcj,csj,mcj/csj,csj/B13,mcj/B13,cj,csj/cj])
    # (2) valley absorbers
    absmask=(d['et']==1)&(d['band']==VLY)&np.isin(d['sh'],[0,1,2])
    shares,Tabs=ion_shares(d,absmask,d['lid'],d['en'],n=12)
    print(f"\n VALLEY ABSORBERS (line-abs 1650-2100 s0-2), total E={Tabs:.3e}:")
    nlte_ige=nlte_ime=other=0.0
    for z,i,E,pct in shares:
        print(f"   {ionname(z,i):10} {pct:5.1f}%  (E={E:.3e})")
        absrows.append([run,ionname(z,i),E,pct])
    # NLTE fraction over ALL absorbers (not just top-12)
    m=absmask&(d['lid']>=0); zz=d['Lz'][d['lid'][m]]; ee=d['en'][m]
    tot=float(d['en'][absmask].sum())
    fIGE=float(ee[np.isin(zz,list(NLTE_IGE))].sum())/tot
    fIME=float(ee[np.isin(zz,list(NLTE_IME))].sum())/tot
    print(f"   => NLTE-IGE(Fe/Co/Ni) share={100*fIGE:.1f}%  NLTE-IME(Si/S/Ca) share={100*fIME:.1f}%  "
          f"(sum SE-solved={100*(fIGE+fIME):.1f}%)")
    # (3) kernel row for valley entries
    hist,elid,een=kernel_row(d,VLY,[0,1,2])
    Ht=hist.sum()
    print(f"\n VALLEY-ENTRY kernel (abs in 1650-2100 s0-2 -> next emission), paired E={Ht:.3e}:")
    for b in range(NB):
        if hist[b]<=0: continue
        conv='UP(red->blue)' if b<VLY else ('in-band' if b==VLY else 'DOWN(blue->red)')
        print(f"   -> {BLAB[b]:14} {100*hist[b]/Ht:5.1f}%  [{conv}]")
        kernrows.append([run,BLAB[b],hist[b],100*hist[b]/Ht])
    # exit ion for valley entries
    esh,eT=ion_shares(d,np.ones(len(elid),bool),elid,een,n=8)
    print(f"   EXIT ion (re-emits valley-absorbed energy):")
    for z,i,E,pct in esh: print(f"      {ionname(z,i):10} {pct:5.1f}%")
    del d
with open(f"{OUT}/v_valley_field.csv","w",newline="") as f:
    csv.writer(f).writerows([['run','shell','mc_J','cs_J','mc_over_cs','cs_over_B13','mc_over_B13','cmfgen_J','cs_over_cmfgen']]+fldrows)
with open(f"{OUT}/v_valley_absorbers.csv","w",newline="") as f:
    csv.writer(f).writerows([['run','ion','E','pct']]+absrows)
with open(f"{OUT}/v_valley_kernel.csv","w",newline="") as f:
    csv.writer(f).writerows([['run','exit_band','E','pct']]+kernrows)
print(f"\n[out] v_valley_field.csv, v_valley_absorbers.csv, v_valley_kernel.csv")
