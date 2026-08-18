#!/usr/bin/env python3
"""PART 3 measurement 1 -- measured effective source function per band.
For B-run (gphall) and stage4, at s0,s1,s2 (deep) and s7,s8 (photosphere control):
  E_abs(band)  = sum packet energy, etype in {1 line-abs, 3 bf-abs}
  E_emit(band) = sum packet energy, etype in {2 line-emit, 4 kpkt-ff, 5 kpkt-fb}
  ratio = E_emit/E_abs = (eta/(chi*J))_band = (S/J)_band   [unit-free]
  mc_J_band, cs_J_band from lumina_coevolve_field.csv (band-averaged J_nu)
  S_eff_band = mc_J_band * ratio            (MC source-function estimate, J_nu units)
Compare to B_nu(T_e=13120) [gas], B_nu(18760) [CMFGEN deep color], cs_J (thermalized).
BLIND SPOTS (restated): CAP128M saturated, iter=11 only, etype 7/8 (e-scatter,
bf-reemit) UNLOGGED -> bf recomb continuum re-emission invisible; report LINE-ONLY
(1 vs 2) alongside ALL-LOGGED to bound the bf gap. Read-only.
"""
import numpy as np, csv, os
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OUT=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/crime_reconstruction"
EV=np.dtype([('pkt_id','<u4'),('line_id','<i4'),('nu','<f4'),('energy','<f4'),
             ('etype','u1'),('shell','u1'),('iter','u1'),('pad','u1')])
C_A=2.99792458e18; H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10
# driver bands (A)
EDGES=[100,300,450,918,1290,1490,1650,2100,4500,20000]
BLAB=['100-300','300-450','450-918','918-1290','1290-1490','1490-1650_COMPLEX',
      '1650-2100_VALLEY','2100-4500','4500+']
def Bnu(lamA,T):
    nu=C_A/lamA; x=H*nu/(KB*T)
    return (2*H*nu**3/C**2)/np.expm1(x) if x<700 else 0.0
def band_center(i):  # geometric-ish center for B evaluation
    return np.sqrt(EDGES[i]*EDGES[i+1])

def load_field(run):
    """return dict[shell]-> (wl[1000], csJ[1000], mcJ[1000])."""
    d={}
    with open(f"{REPO}/logs/coevolve_consume_a10_kx_{run}/lumina_coevolve_field.csv") as f:
        for row in csv.DictReader(f):
            s=int(row['shell']); d.setdefault(s,[]).append(
                (float(row['wavelength_A']),float(row['cs_J']),float(row['mc_J'])))
    out={}
    for s,L in d.items():
        a=np.array(L); out[s]=(a[:,0],a[:,1],a[:,2])
    return out

def field_band_mean(field,shells,i):
    """band-averaged mc_J and cs_J over the given shells and band i."""
    lo,hi=EDGES[i],EDGES[i+1]; mc=[];cs=[]
    for s in shells:
        wl,csJ,mcJ=field[s]; m=(wl>=lo)&(wl<hi)
        if m.any(): mc.append(mcJ[m].mean()); cs.append(csJ[m].mean())
    return (np.mean(mc) if mc else np.nan, np.mean(cs) if cs else np.nan)

def analyze(run):
    mm=np.memmap(f"{REPO}/logs/coevolve_consume_a10_kx_{run}/lumina_events.bin",dtype=EV,mode='r',offset=32)
    et=np.array(mm['etype']); sh=np.array(mm['shell']); nu=np.array(mm['nu']); en=np.array(mm['energy'])
    del mm
    lam=np.where(nu>0,C_A/nu,0.0)
    bi=np.digitize(lam,EDGES)-1
    field=load_field(run)
    rows=[]
    for gname,shells in [('s0-2',[0,1,2]),('s7-8',[7,8])]:
        gm=np.isin(sh,shells)
        for i in range(len(BLAB)):
            bm=gm&(bi==i)
            eabs_l=en[bm&(et==1)].sum(); eemit_l=en[bm&(et==2)].sum()
            eabs_a=en[bm&np.isin(et,(1,3))].sum(); eemit_a=en[bm&np.isin(et,(2,4,5))].sum()
            r_l =eemit_l/eabs_l if eabs_l>0 else np.nan
            r_a =eemit_a/eabs_a if eabs_a>0 else np.nan
            mcJ,csJ=field_band_mean(field,shells,i)
            lamc=band_center(i)
            b13=Bnu(lamc,13120.); b18=Bnu(lamc,18760.)
            Seff=mcJ*r_l if (mcJ==mcJ and r_l==r_l) else np.nan   # line-only S_eff
            rows.append([run,gname,BLAB[i],eabs_l,eemit_l,r_l,eabs_a,eemit_a,r_a,
                         mcJ,csJ,mcJ/csJ if csJ>0 else np.nan,
                         b13,b18,Seff,Seff/b13 if (Seff==Seff and b13>0) else np.nan,
                         Seff/b18 if (Seff==Seff and b18>0) else np.nan,
                         csJ/b13 if b13>0 else np.nan, mcJ/b13 if b13>0 else np.nan])
    return rows

allrows=[]
for run in ['gphall','stage4']:
    print(f"\n{'='*118}\nRUN {run}\n{'='*118}")
    print(f"{'grp':5}{'band':20}{'emit/abs(line)':>15}{'emit/abs(all)':>14}"
          f"{'mcJ/csJ':>10}{'S/J=r_l':>9}{'Seff/B13':>10}{'Seff/B18':>10}{'csJ/B13':>9}{'mcJ/B13':>9}")
    rows=analyze(run); allrows+=rows
    for x in rows:
        print(f"{x[1]:5}{x[2]:20}{x[5]:>15.3f}{x[8]:>14.3f}{x[11]:>10.3f}"
              f"{x[5]:>9.3f}{x[15]:>10.3f}{x[16]:>10.3f}{x[17]:>9.3f}{x[18]:>9.3f}")
with open(f"{OUT}/part3_seff_bands.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(['run','group','band','Eabs_line','Eemit_line','ratio_line','Eabs_all','Eemit_all',
                'ratio_all','mcJ','csJ','mcJ_over_csJ','B13','B18','Seff_line','Seff_over_B13',
                'Seff_over_B18','csJ_over_B13','mcJ_over_B13'])
    w.writerows(allrows)
print(f"\n[out] {OUT}/part3_seff_bands.csv")
