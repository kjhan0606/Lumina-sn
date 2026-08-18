#!/usr/bin/env python3
"""Case V supplement: (a) valley-entry exit id<0 & etype breakdown; (b) 1490-1650
pile mc/cs per run (verify funnel kill 39->1.90); (c) clean valley-vs-CMFGEN dex
table (mc & cs vs actual jnu4 CMFGEN + vs B(T_e_local))."""
import numpy as np, csv
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
EV=np.dtype([('pkt_id','<u4'),('line_id','<i4'),('nu','<f4'),('energy','<f4'),
             ('etype','u1'),('shell','u1'),('iter','u1'),('pad','u1')])
C_A=2.99792458e18;H=6.62607015e-27;KB=1.380649e-16;C=2.99792458e10
EDGES=[100,300,450,918,1290,1490,1650,2100,4500,20000,1e12]
def Bnu(l,T):
    nu=C_A/l;x=H*nu/(KB*T);return (2*H*nu**3/C**2)/np.expm1(x) if x<700 else 0.0
CMFGEN_VLY={0:8.1206e-4,1:6.0442e-4,2:4.4279e-4}
Te={'bsrc.n12':{0:14585,1:14903,2:14991},'gphall':{0:13120,1:13592,2:13912},
    'ltherm':{0:14080,1:14477,2:14882}}

def field(run,lo,hi):
    mc={};cs={}
    with open(f"{REPO}/logs/coevolve_consume_a10_kx_{run}/lumina_coevolve_field.csv") as f:
        for r in csv.DictReader(f):
            s=int(r['shell']);w=float(r['wavelength_A'])
            if lo<=w<hi: mc.setdefault(s,[]).append(float(r['mc_J']));cs.setdefault(s,[]).append(float(r['cs_J']))
    return {s:(np.mean(mc[s]),np.mean(cs[s])) for s in mc}

print("=== (b) 1490-1650 COMPLEX/pile mc/cs (funnel intensity) ===")
for run in ['bsrc.n12','gphall','ltherm']:
    f=field(run,1490,1650)
    print(f"  {run:9}: "+"  ".join(f"s{s} mc/cs={f[s][0]/f[s][1]:.2f}" for s in [0,1,2]))

print("\n=== (c) VALLEY 1650-2100: mc & cs vs ACTUAL CMFGEN and vs B(Te_local) ===")
print(f"  {'run':9}{'sh':>3}{'mc_J':>11}{'cs_J':>11}{'B(Te)':>11}{'mc/CMFGEN':>11}{'cs/CMFGEN':>11}{'mc/B(Te)':>10}{'cs/B(Te)':>10}")
lamc=np.sqrt(1650*2100)
for run in ['bsrc.n12','gphall','ltherm']:
    fv=field(run,1650,2100)
    for s in [0,1,2]:
        mcj,csj=fv[s];cj=CMFGEN_VLY[s];bte=Bnu(lamc,Te[run][s])
        print(f"  {run:9}{s:>3}{mcj:>11.3e}{csj:>11.3e}{bte:>11.3e}{mcj/cj:>11.3f}{csj/cj:>11.3f}{mcj/bte:>10.3f}{csj/bte:>10.3f}")
print(f"  [CMFGEN valley/B(13120): s0={CMFGEN_VLY[0]/Bnu(lamc,13120):.2f} s1={CMFGEN_VLY[1]/Bnu(lamc,13120):.2f} s2={CMFGEN_VLY[2]/Bnu(lamc,13120):.2f}]")

print("\n=== (a) valley-entry exit id<0 / etype breakdown (bsrc.n12) ===")
run='bsrc.n12'
mm=np.memmap(f"{REPO}/logs/coevolve_consume_a10_kx_{run}/lumina_events.bin",dtype=EV,mode='r',offset=32)
pid=np.array(mm['pkt_id']);et=np.array(mm['etype']);sh=np.array(mm['shell'])
nu=np.array(mm['nu']);lid=np.array(mm['line_id']);en=np.array(mm['energy']);del mm
lam=np.where(nu>0,C_A/nu,0.);band=np.digitize(lam,EDGES)-1
N=len(pid);order=np.argsort(pid,kind='stable')
et_s=et[order];sh_s=sh[order];band_s=band[order];lid_s=lid[order];en_s=en[order];pid_s=pid[order]
posn=np.arange(N);is_abs=(et_s==1);is_emit=np.isin(et_s,(2,4,5))
newg=np.empty(N,bool);newg[0]=True;newg[1:]=pid_s[1:]!=pid_s[:-1]
gstart=np.where(newg,posn,-1);gstart=np.maximum.accumulate(gstart)
abspos=np.where(is_abs,posn,-1);runabs=np.maximum.accumulate(abspos)
gov=np.where(runabs>=gstart,runabs,-1)
sel=is_emit&(gov>=0)&(gov!=posn);gpos=gov[sel]
ok=(band_s[gpos]==6)&np.isin(sh_s[gpos],[0,1,2])
si=np.where(sel)[0][ok]
ee=en_s[si];et_e=et_s[si];lid_e=lid_s[si]
tot=ee.sum()
for e in [2,4,5]:
    m=et_e==e;print(f"   exit etype {e}: n={m.sum():,} E={ee[m].sum():.3e} ({100*ee[m].sum()/tot:.1f}%)")
m2=et_e==2
print(f"   of etype-2 exits: id>=0 {100*ee[m2&(lid_e>=0)].sum()/tot:.1f}%  id<0 {100*ee[m2&(lid_e<0)].sum()/tot:.1f}% of total exit E")
