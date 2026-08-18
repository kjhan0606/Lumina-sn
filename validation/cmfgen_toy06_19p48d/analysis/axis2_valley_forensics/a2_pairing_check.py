#!/usr/bin/env python3
"""Diagnostic: is Lumina's abs->emit an intra-ion macro-atom scatter (same ion,
emitter may up-shift freq) or an inter-ion redistribution? For FUV-exit(918-1290)
line-emits at s7-9 (bsrc.n12): report the immediately-preceding event's etype and,
if it's a line-abs, the same-ion fraction between the FUV emitter and that absorber.
Also: for the S II FUV emitters, what band did the SAME packet's immediately
preceding absorption sit in (the S II fluorescence pump)?"""
import numpy as np
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
EV=np.dtype([('pkt_id','<u4'),('line_id','<i4'),('nu','<f4'),('energy','<f4'),
             ('etype','u1'),('shell','u1'),('iter','u1'),('pad','u1')])
LINE=np.dtype([('lam','<f4'),('Z','<u2'),('ion','<u2')])
C_A=2.99792458e18
EDGES=[100,300,450,918,1290,1490,1650,2100,4500,20000,1e12]
BLAB=['100-300','300-450','450-918','918-1290','1290-1490','1490-1650','1650-2100','2100-4500','4500-20000','>20000']
run='bsrc.n12'
p=f"{REPO}/logs/coevolve_consume_a10_kx_{run}"
mm=np.memmap(f"{p}/lumina_events.bin",dtype=EV,mode='r',offset=32)
pid=np.array(mm['pkt_id']);et=np.array(mm['etype']);sh=np.array(mm['shell'])
nu=np.array(mm['nu']);lid=np.array(mm['line_id']);del mm
with open(f"{p}/lumina_events_lines.bin","rb") as f:
    assert f.read(8)==b'LUMLIN01';lr=np.frombuffer(f.read(),dtype=LINE)
Lz=lr['Z'].astype(np.int32);Lion=lr['ion'].astype(np.int32)
lam=np.where(nu>0,C_A/nu,0.);band=np.digitize(lam,EDGES)-1
N=len(pid);order=np.argsort(pid,kind='stable')
pid_s=pid[order];et_s=et[order];sh_s=sh[order];band_s=band[order];lid_s=lid[order]
posn=np.arange(N)
same_pkt=np.zeros(N,bool);same_pkt[1:]=pid_s[1:]==pid_s[:-1]  # prev record same packet
# FUV-exit line-emits at s7-9 (band 3, etype 2)
fuv=(et_s==2)&(band_s==3)&np.isin(sh_s,[7,8,9])&same_pkt
prev=np.where(fuv)[0]-1
prev_et=et_s[prev];prev_band=band_s[prev];prev_lid=lid_s[prev]
emit_lid=lid_s[fuv]
print(f"FUV-exit line-emits at s7-9 with a same-packet predecessor: {fuv.sum():,}")
print("immediately-preceding event etype histogram:")
for e in np.unique(prev_et):
    m=prev_et==e;print(f"   etype {int(e)}: {100*m.mean():.1f}%")
# for predecessors that are line-abs (etype1): same-ion fraction
pa=prev_et==1
ei=emit_lid[pa];ai=prev_lid[pa]
good=(ei>=0)&(ai>=0)
same_ion=(Lz[ei[good]]==Lz[ai[good]])&(Lion[ei[good]]==Lion[ai[good]])
print(f"\npredecessor is line-abs: {pa.mean()*100:.1f}% of FUV emits")
print(f"   same (Z,ion) emitter==absorber: {100*same_ion.mean():.1f}%")
print(f"   -> if HIGH: intra-ion macro-atom up-shift (same atom absorbs red, emits FUV)")
print(f"   -> if LOW: inter-ion redistribution (energy crossed species)")
# S III specifically (Z=16, ion field=2): when S III emits FUV, what did the packet just absorb?
sII=(Lz[emit_lid]==16)&(Lion[emit_lid]==2)&(emit_lid>=0)
paS=pa&sII
aiS=prev_lid[paS]
print(f"\nS III FUV emitters (ion field=2) with line-abs predecessor: {paS.sum():,}")
print("   absorber band of that predecessor (the S III pump entry):")
bb=prev_band[paS]
for b in range(len(BLAB)):
    m=bb==b
    if m.sum()==0: continue
    print(f"      {BLAB[b]:12} {100*m.mean():.1f}%")
gg=aiS>=0
print("   absorber ion (who feeds S II):")
key=Lz[aiS[gg]]*100+Lion[aiS[gg]]
ku,c=np.unique(key,return_counts=True);o=np.argsort(-c)
IONROM={0:'I',1:'II',2:'III',3:'IV',4:'V',5:'VI'};ELEM={14:'Si',16:'S',26:'Fe',27:'Co',28:'Ni'}  # 0-based ion field
for j in o[:6]:
    k=ku[j];print(f"      {ELEM.get(k//100,k//100)} {IONROM.get(k%100,k%100):4} {100*c[j]/gg.sum():.1f}%")
