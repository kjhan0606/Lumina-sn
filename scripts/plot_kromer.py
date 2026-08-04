#!/usr/bin/env python3
"""Kromer plot (ion-by-ion emission/absorption) from lumina_kromer.csv.
Emission stacked by last-emitting ion ABOVE zero; absorption by absorbing ion
BELOW zero; gold + total overplotted. Usage: plot_kromer.py <kromer.csv> <out.png>"""
import sys, csv, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROMAN = ['I','II','III','IV','V','VI','VII']
ELEM = {6:'C',7:'N',8:'O',11:'Na',12:'Mg',13:'Al',14:'Si',16:'S',18:'Ar',20:'Ca',
        22:'Ti',24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni'}
def ionlabel(Z,ion):
    if Z<0: return 'cont/therm'
    return f"{ELEM.get(Z,'Z%d'%Z)} {ROMAN[ion] if 0<=ion<len(ROMAN) else ion}"

csvf = sys.argv[1] if len(sys.argv)>1 else 'lumina_kromer.csv'
outp = sys.argv[2] if len(sys.argv)>2 else 'figures/kromer.png'

el,eZ,ei,al,aZ,ai,en = [],[],[],[],[],[],[]
for r in csv.DictReader(open(csvf)):
    try:
        el.append(float(r['escape_lambda_A'])); eZ.append(int(r['emit_Z'])); ei.append(int(r['emit_ion']))
        al.append(float(r['in_lambda_A'])); aZ.append(int(r['in_Z'])); ai.append(int(r['in_ion']))
        en.append(float(r['energy']))
    except: pass
el=np.array(el);eZ=np.array(eZ);ei=np.array(ei);al=np.array(al);aZ=np.array(aZ);ai=np.array(ai);en=np.array(en)

lo,hi,dl = 3000.,12000.,50.
edges=np.arange(lo,hi+dl,dl); ctr=0.5*(edges[:-1]+edges[1:])
# rank ions by total emitted energy (within range, line emission only)
m=(el>=lo)&(el<hi)
keys={}
for Z,i,e in zip(eZ[m],ei[m],en[m]):
    if Z<0: continue
    keys[(Z,i)]=keys.get((Z,i),0.)+e
top=[k for k,_ in sorted(keys.items(),key=lambda kv:-kv[1])[:10]]
import matplotlib.cm as cm
cols={k:cm.tab20(idx) for idx,k in enumerate(top)}

fig,ax=plt.subplots(figsize=(14,7))
# emission (stacked, above 0)
base=np.zeros(len(ctr))
for k in top:
    sel=m&(eZ==k[0])&(ei==k[1])
    h,_=np.histogram(el[sel],bins=edges,weights=en[sel])
    ax.fill_between(ctr,base,base+h,step='mid',color=cols[k],label=ionlabel(*k),lw=0)
    base+=h
# other emission
sel=m&(eZ>=0)&~np.isin([f'{a}_{b}' for a,b in zip(eZ,ei)],[f'{a}_{b}' for a,b in top])
ho,_=np.histogram(el[m],bins=edges,weights=en[m]); ax.plot(ctr,ho,'k-',lw=1.5,label='total emission')

# absorption (stacked, below 0) by absorbing ion at in_lambda
ma=(al>=lo)&(al<hi)&(aZ>=0)
akeys={}
for Z,i,e in zip(aZ[ma],ai[ma],en[ma]): akeys[(Z,i)]=akeys.get((Z,i),0.)+e
atop=[k for k,_ in sorted(akeys.items(),key=lambda kv:-kv[1])[:10]]
acols={k:cm.tab20(idx) for idx,k in enumerate(atop)}
base=np.zeros(len(ctr))
for k in atop:
    sel=ma&(aZ==k[0])&(ai==k[1])
    h,_=np.histogram(al[sel],bins=edges,weights=en[sel])
    ax.fill_between(ctr,base,base-h,step='mid',color=acols[k],lw=0,alpha=0.85)
    base-=h
# gold overlay (scaled)
try:
    g=np.loadtxt('data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat')
    gw,gf=g[:,0],g[:,1]; gm=(gw>=lo)&(gw<hi)
    scale=np.trapezoid(ho,ctr)/np.trapezoid(gf[gm],gw[gm])
    ax.plot(gw[gm],gf[gm]*scale,'r--',lw=1.8,label='gold (scaled)')
except Exception as e: print('gold overlay skip',e)

ax.axhline(0,color='gray',lw=.6); ax.set_xlim(lo,hi)
ax.set_xlabel('Wavelength (A)'); ax.set_ylabel('emission (up) / absorption (down)')
ax.set_title('LUMINA DDC15 0.976d Kromer plot (ion-by-ion emission/absorption)')
ax.legend(ncol=3,fontsize=7,loc='upper right')
plt.tight_layout(); plt.savefig(outp,dpi=120); print('saved',outp)
# print top emitters by band
print('\n=== top EMISSION ions by band (energy frac) ===')
tote=en[m].sum()
for a,b,nm in [(5000,6500,'grn'),(6500,7500,'red/③'),(8000,9000,'NIR'),(9000,10000,'⑤9290')]:
    bm=m&(el>=a)&(el<b); bt=en[bm].sum()
    kk={}
    for Z,i,e in zip(eZ[bm],ei[bm],en[bm]): kk[(Z,i)]=kk.get((Z,i),0.)+e
    s=sorted(kk.items(),key=lambda kv:-kv[1])[:4]
    print(f'  {nm:9} ({a}-{b}): '+', '.join(f'{ionlabel(*k)} {100*v/bt:.0f}%' for k,v in s))
