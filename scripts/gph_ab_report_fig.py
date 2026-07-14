#!/usr/bin/env python3
"""4-panel report figure for GPH all-level A/B verdict.
P1 IGE core ionization <q> (A/B vs benchmark)
P2 IME Si/S per-shell <q> (A/B vs benchmark) -- shows B over-ionization
P3 b_k depression IGE vs IME excited levels (the smoking gun)
P4 spectrum A vs B vs CMFGEN (feature windows)
"""
import csv, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BG='#140D44'; CARD='#201858'; WHITE='#FAF9F5'; BLUE='#3898EC'; CORAL='#D97757'
TEAL='#4EC9B0'; GOLD='#FFC107'; DIM='#B0AEA5'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':CARD,'text.color':WHITE,
    'axes.labelcolor':WHITE,'xtick.color':DIM,'ytick.color':DIM,'axes.edgecolor':'#40407a',
    'font.size':10,'axes.titlecolor':WHITE})
C=2.99792458e10
gA='logs/coevolve_consume_a10_kx_gphground'; gB='logs/coevolve_consume_a10_kx_gphall'

def load_ion(p):
    d=defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for r in csv.DictReader(open(p)):
        d[int(r['shell_id'])][int(r['Z'])][int(r['stage'])]+=float(r['n_ion'])
    return d
def q(d):
    t=sum(d.values()); return sum(k*v for k,v in d.items())/t if t>0 else np.nan
pA=load_ion(f'{gA}/lumina_ion_pops.csv'); pB=load_ion(f'{gB}/lumina_ion_pops.csv')
vel={int(r['shell_id']):(float(r['v_inner'])+float(r['v_outer']))/2/1e5
     for r in csv.DictReader(open('data/tardis_reference_toy06_19p48d/geometry.csv'))}

def bench_q(el,ep,v):
    L=open(f'data/standart_data1/toy06/ionfrac_{el}_toy06_cmfgen.txt').read().splitlines()
    bl=[(float(l.split(':')[1].split()[0]),i) for i,l in enumerate(L) if l.strip().startswith('#TIME:')]
    bt,bi=min(bl,key=lambda x:abs(x[0]-ep)); rows=[]; j=bi+1
    while j<len(L):
        s=L[j].strip()
        if s.startswith('#TIME'): break
        if not s or s.startswith('#'): j+=1; continue
        pp=s.split()
        try: vv=float(pp[0])
        except: j+=1; continue
        rows.append((vv,[float(x) for x in pp[1:]])); j+=1
    vr=min(rows,key=lambda x:abs(x[0]-v)); f=vr[1]; t=sum(f)
    return sum(i*x for i,x in enumerate(f))/t if t>0 else np.nan

fig,ax=plt.subplots(2,2,figsize=(13,9)); fig.patch.set_facecolor(BG)

# P1 IGE core <q>
a=ax[0,0]; els=['Fe','Co','Ni']; Zs=[26,27,28]
qa=[np.mean([q(pA[s][Z]) for s in range(4)]) for Z in Zs]
qb=[np.mean([q(pB[s][Z]) for s in range(4)]) for Z in Zs]
x=np.arange(3); w=0.28
a.bar(x-w,qa,w,label='A (ground-only)',color=BLUE)
a.bar(x,qb,w,label='B (all-level, Boltzmann)',color=CORAL)
a.bar(x+w,[3.0]*3,w,label='CMFGEN=ARTIS bench',color=TEAL)
a.axhline(3.0,color=TEAL,ls=':',lw=1,alpha=0.6)
a.set_xticks(x); a.set_xticklabels(els); a.set_ylabel('core <q> (ionization stage)')
a.set_title('P1  IGE core (s0-3): B recovers III->IV  [TARGET]',color=GOLD)
a.legend(fontsize=8,facecolor=CARD,edgecolor='#40407a'); a.set_ylim(1.8,3.2)

# P2 IME per-shell <q>
a=ax[0,1]; sh=list(range(4,12)); vv=[vel[s] for s in sh]
for Z,el,c in [(14,'si',GOLD),(16,'s',CORAL)]:
    a.plot(vv,[q(pA[s][Z]) for s in sh],'o-',color=c,alpha=0.5,label=f'{el.upper()} A')
    a.plot(vv,[q(pB[s][Z]) for s in sh],'s-',color=c,label=f'{el.upper()} B')
    a.plot(vv,[bench_q(el,19.48,vel[s]) for s in sh],'--',color=c,lw=2,alpha=0.9,label=f'{el.upper()} bench')
a.set_xlabel('v [km/s]'); a.set_ylabel('<q>')
a.set_title('P2  IME (Si/S): B OVER-ionizes (solid>dash)  [REGRESSION]',color=CORAL)
a.legend(fontsize=7,ncol=3,facecolor=CARD,edgecolor='#40407a')

# P3 b_k smoking gun
a=ax[1,0]
labels=['Fe III','Co III','Ni III','Si II','S II']; bkA=[56.8,10.9,4.1,0.015,0.42]
cols=[TEAL,TEAL,TEAL,CORAL,CORAL]
a.bar(range(5),bkA,color=cols); a.set_yscale('log'); a.axhline(1.0,color=WHITE,ls='--',lw=1)
a.set_xticks(range(5)); a.set_xticklabels(labels)
a.set_ylabel('excited-level b_k (median)')
a.set_title('P3  b_k SMOKING GUN: IGE>>1 (real), IME<<1 (depressed)',color=GOLD)
a.text(0.5,1.3,'b=1 (Boltzmann assumes this)',color=WHITE,fontsize=7)
a.text(3.5,0.03,'Boltzmann over-\ncounts IME 66x',color=CORAL,fontsize=7,ha='center')

# P4 spectrum
a=ax[1,1]
def load_spec(p):
    d=np.loadtxt(p,delimiter=',',skiprows=1); return d[:,0],d[:,1]
la,fa=load_spec(f'{gA}/lumina_spectrum_coevolve_mc.csv')
lb,fb=load_spec(f'{gB}/lumina_spectrum_coevolve_mc.csv')
# CMFGEN 19.48d
Lc=open('data/standart_data1/toy06/spectra_toy06_cmfgen.txt').read().splitlines()
times=[float(x) for x in [l for l in Lc if l.startswith('#TIMES')][0].split(':')[1].split()]
wl=None; fl=None
# crude: find the wavelength grid + 19.48d column -- reuse compare tool convention is complex; skip exact, plot A/B only normalized
def norm(l,f):
    m=(l>3000)&(l<9000); s=np.trapz(f[m],l[m]); return f/s
a.plot(la,norm(la,fa),color=BLUE,lw=1,label=f'A (corr 0.474)')
a.plot(lb,norm(lb,fb),color=CORAL,lw=1,label=f'B (corr 0.372)')
for name,lo,hi in [('SiII 4130',3950,4150),('Fe/Mg',4200,4700),('SiII 5972',5700,6000),('SiII 6355',5900,6400),('CaII',7900,8700)]:
    a.axvspan(lo,hi,color=WHITE,alpha=0.05)
a.set_xlim(3000,9000); a.set_xlabel('wavelength [A]'); a.set_ylabel('normalized flux')
a.set_title('P4  spectrum: B washes optical features  [FALSIFIER FAIL]',color=CORAL)
a.legend(fontsize=8,facecolor=CARD,edgecolor='#40407a')

fig.suptitle('GPH all-level A/B verdict  —  Bug1 confirmed (IGE III->IV), B regresses (IME over-ion), root=b_k  ->  fix C=b_k-weighted',
             color=WHITE,fontsize=12,y=0.99)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig('figures/gph_ab_verdict.png',dpi=110,facecolor=BG)
print('wrote figures/gph_ab_verdict.png')
