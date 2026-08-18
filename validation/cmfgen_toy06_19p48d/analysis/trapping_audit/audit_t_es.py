#!/usr/bin/env python3
"""Audit T: electron-scattering floor + CMFGEN Rosseland/es tau interpolation.
CMFGEN: MEANOPAC cols  R(1e10cm) I Tau_Ross dTau RatRoss ChiRoss ChiRoss2 ChiFlux Chi_es
        Tau_Flux Tau_es RatFlux Rat_es Kappa_R V(km/s).  Tau accumulates surface->in,
        so Tau at velocity v = outward optical depth from v to the surface.
Lumina: n_e (plasma_state) * sigma_T integrated radially outward.
"""
import numpy as np, csv
BASE='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/'
MEAN='/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MEANOPAC'
RVTJ='/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ'
SIGMA_T=6.6524587e-25
T_EXP=19.48*86400.0
TARGET_V=[4264+728*i for i in range(11)]

# --- CMFGEN MEANOPAC ---
Rc=[];Ic=[];TauR=[];Taues=[];ChiR=[];Chies=[];Vc=[]
with open(MEAN) as f:
    for ln in f:
        t=ln.split()
        if len(t)<15: continue
        try: r=float(t[0]); i=int(t[1])
        except: continue
        Rc.append(r);Ic.append(i);TauR.append(float(t[2]));ChiR.append(float(t[5]))
        Chies.append(float(t[8]));Taues.append(float(t[10]));Vc.append(float(t[14]))
Rc=np.array(Rc);TauR=np.array(TauR);Taues=np.array(Taues);ChiR=np.array(ChiR);Chies=np.array(Chies);Vc=np.array(Vc)
o=np.argsort(Vc); Vs=Vc[o];TauRs=TauR[o];Tauess=Taues[o]
print(f'[CMFGEN MEANOPAC] ND={len(Vc)} Vinner={Vc.min():.0f} Vouter={Vc.max():.0f} km/s')
print(f'  total Tau(Ross) inner->surf = {TauR.max():.2f}   total Tau(es) = {Taues.max():.3f}')

# CMFGEN n_e from RVTJ
def parse_block(text,label,ND):
    lines=text.splitlines()
    for i,l in enumerate(lines):
        if l.strip()==label:
            vals=[];j=i+1
            while j<len(lines) and len(vals)<ND:
                tk=lines[j].split()
                try: vals+=[float(x) for x in tk]
                except: break
                j+=1
            return np.array(vals[:ND])
    raise KeyError(label)
rt=open(RVTJ).read(); ND=len(Vc)
Vr=parse_block(rt,'Velocity (km/s)',ND); ne_c=parse_block(rt,'Electron density',ND)
orc=np.argsort(Vr); Vrs=Vr[orc]; ne_cs=ne_c[orc]

def interp_v(vt,x,y): return np.interp(vt,x,y)

# --- Lumina geometry + n_e ---
geo={}
with open(BASE+'data/tardis_reference_toy06_19p48d/geometry.csv') as f:
    r=csv.DictReader(f)
    for row in r:
        s=int(row['shell_id']); geo[s]=(float(row['r_inner']),float(row['r_outer']),
                                        float(row['v_inner'])/1e5,float(row['v_outer'])/1e5)
ne_l={}
with open(BASE+'logs/coevolve_consume_a10_kx_gphall/lumina_plasma_state.csv') as f:
    r=csv.DictReader(f)
    for row in r: ne_l[int(row['shell_id'])]=float(row['n_e'])
nsh=len(geo)
# cumulative es tau outward from inner edge of each shell
dtau=np.array([ne_l[s]*SIGMA_T*(geo[s][1]-geo[s][0]) for s in range(nsh)])
tau_out=np.array([dtau[s:].sum() for s in range(nsh)])  # from r_inner[s] outward

print(f'\n[Lumina] n_shells={nsh}  total es tau (s0 inner->out) = {tau_out[0]:.4f}')
print(f'\n# TAU comparison: outward optical depth from shell to surface')
print(f"{'shell':>5}{'v':>6} | {'CMF_TauRoss':>11}{'CMF_Tau_es':>10} | {'Lum_tau_es':>10} | {'es_ratio':>8} | {'ne_CMF':>10}{'ne_Lum':>10}{'ne_rat':>7}")
rows=[]
for s,vt in enumerate(TARGET_V):
    cr=interp_v(vt,Vs,TauRs); ce=interp_v(vt,Vs,Tauess)
    le=tau_out[s]
    nec=interp_v(vt,Vrs,ne_cs); nel=ne_l[s]
    print(f"{s:>5}{vt:>6} | {cr:>11.3f}{ce:>10.3f} | {le:>10.4f} | {le/ce:>8.3f} | {nec:>10.3e}{nel:>10.3e}{nec/nel:>7.1f}")
    rows.append((s,vt,cr,ce,le,nec,nel))
with open('/tmp/claude-10396/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/ab44c098-ca8e-4ede-a602-b922281a5709/scratchpad/tau_es.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['shell','v_kms','CMF_TauRoss','CMF_Tau_es','Lum_tau_es','ne_CMF','ne_Lum'])
    for r in rows: w.writerow(r)
print('\n[wrote] tau_es.csv')
