#!/usr/bin/env python3
"""Band-resolved field comparison kpr6 vs CMFGEN(jnu4) at s4(v7176) & s6(v8632),
and the Fe III excited-level Gph contribution by threshold-wavelength band.
Tells apart OTS(EUV ground edge) vs BSRC_PHOT(optical line field) as the suppressor."""
import os, sys, math, numpy as np
REPO='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
os.environ['LUMINA_REF_DIR']=f'{REPO}/data/tardis_reference_toy06_19p48d'
os.environ['LUMINA_SIGMA_BIN']=f'{REPO}/data/tardis_reference_toy06_19p48d/cmfgen_sigma_bf.bin'
sys.path.insert(0,f'{REPO}/scripts'); sys.path.insert(0,f'{REPO}/validation/cmfgen_toy06_19p48d/analysis')
import db_photoion_calc as dbp, gamma_from_cmfgen_jnu as G
KPR6=f'{REPO}/logs/coevolve_consume_a10_kx_kpr6'; JNU4='/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4'
Jc,nuc,ND,_=G.read_edd(f'{JNU4}/EDDFACTOR'); rt=open(f'{JNU4}/RVTJ').read()
Vc=G.rvtj_block(rt,'Velocity (km/s)',ND); order=np.argsort(Vc)
def cmf_J(v):
    vi=Vc[order]; i=min(max(np.searchsorted(vi,v),1),len(vi)-1)
    w=(v-vi[i-1])/(vi[i]-vi[i-1]); return G.J_on_sigma_grid((1-w)*Jc[:,order[i-1]]+w*Jc[:,order[i]],nuc)
lam=dbp.C*1e8/dbp.nu_c   # Angstrom per db bin
BANDS=[('EUV<912',0,912),('FUV 912-2000',912,2000),('nearUV 2000-4000',2000,4000),('opt 4000-9000',4000,9000)]
def bandavg(J,lo,hi):
    m=(lam>=lo)&(lam<hi)&(J>0); return float(np.mean(J[m])) if m.any() else 0.0
for shell,v in ((4,7176.0),(6,8632.0)):
    JL=dbp.field(KPR6,shell); JC=cmf_J(v)
    print(f"\n=== s{shell} v={v:.0f}: band-mean J_nu (erg/s/cm2/Hz/sr)  LUM(kpr6) vs CMFGEN(jnu4) ===")
    for name,lo,hi in BANDS:
        jl=bandavg(JL,lo,hi); jc=bandavg(JC,lo,hi)
        print(f"  {name:18} L={jl:.3e}  C={jc:.3e}  L/C={jl/jc if jc>0 else float('nan'):.3e}")
    # Fe III excited-level Gph contribution binned by each level's THRESHOLD wavelength
    Te,ne=dbp.plasma(KPR6,shell); chi0=dbp.CHI[(26,2)]
    idx=np.where((dbp.levZ==26)&(dbp.levI==2))[0]
    x=dbp.levE[idx]/(dbp.KB_EV*Te); U=float(np.sum(np.where(x<50,dbp.levG[idx]*np.exp(-np.minimum(x,50)),0)))
    thr_bins={'edge<912':0.0,'912-2000':0.0,'2000-4000':0.0,'>4000':0.0}
    thrC=dict(thr_bins); tot_l=tot_c=0.0
    for gl in idx:
        RL,chi_l=dbp.R_of_level(gl,JL,chi0); RC,_=dbp.R_of_level(gl,JC,chi0)
        if chi_l<=0: continue
        lam_th=dbp.C*1e8*dbp.H/chi_l
        xl=dbp.levE[gl]/(dbp.KB_EV*Te)
        if xl>=50: continue
        pb=dbp.levG[gl]*math.exp(-xl)/U
        key='edge<912' if lam_th<912 else '912-2000' if lam_th<2000 else '2000-4000' if lam_th<4000 else '>4000'
        thr_bins[key]+=pb*RL; thrC[key]+=pb*RC; tot_l+=pb*RL; tot_c+=pb*RC
    print(f"  Fe III Gph_boltz by level THRESHOLD-wavelength (pb*R summed):")
    for k in thr_bins:
        print(f"     thr {k:10} LUM={thr_bins[k]:.3e} ({100*thr_bins[k]/tot_l if tot_l>0 else 0:4.1f}%)  "
              f"CMFGEN={thrC[k]:.3e} ({100*thrC[k]/tot_c if tot_c>0 else 0:4.1f}%)  L/C={thr_bins[k]/thrC[k] if thrC[k]>0 else float('nan'):.2e}")
    print(f"     TOTAL Gph_boltz LUM={tot_l:.3e}  CMFGEN={tot_c:.3e}  L/C={tot_l/tot_c:.3e}")
