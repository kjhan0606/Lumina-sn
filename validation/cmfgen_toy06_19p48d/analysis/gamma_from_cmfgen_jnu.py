#!/usr/bin/env python3
"""gamma_from_cmfgen_jnu.py -- BONUS (TASK A secondary): field-driven photoionization rate
Gamma(Fe III) and Gamma(Co III) per forming shell from CMFGEN's converged J_nu (EDDFACTOR)
folded with the REAL bound-free cross-sections (Fe III tabulated; Co III real-sigma patch).

SCOPE / HONESTY: this is the FIELD-DRIVEN rate. Gamma_ion = SUM_k p_k * R_k(J), R_k = 4pi
INT sigma_k J/(h nu) dnu above threshold (db_photoion_calc.R_of_level). We weight with
  (a) GROUND state only (p=ground)  and  (b) Boltzmann p_k(Te) at CMFGEN's own Te(depth).
The TRUE NLTE Gamma needs CMFGEN Co III/Fe III level departure coeffs, which are not
readily extractable from a 4-iter non-converged snapshot -- so the shell-to-shell GRADIENT
here isolates the FIELD effect (the campaign question), not the absolute NLTE rate.
Lumina reference denominators (flat T_rad=10470K field): Fe III 8.068e-3, Co III 3.304e-4 s^-1.
"""
import os, sys
import numpy as np

REPO = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
os.environ['LUMINA_REF_DIR'] = f'{REPO}/data/tardis_reference_toy06_19p48d'
os.environ['LUMINA_SIGMA_BIN'] = f'{REPO}/data/tardis_reference_toy06_19p48d/cmfgen_sigma_bf.bin'
os.environ['LUMINA_SIGMA_COIII_PATCH'] = f'{REPO}/data/coiii_real_sigma_patch.npz'
sys.path.insert(0, f'{REPO}/scripts')
import db_photoion_calc as dbp

CLIGHT_A = 2997.92458
TARGET_V = [4264, 5720, 7176, 7904, 8632, 9360, 10088, 10816, 11544]
LABELS = ['s0','s1','s2','s3','s4','s5','s6','s7','s8']
LUM_DENOM = {(26,2): 8.068e-3, (27,2): 3.304e-4}

def read_info(info):
    L = open(info).read().splitlines(); v = L[2].split()
    return dict(ND=int(v[0]), RECL=int(v[1]), WORD=int(v[2]), little=(v[5]=='T'))
def read_edd(edd):
    info=read_info(edd+'_INFO'); ND=info['ND']; nwr=info['RECL']//info['WORD']
    dt='<f8' if info['little'] else '>f8'
    raw=np.fromfile(edd,dtype=dt); raw=raw[:(raw.size//nwr)*nwr].reshape(-1,nwr)
    data=raw[14:]; good=np.isfinite(data[:,:ND]).all(axis=1)&(data[:,ND]>0)
    J=data[good,:ND]; FL=data[good,ND]; nu=FL*1e15
    o=np.argsort(nu); return J[o], nu[o], ND, raw[4,0]
def rvtj_block(text,label,ND):
    lines=text.splitlines()
    for i,ln in enumerate(lines):
        if ln.strip()==label:
            vals=[]; j=i+1
            while j<len(lines) and len(vals)<ND:
                try: vals+=[float(t) for t in lines[j].split()]
                except ValueError: break
                j+=1
            return np.array(vals[:ND])
    raise KeyError(label)

def J_on_sigma_grid(Jdepth, nu_cmf):
    """log-log interpolate CMFGEN J(nu) at one depth onto db.nu_c; 0 outside coverage."""
    ln=np.log(nu_cmf); lj=np.log(np.where(Jdepth>0,Jdepth,1e-300))
    out=np.exp(np.interp(np.log(dbp.nu_c), ln, lj, left=-700, right=-700))
    out[(dbp.nu_c<nu_cmf.min())|(dbp.nu_c>nu_cmf.max())]=0.0
    return out

def gamma_ion(Jgrid, Te, Z, ion):
    chi0=dbp.CHI[(Z,ion)]; idx=np.where((dbp.levZ==Z)&(dbp.levI==ion))[0]
    x=dbp.levE[idx]/(dbp.KB_EV*Te)
    U=float(np.sum(np.where(x<50, dbp.levG[idx]*np.exp(-np.minimum(x,50)), 0.0)))
    G_gnd=G_b=0.0; nlev_sig=0
    for gl in idx:
        R,chi_l=dbp.R_of_level(gl, Jgrid, chi0)
        if R<=0: continue
        nlev_sig+=1
        xl=dbp.levE[gl]/(dbp.KB_EV*Te)
        if xl<50:
            pb=dbp.levG[gl]*np.exp(-xl)/U
            G_b+=pb*R
        if dbp.levN[gl]==0: G_gnd+=R
    return G_gnd, G_b, nlev_sig, dbp.ion_is_kramers(Z,ion)

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--edd',required=True); ap.add_argument('--rvtj',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    J,nu,ND,finish=read_edd(a.edd)
    print(f"[edd] {a.edd} ND={ND} nfreq={J.shape[0]} FINISH_REC={finish}")
    rt=open(a.rvtj).read()
    V=rvtj_block(rt,'Velocity (km/s)',ND); T=rvtj_block(rt,'Temperature (10^4K)',ND)*1e4
    rows=[]
    print(f"\n{'shell':>5} {'v_targ':>7} {'v_cmf':>7} {'depth':>5} {'Te':>7}  "
          f"{'FeIII_Ggnd':>11} {'FeIII_Gb':>11}  {'CoIII_Ggnd':>11} {'CoIII_Gb':>11}")
    store={}
    for lb,vt in zip(LABELS,TARGET_V):
        d=int(np.argmin(np.abs(V-vt)))            # CMFGEN depth nearest target v
        Jg=J_on_sigma_grid(J[:,d], nu)
        fe_g,fe_b,fen,fek=gamma_ion(Jg,T[d],26,2)
        co_g,co_b,con,cok=gamma_ion(Jg,T[d],27,2)
        store[lb]=(vt,V[d],d+1,T[d],fe_g,fe_b,co_g,co_b)
        print(f"{lb:>5} {vt:>7} {V[d]:>7.0f} {d+1:>5} {T[d]:>7.0f}  "
              f"{fe_g:>11.3e} {fe_b:>11.3e}  {co_g:>11.3e} {co_b:>11.3e}")
        rows.append((lb,vt,V[d],d+1,T[d],fe_g,fe_b,co_g,co_b))
    with open(a.out,'w') as f:
        f.write("shell,v_target_km_s,v_cmfgen_km_s,depth,Te_K,FeIII_Ggnd,FeIII_Gboltz,CoIII_Ggnd,CoIII_Gboltz\n")
        for r in rows: f.write(",".join(str(x) for x in r)+"\n")
    print(f"\n[out] -> {a.out}")
    # field gradient deep vs photosphere (ground-state, purely field-driven)
    s0,sN=store['s0'],store['s8']
    for tag,i in (('FeIII',4),('CoIII',6)):
        r=s0[i]/sN[i] if sN[i]>0 else float('nan')
        print(f">>> {tag} Gamma_gnd(s0={s0[0]})/Gamma_gnd(s10={sN[0]}) = {r:.2f} = {np.log10(r):+.2f} dex (FIELD-driven)")
    print(f"\nLumina flat-field denominators: Fe III {LUM_DENOM[(26,2)]:.3e}, Co III {LUM_DENOM[(27,2)]:.3e} s^-1")
    print("NOTE: field+ground/Boltzmann rate, NOT full NLTE. Compare the shell GRADIENT, not absolute.")

if __name__=='__main__':
    main()
