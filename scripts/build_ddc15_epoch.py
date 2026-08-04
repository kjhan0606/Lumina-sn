#!/usr/bin/env python3
"""Build a LUMINA reference for DDC15 at a LATER epoch by HOMOLOGOUS expansion of
the 0.976d hydro file + radioactive decay of the composition. Extends
build_ddc15_initial_epoch.py (which only changed t_exp + decayed, but did NOT
homologously scale R/rho -> wrong for target != 0.976d).

Homologous expansion (R = v t):  v invariant; R(t)=R0*(t/t0); rho(t)=rho0*(t0/t)^3.
Rosseland tau scales as tau(t)=tau0*(t0/t)^2 (chi~rho, dr~t) for the photosphere.
Energy: L_inner = gold bolometric L at the epoch (integrated DDC15 emergent
spectrum at 10 pc); T_inner = (L/(4 pi r_inner^2 sigma))^(1/4).

Usage: build_ddc15_epoch.py KEEPER_REF OUT_REF TARGET_EPOCH_D [tau_phot] [GOLD_SPEC]
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HYDRO = Path("data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d")
T0_D = 0.976
SIGMA_SB = 5.670374e-5
AMU = 1.66053906660e-24
PC = 3.085677581e18
LN2 = np.log(2.0)
Z_LIST = [6, 8, 12, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28]
NAME2Z = {"HYD":1,"HE":2,"CARB":6,"NIT":7,"OXY":8,"FLU":9,"NEON":10,"SOD":11,
    "MAG":12,"ALUM":13,"SIL":14,"PHOS":15,"SUL":16,"CHL":17,"ARG":18,"POT":19,
    "CAL":20,"SCAN":21,"TIT":22,"VAN":23,"CHRO":24,"MAN":25,"IRON":26,"COB":27,
    "NICK":28,"BAR":56}
Z2NAME = {v:k for k,v in NAME2Z.items()}
A_AMU = {1:1.008,2:4.0026,6:12.011,7:14.007,8:15.999,9:18.998,10:20.180,11:22.990,
    12:24.305,13:26.982,14:28.085,15:30.974,16:32.06,17:35.45,18:39.948,19:39.098,
    20:40.078,21:44.956,22:47.867,23:50.942,24:51.996,25:54.938,26:55.845,27:58.933,
    28:58.693,56:137.33}
DECAY = {(28,56):(27,6.075),(27,56):(26,77.236),(28,57):(27,1.483),(27,57):(26,271.74),
    (27,55):(26,17.53),(26,55):(25,1002.0),(26,52):(25,0.3448),(25,52):(24,5.591),
    (24,48):(23,0.898),(23,48):(22,15.973),(22,44):(21,21915.0),(21,44):(20,0.1654),
    (25,51):(24,0.0321),(24,51):(23,27.70),(24,49):(23,0.0294),(23,49):(22,330.0),
    (26,53):(25,0.0059),(22,45):(21,0.1283),(20,47):(21,4.536),(21,47):(22,3.3492),
    (21,46):(22,83.79),(21,43):(20,0.1626)}

def parse_hydro(path):
    blocks, name, vals = {}, None, []
    def isnum(s):
        t=s.split()
        if not t: return False
        try: [float(x) for x in t]; return True
        except ValueError: return False
    for ln in path.read_text().splitlines():
        if isnum(ln): vals.extend(float(x) for x in ln.split())
        else:
            if name and vals: blocks[name]=np.array(vals)
            name, vals = ln.strip(), []
    if name and vals: blocks[name]=np.array(vals)
    return blocks

def build_isotope_elements(blocks, n, target_epoch_d):
    iso={}
    for Z in range(17,29):
        nm=Z2NAME[Z]
        for h in blocks:
            toks=h.split()
            if len(toks)>=2 and toks[0]==nm and toks[1].isdigit():
                iso[(Z,int(toks[1]))]=blocks[h][:n].copy()
    dt=0.01; nstep=int(round((target_epoch_d-T0_D)/dt))
    lam={k:LN2/DECAY[k][1] for k in DECAY}
    for _ in range(max(nstep,0)):
        dX={k:np.zeros(n) for k in iso}
        for k,arr in iso.items():
            if k in lam:
                loss=lam[k]*arr*dt; dX[k]-=loss
                kd=(DECAY[k][0],k[1])
                if kd in dX: dX[kd]+=loss
        for k in iso: iso[k]=np.clip(iso[k]+dX[k],0.0,None)
    Xiso={}
    for Z in range(17,29):
        tot=np.zeros(n)
        for (Zi,A),arr in iso.items():
            if Zi==Z: tot+=arr
        Xiso[Z]=tot
    return Xiso, nstep

def main(keeper, out, target_epoch_d, tau_phot, gold_spec):
    b=parse_hydro(HYDRO)
    def g(k,n=115):
        for h,a in b.items():
            if h.startswith(k): return a[:n]
        raise KeyError(k)
    t_ratio = target_epoch_d / T0_D
    # 0.976d native arrays
    R0 = g("Radius grid")*1e10
    v  = g("Velocity (km/s)")*1e5          # invariant
    T0 = g("Temperature (10^4 K)")*1e4
    rho0 = g("Density")
    ne0 = g("Electron density")
    chi0 = g("Rosseland mean opacity")*1e-10
    atomdens_file = g("Atom density")
    npt=len(R0)
    # --- HOMOLOGOUS expansion to target epoch ---
    R   = R0 * t_ratio
    rho = rho0 * (1.0/t_ratio)**3
    ne  = ne0  * (1.0/t_ratio)**3
    t_exp_s = target_epoch_d*86400.0
    # Rosseland tau at target: chi ~ rho (kappa~const) => tau scales (t0/t)^2
    chi = chi0 * (1.0/t_ratio)**3
    tau=np.zeros_like(R)
    for i in range(1,len(R)):
        tau[i]=tau[i-1]+0.5*(chi[i-1]+chi[i])*abs(R[i-1]-R[i])
    i_phot=int(np.searchsorted(tau,tau_phot))
    if i_phot>=npt: i_phot=npt-1
    v_inner=v[i_phot]; r_inner=R[i_phot]
    # --- Energy: L_inner from gold bolometric (integrate emergent spectrum @10pc) ---
    if gold_spec and Path(gold_spec).exists():
        gs=np.loadtxt(gold_spec); L_inner=4*np.pi*(10*PC)**2*np.trapezoid(gs[:,1],gs[:,0])
        L_src=f"gold bolometric ({gold_spec})"
    else:
        L_inner=4*np.pi*r_inner**2*SIGMA_SB*T0[i_phot]**4; L_src="SB(T_gas) fallback"
    T_inner=(L_inner/(4*np.pi*r_inner**2*SIGMA_SB))**0.25
    # --- decayed composition ---
    Xiso,nstep=build_isotope_elements(b,npt,target_epoch_d)
    Xel={}
    for Z in Z_LIST:
        Xel[Z]= g(Z2NAME[Z]+" mass fraction").copy() if Z<=16 else Xiso[Z]
    Xfull=np.zeros(npt)
    for nm,Z in NAME2Z.items():
        try: Xfull+=g(nm+" mass fraction")
        except KeyError: pass
    # domain: photosphere..outer, inner->outer
    sel=np.arange(0,i_phot+1)[::-1]; n_shells=len(sel)
    R_s,rho_s,ne_s,T_s=R[sel],rho[sel],ne[sel],T0[sel]*(1.0/t_ratio)  # T init ~ t^-1 guess
    Xfull_s=Xfull[sel]
    r_edge=np.empty(n_shells+1)
    r_edge[1:-1]=0.5*(R_s[:-1]+R_s[1:]); r_edge[0]=r_inner
    r_edge[-1]=R_s[-1]+(R_s[-1]-r_edge[-2])
    v_edge=r_edge/t_exp_s
    M=np.vstack([Xel[Z][sel] for Z in Z_LIST]); M=M/np.where(Xfull_s>0,Xfull_s,1.0)
    # --- write ---
    keeper,out=Path(keeper),Path(out); out.mkdir(parents=True,exist_ok=True)
    rewrite={"geometry.csv","density.csv","abundances.csv","abundances.npy",
             "electron_densities.csv","plasma_state.csv","config.json"}
    for f in keeper.iterdir():
        if f.name in rewrite: continue
        link=out/f.name
        if link.exists() or link.is_symlink(): link.unlink()
        link.symlink_to(f.resolve())
    sid=np.arange(n_shells)
    pd.DataFrame({"shell_id":sid,"r_inner":r_edge[:-1],"r_outer":r_edge[1:],
                  "v_inner":v_edge[:-1],"v_outer":v_edge[1:]}).to_csv(out/"geometry.csv",index=False)
    pd.DataFrame({"shell_id":sid,"rho":rho_s}).to_csv(out/"density.csv",index=False)
    pd.DataFrame({"shell_id":sid,"n_e":ne_s}).to_csv(out/"electron_densities.csv",index=False)
    cols=["atomic_number"]+[str(s) for s in sid]
    pd.DataFrame([[Z]+list(M[i]) for i,Z in enumerate(Z_LIST)],columns=cols).to_csv(out/"abundances.csv",index=False)
    W_init=0.5*(r_inner/R_s)**2
    pd.DataFrame({"shell_id":sid,"W":W_init,"T_rad":T_s}).to_csv(out/"plasma_state.csv",index=False)
    with open(keeper/"config.json") as fh: cfg=json.load(fh)
    cfg.update(n_shells=int(n_shells),time_explosion_s=float(t_exp_s),
               luminosity_inner_erg_s=float(L_inner),v_inner_min_cm_s=float(v_inner),
               v_outer_max_cm_s=float(v[sel][-1]),T_inner_K=round(float(T_inner),-1))
    with open(out/"config.json","w") as fh: json.dump(cfg,fh,indent=2)
    # report
    print(f"[ddc15-epoch] {T0_D}d -> {target_epoch_d}d  (t_ratio={t_ratio:.2f}, {nstep} decay steps)")
    print(f"[ddc15-epoch] photosphere tau={tau_phot}: v_inner={v_inner/1e5:.0f} km/s  r_inner={r_inner:.3e}cm  tau_tot={tau[-1]:.2f}")
    print(f"[ddc15-epoch] L_inner={L_inner:.3e} erg/s ({L_src})  T_inner={T_inner:.0f}K")
    print(f"[ddc15-epoch] domain: {n_shells} shells  v=[{v_inner/1e5:.0f},{v[sel][-1]/1e5:.0f}] km/s")
    print(f"[ddc15-epoch] inner-shell composition (decayed): "+"  ".join(f"{Z2NAME[Z]}={M[i,0]:.3f}" for i,Z in enumerate(Z_LIST) if M[i,0]>1e-2))
    # iron-group decay summary at inner shell
    print(f"[ddc15-epoch] Fe-group inner shell: Fe={M[Z_LIST.index(26),0]:.3f} Co={M[Z_LIST.index(27),0]:.3f} Ni={M[Z_LIST.index(28),0]:.3f}")
    print(f"[ddc15-epoch] wrote {out}")

if __name__=="__main__":
    keeper=sys.argv[1]; out=sys.argv[2]; tgt=float(sys.argv[3])
    tau_phot=float(sys.argv[4]) if len(sys.argv)>4 else 2.0/3.0
    gold=sys.argv[5] if len(sys.argv)>5 else ""
    main(keeper,out,tgt,tau_phot,gold)
