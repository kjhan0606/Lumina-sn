#!/usr/bin/env python3
"""Offline closed-physics cell balance (ARTIS-cell mirror) for toy06 outer shells.

Per shell, SELF-CONTAINED (no NLTE matrix, no tdep-IC, no caps):
  ionization:  n_{j+1}/n_j = (Gamma_photo(J_dump) + Gamma_nt) / (n_e * alpha(T_e))
  levels:      Boltzmann at T_e within each ion (resonance-line cooling dominated)
  cooling:     two-level SE per line (collisional supply vs Sobolev-escape radiative)
               Lambda = sum dE * (n_lo*C_lu - n_up*C_ul)   [ETLA form, thin-correct]
  heating:     H_dep (deposition, full) + H_photo (bf excess from J_dump)
  T_e:         bisection on H - Lambda, ionization RE-SOLVED at every T_e evaluation
               (ARTIS thermalbalance.cc:144 simultaneous coupling)

Inputs are ALL real: J_nu dump (converged transport), CMFGEN deposition file,
model composition/density, line list + levels. Compare vs CMFGEN phys/ionfrac.
Uncertainty knobs exposed: --eta (NT fraction), --alpha-scale, --gbar (vR).
"""
import argparse, csv, math, sys, os
import numpy as np

H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10; EV=1.602176634e-12
AMU=1.66053906660e-24; RY_EV=13.605693
T_EXP=19.48*86400.0
BASE="data/tardis_reference_toy06_19p48d"
MASS={14:28.085,16:32.06,20:40.078,6:12.011,8:15.999,26:55.845,27:58.933,28:58.693,
      12:24.305,13:26.98,22:47.867,23:50.94,24:51.996,25:54.938,21:44.956,11:22.99,19:39.1,18:39.95,10:20.18}

def load_shells():
    rho={};
    for r in csv.DictReader(open(f"{BASE}/density.csv")): rho[int(r['shell_id'])]=float(r['rho'])
    dep={}
    for r in csv.DictReader(open(f"{BASE}/deposition_cmfgen.csv")): dep[int(r['shell_id'])]=float(r['heating_rate'])
    geo={}
    for r in csv.DictReader(open(f"{BASE}/geometry.csv")):
        geo[int(r['shell_id'])]=0.5*(float(r['v_inner'])+float(r['v_outer']))/1e5  # km/s
    ab={}  # ab[s][Z]
    rows=list(csv.reader(open(f"{BASE}/abundances.csv")))
    hdr=rows[0]
    for row in rows[1:]:
        Z=int(row[0])
        for i,v in enumerate(row[1:]):
            ab.setdefault(i,{})[Z]=float(v)
    return rho,dep,geo,ab

def load_chi():
    chi={}
    for r in csv.DictReader(open(f"{BASE}/ionization_energies.csv")):
        chi[(int(r['atomic_number']),int(r['ion_number']))]=float(r['ionization_energy_eV'])
    return chi

def load_levels(Zs):
    lev={}  # (Z,ion) -> (E_eV array, g array)
    for r in csv.DictReader(open(f"{BASE}/levels.csv")):
        Z=int(r['atomic_number'])
        if Z not in Zs: continue
        key=(Z,int(r['ion_number']))
        lev.setdefault(key,{})[int(r['level_number'])]=(float(r['energy_eV']),float(r['g']))
    out={}
    for key,d in lev.items():
        n=max(d.keys())+1
        E=np.zeros(n); g=np.ones(n)
        for k,(e,gg) in d.items(): E[k]=e; g[k]=gg
        out[key]=(E,g)
    return out

def load_lines(Zs, lev):
    """per (Z,ion): arrays nu, f_lu, A_ul, E_lo, g_lo, g_up, dE"""
    acc={}
    with open(f"{BASE}/line_list.csv") as fh:
        rd=csv.DictReader(fh)
        for r in rd:
            Z=int(r['atomic_number'])
            if Z not in Zs: continue
            ion=int(r['ion_number']); key=(Z,ion)
            if key not in lev: continue
            E,g=lev[key]
            lo=int(r['level_number_lower']); up=int(r['level_number_upper'])
            if lo>=len(E) or up>=len(E): continue
            acc.setdefault(key,[]).append((float(r['nu']),float(r['f_lu']),float(r['A_ul']),
                                           E[lo],g[lo],g[up]))
    out={}
    for key,rows in acc.items():
        a=np.array(rows)
        out[key]=dict(nu=a[:,0],f_lu=a[:,1],A_ul=a[:,2],E_lo=a[:,3],g_lo=a[:,4],g_up=a[:,5],
                      dE=H*a[:,0])
    return out

def load_J(path="logs/stage1_toy06_jdump/lumina_cmfgen_jnu.csv"):
    Js={}
    for r in csv.DictReader(open(path)):
        s=int(r['shell'])
        Js.setdefault(s,[]).append((float(r['nu']),float(r['J'])))
    out={}
    for s,rows in Js.items():
        rows.sort()
        a=np.array(rows)
        out[s]=(a[:,0],a[:,1])
    return out

_DR={}
def load_dr():
    p="data/atomic/dr_norad/badnell_dr_si_s_ca.txt"
    if not os.path.exists(p): return
    for line in open(p):
        if line.startswith("#") or not line.strip(): continue
        parts=line.split("#")[0].split()
        Z=int(parts[0]); chg=int(parts[1]); n=int(parts[2])
        c=[float(parts[3+2*i]) for i in range(n)]
        E=[float(parts[4+2*i]) for i in range(n)]
        _DR[(Z,chg)]=(c,E)
def alpha_rec(Z, zrec, T, scale, use_dr=True):
    """RR (hydrogenic scaling) + real Badnell DR where tabulated.
    zrec = charge of recombining ion (>=1). scale applies to RR only."""
    a=scale*2.6e-13*zrec**1.6*(T/1e4)**-0.8
    if use_dr and (Z,zrec) in _DR:
        c,E=_DR[(Z,zrec)]
        a+=T**-1.5*sum(ci*math.exp(-min(Ei/T,300.0)) for ci,Ei in zip(c,E))
    return a

def U_of(lev,key,T):
    E,g=lev[key]
    x=E*EV/(KB*T)
    return float(np.sum(g*np.exp(-np.minimum(x,300.0))))

def gamma_photo(nuJ, chi_eV, zeff):
    nu,J=nuJ
    nu0=chi_eV*EV/H
    m=nu>=nu0
    if not m.any(): return 0.0,0.0
    nn=nu[m]; JJ=J[m]
    dnu=np.gradient(nn)
    sig=7.91e-18/max(zeff,1)**2*(nu0/nn)**3
    G=float(np.sum(4*math.pi*sig*JJ/(H*nn)*dnu))
    Hp=float(np.sum(4*math.pi*sig*JJ/(H*nn)*(H*nn-chi_eV*EV)*dnu))
    return G,Hp

class Cell:
    def __init__(s_, s, rho, dep, ab, chi, lev, lines, nuJ, args):
        s_.s=s; s_.dep=dep; s_.chi=chi; s_.lev=lev; s_.lines=lines; s_.nuJ=nuJ; s_.a=args
        s_.nel={Z:ab[Z]*rho/(MASS[Z]*AMU) for Z in args.Zs if ab.get(Z,0)>1e-8}
        s_.natom=sum(s_.nel.values())
        # per (Z, pair j->j+1): Gamma_photo & H_photo weight (from ground of stage j)
        s_.gph={}; s_.hph={}
        for Z in s_.nel:
            for j in range(0,args.maxstage):
                ch=chi.get((Z,j))
                if ch is None: continue
                G,Hp=gamma_photo(nuJ,ch,j)   # zeff ~ charge after? use j+1 conservative
                s_.gph[(Z,j)]=G; s_.hph[(Z,j)]=Hp
        s_.gnt_atom=(args.eta*dep/(args.wion*EV))/max(s_.natom,1e-30)
    def gnt(s_, Z, j):
        """per-stage NT rate: Lotz sigma ~ 1/chi^2 + spectrum softening ->
        Gamma_nt,j = gnt_atom * (chi_ref/chi_j)^p  (chi_ref = wion eV)."""
        ch=s_.chi.get((Z,j),1e9)
        return s_.gnt_atom*(s_.a.wion/ch)**s_.a.ntp
    def ionize(s_, T, n_e):
        """one pass: given n_e,T -> fractions + new n_e"""
        fr={}; ne_new=0.0
        for Z,nZ in s_.nel.items():
            r=[]
            for j in range(0,s_.a.maxstage):
                if (Z,j) not in s_.chi: r.append(0.0); continue
                G=s_.gph.get((Z,j),0.0)+s_.gnt(Z,j)
                al=alpha_rec(Z,j+1,T,s_.a.alpha_scale,use_dr=not s_.a.no_dr)
                r.append(G/(max(n_e,1.0)*al))
            y=[1.0];
            for j in range(len(r)): y.append(y[-1]*max(r[j],0.0))
            y=np.array(y); y/= y.sum()
            fr[Z]=y
            ne_new+=nZ*float(np.sum(np.arange(len(y))*y))
        return fr,ne_new
    def solve_ion(s_, T):
        n_e=0.5*s_.natom
        for it in range(80):
            fr,ne=s_.ionize(T,n_e)
            n_e=0.5*(n_e+max(ne,1e-6*s_.natom))
        return fr,n_e
    def cooling(s_, T, fr, n_e):
        lam=0.0; contrib={}
        for (Z,ion),L in s_.lines.items():
            if Z not in s_.nel or ion>=len(fr[Z]): continue
            nion=s_.nel[Z]*fr[Z][ion]
            if nion<=1e-12: continue
            U=U_of(s_.lev,(Z,ion),T)
            x=L['E_lo']*EV/(KB*T)
            n_lo=nion*L['g_lo']*np.exp(-np.minimum(x,300.0))/U
            dE=L['dE']; nu=L['nu']
            # vR collision strength: Omega = 14.5*gbar*f_lu*g_lo*(Ry/dE)
            Om=14.5*s_.a.gbar*L['f_lu']*L['g_lo']*(RY_EV/(dE/EV))
            qul=8.63e-6*Om/(L['g_up']*math.sqrt(T))
            qlu=qul*(L['g_up']/L['g_lo'])*np.exp(-np.minimum(dE/(KB*T),300.0))
            lam_cm=C/nu
            tau=0.02654*L['f_lu']*lam_cm*n_lo*T_EXP
            beta=np.where(tau<1e-6,1.0,(1.0-np.exp(-np.minimum(tau,700)))/np.maximum(tau,1e-6))
            Jb=np.interp(nu,s_.nuJ[0],s_.nuJ[1])
            Blu=(C*C/(2*H*nu**3))*(L['g_up']/L['g_lo'])*L['A_ul']
            Bul=(C*C/(2*H*nu**3))*L['A_ul']
            Rlu=Blu*Jb*beta; Rul=(L['A_ul']+Bul*Jb)*beta
            Clu=n_e*qlu; Cul=n_e*qul
            nup=n_lo*(Clu+Rlu)/np.maximum(Cul+Rul,1e-300)
            nup_lte=n_lo*(L['g_up']/L['g_lo'])*np.exp(-np.minimum(dE/(KB*T),300.0))
            nup=np.minimum(nup,nup_lte)          # no-pumping guard (ETLA)
            net=float(np.sum(dE*(n_lo*Clu-nup*Cul)))
            lam+=net
            contrib[(Z,ion)]=net
        return lam,contrib
    def heating(s_, fr):
        Hp=0.0
        for Z,nZ in s_.nel.items():
            for j in range(min(len(fr[Z]),s_.a.maxstage)):
                Hp+=nZ*fr[Z][j]*s_.hph.get((Z,j),0.0)
        return s_.dep + max(Hp,0.0)
    def fb_cool(s_, T, fr, n_e):
        """free-bound (recombination) cooling: n_e*n_{j+1}*alpha*(chi + kT)."""
        lam=0.0
        for Z,nZ in s_.nel.items():
            for j in range(0,s_.a.maxstage):
                ch=s_.chi.get((Z,j))
                if ch is None or j+1>=len(fr[Z]): continue
                al=alpha_rec(Z,j+1,T,s_.a.alpha_scale,use_dr=not s_.a.no_dr)
                lam+=n_e*nZ*fr[Z][j+1]*al*(ch*EV+KB*T)
        return lam
    def net(s_, T):
        fr,n_e=s_.solve_ion(T)
        lam,_=s_.cooling(T,fr,n_e)
        lam+=s_.fb_cool(T,fr,n_e)
        return s_.heating(fr)-lam, fr, n_e, lam
    def solve(s_):
        Tlo,Thi=2000.0,140000.0
        flo,_,_,_=s_.net(Tlo); fhi,_,_,_=s_.net(Thi)
        if flo<=0: T=Tlo
        elif fhi>=0: T=Thi
        else:
            for _ in range(40):
                Tm=0.5*(Tlo+Thi)
                fm,_,_,_=s_.net(Tm)
                if fm>0: Tlo=Tm
                else: Thi=Tm
            T=0.5*(Tlo+Thi)
        f,fr,n_e,lam=s_.net(T)
        return T,fr,n_e,lam

def cmfgen_truth():
    T={};
    intime=False; n=0
    for line in open("data/standart_data1/toy06/phys_toy06_cmfgen.txt"):
        if line.startswith("#TIME:"):
            intime=abs(float(line.split()[1])-19.48)<0.01; n=0; continue
        if line.startswith("#") or not line.strip(): continue
        if intime:
            p=line.split(); T[float(p[0])]=float(p[1])
    return T

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--eta",type=float,default=0.05)
    ap.add_argument("--wion",type=float,default=35.0)
    ap.add_argument("--alpha-scale",type=float,default=1.0)
    ap.add_argument("--gbar",type=float,default=0.2)
    ap.add_argument("--shells",default="25,30,35,37,40,43,46,49")
    ap.add_argument("--maxstage",type=int,default=6)
    ap.add_argument("--no-dr",action="store_true")
    ap.add_argument("--ntp",type=float,default=2.0)
    ap.add_argument("--jdump",default="logs/stage1_toy06_jdump/lumina_cmfgen_jnu.csv")
    args=ap.parse_args()
    args.Zs={14,16,20}
    load_dr()
    print(f"DR table entries: {sorted(_DR.keys())}")
    rho,dep,geo,ab=load_shells()
    chi=load_chi()
    print("loading levels/lines/J ...",flush=True)
    lev=load_levels(args.Zs)
    lines=load_lines(args.Zs,lev)
    Js=load_J(args.jdump)
    truth=cmfgen_truth()
    tv=sorted(truth.keys())
    shells=[int(x) for x in args.shells.split(",")]
    print(f"\n knobs: eta={args.eta} wion={args.wion} alpha x{args.alpha_scale} gbar={args.gbar}")
    print(f"{'s':>3} {'v[km/s]':>8} | {'T_pred':>7} {'T_CMFGEN':>8} | {'n_e':>9} | {'Lam':>9} {'H_dep':>9} | Si분율(II,III,IV,V) | S(II,III,IV)")
    for s in shells:
        if s not in Js: print(f"{s:>3}  (no J)"); continue
        cell=Cell(s,rho[s],dep[s],ab[s],chi,lev,lines,Js[s],args)
        T,fr,n_e,lam=cell.solve()
        v=geo[s]
        # CMFGEN T at nearest v
        Tc=truth[min(tv,key=lambda x:abs(x-v))]
        si=fr.get(14,np.zeros(6)); su=fr.get(16,np.zeros(6))
        print(f"{s:>3} {v:8.0f} | {T:7.0f} {Tc:8.0f} | {n_e:9.2e} | {lam:9.2e} {dep[s]:9.2e} | "
              f"{si[1]:.2f},{si[2]:.2f},{si[3]:.2f},{si[4]:.2f} | {su[1]:.2f},{su[2]:.2f},{su[3]:.2f}")
    # s=49 detail: dominant coolants + rates
    s=49
    cell=Cell(s,rho[s],dep[s],ab[s],chi,lev,lines,Js[s],args)
    T,fr,n_e,lam=cell.solve()
    lam2,contrib=cell.cooling(T,fr,n_e)
    print(f"\n=== s=49 detail @T={T:.0f}: Gamma_nt={cell.gnt_atom:.2e}/atom")
    for (Z,ion),val in sorted(contrib.items(),key=lambda kv:-abs(kv[1]))[:6]:
        G=cell.gph.get((Z,ion),0)
        print(f"  Z={Z} ion={ion}: cooling={val:.3e}  Gamma_photo(ground)={G:.2e}")

if __name__=="__main__":
    main()
