#!/usr/bin/env python3
"""Offline co-evolution physics test (2026-07-07): does a HOT decoupled radiation
field (T_R > T_e, dilute W<1 -- what the MC-in-loop rewiring would supply) turn ON
fluorescence, that the COLLAPSED field (J=B(T_e), current Lumina) does not?

Decoupled from "is Lumina's field hot" (Stage-0 showed the deterministic field is
NOT; the rewiring's job is to make it hot via MC transport). This tests the
DOWNSTREAM link: given a hot field, does the NLTE population solve produce b_k>1 on
UV-pumped levels and enhance the OPTICAL line emissivity (= fluorescence)?

Method: reduced multi-level statistical-equilibrium solve for one ion (lowest N
levels), radiative (A_ul + B_ul*J up/down, B_lu*J) + collisional (van Regemorter /
floor) rates, under two fields:
  COLD: J(nu) = B(nu, T_e)                       [thermal, b_k -> 1, LTE]
  HOT:  J(nu) = W * B(nu, T_R), T_R>T_e, W<1     [ARTIS-like dilute-hot]
Then compare, per band, the line emissivity eta = sum_lines n_u * A_ul * h*nu.
FLUORESCENCE = HOT field raises the OPTICAL/UV emissivity ratio vs COLD.

Usage: python3 scripts/offline_fluor_field_test.py [Z ion] [shell] [Nlev] [TR] [W]
  default: Fe II (26 1), shell 10, 120 levels, TR=15000, W=0.1
"""
import sys, csv, math
import numpy as np

H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10; ME=9.1093837e-28
EV=1.602176634e-12; E_RYD=2.1798723611e-11; A0=5.29177e-9
Z    = int(sys.argv[1]) if len(sys.argv)>1 else 26
ION  = int(sys.argv[2]) if len(sys.argv)>2 else 1
SH   = int(sys.argv[3]) if len(sys.argv)>3 else 10
NLEV = int(sys.argv[4]) if len(sys.argv)>4 else 120
TR   = float(sys.argv[5]) if len(sys.argv)>5 else 15000.0
W    = float(sys.argv[6]) if len(sys.argv)>6 else 0.1
LD='data/tardis_reference_toy06_19p48d'

# plasma at shell
ps={int(r['shell_id']):(float(r['T_e']),float(r['n_e'])) for r in
    csv.DictReader(open('logs/stage1_toy06_epay27/lumina_plasma_state.csv'))}
Te,ne = ps[SH]

# lowest NLEV levels of the ion (sorted by energy)
lv=[]
for r in csv.DictReader(open(f'{LD}/levels.csv')):
    if int(r['atomic_number'])==Z and int(r['ion_number'])==ION:
        lv.append((int(r['level_number']),float(r['energy_eV']),float(r['g'])))
lv.sort(key=lambda x:x[1]); lv=lv[:NLEV]
lnmap={ln:i for i,(ln,e,g) in enumerate(lv)}
E=np.array([e for _,e,g in lv])*EV; g=np.array([gg for _,e,gg in lv])
N=len(lv)

# lines among these levels
low=[];up=[];Aul=[];nu=[];flu=[]
for r in csv.DictReader(open(f'{LD}/line_list.csv')):
    if int(r['atomic_number'])==Z and int(r['ion_number'])==ION:
        lo=int(r['level_number_lower']); hi=int(r['level_number_upper'])
        if lo in lnmap and hi in lnmap:
            low.append(lnmap[lo]); up.append(lnmap[hi]); Aul.append(float(r['A_ul']))
            nu.append(float(r['nu'])); flu.append(float(r['f_lu']))
low=np.array(low);up=np.array(up);Aul=np.array(Aul);nu=np.array(nu);flu=np.array(flu)
lam=C/nu*1e8
print(f"{['','I','II','III','IV'][ION+1] if ION+1<5 else ION+1} Z={Z} ion={ION} shell {SH}: "
      f"Te={Te:.0f} ne={ne:.2e} ; {N} levels, {len(nu)} lines ; HOT: TR={TR:.0f} W={W}")

def Bnu(nu_,T):
    x=H*nu_/(KB*T); return np.where(x<500,2*H*nu_**3/C**2/np.expm1(np.clip(x,1e-30,500)),0.0)

def col_vanreg(f,nu_,dE):
    # van Regemorter collisional (de)excitation upward coeff (cm^3/s), allowed lines
    x=dE/(KB*Te)
    gbar=np.maximum(0.2,0.276*np.exp(x)* -np.expm1(-x)*0)  # placeholder; use 0.2 floor
    gbar=np.full_like(f,0.2)
    # C_lu = 5.465e-11 * sqrt(Te) * 14.5 * f * (E_H/dE) * gbar * exp(-x)  (approx)
    return 5.465e-11*np.sqrt(Te)*14.5*np.maximum(f,1e-6)*(E_RYD/dE)*gbar*np.exp(-np.minimum(x,500))

def solve(field):  # field: (Tfield, amp) -> J(nu)=amp*B(nu,Tfield)
    Tf,amp=field
    J=amp*Bnu(nu,Tf)
    Bul=Aul*C**2/(2*H*nu**3); Blu=(g[up]/g[low])*Bul
    dE=E[up]-E[low]
    R_up = Blu*J                      # low->up radiative
    R_dn = Aul + Bul*J                # up->low radiative
    C_up = ne*col_vanreg(flu,nu,dE)   # low->up collisional
    C_dn = C_up*(g[low]/g[up])*np.exp(np.minimum(dE/(KB*Te),500))  # detailed balance
    M=np.zeros((N,N))
    for k in range(len(low)):
        l,u=low[k],up[k]
        up_rate=R_up[k]+C_up[k]; dn_rate=R_dn[k]+C_dn[k]
        M[u,l]+=up_rate; M[l,l]-=up_rate
        M[l,u]+=dn_rate; M[u,u]-=dn_rate
    # replace one row with normalization sum(n)=1
    M[0,:]=1.0; b=np.zeros(N); b[0]=1.0
    n=np.linalg.lstsq(M,b,rcond=None)[0]
    n=np.maximum(n,0)
    return n

# LTE reference populations (Boltzmann at Te) for b_k
nlte_lte=g*np.exp(-np.minimum(E/(KB*Te),500)); nlte_lte/=nlte_lte.sum()

def emis(n):  # per-band line emissivity sum n_u*A_ul*h*nu
    e_uv=np.sum(n[up]*Aul*H*nu*((lam>=2500)&(lam<3700)))
    e_opt=np.sum(n[up]*Aul*H*nu*((lam>=4400)&(lam<7000)))
    return e_uv,e_opt

for name,field in [('COLD J=B(Te)',(Te,1.0)),('HOT J=W*B(TR)',(TR,W))]:
    n=solve(field)
    bk=n/nlte_lte; bk/=bk[0]  # departure relative to ground
    e_uv,e_opt=emis(n)
    # b_k of the UV-absorbing upper levels (levels reached by UV lines from ground-ish)
    uv_up=np.unique(up[(lam>=2500)&(lam<3700)&(low<5)])
    bk_uv=np.median(bk[uv_up]) if len(uv_up) else float('nan')
    print(f"\n {name}:")
    print(f"   median b_k of UV-pumped upper levels = {bk_uv:.2f}  (LTE=1; >1 = pumped)")
    print(f"   line emissivity: UV(2500-3700)={e_uv:.3e}  optical(4400-7000)={e_opt:.3e}  opt/UV={e_opt/max(e_uv,1e-99):.3f}")

print("\nVERDICT: if HOT field gives b_k(UV)>1 AND higher opt/UV than COLD => a hot")
print("decoupled field turns ON fluorescence -> the MC-in-loop rewiring (which supplies")
print("that field) is the correct fix. If opt/UV unchanged => hot field alone insufficient.")
