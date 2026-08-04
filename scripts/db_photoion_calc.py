#!/usr/bin/env python3
"""Detailed-balance ionization calculator over the baked CMFGEN sigma_bf grid.
Mirrors src/lumina_plasma.c: G (5049-5202) three ways, alpha=frozenin_alpha_rr (2708).

SIGMA PROVENANCE (2026-07-17 audit): the default toy06 bin is MIXED --
FeIII/SiII/SIII/CaII carry REAL tabulated CMFGEN sigma_bf, but Co III fell back
to Kramers nu^-3 (its phot_data never matched at bake time; 'Very crude' 386-level
term-grouped file).  So Co III photoion rates here are DATA-INSENSITIVE (Kramers).
The startup _kramers_scan() + per-ion ion_is_kramers() flag this so a data-sensitive
conclusion is never drawn silently.  Override the bin with LUMINA_SIGMA_BIN
(+ LUMINA_SIGMA_BIN_LEVELS companion levels.csv when its NLEV != the ref's).
"""
import os, sys, struct, csv, math
import numpy as np

H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10; ME=9.1093837015e-28
EV=1.602176634e-12; PI=math.pi; KB_EV=8.617333262e-5

# ---- reference dir + sigma bin (LUMINA_SIGMA_BIN override; default unchanged) ----
REF_DIR   = os.environ.get('LUMINA_REF_DIR', 'data/tardis_reference_toy06_19p48d')
SIGMA_BIN = os.environ.get('LUMINA_SIGMA_BIN', f'{REF_DIR}/cmfgen_sigma_bf.bin')
SIGMA_BIN_LEVELS = os.environ.get('LUMINA_SIGMA_BIN_LEVELS', '')  # companion if NLEV != ref

# ---- levels (tool's canonical level order = REF_DIR/levels.csv) ----
levZ=[];levI=[];levN=[];levE=[];levG=[]
for r in csv.DictReader(open(f'{REF_DIR}/levels.csv')):
    levZ.append(int(r['atomic_number']));levI.append(int(r['ion_number']))
    levN.append(int(r['level_number']));levE.append(float(r['energy_eV']));levG.append(int(r['g']))
levZ=np.array(levZ);levI=np.array(levI);levN=np.array(levN);levE=np.array(levE);levG=np.array(levG)
N_CANON=len(levZ)

# ---- sigma grid ----
f=open(SIGMA_BIN,'rb')
magic,ver,NLEV,NFREQ=struct.unpack('<IIii',f.read(16))
numin,numax=struct.unpack('<dd',f.read(16))
_flags_raw=np.frombuffer(f.read(NLEV),dtype=np.int8)
pad=(8-(NLEV%8))%8
f.read(pad)
_SIG_raw=np.memmap(SIGMA_BIN,dtype='<f8',mode='r',offset=32+NLEV+pad,shape=(NLEV,NFREQ))
log_numin=math.log(numin); dlog=(math.log(numax)-log_numin)/NFREQ
nu_c=np.exp(log_numin+(np.arange(NFREQ)+0.5)*dlog)
dnu=np.exp(log_numin+(np.arange(NFREQ)+1)*dlog)-np.exp(log_numin+np.arange(NFREQ)*dlog)

if NLEV==N_CANON:
    SIG=_SIG_raw; flags=_flags_raw
else:
    # remap bin rows -> canonical level order by (Z,ion,level_number); missing -> flags=0
    if not SIGMA_BIN_LEVELS:
        raise SystemExit(f"SIGMA_BIN NLEV={NLEV} != ref levels {N_CANON}; set "
                         f"LUMINA_SIGMA_BIN_LEVELS to the bin's companion levels.csv")
    key2row={}
    for i,r in enumerate(csv.DictReader(open(SIGMA_BIN_LEVELS))):
        key2row[(int(r['atomic_number']),int(r['ion_number']),int(r['level_number']))]=i
    rowmap=np.array([key2row.get((int(levZ[i]),int(levI[i]),int(levN[i])),-1)
                     for i in range(N_CANON)],dtype=np.int64)
    hit=rowmap>=0
    SIG=np.zeros((N_CANON,NFREQ),dtype='<f8'); SIG[hit]=_SIG_raw[rowmap[hit]]
    flags=np.zeros(N_CANON,dtype=np.int8);     flags[hit]=_flags_raw[rowmap[hit]]
    sys.stderr.write(f"[db_photoion] remapped SIGMA_BIN {SIGMA_BIN} ({NLEV} lev) -> "
                     f"{int(hit.sum())}/{N_CANON} canonical levels\n")
assert len(levZ)==SIG.shape[0]

def _row_slope(gl):
    r=np.asarray(SIG[gl]); nz=np.where(r>0)[0]
    if len(nz)<4: return None
    lx=np.log(nu_c[nz]); ly=np.log(r[nz])
    p=np.polyfit(lx,ly,1); pred=np.polyval(p,lx)
    ss=1-np.sum((ly-pred)**2)/max(np.sum((ly-ly.mean())**2),1e-300)
    return p[0],ss

def _is_kramers_row(gl):
    s=_row_slope(gl)
    return s is not None and abs(s[0]+3.0)<1e-3 and s[1]>0.9999

def _kramers_scan(sample=5):
    """Task-4 guard: warn if the WHOLE loaded grid is an all-Kramers nu^-3 fallback."""
    gs=np.where(flags==1)[0]
    if len(gs)==0: return
    pick=gs[np.linspace(0,len(gs)-1,min(sample,len(gs))).astype(int)]
    nk=sum(1 for gl in pick if _is_kramers_row(gl))
    if nk==len(pick):
        sys.stderr.write("\n*** SIGMA IS KRAMERS FALLBACK -- all %d sampled levels have "
                         "slope=-3.000 R2>0.9999. Data-sensitive conclusions INVALID. ***\n\n"%len(pick))

def ion_is_kramers(Z,ion):
    """Per-ion Kramers check (catches the mixed-bin case, e.g. Co III). None if no sigma."""
    idx=np.where((levZ==Z)&(levI==ion)&(flags==1))[0]
    if len(idx)==0: return None
    ck=idx if len(idx)<=8 else idx[np.linspace(0,len(idx)-1,8).astype(int)]
    return all(_is_kramers_row(gl) for gl in ck)

# ---- LUMINA_SIGMA_COIII_PATCH: overlay REAL Co III sigma (bake_coiii_real_sigma.py) ----
_COIII_PATCH=os.environ.get('LUMINA_SIGMA_COIII_PATCH','')
if _COIII_PATCH:
    _pz=np.load(_COIII_PATCH)
    _psig=_pz['sigma']; _pmatch=_pz['matched']
    if not (isinstance(SIG,np.ndarray) and SIG.flags.writeable):
        SIG=np.array(SIG)          # materialize a writable copy off the memmap
    flags=np.array(flags)
    _np=0
    for _k in range(_psig.shape[0]):
        if not _pmatch[_k]: continue
        _sel=np.where((levZ==27)&(levI==2)&(levN==_k))[0]
        if len(_sel): SIG[_sel[0]]=_psig[_k]; flags[_sel[0]]=1; _np+=1
    sys.stderr.write(f"[db_photoion] Co III REAL-sigma patch overlaid: {_np} levels "
                     f"from {_COIII_PATCH}\n")

_kramers_scan()

CHI={}
for r in csv.DictReader(open(f'{REF_DIR}/ionization_energies.csv')):
    CHI[(int(r['atomic_number']),int(r['ion_number']))]=float(r['ionization_energy_eV'])

def field(d,shell):
    Jrow=np.zeros(NFREQ)
    for r in csv.DictReader(open(f'{d}/lumina_coevolve_field.csv')):
        if int(r['shell'])!=shell: continue
        b=int(r['bin']); m=float(r['mc_J']); c=float(r['cs_J'])
        Jrow[b]=m if m>0 else c
    return Jrow

def plasma(d,shell):
    for r in csv.DictReader(open(f'{d}/lumina_plasma_state.csv')):
        if int(r['shell_id'])==shell:
            return float(r['T_e']),float(r['n_e'])

def levelpops(d,Z,ion,shell):
    out={}
    for r in csv.DictReader(open(f'{d}/lumina_levelpop.csv')):
        if int(r['Z'])==Z and int(r['ion'])==ion and int(r['shell'])==shell:
            out[int(r['level_num'])]=(float(r['n_k']),float(r['b_k']))
    return out

def R_of_level(gl, J, chi0_eV):
    """4pi sum sig J/(h nu) dnu above this level's threshold (code loop 5117-5144)"""
    E=levE[gl]; chi_l=(chi0_eV-E)*EV
    if chi_l<=0 or not flags[gl]: return 0.0,chi_l
    nu_th=chi_l/H
    m=(nu_c>=nu_th)&(SIG[gl]>0)&(J>0)
    if not m.any(): return 0.0,chi_l
    return float(np.sum(4*PI*SIG[gl][m]*J[m]/(H*nu_c[m])*dnu[m])),chi_l

def R_planck(gl, T, chi0_eV):
    E=levE[gl]; chi_l=(chi0_eV-E)*EV
    if chi_l<=0 or not flags[gl]: return 0.0,chi_l
    nu_th=chi_l/H
    x=H*nu_c/(KB*T)
    m=(nu_c>=nu_th)&(SIG[gl]>0)&(x<700)
    if not m.any(): return 0.0,chi_l
    B=2*H*nu_c[m]**3/C**2/np.expm1(x[m])
    return float(np.sum(4*PI*SIG[gl][m]*B/(H*nu_c[m])*dnu[m])),chi_l

def analyze(d,label,Z,ion,shell,selftest=False):
    Te,ne=plasma(d,shell)
    J=field(d,shell)
    chi0=CHI[(Z,ion)]
    kT=KB*Te
    idx=np.where((levZ==Z)&(levI==ion))[0]
    if ion_is_kramers(Z,ion):
        print(f"  *** WARNING Z={Z} ion={ion}: sigma_bf is KRAMERS FALLBACK (nu^-3, no CMFGEN "
              f"edge structure) -- this ion's photoion rate is DATA-INSENSITIVE ***")
    # lower-ion partition (code 5059-5063: ALL global levels, x<50)
    x=levE[idx]/(KB_EV*Te)
    U=float(np.sum(np.where(x<50,levG[idx]*np.exp(np.minimum(x,50)*-1),0)))
    # upper ion partition for Milne (code 2721-2736)
    idxu=np.where((levZ==Z)&(levI==ion+1))[0]
    xu=levE[idxu]/(KB_EV*Te)
    Uu=float(np.sum(np.where(xu<50,levG[idxu]*np.exp(-np.minimum(xu,50)),0)))
    if Uu<1: Uu=max(1.0,levG[idxu[0]] if len(idxu) else 1.0)
    lam3=(H*H/(2*PI*ME*KB*Te))**1.5
    pops=levelpops(d,Z,ion,shell)
    ntot=sum(v[0] for v in pops.values())
    G_gnd=G_b=G_n=alpha=0.0
    contrib=[]
    for gl in idx:
        Rb,chi_l=R_planck(gl,Te,chi0)
        if chi_l>0 and flags[gl]:
            alpha+=Rb*lam3*levG[gl]/(2*Uu)*math.exp(min(chi_l/kT,300))
        R,_=(Rb,chi_l) if selftest else R_of_level(gl,J,chi0)
        if R<=0: continue
        xl=levE[gl]/(KB_EV*Te)
        if xl>=50: continue
        pb=levG[gl]*math.exp(-xl)/U
        n=levN[gl]
        nk,bk=pops.get(n,(0.0,float('nan')))
        pn=nk/ntot if ntot>0 else 0.0
        if n==0: G_gnd+=R
        G_b+=pb*R; G_n+=pn*R
        contrib.append((n,levE[gl],bk,pb*R,pn*R))
    r_g=G_gnd/(ne*alpha); r_b=G_b/(ne*alpha); r_n=G_n/(ne*alpha)
    saha=2*Uu/U/lam3*math.exp(-chi0*EV/kT)/ne
    ff=lambda r: r/(1+r)
    print(f"### {label} Z={Z} {ion}->{ion+1} s{shell} Te={Te:.0f} ne={ne:.2e} {'[SELFTEST J=B]' if selftest else ''}")
    print(f"  G_all/G_gnd={G_b/G_gnd if G_gnd>0 else float('nan'):.3g}  G_nlte/G_boltz={G_n/G_b if G_b>0 else float('nan'):.3g}  U_low={U:.1f} U_up={Uu:.1f}")
    print(f"  r_gnd={r_g:.3e} f_up={ff(r_g):.4f} | r_boltz={r_b:.3e} f_up={ff(r_b):.4f} | r_nlte={r_n:.3e} f_up={ff(r_n):.4f} | saha_check r_b/saha={r_b/saha:.3f}")
    contrib.sort(key=lambda c:-c[4])
    print("  top NLTE-G lv(E,b_k):",", ".join(f"{c[0]}({c[1]:.1f},{c[2]:.2g})" for c in contrib[:6]))
    contrib.sort(key=lambda c:-c[3])
    print("  top Boltz-G lv(E,b_k):",", ".join(f"{c[0]}({c[1]:.1f},{c[2]:.2g})" for c in contrib[:6]))
    print()

def _main():
    A='logs/coevolve_consume_a10_kx_gphground'
    B='logs/coevolve_consume_a10_kx_gphall'
    analyze(A,'SELFTEST FeIII',26,2,0,selftest=True)
    analyze(A,'A FeIII',26,2,0)
    analyze(B,'B FeIII',26,2,0)
    analyze(A,'A CoIII',27,2,0)
    analyze(B,'B CoIII',27,2,0)
    analyze(A,'A SII',16,1,10)
    analyze(B,'B SII',16,1,10)
    analyze(A,'A SII',16,1,9)
    analyze(A,'A SiII',14,1,6)
    analyze(B,'B SiII',14,1,6)
    analyze(A,'A SiII s5',14,1,5)
    analyze(B,'B SiII s5',14,1,5)

if __name__=="__main__":
    _main()
