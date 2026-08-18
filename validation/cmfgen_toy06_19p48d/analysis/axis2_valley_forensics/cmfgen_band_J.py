#!/usr/bin/env python3
"""Extract CMFGEN jnu4 band-averaged J_nu at Lumina forming-shell velocities for
ARBITRARY bands (reuses extract_jnu.py's validated EDDFACTOR reader).
Prints band-avg (geometric mean) J per shell for FUV(918-1290) and VALLEY(1650-2100).
Read-only."""
import numpy as np
CLIGHT_A=2997.92458
EDD="/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/EDDFACTOR"
RVTJ="/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ"
# TRUE Lumina field-CSV grid: v(s_n)=4264+728*n (confirmed gradient_budget_shells.csv:
# s0=4264, s2=5720). extract_jnu.py's old TARGET_V was mismatched for s1+.
TARGET_V=[4264+728*n for n in range(9)]  # s0..s8 = 4264,4992,5720,...,10088
LAB=['s0','s1','s2','s3','s4','s5','s6','s7','s8']
BANDS=[('FUV_918_1290',918.,1290.),('VALLEY_1650_2100',1650.,2100.),
       ('COMPLEX_1490_1650',1490.,1650.),('flank_1290_1490',1290.,1490.)]

def read_info(info):
    L=open(info).read().splitlines(); v=L[2].split()
    return dict(ND=int(v[0]),RECL=int(v[1]),WORD=int(v[2]),little=(v[5]=='T'))
def read_edd(edd):
    info=read_info(edd+'_INFO'); ND=info['ND']; nwr=info['RECL']//info['WORD']
    dt='<f8' if info['little'] else '>f8'
    raw=np.fromfile(edd,dtype=dt); n=(raw.size//nwr)*nwr; raw=raw[:n].reshape(-1,nwr)
    finish=raw[4,0]; data=raw[14:]
    good=np.isfinite(data[:,:ND]).all(axis=1)&(data[:,ND]>0)
    return data[good,:ND],data[good,ND],CLIGHT_A/data[good,ND],ND,finish
def parse_block(text,label,ND):
    lines=text.splitlines()
    for i,ln in enumerate(lines):
        if ln.strip()==label:
            vals=[];j=i+1
            while j<len(lines) and len(vals)<ND:
                toks=lines[j].split()
                try: vals+=[float(t) for t in toks]
                except ValueError: break
                j+=1
            return np.array(vals[:ND])
    raise KeyError(label)

J,FL,lam,ND,finish=read_edd(EDD)
print(f"[edd] ND={ND} nfreq={J.shape[0]} FINISH_REC={finish}")
V=parse_block(open(RVTJ).read(),'Velocity (km/s)',ND)
T=parse_block(open(RVTJ).read(),'Temperature (10^4K)',ND)*1e4
dv=np.argsort(V); Vasc=V[dv]
H=6.62607015e-27;KB=1.380649e-16;C=2.99792458e10;C_A=2.99792458e18
def Bnu(lamA,Te):
    nu=C_A/lamA; x=H*nu/(KB*Te)
    return (2*H*nu**3/C**2)/np.expm1(x) if x<700 else 0.0
# T_e at Lumina shells (from bsrc.n12 plasma_state) for B(T) comparison
Te_lum={0:14585,2:14991,7:11190,8:10811}  # rough; refined in report from plasma_state
print(f"\n{'band':20}{'shell':6}{'v':>7}{'CMFGEN J_bandavg':>18}")
res={}
for bn,lo,hi in BANDS:
    bm=(lam>=lo)&(lam<=hi); lam_b=lam[bm]; Jb=J[bm]
    ob=np.argsort(lam_b); lam_b=lam_b[ob]; Jb=Jb[ob]
    for lb,vt in zip(LAB,TARGET_V):
        Jvt=np.empty(lam_b.size)
        for k in range(lam_b.size):
            yk=Jb[k,dv]; yk=np.where(yk>0,yk,np.nan); ly=np.log10(yk); g=np.isfinite(ly)
            Jvt[k]=10**np.interp(vt,Vasc[g],ly[g])
        bavg=np.exp(np.nanmean(np.log(Jvt)))
        res[(bn,lb)]=bavg
        if lb in ('s0','s1','s2','s7','s8'):
            print(f"{bn:20}{lb:6}{vt:>7}{bavg:>18.4e}")
# valley B(13120) reference and 10.5x check
lamc=np.sqrt(1650*2100); b13=Bnu(lamc,13120.)
print(f"\nVALLEY band-center lam={lamc:.0f}A  B(13120K)={b13:.3e}")
for lb in ['s0','s1','s2']:
    jc=res[('VALLEY_1650_2100',lb)]
    print(f"  CMFGEN valley J({lb})={jc:.3e}  = {jc/b13:.2f} x B(13120)")
print(f"\n(For comparison the task states CMFGEN cs-side valley = 10.5x B(13120).)")
import csv
with open("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/axis2_valley_forensics/cmfgen_band_J.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(['band','shell','v_kms','cmfgen_J_bandavg'])
    for (bn,lb),v in res.items():
        vt=TARGET_V[LAB.index(lb)]; w.writerow([bn,lb,vt,f"{v:.6e}"])
print("[out] cmfgen_band_J.csv")
