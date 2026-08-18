#!/usr/bin/env python3
"""Side-by-side CMFGEN vs Lumina per-shell formation map + divergence matrix.

Rebins both CF(band, shell/depth) tables onto a common velocity axis and writes,
per band, the fraction of emergent flux formed in each velocity zone for CMFGEN vs
Lumina, and their difference (CMFGEN% - Lumina%). Both sides are CMF/comoving-frame.
"""
import numpy as np, csv, os
D=os.path.dirname(os.path.abspath(__file__))
BINS=[-1,3900,6000,8000,10000,12000,16000,22000,30000,1e12]
BLAB=["<3.9k(BC/deep)","3.9-6k","6-8k","8-10k","10-12k","12-16k","16-22k","22-30k",">30k"]
BAND_ORDER=["300-450","450-918","918-1290","1490-1650","1290-2000","2000-4500","4500-7000","7000+"]

def load(fn, vcol, fcol):
    rows={}
    with open(fn) as f:
        r=csv.DictReader(f)
        for d in r:
            b=d['band']
            v=d[vcol]
            v=-1.0 if (v.startswith('<') or v=='innerBC') else float(v)
            rows.setdefault(b,[]).append((v,float(d[fcol])))
    return rows

def vbin_frac(pairs):
    out=np.zeros(len(BLAB))
    for v,frac in pairs:
        vv = -1 if v<0 else v
        k=np.searchsorted(BINS,vv,side='right')-1
        k=min(max(k,0),len(BLAB)-1)
        out[k]+=frac
    return out

def main():
    cm=load(os.path.join(D,'cmfgen_CF_band_depth.csv'),'v_kms','CF_frac')
    lm=load(os.path.join(D,'lumina_CF_band_shell.csv'),'v_mid_kms','CF_frac')
    outf=open(os.path.join(D,'formation_divergence_matrix.csv'),'w')
    outf.write("band,vzone,CMFGEN_pct,Lumina_pct,diff_C_minus_L\n")
    print(f"{'band':>10s} {'vzone':>14s} {'CMFGEN%':>8s} {'Lumina%':>8s} {'C-L':>7s}")
    for b in BAND_ORDER:
        if b not in cm or b not in lm: continue
        c=vbin_frac(cm[b])*100; l=vbin_frac(lm[b])*100
        print(f"--- band {b} ---")
        for k in range(len(BLAB)):
            if c[k]<0.05 and l[k]<0.05:
                outf.write(f"{b},{BLAB[k]},{c[k]:.2f},{l[k]:.2f},{c[k]-l[k]:.2f}\n"); continue
            print(f"{'':>10s} {BLAB[k]:>14s} {c[k]:8.1f} {l[k]:8.1f} {c[k]-l[k]:7.1f}")
            outf.write(f"{b},{BLAB[k]},{c[k]:.2f},{l[k]:.2f},{c[k]-l[k]:.2f}\n")
    outf.close()

    # median formation velocity comparison
    cv={r['band']:r for r in csv.DictReader(open(os.path.join(D,'cmfgen_formation_velocity.csv')))}
    lv={r['band']:r for r in csv.DictReader(open(os.path.join(D,'lumina_formation_velocity.csv')))}
    print("\n=== median formation velocity: CMFGEN vs Lumina (km/s) ===")
    print(f"{'band':>10s} {'CMFGEN':>8s} {'Lumina':>8s} {'ratio L/C':>9s}")
    with open(os.path.join(D,'formation_velocity_compare.csv'),'w') as f:
        f.write("band,cmfgen_median_v,lumina_median_v,cmfgen_mean_v,lumina_mean_v\n")
        for b in BAND_ORDER:
            if b in cv and b in lv:
                cmd=float(cv[b]['median_v_kms']); lmd=float(lv[b]['median_v_kms'])
                f.write(f"{b},{cmd:.0f},{lmd:.0f},{cv[b]['mean_v_kms']},{lv[b]['mean_v_kms']}\n")
                rr = lmd/cmd if cmd>0 else float('nan')
                print(f"{b:>10s} {cmd:8.0f} {lmd:8.0f} {rr:9.2f}")
    print(f"\n[wrote] formation_divergence_matrix.csv, formation_velocity_compare.csv")

if __name__=='__main__': main()
