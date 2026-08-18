#!/usr/bin/env python3
"""Ledger item 1: full ionization census, ALL elements, at photosphere s6/s7/s8.
CMFGEN(published StaNdaRT) vs Lumina B-run(gphall) vs Lumina kpr8.
Lumina shell s -> v_mid = 4264 + 728*s (s6=8632,s7=9360,s8=10088 km/s).
CMFGEN published ionfrac interpolated to those velocities.
Outputs: ion_census.csv (per element, per stage-fraction, per shell, all 3 sides).
"""
import os, re, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "..", "data", "standart_data1", "toy06")
BRUN = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx_gphall"
KPR8 = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx_kpr8"

SHELLS = {6: 8632, 7: 9360, 8: 10088}
# Z -> (element symbol, cmfgen ionfrac tag).  Roman-stage index in file: <tag>0=neutral.
ELEMS = [(26,'Fe','fe'),(27,'Co','co'),(28,'Ni','ni'),(14,'Si','si'),
         (16,'S','s'),(20,'Ca','ca'),(8,'O','o'),(6,'C','c')]
STAGE_ROMAN = ['I','II','III','IV','V','VI','VII']

def parse_cmfgen_ionfrac(fn, tgt=19.48):
    if not os.path.exists(fn): return None
    L = open(fn).read().splitlines()
    times, st = [], []
    for i,l in enumerate(L):
        m = re.match(r'#TIME:\s*([\d.]+)', l)
        if m: times.append(float(m.group(1))); st.append(i)
    if not times: return None
    k = int(np.argmin(abs(np.array(times)-tgt)))
    i0=st[k]; i1=st[k+1] if k+1<len(st) else len(L)
    cols=None; v=[]; d=[]
    for l in L[i0:i1]:
        if l.startswith('#vel_mid'):
            cols=l.strip().lstrip('#').split()[1:]; continue
        if l.startswith('#'): continue
        p=l.split()
        if len(p)>=2:
            try: vals=[float(x) for x in p]
            except ValueError: continue
            v.append(vals[0]); d.append(vals[1:])
    if not v: return None
    v=np.array(v); d=np.array(d)
    return v, {c:d[:,j] for j,c in enumerate(cols)}

def load_lumina_ion(rundir):
    """return dict[(shell,Z)] -> np.array over stages 0..5 of n_ion."""
    out={}
    fn=os.path.join(rundir,'lumina_ion_pops.csv')
    with open(fn) as f:
        next(f)
        for line in f:
            p=line.split(',')
            s=int(p[0]); Z=int(p[1]); stg=int(p[2]); n=float(p[3])
            if s not in SHELLS: continue
            out.setdefault((s,Z), np.zeros(7))
            if stg<7: out[(s,Z)][stg]=n
    return out

def main():
    lum_b = load_lumina_ion(BRUN)
    lum_k = load_lumina_ion(KPR8)
    rows=[]
    hdr=("elem,Z,shell,v_kms,stage,"
         "cmfgen_frac,Brun_frac,kpr8_frac")
    print(hdr)
    for Z,sym,tag in ELEMS:
        cf = parse_cmfgen_ionfrac(os.path.join(DATA,f"ionfrac_{tag}_toy06_cmfgen.txt"))
        for s,v in SHELLS.items():
            nb = lum_b.get((s,Z)); nk = lum_k.get((s,Z))
            fb = nb/nb.sum() if (nb is not None and nb.sum()>0) else None
            fk = nk/nk.sum() if (nk is not None and nk.sum()>0) else None
            # cmfgen stage fractions at this v
            fc=None
            if cf is not None:
                vv,dd=cf
                keys=sorted([c for c in dd if c.startswith(tag) and c[len(tag):].isdigit()],
                            key=lambda c:int(c[len(tag):]))
                if keys:
                    fc=np.array([np.interp(v,vv,dd[k]) for k in keys])
                    fc=fc/fc.sum() if fc.sum()>0 else fc
            nstage=max(len(fc) if fc is not None else 0,
                       len(fb) if fb is not None else 0,
                       len(fk) if fk is not None else 0, 6)
            for stg in range(min(nstage,7)):
                c = fc[stg] if (fc is not None and stg<len(fc)) else np.nan
                b = fb[stg] if (fb is not None and stg<len(fb)) else np.nan
                kk = fk[stg] if (fk is not None and stg<len(fk)) else np.nan
                if (np.nan_to_num(c)+np.nan_to_num(b)+np.nan_to_num(kk))<1e-6: continue
                rows.append((sym,Z,s,v,STAGE_ROMAN[stg],c,b,kk))
                print(f"{sym},{Z},s{s},{v},{STAGE_ROMAN[stg]},"
                      f"{c:.4e},{b:.4e},{kk:.4e}")
    with open(os.path.join(HERE,'ion_census.csv'),'w') as f:
        f.write("elem,Z,shell,v_kms,stage,cmfgen_frac,Brun_frac,kpr8_frac\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},s{r[2]},{r[3]},{r[4]},{r[5]:.4e},{r[6]:.4e},{r[7]:.4e}\n")

    # summary: dominant stage + mean-charge per element/shell
    print("\n=== dominant stage & mean charge (CMFGEN | Brun | kpr8) ===")
    print("elem shell    CMFGEN         Brun           kpr8      | zbar C/B/k")
    def zbar(f):
        if f is None: return np.nan
        return sum(i*f[i] for i in range(len(f)))
    def dom(f):
        if f is None: return '--'
        i=int(np.argmax(f)); return f"{STAGE_ROMAN[i]}({f[i]:.2f})"
    for Z,sym,tag in ELEMS:
        cf=parse_cmfgen_ionfrac(os.path.join(DATA,f"ionfrac_{tag}_toy06_cmfgen.txt"))
        for s,v in SHELLS.items():
            nb=lum_b.get((s,Z)); nk=lum_k.get((s,Z))
            fb=nb/nb.sum() if (nb is not None and nb.sum()>0) else None
            fk=nk/nk.sum() if (nk is not None and nk.sum()>0) else None
            fc=None
            if cf is not None:
                vv,dd=cf
                keys=sorted([c for c in dd if c.startswith(tag) and c[len(tag):].isdigit()],
                            key=lambda c:int(c[len(tag):]))
                fc=np.array([np.interp(v,vv,dd[k]) for k in keys]); fc/=fc.sum()
            print(f"{sym:3s} s{s}  {dom(fc):>13s} {dom(fb):>13s} {dom(fk):>13s} | "
                  f"{zbar(fc):.2f}/{zbar(fb):.2f}/{zbar(fk):.2f}")

if __name__=='__main__':
    main()
