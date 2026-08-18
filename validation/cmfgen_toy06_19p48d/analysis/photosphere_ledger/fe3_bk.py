#!/usr/bin/env python3
"""Ledger item 6: Fe III departure coefficients b_k, CMFGEN vs Lumina(B-run,kpr8) at s6/s7/s8.
CMFGEN: FeIIIOUT (per-depth block of NLEV departure coeffs; header line = R,..,V,..,depth).
Lumina: lumina_levelpop.csv (Z=26, ion=2 == Fe III), column b_k.
Question: does CMFGEN show a near-threshold (Rydberg) b_k pileup like Lumina's kpr6 ~1e4?
"""
import os,sys,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
JNU4="/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4"
BRUN="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx_gphall"
KPR8="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx_kpr8"
SHELLS={6:8632,7:9360,8:10088}

def parse_feiiiout(fn):
    """return V[depth], b[depth, NLEV]."""
    toks=open(fn).read().split()
    # find header positions: a run of 7 floats + an integer depth index that increments 1..ND.
    # Simpler: scan for integer tokens equal to expected depth in context of 8-field header line.
    # Robust approach: re-read line-structured.
    lines=open(fn).read().splitlines()
    Vs=[]; blocks=[]; cur=None
    def is_header(p):
        if len(p)!=8: return False
        try:
            [float(x) for x in p[:7]]
            int(p[7]); return not ('E' in p[7] or 'e' in p[7] or '.' in p[7])
        except: return False
    for ln in lines:
        p=ln.split()
        if is_header(p):
            if cur is not None: blocks.append(cur)   # close previous depth block
            cur=[]; Vs.append(float(p[5]))           # 6th field = velocity; start new block
        elif cur is not None:                         # ignore preamble before first header
            for t in p:
                try: cur.append(float(t))
                except: pass
    if cur is not None: blocks.append(cur)
    NLEV=min(len(b) for b in blocks)
    b=np.array([bl[:NLEV] for bl in blocks])   # [ND,NLEV]
    return np.array(Vs), b

def lumina_bk(rundir):
    """dict[shell] -> (E_eV array, b_k array) for Fe III (Z=26, ion=2)."""
    out={s:{'E':[],'b':[]} for s in SHELLS}
    with open(os.path.join(rundir,'lumina_levelpop.csv')) as f:
        next(f)
        for line in f:
            p=line.split(',')
            s=int(p[0]); Z=int(p[1]); ion=int(p[2])
            if Z!=26 or ion!=2 or s not in SHELLS: continue
            out[s]['E'].append(float(p[4])); out[s]['b'].append(float(p[8]))
    for s in SHELLS:
        out[s]['E']=np.array(out[s]['E']); out[s]['b']=np.array(out[s]['b'])
    return out

def stats(b):
    b=b[np.isfinite(b)&(b>0)]
    if b.size==0: return dict(gnd=np.nan,med=np.nan,mx=np.nan,top=np.nan,n=0)
    return dict(gnd=b[0], med=np.median(b), mx=b.max(), top=np.mean(b[-20:]), n=b.size)

def main():
    Vc,bc=parse_feiiiout(JNU4+'/FeIIIOUT')
    print(f"[FeIIIOUT] ND={len(Vc)} NLEV={bc.shape[1]}  V {Vc[0]:.0f}..{Vc[-1]:.0f}")
    print(f"  deep(d90) b: gnd={bc[-1,0]:.3f} med={np.median(bc[-1]):.3f} max={bc[-1].max():.3f}")
    print(f"  outer(d1) b: gnd={bc[0,0]:.3f} med={np.median(bc[0]):.3f} max={bc[0].max():.3f}")
    LB=lumina_bk(BRUN); LK=lumina_bk(KPR8)
    lines=["shell,v_kms,side,b_ground,b_median,b_max,b_top20mean,nlevels"]
    print("\nshell side     b_gnd     b_med     b_max     b_top20   nlev")
    for s,v in SHELLS.items():
        # CMFGEN depth nearest v (V descending)
        d=int(np.argmin(np.abs(Vc-v)))
        sc=stats(bc[d]); sb=stats(LB[s]['b']); sk=stats(LK[s]['b'])
        for side,st in [("CMFGEN",sc),("Brun",sb),("kpr8",sk)]:
            lines.append(f"s{s},{v},{side},{st['gnd']:.4e},{st['med']:.4e},{st['mx']:.4e},{st['top']:.4e},{st['n']}")
            print(f"s{s} {side:7s} {st['gnd']:.3e} {st['med']:.3e} {st['mx']:.3e} {st['top']:.3e} {st['n']}")
    # near-threshold pileup: Lumina levels sorted by E_eV, show top-energy b_k
    print("\n=== Lumina Fe III highest-E levels (near-threshold) b_k, s6 ===")
    E=LB[6]['E']; b=LB[6]['b']; o=np.argsort(E)
    for i in o[-8:]:
        print(f"  E={E[i]:7.3f} eV  b_k(Brun)={b[i]:.3e}")
    with open(os.path.join(HERE,'fe3_bk.csv'),'w') as f:
        f.write("\n".join(lines)+"\n")
    print("\n[wrote] fe3_bk.csv")

if __name__=='__main__':
    main()
