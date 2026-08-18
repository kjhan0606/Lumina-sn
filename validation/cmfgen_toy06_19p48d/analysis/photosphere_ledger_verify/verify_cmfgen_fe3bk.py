#!/usr/bin/env python3
"""Independent verification of CMFGEN Fe III departure coefficients b_k at the photosphere.
Parse FeIIIOUT (jnu4). Per-depth block: header line (8 tokens, field6=V km/s, field8=depth idx),
then 1500 departure coefficients (5 per line). Cross-check T against RVTJ."""
import numpy as np
JNU4="/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4"

def parse(fn):
    Vs=[]; Ts=[]; blocks=[]; cur=None
    for ln in open(fn):
        p=ln.split()
        ishdr = (len(p)==8 and p[7].isdigit())
        if ishdr:
            # verify preceding fields are floats
            try:
                fl=[float(x) for x in p[:7]]
            except ValueError:
                fl=None
            if fl is not None:
                if cur is not None: blocks.append(cur)
                cur=[]; Vs.append(fl[5]); Ts.append(fl[3]); continue
        if cur is not None:
            for t in p:
                try: cur.append(float(t))
                except ValueError: pass
    if cur is not None: blocks.append(cur)
    NLEV=min(len(b) for b in blocks)
    b=np.array([bl[:NLEV] for bl in blocks])
    return np.array(Vs), np.array(Ts), b

V,T,b = parse(JNU4+"/FeIIIOUT")
print(f"ND={len(V)}  NLEV={b.shape[1]}  V[0]={V[0]:.0f} V[-1]={V[-1]:.0f}  T4[0]={T[0]:.4f} T4[-1]={T[-1]:.4f}")
print(f"sanity DEEP (depth {len(V)}, V={V[-1]:.0f}): b_gnd={b[-1,0]:.3f} med={np.median(b[-1]):.3f} max={b[-1].max():.3f} min={b[-1].min():.3f}")
print(f"sanity OUTER(depth 1, V={V[0]:.0f}): b_gnd={b[0,0]:.3f} med={np.median(b[0]):.3f} max={b[0].max():.3f}")
print()
for label,vtarget in [("s6",8632),("s7",9360),("s8",10088)]:
    d=int(np.argmin(np.abs(V-vtarget)))
    bd=b[d]; bpos=bd[bd>0]
    print(f"{label} v_target={vtarget}: matched depth idx={d+1} V={V[d]:.0f} T4={T[d]:.4f}({T[d]*1e4:.0f}K)")
    print(f"   Fe III b_k: gnd={bd[0]:.4f}  min={bpos.min():.4f}  median={np.median(bpos):.4f}  "
          f"mean={bpos.mean():.4f}  max={bpos.max():.4f}  P90={np.percentile(bpos,90):.4f}  P99={np.percentile(bpos,99):.4f}")
    # near-threshold: last 50 levels (highest index ~ highest energy in CMFGEN ordering)
    print(f"   last-20 (near-threshold) b_k mean={np.mean(bd[-20:]):.4f}  range[{bd[-20:].min():.3f},{bd[-20:].max():.3f}]")
