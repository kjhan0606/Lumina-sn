#!/usr/bin/env python3
"""f0a_emission_ledger.py -- OFFLINE F0a deep FUV emission/absorption ledger.

Reads the a10_kx EVENT_LOG (lumina_events.bin, LUMEVT01, 20-byte EventRec;
format from scripts/read_events.py, commit ac8ef44) and tallies per-process
emission / absorption by band (FUV 918-1290, EUV 300-450 A) and shell group
(deep s0-2 vs photosphere s6-10), plus the full per-shell net-source profile.

COVERAGE AUDIT (printed, NOT fabricated): the log is capped at CAP128M and
carries a SINGLE iteration (iter uniq). etypes 7(e-scatter) and 8(bf-reemit)
are absent -> the emission ledger = line-emit(2)+kpkt-ff(4)+kpkt-fb(5); direct
bf recombination re-emission is INVISIBLE. Event energies are packet energies
(not absolutely calibrated to CMFGEN) -> only RELATIVE strengths (deep vs
photosphere, band vs band) are claimable from this log; absolute "N-dex vs
CMFGEN" needs the F0b J/B calibration.

Usage: f0a_emission_ledger.py   (runs both gphall and jtable)
"""
import sys
import numpy as np

REPO='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
RUNS=[("B-run gphall",  f'{REPO}/logs/coevolve_consume_a10_kx_gphall'),
      ("jtable",        f'{REPO}/logs/coevolve_consume_a10_kx_jtable')]

EV=np.dtype([("pkt_id","<u4"),("line_id","<i4"),("nu_comov","<f4"),("energy","<f4"),
             ("etype","u1"),("shell","u1"),("iter","u1"),("pad","u1")])
C_A=2.99792458e18   # A/s
ETN={1:"line-abs",2:"line-emit",3:"bf-abs",4:"kpkt-ff",5:"kpkt-fb",6:"escape",7:"e-scat",8:"bf-reemit"}
EMIT_ET=[2,4,5]     # emission channels present
ABS_ET =[1,3]       # true absorption channels present (7 e-scatter absent)
BANDS=[("FUV_918_1290",918.,1290.),("EUV_300_450",300.,450.)]

def load(d):
    mm=np.memmap(f'{d}/lumina_events.bin',dtype=EV,mode='r',offset=32)
    return (np.array(mm['nu_comov']),np.array(mm['energy']),
            np.array(mm['etype']),np.array(mm['shell']),np.array(mm['iter']))

csv_rows=[]
for tag,d in RUNS:
    nu,en,et,sh,it=load(d)
    lam=np.where(nu>0,C_A/nu,0.0)
    n=len(et)
    print("\n"+"#"*100)
    print(f"# {tag}   ({d})")
    print("#"*100)
    # ---- coverage audit ----
    print("COVERAGE AUDIT:")
    print(f"  n_events={n:,}   (CAP128M={'HIT' if n==128_000_000 else 'not hit'})")
    print(f"  iter uniq={list(np.unique(it))}  -> single transport iteration" )
    ehist={int(e):int((et==e).sum()) for e in np.unique(et)}
    print("  etype histogram: "+", ".join(f"{ETN.get(e,e)}={c:,}" for e,c in sorted(ehist.items())))
    missing=[ETN[e] for e in (7,8) if e not in ehist]
    print(f"  NOT logged: {missing}  -> emission ledger excludes bf-reemit; e-scatter absent")
    print(f"  shell coverage: {int(sh.min())}..{int(sh.max())} ({len(np.unique(sh))} shells)")

    # ---- band x group ledger ----
    for bn,lo,hi in BANDS:
        binb=(lam>=lo)&(lam<hi)
        print(f"\n  ===== band {bn} ({lo:.0f}-{hi:.0f}A): {int(binb.sum()):,} events =====")
        for gname,g0,g1 in [("deep_s0-2",0,2),("phot_s6-10",6,10)]:
            gm=binb&(sh>=g0)&(sh<=g1)
            emitE=0.0; absE=0.0
            print(f"    [{gname}]")
            for e in EMIT_ET:
                m=gm&(et==e); c=int(m.sum()); E=float(en[m].sum()); emitE+=E
                if c: print(f"       EMIT {ETN[e]:>9}: n={c:>10,}  E={E:.4e}")
            for e in ABS_ET:
                m=gm&(et==e); c=int(m.sum()); E=float(en[m].sum()); absE+=E
                if c: print(f"       ABS  {ETN[e]:>9}: n={c:>10,}  E={E:.4e}")
            net=emitE-absE
            print(f"       => gross EMIT={emitE:.4e}  gross ABS={absE:.4e}  NET(emit-abs)={net:+.4e}")
            csv_rows.append([tag,bn,gname,emitE,absE,net])

    # ---- per-shell net FUV source profile (s0..s12) ----
    print(f"\n  ===== per-shell FUV(918-1290) profile =====")
    binf=(lam>=918)&(lam<1290)
    print(f"    {'sh':>3} {'emitE':>11} {'absE':>11} {'netE':>12} {'emit/deep0':>11}")
    e0=None
    for s in range(0,13):
        gm=binf&(sh==s)
        emitE=float(en[gm&np.isin(et,EMIT_ET)].sum())
        absE =float(en[gm&np.isin(et,ABS_ET)].sum())
        if e0 is None: e0=emitE if emitE>0 else 1.0
        print(f"    {s:>3} {emitE:>11.4e} {absE:>11.4e} {emitE-absE:>+12.4e} {emitE/e0:>11.2f}")
    # ---- upconversion / transport-from-below quantification (FUV) ----
    net_by_shell=np.array([float(en[binf&(sh==s)&np.isin(et,EMIT_ET)].sum())
                           -float(en[binf&(sh==s)&np.isin(et,ABS_ET)].sum()) for s in range(50)])
    deepnet=net_by_shell[0:6].sum()     # s0-5 : deep+intermediate feeding region
    photnet=net_by_shell[6:11].sum()    # s6-10: photosphere
    print(f"\n    FUV net source: cumulative s0-5={deepnet:+.4e}   photosphere s6-10={photnet:+.4e}")
    # sign-aware interpretation (a percentage is meaningless when the two have
    # opposite signs -- report sink/source directly instead).
    if deepnet <= 0 and photnet > 0:
        print(f"    -> deep+intermediate (s0-5) is a NET SINK; photosphere (s6-10) is the SOLE net")
        print(f"       FUV SOURCE. ALL of the net FUV is manufactured at the photosphere")
        print(f"       (local upconversion); the deep region does NOT feed FUV upward (it removes it).")
    else:
        tot=deepnet+photnet
        if tot!=0:
            print(f"    photospheric share of net FUV source = {photnet/tot*100:.1f}%")
    # ---- EUV deep emission (answer 3) ----
    bine=(lam>=300)&(lam<450)
    deuv_emit=float(en[bine&(sh<=2)&np.isin(et,EMIT_ET)].sum())
    deuv_n=int((bine&(sh<=2)&np.isin(et,EMIT_ET)).sum())
    deuv_fb=float(en[bine&(sh<=2)&(et==5)].sum()); deuv_fb_n=int((bine&(sh<=2)&(et==5)).sum())
    deuv_ln=float(en[bine&(sh<=2)&(et==2)].sum()); deuv_ln_n=int((bine&(sh<=2)&(et==2)).sum())
    print(f"\n    DEEP EUV(300-450) emission s0-2: n={deuv_n} E={deuv_emit:.4e}"
          f"  [line-emit n={deuv_ln_n} E={deuv_ln:.3e}; kpkt-fb(recomb) n={deuv_fb_n} E={deuv_fb:.3e}]")

# ---- CSV ----
import csv
outp=f'{REPO}/validation/cmfgen_toy06_19p48d/analysis/f0a_emission_ledger.csv'
with open(outp,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['run','band','group','gross_emit_E','gross_abs_E','net_E'])
    w.writerows(csv_rows)
print(f"\n[out] -> {outp}")
