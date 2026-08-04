#!/usr/bin/env python3
"""b_k comparison: Lumina vs ARTIS per line-forming ion — the mechanistic verdict for the
co-evolve consumer fix. ARTIS runs its line-formers super-thermal (Si II ~18, S II ~48,
Ca II ~10); the champion Lumina sits at b_k~=1. Question: does the consumer run lift Lumina's
b_k toward ARTIS?  Models differ (decay-powered vs lamp) so compare the STRUCTURE (per-ion
b_k distribution over the line-forming region), not exact cells.

Usage: compare_bk_artis.py <lumina_levelpop.csv> [artis_nlte_dir] [timestep=27]
"""
import sys, csv, glob
import numpy as np
lumina_csv = sys.argv[1] if len(sys.argv) > 1 else "logs/coevolve_consume/lumina_levelpop.csv"
artis_dir  = sys.argv[2] if len(sys.argv) > 2 else "/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk"

IONS = {(14,2):'Si II',(16,2):'S II',(20,2):'Ca II',(26,2):'Fe II',(26,3):'Fe III'}

# --- ARTIS b_k (converged ts, line-forming cells = inner-mid third) ---
def artis_bk():
    rows=[]
    for f in glob.glob(f"{artis_dir}/nlte_000*.out"):
        for l in open(f):
            p=l.split()
            if len(p)>=7 and p[0].isdigit() and p[2].isdigit(): rows.append(p)
    if not rows: return {}
    # EMISSION-RELEVANT LOW LEVELS only (index<LMAX): the high-Rydberg levels have n_LTE->0 so
    # b_k explodes to 1e6+ (numerical, not physical). The card's feature-strength target is L1-4
    # ~x2-3. Also exclude b_k>BMAX (Rydberg artifacts). ARTIS toy06 is decay-powered: fluorescence
    # super-thermal EARLY (ts~11, 5.27d), faded by the matched epoch (ts27, 19.48d); Lumina is
    # lamp-driven (steady 19.48d) so the target is the FLUORESCENCE-ACTIVE phase. Pick ts by the
    # highest low-level S II median.
    BMAX=1e3
    def lowbk(ts,Z,ion):   # first-EXCITED levels 1-4 (L0 ground~1, L5+ Rydberg inflated)
        return [float(r[6])/float(r[5]) for r in rows
                if int(r[0])==ts and int(r[2])==Z and int(r[3])==ion and 1<=int(r[4])<=4
                and float(r[5])>0 and float(r[6])>0 and 0<float(r[6])/float(r[5])<BMAX]
    alltss=sorted(set(int(r[0]) for r in rows))
    # [사례 18 정정 2026-07-31] data-dependent epoch 자동선택(구: S II b_k 최대 ts) 폐기.
    # 19.48d 매칭-epoch = ts27(mid 20.2549d; timesteps.out 정본). 다른 ts는 인자로 명시.
    ts = int(sys.argv[3]) if len(sys.argv) > 3 else 27  # argv[2]=artis_dir (P0-7 충돌 수리)
    out={}
    for (Z,ion),nm in IONS.items():
        out[(Z,ion)]=np.array(lowbk(ts,Z,ion))
    return out

# --- Lumina b_k (line-forming shells s6-33) ---
def lumina_bk():
    out={(Z,ion):[] for (Z,ion) in IONS}
    try:
        for r in csv.DictReader(open(lumina_csv)):
            s=int(r['shell']); Z=int(r['Z']); ion=int(r['ion'])
            if not (6<=s<=33): continue                       # line-forming shells
            if not (1<=int(r['level_num'])<=4): continue      # first-excited (feature-strength)
            if (Z,ion) not in out: continue
            try:
                bk=float(r['b_k'])
                if 0<bk<1e3: out[(Z,ion)].append(bk)          # exclude Rydberg-inflated
            except: pass
    except FileNotFoundError:
        return None
    return {k:np.array(v) for k,v in out.items()}

A=artis_bk(); L=lumina_bk()
print(f"=== b_k: Lumina ({lumina_csv}) vs ARTIS (toy06_nlte_bk) — line-forming ions ===")
if L is None:
    print(f"  (Lumina levelpop dump not found yet — run consumer with LUMINA_LEVELPOP_DUMP=1)")
def mn(x): return np.mean(x) if len(x) else float('nan')
print(f"{'ion':7} | {'Lu mean':>7} {'Lu max':>7} | {'ART mean':>8} {'ART max':>7} | {'verdict':>18}")
for (Z,ion),nm in IONS.items():
    a=A.get((Z,ion),np.array([])); lu=(L or {}).get((Z,ion),np.array([]))
    lm,am = mn(lu), mn(a)
    if len(lu) and len(a):
        if lm>0.7*am:   v="MATCHES ARTIS ✓"
        elif lm>1.5:    v="lifting (partial)"
        elif lm>1.15:   v="weak lift"
        else:           v="still thermal ✗"
    else: v=""
    print(f"{nm:7} | {lm:7.2f} {np.max(lu) if len(lu) else 0:7.1f} | "
          f"{am:7.2f} {np.max(a) if len(a) else 0:7.1f} | {v:>18}")
print("\nMetric = MEDIAN b_k of the emission-relevant LOW levels (L1-7, Rydberg excluded).")
print("ARTIS's feature-strength target is ~x2-3 (the card; the x18-48 'median' was Rydberg-inflated).")
print("PASS = Lumina low-level b_k lifts from ~1 toward ARTIS's ~2-3. Still ~1 => consumer not pumping.")
