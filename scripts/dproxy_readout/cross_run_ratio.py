import re,csv,math,statistics as st,collections,sys
h=6.62607015e-27; c=2.99792458e10
def load(H):
    E={}
    for r in csv.DictReader(open(H+"/input/model/levels.csv")): E[(int(r["atomic_number"]),int(r["ion_number"]),int(r["level_number"]))]=int(r["g"])
    rows={}; it=None
    for l in open(H+"/stderr.log"):
        m=re.match(r"\[R7\]\[PHASE\] lane=DET iter=(\d) phase=view",l)
        if m: it=int(m.group(1)); continue
        if l.startswith("[A2-10][LINE-SATURATION-ROW]") and it is not None:
            d=dict(t.split("=",1) for t in l.split() if "=" in t)
            if d.get("source_function_defined")=="1": rows[(it,d["line"])]=d
    low={0:collections.defaultdict(list),1:collections.defaultdict(list)}; up={0:{},1:{}}
    for (it,ln),r in rows.items():
        Z=int(r["Z"]); lo=int(r["lower_level"]); u=int(r["upper_level"]); nu=float(r["nu"]); S=float(r["source_function"]); nu_=float(r["n_upper"])
        low[it][(Z,lo)].append(nu_*(E[(Z,1,lo)]/E[(Z,1,u)])*(1.0+2*h*nu**3/(c*c*S))); up[it][(Z,u)]=nu_
    return {it:{k:st.median(v) for k,v in low[it].items()} for it in (0,1)}, up
base,bu=load(sys.argv[1]); arm,au=load(sys.argv[2])
for it in (0,1):
    print(f"== iter {it}: n(arm)/n(base) ==")
    for Z in (26,27,28):
        rl=[arm[it][k]/base[it][k] for k in base[it] if k[0]==Z and k in arm[it]]
        ru=[au[it][k]/bu[it][k] for k in bu[it] if k[0]==Z and k in au[it]]
        allr=sorted(rl+ru); g=arm[it].get((Z,0),float('nan'))/base[it].get((Z,0),float('nan'))
        print(f" Z={Z}: lower n={len(rl)} med={st.median(rl):.6f} spread={max(rl)-min(rl):.2e} | upper n={len(ru)} med={st.median(ru):.6f} spread={max(ru)-min(ru):.2e} | ground ratio={g:.6f} | all distinct(1e-9)={len(set(round(x,9) for x in allr))}")
