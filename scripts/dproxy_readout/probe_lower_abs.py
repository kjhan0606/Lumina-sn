import re,csv,math,statistics as st,collections,sys
H=sys.argv[1].rstrip('/')+'/'
h=6.62607015e-27; c=2.99792458e10
E={}
for r in csv.DictReader(open(H+"input/model/levels.csv")): E[(int(r["atomic_number"]),int(r["ion_number"]),int(r["level_number"]))]=(float(r["energy_eV"]),int(r["g"]))
rows={}; it=None
for l in open(H+"stderr.log"):
    m=re.match(r"\[R7\]\[PHASE\] lane=DET iter=(\d) phase=view",l)
    if m: it=int(m.group(1)); continue
    if l.startswith("[A2-10][LINE-SATURATION-ROW]") and it is not None:
        d=dict(t.split("=",1) for t in l.split() if "=" in t)
        if d.get("source_function_defined")=="1" and d["source_function"] not in ("UNDEFINED_CHI_ZERO","UNAVAILABLE"): rows[(it,d["line"])]=d
byl={0:collections.defaultdict(list),1:collections.defaultdict(list)}
for (it,ln),r in rows.items():
    Z=int(r["Z"]); lo=int(r["lower_level"]); up=int(r["upper_level"]); nu=float(r["nu"]); S=float(r["source_function"]); nu_=float(r["n_upper"])
    gl,gu=E[(Z,1,lo)][1],E[(Z,1,up)][1]
    byl[it][(Z,lo)].append(nu_*(gl/gu)*(1.0+2*h*nu**3/(c*c*S)))
    byl[it][(Z,'u',up)].append(nu_)
for Z in (26,27,28):
    for it in (0,1):
        lows={k:st.median(v) for k,v in byl[it].items() if k[0]==Z and k[1]!='u'}
        ups={k:st.median(v) for k,v in byl[it].items() if k[0]==Z and k[1]=='u'}
        g=lows.get((Z,0),float('nan'))
        # Boltzmann estimate of the full ion from ground: n_ion ≈ n_ground * U(T)/g0 with T=10020 K over ALL deck levels of this ion
        kT=8.617333262e-5*10020.0; g0=E[(Z,1,0)][1]
        U=sum(E[k][1]*math.exp(-E[k][0]/kT) for k in E if k[0]==Z and k[1]==1)
        print(f"Z={Z} iter={it}: n_ground={g:.4e} cm-3, sum(observed lower {len(lows)})={sum(lows.values()):.4e}, sum(observed upper {len(ups)})={sum(ups.values()):.4e}; LTE-extrapolated n_ion ≈ n_ground*U/g0 = {g*U/g0:.4e} (U={U:.2f}, g0={g0})")
