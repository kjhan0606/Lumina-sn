import re,csv,math,statistics as st,collections
H="/gpfs/kjhan/lumina/superlevel_b_h200x2_k36/slb_20260903T080407Z_dproxy_provenance_k42_ion1_h200/"
h=6.62607015e-27; c=2.99792458e10
E={}
for r in csv.DictReader(open(H+"input/model/levels.csv")): E[(int(r["atomic_number"]),int(r["ion_number"]),int(r["level_number"]))]=(float(r["energy_eV"]),int(r["g"]),int(r["metastable"]))
rows={}; it=None
for l in open(H+"stderr.log"):
    m=re.match(r"\[R7\]\[PHASE\] lane=DET iter=(\d) phase=view",l)
    if m: it=int(m.group(1)); continue
    if l.startswith("[A2-10][LINE-SATURATION-ROW]") and it is not None:
        d=dict(t.split("=",1) for t in l.split() if "=" in t); rows[(it,d["line"])]=d
pairs=[k[1] for k in rows if k[0]==0 and (1,k[1]) in rows]
def nl(r):
    Z=int(r["Z"]); lo=int(r["lower_level"]); upl=int(r["upper_level"]); nu=float(r["nu"]); S=float(r["source_function"]); nu_=float(r["n_upper"])
    gl,gu=E[(Z,1,lo)][1],E[(Z,1,upl)][1]
    return nu_*(gl/gu)*(1.0+2*h*nu**3/(c*c*S)), 2*h*nu**3/(c*c*S)
out=[]
for ln in pairs:
    r0,r1=rows[(0,ln)],rows[(1,ln)]
    n0,w0=nl(r0); n1,w1=nl(r1)
    out.append(dict(Z=int(r0["Z"]),lower=int(r0["lower_level"]),upper=int(r0["upper_level"]),dlog_nl=math.log(n1/n0),dlog_nu=math.log(float(r1["n_upper"])/float(r0["n_upper"])),wien=w0,E_l=E[(int(r0["Z"]),1,int(r0["lower_level"]))][0],meta=E[(int(r0["Z"]),1,int(r0["lower_level"]))][2],beta=float(r0["beta"])))
print("pairs",len(out),"; Wien factor 2hnu^3/(c^2 S) med=%.2e (>>1 이면 n_l/n_u≈(g_l/g_u)·factor)"%st.median(o["wien"] for o in out))
# consistency: same lower level, spread of dlog_nl across lines
byl=collections.defaultdict(list)
for o in out: byl[(o["Z"],o["lower"])].append(o["dlog_nl"])
sp=[max(v)-min(v) for v in byl.values() if len(v)>=2]
print(f"lower levels={len(byl)}; with>=2 lines={len(sp)}; dlog_nl spread across lines sharing a lower: med={st.median(sp):.2e} max={max(sp):.2e}")
for Z in (26,27,28):
    L=[(k[1],st.median(v)) for k,v in byl.items() if k[0]==Z]; U=[o for o in out if o["Z"]==Z]
    vl=[x for _,x in L]
    print(f" Z={Z}: lower levels n={len(L)} dlog_n_lower med={st.median(vl):+.4f} [{min(vl):+.4f},{max(vl):+.4f}] | upper (pairs) dlog_n_upper med={st.median(o['dlog_nu'] for o in U):+.4f}")
    for lo_,hi_ in ((0,0.1),(0.1,1.5),(1.5,3),(3,4.5)):
        s=[st.median(v) for k,v in byl.items() if k[0]==Z and lo_<=E[(Z,1,k[1])][0]<hi_]
        if s: print(f"    E_lower [{lo_}-{hi_}eV] n={len(s)} dlog_n_lower med={st.median(s):+.4f} min={min(s):+.4f} max={max(s):+.4f}")
    g0=[st.median(v) for k,v in byl.items() if k[0]==Z and k[1]==0]
    if g0: print(f"    ground level 0: dlog_n = {g0[0]:+.5f}")
