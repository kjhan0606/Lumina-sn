import re,csv,math,statistics as st,collections,bisect
H="/gpfs/kjhan/lumina/superlevel_b_h200x2_k36/slb_20260903T080407Z_dproxy_provenance_k42_ion1_h200/"
rows={}; it=None
for l in open(H+"stderr.log"):
    m=re.match(r"\[R7\]\[PHASE\] lane=DET iter=(\d) phase=view",l)
    if m: it=int(m.group(1)); continue
    if l.startswith("[A2-10][LINE-SATURATION-ROW]") and it is not None:
        d=dict(t.split("=",1) for t in l.split() if "=" in t); rows[(it,d["line"])]=d
pairs=[k[1] for k in rows if k[0]==0 and (1,k[1]) in rows]
sp={0:{},1:{}}
for k in (0,1):
    for r in csv.DictReader(open(H+f"work/physics_comparison/physics_DET_iter000{k}.spectral.csv")):
        if r["shell_id"]=="0": sp[k][int(r["bin_id"])]=(float(r["nu_lo_Hz"]),float(r["nu_hi_Hz"]),float(r["J_nu"]))
los=sorted(sp[0]); lo_arr=[sp[0][b][0] for b in los]
def binJ(k,nu):
    i=bisect.bisect_right(lo_arr,nu)-1; b=los[i]
    return sp[k][b][2] if sp[k][b][0]<=nu<sp[k][b][1] else float("nan")
def midr(v):
    o=sorted(range(len(v)),key=v.__getitem__); r=[0]*len(v); s=0
    while s<len(o):
        e=s+1
        while e<len(o) and v[o[e]]==v[o[s]]: e+=1
        for p_ in range(s,e): r[o[p_]]=(s+1+e)/2
        s=e
    return r
def pear(a,b):
    ma,mb=st.mean(a),st.mean(b); ca=[x-ma for x in a]; cb=[x-mb for x in b]; d=math.sqrt(sum(x*x for x in ca)*sum(y*y for y in cb)); return sum(x*y for x,y in zip(ca,cb))/d if d else float("nan")
out=[]
for ln in pairs:
    r0,r1=rows[(0,ln)],rows[(1,ln)]; nu=float(r0["nu"])
    out.append(dict(Z=int(r0["Z"]),sl=r0["runtime_sl_id"],lower=int(r0["lower_level"]),beta=float(r0["beta"]),
        dlogJ=math.log(float(r1["Jbar"])/float(r0["Jbar"])),dlogS=math.log(float(r1["source_function"])/float(r0["source_function"])),
        dlogn=math.log(float(r1["n_upper"])/float(r0["n_upper"])),dlogJb=math.log(binJ(1,nu)/binJ(0,nu)),E=4.135667696e-15*nu,A=float(r0["A_ul"]),nu=nu))
print("pairs",len(out))
print(f"pair-level: rho(dlogJ,dlogS)={pear(midr([o['dlogJ'] for o in out]),midr([o['dlogS'] for o in out])):+.3f}  rho(dlogJ,dlogJb)={pear(midr([o['dlogJ'] for o in out]),midr([o['dlogJb'] for o in out])):+.3f}  rho(dlogn,dlogJb)={pear(midr([o['dlogn'] for o in out]),midr([o['dlogJb'] for o in out])):+.3f}")
v=sorted(o["dlogJb"] for o in out); print(f"dlogJ_bin at line nu: min={v[0]:+.4f} med={v[len(v)//2]:+.5f} max={v[-1]:+.4f}; |dlogJb|<1e-4: {sum(1 for x in v if abs(x)<1e-4)}/{len(v)}")
for lo_,hi_ in ((0,2),(2,4),(4,6),(6,12)):
    s=[o for o in out if lo_<=o["E"]<hi_]
    if s: print(f"  line E [{lo_}-{hi_}eV] n={len(s)} dlogJb med={st.median(o['dlogJb'] for o in s):+.5f} dlogJ med={st.median(o['dlogJ'] for o in s):+.4f} dlogn med={st.median(o['dlogn'] for o in s):+.4f}")
g=collections.defaultdict(list)
for o in out: g[(o["Z"],o["sl"])].append(o)
PX=[];PY=[]
for Z in (26,27,28):
    X=[];Y=[]
    for k,vv in g.items():
        if k[0]!=Z: continue
        ws=[o["A"]/o["nu"]**3 for o in vv]; X.append(sum(w*o["dlogJb"] for w,o in zip(ws,vv))/sum(ws)); Y.append(st.median(o["dlogn"] for o in vv))
    rx,ry=midr(X),midr(Y); mx,my=st.mean(rx),st.mean(ry); PX+=[a-mx for a in rx]; PY+=[b-my for b in ry]
    print(f" SL-level Z={Z} nSL={len(X)} rho(dlogJ_bin, dlogN_SL)={pear(rx,ry):+.3f}  X med={st.median(X):+.5f} distinctX={len(set(X))}")
print(f" pooled rho_SL(J_bin) = {pear(PX,PY):+.4f}")
gr=[o for o in out if o["lower"]==0]; print(f"ground-lower lines n={len(gr)}: E={min(o['E'] for o in gr):.2f}-{max(o['E'] for o in gr):.2f} eV, dlogJb med={st.median(o['dlogJb'] for o in gr):+.6f}, dlogn med={st.median(o['dlogn'] for o in gr):+.4f}")
