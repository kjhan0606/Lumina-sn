# 재현: python3 repro.py rows.txt  — 초안 실측 3~6 전부 재계산
import re, math, statistics, sys
H,K,C=6.62607015e-27,1.380649e-16,2.99792458e10
def planck(nu,T): return 2*H*nu**3/C**2/math.expm1(H*nu/(K*T))
pat=re.compile(r'(\S+)=(\S+)'); rows=[dict(pat.findall(l)) for l in open(sys.argv[1])]
T_e,T_req=10020.0,19059.411196903675
pred,r_te,sp_b,ratio_nu=[],[],[],[]
for r in rows:
    b,omb=float(r['beta']),float(r['one_minus_beta'])
    jb,jc,sp,nu=float(r['Jbar']),float(r['J_cont']),float(r['S_probe']),float(r['nu'])
    pred.append(jb/(b*jc+omb*sp))
    r_te.append(jb/(b*jc+omb*planck(nu,T_e)))
    sp_b.append(sp/planck(nu,T_req))
    ratio_nu.append((nu,(jb-b*jc)/omb/sp))
for name,v in [("Jbar/pred(S_probe)",pred),("Jbar/pred(B_Te)",r_te),("S_probe/B(T_req)",sp_b)]:
    v=sorted(v);n=len(v);print(f"{name}: n={n} q10={v[int(.1*n)]:.4g} med={v[n//2]:.4g} q90={v[int(.9*n)]:.4g}")
xs=sorted(range(len(ratio_nu)),key=lambda i:ratio_nu[i][0]);ys=sorted(range(len(ratio_nu)),key=lambda i:ratio_nu[i][1])
rx={j:i for i,j in enumerate(xs)};ry={j:i for i,j in enumerate(ys)};n=len(ratio_nu)
print("rank corr(ratio,nu) =",1-6*sum((rx[i]-ry[i])**2 for i in range(n))/(n*(n*n-1)))
