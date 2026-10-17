"""실제 작은 형식해의 AA(1)과 다중 이력 후보 비교. 생산 알고리즘 변경 없음."""
import ctypes as c
import json
import time
import numpy as np

lib=c.CDLL('./probe.so')
ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
lib.acceleration_apply.argtypes=[c.c_int,ptr,ptr]
lib.acceleration_reference.argtypes=[c.c_int,ptr,ptr]
lib.acceleration_solve.argtypes=[c.c_int,c.c_int,c.c_double,ptr,ptr]
lib.acceleration_old_next.argtypes=[ptr,ptr,c.c_double]
lib.acceleration_old_next.restype=None
lib.acceleration_old_reset.restype=None
lib.acceleration_old_count.argtypes=[c.c_int]
lib.acceleration_old_count.restype=c.c_size_t
n=288

def solve(kind,depth):
    x=np.full(n,0. if kind==3 else .8e-7)
    history=[];proposed=False;previous_norm=np.inf;previous_g=None
    rejected=0;accepted=0;rank_losses=0
    lib.acceleration_old_reset()
    for it in range(8192):
        g=np.empty(n)
        assert lib.acceleration_apply(kind,x,g)==0
        assert np.all(g>=0) and np.all(np.isfinite(g))
        difference=abs(g-x)
        rel=np.zeros(n)
        np.divide(difference,abs(x),out=rel,where=x!=0)
        rel[(x==0)&(difference>0)]=np.inf
        norm=float(rel.max())
        if it>0 and norm<1e-10:
            if depth==1:
                accepted=int(lib.acceleration_old_count(0))
                rejected=int(lib.acceleration_old_count(1))
                rank_losses=None
            return g,dict(iterations=it+1,residual=norm,accepted=accepted,
                          rejected=rejected,rank_losses=rank_losses)
        if depth==1:
            lib.acceleration_old_next(x,g,norm)
            continue
        if proposed and (not np.isfinite(norm) or norm>previous_norm):
            x=previous_g.copy();history=[];proposed=False;rejected+=1
            continue
        r=g-x
        candidate=g.copy();use=False
        if history:
            scale=np.maximum(abs(x),abs(g))
            for hg,hr in history:
                scale=np.maximum(scale,np.maximum(abs(hg),abs(hr)))
            active=scale>0
            dr=np.column_stack([(r-hr)[active]/scale[active] for hg,hr in history])
            rhs=r[active]/scale[active]
            coef,_,rank,_=np.linalg.lstsq(dr,rhs,rcond=None)
            rank_losses+=int(rank<len(history))
            if rank>0 and np.all(np.isfinite(coef)):
                candidate=g-sum(v*(g-hg) for v,(hg,hr) in zip(coef,history))
                use=np.all(np.isfinite(candidate)) and np.all(candidate>=0)
            if not use:
                candidate=g.copy();rejected+=1
        history.append((g.copy(),r.copy()))
        history=history[-depth:]
        x=candidate;previous_g=g.copy();previous_norm=norm;proposed=bool(use)
        accepted+=int(use)
    raise RuntimeError((kind,depth,'not converged'))

results=[]
for kind in range(6):
    reference=np.empty(n);bound=np.empty(n)
    ref_iterations=lib.acceleration_reference(kind,reference,bound)
    assert ref_iterations>0 and np.all(bound>=0)
    for depth in (1,2,4,8):
        begin=time.monotonic();x,report=solve(kind,depth)
        scale=np.maximum(abs(reference),abs(x))
        diff=abs(x-reference);active=scale>0
        error=float(np.max(diff[active]/scale[active])) if np.any(active) else 0.
        assert error<1e-7,(kind,depth,error)
        # 동일 수렴 기준과 실제 G(x)-x로 판정한다. bound는 참조해의 상계다.
        results.append(dict(kind=kind,depth=depth,reference_iterations=ref_iterations,
                            relative_to_reference=error,seconds=time.monotonic()-begin,**report))
        print(json.dumps(results[-1]),flush=True)
c_results=[]
for kind in range(6):
    reference=np.empty(n);reference_bound=np.empty(n)
    assert lib.acceleration_reference(kind,reference,reference_bound)>0
    iterations={}
    for depth in (1,2,4):
        x=np.empty(n);bound=np.empty(n)
        count=lib.acceleration_solve(kind,depth,1e-10,x,bound)
        assert count>0 and np.all(x>=0) and np.all(np.isfinite(x))
        difference=abs(x-reference)
        assert np.all(difference<=bound+reference_bound)
        scale=np.maximum(abs(x),abs(reference));active=scale>0
        error=float(np.max(difference[active]/scale[active])) if np.any(active) else 0.
        assert error<1e-7,(kind,depth,error)
        iterations[depth]=count
        row=dict(implementation='C shared solver',kind=kind,depth=depth,
                 iterations=count,relative_to_reference=error,bound_verified=True)
        c_results.append(row);print(json.dumps(row),flush=True)
    if kind in (0,1,2):
        assert iterations[4]<iterations[1],(kind,iterations)
with open('probe.json','x') as f:
    json.dump(dict(scope='small actual CPU formal operator; NOT full model performance or coupled convergence',
                   results=results,C_results=c_results),f,indent=2)
