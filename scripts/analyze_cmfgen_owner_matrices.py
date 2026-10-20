#!/usr/bin/env python3
"""보존한5개 생산 행렬의 구조·수치 풀이를 원래 율 그대로 대조한다."""
import argparse
import csv
import ctypes as C
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svdvals
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from compare_cmfgen_bf_superlevel_rates import sha256_file


class GeneratorDiagnostic(C.Structure):
    _fields_=[("recognized",C.c_int),("column_error",C.c_double),
              ("residual",C.c_double),("minimum",C.c_double),("maximum",C.c_double)]


def solve_gth(lib,A,total):
    a=np.asfortranarray(A,dtype=np.float64);x=np.empty(len(A),dtype=np.float64)
    d=GeneratorDiagnostic()
    status=lib.population_generator_stationary_gth(
        a.ctypes.data_as(C.POINTER(C.c_double)),len(A),total,
        x.ctypes.data_as(C.POINTER(C.c_double)),C.byref(d))
    if status!=0:raise ValueError(f"GTH가 원래 generator를 거부: {status}")
    return x,d


def stage_sums(values,identity):
    result={}
    for i,row in enumerate(identity):
        stage=int(row["spectroscopic_stage"])
        result[stage]=result.get(stage,0.0)+float(values[i])
    return result


def multiprecision_stationary(A,precision):
    """같은 양의 off-diagonal 율로 정의된 generator를 독립 mp LU로 푼다.

    기존 GTH와 마찬가지로 대각은 양의 유출률 합의 음수다. 저장된 double
    대각의 반올림 오차를 정확한 물리 연산자로 간주하지 않는다. 이 함수의
    성공은 생산 raw rank/κ 판정을 변경하지 않는다.
    """
    import mpmath as mp
    A=np.asarray(A,dtype=float)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or not np.isfinite(A).all() or precision<80:
        raise ValueError("고정밀 generator 입력 불일치")
    n=len(A)
    off=A.copy();np.fill_diagonal(off,0)
    if np.any(off<0):raise ValueError("음의 전이율")
    with mp.workdps(precision):
        # mp.mpf(float)는 원래 binary64 값을 직접 가져온다. 십진 재적합 없음.
        matrix=mp.matrix(n,n)
        for j in range(n):
            values=[]
            for i in range(n):
                if i!=j:
                    matrix[i,j]=mp.mpf(float(off[i,j]));values.append(matrix[i,j])
            matrix[j,j]=-mp.fsum(values)
        constrained=matrix.copy()
        for j in range(n):constrained[0,j]=1
        rhs=mp.matrix(n,1);rhs[0]=1
        x=mp.lu_solve(constrained,rhs)
        worst=mp.mpf(0)
        for i in range(n):
            terms=[matrix[i,j]*x[j] for j in range(n)]
            denominator=mp.fsum(abs(v) for v in terms)
            residual=abs(mp.fsum(terms))
            if denominator:worst=max(worst,residual/denominator)
            elif residual:raise ValueError("유출입0인 행의 비영 잔차")
        return dict(precision=precision,population=[mp.nstr(v,precision) for v in x],
                    minimum=mp.nstr(min(x),precision),positive=all(v>0 for v in x),
                    componentwise_residual=mp.nstr(worst,12),
                    conservation_residual=mp.nstr(abs(mp.fsum(x)-1),12))


def multiprecision_case(root,input_path,Z,depth,precisions=(120,160)):
    import mpmath as mp
    from summarize_cmfgen_einstein_response import load_case,input_population
    identity,planes,candidate,files=load_case(root,depth,Z)
    pre,_=input_population(input_path,depth,Z,identity)
    total=float(pre.sum())
    A=sum(planes.values())
    checks=[multiprecision_stationary(A,p) for p in precisions]
    with mp.workdps(max(precisions)):
        vectors=[[mp.mpf(s) for s in check["population"]] for check in checks]
        difference=max(abs(a-b) for a,b in zip(*vectors))
        cases=[]
        for stage in (2,3):
            members=[i for i,row in enumerate(identity) if int(row["spectroscopic_stage"])==stage]
            reference=mp.fsum(vectors[-1][i] for i in members)
            ordinary=float(candidate[members].sum()/total)
            delta=abs(mp.mpf(ordinary)-reference)
            cases.append(dict(stage=stage,multiprecision_fraction=mp.nstr(reference,40),
                              C_GTH_fraction=ordinary,absolute_difference=float(delta),
                              relative_difference=float(delta/reference)))
    return dict(Z=Z,depth_1based=depth,N=len(A),independent_algorithm="mp LU with exact positive-rate generator",
                checks=checks,precision_change_max_abs_probability=mp.nstr(difference,12),stages=cases,
                production_adoption_changed=False,final_goal="NOT_JUDGED",
                evidence_sha256={str(p):sha256_file(p) for p in [input_path,*files.values()]})


def analyze(root,lib,Z,depth):
    stem=f"lumina_ew_iter{depth:04d}_ce00_z{Z:02d}_s{depth-1:03d}_"
    files={role:root/"matrices"/(stem+role+".csv") for role in
           ("matrix_raw","matrix_normalized","identity","solution","diagnostics","boundary_mass")}
    diagnostic={r["key"]:r["value"] for r in csv.DictReader(files["diagnostics"].open())}
    n=int(diagnostic["N"]);A=np.zeros((n,n));Anorm=np.zeros((n,n));b=np.zeros(n)
    for r in csv.DictReader(files["matrix_raw"].open()):
        A[int(r["row"]),int(r["column"])]+=float(r["value"])
    for r in csv.DictReader(files["matrix_normalized"].open()):
        i,j=int(r["row"]),int(r["column"])
        Anorm[i,j]=float(r["value"]);b[i]=float(r["rhs"])
    identity=list(csv.DictReader(files["identity"].open()))
    # The existing identity dump lists active SLs; the explicit scalar's
    # physical identity is independently recorded in boundary_mass.csv.
    if len(identity)==n-1:
        boundary=list(csv.DictReader(files["boundary_mass"].open()))
        if len(boundary)!=1 or int(boundary[0]["Z"])!=Z:
            raise ValueError("terminal identity 불완전")
        identity.append(dict(matrix_index=str(n-1),Z=str(Z),
            spectroscopic_stage=str(int(boundary[0]["terminal_charge"])+1)))
    if [int(r["matrix_index"]) for r in identity]!=list(range(n)):
        raise ValueError("identity 행렬 순서 불일치")
    total=b[0];old=np.empty(n)
    for r in csv.DictReader(files["solution"].open()):
        if r["record_type"] in ("SL","BOUNDARY_MASS"):
            old[int(r["matrix_index"])]=float(r["restored_population"])
    off=A.copy();np.fill_diagonal(off,0)
    if not np.isfinite(A).all() or np.any(off<0):raise ValueError("유한 비음수 generator가 아님")
    count,labels=connected_components(csr_matrix(off.T>0),directed=True,connection="strong")
    closed=np.ones(count,dtype=bool)
    destinations,sources=np.nonzero(off>0)
    for i,j in zip(destinations,sources):
        if labels[i]!=labels[j]:closed[labels[j]]=False
    closed_nodes=[[int(i) for i in np.flatnonzero(labels==label)]
                  for label in np.flatnonzero(closed)]
    x,gd=solve_gth(lib,A,total)
    old_stages=stage_sums(old,identity);new_stages=stage_sums(x,identity)
    max_perm=0.0
    for permutation in (np.arange(n)[::-1],np.random.default_rng(20261019).permutation(n)):
        xp,_=solve_gth(lib,A[np.ix_(permutation,permutation)],total)
        restored=np.empty(n);restored[permutation]=xp
        stages=stage_sums(restored,identity)
        max_perm=max(max_perm,max(abs(stages[k]-new_stages[k])/total for k in stages))
    # Exact nonsingular row/column coordinate changes only; no rate changes.
    # This is an offline diagnostic, never a route to production adoption.
    ld=np.longdouble
    weighted=Anorm.astype(ld)*(x/total).astype(ld)[None,:]
    scale=np.max(np.abs(weighted),axis=1)
    balanced=np.asarray(weighted/scale[:,None],dtype=float)
    singular=svdvals(balanced)
    row_norms=np.max(np.abs(A[1:]),axis=1)
    reduced=A[1:]/row_norms[:,None]
    reduced_singular=svdvals(reduced)
    return dict(Z=Z,depth_1based=depth,N=n,positive_rate_edges=int(len(sources)),
        strong_components=count,closed_class_count=int(np.sum(closed)),closed_classes=closed_nodes,
        graph_generator_rank=n-int(np.sum(closed)),old_rank=int(diagnostic["raw_rank"]),
        old_kappa=float(diagnostic["kappa_2"]),
        old_permutation_fraction_max=float(diagnostic["permutation_ion_fraction_max_abs"]),
        old_pivot_growth=float(diagnostic["pivot_growth"]),
        gth_recognized=bool(gd.recognized),gth_column_error=gd.column_error,
        gth_exact_generator_residual=gd.residual,
        gth_permutation_fraction_max=max_perm,
        old_vs_gth_max_ion_fraction_difference=max(abs(old_stages[k]-new_stages[k])/total for k in old_stages),
        old_ion_fractions={str(k):v/total for k,v in old_stages.items()},
        gth_ion_fractions={str(k):v/total for k,v in new_stages.items()},
        stationary_coordinate_kappa=float(singular[0]/singular[-1]),
        row_scaled_SE_singular_min_over_max=float(reduced_singular[-1]/reduced_singular[0]),
        input_sha256={str(p):sha256_file(p) for p in files.values()})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True);ap.add_argument("--library",type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();out=root/"matrix_analysis.json"
    if out.exists():raise ValueError("기존 분석 보존")
    lib=C.CDLL(str(a.library.resolve()))
    lib.population_generator_stationary_gth.argtypes=[C.POINTER(C.c_double),C.c_size_t,C.c_double,
                                                    C.POINTER(C.c_double),C.POINTER(GeneratorDiagnostic)]
    lib.population_generator_stationary_gth.restype=C.c_int
    result=dict(scope="fixed stored production matrices; no rates or criteria changed",
                final_goal="NOT_JUDGED",library_sha256=sha256_file(a.library),
                analyzer_sha256=sha256_file(Path(__file__)),
                cases=[analyze(root,lib,z,d) for z,d in ((14,1),(14,61),(16,1),(16,19),(16,20))])
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for row in result["cases"]:
        print(json.dumps({k:v for k,v in row.items() if k not in
                         ("input_sha256","closed_classes","old_ion_fractions","gth_ion_fractions")}))


if __name__=="__main__":main()
