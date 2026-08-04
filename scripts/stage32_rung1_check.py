#!/usr/bin/env python3
"""Fail-closed reader and independent Rung-1 branch/energy judge."""
from __future__ import annotations
import argparse, hashlib, json, re, struct
from pathlib import Path
import numpy as np

MAGIC=b"LCMFR101";ENDIAN=0x01020304;VERSION=3;C=2.99792458e10
HEADER=struct.Struct("<8sIIQQQIIIIQdddII")
ROW=np.dtype([("line","<u4"),("shell","<u4"),("bin","<u4"),("primary_status","<u4"),
              ("lambda_A","<f8"),("tau","<f8"),("beta","<f8"),
              ("eps0_raw","<f8"),("eps_prime","<f8"),("eps_applied","<f8"),
              ("chi_es","<f8"),("chi_tot","<f8"),("lambda_star","<f8"),
              ("rho_local","<f8"),
              ("source_S","<f8"),("eta_energy","<f8"),
              ("branch_evidence","<u4"),("disposition","<u4")])
NAMES=("legacy_source","thick_exempt","rate_shape_replaced","scalar_rescaled")
EV_REACHED=1;EV_ELIGIBLE=2;EV_THICK=4;EV_EPAY2=8
EV_ACCW=16;EV_HOT=32;EV_BRANCH=64;EV_ASSEMBLED=128
class R1Error(ValueError): pass

def read_check(path: Path, *, expected_iteration: int,
               expected_field_generation: int,
               expect_unscaled_energy: bool=False):
    path=Path(path);raw=path.read_bytes()
    if len(raw)<HEADER.size: raise R1Error("truncated header")
    vals=HEADER.unpack_from(raw)
    (magic,endian,version,it,gen,lambda_gen,ns,nb,nl,reserved,nrows,llo,lhi,texp,
     rowbytes,flags)=vals
    if (magic,endian,version)!=(MAGIC,ENDIAN,VERSION): raise R1Error("schema identity mismatch")
    if reserved or flags or rowbytes!=ROW.itemsize or len(raw)!=HEADER.size+nrows*rowbytes:
        raise R1Error("header/length mismatch")
    match=re.search(r"\.iter([0-9]{3,})$",path.name)
    if not match or int(match.group(1))!=it: raise R1Error("filename/header generation mismatch")
    if it!=expected_iteration or gen!=expected_field_generation:
        raise R1Error("iteration/field_generation expectation mismatch")
    if lambda_gen!=gen:
        raise R1Error(f"chi/lambda generation mismatch: assembly={gen} lambda={lambda_gen}")
    if not (gen>0 and ns>0 and nb>0 and nl>0 and (llo,lhi)==(600.0,3000.0)
            and np.isfinite(texp) and texp>0):
        raise R1Error("invalid dimensions/window/lineage")
    rows=np.frombuffer(raw,dtype=ROW,count=nrows,offset=HEADER.size).copy()
    finite=("lambda_A","tau","beta","eps0_raw","eps_prime","eps_applied",
            "chi_es","chi_tot","lambda_star","source_S","eta_energy")
    if any(not np.isfinite(rows[n]).all() for n in finite): raise R1Error("nonfinite row")
    if (np.any(rows["tau"]<=0) or np.any(rows["beta"]<=0) or np.any(rows["beta"]>1)
        or np.any(rows["lambda_star"]<0) or np.any(rows["lambda_star"]>1)
        or np.any(rows["eps0_raw"]<0) or np.any(rows["eps0_raw"]>1)
        or np.any(rows["eps_prime"]<0) or np.any(rows["eps_prime"]>1)
        or np.any(rows["chi_es"]<0) or np.any(rows["chi_tot"]<0)
        or np.any(rows["source_S"]<0) or np.any(rows["eta_energy"]<0)
        or np.any(rows["primary_status"]>1)
        or np.any(rows["disposition"]>3) or np.any(rows["branch_evidence"]>255)):
        raise R1Error("row outside exact mathematical domain")
    beta_ref=-np.expm1(-rows["tau"])/rows["tau"]
    if not np.allclose(rows["beta"],beta_ref,rtol=1e-12,atol=0):
        raise R1Error("KA-3.2.3 FAIL: Sobolev beta differs from analytic oracle")
    eps_prime_ref=rows["eps0_raw"]/(rows["eps0_raw"]+
                      (1.0-rows["eps0_raw"])*rows["beta"])
    if not np.allclose(rows["eps_prime"],eps_prime_ref,rtol=2e-14,atol=0):
        raise R1Error("secondary eps_prime identity failure")
    defined=rows["primary_status"]==0;undefined=~defined
    if np.any(rows["chi_tot"][defined]==0) or np.any(~np.isfinite(rows["rho_local"][defined])):
        raise R1Error("defined primary row has zero chi_tot or nonfinite rho")
    if np.any(rows["chi_tot"][undefined]!=0) or np.any(~np.isnan(rows["rho_local"][undefined])):
        raise R1Error("chi_tot==0 was not recorded as undefined")
    rho_ref=(rows["chi_es"][defined]/rows["chi_tot"][defined])*rows["lambda_star"][defined]
    if not np.array_equal(rows["rho_local"][defined],rho_ref):
        raise R1Error("rho_local production-array identity failure")
    if np.any(rows["rho_local"][defined]<0):
        raise R1Error("negative local spectral radius")

    # This independent formula is valid only for an externally asserted
    # eps_phys=0 run.  For eps_phys=1 the checker deliberately does not copy
    # production's epsilon branch; the authoritative cell census below judges
    # whether both owners received the exact eta_l assembled by production.
    if expect_unscaled_energy:
        frac=np.where(rows["tau"]>1e-6,-np.expm1(-rows["tau"]),rows["tau"])
        line_nu=C/(rows["lambda_A"]*1e-8)
        assembled=(rows["branch_evidence"]&EV_ASSEMBLED)!=0
        eta_ref=frac*line_nu/(C*texp)*rows["source_S"]
        eta_ref=np.where(assembled,eta_ref,0.0)
        if not np.allclose(rows["eta_energy"],eta_ref,rtol=8e-15,atol=0):
            raise R1Error("line energy FAIL: not production w_l*S_l*dnu for eps_phys=0")

    # The disposition is checked from separately recorded branch-site facts.
    # In particular rate_shape requires acc_w>0; a reconstructed predicate
    # that omits EV_ACCW cannot pass this oracle.
    ev=rows["branch_evidence"].astype(np.uint32)
    if np.any((ev&EV_REACHED)==0) or np.any((ev&EV_BRANCH)==0):
        raise R1Error("hard gate: missing branch-site evidence")
    expected=np.zeros(nrows,dtype=np.uint32)
    eligible=(ev&EV_ELIGIBLE)!=0;thick=(ev&EV_THICK)!=0
    rate=((ev&EV_EPAY2)!=0)&((ev&EV_ACCW)!=0)&((ev&EV_HOT)!=0)
    expected[eligible&thick]=1
    expected[eligible&~thick&rate]=2
    expected[eligible&~thick&~rate]=3
    if not np.array_equal(rows["disposition"],expected):
        raise R1Error("hard gate FAIL: branch-site disposition mismatch (including acc_w)")

    # KA-3.2.3 closed-form S,Jbar, independently using beta_ref.
    jext=0.37;B=1.41;eps=rows["eps0_raw"]
    D=eps+rows["beta"]-eps*rows["beta"];Dr=eps+beta_ref-eps*beta_ref
    S=((1-eps)*rows["beta"]*jext+eps*B)/D
    Sr=((1-eps)*beta_ref*jext+eps*B)/Dr
    J=(1-rows["beta"])*S+rows["beta"]*jext
    Jr=(1-beta_ref)*Sr+beta_ref*jext
    scaleS=np.maximum(np.abs(Sr),np.finfo(float).tiny)
    scaleJ=np.maximum(np.abs(Jr),np.finfo(float).tiny)
    if np.max(np.abs(S-Sr)/scaleS)>1e-12 or np.max(np.abs(J-Jr)/scaleJ)>1e-12:
        raise R1Error("KA-3.2.3 FAIL: analytic S/Jbar relative error > 1e-12")

    mf=json.loads(Path(str(path)+".manifest.json").read_text())
    if mf.get("schema")!="LCMFR101-v3" or mf.get("sha256")!=hashlib.sha256(raw).hexdigest():
        raise R1Error("manifest schema/SHA mismatch")
    if (mf.get("iteration")!=it or mf.get("field_generation")!=gen or
        mf.get("lambda_generation")!=lambda_gen or mf.get("rows")!=nrows):
        raise R1Error("manifest generation/count mismatch")
    eps_floor=float(mf.get("eps_floor",np.nan));eps_cap=float(mf.get("eps_cap",np.nan))
    if not np.isfinite(eps_floor) or not np.isfinite(eps_cap):
        raise R1Error("secondary epsilon limits missing/nonfinite")
    eps_applied_ref=np.where(rows["eps_prime"]<eps_floor,eps_floor,rows["eps_prime"])
    eps_applied_ref=np.where(eps_applied_ref>eps_cap,eps_cap,eps_applied_ref)
    if not np.array_equal(rows["eps_applied"],eps_applied_ref):
        raise R1Error("secondary eps_applied identity failure")
    eps_diff=int(np.count_nonzero(rows["eps_applied"]!=rows["eps_prime"]))
    rho_undefined=int(np.count_nonzero(undefined))
    if (mf.get("eps_applied_diff_rows")!=eps_diff or
        mf.get("rho_undefined_chi_tot_zero_rows")!=rho_undefined):
        raise R1Error("secondary clamp/primary undefined census mismatch")
    counts={name:int(np.count_nonzero(rows["disposition"]==i)) for i,name in enumerate(NAMES)}
    energies={name:float(rows["eta_energy"][rows["disposition"]==i].sum(dtype=np.longdouble))
              for i,name in enumerate(NAMES)}
    if mf.get("disposition_row_counts")!=counts: raise R1Error("disposition count census mismatch")
    me=mf.get("disposition_energy",{})
    for name in NAMES:
        if not np.isclose(me.get(name,np.nan),energies[name],rtol=2e-15,atol=0):
            raise R1Error(f"disposition energy census mismatch: {name}")
    selected=float(rows["eta_energy"].sum(dtype=np.longdouble))
    authoritative=float(mf.get("authoritative_pre_epay_window_energy",np.nan))
    boundary=float(mf.get("boundary_nonselected_line_energy",np.nan))
    stated_selected=float(mf.get("selected_row_energy",np.nan))
    residual=float(mf.get("closure_residual",np.nan))
    if not all(np.isfinite(x) and x>=0 for x in (authoritative,boundary,stated_selected)):
        raise R1Error("independent energy census missing/nonfinite")
    if not np.isclose(stated_selected,selected,rtol=2e-15,atol=0):
        raise R1Error("selected-row energy census mismatch")
    if not np.isclose(residual,authoritative-selected-boundary,rtol=0,
                      atol=32*np.finfo(float).eps*max(authoritative,selected+boundary,np.finfo(float).tiny)):
        raise R1Error("closure residual identity mismatch")
    tol=64*np.finfo(float).eps*(nl+1)*max(authoritative,selected+boundary,np.finfo(float).tiny)
    if abs(residual)>tol:
        raise R1Error("authoritative pre-EPAY energy census does not close: "
                      f"residual={residual:.17g} tol={tol:.17g}")
    if counts["rate_shape_replaced"]<=0 or not (energies["rate_shape_replaced"]>0):
        raise R1Error("hard gate: rate_shape_replaced lacks count/energy evidence")
    target=(rows["shell"]==8)&defined
    target_energy=float(rows["eta_energy"][target].sum(dtype=np.longdouble))
    median=None
    if target_energy>0:
        order=np.argsort(rows["rho_local"][target])
        rv=rows["rho_local"][target][order];ew=rows["eta_energy"][target][order]
        median=float(rv[np.searchsorted(np.cumsum(ew,dtype=np.longdouble),target_energy/2)])
    gain=None if median is None or median==1.0 else 1.0/(1.0-median)
    pred1=median is not None and 0.90<=median<0.98
    pred2=gain is not None and 10.0<=gain<=50.0
    return {"rows":int(nrows),"sha256":mf["sha256"],"field_generation":int(gen),
            "lambda_generation":int(lambda_gen),
            "disposition_counts":counts,"disposition_energy":energies,"KA-3.2.3":"PASS",
            "authoritative_energy":authoritative,"boundary_nonselected_energy":boundary,
            "closure_residual":residual,"eps_applied_diff_rows":eps_diff,
            "rho_undefined_chi_tot_zero_rows":rho_undefined,
            "rho_target_shell":8,"rho_target_energy":target_energy,
            "rho_energy_weighted_median":median,
            "geometric_amplification_1_over_1_minus_rho":gain,
            "prediction_1_status":"MATCH" if pred1 else "OUTSIDE_DISCOVERY",
            "prediction_2_status":"MATCH" if pred2 else "OUTSIDE_DISCOVERY"}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("path",type=Path)
    ap.add_argument("--expected-iteration",type=int,required=True)
    ap.add_argument("--expected-field-generation",type=int,required=True)
    ap.add_argument("--expect-eps-phys-zero",action="store_true")
    a=ap.parse_args()
    try:
        print(json.dumps(read_check(a.path,expected_iteration=a.expected_iteration,
                                    expected_field_generation=a.expected_field_generation,
                                    expect_unscaled_energy=a.expect_eps_phys_zero),
                         indent=2,sort_keys=True));return 0
    except Exception as e: print(f"FAIL: {e}");return 1
if __name__=="__main__": raise SystemExit(main())
