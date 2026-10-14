#!/usr/bin/env python3
"""COMMON 입력 에너지와 공개 기준 대역을 대조한다. 새 물리 해나 수렴 판정이 아니다."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

MODEL = Path("/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/model")
PUBLIC = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06")
DAY = 19.48


def matrix_at_epoch(path):
    text = path.read_text().splitlines()
    times = next(x for x in text if x.startswith("#TIMES[d]:")).split()[1:]
    hits = [i+1 for i,t in enumerate(times) if abs(float(t)-DAY)<1e-8]
    assert len(hits)==1
    data = np.loadtxt(path)
    assert data.shape[1]==len(times)+1 and np.all(np.diff(data[:,0])>0)
    return data[:,0],data[:,hits[0]]


def integrate(x,y,low,high):
    assert x[0]<=low<high<=x[-1]
    selected = (x>low)&(x<high)
    xx = np.concatenate(([low],x[selected],[high]))
    yy = np.interp(xx,x,y)
    return float(np.trapezoid(yy,xx))


def main():
    assert os.environ.get("SLURM_JOB_ID"), "Slurm 안에서 계산한다"
    config = json.loads((MODEL/"config.json").read_text())
    with (MODEL/"geometry.csv").open() as handle:
        geometry = list(csv.DictReader(handle))
    with (MODEL/"deposition_cmfgen.csv").open() as handle:
        deposition = {int(r["shell_id"]):float(r["heating_rate"]) for r in csv.DictReader(handle)}
    volume = np.array([4*math.pi/3*(float(r["r_outer"])**3-float(r["r_inner"])**3) for r in geometry])
    velocity = np.array([(float(r["v_inner"])+float(r["v_outer"]))/2e5 for r in geometry])
    heat = np.array([deposition[int(r["shell_id"])] for r in geometry])
    assert np.all(heat>=0) and np.all(volume>0)
    lam,L = matrix_at_epoch(PUBLIC/"spectra_toy06_cmfgen.txt")
    Lcommon = integrate(lam,L,2500,25500)
    bol = np.loadtxt(PUBLIC/"lbol_edep_toy06_cmfgen.txt")
    hit = bol[np.abs(bol[:,0]-DAY)<1e-8]
    assert len(hit)==1
    Lbol,Edep = map(float,hit[0,1:3])
    vdep,hdep = matrix_at_epoch(PUBLIC/"edep_toy06_cmfgen.txt")
    assert vdep[0]<=velocity.min() and velocity.max()<=vdep[-1]
    candidates = {"linear_velocity_volumetric":np.interp(velocity,vdep,hdep)}
    if np.all(hdep>0):
        candidates["linear_velocity_log_volumetric"] = np.exp(np.interp(velocity,vdep,np.log(hdep)))
    comparison = {}
    for name,predicted in candidates.items():
        scale = np.maximum(np.maximum(np.abs(heat),np.abs(predicted)),1e-300)
        rel = np.abs(heat-predicted)/scale
        comparison[name] = dict(max_symmetric_relative=float(rel.max()),
            median_symmetric_relative=float(np.median(rel)),
            matching_printed_6e=sum(f"{a:.6e}"==f"{b:.6e}" for a,b in zip(heat,predicted)),
            shells=len(heat))
    Loutward = float(config["luminosity_inner_erg_s"])
    Lheat = float(heat@volume)
    gross = Loutward+Lheat
    result = dict(scope="input-only; no return flux, expansion work or time storage measured",
        target_day=DAY,inner_outward_L=Loutward,volume_integrated_input_deposition=Lheat,
        public_Lbol=Lbol,public_Edep=Edep,public_L_2500_25500=Lcommon,
        gross_input_sum=gross,gross_sum_over_public_Lbol=gross/Lbol,
        gross_sum_over_public_common=gross/Lcommon,
        gross_minus_90pct_common=gross-.9*Lcommon,
        blackbody_T_inner_K=config["T_inner_K"],
        deposition_reconstruction=comparison,
        warning="L_inner is outward, not net; input sum is not predicted emergent luminosity or proof of a time-storage term")
    paths = [MODEL/n for n in ("config.json","geometry.csv","deposition_cmfgen.csv")]
    paths += [PUBLIC/n for n in ("spectra_toy06_cmfgen.txt","lbol_edep_toy06_cmfgen.txt","edep_toy06_cmfgen.txt")]
    result["sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]}
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
