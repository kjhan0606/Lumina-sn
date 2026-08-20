#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from compare_physics_snapshots import SnapshotError, compare_snapshots, load_snapshot


SHELL_COLUMNS=(
    "shell_id","r_inner_cm","r_outer_cm","v_inner_cm_s","v_outer_cm_s",
    "T_e_K","n_e_cm3","n_atom_cm3","u_atom_erg",
    "q_ad_temperature_gradient","q_ad_velocity_divergence",
    "q_ad_electron_fraction_gradient","q_ad_internal_energy_gradient",
    "q_ad_signed_total","q_ad_heating","q_ad_cooling","photo_heat",
    "line_abs_heat","ff_abs_heat","compton_heat","gamma_heat",
    "nonthermal_heat","recomb_cool","line_emit_cool","coll_line_cool",
    "ff_emit_cool","compton_cool","sum_heating","sum_cooling","residual")
SPECTRAL_COLUMNS=("shell_id","bin_id","nu_lo_Hz","nu_hi_Hz","J_nu",
    "chi_es_cm1","chi_bb_cm1","chi_bf_cm1","chi_ff_cm1","chi_total_cm1",
    "eta_bb","eta_bf","eta_ff","eta_true_total")


def grid_manifest_sha256(edges:list[float])->str:
    digest=hashlib.sha256()
    digest.update(b"A2-09:grid-manifest:Hz:bin-edges:IEEE754:v1")
    digest.update(struct.pack(">Q",len(edges)-1))
    for edge in edges:digest.update(struct.pack(">d",edge))
    return digest.hexdigest()


def coarse_index(edges:list[float], x:float)->int:
    for i in range(len(edges)-1):
        if edges[i] <= x < edges[i+1] or (i==len(edges)-2 and x==edges[-1]):
            return i
    raise AssertionError(x)


def write_fixture(root:Path,name:str,shell_edges:list[float],freq_edges:list[float])->Path:
    shell_name=f"{name}.shell.csv"; spectral_name=f"{name}.spectral.csv"
    coarse_shell=[1.0,2.0,3.0]; coarse_freq=[1.0,2.0,4.0]
    with (root/shell_name).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=SHELL_COLUMNS);writer.writeheader()
        for s,(ri,ro) in enumerate(zip(shell_edges,shell_edges[1:])):
            base=10.0+coarse_index(coarse_shell,0.5*(ri+ro))
            row={key:0.0 for key in SHELL_COLUMNS}
            row.update(shell_id=s,r_inner_cm=ri,r_outer_cm=ro,
                v_inner_cm_s=ri/10.0,v_outer_cm_s=ro/10.0,
                T_e_K=5000*base,n_e_cm3=base,n_atom_cm3=2*base,
                u_atom_erg=1e-12*base,q_ad_temperature_gradient=0.1*base,
                q_ad_velocity_divergence=0.2*base,
                q_ad_electron_fraction_gradient=-0.1*base,
                q_ad_internal_energy_gradient=0.4*base,q_ad_signed_total=0.6*base,
                q_ad_heating=0.0,q_ad_cooling=0.6*base,photo_heat=2*base,
                line_abs_heat=base,recomb_cool=base,sum_heating=3*base,
                sum_cooling=1.6*base,residual=1.4*base)
            writer.writerow(row)
    with (root/spectral_name).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=SPECTRAL_COLUMNS);writer.writeheader()
        for s,(ri,ro) in enumerate(zip(shell_edges,shell_edges[1:])):
            sc=coarse_index(coarse_shell,0.5*(ri+ro))
            for b,(lo,hi) in enumerate(zip(freq_edges,freq_edges[1:])):
                fc=coarse_index(coarse_freq,0.5*(lo+hi));base=1.0+2*sc+fc
                writer.writerow(dict(shell_id=s,bin_id=b,nu_lo_Hz=lo,nu_hi_Hz=hi,
                    J_nu=base,chi_es_cm1=base,chi_bb_cm1=2*base,
                    chi_bf_cm1=3*base,chi_ff_cm1=4*base,chi_total_cm1=10*base,
                    eta_bb=base,eta_bf=2*base,eta_ff=3*base,eta_true_total=6*base))
    manifest={
        "schema":"LUMINA_PHYSICS_COMPARISON_V1","transaction_status":"COMMITTED",
        "code":name,"lane":"DET","iteration":0,"epoch_s":10.0,
        "n_shells":len(shell_edges)-1,"n_bins":len(freq_edges)-1,
        "frame":"SHELL_COMOVING","frequency_coordinate":"HZ",
        "opacity_units":"CM^-1",
        "emissivity_units":"ERG_S^-1_CM^-3_HZ^-1_SR^-1",
        "volume_rate_units":"ERG_S^-1_CM^-3","eta_is_per_sr":True,
        "radiative_integral_factor":4*math.pi,
        "adiabatic_positive_is_cooling":True,"shell_weight":"SPHERICAL_VOLUME",
        "frequency_regrid":"INTEGRAL_PRESERVING_PIECEWISE_CONSTANT",
        "atomic_model_sha256":"a"*64,"geometry_sha256":"b"*64,
        "te_manifest_sha256":"c"*64,
        "grid_manifest_sha256":grid_manifest_sha256(freq_edges),
        "radiation_generation":1,"population_generation":1,"te_generation":1,
        "opacity_generation":1,"emissivity_generation":1,
        "shell_file":shell_name,"spectral_file":spectral_name}
    path=root/f"{name}.manifest.json"
    path.write_text(json.dumps(manifest),encoding="utf-8")
    return path


def expect_blocked(path:Path,needle:str)->None:
    try: load_snapshot(path)
    except SnapshotError as exc:
        assert needle in str(exc),(needle,str(exc));return
    raise AssertionError(f"negative control did not block: {needle}")


def main()->int:
    with tempfile.TemporaryDirectory(prefix="physics-compare-selftest-") as tmp:
        root=Path(tmp)
        left=write_fixture(root,"left",[1,2,3],[1,2,4])
        right=write_fixture(root,"right",[1,1.5,2,2.5,3],[1,1.5,2,3,4])
        result=compare_snapshots(load_snapshot(left),load_snapshot(right),rtol=1e-14,atol=0)
        assert result["verdict"]=="PASS",result["failed_columns"]
        assert result["left_shell_coverage_fraction"]==1.0
        assert result["right_frequency_coverage_fraction"]==1.0

        data=json.loads(right.read_text());data["emissivity_units"]="ERG_S^-1_CM^-3_HZ^-1"
        bad=root/"bad_units.manifest.json";bad.write_text(json.dumps(data))
        expect_blocked(bad,"emissivity_units")
        data=json.loads(right.read_text());data["radiative_integral_factor"]=1.0
        bad=root/"bad_fourpi.manifest.json";bad.write_text(json.dumps(data))
        expect_blocked(bad,"4*pi")
        data=json.loads(right.read_text());data["adiabatic_positive_is_cooling"]=False
        bad=root/"bad_sign.manifest.json";bad.write_text(json.dumps(data))
        expect_blocked(bad,"adiabatic sign")
        data=json.loads(right.read_text());data["emissivity_generation"]=2
        bad=root/"bad_generation.manifest.json";bad.write_text(json.dumps(data))
        expect_blocked(bad,"generation mismatch")
    print("PHYSICS_COMPARISON_REGRID_SELFTEST PASS shell=VOLUME frequency=INTEGRAL "
          "negative_controls=4")
    return 0


if __name__=="__main__":raise SystemExit(main())
