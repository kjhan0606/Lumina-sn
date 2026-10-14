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


def physical_profile_at_epoch(path, day=DAY):
    """공개 물질 표를 읽는다. 복사장이나 생산 입력으로 바꾸지 않는다."""
    lines = path.read_text().splitlines()
    starts = [i for i,line in enumerate(lines)
              if line.startswith("#TIME:") and abs(float(line.split()[1])-day)<1e-8]
    assert len(starts)==1
    start = starts[0]
    assert lines[start+1].startswith("#NVEL:")
    count = int(lines[start+1].split()[1])
    assert lines[start+2].split()==["#vel_mid[km/s]","temp[K]","rho[gcc]","ne[/cm^3]","natom[/cm^3]"]
    data = np.array([[float(v) for v in line.split()]
                     for line in lines[start+3:start+3+count]])
    assert data.shape==(count,5) and np.isfinite(data).all()
    assert np.all(data>0) and np.all(np.diff(data[:,0])>0)
    return data


def integrate_time_weighted_linear(seconds, power):
    """구간별 선형 power를 가정한 integral(t*power dt), 단위 erg*s."""
    seconds, power = np.asarray(seconds), np.asarray(power)
    assert seconds.ndim == power.ndim == 1 and len(seconds) == len(power) >= 2
    assert np.isfinite(seconds).all() and np.isfinite(power).all()
    assert seconds[0] > 0 and np.all(np.diff(seconds) > 0)
    dt = np.diff(seconds)
    # 각 구간의 두 끝값에 곱할 가중치. t*power는 선형이 아닌 이차함수다.
    return float(np.sum(dt/6 * ((2*seconds[:-1]+seconds[1:])*power[:-1]
                              +(seconds[:-1]+2*seconds[1:])*power[1:])))


def radiation_history_budget(bol):
    """전체 방출물의 근사 에너지식. 초기 E는 미지수이며 생산에 주입하지 않는다."""
    assert bol.ndim == 2 and bol.shape[1] == 3 and np.isfinite(bol).all()
    assert np.all(bol > 0) and np.all(np.diff(bol[:, 0]) > 0)
    hit = np.flatnonzero(np.abs(bol[:, 0]-DAY) < 1e-8)
    assert len(hit) == 1 and hit[0] > 0
    history = bol[:hit[0]+1]
    t, luminosity, deposited = history[:, 0]*86400, history[:, 1], history[:, 2]
    increment = integrate_time_weighted_linear(t, deposited-luminosity)
    alternate = float(np.trapezoid(t*(deposited-luminosity), t))
    present = t[-1]
    net_release = float(luminosity[-1]-deposited[-1])
    # 미지 E(t0)>=0, 위 근사식과 보간이 성립할 때의 조건부 하한이다.
    # 공개 표만으로 CMFGEN의 실제 radiation-storage 적분을 측정했다고 하지 않는다.
    energy_min = max(0.0, increment/present)
    analytic_t = np.array([1.0, 1.3, 2.6, 4.0])*86400
    for constant, slope in ((3.0, 0.0), (0.0, 2.0), (-4.0, 3.0)):
        power = constant+slope*analytic_t
        exact = (constant*(analytic_t[-1]**2-analytic_t[0]**2)/2
                 +slope*(analytic_t[-1]**3-analytic_t[0]**3)/3)
        assert math.isclose(integrate_time_weighted_linear(analytic_t, power),
                            exact, rel_tol=1e-14)
    return dict(
        equation="dE/dt = Q-L-E/t; whole ejecta, radiation dominated, nonrelativistic homologous expansion",
        reference="https://arxiv.org/abs/1301.6766",
        start_day=float(history[0, 0]), end_day=float(history[-1, 0]), samples=len(history),
        initial_energy="unknown nonnegative E(t0); not assumed zero",
        initial_energy_coefficient_in_E_at_target=float(t[0]/present),
        known_contribution_to_E_erg=increment/present,
        time_weighted_integral_erg_s=increment,
        weighted_table_trapezoid_erg_s=alternate,
        quadrature_difference_relative=abs(increment-alternate)/max(abs(increment), abs(alternate)),
        conditional_E_lower_erg=energy_min,
        conditional_expansion_loss_lower_erg_s=energy_min/present,
        public_L_minus_Q_erg_s=net_release,
        public_L_minus_Q_over_L=net_release/float(luminosity[-1]),
        conditional_energy_release_lower_erg_s=energy_min/present+net_release,
        conditional_energy_release_lower_over_L=(energy_min/present+net_release)/float(luminosity[-1]),
        analytic_integral_tests="PASS: constant and affine power on nonuniform times",
        warning="Conditional whole-ejecta inference under the stated approximations, not a rigorous "
                "CMFGEN energy measurement or error bound. Sparse-time interpolation, frame/relativistic "
                "effects and gas/internal energy are not controlled by the quadrature comparison. "
                "No local radiation spectrum, truncated-domain storage or production correction is inferred.")


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
    profile = physical_profile_at_epoch(PUBLIC/"phys_toy06_cmfgen.txt")
    vin = float(geometry[0]["v_inner"])/1e5
    vout = float(geometry[-1]["v_outer"])/1e5
    assert profile[0,0]<=vin<vout<=profile[-1,0]
    # 자유전자 산란 깊이만 계산한다. 열평형화 깊이·복사 저장·순경계 광도가 아니다.
    electron_tau = 6.6524587321e-25*config["time_explosion_s"]*1e5*integrate(
        profile[:,0],profile[:,3],vin,vout)
    public_T_at_cut = float(np.interp(vin,profile[:,0],profile[:,1]))
    boundary = dict(public_profile_min_velocity_km_s=float(profile[0,0]),
        model_inner_face_velocity_km_s=vin,model_outer_face_velocity_km_s=vout,
        public_gas_temperature_at_model_inner_face_K=public_T_at_cut,
        model_fixed_outgoing_blackbody_temperature_K=config["T_inner_K"],
        public_electron_scattering_tau_across_model=electron_tau,
        hypothetical_full_blackbody_outward_L_at_public_gas_T=
            Loutward*(public_T_at_cut/config["T_inner_K"])**4,
        warning="Public gas T is not a measured boundary radiation temperature. "
            "Replacing T_inner with it would change injected power and is not a diffusion boundary solution. "
            "No public temperature or population is injected into production.")
    result = dict(scope="input-only; no return flux, expansion work or time storage measured",
        target_day=DAY,inner_outward_L=Loutward,volume_integrated_input_deposition=Lheat,
        public_Lbol=Lbol,public_Edep=Edep,public_L_2500_25500=Lcommon,
        gross_input_sum=gross,gross_sum_over_public_Lbol=gross/Lbol,
        gross_sum_over_public_common=gross/Lcommon,
        gross_minus_90pct_common=gross-.9*Lcommon,
        blackbody_T_inner_K=config["T_inner_K"],
        boundary_domain_comparison=boundary,
        deposition_reconstruction=comparison,
        public_radiation_history_budget=radiation_history_budget(bol),
        warning="L_inner is outward, not net; input sum is not predicted emergent luminosity or proof of a time-storage term")
    paths = [MODEL/n for n in ("config.json","geometry.csv","deposition_cmfgen.csv")]
    paths += [PUBLIC/n for n in ("spectra_toy06_cmfgen.txt","lbol_edep_toy06_cmfgen.txt","edep_toy06_cmfgen.txt","phys_toy06_cmfgen.txt")]
    result["sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]}
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
