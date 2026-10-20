#!/usr/bin/env python3
"""봉인 CMFGEN pre-solve 상태를 실제 C Si/S owner 입력으로 내보낸다.

수치 SE는 수행하지 않는다. 원래 SCRTEMP superlevel 인구를 보존하고,
fine-level 내부 분배는 C 생산 owner가 같은 T에서 재구성한다.
"""
import argparse
import csv
import json
import re
import struct
from pathlib import Path

import numpy as np

import compare_cmfgen_bf_superlevel_rates as rates
from ptf_gated_metrics import ScrtempReader, read_dimensions, read_variables, read_rvtj_block


def stage_atomic_view(deck, run, out, state, variables, rho, loaded):
    """동일 원자 파일 + 실제 첫 CMFGEN 점의 물질 초기화. 원본 덱은 불변."""
    view = out / "atomic_view"
    view.mkdir()
    # A single, explicitly selected small deck directory; no recursive walk.
    for source in deck.iterdir():
        if source.name.startswith("abundances.csv") or not source.is_file():
            continue
        (view / source.name).symlink_to(source)
    masses = list(csv.DictReader((deck / "atom_masses.csv").open()))
    species = {14: "Si", 16: "S", 20: "Ca", 26: "Fe", 27: "Co", 28: "Ni"}
    mass_rows = []
    with (view / "abundances.csv").open("x", newline="") as f:
        writer = csv.writer(f); writer.writerow(["atomic_number", "0"])
        for row in masses:
            Z, mass = int(row["atomic_number"]), float(row["mass_amu"])
            indexes = [v.index-1 for v in variables if v.species == species[Z]]
            if not indexes:
                raise ValueError(f"CMFGEN 물질 원소 누락 Z={Z}")
            value = float(state[indexes, 0].sum()*mass*1.660539066e-24/rho[0])
            writer.writerow([Z, format(value, ".17g")]); mass_rows.append((Z, value))
    # Only IV needs the legacy target map. II/III consume their own PHOT CSR,
    # and V uses the explicit terminal scalar. Resolve IV from sealed sources.
    levels = list(csv.DictReader((deck / "levels.csv").open()))
    level_index = {(int(r["atomic_number"]), int(r["ion_number"])+1,
                    int(r["level_number"])+1): i for i,r in enumerate(levels)}
    target_rows = {}
    from types import SimpleNamespace
    for Z in (14,16):
        upper = loaded["model"][(Z,5)]
        source = loaded["links"][(Z,5)]
        osc = rates.atomic.parse_osc(source["osc"])
        count = upper["n_full"]
        mapping = np.array([loaded["runtime"][(upper["label"],k)]["superlevel"]
                            for k in range(1,count+1)],dtype=int)
        data = dict(levels=osc.levels,n_kept=count,ftos=SimpleNamespace(sl_of_fl=mapping))
        phot_paths = rates.atomic._phot_route_paths(Path(loaded["links"][(Z,4)]["phot"]))
        if len(phot_paths) != 1:
            raise ValueError("IV legacy target map requires one sealed PHOT grid")
        phot = rates.atomic.parse_phot(phot_paths[0])
        components = rates.atomic._phot_target_components(phot,data,level_index,Z,5)
        for i,row in enumerate(levels):
            if int(row["atomic_number"])==Z and int(row["ion_number"])==3:
                target_rows[i]=[(int(c["upper_global"]),float(c["probability"])) for c in components]
    offsets=[0];targets=[];probabilities=[]
    for i in range(len(levels)):
        for target,probability in target_rows.get(i,[]):
            targets.append(target);probabilities.append(probability)
        offsets.append(len(targets))
    path=view/"ma_radrecomb_target.bin"
    if path.exists(): raise ValueError("기존 target map 보존")
    with path.open("xb") as f:
        f.write(struct.pack("<IIiii",0x4D415254,2,len(levels),2,len(targets)))
        f.write(np.asarray(offsets,dtype="<i4").tobytes())
        f.write(np.asarray(targets,dtype="<i4").tobytes())
        f.write(np.asarray(probabilities,dtype="<f8").tobytes())
    return dict(path=str(view), material_initialization="first matched CMFGEN node, not normalized",
                mass_fractions=mass_rows, abundances_sha256=rates.sha256_file(view/"abundances.csv"),
                legacy_IV_target_sha256=rates.sha256_file(path))


def write_input(path, nu, profiles, columns, populations, field, epoch):
    nd = len(profiles)
    if (profiles.shape != (nd, 5) or populations.shape != (nd, len(columns)) or
            field.shape != (len(nu), nd) or len(nu) < 2 or
            not np.all(np.diff(nu) > 0) or not np.isfinite(field).all() or
            np.any(field < 0) or not np.isfinite(populations).all() or
            np.any(populations < 0) or not np.isfinite(profiles).all() or
            np.any(profiles[:, :3] <= 0) or not np.isfinite(epoch) or epoch <= 0 or
            np.any(profiles[:, 3] < 0) or
            np.any(profiles[:, 4] <= profiles[:, 3])):
        raise ValueError("잘못된 CMFGEN owner 입력")
    edges = rates.l1bf.canonical_edges()
    with path.open("xb") as out:
        out.write(struct.pack("<8sIIIId", b"CMFOWN01", nd, len(nu), len(columns),
                              len(edges) - 1, epoch))
        out.write(np.asarray(nu, dtype="<f8").tobytes())
        out.write(np.asarray(columns, dtype="<i4").tobytes())
        for depth in range(nd):
            j, valid = rates.l1bf.conservative_rebin(nu, field[:, depth], edges)
            for a in (profiles[depth], populations[depth], j):
                out.write(np.asarray(a, dtype="<f8").tobytes())
            out.write(np.asarray(valid, dtype="<i4").tobytes())
            out.write(np.asarray(field[:, depth], dtype="<f8").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmfgen-run", type=Path, required=True)
    ap.add_argument("--deck", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run, deck, out = (p.resolve() for p in (args.cmfgen_run, args.deck, args.out))
    if out.exists():
        raise ValueError(f"기존 출력 보존: {out}")
    _, _, loaded = rates.verify_sealed_deck(run, deck)
    model = (run / "MODEL").read_text()
    if not re.search(r"^\s*F\s+\[TRT_NON_TE\]", model, re.M):
        raise ValueError("현재 adapter는 비열적 전자 비활성 CMFGEN 상태만 지원")
    field, nu, nd, finish = rates.read_eddfactor(str(run / "EDDFACTOR"))
    if finish != 1.0:
        raise ValueError("미완성 EDDFACTOR")
    scr_nd, nt = read_dimensions(run)
    if nd != scr_nd:
        raise ValueError("EDDFACTOR/SCRTEMP 깊이 불일치")
    variables = read_variables(run, nt - 2)
    indices = {(v.ion, v.superlevel): v.index - 1 for v in variables}
    profiles = {}
    for Z, stage in sorted(rates.TARGETS):
        label = loaded["model"][(Z, stage)]["label"]
        path = run / f"{label}PRRR"
        prrr = rates.parse_prrr_summary(path, label, nd,
                                        loaded["model"][(Z, stage)]["n_super"])
        profiles[label] = dict(next_ion=loaded["model"][(Z, stage + 1)]["label"],
                               ion_density=np.asarray(prrr["ion_density"]),
                               electron_density=np.asarray(prrr["electron_density"]),
                               temperature=rates.parse_prrr_temperature(path, nd))
    reader = ScrtempReader(run / "SCRTEMP", nd, nt)
    record, state, matched = rates.match_prrr_scrtemp_state(
        reader, indices, profiles, nt - 2, nt - 1)
    selected = [v for v in variables if v.species in ("Si", "S")]
    columns = [(14 if v.species == "Si" else 16, v.charge, v.superlevel - 1)
               for v in selected]
    if len(set(columns)) != len(columns):
        raise ValueError("중복 population 키")
    for Z in (14, 16):
        for charge in range(1, 5):
            spec = loaded["model"][(Z, charge + 1)]
            actual = {sl for z, c, sl in columns if (z, c) == (Z, charge)}
            if actual != set(range(spec["n_super"])):
                raise ValueError(f"활성 SL 누락 Z={Z}, charge={charge}")
        if sum(z == Z and c == 5 for z, c, _ in columns) != 1:
            raise ValueError(f"terminal VI 변수 누락/중복 Z={Z}")
    if any(c < 1 or c > 5 for _, c, _ in columns):
        raise ValueError("II--VI 밖의 population")
    rho = read_rvtj_block(run / "RVTJ", "Mass Density (gm/cm^3)", nd)
    velocity = read_rvtj_block(run / "RVTJ", "Velocity (km/s)", nd)
    if rho is None or velocity is None or np.any(velocity <= 0):
        raise ValueError("질량밀도/속도 누락")
    epochs = np.asarray(reader.radius) * 1e10 / (velocity * 1e5)
    epoch = float(np.median(epochs))
    if np.max(np.abs(epochs / epoch - 1)) > 1e-5:
        raise ValueError("상동팽창 시각 불일치")
    # Local material evaluation at CMFGEN depth nodes, not a new transfer grid.
    v = velocity * 1e5
    mid = 0.5 * (v[:-1] + v[1:])
    vin = np.r_[mid, v[-1] - (mid[-1] - v[-1])]
    vout = np.r_[v[0] + (v[0] - mid[0]), mid]
    state_profiles = np.column_stack((rho, state[nt - 1] * 1e4,
                                      state[nt - 2], vin, vout))
    populations = state[[v.index - 1 for v in selected]].T
    out.mkdir()
    binary = out / "owner_input.bin"
    write_input(binary, nu, state_profiles, columns, populations, field, epoch)
    atomic_view = stage_atomic_view(deck, run, out, state, variables, rho, loaded)
    inputs = [run / name for name in ("MODEL", "SCRTEMP", "RVTJ", "EDDFACTOR",
                                     "LEVEL_SL_STEQ_LINKS", "MODEL_SPEC")]
    inputs += [run / f"{label}PRRR" for label in profiles]
    manifest = dict(schema="cmfgen-owner-input-v1", cmfgen_run=str(run), deck=str(deck),
                    depths=nd, native_frequencies=len(nu), population_columns=len(columns),
                    atomic_view=atomic_view,
                    matched_scrtemp_record=record, phase_match=matched, epoch_seconds=epoch,
                    radiation="CMFGEN EDDFACTOR, piecewise linear in frequency",
                    fine_distribution="production Boltzmann within SL; exact input SL totals",
                    nonthermal="TRT_NON_TE=F; zero nonthermal source for SE only; no RE claim",
                    transfer_grid="none; publication bounds around CMFGEN nodes only",
                    CMFGEN_converged=False, final_goal="NOT_JUDGED",
                    binary_sha256=rates.sha256_file(binary),
                    input_sha256={str(p): rates.sha256_file(p) for p in inputs},
                    preparer_sha256=rates.sha256_file(Path(__file__)))
    (out / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("depths", "native_frequencies",
                                              "matched_scrtemp_record", "binary_sha256")}))


if __name__ == "__main__":
    main()
