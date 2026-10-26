#!/usr/bin/env python3
"""실제 물질 한 점에서 OSC·COL 양쪽에 없는 Si/S 충돌 BB 쌍만 계량한다.

이 쌍들의 native OMEGA_GEN_V3 율은 정확히 해당 이온의 OMEGA_SET이다.
복사선 또는 표가 있는 쌍, 같은 SL의 용해 이온화, 전체 SE/RE는 세지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from pathlib import Path

import numpy as np

import build_cmfgen_coldata_all as col
import cmfgen_model_continuum as material
from cmfgen_parser import parse_osc
from expand_atomic_data_cmfgen import load_cmfgen_links, load_cmfgen_model_spec

IONS = ((14, 1, "Si II", 406), (14, 2, "Si III", 178),
        (16, 1, "S II", 2415), (16, 2, "S III", 227))
KB_EV = 8.617333262145e-5
EV_ERG = 1.602176634e-12


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def occupation(path, manifest, levels, shells):
    record = json.loads(manifest.read_text())
    require(record["state"] == "PASS" and record["input_identity"]["material_record"] == 70,
            "다른 점유확률 세대")
    require(material.sha(path) == record["inputs_sha256"][path.name], "점유확률 신원 변경")
    with path.open("rb") as stream:
        tag, nl, ns = struct.unpack("<8sII", stream.read(16))
    require((tag, nl, ns) == (b"LMATOCC1", levels, shells), "점유확률 차원 오류")
    require(path.stat().st_size == 16 + 8*levels*shells, "부분 점유확률")
    values = np.memmap(path, mode="r", offset=16, dtype="<f8", shape=(levels, shells))
    require(np.isfinite(values).all() and np.all((values > 0) & (values <= 1)),
            "영 또는 손상된 점유확률")
    return values


def omega_set(path):
    text = path.read_text(errors="replace")
    hits = re.findall(r"^\s*(\S+)\s+!\s*Value for OMEGA if f=0", text, re.M)
    require(len(hits) == 1, "OMEGA_SET 선언 누락/중복")
    value = float(hits[0].replace("D", "E"))
    require(np.isfinite(value) and value > 0, "OMEGA_SET 오류")
    return value


def one_ion(Z, ion0, name, expected_tab, rows, state, models, links, node):
    stage = ion0 + 1
    level_ids = [k for k, row in enumerate(rows)
                 if int(row["atomic_number"]) == Z and int(row["ion_number"]) == ion0]
    n = int(models[Z, stage]["n_full"])
    require(len(level_ids) == n and
            [int(rows[k]["level_number"]) for k in level_ids] == list(range(n)),
            f"{name}의 실제 FL 배열 불일치")
    osc_path, col_path = links[Z, stage]["osc"], links[Z, stage]["col"]
    osc = parse_osc(osc_path)
    require(len(osc.levels) >= n and len(osc.transitions) == osc.declared_n_transitions,
            f"{name} OSC 선언 불일치")
    for i, k in enumerate(level_ids):
        row = rows[k]
        require(row["configuration"] == osc.levels["config"][i] and
                float(row["g"]) == float(osc.levels["g"][i]) and
                abs(float(row["energy_eV"]) - float(osc.levels["E_cm"][i])*1.239841984e-4) <= 5.1e-11,
                f"{name} OSC/FL 신원 불일치")
    name_to_index, _, weights, n_osc = col.parse_osc(str(osc_path))
    table, _, _, stats = col.parse_col(str(col_path), name_to_index, weights, n_osc)
    table = {(i, j) for i, j in table if i < n and j < n}
    require(len(table) == expected_tab and stats["n_read"] == stats["n_decl"] and
            stats["n_nonpos"] == 0 and stats["n_term_reject"] == 0,
            f"{name} native 표 수/품질 불일치")
    marked = np.zeros((n, n), dtype=bool)
    for i, j in table:
        marked[i, j] = True
    raw_pairs = set()
    for tr in osc.transitions:
        i, j = int(tr["i"])-1, int(tr["j"])-1
        if 0 <= i < j < n:
            raw_pairs.add((i, j))
            marked[i, j] = True
    lower, upper = np.triu_indices(n, 1)
    sl = np.array([int(rows[k]["super_level"]) for k in level_ids], dtype="i4")
    mask = (sl[lower] != sl[upper]) & ~marked[lower, upper]
    lower, upper = lower[mask], upper[mask]
    global_ids = np.asarray(level_ids)
    e = np.array([float(rows[k]["energy_eV"]) for k in level_ids])
    g = np.array([float(rows[k]["g"]) for k in level_ids])
    gap = e[upper] - e[lower]
    # native는 동률 에너지의 서로 다른 FL 쌍도 OMEGA_SET 루프에 남긴다.
    # 그 쌍의 충돌률은 정의되지만 충돌 열량은 정확히 0이다.
    require(np.all(g > 0) and np.all(gap >= 0), f"{name} FL 에너지/중량 오류")
    pop = state["full"][global_ids, node]
    w = state["occupation"][global_ids, node]
    temp, ne = float(state["temperature"][node]), float(state["ne"][node])
    require(np.isfinite(pop).all() and np.all(pop >= 0) and np.all(w > 0) and
            temp > 0 and ne > 0, f"{name} 물질 상태 오류")
    strength = omega_set(col_path)
    common = ne * 8.63e-6 * strength / np.sqrt(temp)
    # SUBCOL_MULTI_V6: 초준위로 투영하면 HN_F/HN_S가 곱해지고,
    # 상향에는 W_F(J)/W_F(I)가 한 번 더 남는다. 하향에는 이 인자가 없다.
    up = pop[lower] * common/g[lower] * np.exp(-gap/(KB_EV*temp)) * w[upper]/w[lower]
    down = pop[upper] * common/g[upper]
    net = up-down
    exchange = net*gap*EV_ERG
    require(all(np.isfinite(x).all() and np.all(x >= 0) for x in (up, down)) and
            np.isfinite(exchange).all(), f"{name} 충돌율 비유한")
    largest = int(np.argmax(abs(exchange))) if len(exchange) else -1
    dominant = None if largest < 0 else dict(
        lower_level=int(lower[largest])+1, upper_level=int(upper[largest])+1,
        lower_configuration=rows[level_ids[lower[largest]]]["configuration"],
        upper_configuration=rows[level_ids[upper[largest]]]["configuration"],
        lower_superlevel=int(sl[lower[largest]])+1,
        upper_superlevel=int(sl[upper[largest]])+1,
        energy_gap_eV=float(gap[largest]),
        lower_population_cm3=float(pop[lower[largest]]),
        upper_population_cm3=float(pop[upper[largest]]),
        excitation_cm3_s=float(up[largest]),
        deexcitation_cm3_s=float(down[largest]),
        gas_cooling_erg_cm3_s=float(exchange[largest]))
    return dict(ion=name, full_levels=n, cross_super_unlisted_pairs=len(lower),
                zero_gap_pairs=int(np.count_nonzero(gap == 0)),
                raw_osc_pairs=len(raw_pairs),
                tabulated_pairs=len(table), omega_set=strength, native_node=node,
                temperature_K=temp, electron_density_cm3=ne,
                gross_excitation_cm3_s=float(np.sum(up, dtype="longdouble")),
                gross_deexcitation_cm3_s=float(np.sum(down, dtype="longdouble")),
                signed_excitation_cm3_s=float(np.sum(net, dtype="longdouble")),
                absolute_net_exchange_erg_cm3_s=float(np.sum(abs(exchange), dtype="longdouble")),
                signed_gas_cooling_erg_cm3_s=float(np.sum(exchange, dtype="longdouble")),
                max_abs_pair_exchange_erg_cm3_s=float(np.max(abs(exchange), initial=0.)),
                dominant_abs_exchange_pair=dominant,
                ion_density_cm3=float(np.sum(pop, dtype="longdouble")),
                osc_sha256=material.sha(osc_path), col_sha256=material.sha(col_path))


def run(args):
    model_spec = args.material_run/"MODEL_SPEC"
    links_path = args.material_run/"atomic_links.txt"
    levels = args.atomic/"levels.csv"
    seed_manifest = args.atomic.parent/"material-seed.json"
    seed = json.loads(seed_manifest.read_text())
    require(seed["source_record"] == 70 and seed["material_points"] == 90 and
            args.material_nodes.resolve() == (args.atomic.parent/"native_material_nodes.bin").resolve() and
            material.sha(args.material_nodes) == seed["output_sha256"]["native_material_nodes.bin"],
            "다른 물질70 원90점 입력")
    models = load_cmfgen_model_spec(model_spec, links_path)
    links = load_cmfgen_links(links_path)
    sources = [model_spec, links_path, levels, args.material_nodes,
               args.occupation, args.occupation_manifest, seed_manifest, Path(__file__)]
    sources.extend(links[z,ion0+1][kind] for z,ion0,_,_ in IONS for kind in ("osc","col"))
    before = {str(path): material.sha(path) for path in sources}
    with levels.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with args.material_nodes.open("rb") as stream:
        tag, shells, _, nlevels, _ = struct.unpack("<8sIIId", stream.read(28))
    require(tag in (b"LUMNOD01", b"LUMCEL01") and nlevels == len(rows) and
            shells == 90 and 0 <= args.node < shells, "실제 물질70/90점 차원 오류")
    w = occupation(args.occupation, args.occupation_manifest, nlevels, shells)
    state = material.read_material(args.material_nodes, rows, models, w)
    result = [one_ion(*ion, rows, state, models, links, args.node) for ion in IONS]
    require(all(material.sha(Path(path)) == digest for path, digest in before.items()),
            "물질/코드 입력이 계산 중 변경됨")
    return dict(schema="lumina-native-unlisted-collision-v1", material_record=70,
                scope="Si/S II,III에서 OSC와 COL 양쪽에 없는 cross-SL BB 쌍의 정확한 OMEGA_SET 항",
                exclusions=["표/OSC에 있는 쌍", "same-SL 및 용해 이온화", "전체 충돌망/SE/RE/새 수송"],
                input_sha256=before, rows=result, physical_RT_executions=0,
                material_updates=0, final_goal="NOT_MET")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("material-run", "atomic", "material-nodes", "occupation",
                 "occupation-manifest"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--node", type=int, default=49)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, allow_nan=False), flush=True)
