"""실제 NT 자료와 원자 준위의 연결표. 없는 경로를 바닥상태로 대체하지 않는다.

source line/global level ID/명시적 미연결 상태를 보존한다. 에너지가 허용하고
실제 준위가 있는 경로만 별도 CSV로 내보낸다. 이 CSV는 불완전한 원자 모델의
명시적인 대조 입력이며 누락 경로가 물리적으로0이라는 뜻이 아니다.
"""
import csv
import hashlib
import json
import re
from pathlib import Path


def read_routes(path):
    lines = path.read_text().splitlines()
    routes, i = [], 0
    while i < len(lines):
        line, number = lines[i].strip(), i + 1
        i += 1
        if not line or line.startswith("!"):
            continue
        fields = line.split(maxsplit=9)
        z, bound, n, ell = map(int, fields[:4])
        threshold = float(fields[4])
        try:
            tuple(map(float, fields[5:9]))
            assert len(fields[5:9]) == 4
            terms = fields[9] if len(fields) == 10 else ""
            ntab = 0
        except (ValueError, AssertionError):
            fields = line.split(maxsplit=6)
            ntab = int(fields[5])
            terms = fields[6] if len(fields) == 7 else ""
            assert len(lines[i].split()) == ntab and len(lines[i+1].split()) == ntab
            i += 2
        routes.append(dict(source_line=number, Z=z, charge=z-bound,
                           principal_n=n, ell=ell, threshold_eV=threshold,
                           n_tab=ntab, terms=terms))
    return routes


def term(name):
    return name.split("[")[0]


def matches(aliases, name):
    # 부분 문자열(예: 3d vs 3d2)을 같은 term으로 취급하지 않는다.
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) +
                          r"(?![A-Za-z0-9_])", aliases))


def main():
    paths = [Path("input/arnaud_rothenflug_23jun15"), Path("input/levels.csv"),
             Path("input/ionization_energies.csv"), Path("input/bf_channels.terminal.csv")]
    routes = read_routes(paths[0])
    assert len(routes) == 407
    levels, ions = [], {}
    with paths[1].open() as fp:
        for row in csv.DictReader(fp):
            entry = dict(global_level=len(levels), Z=int(row["atomic_number"]),
                         charge=int(row["ion_number"]), level_number=int(row["level_number"]),
                         energy_eV=float(row["energy_eV"]), g=int(row["g"]),
                         configuration=row["configuration"], term=term(row["configuration"]))
            assert entry["g"] > 0
            levels.append(entry)
            ions.setdefault((entry["Z"], entry["charge"]), []).append(entry)
    with paths[2].open() as fp:
        ip = {(int(r["atomic_number"]), int(r["ion_number"])): float(r["ionization_energy_eV"])
              for r in csv.DictReader(fp)}
    # 실제 catalog를 읽되 full level을 만들지 않는다. terminal은 원자 모델의
    # 마지막 이온 인구이며 fine-level 여기나 추가 전리 채널을 뜻하지 않는다.
    with paths[3].open() as fp:
        catalog = {(int(r["Z"]), int(r["ion_stage_0based"])): r for r in csv.DictReader(fp)}
    terminal = {(z, max(q for zz, q in ions if z == zz) + 1) for z, _ in ions}
    assert terminal == set(catalog)
    for c in catalog.values():
        assert int(c["level_index"]) == 0 and float(c["E_cm-1"]) == 0 and float(c["g"]) > 0
    output, missing_ions = [], []
    for key, lev in sorted(ions.items()):
        ground = [x for x in lev if x["level_number"] == 0]
        assert len(ground) == 1 and ground[0]["energy_eV"] == 0
        selected = [r for r in routes if (r["Z"], r["charge"]) == key]
        if not selected:
            missing_ions.append(dict(Z=key[0], charge=key[1], reason="NO_ATOMIC_CROSS_SECTION"))
        for raw in selected:
            r = dict(raw)
            label = raw["terms"].strip()
            if label.startswith("*"):
                lower_term, _, label = label[1:].partition(" ")
                label = label.strip()
            else:
                lower_term = ground[0]["term"]
            lower = [x for x in lev if x["term"] == lower_term]
            r["lower_levels"] = [x["global_level"] for x in lower]
            next_key = (key[0], key[1]+1)
            # '&' 뒤는 다음다음 이온 후보다. 일반 후보와 혼합하지 않는다.
            single_label = " ".join(part.split("&")[0] for part in re.split(r"[/\\]", label))
            upper = [x for x in ions.get(next_key, []) if matches(single_label, x["term"])]
            # '/'가 항상 철자 alias는 아니다. 다른 configuration/term을
            # ground 대체 이름으로 적은 행을 실제 들뜬 상태 연결로 오인하지 않는다.
            primary = re.split(r"[/\\]", label)[0].split("&")[0].strip()
            canonical = primary.replace("_", "")
            primary_upper = [x for x in upper if x["term"].replace("_", "") == canonical]
            target_key = next_key
            if not lower:
                r["status"] = "LOWER_TERM_NOT_PRESENT"
            elif next_key in terminal:
                r["status"] = "TERMINAL_REQUIRES_CATALOG_BINDING"
            elif upper:
                r["status"] = "SINGLE_IONIZATION_MAPPED" if primary_upper else "NONIDENTICAL_ALTERNATIVE_TARGET"
                if primary_upper:
                    upper = primary_upper
            elif "&" in label:
                target_key = (key[0], key[1]+2)
                auger_label = " ".join(label.split("&")[1:])
                upper = [x for x in ions.get(target_key, []) if matches(auger_label, x["term"])]
                r["status"] = "AUGER_REQUIRES_ENERGY_KERNEL" if upper or target_key in terminal else "UPPER_TERM_NOT_PRESENT"
            else:
                r["status"] = "UPPER_TERM_NOT_PRESENT"
            r["target_charge"] = target_key[1]
            r["upper_levels"] = [x["global_level"] for x in upper]
            weight = sum(x["g"] for x in upper)
            r["upper_weights"] = [x["g"]/weight for x in upper]
            if target_key in terminal:
                r["terminal_catalog"] = catalog[target_key]
            if lower and (upper or target_key in terminal):
                # 원자료 threshold와 실제 준위 간 내부에너지 차이를 별도 기록한다.
                binding = sum(ip[(key[0], q)] for q in range(key[1], target_key[1]))
                ends = [u["energy_eV"] for u in upper] if upper else [0.0]
                delta = [binding + end - lo["energy_eV"] for lo in lower for end in ends]
                r["level_energy_change_eV_range"] = [min(delta), max(delta)]
                r["threshold_minus_level_change_eV_range"] = [raw["threshold_eV"]-max(delta), raw["threshold_eV"]-min(delta)]
                if target_key[1] == key[1]+2:
                    r["auger_energy_feasible_all_branches"] = raw["threshold_eV"] > max(delta)
            output.append(r)
    bindings, unresolved, selected_lines = [], [], set()
    for r in output:
        key = (r["Z"], r["charge"])
        target = (r["Z"], r["target_charge"])
        if r.get("auger_energy_feasible_all_branches") is False:
            unresolved.append(dict(source_line=r["source_line"], reason="NO_ENERGY_FOR_ASSIGNED_AUGER"))
            continue
        if target in terminal and r["target_charge"] == r["charge"]+1:
            single_label = " ".join(part.split("&")[0] for part in re.split(r"[/\\]", r["terms"]))
            if not any(matches(single_label, alias) for alias in catalog[target]["label"].split("/")):
                unresolved.append(dict(source_line=r["source_line"], reason="TERMINAL_EXCITED_STATE_NOT_REPRESENTED"))
                continue
        if r["status"] not in ("SINGLE_IONIZATION_MAPPED", "AUGER_REQUIRES_ENERGY_KERNEL", "TERMINAL_REQUIRES_CATALOG_BINDING"):
            unresolved.append(dict(source_line=r["source_line"], reason=r["status"]))
            continue
        assert r["lower_levels"] and (r["upper_levels"] or target in terminal)
        lowers = [levels[i] for i in r["lower_levels"]]
        reference = min(lo["energy_eV"] for lo in lowers)
        for lo in lowers:
            for upper, weight in zip(r["upper_levels"] or [-1], r["upper_weights"] or [1.0]):
                change = sum(ip[(key[0], q)] for q in range(key[1], target[1])) - lo["energy_eV"]
                if upper >= 0:
                    change += levels[upper]["energy_eV"]
                if target[1] == key[1]+2:
                    threshold = r["threshold_eV"] - (lo["energy_eV"]-reference)
                    auger = threshold-change
                    assert auger > 0
                else:
                    threshold, auger = change, 0.0
                assert threshold > 0
                bindings.append([r["source_line"], lo["global_level"], upper, target[0], target[1],
                                 threshold, auger, weight])
                selected_lines.add(r["source_line"])
    assert len(selected_lines)+len(unresolved) == len(output)
    with Path("nonthermal_resolved_bindings.csv").open("x") as fp:
        fp.write(f"# resolved_routes={len(selected_lines)} unresolved_routes={len(unresolved)} missing_ions={len(missing_ions)}\n")
        writer = csv.writer(fp)
        writer.writerow(["source_line", "lower_global", "upper_global", "target_Z", "target_charge",
                         "threshold_eV", "auger_eV", "branch_weight"])
        writer.writerows(bindings)
    summary = {}
    for r in output:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    result = dict(source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  actual_levels=len(levels), actual_explicit_ions=len(ions),
                  status_counts=summary, ions_without_ionization_data=missing_ions,
                  resolved_routes=len(selected_lines), resolved_channels=len(bindings),
                  unresolved_routes=unresolved,
                  binding_policy="explicit_resolved_subset; sigma_at_same_excess_energy; not_full_atomic_model",
                  routes=output, NLTE_connected=False, spectrum_compared=False)
    with Path("nonthermal_atom_map.json").open("x") as fp:
        json.dump(result, fp, indent=2)
        fp.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("routes", "source_sha256")}))
    for r in output:
        if r.get("auger_energy_feasible_all_branches") is False:
            print("NT_ENERGETICALLY_INVALID_AUGER", json.dumps(r))
        if r["Z"] in (14,16) and r["charge"] in (1,2):
            print("NT_TARGET_ROUTE", json.dumps(r))
    for z in (14,16):
        for charge in (1,2):
            assert any(r["Z"] == z and r["charge"] == charge and r["status"] == "SINGLE_IONIZATION_MAPPED" for r in output)


if __name__ == "__main__":
    main()
