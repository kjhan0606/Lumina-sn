"""실제 NT 자료와 원자 준위의 연결표. 없는 경로를 바닥상태로 대체하지 않는다.

아직 NLTE 소비자가 아니다. source line/global level ID/명시적 미연결 상태를
보존하여 실제 전이율 공급자의 입력으로 사용할 연결을 정한다.
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
    # terminal 파일은 정체성만 보존한다. 포맷을 추측해 인공 준위를 만들지 않는다.
    terminal = {(z, max(q for zz, q in ions if z == zz) + 1) for z, _ in ions}
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
            target_key = next_key
            if not lower:
                r["status"] = "LOWER_TERM_NOT_PRESENT"
            elif next_key in terminal:
                r["status"] = "TERMINAL_REQUIRES_CATALOG_BINDING"
            elif upper:
                r["status"] = "SINGLE_IONIZATION_MAPPED"
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
            if lower and upper and target_key == next_key:
                # 원자료 threshold와 실제 준위 간 내부에너지 차이를 별도 기록한다.
                delta = [ip[key] + u["energy_eV"] - lo["energy_eV"] for lo in lower for u in upper]
                r["level_energy_change_eV_range"] = [min(delta), max(delta)]
                r["threshold_minus_level_change_eV_range"] = [raw["threshold_eV"]-max(delta), raw["threshold_eV"]-min(delta)]
            output.append(r)
    summary = {}
    for r in output:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    result = dict(source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  actual_levels=len(levels), actual_explicit_ions=len(ions),
                  status_counts=summary, ions_without_ionization_data=missing_ions,
                  routes=output, NLTE_connected=False, spectrum_compared=False)
    with Path("nonthermal_atom_map.json").open("x") as fp:
        json.dump(result, fp, indent=2)
        fp.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("routes", "source_sha256")}))
    for r in output:
        if r["Z"] in (14,16) and r["charge"] in (1,2):
            print("NT_TARGET_ROUTE", json.dumps(r))
    for z in (14,16):
        for charge in (1,2):
            assert any(r["Z"] == z and r["charge"] == charge and r["status"] == "SINGLE_IONIZATION_MAPPED" for r in output)


if __name__ == "__main__":
    main()
