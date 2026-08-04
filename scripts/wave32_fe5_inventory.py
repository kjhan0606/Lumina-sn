#!/usr/bin/env python3
"""Print the Wave-3.2 R6 Fe II--V data inventory from a model projection."""

import argparse
import csv
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path,
                        default=ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv")
    return parser.parse_args()


def main():
    model = arguments().model.resolve()
    levels = list(csv.DictReader((model / "levels.csv").open()))
    fe = {}
    for index, row in enumerate(levels):
        if int(row["atomic_number"]) == 26:
            fe.setdefault(int(row["ion_number"]), []).append(index)
    with (model / "cmfgen_sigma_bf.bin").open("rb") as handle:
        magic, version, nlevels, nfreq = struct.unpack("<IIii", handle.read(16))
        nu_min, nu_max = struct.unpack("<dd", handle.read(16))
        flags = handle.read(nlevels)
    with (model / "ma_radrecomb_target.bin").open("rb") as handle:
        ma_magic, ma_version, ma_levels, ma_ions = struct.unpack(
            "<IIii", handle.read(16))
        if ma_version == 1:
            targets = struct.unpack(f"<{ma_levels}i", handle.read(4 * ma_levels))
            offsets = probabilities = route_targets = None
        elif ma_version == 2:
            nroutes, = struct.unpack("<i", handle.read(4))
            offsets = struct.unpack(f"<{ma_levels + 1}i",
                                    handle.read(4 * (ma_levels + 1)))
            route_targets = struct.unpack(f"<{nroutes}i", handle.read(4 * nroutes))
            probabilities = struct.unpack(f"<{nroutes}d", handle.read(8 * nroutes))
            targets = None
        else:
            raise ValueError(f"unsupported MART version {ma_version}")
    ionization = {(int(row["atomic_number"]), int(row["ion_number"])):
                  float(row["ionization_energy_eV"])
                  for row in csv.DictReader(
                      (model / "ionization_energies.csv").open())}
    if magic != 0x434D4644 or nlevels != len(levels):
        raise ValueError("CMFD header/model mismatch")
    if ma_magic != 0x4D415254 or ma_levels != len(levels):
        raise ValueError("MART header/model mismatch")
    print(f"model={model}")
    print(f"sigma_header=CMFD/v{version} levels={nlevels} bins={nfreq} "
          f"nu={nu_min:.9e}..{nu_max:.9e}")
    print(f"ma_header=MART/v{ma_version} levels={ma_levels} ions={ma_ions}")
    print("ion_number,spectroscopic_stage,levels,sigma_rows,sigma_coverage_pct,"
          "ma_source_levels,ma_routes,valid_target_routes,target_ion_numbers,"
          "ionization_energy_eV")
    for ion in range(1, 5):
        indices = fe.get(ion, [])
        sigma = sum(bool(flags[index]) for index in indices)
        valid = routes = sources = 0
        target_ions = set()
        for index in indices:
            if ma_version == 1:
                route_list = [(targets[index], 1.0)] if targets[index] >= 0 else []
            else:
                route_list = [(route_targets[r], probabilities[r])
                              for r in range(offsets[index], offsets[index + 1])]
            if route_list:
                sources += 1
            routes += len(route_list)
            for target, probability in route_list:
                if 0 <= target < len(levels) and probability > 0.0:
                    valid += 1
                    target_ions.add(int(levels[target]["ion_number"]))
        coverage = 100.0 * sigma / len(indices) if indices else 0.0
        print(f"{ion},Fe {ion + 1},{len(indices)},{sigma},{coverage:.6f},"
              f"{sources},{routes},{valid},"
              f"{'|'.join(map(str, sorted(target_ions))) or '-'},"
              f"{ionization.get((26, ion), float('nan')):.10f}")
    fe5_complete = bool(fe.get(4)) and all(flags[index] for index in fe[4])
    fe5_ma = sum((targets[index] >= 0) if ma_version == 1 else
                 (offsets[index + 1] > offsets[index]) for index in fe.get(4, []))
    verdict = "AVAILABLE" if fe5_complete and fe5_ma == len(fe[4]) else "INSUFFICIENT"
    print(f"FeV_full_stage_verdict={verdict} "
          f"(FeV ma_rr sources {fe5_ma}/{len(fe.get(4, []))})")


if __name__ == "__main__":
    main()
