#!/usr/bin/env python3
"""Si II 저장 단면적을 로컬 CMFGEN 원자료와 대조한다(생산 변경 없음).

기존 baker의 함수를 재사용하되 main/writer는 호출하지 않는다.
공명 단면적 계보 확인은 별도 DR과의 중복량 측정과 다르다.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import struct

import numpy as np
import pandas as pd

import bake_cmfgen_sigma_bf_carsus as BA
import det_si_trace_readout as TR
import expand_atomic_data_cmfgen as AVG


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--bin-average", action="store_true",
                    help="현재 단면적의 구간 평균 규약으로 대조; 기본은 구형 중심점 표본 대조")
    args = ap.parse_args()
    tr = args.run / "si_trace"
    sm = [r for r in TR.read_trace(tr / "si10_slmap.csv")
          if r["call"] == "36" and r["upper_ion"] == "0"]
    df = pd.DataFrame([{"global_idx": int(r["global_level"]), "energy_eV": float(r["E_eV"]),
                        "g": int(r["g"]), "level_num": int(r["level_num"])} for r in sm])
    if len(df) != 125 or not df.global_idx.is_unique:
        raise ValueError("expected 125 unique lower levels")
    raw = BA.CMFGEN_ROOT / "SIL/II/19apr23"
    osc_path, phot_path = raw / "osc_data", raw / "phot_data_A"
    osc, phot = BA.parse_osc(osc_path), BA.parse_phot(phot_path)
    mapping = BA.match_levels_to_cmfgen(df, osc.levels)
    exact, term = defaultdict(list), defaultdict(list)
    normalize = lambda s: s.strip().lower()
    term_name = lambda s: re.sub(r"\[[^\[\]]*\]\s*$", "", normalize(s))
    for entry in phot.entries:
        exact[normalize(entry.config)].append(entry)
        term[term_name(entry.config)].append(entry)
    sigma_path = args.run / "input/model/cmfgen_sigma_bf.bin"
    rows = []
    with sigma_path.open("rb") as f:
        magic, version, nl, nf = struct.unpack("<IIii", f.read(16))
        nu_min, nu_max = struct.unpack("<dd", f.read(16))
        if magic != 0x434D4644 or version != 1:
            raise ValueError("unknown sigma format")
        flags = f.read(nl)
        offset = 32 + nl + (-nl) % 8
        nu = nu_min * np.exp((np.arange(nf) + 0.5) * np.log(nu_max / nu_min) / nf)
        edges = nu_min * np.exp(np.arange(nf + 1) * np.log(nu_max / nu_min) / nf)
        for r in df.itertuples():
            cfg = mapping.get(r.global_idx)
            entries = (exact.get(normalize(cfg)) or term.get(term_name(cfg), [])) if cfg else []
            threshold = (osc.ionization_eV - r.energy_eV) * BA.EV_TO_ERG / BA.H_CGS
            if args.bin_average and cfg:
                raw_level = next(v for v in osc.levels if v["config"] == cfg)
                threshold = (osc.ionization_eV - float(raw_level["E_cm"]) * 1.239841984e-4) * BA.EV_TO_ERG / BA.H_CGS
                rebuilt = np.zeros(nf)
                for entry in entries:
                    model = AVG._sigma_model(entry.cs_type, entry.energy, entry.sigma_Mb, threshold,
                                             zion=float(osc.z_screen),
                                             nef=AVG._cmfgen_nef(14, float(osc.z_screen), threshold))
                    if model is not None:
                        fn, knots, start, _ = model
                        rebuilt += AVG._bin_average_sigma(fn, knots, edges, np.diff(edges), start)
            else:
                rebuilt = sum((BA.bake_one_level(e, nu, threshold) for e in entries), np.zeros(nf))
            f.seek(offset + r.global_idx * nf * 8)
            saved = np.frombuffer(f.read(nf * 8), dtype="<f8")
            scale = max(float(np.max(abs(saved))), float(np.max(abs(rebuilt))))
            relative = float(np.max(abs(saved - rebuilt))) / scale if scale else 0.0
            rows.append({"level_num": r.level_num, "global_level": r.global_idx, "config": cfg,
                         "saved_flag": flags[r.global_idx], "raw_types": [e.cs_type for e in entries],
                         "relative_max_difference": relative,
                         "compatible_1e_8": relative <= 1e-8})
    source = Path("/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/lumina_plasma.c")
    text = source.read_text()
    table = text[text.index("static const DRCoefficient DR_TABLE[]") : text.index("#define DR_N_ENTRIES")]
    first = re.search(r"\{14,\s*2,\s*5,\s*\{([^}]+)\},\s*\{([^}]+)\},\s*DR_SOURCE_(\w+)", table)
    if first is None:
        raise ValueError("Si III first DR entry not found")
    coeff, energy = ([float(v) for v in first[k].split(",")] for k in (1, 2))
    call = next(r for r in TR.read_trace(tr / "si10_calls.csv") if r["call"] == "36")
    T = float(call["T_e"])
    alpha = math.fsum(c * math.exp(-e / T) for c, e in zip(coeff, energy)) / T ** 1.5
    dr = next(r for r in TR.read_trace(tr / "si10_ops.csv") if r["call"] == "36" and r["site"] == "8")
    paths = [osc_path, phot_path, sigma_path, source, tr / "si10_slmap.csv", tr / "si10_calls.csv",
             tr / "si10_ops.csv", Path(BA.__file__), Path(AVG.__file__), Path(TR.__file__), Path(__file__)]
    print(json.dumps({"job": os.environ.get("SLURM_JOB_ID"), "scope": "Si II raw-to-saved sigma and DR source only",
                      "sampling": "bin-average" if args.bin_average else "legacy-bin-centre",
                      "exact_hydrogenic": AVG.SIGMA_EXACT_HYD,
                      "matched_levels": len(mapping), "saved_positive": sum(r["saved_flag"] != 0 for r in rows),
                      "compatible_rows_1e_8": sum(r["compatible_1e_8"] for r in rows),
                      "saved_positive_with_type20": sum(r["saved_flag"] != 0 and 20 in r["raw_types"] for r in rows),
                      "DR_first_source": first[3], "DR_alpha_fit": alpha, "DR_alpha_trace": float(dr["x1"]),
                      "DR_fit_to_trace": alpha / float(dr["x1"]), "rows": rows,
                      "sha256": {str(p): TR.sha(p) for p in paths}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
