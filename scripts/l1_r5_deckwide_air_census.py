#!/usr/bin/env python3
"""층 1 R-5 부수 측정 — 덱 전량 공기파장 오염 census (자기완결).

R-5 가 상위 기여 5이온에서 확정한 기전:
  덱 빌더가 `f = A_file*(g_up/g_lo)*lam_file^2*m_e*c/(8pi^2 e^2)` 로 f 를 역산하는데
  CMFGEN osc 파일은 헤더가 명시하듯 lambda>2000A 를 **공기파장**으로 적는다.
  A<->f 변환과 선 위치는 **진공** nu 를 요구하므로 그 구간이 오염된다.

여기서는 원본 osc 를 읽지 않고 **덱 안에서만** 판별한다:
  덱 nu (= c/wavelength_cm)  대  덱 준위에너지 유래 진공 nu
둘의 비가 굴절률(n_air-1 ~ 2.8e-4)이면 그 이온은 공기 규약 파일에서 왔다.

전 이온이 같지 않다.  vintage 별로 규약이 다르므로 이온 단위로 분류한다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
H_EV = 4.135667696e-15      # h [eV s]  (SI-2019)
C_CGS = 2.99792458e10
AIR_BAND_A = 2000.0         # CMFGEN 규약 경계
CONTAM_CUT = 1.0e-5         # 굴절률 서명 판정 임계 (진공 이온은 ~1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deck", type=Path,
                   default=ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos")
    p.add_argument("--out", type=Path,
                   default=ROOT / "validation/layer1/L1_R5_DECKWIDE_AIR_CENSUS.json")
    a = p.parse_args()

    lv = pd.read_csv(a.deck / "levels.csv",
                     usecols=["atomic_number", "ion_number", "level_number", "energy_eV"])
    key = (lv.atomic_number.astype("int64") * 1000000
           + lv.ion_number.astype("int64") * 100000
           + lv.level_number.astype("int64"))
    emap = pd.Series(lv.energy_eV.values, index=key.values)

    per_ion = {}
    n_join = n_air = n_vac = n_drop = 0
    it = pd.read_csv(a.deck / "line_list.csv",
                     usecols=["atomic_number", "ion_number", "level_number_lower",
                              "level_number_upper", "nu", "wavelength_cm"],
                     chunksize=2_000_000)
    for ch in it:
        base = ch.atomic_number.astype("int64") * 1000000 + ch.ion_number.astype("int64") * 100000
        El = emap.reindex((base + ch.level_number_lower.astype("int64")).values).values
        Eu = emap.reindex((base + ch.level_number_upper.astype("int64")).values).values
        ok = np.isfinite(El) & np.isfinite(Eu) & (Eu > El)
        n_drop += int((~ok).sum())
        nu_lev = (Eu[ok] - El[ok]) / H_EV
        r = ch.nu.values[ok] / nu_lev - 1.0
        lamA = ch.wavelength_cm.values[ok] * 1e8
        air = lamA > AIR_BAND_A
        Z = ch.atomic_number.values[ok]
        io = ch.ion_number.values[ok]
        n_join += int(ok.sum()); n_air += int(air.sum()); n_vac += int((~air).sum())
        for z, i in set(zip(Z.tolist(), io.tolist())):
            sel = (Z == z) & (io == i)
            d = per_ion.setdefault(f"{z}_{i}",
                                   {"Z": int(z), "ion_number": int(i),
                                    "air_lines": 0, "vac_lines": 0,
                                    "_air_r": [], "_vac_r": []})
            sa, sv = sel & air, sel & ~air
            d["air_lines"] += int(sa.sum()); d["vac_lines"] += int(sv.sum())
            if sa.any():
                d["_air_r"].append(float(np.median(r[sa])))
            if sv.any():
                d["_vac_r"].append(float(np.median(r[sv])))

    contaminated_air = contaminated_total = 0
    n_contam_ions = 0
    for k, d in per_ion.items():
        d["air_band_median_r"] = float(np.median(d.pop("_air_r"))) if d["_air_r"] else None
        d["vac_band_median_r"] = float(np.median(d.pop("_vac_r"))) if d["_vac_r"] else None
        d["contaminated"] = bool(d["air_band_median_r"] is not None
                                 and d["air_band_median_r"] > CONTAM_CUT)
        if d["contaminated"]:
            n_contam_ions += 1
            contaminated_air += d["air_lines"]
            contaminated_total += d["air_lines"] + d["vac_lines"]
        d["implied_shift_km_s"] = (abs(d["air_band_median_r"]) * C_CGS / 1e5
                                   if d["air_band_median_r"] is not None else None)

    out = {
        "schema": "lumina-layer1-r5-deckwide-air-census-v1",
        "criterion": ("이온의 lambda>2000A 대역에서 deck_nu/levelenergy_nu-1 이 "
                      f"{CONTAM_CUT:g} 를 넘으면 공기 규약 파일에서 온 것"),
        "lines_joined": n_join, "lines_dropped_no_level": n_drop,
        "air_band_lines": n_air, "vac_band_lines": n_vac,
        "ions_total": len(per_ion), "ions_contaminated": n_contam_ions,
        "contaminated_air_band_lines": contaminated_air,
        "contaminated_ion_all_lines": contaminated_total,
        "contaminated_air_fraction_of_deck": contaminated_air / max(n_join, 1),
        "per_ion": dict(sorted(per_ion.items(),
                               key=lambda kv: -kv[1]["air_lines"])),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "per_ion"},
                     indent=1, ensure_ascii=False))
    print("\n오염 이온 (공기대역 선 수 순):")
    for k, d in out["per_ion"].items():
        if d["contaminated"]:
            print("  Z=%2d ion=%d  air=%7d vac=%7d  r=%.3e  shift=%.1f km/s"
                  % (d["Z"], d["ion_number"], d["air_lines"], d["vac_lines"],
                     d["air_band_median_r"], d["implied_shift_km_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
