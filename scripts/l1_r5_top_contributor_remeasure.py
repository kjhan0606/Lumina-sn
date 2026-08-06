#!/usr/bin/env python3
"""층 1 R-5 — 구 I2 상위 기여 이온의 `_ftos` 동일-원본 재측정.

구 I2(880,406선 중 75,075 불일치)의 상위 5 이온은 Ni III 28.4% · Ni II 21.9% ·
Co III 17.4% · Ca IV 11.3% · S III 7.4% (합 86%) 이다.  Fable L3 검수가 확정했듯
구 수치는 **교차-vintage A열 대 A열** 비교였고, `_ftos`에서는 덱과 CMFGEN 런이
같은 파일을 가리킨다(R-1).  따라서 여기서 재는 것은 전혀 다른 양이다.

★검수가 찾은 결정적 사실: CMFGEN 은 osc 파일의 **A 열을 읽지 않는다.**
genosc_v6.f:278 이 f 만 읽고, :313-317 이

    T1        = OPLIN/EMLIN*TWOHCSQ
    EINA(J,I) = T1 * f * STAT_WT(I)/STAT_WT(J) * (FEDGE(I)-FEDGE(J))**2

로 A 를 재계산한다.  COMMON/CONSTANTS/·COMMON/LINE/ 을 통해 상수가 들어오며
(new_main/cmfgen.f:120-122) T1 은 해석적으로 8*pi^2*e^2/(m_e*c^3)*1e30 으로 환원된다
(h 는 EMLIN 과 TWOHCSQ 사이에서 상쇄).  FEDGE 는 :205 에서 준위 에너지로부터
재계산되므로 파일 4열(6자리)이 아니라 E(cm^-1) 전정밀도를 쓴다.

비교자 6종 — 임계 1e-6 / 1e-9 / 1e-12, 대칭 상대차 |a-b|/max(|a|,|b|):

  C1  deck f   vs source f      임포트 충실도            기대 exact
  C2  deck A   vs source A열    Lumina 가 앵커한 것      기대 exact
  C3  deck A   vs A_CMFGEN      ★실제로 물리에 들어가는 괴리
  C4  source A vs A_CMFGEN      파일 자신의 A<->f 불일치 = C3 의 하한(잡음 바닥)
  C5  deck lam vs source lam    파장 임포트
  C6  A_CMFGEN(e_Lumina) vs A_CMFGEN(e_CMFGEN)   상수 선택만의 기여

판별식 (사전 등록):
  C2 > 0                -> Lumina 임포트 결함
  C2 = 0 이고 C3 ~= C4  -> 결함 아님.  A-우선(Lumina) 대 f-우선(CMFGEN) **선택**
  C2 = 0 이고 C3 >> C4  -> 조인·파장·g 등 다른 경로 결함
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cmfgen_parser import parse_osc  # noqa: E402

# --- CMFGEN 의 상수 (subs/phys_con.f) — 런이 실제로 쓰는 값 ---
C_CMF = 2.99792458e10          # :19  cm/s (exact)
ME_CMF = 9.1093837015e-28      # :59  g
E_CMF = 4.80320427e-10         # :67  esu
PI_CMF = 3.141592653589793238462643  # :204

# --- Lumina 덱 빌더가 쓴 값 (scripts/finalize_cmfgen_ref_npy.py:82) ---
E_DECK = 4.80320425e-10

# A = PRE * f * (g_lo/g_up) * nu^2   [nu in Hz]
PRE_CMF = 8.0 * PI_CMF**2 * E_CMF**2 / (ME_CMF * C_CMF**3)
PRE_DECK = 8.0 * PI_CMF**2 * E_DECK**2 / (ME_CMF * C_CMF**3)

# 구 I2 상위 기여 이온 (기여율은 구 덱 `_sivcaiv` 기준 — 분모로 쓰지 말 것)
IONS = [
    ("Ni III", 28, 2, "NICK/III/18oct00/nkiii_osc.dat", 28.4),
    ("Ni II", 28, 1, "NICK/II/18oct00/nkii_osc.dat", 21.9),
    ("Co III", 27, 2, "COB/III/18oct00/coiii_osc.dat", 17.4),
    ("Ca IV", 20, 3, "CA/IV/10apr99/osc_op_sp.dat", 11.3),
    ("S III", 16, 2, "SUL/III/3oct00/siiiosc_fin.dat", 7.4),
]

THRESHOLDS = (1e-6, 1e-9, 1e-12)


def reldiff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = np.maximum(np.abs(a), np.abs(b))
    out = np.zeros_like(den)
    nz = den > 0
    out[nz] = np.abs(a[nz] - b[nz]) / den[nz]
    return out


def stat(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    d = {
        "n": int(x.size),
        "exact_zero": int((x == 0.0).sum()),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }
    for t in THRESHOLDS:
        d[f"over_{t:g}"] = int((x > t).sum())
    return d


def measure(name, Z, ion, rel, share, atomic_root, levels, lines):
    src = atomic_root / rel
    res = {"source": rel, "old_i2_share_pct_sivcaiv": share,
           "source_exists": src.exists()}
    if not src.exists():
        res["error"] = "source missing"
        return res
    o = parse_osc(src)
    t = o.transitions
    res["source_transitions"] = int(len(t))
    res["source_levels"] = int(o.n_levels)

    sub = lines[(lines.atomic_number == Z) & (lines.ion_number == ion)]
    res["deck_lines"] = int(len(sub))
    if len(sub) == 0:
        res["error"] = "no deck lines"
        return res

    # 덱 준위 인덱스 기준 (0- 또는 1-based) 실측
    off = 1 if int(sub.level_number_lower.min()) == 0 else 0
    res["deck_level_index_offset"] = off

    key_d = (sub.level_number_lower.values.astype(np.int64) + off) * 1000000 \
        + (sub.level_number_upper.values.astype(np.int64) + off)
    key_s = t["i"].astype(np.int64) * 1000000 + t["j"].astype(np.int64)
    if pd.Series(key_d).duplicated().any():
        res["error"] = "deck key not unique"
        return res
    pos = pd.Series(np.arange(len(sub)), index=key_d)
    idx = pos.reindex(key_s).values
    m = np.isfinite(idx)
    idx = idx[m].astype(int)
    res["matched"] = int(m.sum())
    res["source_unmatched"] = int((~m).sum())
    res["deck_unmatched"] = int(len(sub) - len(set(idx.tolist())))

    ti, tj = t["i"][m], t["j"][m]
    f_src = np.abs(t["f"][m].astype(float))   # genosc_v6:305-309 음수 f 는 abs
    A_src = t["A"][m].astype(float)
    lam_src = t["lam_A"][m].astype(float)
    res["negative_f_in_source"] = int((t["f"][m].astype(float) < 0).sum())

    f_deck = sub.f_lu.values[idx].astype(float)
    A_deck = sub.A_ul.values[idx].astype(float)
    lam_deck = sub.wavelength_cm.values[idx].astype(float) * 1e8  # cm -> Angstrom

    # 원본은 lam 을 음수로도 적는다(규약).  v1 의 C5 는 그 부호를 재고 있었다 —
    # 물리가 아니므로 |lam| 로 비교한다.  n_neg 는 따로 남긴다.
    res["source_negative_lambda"] = int((lam_src < 0).sum())
    lam_src = np.abs(lam_src)

    # --- CMFGEN 메모리 A: 준위 에너지에서 nu 재계산 (genosc_v6:205,313-317) ---
    E = o.levels["E_cm"].astype(float)
    g = o.levels["g"].astype(float)
    nu = (E[tj - 1] - E[ti - 1]) * C_CMF            # Hz
    g_lo, g_up = g[ti - 1], g[tj - 1]
    A_cmf = PRE_CMF * f_src * (g_lo / g_up) * nu**2
    A_cmf_edeck = PRE_DECK * f_src * (g_lo / g_up) * nu**2

    # --- 공기/진공 분해 ---------------------------------------------------
    # 원본 헤더: "Wavelengths in air for lambda > 2000 Ang, else vacuum".
    # 준위 에너지 유래 lambda 는 항상 진공이므로 그 비가 굴절률을 실측한다.
    lam_lev = 1.0e8 / (E[tj - 1] - E[ti - 1])          # Angstrom, 진공
    air = lam_src > 2000.0
    res["air_band_lines"] = int(air.sum())
    res["vac_band_lines"] = int((~air).sum())
    r_lam = lam_src / lam_lev - 1.0
    res["refractive_index_measured"] = {
        "air_band_median": float(np.median(r_lam[air])) if air.any() else None,
        "vac_band_median": float(np.median(r_lam[~air])) if (~air).any() else None,
    }

    # 덱 f 가 실제로 어떻게 만들어졌는지 재구성(기계정밀도로 확인된 가설):
    #   f = A_file * (g_up/g_lo) * lam_file^2 * m_e*c/(8 pi^2 e_deck^2)
    inv = ME_CMF * C_CMF / (8.0 * PI_CMF**2 * E_DECK**2)
    f_recon = A_deck * (g_up / g_lo) * (lam_src * 1e-8) ** 2 * inv

    res["comparators"] = {
        "C1_deck_f_vs_source_f": stat(reldiff(f_deck, f_src)),
        "C2_deck_A_vs_source_Acol": stat(reldiff(A_deck, A_src)),
        "C3_deck_A_vs_A_cmfgen": stat(reldiff(A_deck, A_cmf)),
        "C4_source_Acol_vs_A_cmfgen": stat(reldiff(A_src, A_cmf)),
        "C5_deck_lam_vs_source_absLam": stat(reldiff(lam_deck, lam_src)),
        "C6_e_constant_only": stat(reldiff(A_cmf_edeck, A_cmf)),
        "C8_deck_f_reconstruction": stat(reldiff(f_recon, f_deck)),
        "C9_deck_f_vs_source_f_AIR_band": stat(reldiff(f_deck[air], f_src[air]))
        if air.any() else None,
        "C10_deck_f_vs_source_f_VAC_band": stat(reldiff(f_deck[~air], f_src[~air]))
        if (~air).any() else None,
    }
    # 부호 있는 편향 — 계통성 판별용(대칭 상대차는 부호를 지운다)
    res["signed_bias"] = {
        "f_deck_over_f_src_minus1_median_all": float(np.median(f_deck / f_src - 1.0)),
        "f_deck_over_f_src_minus1_median_air":
            float(np.median((f_deck / f_src - 1.0)[air])) if air.any() else None,
        "f_deck_over_f_src_minus1_median_vac":
            float(np.median((f_deck / f_src - 1.0)[~air])) if (~air).any() else None,
    }
    # 덱 g 가 원본 g 와 같은지 (조인 건전성)
    dl = levels[(levels.atomic_number == Z) & (levels.ion_number == ion)]
    if len(dl):
        gmap = dl.set_index("level_number").g
        g_deck_lo = gmap.reindex(ti - off).values.astype(float)
        res["comparators"]["C7_deck_g_vs_source_g"] = stat(reldiff(g_deck_lo, g_lo))
    return res


def verdict(ions):
    """사전 등록한 판별식을 기계적으로 적용.

    v1 은 A 축만 보고 'A-우선 대 f-우선 선택'이라 판정했으나 그것은 틀렸다.
    CMFGEN 이 소비하는 것은 f 이므로(genosc_v6.f:278-286 이 f 와 준위 인덱스만
    읽고 A·lam 토큰은 건너뛴다) f 축을 함께 봐야 한다.
    """
    out = {}
    for name, r in ions.items():
        c = r.get("comparators")
        if not c:
            out[name] = {"verdict": "NOT_MEASURED"}
            continue
        v = {}
        v["A_axis"] = ("EXACT_IMPORT"
                       if c["C2_deck_A_vs_source_Acol"]["over_1e-12"] == 0
                       else "LUMINA_A_IMPORT_DEFECT")
        air = c.get("C9_deck_f_vs_source_f_AIR_band")
        vac = c.get("C10_deck_f_vs_source_f_VAC_band")
        b = r.get("signed_bias", {})
        # 공기대역 f 편향이 진공대역보다 한 자릿수 이상 크고 한쪽 부호이면
        # A->f 역산에 공기파장을 쓴 것이다.
        if air and vac and air["median"] > 10.0 * max(vac["median"], 1e-30) \
                and b.get("f_deck_over_f_src_minus1_median_air") is not None \
                and b["f_deck_over_f_src_minus1_median_air"] < 0:
            v["f_axis"] = "LUMINA_AIR_WAVELENGTH_IN_A_TO_F_CONVERSION"
        elif air and vac:
            v["f_axis"] = "NO_AIR_VACUUM_SIGNATURE"
        else:
            v["f_axis"] = "SINGLE_BAND_ONLY"
        rec = c.get("C8_deck_f_reconstruction")
        v["deck_f_recipe_reconstructed"] = bool(rec and rec["max"] < 1e-9)
        out[name] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deck", type=Path,
                   default=ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos")
    p.add_argument("--atomic-root", type=Path,
                   default=Path("/gpfs/kjhan/cmfgen_21jun23/atomic"))
    p.add_argument("--out", type=Path,
                   default=ROOT / "validation/layer1/L1_R5_TOP_CONTRIBUTORS.json")
    a = p.parse_args()

    levels = pd.read_csv(a.deck / "levels.csv",
                         usecols=["atomic_number", "ion_number", "level_number", "g"])
    want = {(Z, i) for _, Z, i, _, _ in IONS}
    it = pd.read_csv(a.deck / "line_list.csv",
                     usecols=["atomic_number", "ion_number", "level_number_lower",
                              "level_number_upper", "wavelength_cm", "f_lu", "A_ul"],
                     chunksize=2_000_000)
    lines = pd.concat(
        [ch[[(z, i) in want for z, i in zip(ch.atomic_number, ch.ion_number)]]
         for ch in it],
        ignore_index=True)

    out = {
        "schema": "lumina-layer1-r5-top-contributors-v1",
        "purpose": "구 I2 상위 기여 이온을 `_ftos` 동일-원본 조건에서 재측정",
        "cmfgen_recompute": {
            "source": "subs/genosc_v6.f:278,290,313-317 + subs/phys_con.f",
            "formula": "A = 8*pi^2*e^2/(m_e*c^3) * f * g_lo/g_up * nu^2",
            "nu_from": "genosc_v6.f:205 — (E_ion - E_lev)*c, 준위표 전정밀도",
            "e_cmfgen": E_CMF, "e_lumina_deck": E_DECK,
            "m_e": ME_CMF, "c": C_CMF,
        },
        "thresholds": list(THRESHOLDS),
        "ions": {},
    }
    for name, Z, ion, rel, share in IONS:
        out["ions"][name] = measure(name, Z, ion, rel, share,
                                    a.atomic_root, levels, lines)
        print(f"[R5] {name} done", flush=True)
    out["verdict_per_ion"] = verdict(out["ions"])

    tot = {"matched": 0, "air_band_lines": 0, "vac_band_lines": 0,
           "over_1e-6_C1_f": 0, "over_1e-6_C2_A": 0, "over_1e-6_C3_A": 0,
           "over_1e-6_C4_A": 0}
    for r in out["ions"].values():
        c = r.get("comparators")
        if not c:
            continue
        tot["matched"] += r["matched"]
        tot["air_band_lines"] += r["air_band_lines"]
        tot["vac_band_lines"] += r["vac_band_lines"]
        tot["over_1e-6_C1_f"] += c["C1_deck_f_vs_source_f"]["over_1e-06"]
        tot["over_1e-6_C2_A"] += c["C2_deck_A_vs_source_Acol"]["over_1e-06"]
        tot["over_1e-6_C3_A"] += c["C3_deck_A_vs_A_cmfgen"]["over_1e-06"]
        tot["over_1e-6_C4_A"] += c["C4_source_Acol_vs_A_cmfgen"]["over_1e-06"]
    out["totals_5_ions"] = tot

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"totals": tot, "verdict": out["verdict_per_ion"]},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
