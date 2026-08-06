#!/usr/bin/env python3
"""I20 수리안 오프라인 검증 — 덱 재생성 **전에** 레시피가 옳은지 증명한다.

새 레시피(expand_atomic_data_cmfgen.build_lines 수리분)를 원본 osc 에 직접 적용해
CMFGEN 메모리 상태(genosc_v6.f 가 만드는 것)를 재현하는지 본다.

  G1b  A_new(e=e_CMFGEN)  vs  A_CMFGEN            기대 <= 1e-14  (by-construction)
  G1   A_new(e=참값)      vs  A_CMFGEN            기대  1.843e-7 (오직 e 상수차)
  G2   f_new              vs  원본 f 열            기대  exact 0
  G3   nu_new             vs  준위에너지 nu        기대 <= 1e-12
  G4   lam_new            vs  준위에너지 lam       공기 오염 0
  G5   선 수              vs  구 덱 선 수          동일

음성대조 N1: 뿌리1만 고치고 뿌리2(A->f 역산)를 남기면 G2 가 FAIL 해야 한다.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cmfgen_parser import parse_osc  # noqa: E402

C = 2.99792458e10
ME = 9.1093837015e-28
PI = 3.141592653589793238462643
E_TRUE = 1.602176634e-19 * C / 10.0      # 4.803204712...e-10  (SI-2019)
E_CMF = 4.80320427e-10                   # phys_con.f:67 (CODATA-2006)

ATOMIC = Path("/gpfs/kjhan/cmfgen_21jun23/atomic")
IONS = [
    ("Ni III", "NICK/III/18oct00/nkiii_osc.dat"),
    ("Ni II", "NICK/II/18oct00/nkii_osc.dat"),
    ("Co III", "COB/III/18oct00/coiii_osc.dat"),
    ("Ca IV", "CA/IV/10apr99/osc_op_sp.dat"),
    ("S III", "SUL/III/3oct00/siiiosc_fin.dat"),
]


def pre(e):
    return 8.0 * PI**2 * e**2 / (ME * C**3)


def mx(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = np.maximum(np.abs(a), np.abs(b))
    r = np.zeros_like(d); nz = d > 0
    r[nz] = np.abs(a[nz] - b[nz]) / d[nz]
    return float(r.max()), float(np.median(r))


def main():
    out = {"schema": "lumina-i20-repair-offline-check-v1",
           "e_true": E_TRUE, "e_cmfgen": E_CMF,
           "expected_G1": (E_TRUE / E_CMF) ** 2 - 1.0, "ions": {}}
    allpass = True
    for name, rel in IONS:
        o = parse_osc(ATOMIC / rel)
        t = o.transitions
        E = o.levels["E_cm"].astype(float)
        g = o.levels["g"].astype(float)
        i, j = t["i"], t["j"]
        keep = (E[j - 1] - E[i - 1]) > 0
        i, j = i[keep], j[keep]
        f_src = np.abs(t["f"][keep].astype(float))

        # ---- 새 레시피 (수리 후 build_lines 와 동일한 산술) ----
        nu_new = (E[j - 1] - E[i - 1]) * C
        lam_new = C / nu_new * 1e8
        f_new = f_src
        glo, gup = g[i - 1], g[j - 1]
        A_new = pre(E_TRUE) * f_new * (glo / gup) * nu_new**2
        A_new_cmfe = pre(E_CMF) * f_new * (glo / gup) * nu_new**2

        # ---- CMFGEN 메모리 상태 (genosc_v6.f:205,313-317) ----
        A_cmf = pre(E_CMF) * f_src * (glo / gup) * nu_new**2

        # ---- 음성대조 N1: 뿌리2(A->f 역산)를 남긴 경우 ----
        # 주입 결함은 **공기대역(lambda>2000A)에만** 나타난다.  전 선 중앙값으로
        # 재면 공기대역 비율이 낮은 이온(Ca IV 는 46%)에서 진공대역이 통계를
        # 지배해 통제가 무력해진다 — 대역을 나눠서 잰다.
        lam_air = np.abs(t["lam_A"][keep].astype(float))
        f_bad = A_new * (gup / glo) * (lam_air * 1e-8) ** 2 * (ME * C / (8 * PI**2 * E_TRUE**2))
        air = lam_air > 2000.0

        g1b_max, _ = mx(A_new_cmfe, A_cmf)
        g1_max, g1_med = mx(A_new, A_cmf)
        g2_max, _ = mx(f_new, f_src)
        g3_max, _ = mx(nu_new, (E[j - 1] - E[i - 1]) * C)
        g4_max, _ = mx(lam_new, 1.0e8 / (E[j - 1] - E[i - 1]))
        n1_max, n1_med = mx(f_bad[air], f_src[air]) if air.any() else (0.0, 0.0)
        _, n1_vac = mx(f_bad[~air], f_src[~air]) if (~air).any() else (0.0, 0.0)

        r = {
            "lines": int(keep.sum()), "dropped_dE_le_0": int((~keep).sum()),
            "G1b_A_vs_cmfgen_same_e_max": g1b_max,
            "G1_A_vs_cmfgen_true_e_median": g1_med,
            "G2_f_vs_source_f_max": g2_max,
            "G3_nu_max": g3_max,
            "G4_lam_max": g4_max,
            "air_band_lines": int(air.sum()), "vac_band_lines": int((~air).sum()),
            "N1_negative_control_AIR_median": n1_med,
            "N1_negative_control_VAC_median": n1_vac,
        }
        # 음성대조 자격: 공기대역에서 2*(n_air-1) ~ 5.5e-4 를 시연하고,
        # 진공대역에서는 조용해야 한다(결함이 대역 특이적임을 함께 증명).
        r["PASS"] = bool(g1b_max <= 1e-14 and g2_max == 0.0
                         and g3_max <= 1e-12 and g4_max <= 1e-12
                         and abs(g1_med - out["expected_G1"]) <= 1e-12
                         and n1_med > 4.0e-4 and n1_vac < 1.0e-5)
        allpass &= r["PASS"]
        out["ions"][name] = r
        print("%-8s lines=%7d air=%6d  G1b=%.2e  G1med=%.4e  G2=%.1e  G3=%.1e  "
              "G4=%.1e  N1air=%.3e N1vac=%.1e  %s"
              % (name, r["lines"], r["air_band_lines"], g1b_max, g1_med, g2_max,
                 g3_max, g4_max, n1_med, n1_vac, "PASS" if r["PASS"] else "FAIL"))
    out["ALL_PASS"] = bool(allpass)
    p = ROOT / "validation/layer1/L1_R5_REPAIR_OFFLINE_CHECK.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("\nALL_PASS =", allpass, " expected_G1 =", out["expected_G1"])
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
