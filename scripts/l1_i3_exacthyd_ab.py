#!/usr/bin/env python3
"""I3 — 수소꼴 fit(타입 2/3/8/9) 대용 대 정확 평가 A/B.

베이커가 매 빌드마다 스스로 경고해 온 결함:
  "[bakefix] WARNING: N levels carry the known-wrong params[0]-as-sigma_0 stand-in"
타입 2/3/8 은 수소꼴이라 params[0] 이 **주양자수 n**(또는 타입 3 의 스케일)이고
단면적이 아니다. 그것을 sigma_0[Mb] 로 읽는 것은 조작된 수다. 타입 9 는 Verner fit.

BAKEFIX2 가 네 타입의 정확 평가기를 newsubs/sub_phot_gen.f 에서 문장 단위로 이식해
뒀으나 게이트 CMFGEN_EXACT_HYD 가 꺼져 있었다. 이 스크립트는 켠 덱과 끈 덱을 대조한다.

사전 등록한 기대:
  - 변하는 준위 = 경고가 센 수와 일치해야 한다
  - 변하지 않는 준위(타입 1/7/20/21/22)는 **비트 동일**이어야 한다(단일 변수 변경)
  - 대용은 임계에서 과소로 알려져 있다(주석 기록: legacy/CMFGEN = Co III 0.85 ·
    Fe II 0.41 · Co II 0.41 · Sc I 0.09) ⟹ exact/standin > 1 이 다수일 것
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def read_sigma(path: Path):
    with open(path, "rb") as f:
        magic, ver = struct.unpack("<II", f.read(8))
        nl, nb = struct.unpack("<ii", f.read(8))
        lo, hi = struct.unpack("<dd", f.read(16))
        has = np.frombuffer(f.read(nl), dtype="i1").copy()
        f.read((8 - (nl % 8)) % 8)
        sig = np.frombuffer(f.read(), dtype="f8").reshape(nl, nb)
    return sig, has, nb, lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--standin", type=Path,
                   default=ROOT / "data/tardis_reference_toy06_19p48d_ophys")
    p.add_argument("--exact", type=Path,
                   default=ROOT / "data/tardis_reference_toy06_19p48d_ophys_exacthyd")
    p.add_argument("--out", type=Path,
                   default=ROOT / "validation/layer1/L1_I3_EXACTHYD_AB.json")
    a = p.parse_args()

    sa, ha, nb, lo, hi = read_sigma(a.standin / "cmfgen_sigma_bf.bin")
    sb, hb, nb2, lo2, hi2 = read_sigma(a.exact / "cmfgen_sigma_bf.bin")
    if (sa.shape != sb.shape) or (nb != nb2) or (lo, hi) != (lo2, hi2):
        raise SystemExit(f"grid/shape mismatch: {sa.shape} vs {sb.shape}")

    dlog = (np.log(hi) - np.log(lo)) / nb
    edges = lo * np.exp(np.arange(nb + 1) * dlog)
    nu_c = np.sqrt(edges[:-1] * edges[1:])
    dnu = np.diff(edges)
    # Gamma 적분의 형상 가중치 (J 없이): w_b = dnu_b / nu_b
    w = dnu / nu_c

    lev = pd.read_csv(a.exact / "levels.csv",
                      usecols=["atomic_number", "ion_number", "level_number"])
    if len(lev) != sa.shape[0]:
        raise SystemExit(f"levels.csv {len(lev)} != sigma rows {sa.shape[0]}")

    changed = ~np.all(sa == sb, axis=1)
    out = {
        "schema": "lumina-layer1-i3-exacthyd-ab-v1",
        "standin_deck": str(a.standin), "exact_deck": str(a.exact),
        "n_levels": int(sa.shape[0]), "n_bins": nb,
        "levels_changed": int(changed.sum()),
        "levels_bit_identical": int((~changed).sum()),
        "has_cmfgen_changed": int((ha != hb).sum()),
    }

    # Gamma 형상 적분 비 (exact / standin)
    ga = sa @ w
    gb = sb @ w
    ok = changed & (ga > 0) & (gb > 0)
    r = gb[ok] / ga[ok]
    out["gamma_shape_ratio_exact_over_standin"] = {
        "n": int(ok.sum()),
        "median": float(np.median(r)), "p05": float(np.percentile(r, 5)),
        "p95": float(np.percentile(r, 95)),
        "min": float(r.min()), "max": float(r.max()),
        "frac_gt_1": float((r > 1).mean()),
    }
    out["changed_levels_now_zero"] = int((changed & (gb <= 0) & (ga > 0)).sum())
    out["changed_levels_were_zero"] = int((changed & (ga <= 0) & (gb > 0)).sum())

    per_ion = {}
    for (Z, io), grp in lev.groupby(["atomic_number", "ion_number"]):
        idx = grp.index.to_numpy()
        c = changed[idx]
        if not c.any():
            continue
        m = idx[c]
        good = (ga[m] > 0) & (gb[m] > 0)
        rr = gb[m][good] / ga[m][good]
        per_ion[f"Z{int(Z)}_ion{int(io)}"] = {
            "levels_total": int(len(idx)), "levels_changed": int(c.sum()),
            "gamma_ratio_median": float(np.median(rr)) if rr.size else None,
            "gamma_ratio_p95": float(np.percentile(rr, 95)) if rr.size else None,
            "now_zero": int(((ga[m] > 0) & (gb[m] <= 0)).sum()),
        }
    out["per_ion"] = dict(sorted(per_ion.items(),
                                 key=lambda kv: -kv[1]["levels_changed"]))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    head = {k: v for k, v in out.items() if k != "per_ion"}
    print(json.dumps(head, indent=1, ensure_ascii=False))
    print("\n이온별 상위 12 (변경 준위 수 순):")
    for k, v in list(out["per_ion"].items())[:12]:
        print("  %-12s changed=%-5d/%-5d  Γ비 median=%-8s p95=%-8s now_zero=%d"
              % (k, v["levels_changed"], v["levels_total"],
                 ("%.3f" % v["gamma_ratio_median"]) if v["gamma_ratio_median"] else "-",
                 ("%.3f" % v["gamma_ratio_p95"]) if v["gamma_ratio_p95"] else "-",
                 v["now_zero"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
