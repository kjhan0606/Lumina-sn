#!/usr/bin/env python3
"""Strong-line audit: carsus vs CMFGEN.

For each band + ion, look ONLY at strong lines (A_ul > 1e7, log10 weight ~ 7).
Report:
  N_carsus_strong, N_cmfgen_strong, N_matched (within ±0.1 Å)
  median ratio of MATCHED strong lines (A_cmfgen / A_carsus)
  top 3 strongest carsus lines + their CMFGEN counterpart (or NONE)
  top 3 strongest CMFGEN lines + their carsus counterpart (or NONE)
Goal: distinguish "wrong A_ul" from "wrong line set".
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
CARSUS = ROOT / "data/tardis_reference_strat6_higherL_aulboost_L19/line_list.csv"
CMFGEN = ROOT / "data/tardis_reference_cmfgen/line_list.csv"

BANDS = [
    ("UVbl_iron3", 3000, 3100),
    ("D_4700",     4600, 4800),
    ("F_6000",     5900, 6100),
]
IONS = [(14, 1), (24, 1), (24, 2), (26, 1), (26, 2), (27, 1), (27, 2),
        (28, 1), (28, 2)]
A_STRONG = 1e7
TOL = 0.1  # tighter: same transition

def main():
    print("Loading...", flush=True)
    car = pd.read_csv(CARSUS, usecols=["atomic_number","ion_number",
                                        "wavelength","A_ul"])
    cmf = pd.read_csv(CMFGEN, usecols=["atomic_number","ion_number",
                                        "wavelength","A_ul"])
    print(f"  carsus: {len(car):,}  CMFGEN: {len(cmf):,}\n")

    for band, lo, hi in BANDS:
        print(f"=== {band}  [{lo},{hi}]Å  ===")
        print(f"{'Z':>3} {'ion':>3} | {'N_car>1e7':>9} {'N_cmf>1e7':>9} "
              f"{'matched':>7} {'med ratio':>9}")
        for Z, ion in IONS:
            cs = car[(car.atomic_number==Z) & (car.ion_number==ion) &
                     (car.wavelength>=lo) & (car.wavelength<=hi) &
                     (car.A_ul>=A_STRONG)].copy()
            ms = cmf[(cmf.atomic_number==Z) & (cmf.ion_number==ion) &
                     (cmf.wavelength>=lo-TOL) & (cmf.wavelength<=hi+TOL) &
                     (cmf.A_ul>=A_STRONG)].copy()
            n_c, n_m = len(cs), len(ms)
            if n_c == 0 and n_m == 0:
                continue
            # match strong-to-strong within TOL
            matched = []
            ms_lams = ms.wavelength.values
            ms_aul = ms.A_ul.values
            for _, r in cs.iterrows():
                if len(ms_lams) == 0: break
                j = int(np.argmin(np.abs(ms_lams - r.wavelength)))
                if abs(ms_lams[j] - r.wavelength) <= TOL:
                    matched.append((r.wavelength, r.A_ul, ms_aul[j]))
            ratio_str = "n/a"
            if matched:
                ratios = [m[2]/m[1] for m in matched]
                ratio_str = f"{np.median(ratios):.3f}"
            print(f"{Z:>3} {ion:>3} | {n_c:>9} {n_m:>9} {len(matched):>7} {ratio_str:>9}")

            # show strongest 3 carsus, then strongest 3 CMFGEN
            if n_c > 0:
                top_c = cs.nlargest(3, "A_ul")
                for _, r in top_c.iterrows():
                    if len(ms_lams):
                        j = int(np.argmin(np.abs(ms_lams - r.wavelength)))
                        if abs(ms_lams[j]-r.wavelength) <= TOL:
                            note = f"CMF λ={ms_lams[j]:.2f} A={ms_aul[j]:.2e}"
                        else:
                            note = "no CMF strong match"
                    else:
                        note = "no CMF strong match"
                    print(f"    car λ={r.wavelength:7.2f} A={r.A_ul:.2e}  → {note}")
            if n_m > 0:
                top_m = ms.nlargest(3, "A_ul")
                cs_lams = cs.wavelength.values
                cs_aul = cs.A_ul.values
                for _, r in top_m.iterrows():
                    if len(cs_lams):
                        j = int(np.argmin(np.abs(cs_lams - r.wavelength)))
                        if abs(cs_lams[j]-r.wavelength) <= TOL:
                            note = f"car λ={cs_lams[j]:.2f} A={cs_aul[j]:.2e}"
                        else:
                            note = "no car strong match"
                    else:
                        note = "no car strong match"
                    print(f"    cmf λ={r.wavelength:7.2f} A={r.A_ul:.2e}  → {note}")
        print()


if __name__ == "__main__":
    main()
