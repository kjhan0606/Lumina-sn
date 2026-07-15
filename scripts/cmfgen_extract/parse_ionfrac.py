#!/usr/bin/env python3
"""
parse_ionfrac.py -- convert dispgen IF_<species> WXY dumps (per-element ionization
fractions vs depth index) into a tidy CSV matching Lumina's ion_pops / the
bench_ionfrac_at.py convention: columns  Z, element, ion_stage, depth_index,
v_kms, log_frac, frac.

INPUT: a dispgen_ionfrac.sh manifest with lines  "<wxy_file> <SPECIES> <Z>".
Each WXY file (GRAMON WXY, LAM_ST==LAM_END branch) has:
    line1 : n_plots  (= number of ionization stages dumped for that element)
    line2 : points-per-plot (= ND, all equal)
    rows  : n_plots (X,Y) pairs; X = depth index (dispgen 'XN' abscissa),
            Y = log10( N_ion / N_species ).
Stages are dumped in ascending ionization order; the lowest included stage for
every toy06 species is III, so plot #k -> ionization stage (k+2) [III=3, IV=4,...].

v_kms per depth is joined from rvtj.csv when given.
"""
import sys, csv, argparse, os

ROMAN = {3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}
LOWEST_STAGE = 3   # III for all toy06 species


def read_wxy(path):
    with open(path) as f:
        toks = f.read().split()
    i = 0
    npl = int(float(toks[i])); i += 1
    npts = [int(float(toks[i + j])) for j in range(npl)]; i += npl
    nmax = max(npts)
    data = toks[i:]
    ncol = 2 * npl
    if len(data) < nmax * ncol:
        nmax = len(data) // ncol
    X = [0.0] * nmax
    Y = [[0.0] * nmax for _ in range(npl)]
    for r in range(nmax):
        b = r * ncol
        X[r] = float(data[b].replace("D", "E"))
        for p in range(npl):
            Y[p][r] = float(data[b + 2 * p + 1].replace("D", "E"))
    return npl, npts, X, Y


def load_vmap(rvtj_csv):
    v = {}
    if rvtj_csv and os.path.exists(rvtj_csv):
        with open(rvtj_csv) as f:
            for row in csv.DictReader(f):
                try:
                    v[int(row["depth_index"])] = float(row["v_kms"])
                except (KeyError, ValueError):
                    pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="dispgen_ionfrac.sh manifest")
    ap.add_argument("out_csv")
    ap.add_argument("--rvtj-csv", default=None)
    a = ap.parse_args()

    vmap = load_vmap(a.rvtj_csv)
    rows = []
    with open(a.manifest) as f:
        entries = [ln.split() for ln in f if ln.split()]
    for ent in entries:
        wxy, species, Z = ent[0], ent[1], int(ent[2])
        if not os.path.exists(wxy):
            print(f"[parse_ionfrac] missing {wxy}", file=sys.stderr); continue
        npl, npts, X, Y = read_wxy(wxy)
        for p in range(npl):
            stage = LOWEST_STAGE + p
            for r in range(len(X)):
                d = int(round(X[r]))
                logf = Y[p][r]
                frac = 10.0 ** logf if logf > -300 else 0.0
                rows.append((Z, species, stage, ROMAN.get(stage, str(stage)),
                             d, vmap.get(d, ""), f"{logf:.6E}", f"{frac:.6E}"))

    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Z", "element", "ion_stage", "ion_roman",
                    "depth_index", "v_kms", "log_frac", "frac"])
        w.writerows(rows)
    print(f"[parse_ionfrac] {len(entries)} elements -> {len(rows)} rows in {a.out_csv}")


if __name__ == "__main__":
    main()
