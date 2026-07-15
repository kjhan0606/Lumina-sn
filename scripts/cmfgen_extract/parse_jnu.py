#!/usr/bin/env python3
"""
parse_jnu.py -- convert a plt_jh WXY dump of J(lambda) at every depth into a CSV.

INPUT  (produced by jnu_dump.sh driving plt_jh.exe; format = pgplt/gramon_pgplot.f
        WXY option, LAM_ST==LAM_END branch, gramon_pgplot.f:2415-2436):
    line 1            : NP  (number of plots = number of depths dumped)
    line 2            : NP integers = point count per plot (all = NCF)
    then NCF rows     : NP pairs  (lambda_Angstrom  J_nu)   per row,
                        format '(2X,ES14.7,ES14.6)' per pair; the lambda column is
                        identical across all plots (shared CMF frequency grid).
    (VERIFIED live against the running toy06_2d EDDFACTOR: NP=1 and NP=3 dumps.)

OUTPUT CSV columns:  depth_index, v_kms, lambda_A, J_nu
    v_kms per depth is joined from rvtj.csv (parse_rvtj.py) when available, else
    from a --vmap file (depth_index,v_kms), else left blank.

By default only lines in a wavelength band are written (--lam-min/--lam-max,
default 900-1500 A per the task) to keep the CSV tractable; pass --all for the
full grid.
"""
import sys, csv, argparse, os


def read_wxy(path):
    with open(path) as f:
        toks = f.read().split()
    idx = 0
    npl = int(float(toks[idx])); idx += 1
    npts = [int(float(toks[idx + i])) for i in range(npl)]; idx += npl
    nmax = max(npts)
    data = toks[idx:]
    ncol = 2 * npl
    # data is row-major: nmax rows x ncol values
    if len(data) < nmax * ncol:
        # tolerate a short/truncated final row
        nmax = len(data) // ncol
    lam = [0.0] * nmax
    J = [[0.0] * nmax for _ in range(npl)]
    for r in range(nmax):
        base = r * ncol
        lam[r] = float(data[base].replace("D", "E"))
        for p in range(npl):
            J[p][r] = float(data[base + 2 * p + 1].replace("D", "E"))
    return npl, npts, lam, J


def load_vmap(rvtj_csv, vmap):
    v = {}
    if rvtj_csv and os.path.exists(rvtj_csv):
        with open(rvtj_csv) as f:
            for row in csv.DictReader(f):
                try:
                    v[int(row["depth_index"])] = float(row["v_kms"])
                except (KeyError, ValueError):
                    pass
    elif vmap and os.path.exists(vmap):
        with open(vmap) as f:
            for ln in f:
                p = ln.replace(",", " ").split()
                if len(p) >= 2:
                    try:
                        v[int(float(p[0]))] = float(p[1])
                    except ValueError:
                        pass
    return v


def load_manifest(path):
    """-> list of (wxy_file, first_depth) from a jnu_dump manifest."""
    out = []
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 2:
                out.append((p[0], int(p[1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wxy", nargs="?", help="single plt_jh WXY dump file")
    ap.add_argument("out_csv")
    ap.add_argument("--manifest", default=None,
                    help="jnu_dump manifest (merges batched dumps)")
    ap.add_argument("--depth-base", type=int, default=1,
                    help="global depth index of plot #1 in a single file")
    ap.add_argument("--rvtj-csv", default=None, help="rvtj.csv for depth->v_kms")
    ap.add_argument("--vmap", default=None, help="alt: depth_index,v_kms file")
    ap.add_argument("--lam-min", type=float, default=900.0)
    ap.add_argument("--lam-max", type=float, default=1500.0)
    ap.add_argument("--all", action="store_true", help="write full grid")
    a = ap.parse_args()

    if a.manifest:
        batches = load_manifest(a.manifest)
    elif a.wxy:
        batches = [(a.wxy, a.depth_base)]
    else:
        ap.error("provide a WXY file or --manifest")

    vmap = load_vmap(a.rvtj_csv, a.vmap)
    n_written = 0
    ndepth = 0
    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth_index", "v_kms", "lambda_A", "J_nu"])
        for wxy, base in batches:
            npl, npts, lam, J = read_wxy(wxy)
            ndepth = max(ndepth, base + npl - 1)
            for r in range(len(lam)):
                L = lam[r]
                if not a.all and not (a.lam_min <= L <= a.lam_max):
                    continue
                for p in range(npl):
                    d = base + p
                    w.writerow([d, vmap.get(d, ""), f"{L:.7E}", f"{J[p][r]:.6E}"])
                    n_written += 1
    band = "full grid" if a.all else f"band {a.lam_min}-{a.lam_max} A"
    print(f"[parse_jnu] {len(batches)} batch(es), {ndepth} depths -> "
          f"{n_written} rows in {a.out_csv} ({band})")


if __name__ == "__main__":
    main()
