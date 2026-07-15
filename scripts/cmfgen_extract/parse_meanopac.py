#!/usr/bin/env python3
"""
parse_meanopac.py -- parse a CMFGEN MEANOPAC file (per-depth mean opacities and
optical-depth scales) into CSV.  MEANOPAC is written every iteration, so this is
usable on a PARTIAL (still-running) model as well as a converged one.

Columns in MEANOPAC (header row, whitespace table):
  R  I  Tau(Ross)  /\\Tau  Rat(Ross)  Chi(Ross) Chi(ross) Chi(Flux) Chi(es)
  Tau(Flux) Tau(es) Rat(Flux) Rat(es) Kappa(R)  V(km/s)

OUTPUT CSV: depth_index, r_1e10cm, v_kms, tau_ross, chi_ross, chi_flux, chi_es,
            kappa_ross.
Also (with --vmap OUT) writes a plain "depth_index v_kms" map that parse_jnu can
use to fill v_kms before an RVTJ exists.
"""
import sys, csv, argparse


def parse(path):
    rows = []
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) < 15:
                continue
            try:
                r = float(p[0]); idx = int(p[1])
            except ValueError:
                continue
            try:
                rows.append({
                    "depth_index": idx,
                    "r_1e10cm": r,
                    "tau_ross": float(p[2]),
                    "chi_ross": float(p[5]),
                    "chi_flux": float(p[7]),
                    "chi_es":   float(p[8]),
                    "kappa_ross": float(p[13]),
                    "v_kms":    float(p[14]),
                })
            except (ValueError, IndexError):
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meanopac")
    ap.add_argument("out_csv")
    ap.add_argument("--vmap", default=None, help="also write depth_index v_kms map")
    a = ap.parse_args()
    rows = parse(a.meanopac)
    cols = ["depth_index", "r_1e10cm", "v_kms", "tau_ross",
            "chi_ross", "chi_flux", "chi_es", "kappa_ross"]
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    if a.vmap:
        with open(a.vmap, "w") as f:
            for r in rows:
                f.write(f"{r['depth_index']} {r['v_kms']:.6E}\n")
    print(f"[parse_meanopac] {len(rows)} depths -> {a.out_csv}"
          + (f" (+ vmap {a.vmap})" if a.vmap else ""))


if __name__ == "__main__":
    main()
