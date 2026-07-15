#!/usr/bin/env python3
"""
parse_rvtj.py -- parse a CMFGEN RVTJ file (r/v/T_e/n_e/opacities/moments and
per-element number densities per depth) into a wide CSV.

RVTJ FORMAT  (written by new_main/cmfgen_sub.f:4398-4468, plain ASCII):
  header  : ' Output format date:' ... ' ND:  NN' ... ' Species naming convention:'
  then, from ' Radius (10^10 cm)' onward, repeated blocks of
        <label line>            e.g.  ' Velocity (km/s)'
        <ND values, 8 per line> (R,V in ES18.10; the rest in 1P8E16.7)
  Vector order: Radius, Velocity, dlnV/dlnr-1, Electron density, Temperature
  (10^4K), Grey temperature, Heating: radioactive decay, Rosseland/Flux/Planck/
  Absorption Mean Opacity, J/H/K moment, Atom Density, Ion Density, Mass Density,
  Clumping Factor, then '<Element> Density' for each species present.

OUTPUT CSV: depth_index, then one column per vector (sanitized label), plus a
derived  T_e_K = Temperature*1e4  and  v_kms alias, so downstream tools (and
parse_jnu's depth->v join) can read it directly.
"""
import sys, re, csv, argparse

ANCHOR = "Radius"          # first vector label; header ends here
NUM_RE = re.compile(r"^[\s0-9EeDd+\-.]+$")


def sanitize(label):
    s = label.strip().strip("'").strip()
    s = re.sub(r"\(.*?\)", "", s)          # drop units in parens
    s = s.replace(":", "").strip()
    s = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_").lower()
    return s or "col"


def parse(path):
    with open(path) as f:
        lines = f.readlines()
    # ND
    nd = None
    for ln in lines:
        m = re.match(r"\s*ND:\s*(\d+)", ln)
        if m:
            nd = int(m.group(1)); break
    if nd is None:
        raise ValueError("RVTJ: no 'ND:' header found")

    # find anchor
    start = None
    for i, ln in enumerate(lines):
        if ANCHOR in ln and ":" not in ln.split(ANCHOR)[0]:
            start = i; break
        if ln.strip().startswith(ANCHOR):
            start = i; break
    if start is None:
        raise ValueError("RVTJ: no 'Radius' vector block found")

    vectors = {}       # label -> list of floats
    order = []
    cur, buf = None, []

    def flush():
        if cur is not None:
            vectors[cur] = buf[:nd]
    i = start
    while i < len(lines):
        ln = lines[i].rstrip("\n")
        if ln.strip() == "":
            i += 1; continue
        toks = ln.replace("D", "E").replace("d", "e").split()
        is_num = bool(toks) and all(NUM_RE.match(t) or _isfloat(t) for t in toks) \
            and any(c.isdigit() for c in ln) and not re.search(r"[A-Za-z]{2,}", ln)
        if is_num:
            for t in toks:
                try:
                    buf.append(float(t))
                except ValueError:
                    pass
        else:
            # new label
            flush()
            name = sanitize(ln)
            # de-duplicate element-density labels
            base = name; k = 2
            while name in vectors:
                name = f"{base}_{k}"; k += 1
            cur = name; order.append(name); buf = []
        i += 1
    flush()
    return nd, order, vectors


def _isfloat(t):
    try:
        float(t); return True
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rvtj")
    ap.add_argument("out_csv")
    a = ap.parse_args()
    nd, order, vectors = parse(a.rvtj)

    # canonical aliases for the most-used quantities
    def find(*subs):
        for name in order:
            if all(s in name for s in subs):
                return name
        return None
    v_col = find("velocity")
    t_col = find("temperature") if find("temperature") and "grey" not in (find("temperature") or "") else None
    # prefer the plain temperature (not grey)
    for name in order:
        if name.startswith("temperature") and "grey" not in name:
            t_col = name; break

    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["depth_index"]
        if v_col: header.append("v_kms")
        if t_col: header.append("T_e_K")
        header += order
        w.writerow(header)
        for d in range(nd):
            row = [d + 1]
            if v_col: row.append(f"{vectors[v_col][d]:.7E}")
            if t_col: row.append(f"{vectors[t_col][d]*1e4:.7E}")
            for name in order:
                vals = vectors.get(name, [])
                row.append(f"{vals[d]:.7E}" if d < len(vals) else "")
            w.writerow(row)
    print(f"[parse_rvtj] ND={nd}, {len(order)} vectors -> {a.out_csv}")
    print("  vectors:", ", ".join(order))


if __name__ == "__main__":
    main()
