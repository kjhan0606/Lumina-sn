#!/usr/bin/env python3
"""Read-only Stage4 Jbar arithmetic census for a sealed A2-10 stderr log.

This does not alter Lumina values and never supplies a solver value.  It only
checks whether each recorded saturation row contains enough independent fields
to recompute Jbar = beta*Jcont + (1-beta)*Sprobe.  If Jcont is absent, it may be
algebraically inferred from the already recorded Jbar/S pair, but that is
explicitly reported as NON-INDEPENDENT and cannot predict an rc=0 run.
"""
import argparse, json, math, re, hashlib
from pathlib import Path

ROW = re.compile(r"\[A2-10\]\[LINE-SATURATION-ROW\] (?P<body>.*)$")
FIELD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)")

def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except ValueError:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stderr", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raw = args.stderr.read_bytes()
    rows = []
    for line in raw.decode("utf-8", "replace").splitlines():
        m = ROW.search(line)
        if not m:
            continue
        d = {k: v for k, v in FIELD.findall(m.group("body"))}
        needed = {k: fnum(d.get(k, "")) for k in ("beta", "Jbar", "source_function")}
        independent_jc = fnum(d.get("J_cont", ""))
        independent_sp = fnum(d.get("S_probe", ""))
        declared_independent = d.get("independent_fields_defined") == "1"
        rec = {"phase": d.get("phase"), "shell": d.get("shell"),
               "line": d.get("line"), "Z": d.get("Z"), "ion": d.get("ion"),
               "independent_J_cont": independent_jc is not None,
               "independent_S_probe": independent_sp is not None,
               "declared_independent_fields": declared_independent}
        if all(v is not None for v in needed.values()):
            beta, jbar, source = needed["beta"], needed["Jbar"], needed["source_function"]
            if beta != 0.0:
                jc = independent_jc if independent_jc is not None else (jbar - (1.0-beta)*source)/beta
                probe = independent_sp if independent_sp is not None else source
                reconstructed = beta*jc + (1.0-beta)*probe
                rec.update({"finite": True, "inferred_J_cont": independent_jc is None,
                            "inferred_S_probe": independent_sp is None,
                            "reconstructed_Jbar": reconstructed,
                            "absolute_residual": abs(reconstructed-jbar),
                            "independent_prediction": declared_independent and
                            independent_jc is not None and independent_sp is not None})
            else:
                rec.update({"finite": True, "zero_beta": True, "independent_prediction": False})
        else:
            rec.update({"finite": False, "independent_prediction": False})
        rows.append(rec)
    independent = sum(bool(r.get("independent_prediction")) for r in rows)
    finite = sum(bool(r.get("finite")) for r in rows)
    out = {
        "schema": "lumina-a210-stage4-jbar-offline-v1",
        "input": str(args.stderr),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(rows), "finite_rows": finite,
        "independent_prediction_rows": independent,
        "prediction_status": "READY" if independent == len(rows) and rows else "INSUFFICIENT_INDEPENDENT_FIELDS",
        "physical_values_modified": False,
        "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("schema", "row_count", "finite_rows", "independent_prediction_rows", "prediction_status", "input_sha256")}))

if __name__ == "__main__":
    main()
