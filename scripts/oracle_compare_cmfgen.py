#!/usr/bin/env python3
"""REPORT-ONLY Gate-B Phase-1.6 comparison against frozen CMFGEN outputs.

The parser is deliberately fail-closed: RVTJ, PRRR, ion OUT and GENCOOL blocks
must have their source-declared dimensions.  Missing identical quantities stay
as rows with reasons; this script contains no threshold or gate verdict.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "validation/cmfgen_toy06_19p48d/analysis"
GAMMA_DIR = ANALYSIS / "gamma_coiii_alllevel"
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(GAMMA_DIR))
from cmp_rvtj_T_ne_vs_published import read_rvtj  # noqa: E402
from gamma_coiii_alllevel import read_eddfactor  # noqa: E402

SHELLS = (0, 8, 43)
ION_FILE = {
    (14, 1): ("Sk2", "Sk2PRRR", "Sk2OUT"),
    (14, 2): ("SkIII", "SkIIIPRRR", "SkIIIOUT"),
    (16, 1): ("S2", "S2PRRR", "S2OUT"),
    (16, 2): ("SIII", "SIIIPRRR", "SIIIOUT"),
    (26, 1): ("Fe2", "Fe2PRRR", "Fe2OUT"),
    (26, 2): ("FeIII", "FeIIIPRRR", "FeIIIOUT"),
    (26, 3): ("FeIV", "FeIVPRRR", "FeIVOUT"),
    (27, 2): ("CoIII", "CoIIIPRRR", "CoIIIOUT"),
}


def numeric_line(line: str) -> list[float] | None:
    toks = line.split()
    if not toks:
        return None
    try:
        vals = []
        for token in toks:
            token = token.replace("D", "E").replace("d", "e")
            if "e" not in token.lower():
                token = re.sub(r"(?<=\d)([+-]\d{2,3})$", r"E\1", token)
            vals.append(float(token))
        return vals
    except ValueError:
        return None


def read_geometry(path: Path) -> dict[int, float]:
    with path.open() as fh:
        return {
            int(r["shell_id"]): 0.5 * (float(r["v_inner"]) + float(r["v_outer"])) / 1e5
            for r in csv.DictReader(fh)
        }


def model_superlevels(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    rx = re.compile(r"^\s*(\d+)\s*,.*\[([A-Za-z0-9]+)_ISF\]")
    for line in path.read_text().splitlines():
        m = rx.search(line)
        if m:
            out[m.group(2)] = int(m.group(1))
    return out


def parse_prrr(path: Path, ion_label: str, nd: int, nsl: int) -> dict[str, np.ndarray]:
    """Parse every depth chunk, requiring exactly N_SL photo-rate rows."""
    lines = path.read_text(errors="strict").replace("\f", "\n").splitlines()
    ion_density = np.full(nd, np.nan)
    electron_density = np.full(nd, np.nan)
    photo_sum = np.full(nd, np.nan)
    alpha = np.full(nd, np.nan)
    chunks: set[int] = set()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "Depth index":
            i += 1
            continue
        depth_vals = numeric_line(lines[i + 1])
        if not depth_vals:
            raise ValueError(f"{path}: missing depth-index row at physical line {i + 2}")
        depths = [int(v) - 1 for v in depth_vals]
        if any(d < 0 or d >= nd for d in depths):
            raise ValueError(f"{path}: depth out of range: {depths}")
        j0, width = depths[0], len(depths)
        chunks.add(j0)
        k = i + 2
        while k < len(lines) and lines[k].strip() != "Depth index":
            label = lines[k].strip()
            if label in {"Ion Density", "Electron Density",
                         "Radiative Recombination Coefficient for explicitly treated levels."}:
                vals = numeric_line(lines[k + 1])
                if vals is None or len(vals) != width:
                    raise ValueError(f"{path}:{k + 2}: {label} width != {width}")
                arr = (ion_density if label == "Ion Density" else
                       electron_density if label == "Electron Density" else alpha)
                arr[j0:j0 + width] = vals
                k += 2
                continue
            if label == f"{ion_label} Photoionization Rates":
                block = []
                for row in range(nsl):
                    vals = numeric_line(lines[k + 1 + row])
                    if vals is None or len(vals) != width:
                        raise ValueError(
                            f"{path}:{k + 2 + row}: photo row {row + 1}/{nsl} invalid")
                    block.append(vals)
                if numeric_line(lines[k + 1 + nsl]) is not None:
                    raise ValueError(f"{path}: more than MODEL_SPEC N_SL={nsl} photo rows")
                photo_sum[j0:j0 + width] = np.sum(np.asarray(block), axis=0)
                k += 1 + nsl
                continue
            k += 1
        i = k
    expected_chunks = set(range(0, nd, 10))
    if chunks != expected_chunks:
        raise ValueError(f"{path}: depth chunks {sorted(chunks)} != {sorted(expected_chunks)}")
    for name, arr in [("Ion Density", ion_density), ("Electron Density", electron_density),
                      ("photo sum", photo_sum), ("alpha", alpha)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{path}: incomplete {name}, finite={np.isfinite(arr).sum()}/{nd}")
    gamma = photo_sum / ion_density
    return {"gamma": gamma, "alpha": alpha, "ion_density": ion_density,
            "electron_density": electron_density}


def parse_out_bk(path: Path, nd: int) -> np.ndarray:
    """Parse the source-declared NLEV departure coefficients for exactly ND depths."""
    lines = path.read_text(errors="strict").splitlines()
    meta = numeric_line(lines[3])
    if meta is None or len(meta) < 3:
        raise ValueError(f"{path}: invalid fourth-line metadata")
    nlev = int(meta[2])
    out = np.full((nd, nlev), np.nan)
    k = 4
    for expected_depth in range(1, nd + 1):
        while k < len(lines) and numeric_line(lines[k]) is None:
            k += 1
        hdr = numeric_line(lines[k]) if k < len(lines) else None
        if hdr is None or len(hdr) != 8 or int(round(hdr[-1])) != expected_depth:
            raise ValueError(f"{path}:{k + 1}: expected depth header {expected_depth}")
        k += 1
        vals: list[float] = []
        while len(vals) < nlev and k < len(lines):
            row = numeric_line(lines[k])
            if row is None:
                raise ValueError(f"{path}:{k + 1}: nonnumeric inside b block")
            vals.extend(row)
            k += 1
        if len(vals) != nlev:
            raise ValueError(f"{path}: depth {expected_depth} has {len(vals)} != {nlev}")
        out[expected_depth - 1] = vals
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{path}: nonfinite b coefficient")
    return out


def parse_gencool(path: Path, nd: int) -> dict[str, np.ndarray]:
    """Sum all ion BF, collisional and FF blocks; read the explicit net block."""
    lines = path.read_text(errors="strict").replace("\f", "\n").splitlines()
    totals = {q: np.zeros(nd) for q in ("bf", "coll", "ff")}
    totals["net"] = np.full(nd, np.nan)
    depths: list[int] | None = None
    seen_depths: set[int] = set()
    i = 0
    while i < len(lines):
        label = lines[i].strip()
        if label == "Depth":
            vals = numeric_line(lines[i + 1])
            if not vals:
                raise ValueError(f"{path}:{i + 2}: missing depth row")
            depths = [int(v) - 1 for v in vals]
            if any(d < 0 or d >= nd for d in depths):
                raise ValueError(f"{path}: GENCOOL depth out of range")
            seen_depths.update(depths)
            i += 2
            continue
        if depths is None:
            i += 1
            continue
        if label.endswith("Bound-Free Cooling [ergs/cm**3/s]"):
            k, rows = i + 1, 0
            while k < len(lines):
                vals = numeric_line(lines[k])
                if vals is None:
                    break
                if len(vals) != len(depths):
                    raise ValueError(f"{path}:{k + 1}: BF width mismatch")
                totals["bf"][depths] += vals
                rows += 1
                k += 1
            if rows == 0:
                raise ValueError(f"{path}:{i + 1}: empty BF block")
            i = k
            continue
        kind = None
        if label.endswith("Collisional Cooling"):
            kind = "coll"
        elif label.endswith("(ion) Free-Free Cooling"):
            kind = "ff"
        elif label == "Net Cooling Rate [ergs/cm**3/sec]":
            kind = "net"
        if kind:
            k = i + 1
            while k < len(lines) and numeric_line(lines[k]) is None:
                k += 1
            vals = numeric_line(lines[k]) if k < len(lines) else None
            if vals is None or len(vals) != len(depths):
                raise ValueError(f"{path}:{i + 1}: {kind} width mismatch")
            if kind == "net":
                totals[kind][depths] = vals
            else:
                totals[kind][depths] += vals
            i = k + 1
            continue
        i += 1
    if seen_depths != set(range(nd)):
        raise ValueError(f"{path}: GENCOOL depths incomplete")
    if not np.all(np.isfinite(totals["net"])):
        raise ValueError(f"{path}: GENCOOL net incomplete")
    return totals


def read_oracles(directory: Path) -> list[dict[str, str]]:
    rows = []
    for shell in SHELLS:
        path = directory / f"lumina_oracle_cell_s{shell}.csv"
        with path.open() as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def locate_value_lines(path: Path, labels: list[str], nd: int) -> dict[str, list[int]]:
    lines = path.read_text().splitlines()
    out = {}
    for label in labels:
        start = next(i for i, line in enumerate(lines) if label in line) + 1
        loc, got, j = [], 0, start
        while got < nd:
            vals = numeric_line(lines[j])
            if vals is None:
                raise ValueError(f"{path}:{j + 1}: nonnumeric in {label}")
            loc.extend([j + 1] * len(vals))
            got += len(vals)
            j += 1
        if got != nd:
            raise ValueError(f"{path}: {label} has {got} != ND={nd}")
        out[label] = loc
    return out


def source_evidence(source_dir: Path) -> dict[str, tuple[Path, int, str]]:
    writer = source_dir / "new_main/cmfgen_sub.f"
    module = source_dir / "new_main/mod_subs/mod_cmfgen.f"
    recomb = source_dir / "subs/wrrecomchk_v3.f"
    cool = source_dir / "subs/wrcoolgen_v2.f"
    checks = {
        "ne_writer_header": (writer, r"WRITE.*Electron density"),
        "ne_writer_value": (writer, r"WRITE.*\)ED\s*$"),
        "ne_unit_declaration": (module, r"ED\(:\).*!Electron density \(#/cm\^3\)"),
        "alpha_division": (recomb, r"TOTRR\(J\)=TOTRR\(J\)/ED\(J\)/DHYD\(J\)"),
        "gencool_unit": (cool, r"Bound-Free Cooling.*ergs/cm\*\*3/s"),
    }
    out = {}
    for key, (path, rx) in checks.items():
        # splitlines() treats Fortran form-feed as a line boundary and therefore
        # does not report newline-defined physical lines.  Evidence CSV line
        # numbers are explicitly newline based (matching nl/sed/editors).
        lines = path.read_text(errors="strict").split("\n")
        match = next(((i + 1, line.strip()) for i, line in enumerate(lines)
                      if re.search(rx, line, re.I)), None)
        if match is None:
            raise ValueError(f"CMFGEN source evidence missing: {key} in {path}")
        out[key] = (path, match[0], match[1])
    return out


def coverage_disposition(row: dict[str, str]) -> tuple[str, str]:
    """Classify every non-compared row; unknown or blank reasons fail closed."""
    status = row["status"]
    note = row["note"].strip()
    if status == "compared":
        return "strict_identical_quantity", "closed"
    if status == "context_only_nonidentical":
        return "numeric_context_nonidentical_quantity", "closed"
    if not note:
        raise ValueError(f"coverage orphan without reason: {row}")
    low = note.lower().replace("_", " ")
    if status.startswith("lumina_unavailable"):
        if "raw-jbar" in low or "raw jbar" in low:
            return "capture_gate:LUMINA_GATEB_ORACLE_CAPTURE", "recoverable_on_rerun"
        if "lower member of an assembled nlte pair" in low:
            return "structural:no_lower_pair_in_frozen_nlte_topology", "irrecoverable_for_snapshot"
        if ("positive-population" in low or "positive lower-level" in low or
                "no representative transition" in low):
            return "snapshot:no_positive_population_flow", "irrecoverable_for_snapshot"
        if "macro-atom" in low or "per-shell volumetric ledger" in low:
            return "capture_gate:LUMINA_GATEB_ORACLE_CAPTURE", "recoverable_on_rerun"
        if ("lumina ma line destruct.csv" in low or
                "shell ownership and packet energy/volume normalization" in low):
            return "archive_loss:missing_per_shell_ma_ledger", "unavailable_in_selected_archive"
        if "non-sentinel recorded b_k" in low:
            return "snapshot:no_recorded_representative_bk", "irrecoverable_for_snapshot"
        if "census sums to zero" in low or "all recorded stages summed" in low:
            return "snapshot:zero_element_population_census", "irrecoverable_for_snapshot"
        if "not reached" in low:
            return "production_call_not_reached", "recoverable_on_rerun"
        raise ValueError(f"unregistered Lumina-unavailable reason: {row}")
    return "cmfgen_output_has_no_identical_anchor", "irrecoverable_with_selected_outputs"


def fmt(x: float | None) -> str:
    return "" if x is None or not math.isfinite(x) else f"{x:.9e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", type=Path,
                    default=ROOT / "validation/gate_b_dual_oracle/phase1_6")
    ap.add_argument("--model-dir", type=Path,
                    default=ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv")
    ap.add_argument("--model-spec", type=Path,
                    help="defaults to CMFGEN output directory MODEL_SPEC")
    ap.add_argument("--cmfgen-dir", type=Path,
                    default=Path("/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern"))
    ap.add_argument("--cmfgen-source-dir", type=Path,
                    default=Path("/gpfs/kjhan/cmfgen_src/cur_cmf"))
    ap.add_argument("--edd-dir", type=Path,
                    default=Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4"))
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    out_dir = args.out_dir or args.oracle_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_oracles(args.oracle_dir)
    velocities = read_geometry(args.model_dir / "geometry.csv")
    V, Ne, Te = read_rvtj(str(args.cmfgen_dir / "RVTJ"))
    nd = len(V)
    mapping = {}
    for shell in SHELLS:
        v = velocities[shell]
        d = int(np.argmin(np.abs(V - v)))
        inside = bool(V.min() <= v <= V.max())
        if not inside:
            raise ValueError(f"s{shell} velocity {v} lies outside CMFGEN [{V.min()}, {V.max()}]")
        mapping[shell] = {
            "lumina_velocity_km_s": v, "cmfgen_depth_1based": d + 1,
            "cmfgen_velocity_km_s": float(V[d]), "delta_v_km_s": float(V[d] - v),
            "in_cmfgen_range": inside,
        }

    model_spec_path = args.model_spec or (args.cmfgen_dir / "MODEL_SPEC")
    nsl = model_superlevels(model_spec_path)
    prrr, bout = {}, {}
    for key, (label, prname, outname) in ION_FILE.items():
        if label not in nsl:
            raise ValueError(f"{model_spec_path}: no {label}_ISF")
        prrr[key] = parse_prrr(args.cmfgen_dir / prname, label, nd, nsl[label])
        bout[key] = parse_out_bk(args.cmfgen_dir / outname, nd)
    cool = parse_gencool(args.cmfgen_dir / "GENCOOL", nd)
    evidence = source_evidence(args.cmfgen_source_dir)

    Jedd, nuedd, nd_edd, finish = read_eddfactor(str(args.edd_dir / "EDDFACTOR"))
    if nd_edd != nd or not np.isfinite(finish) or finish == 0:
        raise ValueError("EDDFACTOR dimension/completion mismatch")

    compared = []
    for row in rows:
        shell = int(row["shell"])
        depth0 = mapping[shell]["cmfgen_depth_1based"] - 1
        q, Z, stage = row["quantity"], int(row["Z"]), int(row["stage"])
        try:
            lum = float(row["value"]) if row["value"] else None
        except ValueError:
            lum = None
        cmf, source, status, note = None, "", "unavailable", ""
        key = (Z, stage)
        if q == "T_e":
            cmf, source, status = float(Te[depth0]), "RVTJ:Temperature (10^4K)*1e4", "compared"
        elif q == "n_e":
            cmf, source, status = float(Ne[depth0]), "RVTJ:Electron density; source-certified #/cm^3", "compared"
        elif q == "n_ion" and key in prrr:
            cmf, source, status = float(prrr[key]["ion_density"][depth0]), f"{ION_FILE[key][1]}:Ion Density", "compared"
        elif q == "ion_fraction":
            note = "CMFGEN target PRRR files do not constitute a complete element-stage census"
        elif q == "b_k_representative" and key in bout:
            m = re.fullmatch(r"level(\d+)", row["transition"])
            lev = int(m.group(1)) if m else -1
            if 0 <= lev < bout[key].shape[1]:
                cmf = float(bout[key][depth0, lev])
                source, status = f"{ION_FILE[key][2]}:depth-block level{lev}", "compared"
            else:
                note = f"representative level {lev} outside CMFGEN OUT NLEV={bout[key].shape[1]}"
        elif q == "Gamma_photoion_total" and key in prrr:
            cmf, source, status = float(prrr[key]["gamma"][depth0]), f"{ION_FILE[key][1]}:sum(exact N_SL PR)/Ion Density", "compared"
        elif q == "alpha_recomb_total" and key in prrr:
            cmf, source, status = float(prrr[key]["alpha"][depth0]), f"{ION_FILE[key][1]}:RR/(ED*ion density), source-certified cm^3/s", "compared"
        elif q in {"alpha_recomb_spont", "alpha_recomb_stim"}:
            note = "CMFGEN PRRR does not expose a separately labelled spontaneous/stimulated alpha split"
        elif q in {"jbar_representative", "jbar_input_raw"} and row["frequency_Hz"]:
            nu = float(row["frequency_Hz"])
            fi = int(np.argmin(np.abs(nuedd - nu)))
            cmf, source, status = float(Jedd[fi, depth0]), f"EDDFACTOR:nu={nuedd[fi]:.9e}Hz", "compared"
            note = "nearest native CMFGEN frequency at speed-matched depth"
        elif q in {"cooling_ff", "cooling_ff_grid"}:
            cmf, source, status = float(cool["ff"][depth0]), "GENCOOL:sum(all ion Free-Free Cooling)", "context_only_nonidentical"
            note = "Lumina row is emissive cooling; GENCOOL FF is net emission-minus-absorption"
        elif q == "cooling_bf_net":
            cmf, source, status = float(cool["bf"][depth0]), "GENCOOL:sum(all ion Bound-Free Cooling)", "compared"
        elif q == "cooling_bf":
            cmf, source, status = float(cool["bf"][depth0]), "GENCOOL:sum(all ion Bound-Free Cooling)", "context_only_nonidentical"
            note = "Lumina row is emission only; GENCOOL BF is net emission-minus-photo-heating; use cooling_bf_net for identical sign accounting"
        elif q == "cooling_bb_collisional":
            cmf, source, status = float(cool["coll"][depth0]), "GENCOOL:sum(all ion Collisional Cooling)", "compared"
        elif q == "thermal_net":
            cmf, source, status = -float(cool["net"][depth0]), "GENCOOL:-Net Cooling Rate (converted C-H to Lumina H-C)", "compared"
        elif q == "heating_deposition":
            note = "GENCOOL has no separately labelled deposition row identical to Lumina's external deposition input"
        elif q == "heating_photoion":
            note = "GENCOOL BF block is net bound-free cooling, not a separate photoion heating rate"
        elif q == "heating_MA_LINE_DESTRUCT":
            note = "frozen Lumina transport wrote no per-cell macro-atom destruction ledger"
        elif q == "cooling_adiabatic":
            note = "GENCOOL has no separately labelled adiabatic cooling row"
        elif q.startswith(("chi_bf_", "eta_bf_", "chi_ff_", "eta_ff_")):
            note = "selected CMFGEN files expose no identical monochromatic coefficient"
        elif q in {"C_lu", "C_ul"}:
            note = "GENCOOL exposes aggregate collisional cooling, not the representative transition coefficient"
        elif q.startswith("R_") or q == "sobolev_beta":
            note = "GENCOOL/EDDFACTOR do not expose the matched transition rate/beta"
        elif q in {"T_rad", "W"}:
            note = "RVTJ has no identical Lumina dilute-field scalar"
        else:
            note = "no registered identical CMFGEN anchor"

        if row["status"] != "available":
            status = "lumina_" + row["status"]
            note = row["note"] or note
        ratio = (lum / cmf if status == "compared" and lum is not None and
                 cmf is not None and math.isfinite(lum) and math.isfinite(cmf) and cmf != 0 else None)
        compared.append({
            "shell": shell, "lumina_velocity_km_s": fmt(mapping[shell]["lumina_velocity_km_s"]),
            "cmfgen_depth_1based": mapping[shell]["cmfgen_depth_1based"],
            "cmfgen_velocity_km_s": fmt(mapping[shell]["cmfgen_velocity_km_s"]),
            "category": row["category"], "quantity": q, "Z": Z, "stage": stage,
            "transition": row["transition"], "lumina_value": fmt(lum),
            "cmfgen_value": fmt(cmf), "lumina_over_cmfgen": fmt(ratio),
            "unit": row["unit"], "cmfgen_source": source, "status": status, "note": note,
        })

    out_csv = out_dir / "oracle_vs_cmfgen.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(compared[0]))
        w.writeheader(); w.writerows(compared)
    map_csv = out_dir / "shell_cmfgen_depth_map.csv"
    with map_csv.open("w", newline="") as fh:
        fields = ["shell", *next(iter(mapping.values())).keys()]
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for shell, data in mapping.items():
            w.writerow({"shell": shell, **data})

    line_map = locate_value_lines(args.cmfgen_dir / "RVTJ",
                                  ["Velocity (km/s)", "Electron density",
                                   "Temperature (10^4K)"], nd)
    roundtrip = out_dir / "cmfgen_parser_roundtrip.csv"
    with roundtrip.open("w", newline="") as fh:
        fields = ["shell", "quantity", "raw_file", "header_label",
                  "physical_line_1based", "depth_1based", "raw_value",
                  "post_conversion_value", "conversion", "unit_evidence"]
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for shell in SHELLS:
            d = mapping[shell]["cmfgen_depth_1based"]; i = d - 1
            for label, q, raw, post, conv, proof in [
                ("Velocity (km/s)", "velocity", V[i], V[i], "identity km/s", "RVTJ header"),
                ("Electron density", "n_e", Ne[i], Ne[i], "identity #/cm^3",
                 "mod_cmfgen.f ED(:) declaration + cmfgen_sub.f RVTJ WRITE ED"),
                ("Temperature (10^4K)", "T_e", Te[i] / 1e4, Te[i], "multiply 1e4 K", "RVTJ header"),
            ]:
                w.writerow({"shell": shell, "quantity": q,
                    "raw_file": str(args.cmfgen_dir / "RVTJ"), "header_label": label,
                    "physical_line_1based": line_map[label][i], "depth_1based": d,
                    "raw_value": f"{raw:.10e}", "post_conversion_value": f"{post:.10e}",
                    "conversion": conv, "unit_evidence": proof})

    ev_csv = out_dir / "cmfgen_source_evidence.csv"
    with ev_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["evidence", "source_file",
                                           "physical_line_1based", "source_text"])
        w.writeheader()
        for key, (path, line, source_text) in evidence.items():
            w.writerow({"evidence": key, "source_file": str(path),
                        "physical_line_1based": line, "source_text": source_text})

    dim_csv = out_dir / "cmfgen_parser_dimensions.csv"
    with dim_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Z", "stage", "ion_label", "ND",
                                           "PRRR_N_SL", "OUT_NLEV",
                                           "model_spec_file"])
        w.writeheader()
        for key, (label, _prname, _outname) in ION_FILE.items():
            w.writerow({"Z": key[0], "stage": key[1], "ion_label": label,
                        "ND": nd, "PRRR_N_SL": nsl[label],
                        "OUT_NLEV": bout[key].shape[1],
                        "model_spec_file": str(model_spec_path)})

    snap_csv = out_dir / "cmfgen_snapshot_consistency.csv"
    with snap_csv.open("w", newline="") as fh:
        fields = ["shell", "Z", "stage", "depth_1based", "rvtj_ne",
                  "prrr_ne", "relative_difference", "status"]
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for shell in SHELLS:
            d = mapping[shell]["cmfgen_depth_1based"] - 1
            for key, data in prrr.items():
                rel = data["electron_density"][d] / Ne[d] - 1.0
                w.writerow({"shell": shell, "Z": key[0], "stage": key[1],
                            "depth_1based": d + 1, "rvtj_ne": f"{Ne[d]:.10e}",
                            "prrr_ne": f"{data['electron_density'][d]:.10e}",
                            "relative_difference": f"{rel:.10e}",
                            "status": "same_snapshot" if abs(rel) <= 5e-5 else
                                      "different_snapshot_or_output_generation"})

    census = Counter(r["status"] for r in compared)
    coverage = Counter()
    for row in compared:
        disposition, recoverability = coverage_disposition(row)
        coverage[(row["category"], row["status"], disposition,
                  recoverability, row["note"])] += 1
    coverage_csv = out_dir / "coverage_disposition.csv"
    with coverage_csv.open("w", newline="") as fh:
        fields = ["category", "status", "count", "disposition",
                  "recoverability", "reason"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (category, status, disposition, recoverability, reason), count in sorted(
                coverage.items()):
            w.writerow({"category": category, "status": status, "count": count,
                        "disposition": disposition,
                        "recoverability": recoverability, "reason": reason})
    md = out_dir / "oracle_vs_cmfgen.md"
    with md.open("w") as fh:
        fh.write("# Gate-B Phase-1.6 Lane C comparison (REPORT-ONLY)\n\n")
        fh.write("No threshold or gate verdict is implemented here.\n\n")
        fh.write("## Speed mapping\n\n")
        fh.write("| shell | Lumina km/s | CMFGEN depth | CMFGEN km/s | delta | in range |\n")
        fh.write("|---:|---:|---:|---:|---:|:---:|\n")
        for shell, m in mapping.items():
            fh.write(f"| s{shell} | {m['lumina_velocity_km_s']:.3f} | "
                     f"{m['cmfgen_depth_1based']} | {m['cmfgen_velocity_km_s']:.3f} | "
                     f"{m['delta_v_km_s']:+.3f} | {m['in_cmfgen_range']} |\n")
        fh.write("\nThe outer cell is s43, the outermost recorded Lumina cell whose "
                 "centre remains inside the CMFGEN RVTJ velocity range.\n\n")
        fh.write("## Census\n\n")
        for status in sorted(census):
            fh.write(f"- `{status}`: {census[status]}\n")
        numeric = census["compared"] + census["context_only_nonidentical"]
        fh.write(f"\nStrict identical coverage: **{census['compared']}/{len(compared)} "
                 f"= {100.0*census['compared']/len(compared):.2f}%**. "
                 f"Including explicitly nonidentical numeric context: "
                 f"**{numeric}/{len(compared)} = {100.0*numeric/len(compared):.2f}%**.\n")
        fh.write("\n`coverage_disposition.csv` accounts for every non-compared row. "
                 "An unknown or blank unavailability reason aborts generation.\n")
        fh.write("\n## Parser and unit evidence\n\n")
        fh.write("- RVTJ n_e is identity-transcribed. `cmfgen_source_evidence.csv` "
                 "records the RVTJ header writer, the following `WRITE ... ED`, and "
                 "the `ED(:) !Electron density (#/cm^3)` declaration.\n")
        fh.write("- PRRR requires exactly MODEL_SPEC N_SL rows in every 10-depth "
                 "chunk and all ND chunks. Alpha is source-certified by "
                 "`TOTRR=TOTRR/ED/DHYD`.\n")
        fh.write("- Ion OUT requires exactly source-declared NLEV coefficients for "
                 "all ND depth blocks. GENCOOL requires all depths and reads its "
                 "explicit volumetric-rate headers.\n")
        fh.write("- `cmfgen_snapshot_consistency.csv` records RVTJ-versus-PRRR n_e; "
                 "a mismatch is disclosed and never silently treated as one snapshot.\n")
        fh.write("- Every unavailable comparison remains in `oracle_vs_cmfgen.csv` "
                 "with its reason.\n")

    for path in (out_csv, md, map_csv, roundtrip, ev_csv, dim_csv, snap_csv,
                 coverage_csv):
        print(f"[REPORT-ONLY] wrote {path}")


if __name__ == "__main__":
    main()
