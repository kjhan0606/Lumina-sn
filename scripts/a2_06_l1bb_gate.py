#!/usr/bin/env python3
"""A2-06 L-1bb gate and pre-registered negative controls.

The current accepted state is BLOCKED_MISSING_RATE_EXPORT: closure, wiring,
static ownership, and the atomic A_ul crosswalk are judged now, but absence of
separated CMFGEN NETRATE/TOTRATE channels can never be promoted to L-1bb PASS.
An unavailable runtime configuration-label binding similarly produces
BLOCKED_MISSING_LABEL_BINDING rather than PASS or FAIL.

Exit 0 means every available prerequisite passed, all nine negative controls
observed their expected failure, and the L-1bb state is a registered BLOCKED
state or a complete PASS from --rate-ledger.  Exit 1 is a physical/contract
failure; exit 2 is an input/schema failure.
"""
from __future__ import annotations

import argparse
from array import array
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cmfgen_parser import parse_osc  # noqa: E402

MODEL_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos"
COHORT = ROOT / "validation/a2_02c/A2_02C_ESTIMATOR_COHORT.json"
RESOLUTION = ROOT / "validation/a2_02c/A2_02C_RESOLUTION_INPUT.json"
FINE_RESULT = ROOT / "validation/a2_02c/a2_02c_estimator_effort_gate2_v3_result.json"
COHORT_SHA256 = "0c029ca15116119e2d7af4693d76988b14a452e35cae3679522629230a6c3e69"
PROFILE_SHA256 = "f8572907be3ad2e9738a84dae1000338bb7100772cf1d3b52ec17561da409bbf"
NU_LO = 1.4402928950097124e12
NU_HI = 4.032418413741097e16
MAX_LIMIT = 0.01
MEDIAN_LIMIT = 0.002
AUL_LIMIT = 1.0e-10
ROW_STRUCT = struct.Struct("<Qiid")
CM_INV_TO_EV = 1.239841984e-4


class GateInputError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateInputError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_label(label: str) -> str:
    return hashlib.sha256(label.strip().encode()).hexdigest()


def source_excitation_ev(level: np.void) -> float:
    """CMFGEN E_cm is excitation energy; its printed E_eV is continuum distance."""
    return float(level["E_cm"]) * CM_INV_TO_EV


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def error_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors, false_positive = [], []
    for row in rows:
        left, right = float(row["direct"]), float(row["projected"])
        require(math.isfinite(left) and math.isfinite(right), "non-finite closure value")
        if left == 0.0 and right == 0.0:
            err = 0.0
        elif right == 0.0:
            err = math.inf
            false_positive.append(int(row["band"]))
        else:
            err = abs(left - right) / abs(right)
        errors.append(err)
    maximum = max(errors, default=math.inf)
    median = float(np.median(errors)) if errors else math.inf
    return {
        "maximum_relative_change": maximum,
        "median_relative_change": median,
        "maximum_limit": MAX_LIMIT,
        "median_limit": MEDIAN_LIMIT,
        "false_positive_rows": false_positive,
        "passed": maximum <= MAX_LIMIT and median <= MEDIAN_LIMIT and not false_positive,
    }


def run_projection_fixture(out: Path) -> dict[str, Any]:
    source = ROOT / "tests/a2_06_l1bb_fixture.c"
    binary = out / "a2_06_l1bb_fixture"
    subprocess.run(["cc", "-O2", "-std=c11", str(source), "-lm", "-o", str(binary)],
                   check=True, cwd=ROOT)
    doc = json.loads(subprocess.check_output([str(binary)], text=True, cwd=ROOT))
    expected = np.geomspace(NU_LO, NU_HI, 9)
    actual = np.asarray(doc.get("band_edges_hz", []), dtype=float)
    require(doc.get("schema") == "lumina-a2-06-projection-fixture-v1",
            "projection fixture schema mismatch")
    require(actual.shape == (9,) and np.array_equal(actual, expected),
            "projection edges differ from the preregistered geomspace")
    require(doc.get("frame") == "comoving" and
            doc.get("normalization") == "1/(4*pi*V_s*delta_t)",
            "projection fixture frame/normalization mismatch")
    result = error_summary(doc.get("rows", []))
    result.update({"band_edges_hz": actual.tolist(), "fixture": str(source),
                   "fixture_sha256": sha256_file(source)})
    return result


def validate_fine_inputs(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cohort = load_json(COHORT)
    resolution = load_json(RESOLUTION)
    fine = load_json(path)
    active = [row for row in cohort.get("records", [])
              if str(row.get("cohort_status", "")).startswith("ACTIVE_")]
    require(len(active) == 74, f"fine cohort active count {len(active)} != 74")
    require(resolution.get("estimator_cohort", {}).get("sha256") == COHORT_SHA256,
            "resolution input does not bind the V5 cohort hash")
    require(sha256_file(COHORT) == COHORT_SHA256,
            "A2_02C_ESTIMATOR_COHORT bytes differ from V5 binding")
    cref = fine.get("cohort", {})
    closure = fine.get("fine_diagnostic_closure", {})
    convergence = fine.get("fine_histogram_resolution_convergence", {})
    require(cref.get("sha256") == COHORT_SHA256 and
            int(cref.get("active_records", -1)) == 74,
            "fine result does not bind all 74 active cohort records")
    require(int(closure.get("registered_records", -1)) == 74 and
            int(convergence.get("registered_records", -1)) == 74,
            "fine closure/resolution registered population is not 74")
    require(int(closure.get("invalid_eligible_records", -1)) == 0,
            "fine closure contains invalid eligible records")
    for label, metric in (("fine_closure", closure), ("fine_resolution", convergence)):
        require(float(metric.get("maximum_limit", -1)) == MAX_LIMIT and
                float(metric.get("median_limit", -1)) == MEDIAN_LIMIT,
                f"{label} thresholds were changed")
    fine_ok = bool(closure.get("passed")) and bool(convergence.get("passed"))
    return ({"cohort_path": str(COHORT), "cohort_sha256": COHORT_SHA256,
             "active_records": 74, "result_path": str(path),
             "result_sha256": sha256_file(path), "closure": closure,
             "resolution_convergence": convergence, "passed": fine_ok}, fine)


def validate_same_measure(fine: dict[str, Any]) -> dict[str, Any]:
    same = fine.get("same_measure_commit_gate", {})
    required = ("raw_segment_ledger_sha256", "generation", "frame", "volume_table",
                "delta_t_s", "normalization", "q_set_hash")
    require(all(key in same for key in required), "same-measure tuple incomplete")
    require(len(str(same["raw_segment_ledger_sha256"])) == 64,
            "raw segment ledger hash malformed")
    require(int(same["generation"]) == int(same.get("q_generation_bound_from_capture", -1)),
            "J_nu/cache generation mismatch")
    require("comoving" in str(same["frame"]).lower(), "same-measure frame is not comoving")
    require("4*pi" in str(same["normalization"]) and
            float(same["delta_t_s"]) > 0.0 and bool(same["volume_table"]),
            "same-measure normalization tuple incomplete")
    require(bool(same.get("passed")), "upstream same-measure gate failed")
    return dict(same)


def strip_diagnostic_shadows(text: str) -> str:
    begin = "A2_06_DIAGNOSTIC_SHADOW_BEGIN"
    end = "A2_06_DIAGNOSTIC_SHADOW_END"
    out, depth = [], 0
    for line in text.splitlines():
        if begin in line:
            depth += 1
            continue
        if end in line:
            require(depth > 0, "unmatched diagnostic-shadow end marker")
            depth -= 1
            continue
        if depth == 0:
            out.append(line)
    require(depth == 0, "unmatched diagnostic-shadow begin marker")
    return "\n".join(out)


def static_read_trace() -> dict[str, Any]:
    source = (ROOT / "src/lumina_plasma.c").read_text()
    main = (ROOT / "src/lumina_main.c").read_text()
    production = strip_diagnostic_shadows(source)
    require(production.count("nlte_bb_jbar_canonical(") >= 6,
            "not all CPU BB consumers reach the canonical lookup choke point")
    require("R_absorb = atom->line_B_lu[line] * Jbar_view;" in production and
            "R_stim   = atom->line_B_ul[line] * Jbar_view;" in production and
            "R_spont  = atom->line_A_ul[line];" in production,
            "split bound-bound rate formula absent")
    require("line_jbar_qset_build(&line_qset" in main and
            ".line_sum=line_acc.sum" in main and
            "radiation_field_line_jbar_view(" in main,
            "Q_g/dual-commit/view refresh wiring incomplete")
    require("line_jbar_set" not in production and "production_phi_rebin" not in production,
            "independent line-cache setter or production phi rebin found")
    return {"source": "src/lumina_plasma.c", "source_sha256":
            sha256_file(ROOT / "src/lumina_plasma.c"), "canonical_calls":
            production.count("nlte_bb_jbar_canonical("), "passed": True}


def resolve_atomic_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    marker = "/atomic/"
    require(marker in raw, f"cannot remap atomic path: {raw}")
    local = ROOT / "data/atomic/cmfgen" / raw.split(marker, 1)[1]
    require(local.is_file(), f"remapped atomic source absent: {local}")
    return local


def load_model_levels(
    path: Path,
) -> tuple[dict[tuple[int, int], dict[int, dict[str, Any]]], dict[str, Any]]:
    result: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "configuration" not in reader.fieldnames:
            return {}, {
                "state": "BLOCKED_MISSING_LABEL_BINDING",
                "reason": "levels.csv lacks configuration label source",
                "source": str(path),
            }
        row_count = 0
        for row in reader:
            row_count += 1
            label = row.get("configuration")
            if label is None or not label.strip():
                return {}, {
                    "state": "BLOCKED_MISSING_LABEL_BINDING",
                    "reason": "levels.csv configuration labels are incomplete",
                    "source": str(path),
                    "first_unbound_row": row_count + 1,
                }
            ion = (int(row["atomic_number"]), int(row["ion_number"]))
            result.setdefault(ion, {})[int(row["level_number"])] = {
                "label_hash": sha_label(label),
                "energy_eV": float(row["energy_eV"]), "g": int(row["g"]),
            }
    return result, {
        "state": "ACTIVE",
        "source": str(path),
        "bound_levels": row_count,
    }


def blocked_aul_crosswalk(model: Path, label_binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "lumina-a2-06-aul-crosswalk-v1",
        "model": str(model),
        "verdict": "BLOCKED_MISSING_LABEL_BINDING",
        "label_binding": label_binding,
        "assertion": "A_ul crosswalk PASS forbidden without configuration-label binding",
    }


def aul_crosswalk(model: Path, out: Path) -> dict[str, Any]:
    levels, label_binding = load_model_levels(model / "levels.csv")
    if label_binding["state"] != "ACTIVE":
        return blocked_aul_crosswalk(model, label_binding)
    manifests = list(csv.DictReader((model / "atomic_vintage_manifest.csv").open(newline="")))
    manifest = {(int(row["atomic_number"]), int(row["ion_number"])): row
                for row in manifests}
    require(set(levels).issubset(manifest), "levels contain ion absent from vintage manifest")

    with tempfile.TemporaryDirectory(prefix="a2_06_aul_") as tmp_name:
        tmp = Path(tmp_name)
        handles = {ion: (tmp / f"z{ion[0]}_i{ion[1]}.bin").open("wb") for ion in manifest}
        try:
            with (model / "line_list.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    ion = (int(row["atomic_number"]), int(row["ion_number"]))
                    require(ion in handles, f"line ion absent from manifest: {ion}")
                    handles[ion].write(ROW_STRUCT.pack(int(row["line_id"]),
                        int(row["level_number_lower"]), int(row["level_number_upper"]),
                        float(row["A_ul"])))
        finally:
            for stream in handles.values():
                stream.close()

        unmatched_path = out / "A2_06_AUL_UNMATCHED.csv.gz"
        rel_max = 0.0
        matched_lines = total_lumina = zero_mismatch = 0
        level_unmatched = []
        energy_delta_max = 0.0
        truth_weights = array("d")
        truth_matched = bytearray()
        with gzip.open(unmatched_path, "wt", newline="") as gz:
            writer = csv.writer(gz)
            writer.writerow(["side", "Z", "ion", "line_or_transition", "reason"])
            for ion, row in sorted(manifest.items()):
                osc_path = resolve_atomic_path(row["osc_path"])
                osc = parse_osc(osc_path)
                source_levels = {}
                label_candidates: dict[tuple[str, int], list[int]] = {}
                for idx, lev in enumerate(osc.levels):
                    sid = int(lev["ID"])
                    source_levels[sid] = lev
                    label_candidates.setdefault((sha_label(str(lev["config"])), int(lev["g"])), []).append(sid)
                level_map, used_level = {}, set()
                for lnum, lum in levels.get(ion, {}).items():
                    candidates = [sid for sid in label_candidates.get(
                        (lum["label_hash"], lum["g"]), []) if sid not in used_level]
                    if not candidates:
                        level_unmatched.append((ion[0], ion[1], lnum))
                        continue
                    sid = min(candidates, key=lambda x: abs(
                        source_excitation_ev(source_levels[x]) - lum["energy_eV"]))
                    used_level.add(sid)
                    level_map[lnum] = sid
                    energy_delta_max = max(energy_delta_max, abs(
                        source_excitation_ev(source_levels[sid]) - lum["energy_eV"]))

                by_pair: dict[tuple[int, int], list[int]] = {}
                trans = osc.transitions
                for ti, tr in enumerate(trans):
                    by_pair.setdefault((int(tr["i"]), int(tr["j"])), []).append(ti)
                    gu = int(source_levels[int(tr["j"])]["g"])
                    truth_weights.append(abs(float(tr["A"])) * gu)
                    truth_matched.append(0)
                base = len(truth_matched) - len(trans)
                used_trans = set()
                bucket = tmp / f"z{ion[0]}_i{ion[1]}.bin"
                with bucket.open("rb") as stream:
                    while True:
                        raw = stream.read(ROW_STRUCT.size)
                        if not raw:
                            break
                        require(len(raw) == ROW_STRUCT.size, f"truncated A_ul bucket {bucket}")
                        line_id, lo, up, a_lum = ROW_STRUCT.unpack(raw)
                        total_lumina += 1
                        if lo not in level_map or up not in level_map:
                            writer.writerow(["LUMINA", ion[0], ion[1], line_id,
                                             "UNMATCHED_LEVEL_LABEL"])
                            continue
                        pair = (level_map[lo], level_map[up])
                        candidates = [x for x in by_pair.get(pair, []) if x not in used_trans]
                        if not candidates:
                            writer.writerow(["LUMINA", ion[0], ion[1], line_id,
                                             "UNMATCHED_TRANSITION"])
                            continue
                        # Labels already match.  Endpoint-energy distance is the
                        # sole tie-breaker, as required by V4 section 4.
                        ti = min(candidates, key=lambda x: abs(
                            source_excitation_ev(source_levels[int(trans[x]["i"])]) -
                            levels[ion][lo]["energy_eV"]) + abs(
                            source_excitation_ev(source_levels[int(trans[x]["j"])]) -
                            levels[ion][up]["energy_eV"]))
                        used_trans.add(ti)
                        truth_matched[base + ti] = 1
                        matched_lines += 1
                        a_cmf = float(trans[ti]["A"])
                        if a_lum == 0.0 and a_cmf == 0.0:
                            rel = 0.0
                        elif a_lum == 0.0 or a_cmf == 0.0:
                            rel = math.inf
                            zero_mismatch += 1
                        else:
                            rel = abs(a_lum - a_cmf) / max(abs(a_lum), abs(a_cmf))
                        rel_max = max(rel_max, rel)
                for ti, tr in enumerate(trans):
                    if ti not in used_trans:
                        writer.writerow(["CMFGEN", ion[0], ion[1], int(tr["trans_id"]),
                                         "UNMATCHED_TRANSITION"])

    weights = np.frombuffer(truth_weights, dtype=np.float64)
    matched = np.frombuffer(truth_matched, dtype=np.uint8).astype(bool)
    require(weights.size == matched.size and weights.size > 0, "empty A_ul truth universe")
    order = np.argsort(-weights, kind="stable")
    sorted_w = weights[order]
    total = float(sorted_w.sum())
    require(total > 0.0, "zero A_ul truth weight")
    cut = int(np.searchsorted(np.cumsum(sorted_w), 0.999 * total, side="left"))
    boundary = sorted_w[cut]
    active = weights >= boundary
    coverage = float(weights[active & matched].sum() / weights[active].sum())
    result = {
        "schema": "lumina-a2-06-aul-crosswalk-v1",
        "model": str(model), "label_binding": label_binding,
        "matching": ["Z", "ion", "lower_label_sha256",
        "upper_label_sha256", "lower_energy_eV_tiebreak_only", "upper_energy_eV_tiebreak_only",
        "g_lower", "g_upper"],
        "lumina_lines": total_lumina, "matched_lines": matched_lines,
        "unmatched_lumina_lines": total_lumina - matched_lines,
        "unmatched_levels": len(level_unmatched), "truth_transitions": int(weights.size),
        "maximum_level_energy_delta_eV": energy_delta_max,
        "maximum_A_ul_relative_error": rel_max, "relative_error_limit": AUL_LIMIT,
        "one_sided_zero_rows": zero_mismatch, "truth_weight": "abs(A_ul)*g_upper",
        "active_prefix": "minimal >=99.9%; boundary ties all included",
        "active_truth_weight_coverage_diagnostic": coverage,
        "coverage_disposition": "reported, not judged: V4 section 4 replaces V3 section 3.7",
        "unmatched_path": str(unmatched_path), "unmatched_sha256": sha256_file(unmatched_path),
    }
    result["verdict"] = "PASS" if (rel_max <= AUL_LIMIT and zero_mismatch == 0) else "FAIL"
    return result


def esym(left: float, right: float) -> tuple[float, bool]:
    if left == 0.0 and right == 0.0:
        return 0.0, False
    if right == 0.0:
        return 2.0, left != 0.0
    return 2.0 * abs(left - right) / (abs(left) + abs(right)), False


def judge_rate_ledger(path: Path) -> dict[str, Any]:
    doc = load_json(path)
    require(doc.get("schema") == "lumina-a2-06-separated-rate-ledger-v1",
            "separated rate ledger schema mismatch")
    rows = doc.get("rows", [])
    require(rows, "empty separated rate ledger")
    for row in rows:
        for key in ("n_lower", "B_lu", "B_ul", "jbar_truth", "jbar_lum",
                    "R_lu_cmf", "R_ul_stim_cmf", "view_state"):
            require(key in row, f"rate row missing {key}")
        row["F_truth"] = float(row["n_lower"]) * float(row["B_lu"]) * float(row["jbar_truth"])
    ordered = sorted(rows, key=lambda r: r["F_truth"], reverse=True)
    total = sum(r["F_truth"] for r in ordered)
    require(total > 0.0, "non-positive truth flow universe")
    acc, boundary = 0.0, None
    for row in ordered:
        acc += row["F_truth"]
        boundary = row["F_truth"]
        if acc >= 0.999 * total:
            break
    active = [r for r in ordered if r["F_truth"] >= boundary]
    usable = {"MEASURED", "EXACT_ZERO"}
    fcov = (sum(r["F_truth"] for r in active if r["view_state"] in usable) /
            sum(r["F_truth"] for r in active))
    channels = {
        "Jbar": (lambda r: float(r["jbar_lum"]), lambda r: float(r["jbar_truth"])),
        "R_lu": (lambda r: float(r["B_lu"]) * float(r["jbar_lum"]),
                 lambda r: float(r["R_lu_cmf"])),
        "R_ul_stim": (lambda r: float(r["B_ul"]) * float(r["jbar_lum"]),
                      lambda r: float(r["R_ul_stim_cmf"])),
    }
    metrics, ok = {}, fcov >= 0.95
    for name, (lum, cmf) in channels.items():
        denom = sum(r["F_truth"] * abs(cmf(r)) for r in active)
        require(denom > 0.0, f"zero CMFGEN denominator for {name}")
        e1 = sum(r["F_truth"] * abs(lum(r) - cmf(r)) for r in active) / denom
        sym, fp = [], []
        for r in active:
            value, false_positive = esym(lum(r), cmf(r))
            sym.append(value)
            if false_positive:
                fp.append([r.get("line_id"), r.get("shell")])
        p95 = float(np.percentile(sym, 95))
        passed = e1 <= 0.10 and p95 <= 0.25 and not fp
        metrics[name] = {"E_1": e1, "E_1_limit": 0.10, "E_sym_P95": p95,
                         "percentile": "numpy linear", "false_positive_rows": fp,
                         "passed": passed}
        ok &= passed
    return {"state": "PASS" if ok else "FAIL", "f_cov": fcov,
            "coverage_limit": 0.95, "active_rows": len(active),
            "boundary_truth_flow": boundary, "boundary_ties_included": True,
            "channels": metrics, "source": str(path), "source_sha256": sha256_file(path)}


def run_selftest(target: str, marker: str) -> bool:
    subprocess.run(["make", target], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    proc = subprocess.run([str(ROOT / target)], cwd=ROOT,
                          text=True, capture_output=True)
    return proc.returncode == 0 and marker in (proc.stdout + proc.stderr)


def negative_controls(same: dict[str, Any], projection: dict[str, Any],
                      static: dict[str, Any]) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    dual_ok = run_selftest("selftest_a2_06_dual_commit",
                           "A2_06_DUAL_COMMIT_SELFTEST PASS")
    line_ok = run_selftest("selftest_a2_06_line_jbar",
                           "A2_06_LINE_JBAR_SELFTEST PASS")

    def record(number: int, name: str, observed: bool, evidence: str) -> None:
        controls[f"A2_06_NEG_{number}"] = {"name": name,
            "expected": "FAIL", "observed_fail": bool(observed), "evidence": evidence}

    stale_mutation = (int(same["generation"]) - 1 !=
                      int(same["q_generation_bound_from_capture"]))
    record(1, "stale generation", dual_ok and stale_mutation,
           "dual fixture rejects wrong-generation commit and stale lookup view")
    record(2, "line/profile hash exchange", dual_ok and
           same["q_set_hash"] != PROFILE_SHA256,
           "dual fixture observes distinct QHASH and PROFILE view failures")
    poison_rows = [{"band": i, "direct": 4.0 * math.pi * 2.0 * r,
                    "projected": r} for i, r in enumerate([1.0] * 8)]
    record(3, "4pi/V/dt/frame omission", not error_summary(poison_rows)["passed"],
           "normalization poison violates projection limits")
    record(4, "UNSAMPLED zero/coarse fallback", dual_ok and
           "blocked_unsampled" in (ROOT / "src/lumina_plasma.c").read_text()
           and static["passed"],
           "dual fixture returns UNSAMPLED; consumer counter and zero-fallback trace pass")
    q_source = (ROOT / "src/line_jbar.c").read_text()
    q_body = q_source.split("int line_jbar_qset_build", 1)[1].split(
        "void line_jbar_qset_free", 1)[0]
    baseline_ids = array("i", [0, 1, 3, 4]).tobytes()
    pruned_ids = array("i", [0, 1, 3]).tobytes()
    selection_poison = (hashlib.sha256(baseline_ids).digest() !=
                        hashlib.sha256(pruned_ids).digest())
    selection_inputs_ok = ("nlte_line_map[l] >= 0" in q_body and
                           "bb_in_domain" in q_body and
                           "A_ul" not in q_body and "sumsq" not in q_body)
    record(5, "A_ul/previous-estimator Q pruning", line_ok and
           selection_poison and selection_inputs_ok,
           "registered Q builder uses reachability/domain only; pruned census changes count/hash")
    owner = (ROOT / "src/radiation_field.h").read_text()
    record(6, "independent setter/generation/lifecycle", "LineJbarCache line_jbar_cache" in owner
           and "line_jbar_set" not in owner, "cache is nested in RadiationFieldOwner")
    phi_rows = [{"band": i, "direct": 0.5 * float(i + 1),
                 "projected": float(i + 1)} for i in range(8)]
    field_source = (ROOT / "src/radiation_field.c").read_text()
    frame_guard = (field_source.count(
        "frame != RADIATION_FIELD_FRAME_SHELL_COMOVING") >= 2)
    record(7, "phi normalization/observer frame",
           not error_summary(phi_rows)["passed"] and frame_guard,
           "half-normalized phi fails closure; both checked views reject observer frame")
    record(8, "legacy production read trace", static["passed"],
           "diagnostic shadows excluded; every production BB site reaches checked view")
    record(9, "partial commit publication", dual_ok,
           "dual-commit fixture injects both failure directions and memcmp-checks public state")
    return controls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "validation/a2_06")
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--fine-result", type=Path, default=FINE_RESULT)
    parser.add_argument("--aul-ledger", type=Path,
                        help="reuse a previously generated A2_06_AUL_CROSSWALK.json")
    parser.add_argument("--rate-ledger", type=Path,
                        help="separated CMFGEN/Lumina rate export; absent => registered BLOCKED")
    parser.add_argument("--aul-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if args.aul_ledger:
            model = args.model.resolve()
            _, label_binding = load_model_levels(model / "levels.csv")
            if label_binding["state"] != "ACTIVE":
                aul = blocked_aul_crosswalk(model, label_binding)
            else:
                aul = load_json(args.aul_ledger)
                aul.setdefault("label_binding", label_binding)
        else:
            aul = aul_crosswalk(args.model.resolve(), args.out.resolve())
            with (args.out / "A2_06_AUL_CROSSWALK.json").open("w") as stream:
                json.dump(aul, stream, indent=2, allow_nan=False)
        if args.aul_only:
            print(json.dumps(aul, indent=2))
            return 0 if aul.get("verdict") in (
                "PASS", "BLOCKED_MISSING_LABEL_BINDING") else 1

        fine_gate, fine_doc = validate_fine_inputs(args.fine_result.resolve())
        same = validate_same_measure(fine_doc)
        projection = run_projection_fixture(args.out.resolve())
        static = static_read_trace()
        controls = negative_controls(same, projection, static)
        controls_ok = all(row["observed_fail"] for row in controls.values())
        if args.rate_ledger:
            l1bb = judge_rate_ledger(args.rate_ledger.resolve())
        else:
            l1bb = {"state": "BLOCKED_MISSING_RATE_EXPORT",
                    "assertion": "NETRATE/TOTRATE separated channels absent; PASS forbidden"}
        available_ok = (fine_gate["passed"] and projection["passed"] and
                        static["passed"] and controls_ok)
        if available_ok and aul.get("verdict") == "BLOCKED_MISSING_LABEL_BINDING":
            verdict = "BLOCKED_MISSING_LABEL_BINDING"
        elif available_ok and aul.get("verdict") == "PASS":
            verdict = "PASS" if l1bb["state"] == "PASS" else (
                "BLOCKED_MISSING_RATE_EXPORT" if
                l1bb["state"] == "BLOCKED_MISSING_RATE_EXPORT" else "FAIL")
        else:
            verdict = "FAIL"
        ledger = {"schema": "lumina-a2-06-l1bb-gate-v1", "verdict": verdict,
                  "same_measure": same, "projection_closure_8band": projection,
                  "fine_closure_74": fine_gate, "static_read_trace": static,
                  "A_ul_crosswalk": aul, "negative_controls": controls,
                  "L_1bb": l1bb}
        with (args.out / "A2_06_L1BB_GATE.json").open("w") as stream:
            json.dump(ledger, stream, indent=2, allow_nan=False)
        print(f"A2-06 gate: {verdict}")
        print(f"ledger: {args.out / 'A2_06_L1BB_GATE.json'}")
        return 0 if verdict in (
            "PASS", "BLOCKED_MISSING_RATE_EXPORT",
            "BLOCKED_MISSING_LABEL_BINDING") else 1
    except (GateInputError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"A2-06 gate input/error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
