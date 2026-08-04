#!/usr/bin/env python3
"""Build the CMFGEN-active deck and quarantine the 32-ion source inventory.

The source and all four historical decks are read-only.  This driver refuses
to overwrite its output and emits a draft manifest; ``seal_atomic_quarantine``
must run after the derived sidecars are baked.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos"
TARGET = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_active"
EXPAND = ROOT / "scripts/expand_atomic_data_cmfgen.py"

MODEL_COMPANIONS = {
    "config.json", "density.csv", "deposition_cmfgen.csv",
    "electron_densities.csv", "geometry.csv", "plasma_state.csv",
    "coiii_col_cmfgen.csv", "feiii_col_zhang.bin", "s2_col_cmfgen.csv",
    "si2_col_cmfgen.csv", "siii_col_cmfgen.csv",
    "siii_terms_col_cmfgen.csv", "siii_terms_col_cmfgen_provenance.txt",
}

ACTIVE_CORE = {
    "abundances.csv", "active_ions.csv", "atom_masses.csv",
    "atomic_data_cmfgen.h5", "atomic_vintage_manifest.csv",
    "cmfgen_sigma_bf.bin", "ionization_energies.csv", "levels.csv",
    "line_list.csv", "macro_atom_data.csv", "macro_atom_references.csv",
    "zeta_data.npy", "zeta_ions.csv", "zeta_temps.csv",
}

CLASS_A_ELEMENTS = {14, 16, 20, 26, 27, 28}
EXPECTED_COUNTS = {"a": 6, "b": 0, "c": 26}
EXPECTED_EXTRA_LEVELS = 10_607


def validate_composition_shape(deck: Path) -> None:
    with (deck / "abundances.csv").open(newline="") as stream:
        header = next(csv.reader(stream), [])
    abundance_columns = max(len(header) - 1, 0)
    with (deck / "geometry.csv").open(newline="") as stream:
        rows = csv.reader(stream)
        next(rows, None)
        geometry_rows = sum(1 for row in rows if row)
    if abundance_columns != geometry_rows:
        raise SystemExit(
            f"abundances/geometry shape mismatch in {deck}: "
            f"abundance_columns={abundance_columns}, geometry_rows={geometry_rows}"
        )


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ordered_model_scope(model_spec: Path) -> list[dict]:
    pattern = re.compile(
        r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s+\[([^]]+)_ISF\]"
    )
    rows = []
    for line in model_spec.read_text(encoding="latin-1").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        nsl, nsl_repeat, nf = map(int, match.group(1, 2, 3))
        if nsl != nsl_repeat:
            raise RuntimeError(
                f"MODEL_SPEC N_SL columns disagree for {match.group(4)}: "
                f"{nsl}/{nsl_repeat}"
            )
        rows.append({"label": match.group(4), "n_sl": nsl, "n_full": nf})
    return rows


def source_inventory() -> tuple[set[tuple[int, int]], dict, dict]:
    levels: dict[tuple[int, int], list[dict[str, str]]] = {}
    with (SOURCE / "levels.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["atomic_number"]), int(row["ion_number"]))
            levels.setdefault(key, []).append(row)
    vintage: dict[tuple[int, int], dict[str, str]] = {}
    with (SOURCE / "atomic_vintage_manifest.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["atomic_number"]), int(row["ion_number"]))
            if key in vintage:
                raise RuntimeError(f"duplicate source vintage row {key}")
            vintage[key] = row
    if set(levels) != set(vintage):
        raise RuntimeError("source levels/vintage ion sets differ")
    return set(levels), levels, vintage


def source_abundances() -> dict[int, list[float]]:
    result = {}
    with (SOURCE / "abundances.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            z = int(row.pop("atomic_number"))
            result[z] = [float(value) for value in row.values()]
    return result


def write_active_ions(path: Path, scopes: list[dict], expand) -> None:
    fields = [
        "atomic_number", "ion_number", "ion_stage", "spectroscopic",
        "model_label", "n_full", "n_super", "osc_path", "f_to_s_path",
        "osc_sha256", "f_to_s_sha256",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in scopes:
            z, stage = item["key"]
            sources = expand.CMFGEN_LINK_MAP[(z, stage)]
            writer.writerow({
                "atomic_number": z,
                "ion_number": stage - 1,
                "ion_stage": stage,
                "spectroscopic": f"{expand.SYM[z]} {expand.ROMAN[stage]}",
                "model_label": item["label"],
                "n_full": item["n_full"],
                "n_super": item["n_sl"],
                "osc_path": sources["osc"],
                "f_to_s_path": sources["f_to_s"],
                "osc_sha256": sha256_file(sources["osc"]),
                "f_to_s_sha256": sha256_file(sources["f_to_s"]),
            })


def write_filtered_elements(output: Path, active_z: set[int]) -> None:
    for name in ("atom_masses.csv", "abundances.csv"):
        with (SOURCE / name).open(newline="") as inp:
            reader = csv.DictReader(inp)
            rows = [row for row in reader if int(row["atomic_number"]) in active_z]
            fields = list(reader.fieldnames or ())
        with (output / name).open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def write_filtered_zeta(output: Path, active: set[tuple[int, int]]) -> None:
    source_rows = []
    keep = []
    with (SOURCE / "zeta_ions.csv").open(newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if (int(row["atomic_number"]), int(row["ion_number"])) in active:
                keep.append(index)
                source_rows.append(row)
    zeta = np.load(SOURCE / "zeta_data.npy", allow_pickle=False)
    if zeta.shape[0] != sum(1 for _ in (SOURCE / "zeta_ions.csv").open()) - 1:
        raise RuntimeError("source zeta row count differs from zeta_ions.csv")
    with (output / "zeta_ions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["atomic_number", "ion_number"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(source_rows)
    np.save(output / "zeta_data.npy", zeta[np.asarray(keep, dtype=np.int64)])
    shutil.copy2(SOURCE / "zeta_temps.csv", output / "zeta_temps.csv")


def archive_source(output: Path) -> dict[str, dict[str, int | str]]:
    snapshot = output / "quarantine/source_deck_snapshot"
    snapshot.mkdir(parents=True)
    hashes = {}
    for source in sorted(SOURCE.iterdir()):
        if not source.is_file():
            raise RuntimeError(f"unexpected non-file in source deck: {source}")
        target = snapshot / source.name
        shutil.copy2(source, target)
        hashes[source.name] = {
            "sha256": sha256_file(target), "bytes": target.stat().st_size,
        }
    return hashes


def build_quarantine_manifest(
    output: Path,
    active: set[tuple[int, int]],
    original: set[tuple[int, int]],
    levels: dict,
    vintage: dict,
    expand,
    archive_hashes: dict,
) -> None:
    extra = sorted(original - active)
    abundances = source_abundances()
    ions = []
    counts = {"a": 0, "b": 0, "c": 0}
    total_levels = 0
    for z, ion0 in extra:
        rows = levels[(z, ion0)]
        stage = ion0 + 1
        source = Path(vintage[(z, ion0)]["osc_path"])
        atomic_exists = source.is_file()
        classification = "c" if z not in CLASS_A_ELEMENTS else (
            "a" if atomic_exists else "b"
        )
        counts[classification] += 1
        total_levels += len(rows)
        abundance = abundances.get(z, [])
        nonzero = sum(value > 0.0 for value in abundance)
        reason = {
            "a": "COMPOSITION_PRESENT_BUT_ION_NOT_LINKED",
            "b": "CMFGEN_ATOMIC_DATA_ABSENT",
            "c": "ELEMENT_ABSENT_FROM_CMFGEN_COMPOSITION",
        }[classification]
        ions.append({
            "ion": {
                "Z": z, "ion0": ion0,
                "spectroscopic": f"{expand.SYM[z]} {expand.ROMAN[stage]}",
            },
            "status": "quarantined",
            "classification": {
                "primary": classification, "reason_code": reason,
                "precedence": "c>a>b",
            },
            "cmfgen": {
                "model_spec_present": False,
                "atomic_link_present": False,
                "atomic_data_exists": atomic_exists,
                "composition_present": z in CLASS_A_ELEMENTS,
                "atomic_paths": [str(source)],
                "evidence_hashes": {
                    "osc_sha256": sha256_file(source) if source.is_file() else None,
                },
            },
            "lumina_before": {
                "abundance_min": min(abundance) if abundance else 0.0,
                "abundance_max": max(abundance) if abundance else 0.0,
                "nonzero_shells": nonzero,
                "physical_activity": (
                    "PHYSICAL_CONTAMINATION" if classification == "c" else
                    "ACTIVE_UNLINKED_ION"
                ),
                "full_levels": len(rows),
                "super_levels": len({int(row["super_level"]) for row in rows}),
            },
            "archive": {
                "snapshot": "quarantine/source_deck_snapshot",
                "reversible": True,
            },
            "restore_requirements": [
                f"CMFGEN target set links ({z},{ion0})",
                f"CMFGEN composition includes Z={z}",
                "all bidirectional identity and quarantine-leak gates pass",
            ],
        })
    if counts != EXPECTED_COUNTS or total_levels != EXPECTED_EXTRA_LEVELS:
        raise RuntimeError(
            f"pre-registered quarantine census changed: counts={counts}, "
            f"levels={total_levels}"
        )

    manifest = {
        "schema_version": 1,
        "state": "draft",
        "source_deck": str(SOURCE),
        "active_deck": str(TARGET),
        "preserved_original_ion_count": len(original),
        "active_ion_count": len(active),
        "quarantined_ion_count": len(extra),
        "classification_counts": counts,
        "logical_active_system": {
            "max_n": 240,
            "allocator_reduction_claimed": False,
            "note": (
                "The current src allocator still sizes from full-level offsets; "
                "quarantine alone does not reduce GPU allocation."
            ),
        },
        "archive_files": archive_hashes,
        "active_files": {},
        "pre_registration": {
            "physics_changes": True,
            "expected_direction": (
                "Opacity from positive-abundance C/O/Mg/Al/Sc/Ti/V/Cr/Mn "
                "disappears, so the ejecta must become more transparent by that "
                "contribution; the opposite direction falsifies the mechanism."
            ),
            "required_comparison": (
                "Compare the immutable regression ledger before and after the run; "
                "this build does not write validation/regression_ledger/."
            ),
        },
        "loader_contract": {
            "root_only": True,
            "recursive_glob_forbidden": True,
            "leak_fatal_tag": "[ATOMIC-ACTIVE-SET-LEAK]",
            "sentinel": "quarantine/DO_NOT_LOAD",
        },
        "restore": {
            "in_place": False,
            "snapshot": "quarantine/source_deck_snapshot",
            "procedure": "docs/CODEX_ATOMIC_QUARANTINE.md#restore",
            "events": [],
        },
        "ions": ions,
    }
    quarantine = output / "quarantine"
    (quarantine / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    fields = [
        "atomic_number", "ion_number", "spectroscopic", "classification",
        "reason_code", "full_levels", "super_levels", "abundance_min",
        "abundance_max", "nonzero_shells", "physical_activity",
    ]
    with (quarantine / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in ions:
            writer.writerow({
                "atomic_number": item["ion"]["Z"],
                "ion_number": item["ion"]["ion0"],
                "spectroscopic": item["ion"]["spectroscopic"],
                "classification": item["classification"]["primary"],
                "reason_code": item["classification"]["reason_code"],
                "full_levels": item["lumina_before"]["full_levels"],
                "super_levels": item["lumina_before"]["super_levels"],
                "abundance_min": item["lumina_before"]["abundance_min"],
                "abundance_max": item["lumina_before"]["abundance_max"],
                "nonzero_shells": item["lumina_before"]["nonzero_shells"],
                "physical_activity": item["lumina_before"]["physical_activity"],
            })
    sentinel = quarantine / "DO_NOT_LOAD"
    sentinel.write_text(
        "This file is intentionally unreadable. Any loader traversal into "
        "quarantine is [ATOMIC-ACTIVE-SET-LEAK].\n"
    )
    sentinel.chmod(0)


def main() -> int:
    required = {
        "CMFGEN_FULL_LEVELS": "1",
        "CMFGEN_SUPER_LEVELS": "1",
        "CMFGEN_LINK_FTOS": "1",
    }
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise SystemExit(f"{name}={expected} is required")
    cmf_run = Path(os.environ.get(
        "CMFGEN_RUN", "/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4"
    ))
    links_path = Path(os.environ.get("CMFGEN_LINKS", cmf_run / "atomic_links.txt"))
    if not (cmf_run / "MODEL_SPEC").is_file() or not links_path.is_file():
        raise SystemExit(f"missing MODEL_SPEC/atomic_links below {cmf_run}")
    if not SOURCE.is_dir():
        raise SystemExit(f"source R4 deck absent: {SOURCE}")
    validate_composition_shape(SOURCE)
    if TARGET.exists():
        raise SystemExit(f"refusing to overwrite existing output: {TARGET}")
    build = Path(os.environ.get(
        "QUARANTINE_BUILD_DIR", f"{TARGET}.building-{os.environ.get('SLURM_JOB_ID', 'local')}"
    ))
    if build.exists():
        raise SystemExit(f"refusing to reuse build directory: {build}")

    # expand_atomic_data_cmfgen resolves its link map at import time.
    os.environ["CMFGEN_LINKS"] = str(links_path)
    expand = load_module("deck_expand_quarantine", EXPAND)
    ordered_keys_module = load_module(
        "r1_scope_for_quarantine", ROOT / "scripts/verify_deck_r1_vintage.py"
    )
    ordered_keys = ordered_keys_module.ordered_osc_link_keys(expand, links_path)
    model_rows = ordered_model_scope(cmf_run / "MODEL_SPEC")
    if len(ordered_keys) != 27 or len(model_rows) != 27:
        raise SystemExit(
            f"expected 27 MODEL_SPEC/link ions, got {len(model_rows)}/{len(ordered_keys)}"
        )
    scopes = []
    for key, model in zip(ordered_keys, model_rows, strict=True):
        scopes.append({**model, "key": key})
    active_stage = set(ordered_keys)
    active = {(z, stage - 1) for z, stage in active_stage}
    original, source_levels, source_vintage = source_inventory()
    if len(original) != 59 or len(original - active) != 32:
        raise SystemExit(
            f"source census changed: original={len(original)} extra={len(original-active)}"
        )

    build.mkdir(parents=True)
    (build / "quarantine").mkdir()
    archive_hashes = archive_source(build)

    expand.ROOT = ROOT
    expand.CMFGEN_ROOT = ROOT / "data/atomic/cmfgen"
    expand.OUT_DIR = build
    expand.OUT_H5 = build / "atomic_data_cmfgen.h5"
    expand.OUT_SIGMA_BIN = build / "cmfgen_sigma_bf.bin"
    expand.ION_LEVEL_CAPS = {item["key"]: item["n_full"] for item in scopes}
    if set(expand.CMFGEN_LINK_MAP) != active_stage:
        raise SystemExit("atomic_links parsed set differs from active MODEL_SPEC set")
    ion_data = expand.parse_all_ions()
    if set(ion_data) != active_stage:
        raise SystemExit("parsed ion set differs from exact CMFGEN active set")

    for item in scopes:
        key = item["key"]
        data = ion_data[key]
        nf = item["n_full"]
        if data["osc"].n_levels < nf or data.get("ftos") is None:
            raise RuntimeError(f"linked source cannot supply NF/f_to_s for {key}")
        data["n_kept"] = nf
        data["levels"] = data["osc"].levels[:nf]
        trans = data["trans"]
        data["trans"] = trans[(trans["i"] <= nf) & (trans["j"] <= nf)]
        prefix = data["ftos"].sl_of_fl[:nf]
        if len(set(prefix.tolist())) != item["n_sl"]:
            raise RuntimeError(
                f"MODEL_SPEC/f_to_s prefix N_SL mismatch for {key}: "
                f"{len(set(prefix.tolist()))}/{item['n_sl']}"
            )

    levels_rows, lookup, per_ion_g = expand.build_global_levels(ion_data)
    lines = expand.build_lines(ion_data, lookup, per_ion_g)
    expand.write_levels_csv(levels_rows, build / "levels.csv")
    expand.write_atomic_vintage_manifest(ion_data, build / "atomic_vintage_manifest.csv")
    expand.write_line_list_csv(lines, levels_rows, build / "line_list.csv")
    expand.write_macro_atom(
        lines, levels_rows, lookup,
        build / "macro_atom_data.csv", build / "macro_atom_references.csv",
    )
    expand.write_ionization_csv(ion_data, build / "ionization_energies.csv")
    expand.write_phot_col_h5(ion_data, build / "atomic_data_cmfgen.h5")
    expand.bake_sigma_bf_grid(
        ion_data, ion_data, levels_rows, lookup, build / "cmfgen_sigma_bf.bin"
    )

    for name in MODEL_COMPANIONS:
        source = SOURCE / name
        if source.is_file():
            shutil.copy2(source, build / name)
    write_filtered_elements(build, {z for z, _ in active})
    validate_composition_shape(build)
    write_filtered_zeta(build, active)
    write_active_ions(build / "active_ions.csv", scopes, expand)
    config_path = build / "config.json"
    config = json.loads(config_path.read_text())
    config["n_lines"] = int(lines["Z"].size)
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    build_quarantine_manifest(
        build, active, original, source_levels, source_vintage, expand, archive_hashes
    )
    build.rename(TARGET)
    print(
        f"created unsealed active deck: {TARGET} "
        f"({len(active)} active / {len(original-active)} quarantined ions)"
    )
    print("NEXT: bake derived sidecars, then run seal_atomic_quarantine.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
