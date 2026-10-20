#!/usr/bin/env python3
"""Create the R1 link-pinned full-coverage deck without overwriting any deck."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_fullcov"
NEW_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_links"
NEW = Path(os.environ.get("LUMINA_CMFGEN_OUTPUT_DECK", str(NEW_DEFAULT))).expanduser().resolve()
EXPAND = ROOT / "scripts/expand_atomic_data_cmfgen.py"

REBUILT = {
    "atom_masses.csv",
    "atomic_data_cmfgen.h5",
    "atomic_vintage_manifest.csv",
    "atomic_source_identity.json",
    "DECK_PROVENANCE.json",
    "cmfgen_phot_smoothing.so",
    "cmfgen_sigma_bf.bin",
    "cmfgen_bf_channels.bin",
    "cmfgen_bf_channels_manifest.json",
    "coldata_cmfgen_manifest.csv",
    "ionization_energies.csv",
    "level_multiplicity.csv",
    "levels.csv",
    "line2macro_level_upper.npy",
    "line_list.csv",
    "kshape_contract.txt",
    "ma_radrecomb_target.bin",
    "ma_radrecomb_target_manifest.csv",
    "macro_atom_data.csv",
    "macro_atom_references.csv",
    "transition_probabilities.npy",
    "tau_sobolev.npy",
    "verification.log",
    "zeta_data.npy",
    "zeta_ions.csv",
    "zeta_temps.csv",
}


def validate_composition_shape(deck: Path) -> int:
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
    return geometry_rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic_source_identity(manifest_path: Path, output_path: Path) -> dict:
    """Hash every exact osc/F_TO_S/PHOT/col file used by the active atom set."""
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite source identity: {output_path}")
    source_cache: dict[str, dict] = {}
    ions = []
    with manifest_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"ion", "selection_source", "model_spec_nf", "active_superlevels"}
        required.update(f"{kind}_path" for kind in ("osc", "f_to_s", "phot", "col"))
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit("atomic vintage manifest lacks required active-source columns")
        for row in reader:
            if row["selection_source"] != "links":
                raise SystemExit(
                    f"MODEL_SPEC deck contains non-linked ion {row['ion']}: "
                    f"{row['selection_source']}"
                )
            if not row["model_spec_nf"] or not row["active_superlevels"]:
                raise SystemExit(f"MODEL_SPEC counts missing for {row['ion']}")
            sources = {}
            for kind in ("osc", "f_to_s", "phot", "col"):
                raw_path = row[f"{kind}_path"]
                if not raw_path:
                    raise SystemExit(f"{row['ion']} has no {kind} source path")
                source = Path(raw_path)
                resolved = source.resolve(strict=True)
                if not resolved.is_file():
                    raise SystemExit(f"atomic source is not a regular file: {resolved}")
                key = str(resolved)
                if key not in source_cache:
                    source_cache[key] = {
                        "resolved_path": key,
                        "bytes": resolved.stat().st_size,
                        "sha256": sha256_file(resolved),
                    }
                sources[kind] = {
                    "path": raw_path,
                    **source_cache[key],
                }
            ions.append({
                "ion": row["ion"],
                "atomic_number": int(row["atomic_number"]),
                "ion_stage": int(row["ion_stage"]),
                "model_spec_nf": int(row["model_spec_nf"]),
                "active_superlevels": int(row["active_superlevels"]),
                "sources": sources,
            })
    identity = {
        "schema": "lumina-cmfgen-active-atomic-source-identity-v1",
        "atomic_vintage_manifest": str(manifest_path.resolve()),
        "atomic_vintage_manifest_sha256": sha256_file(manifest_path),
        "n_active_ions": len(ions),
        "n_unique_source_files": len(source_cache),
        "source_files": [source_cache[key] for key in sorted(source_cache)],
        "ions": ions,
    }
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return identity


def write_deck_provenance(module, links: Path, model_spec: Path,
                          level_map: Path, geometry_shells: int,
                          source_identity: dict) -> None:
    """Replace inherited template provenance with this run's builder stamp."""
    destination = NEW / "DECK_PROVENANCE.json"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite deck provenance: {destination}")
    overrides = []
    for record in getattr(module, "ATOMIC_LOCAL_OVERRIDES", []):
        path = Path(record["path"]).resolve(strict=True)
        overrides.append({**record, "sha256": sha256_file(path)})
    builder_paths = [
        Path(__file__).resolve(),
        EXPAND.resolve(),
        (ROOT / "scripts/cmfgen_parser.py").resolve(),
        (ROOT / "scripts/cmfgen_phot_smoothing.py").resolve(),
        (ROOT / "scripts/cmfgen_recombination_policy.py").resolve(),
    ]
    hyd_fit_inputs = []
    for raw_path in getattr(module, "HYD_BF_DATA_FILES", ()):
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file():
            raise SystemExit(f"CMFGEN hydrogenic fit table is not a file: {path}")
        hyd_fit_inputs.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if len(hyd_fit_inputs) != 2:
        raise SystemExit("expected both CMFGEN HYD_L_DATA and GBF_N_DATA inputs")
    identity_path = NEW / "atomic_source_identity.json"
    smoother = getattr(module, "PHOT_SMOOTHER", None)
    phot_preprocessing = smoother.provenance() if smoother is not None else None
    if smoother is not None and not smoother.sources:
        raise SystemExit("PHOT preprocessing lacks original CMFGEN source identities")
    if smoother is not None and smoother.library is not None:
        library_copy = NEW / "cmfgen_phot_smoothing.so"
        if library_copy.exists():
            raise SystemExit(f"refusing to overwrite PHOT library: {library_copy}")
        shutil.copy2(smoother.library, library_copy)
        phot_preprocessing["library"] = {
            "path": str(library_copy), "sha256": sha256_file(library_copy)}
    stamp = {
        "schema": "lumina-deck-provenance-v1",
        "deck": str(NEW),
        "cmfgen_links": str(links.resolve()),
        "cmfgen_links_sha256": sha256_file(links),
        "cmfgen_model_spec": str(model_spec.resolve()),
        "cmfgen_model_spec_sha256": sha256_file(model_spec),
        "level_sl_steq_links": str(level_map.resolve()),
        "level_sl_steq_links_sha256": sha256_file(level_map),
        "provenance_base": str(OLD),
        "n_shells": geometry_shells,
        # 이 드라이버는 expand만 호출한다. NPY/매크로/K-SHAPE finalizer가
        # 실행되지 않은 원자 core를 완성된 생산 덱으로 표시하지 않는다.
        "finalized": False,
        "line_einstein_convention": "J_nu",
        "builder": {
            str(path): sha256_file(path) for path in builder_paths
        },
        "atomic_source_identity": {
            "path": str(identity_path),
            "sha256": sha256_file(identity_path),
            "n_active_ions": source_identity["n_active_ions"],
            "n_unique_source_files": source_identity["n_unique_source_files"],
        },
        "hydrogenic_fit_data": hyd_fit_inputs,
        "phot_preprocessing": phot_preprocessing,
        "active_ions": {
            row["ion"]: {
                "model_spec_nf": row["model_spec_nf"],
                "active_superlevels": row["active_superlevels"],
            }
            for row in source_identity["ions"]
        },
        "atomic_local_overrides": overrides,
        "env": {
            name: os.environ.get(name, "")
            for name in (
                "CMFGEN_LINK_FTOS", "CMFGEN_VINTAGE_MATCH",
                "CMFGEN_EXACT_HYD",
                "CMFGEN_FULL_LEVELS", "CMFGEN_SUPER_LEVELS",
                "LUMINA_SIII_PHOT_PATH", "LUMINA_COII_PHOT_PATH",
            )
        },
        "note": "Active ion set and levels are sealed by this run's MODEL_SPEC and links.",
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(stamp, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_expand_module():
    spec = importlib.util.spec_from_file_location("deck_expand_r1", EXPAND)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EXPAND}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_companions() -> None:
    """Copy immutable model companions; runtime atomic sidecars are rebuilt."""
    for source in sorted(OLD.iterdir()):
        if source.name in REBUILT or source.name.startswith("ige_col_"):
            continue
        target = NEW / source.name
        if target.exists():
            continue
        resolved = source.resolve(strict=True)
        if resolved.is_file():
            shutil.copy2(resolved, target)


def main() -> None:
    if os.environ.get("CMFGEN_FULL_LEVELS") != "1":
        raise SystemExit("CMFGEN_FULL_LEVELS=1 is required")
    if os.environ.get("CMFGEN_SUPER_LEVELS") != "1":
        raise SystemExit("CMFGEN_SUPER_LEVELS=1 is required")
    links = os.environ.get("CMFGEN_LINKS", "")
    if not links or not Path(links).is_file():
        raise SystemExit(f"CMFGEN_LINKS must name an existing atomic_links.txt: {links}")
    model_spec = os.environ.get("CMFGEN_MODEL_SPEC", "")
    if not model_spec or not Path(model_spec).is_file():
        raise SystemExit(f"CMFGEN_MODEL_SPEC must name the matching run's MODEL_SPEC: {model_spec}")
    if Path(links).resolve().parent != Path(model_spec).resolve().parent:
        raise SystemExit("CMFGEN_LINKS and CMFGEN_MODEL_SPEC must belong to the same run root")
    level_map = Path(os.environ.get(
        "CMFGEN_LEVEL_SL_STEQ_LINKS",
        str(Path(model_spec).resolve().parent / "LEVEL_SL_STEQ_LINKS"),
    )).expanduser().resolve()
    if not level_map.is_file():
        raise SystemExit(f"CMFGEN_LEVEL_SL_STEQ_LINKS must exist: {level_map}")
    if os.environ.get("CMFGEN_LINK_FTOS") != "1":
        raise SystemExit("CMFGEN_LINK_FTOS=1 is required for exact linked superlevels")
    if NEW.exists():
        raise SystemExit(f"refusing to overwrite existing output: {NEW}")
    if NEW == OLD or OLD in NEW.parents:
        raise SystemExit(f"output deck must not overwrite or nest inside template: {NEW}")
    if not OLD.is_dir():
        raise SystemExit(f"missing full-coverage provenance deck: {OLD}")

    module = load_expand_module()
    module.ROOT = ROOT
    module.CMFGEN_ROOT = ROOT / "data/atomic/cmfgen"
    module.OUT_DIR = NEW
    module.OUT_H5 = NEW / "atomic_data_cmfgen.h5"
    module.OUT_SIGMA_BIN = NEW / "cmfgen_sigma_bf.bin"
    module.OUT_BF_CHANNEL_BIN = NEW / "cmfgen_bf_channels.bin"
    module.main()
    copy_companions()
    geometry_shells = validate_composition_shape(NEW)
    source_identity = write_atomic_source_identity(
        NEW / "atomic_vintage_manifest.csv", NEW / "atomic_source_identity.json")
    write_deck_provenance(
        module, Path(links), Path(model_spec), level_map,
        geometry_shells, source_identity)
    print(f"created core R1 link-pinned deck: {NEW}")


if __name__ == "__main__":
    main()
